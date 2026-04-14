import argparse

import os
import sys
import shutil
import json
import pickle
from glob import glob
from time import time
from datetime import datetime
import warnings

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

import torch
from torch import nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torch.utils.tensorboard import SummaryWriter
torch.set_default_dtype(torch.float64)

from shutil import SameFileError


def get_cli():
    parser = argparse.ArgumentParser()
    # source parameters
    parser.add_argument("--dataset-rootdir", type=str,
                        help="Dataset directory")
    parser.add_argument("--rescale-rootdir", type=str,
                        help="Directory of the rescaling files")
    parser.add_argument("--base-network-path", type=str,
                        help="Path to the pre-rescaled network")
    parser.add_argument("--output-rootdir", type=str,
                        help="Output directory")
    parser.add_argument("--run-title", type=str, default="npe",
                        help="Title of the run")
    parser.add_argument("--run-type", type=str, default="BH",
                        help="Type of compact binary event")
    parser.add_argument("--dataset-seed", type=int, default=1234,
                        help="Set RNG seed for dataset generation.")
    parser.add_argument("--training-seed", type=int, default=-1,
                        help="Set RNG seed for training.")
    parser.add_argument("--data-dim", type=int, default=640,
                        help="Set dimensionality of dephasing frequency grid (Default: 640).")
    parser.add_argument("--num-epochs-shape", type=int, default=50,
                        help="Number of epochs used for training the primary network.")
    parser.add_argument("--num-epochs-scale", type=int, default=50,
                        help="Number of epochs to used for training the secondary network.")
    parser.add_argument("--num-epochs-rescale", type=int, default=50,
                        help="Number of epochs to use for rescaling the secondary network.")
    parser.add_argument("--include-tidal-params", action=argparse.BooleanOptionalAction, default=False, required=False,
                        help="Include tidal deformation terms in the parameter space.")
    parser.add_argument("--include-tidal-params-full", action=argparse.BooleanOptionalAction, default=False, required=False,
                        help="Include tidal and spin induced deformation terms in the parameter space.")
    parser.add_argument("--include-tidal-data", action=argparse.BooleanOptionalAction, default=False, required=False,
                        help="Include tidal deformation waveform data, up to 5PN.")
    parser.add_argument("--include-tidal-data-2p5and4", action=argparse.BooleanOptionalAction, default=None, required=False,
                        help="Include tidal deformation waveform data which may be degenerate with waveform parameters, specifically at 2.5PN and 4PN")
    parser.add_argument("--include-2PN-data", action=argparse.BooleanOptionalAction, default=True, required=False,
                        help="Include dephasing training data for the 2PN order.")
    parser.add_argument("--include-1p5PN-data", action=argparse.BooleanOptionalAction, default=True, required=False,
                        help="Include dephasing training data for the 1.5PN order.")
    parser.add_argument("--rescale-2p5and4", action=argparse.BooleanOptionalAction, default=None, required=False,
                        help="Whether or not to suppress latent space support at 2.5PN and 4PN orders.")
    parser.add_argument("--rescale-2p5and4-only", action=argparse.BooleanOptionalAction, default=None, required=False,
                        help="Whether or not to only rescale by suppressing latent space support at 2.5PN and 4PN orders.")
    parser.add_argument("--penalize-highPN", action=argparse.BooleanOptionalAction, default=None, required=False,
                        help="Penalize latent space from learning high-PN-like dephasing functions in the non-PN region.")
    
    parser.add_argument("--train-lr", type=float, default=1e-4,
                        help="Base learning rate to use for training.")
    parser.add_argument("--train-wd", type=float, default=1e-4,
                        help="Weight decay to use for training.")
    parser.add_argument("--train-gamma", type=float, default=0.9,
                        help="Multiplicative learning rate decay factor (<1).")
    parser.add_argument("--train-kl-coeff", type=float, default=1e-6,
                        help="Coefficient to weight KL divergence term in loss function relative to reconstruction error term.")
    args = parser.parse_args()
    return args


# ===================

def get_fc_layers(input_dim, output_dim, width, depth, 
                  activation=nn.ReLU, batchnorm=nn.Identity):
    layers = [
        nn.Linear(input_dim, width),
        batchnorm(width),
        activation(),
    ]
    for _ in range(depth):
        layers += [
            nn.Linear(width, width),
            batchnorm(width),
            activation(),
        ]
    layers.append(nn.Linear(width, output_dim))
    layers = nn.Sequential(*layers)
    return layers


class Encoder(nn.Module):

    def __init__(self, 
                 width=64, depth=2, 
                 latent_dim=2, data_dim=640, 
                 cond_dim=4, use_cond=True,
                 activation=nn.ReLU, batchnorm=nn.Identity):
        # Assume (N, data_dim) -> (N, latent_dim) + (N, latent_dim)
        super().__init__()
        self.latent_dim = latent_dim
        self.data_dim = data_dim
        self.cond_dim = cond_dim
        self.use_cond = use_cond
        input_dim = data_dim + use_cond * cond_dim
        output_dim = latent_dim * 2 - 1
        self.fc = get_fc_layers(input_dim, output_dim, width, depth, 
                                activation=activation, batchnorm=batchnorm)

    def forward(self, x, cond):
        if self.use_cond:
            x = torch.cat([x, cond], dim=-1)
        x = self.fc(x)
        mu, logvar = x[:,:self.latent_dim], x[:,self.latent_dim:]
        logvar = torch.cat([logvar, logvar], dim=-1)
        return mu, logvar

        
class Decoder(nn.Module):
    
    def __init__(self, 
                 width=64, depth=2, 
                 latent_dim=2, data_dim=640, 
                 cond_dim=4, use_cond=True,
                 activation=nn.ReLU, batchnorm=nn.Identity):
        # Assume mapping (N, latent_dim) + (N, cond_dim) -> (N, data_dim)
        super().__init__()
        self.data_dim = data_dim
        self.cond_dim = cond_dim
        self.latent_dim = latent_dim
        self.use_cond = use_cond
        input_dim = latent_dim + use_cond * cond_dim
        output_dim = data_dim
        self.fc = get_fc_layers(input_dim, output_dim, width, depth,
                                activation=activation, batchnorm=batchnorm)

    def forward(self, z, cond):
        if self.use_cond:
            z = torch.cat([z, cond], dim=-1)
        x = self.fc(z)
        return x


class VAE(nn.Module):
    
    def __init__(self, 
                 width=64, depth=2, 
                 width_scale=None, depth_scale=None, 
                 latent_dim=2, data_dim=640, cond_dim=4, grid_dim=None,
                 freeze_shape=False, freeze_scale=False,
                 activation=nn.ReLU, batchnorm=nn.Identity):
        super().__init__()
        assert latent_dim == 2
        if width_scale is None:
            width_scale = width
        if depth_scale is None:
            depth_scale = depth
        if grid_dim is None:
            grid_dim = data_dim
        self.data_dim = data_dim
        self.grid_dim = grid_dim
        self.cond_dim = cond_dim
        self.freeze_shape = freeze_shape
        self.freeze_scale = freeze_scale
        self.raw_encoder = Encoder(width, depth, 
                                   latent_dim, data_dim, 
                                   cond_dim, use_cond=False,
                                   activation=activation, batchnorm=batchnorm)
        self.raw_decoder0 = Decoder(width_scale, depth_scale, 
                                    latent_dim, 1,
                                    cond_dim, use_cond=True,
                                    activation=activation, batchnorm=batchnorm)
        self.raw_decoder1 = Decoder(width, depth, 
                                    latent_dim, grid_dim, 
                                    cond_dim, use_cond=False,
                                    activation=activation, batchnorm=batchnorm)
        self.raw_decoder2 = Decoder(width, depth, 
                                    latent_dim, grid_dim, 
                                    cond_dim, use_cond=False,
                                    activation=activation, batchnorm=batchnorm)
        if freeze_shape:
            for m in [self.raw_encoder, self.raw_decoder1, self.raw_decoder2]:
                for p in m.parameters():
                    p.requires_grad = False
        if freeze_scale:
            for p in self.raw_decoder0.parameters():
                p.requires_grad = False
        
    def encoder(self, x, cond):
        batch_size = x.shape[0]
        x = x / torch.sqrt(torch.mean(x*x, dim=-1, keepdim=True))
        x = x.view(-1, self.raw_encoder.data_dim)
        x = torch.cat([x, -x], dim=0)
        cond = torch.cat([cond, cond], dim=0)
        mu, logvar = self.raw_encoder(x, cond)
        mu = (mu[:batch_size] - mu[batch_size:]) / 2
        mu = mu / torch.sqrt(torch.sum(mu*mu, dim=-1, keepdim=True))
        logvar = (logvar[:batch_size] + logvar[batch_size:]) / 2
        return mu, logvar
    
    def grid_decoder(self, z, cond):
        batch_size = z.shape[0]
        z = torch.cat([z, -z], dim=0)
        cond = torch.cat([cond, cond], dim=0)
        x0 = self.raw_decoder0(z, cond)
        x1 = self.raw_decoder1(z, cond)
        x2 = self.raw_decoder2(z, cond)
        x0 = (x0[:batch_size] + x0[batch_size:]) / 2
        x1 = (x1[:batch_size] - x1[batch_size:]) / 2
        x2 = (x2[:batch_size] + x2[batch_size:]) / 2
        x1 = torch.exp(x0) * x1
        return x1, x2
    
    def decoder(self, z, cond, xin=None):
        batch_size = z.shape[0]
        x1, x2 = self.grid_decoder(z, cond)
        if xin is None:
            xin = torch.linspace(0, 1, self.data_dim, device=x1.device)
            xin = xin.repeat(batch_size, 1)
        xout = torch.sum(x1[:,None,:] * torch.exp(x2[:,None,:] * xin[:,:,None]), dim=-1)
        xout = xout.view(batch_size, 1, -1)
        return xout
    
    def varier(self, mu, logvar):
        z = mu + torch.exp(0.5*logvar) * torch.randn_like(mu)
        z = z / torch.sqrt(torch.sum(z*z, dim=-1, keepdim=True))
        return z

    def forward(self, x, cond):
        mu, logvar = self.encoder(x, cond)
        z = self.varier(mu, logvar)
        x_recon = self.decoder(z, cond)
        return x_recon, mu, logvar


# ===================

from utils.workflow import train
from utils.data import PhasingDataset
from utils.train import align_data_format_with_model
from utils import loss as loss_funcs

def get_shape_and_scale(v):
    scale = torch.sqrt(torch.mean(v*v, dim=-1, keepdim=True))
    shape = v / scale
    return shape, scale

def get_loss_shape(shape1, shape2):
    return F.mse_loss(shape1, shape2)

def get_loss_scale(scale1, scale2):
    log_scale1 = torch.log(scale1)
    log_scale2 = torch.log(scale2)
    return F.mse_loss(log_scale1, log_scale2)

def get_loss_recon_var(v, v_recon):
    shape1, scale1 = get_shape_and_scale(v)
    shape2, scale2 = get_shape_and_scale(v_recon)
    loss_shape = get_loss_shape(shape1, shape2)
    loss_scale = get_loss_scale(scale1, scale2)
    return loss_shape, loss_scale

def get_loss_recon(v, v_recon):
    shape1, scale1 = get_shape_and_scale(v)
    shape2, scale2 = get_shape_and_scale(v_recon)
    loss_shape = get_loss_shape(v, scale1*shape2)
    loss_scale = get_loss_scale(scale1, scale2)
    return loss_shape, loss_scale

def kl_div_diagonal_gaussian_to_standard_gaussian(mu, logvar, dim=None, with_mu=True):
    # changed to ignore the mu effect
    if with_mu:
        kld = -0.5 * (1 + logvar - torch.square(mu) - torch.exp(logvar))
    else:
        kld = -0.5 * (1 + logvar - torch.exp(logvar))
    if dim is None:
        kld = torch.sum(kld)
    else:
        kld = torch.sum(kld, dim=dim)
    return kld

def vae_loss_fn(model, v, l, t, *args, 
                kl_coeff=1., 
                shape_coeff=1., scale_coeff=1., 
                recon_coeff=1., recon_scale_coeff=1.,
                with_mu=True, recon_use_mse=False,
                **kwargs):
    v, l, t = align_data_format_with_model(model, v, l, t)
    
    mu, logvar = model.encoder(v, l)
    if kl_coeff == 0.:
        loss_kld = 0.
    else:
        loss_kld = kl_div_diagonal_gaussian_to_standard_gaussian(mu, logvar, dim=-1, with_mu=with_mu)
        loss_kld = kl_coeff * torch.mean(loss_kld)
    
    if shape_coeff == 0. and scale_coeff == 0.:
        loss_var = 0.
    else:
        z = model.varier(mu, logvar)
        v_recon_var = model.decoder(z, l)
        loss_shape, loss_scale = get_loss_recon_var(v, v_recon_var)
        loss_var = shape_coeff * loss_shape + scale_coeff * loss_scale

    if recon_coeff == 0. and (recon_use_mse or recon_scale_coeff == 0.):
        loss_recon == 0.
    else:
        v_recon = model.decoder(mu, l)
        if recon_use_mse:
            loss_recon = recon_coeff * F.mse_loss(v, v_recon)
        else:
            loss_shape_recon, loss_scale_recon = get_loss_recon(v, v_recon)
            loss_recon = recon_coeff * loss_shape_recon + recon_scale_coeff * loss_scale_recon
        
    loss = loss_kld + loss_var + loss_recon
    return loss


def vae_diagnosis_fn(model, v, l, t, *args, with_mu=True, **kwargs):
    v, l, t = align_data_format_with_model(model, v, l, t)
    
    mu, logvar = model.encoder(v, l)
    loss_kld = kl_div_diagonal_gaussian_to_standard_gaussian(mu, logvar, dim=-1, with_mu=with_mu)
    loss_kld = torch.mean(loss_kld)
    
    v_recon = model.decoder(mu, l)
    loss_shape, loss_scale = get_loss_recon_var(v, v_recon)
    loss_shape_recon, _ = get_loss_recon(v, v_recon)
    loss_recon = F.mse_loss(v, v_recon)
    
    metrics = dict(
        kl_div=loss_kld,
        err_recon=loss_recon,
        err_shape_rescaled=loss_shape_recon,
        err_shape=loss_shape,
        err_scale=loss_scale,
    )
    distrib = dict(
        values=v,
        labels=l,
        theory=t,
        mu=mu,
        logvar=logvar,
        recon=v_recon,
    )
    metrics = {k: v.item() for k, v in metrics.items()}
    distrib = {k: v.detach().cpu().numpy() for k, v in distrib.items()}
    return metrics, distrib


# ===================

def get_data_rescale_function(data_filename, rescale_rootdir):
    rescale_filename = data_filename.rpartition('.')[0] + '-phase-rescale-fac.npy'
    print(f'Loading rescale factors from: {os.path.join(rescale_rootdir, rescale_filename)}')
    rescale_factors = np.load(os.path.join(rescale_rootdir, rescale_filename))
    def prep_fn(df):
        df['phases'] = df['phases'] * rescale_factors
        return df
    return prep_fn

# ===================

def main():
    
    args = get_cli()

    # args = type('TrainingArguments', (), {})

    # args.run_title = "npE_network"
    # args.resume_title = args.run_title
    # args.resume_epochs = 0
    # args.add_epochs = 50
    # args.epochs_per_latent_plot = [(10, 1), (None, 10)]
    # args.epochs_per_checkpoint = 50
    # args.optimizer_override = True
    # args.scheduler_override = True

    # args.batch_size_train = 64
    # args.batch_size_val = 1024
    # args.batches_per_summary = 0.1

    # args.lr = 1e-4
    # args.wd = 1e-4
    # args.gamma = 0.9
    # args.loss_kwargs = dict(
    #     kl_coeff=1e-6, 
    #     shape_coeff=1., scale_coeff=0., 
    #     recon_coeff=0., recon_scale_coeff=0.,
    #     with_mu=False, recon_use_mse=False, 
    # )
    # args.diagnosis_kwargs = dict(with_mu=False)

    # args.model_type = VAE
    # args.model_kwargs = dict(
    #     depth=4, width=512, 
    #     data_dim=640, grid_dim=2, 
    #     freeze_scale=True,
    # )
    # args.optimizer_type = torch.optim.AdamW
    # args.scheduler_type = torch.optim.lr_scheduler.ExponentialLR
    # args.loss_fn = vae_loss_fn
    # args.diagnosis_fn = vae_diagnosis_fn
    # args.training_device = None
    # args.training_seed = None

    # args.npoints_for_latent_plot = int(1e2)
    # args.npoints_for_generation = 16
    # args.show_plot = False

    # args.dataset_recipe_from_file = None
    # args.dataset_recipe_save_file = None
    # args.dataset_filenames = [
    #     "ppe-minus13.pkl",
    #     "ppe-minus11.pkl",
    #     "ppe-minus9.pkl",
    #     "ppe-minus7.pkl",
    #     "ppe-minus5.pkl",
    #     "ppe-minus3.pkl",
    #     "ppe-minus1.pkl",
    # ]
    # args.dataset_prep_fns = [
    #     get_data_rescale_function(filename, args.rescale_rootdir)
    #     for filename in args.dataset_filenames
    # ]
    # args.dataset_type = PhasingDataset
    # args.dataset_n_ppe = 1
    # args.dataset_norm_fac = {}
    # args.dataset_sample_size = 0.25
    # args.dataset_subset_split = [0.8, 0.1, 0.1]
    # args.dataset_seed = 1234

    # train(args)


    # ===================

    # args.run_title = "npE_network"
    args.resume_title = args.run_title
    args.resume_epochs = args.num_epochs_shape + args.num_epochs_scale
    args.add_epochs = args.num_epochs_rescale
    args.epochs_per_latent_plot = min(10, args.num_epochs_rescale)
    args.epochs_per_checkpoint = min(10, args.num_epochs_rescale)
    args.optimizer_override = True
    args.scheduler_override = True

    args.batch_size_train = 64
    args.batch_size_val = 1024
    args.batches_per_summary = 0.1

    args.show_plot = False

    args.lr = args.train_lr
    args.wd = args.train_wd
    args.gamma = args.train_gamma
    args.loss_kwargs = dict(
        kl_coeff=0., 
        shape_coeff=0., scale_coeff=0., 
        recon_coeff=0., recon_scale_coeff=1.,
        with_mu=False, recon_use_mse=False, 
    )
    args.diagnosis_kwargs = dict(with_mu=False)

    # TODO: CLI arg support for variable model architecture structure? BNS vs. BBH
    # i.e. cond_dim ~ # of intrinsic binary parameters, so we may need to expand it for BNS tidal deformability
    args.structure_kwargs = dict(
        depth=4, width=512,
        data_dim=args.data_dim, grid_dim=2,
        cond_dim=8 if args.include_tidal_params_full else 6 if args.include_tidal_params else 4,
        )

    args.model_type = VAE
    args.model_kwargs = dict(
        **args.structure_kwargs,
        freeze_shape=True,
    )
    args.optimizer_type = torch.optim.AdamW
    args.scheduler_type = torch.optim.lr_scheduler.ExponentialLR
    args.loss_fn = vae_loss_fn
    args.diagnosis_fn = vae_diagnosis_fn
    args.training_device = None
    #args.training_seed = None
    args.training_seed = args.training_seed if args.training_seed > 0 else None

    args.npoints_for_latent_plot = int(1e2)
    args.npoints_for_generation = 16
    args.show_plot = False

    args.include_tidal_data = bool(args.include_tidal_data)
    args.include_tidal_data_2p5and4 = bool(args.include_tidal_data_2p5and4) if args.include_tidal_data_2p5and4 is not None else args.include_tidal_data
    args.rescale_2p5and4 = (bool(args.rescale_2p5and4) or bool(args.rescale_2p5and4_only)) and args.include_tidal_data_2p5and4
    args.rescale_2p5and4_only = bool(args.rescale_2p5and4_only)
    args.penalize_highPN = bool(args.penalize_highPN)

    args.dataset_recipe_from_file = None
    args.dataset_recipe_save_file = None
    if args.run_type in ['NS', 'NSBH']:
        args.dataset_filenames = [
            'ppe-minus13.pkl',
            'ppe-minus12.pkl',
            'ppe-minus11.pkl',
            'ppe-minus10.pkl',
            'ppe-minus9.pkl',
            'ppe-minus8.pkl',
            'ppe-minus7.pkl',
            'ppe-minus6.pkl',
            'ppe-minus5.pkl',
            'ppe-minus4.pkl',
            'ppe-minus3.pkl',
        ]
        if args.include_1p5PN_data:
            args.dataset_filenames += [
                'ppe-minus2.pkl'
            ]
        if args.include_2PN_data:
            args.dataset_filenames += [
                'ppe-minus1.pkl'
            ]
        if args.include_tidal_data:
            if args.include_tidal_data_2p5and4:
                args.dataset_filenames += [
                    'ppe-minus0.pkl',
                    'ppe-plus1.pkl',
                    'ppe-plus2.pkl',
                    'ppe-plus3.pkl',
                    'ppe-plus4.pkl',
                    'ppe-plus5.pkl'
                ]
            else:
                args.dataset_filenames += [
                    #'ppe-minus0.pkl',
                    'ppe-plus1.pkl',
                    'ppe-plus2.pkl',
                    #'ppe-plus3.pkl',
                    'ppe-plus4.pkl',
                    'ppe-plus5.pkl'
                ]
    else:
        args.dataset_filenames = [
            'ppe-minus13.pkl',
            'ppe-minus12.pkl',
            'ppe-minus11.pkl',
            'ppe-minus10.pkl',
            'ppe-minus9.pkl',
            'ppe-minus8.pkl',
            'ppe-minus7.pkl',
            'ppe-minus6.pkl',
            'ppe-minus5.pkl',
            'ppe-minus4.pkl',
            'ppe-minus3.pkl'
        ]
        if args.include_1p5PN_data:
            args.dataset_filenames += [
                'ppe-minus2.pkl'
            ]
        if args.include_2PN_data:
            args.dataset_filenames += [
                'ppe-minus1.pkl'
            ]
    
    def prep_fn_suppressed(df):
        #df['phases'] = df['phases'].apply(lambda x: np.inf)
        #df['phases'] = df['phases'].apply(lambda x: 0.)
        #df['phases'] = df['phases'].apply(lambda x: x * 1e-10)
        df['phases'] = df['phases'].apply(lambda x: x * 1e10)
        return df
    
    def prep_fn_ignored(df):
        return df
    
    degenerate_filenames = ['ppe-minus0.pkl', 'ppe-plus0.pkl', 'ppe-plus3.pkl'] if args.rescale_2p5and4 else []
    
    args.dataset_prep_fns = [
        prep_fn_ignored if args.rescale_2p5and4_only else get_data_rescale_function(filename, args.rescale_rootdir)
        if filename not in degenerate_filenames
        else prep_fn_suppressed
        for filename in args.dataset_filenames
    ]
    args.dataset_type = PhasingDataset
    args.dataset_kwargs = dict(use_tidal_params=args.include_tidal_params, 
                               use_tidal_params_full=args.include_tidal_params_full)
    args.dataset_n_ppe = None # just use the dephasing provided by the dataset, no scratch construction from the ppE coefficients
    args.dataset_norm_fac = {}
    args.dataset_sample_size = 0.25
    args.dataset_subset_split = [0.8, 0.1, 0.1]
    args.dataset_seed = args.dataset_seed if args.dataset_seed > 0 else None

    resume_cpdir = os.path.join(args.output_rootdir, "checkpoints/{}/".format(args.resume_title))
    resume_filename = "{}_{}-epochs.pt".format(args.resume_title, args.resume_epochs)
    resume_filepath = os.path.join(resume_cpdir, resume_filename)
    os.makedirs(resume_cpdir, exist_ok=True)
    try:
        shutil.copyfile(args.base_network_path, resume_filepath)
    except SameFileError:
        print(f"Base network file already exists at the resume path: {resume_filepath}")
        print("Proceeding with training using the existing file.")
    except Exception as e:
        print(f"Error copying base network file: {e}")
        print(f"Please ensure that the base network file exists at the specified path: {args.base_network_path}")
        raise e

    train(args)



if __name__ == '__main__':
    main()

