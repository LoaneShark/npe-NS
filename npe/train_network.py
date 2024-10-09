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


def get_cli():
    parser = argparse.ArgumentParser()
    # source parameters
    parser.add_argument("--dataset-rootdir", type=str,
                        help="Dataset directory")
    parser.add_argument("--output-rootdir", type=str,
                        help="Output directory")
    parser.add_argument("--run-title", type=str, default="npe",
                        help="Title of the run")
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
    loss_kld = kl_div_diagonal_gaussian_to_standard_gaussian(mu, logvar, dim=-1, with_mu=with_mu)
    loss_kld = kl_coeff * torch.mean(loss_kld)
    
    z = model.varier(mu, logvar)
    v_recon_var = model.decoder(z, l)
    loss_shape, loss_scale = get_loss_recon_var(v, v_recon_var)
    loss_var = shape_coeff * loss_shape + scale_coeff * loss_scale

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

def main():
    
    args = get_cli()

    # args = type('TrainingArguments', (), {})

    # args.run_title = "npE_network"
    args.resume_title = args.run_title
    args.resume_epochs = 0
    args.add_epochs = 50
    args.epochs_per_latent_plot = [(10, 1), (None, 10)]
    args.epochs_per_checkpoint = 50
    args.optimizer_override = True
    args.scheduler_override = True

    args.batch_size_train = 64
    args.batch_size_val = 1024
    args.batches_per_summary = 0.1

    args.lr = 1e-4
    args.wd = 1e-4
    args.gamma = 0.9
    args.loss_kwargs = dict(
        kl_coeff=1e-6, 
        shape_coeff=1., scale_coeff=0., 
        recon_coeff=0., recon_scale_coeff=0.,
        with_mu=False, recon_use_mse=False, 
    )
    args.diagnosis_kwargs = dict(with_mu=False)

    args.model_type = VAE
    args.model_kwargs = dict(
        depth=4, width=512, 
        data_dim=640, grid_dim=2, 
        freeze_scale=True,
    )
    args.optimizer_type = torch.optim.AdamW
    args.scheduler_type = torch.optim.lr_scheduler.ExponentialLR
    args.loss_fn = vae_loss_fn
    args.diagnosis_fn = vae_diagnosis_fn
    args.training_device = None
    args.training_seed = None

    args.npoints_for_latent_plot = int(1e2)
    args.npoints_for_generation = 16
    args.show_plot = False

    args.dataset_recipe_from_file = None
    args.dataset_recipe_save_file = None
    args.dataset_filenames = [
        "ppe-minus13.pkl",
        "ppe-minus11.pkl",
        "ppe-minus9.pkl",
        "ppe-minus7.pkl",
        "ppe-minus5.pkl",
        "ppe-minus3.pkl",
        "ppe-minus1.pkl",
    ]
    args.dataset_type = PhasingDataset
    args.dataset_n_ppe = 1
    args.dataset_norm_fac = {}
    args.dataset_sample_size = 0.25
    args.dataset_subset_split = [0.8, 0.1, 0.1]
    args.dataset_seed = 1234

    train(args)


    # ===================

    # args.run_title = "npE_network"
    args.resume_title = args.run_title
    args.resume_epochs = 50
    args.add_epochs = 50
    args.epochs_per_latent_plot = 50
    args.epochs_per_checkpoint = 50
    args.optimizer_override = True
    args.scheduler_override = True

    args.batch_size_train = 64
    args.batch_size_val = 1024
    args.batches_per_summary = 0.1

    args.show_plot = False

    args.lr = 1e-4
    args.wd = 1e-4
    args.gamma = 0.9
    args.loss_kwargs = dict(
        kl_coeff=0., 
        shape_coeff=0., scale_coeff=0., 
        recon_coeff=0., recon_scale_coeff=1.,
        with_mu=False, recon_use_mse=False, 
    )
    args.diagnosis_kwargs = dict(with_mu=False)

    args.model_type = VAE
    args.model_kwargs = dict(
        depth=4, width=512, 
        data_dim=640, grid_dim=2, 
        freeze_shape=True,
    )

    train(args)



if __name__ == '__main__':
    main()

