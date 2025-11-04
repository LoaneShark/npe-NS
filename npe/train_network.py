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

import PIL.Image
if not hasattr(PIL.Image, 'Resampling'):  # Fix for Pillow<9.0
    PIL.Image.Resampling = PIL.Image

import matplotlib.pyplot as plt
import torch
from torch import nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torch.utils.tensorboard import SummaryWriter
from scipy.interpolate import interp1d

from generate_dataset import _get_pn_coeffs, _get_coeff_bound, _convert_f_to_fgeom, _convert_f_to_fgeom_mc

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
    parser.add_argument("--run-type", type=str, default="BH",
                        help="Toggle BH or NS binaries")
    parser.add_argument("--dataset_seed", type=int, default=1234,
                        help="Set RNG seed for dataset generation.")
    parser.add_argument("--training_seed", type=int, default=-1,
                        help="Set RNG seed for training.")
    parser.add_argument("--data-dim", type=int, default=640,
                        help="Number of frequency points used (dimensionality of input/output layer).")
    parser.add_argument("--include-tidal-params", action=argparse.BooleanOptionalAction, default=False, required=False,
                        help="Include tidal deformation terms in the condition parameter space.")
    parser.add_argument("--include-tidal-params-full", action=argparse.BooleanOptionalAction, default=False, required=False,
                        help="Include tidal and spin induced deformation terms in the condition parameter space.")
    parser.add_argument("--include-tidal-data", action=argparse.BooleanOptionalAction, default=False, required=False,
                        help="Include tidal deformation waveform data, up to 5PN.")
    parser.add_argument("--include-tidal-data-2p5and4", action=argparse.BooleanOptionalAction, default=None, required=False,
                        help="Include tidal deformation waveform data which may be degenerate with waveform parameters, specifically at 2.5PN and 4PN.")
    parser.add_argument("--rescale-2p5and4", action=argparse.BooleanOptionalAction, default=None, required=False,
                        help="Whether or not to suppress latent space support at 2.5PN and 4PN orders.")
    parser.add_argument("--penalize-highPN", action=argparse.BooleanOptionalAction, default=None, required=False,
                        help="Penalize latent space from learning high-PN-like dephasing functions.")
    parser.add_argument("--penalize-nonPN", action=argparse.BooleanOptionalAction, default=None, required=False,
                        help="Penalize latent space from learning PN-like dephasing functions in the non-PN region of latent space.")
    parser.add_argument("--train-recon-phase", action=argparse.BooleanOptionalAction, default=None, required=False,
                        help="Additional training phase including recon error, to facilitate encoder-decoder latent space parity.")
    parser.add_argument("--num-epochs-shape", type=int, default=50,
                        help="Number of epochs to use for training the primary network.")
    parser.add_argument("--num-epochs-scale", type=int, default=50,
                        help="Number of epochs to use for training the secondary network.")
    parser.add_argument("--num-epochs-recon", type=int, default=50,
                        help="Number of epochs to use for training the recon phase.")
    parser.add_argument("--num-epochs-highPN", type=int, default=0,
                        help="Number of epochs (at the end) of the shape train phase to use for penalizing high PN-like behavior in the latent space.")
    parser.add_argument("--num-epochs-nonPN", type=int, default=0,
                        help="Number of epochs (at the end) of the shape train phase to use for penalizing PN-like behavior in the non-PN region of latent space.")
    parser.add_argument("--freqs-using-mc", action='store_true', default=False, 
                        help="Frequency in natural units defined with chirp mass instead of total mass.")
    # set individual training parameters
    parser.add_argument("--train-lr", type=float, default=1e-4,
                        help="Base learning rate to use for training.")
    parser.add_argument("--train-lr-shape", type=float, default=None,
                        help="Base learning rate to use for training. Overrides train-lr if set.")
    parser.add_argument("--train-lr-scale", type=float, default=None,
                        help="Base learning rate to use for training. Overrides train-lr if set.")
    parser.add_argument("--train-lr-recon", type=float, default=None,
                        help="Base learning rate to use for training. Overrides train-lr if set.")
    parser.add_argument("--train-wd", type=float, default=1e-4,
                        help="Weight decay to use for training.")
    parser.add_argument("--train-wd-shape", type=float, default=None,
                        help="Weight decay to use for training. Overrides train-wd if set.")
    parser.add_argument("--train-wd-scale", type=float, default=None,
                        help="Weight decay to use for training. Overrides train-wd if set.")
    parser.add_argument("--train-wd-recon", type=float, default=None,
                        help="Weight decay to use for training. Overrides train-wd if set.")
    parser.add_argument("--train-gamma", type=float, default=0.9,
                        help="Multiplicative learning rate decay factor (<1).")
    parser.add_argument("--train-gamma-shape", type=float, default=None,
                        help="Multiplicative learning rate decay factor (<1). Overrides train-gamma if set.")
    parser.add_argument("--train-gamma-scale", type=float, default=None,
                        help="Multiplicative learning rate decay factor (<1). Overrides train-gamma if set.")
    parser.add_argument("--train-gamma-recon", type=float, default=None,
                        help="Multiplicative learning rate decay factor (<1). Overrides train-gamma if set.")
    parser.add_argument("--train-kl-coeff", type=float, default=1e-6,
                        help="Coefficient to weight KL divergence term in loss function relative to reconstruction error term.")
    parser.add_argument("--train-kl-coeff-shape", type=float, default=None,
                        help="Coefficient to weight KL divergence term in loss function relative to reconstruction error term. Overrides train-kl-coeff if set.")
    parser.add_argument("--train-kl-coeff-scale", type=float, default=None,
                        help="Coefficient to weight KL divergence term in loss function relative to reconstruction error term. Overrides train-kl-coeff if set.")
    parser.add_argument("--train-kl-coeff-recon", type=float, default=None,
                        help="Coefficient to weight KL divergence term in loss function relative to reconstruction error term. Overrides train-kl-coeff if set.")
    parser.add_argument("--train-highPN-coeff", type=float, default=0.,
                        help="Coefficient to weight penalization of high-PN behavior in latent space.")
    parser.add_argument("--train-highPN-coeff-shape", type=float, default=None,
                        help="Coefficient to weight penalization of high-PN behavior in latent space. Overrides train-highPN-coeff if set.")
    parser.add_argument("--train-highPN-coeff-scale", type=float, default=None,
                        help="Coefficient to weight penalization of high-PN behavior in latent space. Overrides train-highPN-coeff if set.")
    parser.add_argument("--train-highPN-coeff-recon", type=float, default=None,
                        help="Coefficient to weight penalization of high-PN behavior in latent space. Overrides train-highPN-coeff if set.")
    parser.add_argument("--train-nonPN-coeff", type=float, default=0.,
                        help="Coefficient to weight penalization of PN-like behavior in non-PN regions of latent space.")
    parser.add_argument("--train-nonPN-coeff-shape", type=float, default=None,
                        help="Coefficient to weight penalization of PN-like behavior in non-PN latent space. Overrides train-highPN-coeff if set.")
    parser.add_argument("--train-nonPN-coeff-scale", type=float, default=None,
                        help="Coefficient to weight penalization of non-PN behavior in non-PN latent space. Overrides train-highPN-coeff if set.")
    parser.add_argument("--train-nonPN-coeff-recon", type=float, default=None,
                        help="Coefficient to weight penalization of non-PN behavior in non-PN latent space. Overrides train-highPN-coeff if set.")
    
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

# TODO: Implement effective cycles recon error
def get_loss_recon_EC(v, v_recon):
    #shape1, scale1 = get_shape_and_scale(v)
    #shape2, scale2 = get_shape_and_scale(v_recon)
    #loss_shape = get_loss_shape(v, scale1*shape2)
    #loss_scale = get_loss_scale(scale1, scale2)

    # TODO: For now, assume h_c ~ (Mf)^(-7/6)
    amplitudes = None
    #amplitudes = model.loggeom_freqs
    asd = None
    #asd = model.asd
    
    if amplitudes is None or asd is None:
        return None
    
    # L_recon (v, v_recon) = \sum_{f} (A(f) / ASD(f)^2) (v(f) - v_recon(f))^2
    loss_recon = np.sum(
        (amplitudes / (asd**2)) * 
        (v - v_recon)**2, axis=-1
    )

    return loss_recon

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

# TODO: Finish implementing scale-function dependence for high-PN terms?
def get_highPN_loss_with_scale(v_recon, l):

    # By default, assume 10Hz detector minimum and IMRPhenom inspiral cutoff maximum frequency bounds
    # TODO: Avoid hardcoding all of these values (incl. frequency grid size and spacing)
    ref_min = _convert_f_to_fgeom(10, l.T[0] + l.T[1])
    ref_max = 0.018
    v_min = (np.pi * ref_min) ** (1/3)
    v_max = (np.pi * ref_max) ** (1/3)
    freqs = np.logspace(np.log10(ref_min), np.log10(ref_max), 640)
    v = (np.pi * freqs) ** (1/3)
    if len(l.T) == 6:
        l1 = l.T[4]
        l2 = l.T[5]
    else:
        l1 = 0.
        l2 = 0.
    v_coeffs, _, _ = _get_pn_coeffs([l.T[0]], l.T[1], l.T[2], l.T[3], l1, l2)

    # Get latent space z samples
    loss_highPN_arr = []
    for k in range(0, 7):
        # Get high-PN ppE-like modifications
        #print('k: ', k)
        coeff_bound = _get_coeff_bound(np.array([k]), v_coeffs, v_min, v_max)
        v_highPN_k = coeff_bound * v ** (k)
        #print('v_highPN_k: ', v_highPN_k.shape)
        loss_highPN_arr.append(F.mse_loss(v_highPN_k, v_recon))
        loss_highPN_arr.append(F.mse_loss(-v_highPN_k, v_recon))

    # Sum over k: MSE of v_recon and v_highPN_k
    loss_highPN = sum(loss_highPN_arr)
    #print('loss_highPN: ', loss_highPN)
    return loss_highPN

def get_highPN_loss(v_recon):
    # By default, assume 10Hz detector minimum and IMRPhenom inspiral cutoff maximum frequency bounds
    # TODO: Avoid hardcoding all of these values (incl. frequency grid size and spacing)
    ref_min = 4e-5
    #ref_min = _convert_f_to_fgeom(10, l.T[0] + l.T[1])
    ref_max = 0.018
    #v_min = (np.pi * ref_min) ** (1/3)
    #v_max = (np.pi * ref_max) ** (1/3)
    num_f = v_recon.shape[2]
    #freqs = np.linspace(ref_min, ref_max, num_f)
    freqs = np.logspace(np.log10(ref_min), np.log10(ref_max), num_f)
    v_0 = (np.pi * freqs) ** (1/3)
    #print('l: ', l.T)
    #print('l[0]: ', l.T[0])
    #if len(l.T) == 6:
    #    l1 = l.T[4]
    #    l2 = l.T[5]
    #else:
    #    l1 = 0.
    #    l2 = 0.
    #v_coeffs, _, _ = _get_pn_coeffs([l.T[0]], l.T[1], l.T[2], l.T[3], l1, l2)
    #print('v_coeffs: ', len(v_coeffs))
    #print(v_coeffs)
    #print('v_recon: ', v_recon.shape)
    #print('v_recon: ', v_recon.dtype)
    #print('v_recon: ', v_recon)

    #shape1, scale1 = get_shape_and_scale(v_recon)
    shape1, scale1 = get_shape_and_scale(torch.abs(v_recon))

    # Get latent space z samples
    loss_highPN_arr = []
    signs = [1.] 
    #signs = [-1., 1.]
    loss_shape_weights = {0: 100., 1: 0.01, 2: 0.01, 3: 100., 4: 0.01, 5: 0.01, 6: 0.1}
    #loss_shape_weights = {}
    for k in range(0, 8):
        # Get high-PN ppE-like modifications
        #print('k: ', k)
        #coeff_bound = _get_coeff_bound(np.array([k]), v_coeffs, v_min, v_max)
        coeff_bound = 1.
        for s in signs:
            #print('s:  +' if s == 1 else 's: -')
            #print('v: ', v.shape)
            v_highPN_k = torch.tensor(s * coeff_bound * v_0 ** k)
            #v_highPN_k = v_highPN_k.resize(v_recon.shape[0], 1, num_f)
            v_highPN = v_highPN_k.unsqueeze(0).repeat(v_recon.shape[0], 1)
            v_highPN.resize(v_recon.shape[0], 1, num_f)

            shape2, scale2 = get_shape_and_scale(v_highPN)
            shape2 = shape2.reshape(v_recon.shape[0], 1, num_f)
            #print('v_recon: ', v_recon.shape)
            #print('v_highPN_k: ', v_highPN_k.shape)
            #print('v_highPN: ', v_highPN.shape)
            #print('shape_1: ', shape1.shape)
            #print('shape_2: ', shape2.shape)
            v_loss_highPN = get_loss_shape(shape1, shape2)
            loss_shape_coeff = 1. if k not in loss_shape_weights else loss_shape_weights[k]
            #print('loss_highPN: ', v_loss_highPN)
            loss_highPN_arr.append(loss_shape_coeff * v_loss_highPN)
            #print('v_highPN_k')
            #print(v_highPN_k)
            #print('shape_2')
            #print(shape2)

    # Sum over k: MSE of v_recon and v_highPN_k
    #print(loss_highPN_arr[0].shape)
    #print(loss_highPN_arr)
    #loss_highPN = 1. / sum(loss_highPN_arr)
    #loss_highPN = sum(1. / loss_highPN_arr)
    #loss_highPN = sum(loss_highPN_arr)
    #print('loss_highPN: ', loss_highPN)

    loss_highPN_vals = 1. / torch.tensor(loss_highPN_arr)
    loss_highPN = torch.sum(loss_highPN_vals)

    return loss_highPN

def get_nonPN_loss(v_recon):
    if v_recon.shape[0] <= 0:
        return torch.tensor(0.0)
    else:
        norm_fac = 1. / v_recon.shape[0]

    # By default, assume 10Hz detector minimum and IMRPhenom inspiral cutoff maximum frequency bounds
    # TODO: Avoid hardcoding all of these values (incl. frequency grid size and spacing)
    ref_min = 4e-5
    #ref_min = _convert_f_to_fgeom(10, l.T[0] + l.T[1])
    ref_max = 0.018
    #v_min = (np.pi * ref_min) ** (1/3)
    #v_max = (np.pi * ref_max) ** (1/3)
    num_f = v_recon.shape[2]
    #freqs = np.linspace(ref_min, ref_max, num_f)
    freqs = np.logspace(np.log10(ref_min), np.log10(ref_max), num_f)
    v_0 = (np.pi * freqs) ** (1/3)

    shape1, scale1 = get_shape_and_scale(torch.abs(v_recon))

    coeff_bound = 1.
    signs = [1.]
    #loss_nonPN_arr = []
    k_min = -14
    #k_max = 6
    k_max = 0
    k_num = k_min - k_max - 1
    loss_nonPN_arr = []
    #loss_nonPN_vals = torch.zeros(size=(v_recon.shape[0], 1, k_num))
    #for k in range(-14, 6):
    for k_idx, k in enumerate(range(k_min, k_max)):
        for s in signs:
            v_nonPN_k = torch.tensor(s * coeff_bound * v_0 ** k)
            v_nonPN = v_nonPN_k.unsqueeze(0).repeat(v_recon.shape[0], 1)
            v_nonPN.resize(v_recon.shape[0], 1, num_f)

            shape2, scale2 = get_shape_and_scale(v_nonPN)
            shape2 = shape2.reshape(v_recon.shape[0], 1, num_f)
            
            v_loss_nonPN = get_loss_shape(shape1, shape2)
            #print('shape1: ', shape1.shape)
            #print('shape2: ', shape2.shape)
            #print('v_loss_nonPN.shape: ', v_loss_nonPN.shape)
            #print('v_loss_nonPN: ', v_loss_nonPN)
            
            loss_nonPN_arr.append(v_loss_nonPN)

    loss_nonPN_vals = torch.tensor(loss_nonPN_arr)
    #print('loss_nonPN_vals.shape: ', loss_nonPN_vals.shape)
    #print('loss_nonPN_vals: ', loss_nonPN_vals)
    #loss_nonPN[v_is_nonPN] = loss_nonPN[v_is_nonPN]
    #loss_nonPN_vals[v_is_nonPN] = 1. / loss_nonPN_vals[v_is_nonPN]
    #loss_nonPN_vals[~v_is_nonPN] = 0.0
    loss_nonPN_vals = 1. / loss_nonPN_vals
    #loss_nonPN = 1. / sum(loss_nonPN_arr)
    loss_nonPN = norm_fac * torch.sum(loss_nonPN_vals)
    #print('loss_nonPN: ', loss_nonPN)
    return loss_nonPN

def check_nonPN_region(model, l, v_recon, mu, logvar):
    # By default, assume 10Hz detector minimum and IMRPhenom inspiral cutoff maximum frequency bounds
    # TODO: Avoid hardcoding all of these values (incl. frequency grid size and spacing)
    ref_min = 4e-5
    #ref_min = _convert_f_to_fgeom(10, l.T[0] + l.T[1])
    ref_max = 0.018
    #v_min = (np.pi * ref_min) ** (1/3)
    #v_max = (np.pi * ref_max) ** (1/3)
    num_f = v_recon.shape[2]
    #freqs = np.linspace(ref_min, ref_max, num_f)
    freqs = np.logspace(np.log10(ref_min), np.log10(ref_max), num_f)
    v_0 = torch.tensor((np.pi * freqs) ** (1/3))
    v_0 = v_0.unsqueeze(0).repeat(v_recon.shape[0], 1)
    v_0.resize(v_recon.shape[0], 1, num_f)

    #print('v_recon.shape: ', v_recon.shape)
    #print('v_0.shape: ', v_recon.shape)
    #print('l.shape: ', l.shape)
    #print('mu.shape: ', mu.shape)
    #print('logvar.shape: ', logvar.shape)

    v_m1_pos = 1. * v_0 ** (-1)
    v_m1_pos.resize(v_recon.shape[0], 1, num_f)
    #print('v_m1_pos.shape: ', v_m1_pos.shape)
    mu_m1p, logvar_m1p = model.encoder(v_m1_pos, l)

    v_m1_neg = -1. * v_0 ** (-1)
    v_m1_neg.resize(v_recon.shape[0], 1, num_f)
    mu_m1n, logvar_m1n = model.encoder(v_m1_neg, l)

    v_m13_pos = 1. * v_0 ** (-13)
    v_m13_pos.resize(v_recon.shape[0], 1, num_f)
    mu_m13p, logvar_m13p = model.encoder(v_m13_pos, l)

    v_m13_neg = -1. * v_0 ** (-13)
    v_m13_neg.resize(v_recon.shape[0], 1, num_f)
    mu_m13n, logvar_m13n = model.encoder(v_m13_neg, l)
    
    #is_nonPN_v = torch.zeros_like(mu[:,:,0])
    #is_nonPN_v = torch.zeros(size=mu.shape[:-1])
    is_nonPN_v = torch.full(size=mu.shape[:-1], fill_value=False)

    for mu_min, logvar_min, mu_max, logvar_max in [(mu_m1p, logvar_m1p, mu_m13n, logvar_m13n), 
                                                   (mu_m13p, logvar_m13p, mu_m1n, logvar_m1n)]:
        #theta1 = torch.arctan(torch.sqrt(mu_min[:,0]**2 + mu_min[:,1]**2))
        theta1 = torch.arcsin(mu_min[:,1] / torch.sqrt(mu_min[:,0]**2 + mu_min[:,1]**2))
        dtheta1 = torch.arctan(torch.sqrt((10.**logvar_min[:,0])**2 + (10.**logvar_min[:,1])**2))
        theta1_min = (theta1 - dtheta1) % (2*np.pi)
        theta1_max = (theta1 + dtheta1) % (2*np.pi)
        #theta2 = torch.arctan(torch.sqrt(mu_max[:,0]**2 + mu_max[:,1]**2))
        theta2 = torch.arcsin(mu_max[:,1] / torch.sqrt(mu_max[:,0]**2 + mu_max[:,1]**2))
        dtheta2 = torch.arctan(torch.sqrt((10.**logvar_max[:,0])**2 + (10.**logvar_max[:,1])**2))
        theta2_min = (theta2 - dtheta2) % (2*np.pi)
        theta2_max = (theta2 + dtheta2) % (2*np.pi)

        # assume nonPN region is < pi / 2
        #zero_cross = (torch.sign(mu_min[:,:,0]) < 0 & torch.sign(mu_min[:,:,1]) < 0 \
        #                   & torch.sign(mu_max[:,:,0]) > 0 & torch.sign(mu_max[:,:,1]) > 0)
        # assume nonPN region is < pi / 2
        zero_cross = ((theta1_min > theta2_max % (2*np.pi)) \
                        & (((theta1_min + np.pi/2.) % (2*np.pi)) < ((theta2_max + np.pi/2.) % (2*np.pi))) \
                        & (((theta1_min - np.pi/2.) % (2*np.pi)) < ((theta2_max - np.pi/2.) % (2*np.pi))) \
                        & (((theta1_min + np.pi)    % (2*np.pi)) < ((theta2_max + np.pi)    % (2*np.pi)))) | \
                     ((theta1_max < theta2_min % (2*np.pi)) \
                        & (((theta1_max + np.pi/2.) % (2*np.pi)) > ((theta2_min + np.pi/2.) % (2*np.pi))) \
                        & (((theta1_max - np.pi/2.) % (2*np.pi)) > ((theta2_min - np.pi/2.) % (2*np.pi))) \
                        & (((theta1_max + np.pi)    % (2*np.pi)) > ((theta2_min + np.pi)    % (2*np.pi))))

        # theta1 < theta2
        sign_mask = ((theta1_max < theta2_min) & ~zero_cross) | ((theta1_min > theta2_max) & zero_cross)

        theta_ref_min = theta1_max
        theta_ref_min[~sign_mask] = theta2_max[~sign_mask]
        theta_ref_max = theta2_min
        theta_ref_max[~sign_mask] = theta1_min[~sign_mask]

        z_v = torch.sqrt(mu[:,0]**2 + mu[:,1]**2)
        dz_v = torch.sqrt((10.**logvar[:,0])**2 + (10.**logvar[:,1])**2)
        #print('z_v.shape: ', z_v.shape)
        #print('z_v: ', z_v)
        z_nonGR = z_v > dz_v
        #z_nonGR = z_v > 0
        #theta_v = torch.full_like(z_v, False, dtype=bool)
        theta_v = torch.zeros_like(z_v)
        theta_v[z_nonGR] = (torch.arcsin(mu[:,1] / z_v)[z_nonGR]) % (2*np.pi)
        #print('theta_v.shape: ', theta_v.shape)
        #print('theta_v: ', theta_v)
        dtheta_v = torch.zeros_like(z_v)
        dtheta_v[z_nonGR] = torch.arctan(dz_v / z_v)[z_nonGR]
        #print('dtheta_v.shape: ', dtheta_v.shape)
        #print('dtheta_v: ', dtheta_v)

        theta_v_min = (theta_v - dtheta_v) % (2*np.pi)
        theta_v_max = (theta_v + dtheta_v) % (2*np.pi)
        #theta_v_min = theta_v
        #theta_v_max = theta_v

        theta_nonPN_mask = z_nonGR & \
                            (~zero_cross & (theta_v_max < theta_ref_max) & (theta_v_min > theta_ref_min) | \
                            (zero_cross & (((theta_v_max + np.pi) % (2*np.pi)) < ((theta_ref_max + np.pi) % (2*np.pi))) & \
                                          (((theta_v_min + np.pi) % (2*np.pi)) > ((theta_ref_min + np.pi) % (2*np.pi)))))
        
        is_nonPN_v[theta_nonPN_mask] = True

    #print('mu.shape: ', mu.shape)
    #print('mu: ', mu)
    #print('logvar.shape: ', logvar.shape)
    #print('logvar: ', logvar)

    #print('is_nonPN_v.shape: ', is_nonPN_v.shape)
    #print('any is_nonPN_v: ', torch.any(is_nonPN_v))
    #print('is_nonPN_v: ', is_nonPN_v)

    #raise ValueError

    return is_nonPN_v


def vae_loss_fn(model: VAE , v, l, t, *args, 
                kl_coeff=1., 
                shape_coeff=1., scale_coeff=1., 
                recon_coeff=1., recon_scale_coeff=1.,
                nonPN_coeff=1., highPN_coeff=1.,
                with_mu=True, recon_use_mse=False,
                **kwargs):
    v, l, t = align_data_format_with_model(model, v, l, t)

    #print('v.shape: ', v.shape)
    #print('l.shape: ', l.shape)
    #print('t.shape: ', t.shape)
    
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
    
    # Penalize high-order PN-like dephasing behavior across latent space
    if highPN_coeff != 0.:
        #loss_highPN = get_highPN_loss_with_scale(v_recon, l)
        loss_highPN = highPN_coeff * get_highPN_loss(v_recon)
    else:
        loss_highPN = torch.tensor(0.)

    # Penalize PN-like dephasing behavior in the non-PN region of latent space only
    if nonPN_coeff != 0.:
        v_is_nonPN = check_nonPN_region(model, l, v_recon, mu, logvar)
        loss_nonPN = nonPN_coeff * get_nonPN_loss(v_recon[v_is_nonPN])
    else:
        loss_nonPN = torch.tensor(0.)

    #print('loss_kld: ', loss_kld.shape)
    #print('loss_kld: ', loss_kld)
    #print('loss_var: ', loss_var.shape)
    #print('loss_var: ', loss_var)
    #print('loss_recon: ', loss_recon.shape)
    #print('loss_recon: ', loss_recon)
    #print('loss_highPN: ', loss_highPN.shape)
    #print('loss_highPN: ', loss_highPN)
    #print('loss_nonPN: ', loss_nonPN.shape)
    #print('loss_nonPN: ', loss_nonPN)
        
    loss = loss_kld + loss_var + loss_recon + loss_highPN + loss_nonPN
    return loss

# TODO: Modify vae loss function to use effective cycles as reconstruction loss
def vae_loss_fn_EC(model, v, l, t, *args, 
                   kl_coeff=1., 
                   shape_coeff=1., scale_coeff=1., 
                   recon_coeff=1., recon_scale_coeff=1.,
                   with_mu=True, recon_use_mse=False,
                   recon_use_effective_cycles=False,
                   **kwargs):
    v, l, t = align_data_format_with_model(model, v, l, t)
    
    # Run encoder to get mu and variance of latent space
    # Note: v is the dephasing function, l is the labels (e.g., intrinsic parameters)
    mu, logvar = model.encoder(v, l)

    # KL divergence loss
    loss_kld = kl_div_diagonal_gaussian_to_standard_gaussian(mu, logvar, dim=-1, with_mu=with_mu)
    loss_kld = kl_coeff * torch.mean(loss_kld)
    
    # Sample from the latent space
    z = model.varier(mu, logvar)

    # Reconstruction from the latent space sample
    v_recon = model.decoder(z, l)

    if recon_use_effective_cycles:
        # Use effective cycles reconstruction loss
        loss_recon = get_loss_recon_EC(v, v_recon)
    else:
        # Sample reconstruction loss
        # Note: v_recon_var is the reconstruction of the dephasing function from the latent space sample (z)
        v_recon_var = v_recon
        loss_shape, loss_scale = get_loss_recon_var(v, v_recon_var)
        loss_recon_var = shape_coeff * loss_shape + scale_coeff * loss_scale

        # Mean reconstruction loss
        # Note: v_recon_mean is the reconstruction of the dephasing function from the latent space mean (mu)
        v_recon_mean = model.decoder(mu, l)
        if recon_use_mse:
            loss_recon_mean = recon_coeff * F.mse_loss(v, v_recon_mean)
        else:
            loss_shape_recon, loss_scale_recon = get_loss_recon(v, v_recon_mean)
            loss_recon_mean = recon_coeff * loss_shape_recon + recon_scale_coeff * loss_scale_recon
        
        loss_recon = loss_recon_var + loss_recon_mean

    loss = loss_kld + loss_recon
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
    loss_highPN = get_highPN_loss(v_recon)
    v_is_nonPN = check_nonPN_region(model, l, v_recon, mu, logvar)
    loss_nonPN = get_nonPN_loss(v_recon[v_is_nonPN])
    
    metrics = dict(
        kl_div=loss_kld,
        err_recon=loss_recon,
        err_shape_rescaled=loss_shape_recon,
        err_shape=loss_shape,
        err_scale=loss_scale,
        err_highPN=loss_highPN,
        err_nonPN=loss_nonPN,
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

    # Train primary network

    # args.run_title = "npE_network"
    args.resume_title = args.run_title
    args.resume_epochs = 0
    args.add_epochs = args.num_epochs_shape
    args.nonPN_epochs = args.num_epochs_nonPN if args.penalize_nonPN else 0.
    args.epochs_per_latent_plot = [(10, 1), (None, 10)]
    args.epochs_per_checkpoint = min(50, args.num_epochs_shape, args.num_epochs_scale)
    args.optimizer_override = True
    args.scheduler_override = True

    args.batch_size_train = 64
    args.batch_size_val = 1024
    args.batches_per_summary = 0.1

    args.penalize_highPN = bool(args.penalize_highPN)

    args.lr = args.train_lr_shape if args.train_lr_shape is not None else args.train_lr
    args.wd = args.train_wd_shape if args.train_wd_shape is not None else args.train_wd
    args.gamma = args.train_gamma_shape if args.train_gamma_shape is not None else args.train_gamma
    args.kl_coeff = args.train_kl_coeff_shape if args.train_kl_coeff_shape is not None else args.train_kl_coeff
    args.highPN_coeff = (args.train_highPN_coeff_shape if args.train_highPN_coeff_shape is not None else args.train_highPN_coeff) if args.penalize_highPN else 0.
    args.nonPN_coeff = (args.train_nonPN_coeff_shape if args.train_nonPN_coeff_shape is not None else args.train_nonPN_coeff) if args.penalize_nonPN else 0.
    args.loss_kwargs = dict(
        kl_coeff=args.kl_coeff, 
        shape_coeff=1., scale_coeff=0., 
        recon_coeff=0., recon_scale_coeff=0.,
        #highPN_coeff=args.highPN_coeff,
        highPN_coeff=[(max(1, args.num_epochs_shape - args.num_epochs_highPN), 0.), (None, args.highPN_coeff)],
        nonPN_coeff=[(max(1, args.num_epochs_shape - args.num_epochs_nonPN), 0.), (None, args.nonPN_coeff)],
        with_mu=False, recon_use_mse=False, 
    )
    args.diagnosis_kwargs = dict(with_mu=False)

    if args.run_type in ['NS', 'NSBH', 'CBC']:
        data_dim = args.data_dim
        if args.include_tidal_params_full:
            cond_dim = 8
        elif args.include_tidal_params:
            cond_dim = 6
        else:
            cond_dim = 4
        grid_dim = 2
        depth = 4
        width = 512
    elif args.run_type == 'BH':
        data_dim = args.data_dim
        cond_dim = 4
        grid_dim = 2
        depth = 4
        width = 512
    else:
        raise ValueError('run_type should be one of: \'NS\', \'BH\'')
    
    args.structure_kwargs = dict(
        depth=depth, width=width,
        data_dim=data_dim, grid_dim=grid_dim,
        cond_dim=cond_dim,
        )

    args.model_type = VAE
    args.model_kwargs = dict(
        **args.structure_kwargs,
        freeze_scale=True,
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
            'ppe-minus2.pkl',
            'ppe-minus1.pkl'
        ]
        if args.include_tidal_data:
            if args.include_tidal_data_2p5and4 is None or bool(args.include_tidal_data_2p5and4):
                args.dataset_filenames += [
                    'ppe-minus0.pkl',
                    'ppe-plus1.pkl',
                    'ppe-plus2.pkl',
                    'ppe-plus3.pkl',
                    'ppe-plus4.pkl',
                    'ppe-plus5.pkl'
                ]
                if bool(args.rescale_2p5and4):
                    def prep_fn_suppressed(df):
                        #df['phases'] = df['phases'].apply(lambda x: np.inf)
                        #df['phases'] = df['phases'].apply(lambda x: 0.)
                        df['phases'] = df['phases'].apply(lambda x: x * 1e-8)
                        return df
                    
                    args.dataset_prep_fns = [lambda x:x] * len(args.dataset_filenames)
                    args.dataset_prep_fns[args.dataset_filenames.index('ppe-minus0.pkl')] = prep_fn_suppressed
                    args.dataset_prep_fns[args.dataset_filenames.index('ppe-plus3.pkl')]  = prep_fn_suppressed
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
            'ppe-minus3.pkl',
            'ppe-minus2.pkl',
            'ppe-minus1.pkl'
        ]
        if args.include_tidal_data:
            args.dataset_filenames += [
                #'ppe-minus0.pkl',
                #'ppe-plus1.pkl',
                #'ppe-plus2.pkl',
                #'ppe-plus3.pkl',
                #'ppe-plus4.pkl',
                #'ppe-plus5.pkl'
            ]
    args.dataset_type = PhasingDataset
    args.dataset_kwargs = {'use_tidal_params': args.include_tidal_params, 'use_tidal_params_full': args.include_tidal_params_full}
    args.dataset_n_ppe = 1
    args.dataset_norm_fac = {}
    args.dataset_sample_size = 0.25
    args.dataset_subset_split = [0.8, 0.1, 0.1]
    args.dataset_seed = args.dataset_seed if args.dataset_seed > 0 else None

    train(args)


    # ===================

    # Train secondary network

    # args.run_title = "npE_network"
    args.resume_title = args.run_title
    args.resume_epochs = args.num_epochs_shape
    args.add_epochs = args.num_epochs_scale
    args.epochs_per_latent_plot =  min(50, args.num_epochs_shape, args.num_epochs_scale)
    args.epochs_per_checkpoint =  min(50, args.num_epochs_shape, args.num_epochs_scale)
    args.optimizer_override = True
    args.scheduler_override = True

    args.batch_size_train = 64
    args.batch_size_val = 1024
    args.batches_per_summary = 0.1

    args.show_plot = False

    args.lr = args.train_lr_scale if args.train_lr_scale is not None else args.train_lr
    args.wd = args.train_wd_scale if args.train_wd_scale is not None else args.train_wd
    args.gamma = args.train_gamma_scale if args.train_gamma_scale is not None else args.train_gamma
    args.kl_coeff = args.train_kl_coeff_scale if args.train_kl_coeff_scale is not None else 0.
    args.highPN_coeff = args.train_highPN_coeff_scale if args.train_highPN_coeff_scale is not None else 0.
    args.nonPN_coeff = args.train_nonPN_coeff_scale if args.train_nonPN_coeff_scale is not None else 0.
    args.loss_kwargs = dict(
        kl_coeff=args.kl_coeff, 
        shape_coeff=0., scale_coeff=0., 
        recon_coeff=0., recon_scale_coeff=1.,
        highPN_coeff=args.highPN_coeff,
        nonPN_coeff=args.nonPN_coeff,
        with_mu=False, recon_use_mse=False, 
    )
    args.diagnosis_kwargs = dict(with_mu=False)

    args.model_type = VAE
    #args.model_kwargs = dict(
    #    depth=4, width=512, 
    #    data_dim=640, grid_dim=2, 
    #    freeze_shape=True,
    #)

    args.model_kwargs = dict(
        **args.structure_kwargs,
        freeze_shape=True,
    )

    train(args)


    # ===================

    # Train on reconstruction error

    train_recon_phase = args.train_recon_phase and args.num_epochs_recon > 0.
    if train_recon_phase:

        # args.run_title = "npE_network"
        args.resume_title = args.run_title
        args.resume_epochs = args.num_epochs_shape + args.num_epochs_scale
        args.add_epochs = args.num_epochs_recon
        args.epochs_per_latent_plot =  min(50, args.num_epochs_shape, args.num_epochs_scale, args.num_epochs_recon)
        args.epochs_per_checkpoint =  min(50, args.num_epochs_shape, args.num_epochs_scale, args.num_epochs_recon)
        args.optimizer_override = True
        args.scheduler_override = True

        args.batch_size_train = 64
        args.batch_size_val = 1024
        args.batches_per_summary = 0.1

        args.show_plot = False

        args.lr = args.train_lr_recon if args.train_lr_recon is not None else args.train_lr
        args.wd = args.train_wd_recon if args.train_wd_recon is not None else args.train_wd
        args.gamma = args.train_gamma_recon if args.train_gamma_recon is not None else args.train_gamma
        args.kl_coeff = args.train_kl_coeff_recon if args.train_kl_coeff_recon is not None else 0.
        args.highPN_coeff = (args.train_highPN_coeff_recon if args.train_highPN_coeff_recon is not None else args.train_highPN_coeff) if args.penalize_highPN else 0.
        args.nonPN_coeff = (args.train_nonPN_coeff_recon if args.train_nonPN_coeff_recon is not None else args.train_nonPN_coeff) if args.penalize_nonPN else 0.
        args.loss_kwargs = dict(
            kl_coeff=args.kl_coeff, 
            shape_coeff=0.5, scale_coeff=0., 
            recon_coeff=1., recon_scale_coeff=0.5,
            highPN_coeff=args.highPN_coeff,
            nonPN_coeff=args.nonPN_coeff,
            with_mu=False, recon_use_mse=False, 
        )
        args.diagnosis_kwargs = dict(with_mu=False)

        args.model_type = VAE
        #args.model_kwargs = dict(
        #    depth=4, width=512, 
        #    data_dim=640, grid_dim=2, 
        #    freeze_shape=True,
        #)

        args.model_kwargs = dict(
            **args.structure_kwargs,
            freeze_shape=True,
        )

        train(args)

if __name__ == '__main__':
    main()

