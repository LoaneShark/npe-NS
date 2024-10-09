import numpy as np

import torch
from torch import nn
import torch.nn.functional as F

import lal
import lalsimulation
import bilby

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

    def extended_decoder(self, z, cond, xin=None):
        batch_size = z.shape[0]
        x1, x2 = self.grid_decoder(z, cond)
        if xin is None:
            xin = torch.linspace(0, 1, self.data_dim, device=x1.device)
            xin = xin.repeat(batch_size, 1)
        xout_comp = x1[:,None,:] * torch.exp(x2[:,None,:] * xin[:,:,None])
        xout = torch.sum(xout_comp, dim=-1).view(batch_size, 1, -1)
        dext = torch.sum(x2 * xout_comp[:,-1,:], dim=-1).view(batch_size, 1)
        return xout, dext
    
    def decoder(self, z, cond, xin=None):
        xout, _ = self.extended_decoder(z, cond, xin)
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
    


MSUN_KM = lal.MSUN_SI * lal.G_SI / lal.C_SI ** 2 / 1e3
MSUN_S  = MSUN_KM / lal.C_SI * 1e3
    
class PhaseModificationAnalysis:
    device = torch.device('cpu')

    def __init__(self, filepath, model_kwargs, norm_fac=1.):
        self.model = VAE(**model_kwargs).to(self.device)
        self.model.load_state_dict(torch.load(filepath, map_location=self.device)['model'])
        self.model.eval()
        self.model.train(False)
        self.model_loggeom_freqs = np.linspace(np.log10(4e-4), np.log10(1.8e-2), 640)
        self.norm_fac = norm_fac

    @classmethod
    def gw_params_to_vae_labels(cls, mass_1, mass_2, chi_1, chi_2):
        mc = bilby.gw.conversion.component_masses_to_chirp_mass(mass_1, mass_2)
        q = mass_2 / mass_1
        chi_sym = 0.5 * (chi_1 + chi_2)
        chi_asym = 0.5 * (chi_1 - chi_2)
        labels = [np.log(mc), q, chi_sym, chi_asym]
        return labels
    
    @classmethod
    def vae_labels_to_gw_params(cls, labels):
        mc = np.exp(labels[0])
        q = labels[1]
        chi_sym = labels[2]
        chi_asym = labels[3]
        m1, m2 = bilby.gw.conversion.chirp_mass_and_mass_ratio_to_component_masses(mc, q)
        chi1 = chi_sym + chi_asym
        chi2 = chi_sym - chi_asym
        return m1, m2, chi1, chi2
    
    def phase_mod(self, freqs, mass_1, mass_2, chi_1, chi_2, z_1, z_2):
        z_abs = np.sqrt(z_1*z_1 + z_2*z_2)
        if z_abs == 0.:
            return np.zeros_like(freqs)
        z = torch.tensor([z_1/z_abs, z_2/z_abs], dtype=torch.float32, device=self.device).view(1, -1)
        l = self.gw_params_to_vae_labels(mass_1, mass_2, chi_1, chi_2)
        l = torch.tensor(l, dtype=torch.float32, device=self.device).view(1, -1)
        geom_freqs = freqs * (mass_1 + mass_2) * MSUN_S
        geom_freqs_cutoff = 10**self.model_loggeom_freqs[-1]
        # mask_low = geom_freqs < 10**self.model_loggeom_freqs[0]
        # mask_high = geom_freqs > 10**self.model_loggeom_freqs[-1]
        mask_low = geom_freqs <= 0
        mask_high = geom_freqs > geom_freqs_cutoff
        mask_mid = ~(mask_low|mask_high)
        loggeom_freqs = np.log10(np.append(geom_freqs[mask_mid], geom_freqs_cutoff))
        loggeom_freq_range = self.model_loggeom_freqs[-1] - self.model_loggeom_freqs[0]
        xin = (loggeom_freqs - self.model_loggeom_freqs[0]) / loggeom_freq_range
        xin = torch.tensor(xin, dtype=torch.float32, device=self.device).view(1, -1)
        xout, dext = self.model.extended_decoder(z, l, xin)
        xout, dext = xout.view(-1).cpu().detach().numpy().flatten(), dext.item()
        phases_mod = np.zeros_like(geom_freqs)
        phases_mod[mask_mid] = xout[:-1] - xout[-1] - dext / loggeom_freq_range / np.log(10) * (geom_freqs[mask_mid] / geom_freqs_cutoff - 1.)
        phases_mod *= z_abs * self.norm_fac
        return phases_mod

    def extract_latent(self, phase_func, mass_1, mass_2, chi_1, chi_2):
        mtot = mass_1 + mass_2
        freqs = 10**self.model_loggeom_freqs / mtot / MSUN_S
        phases = phase_func(freqs) / self.norm_fac
        norm = np.sqrt(np.mean(phases*phases, axis=-1))
        if norm == 0.:
            return 0., 0.
        phases = torch.tensor(phases, dtype=torch.float32, device=self.device).view(1, 1, -1)
        labels = self.gw_params_to_vae_labels(mass_1, mass_2, chi_1, chi_2)
        labels = torch.tensor(labels, dtype=torch.float32, device=self.device).view(1, -1)
        z, _ = self.model.encoder(phases, labels)
        norm_std = self.model.decoder(z, labels)
        norm_std = torch.sqrt(torch.mean(norm_std*norm_std, dim=-1)).squeeze().item()
        z_1, z_2 = z.cpu().detach().numpy().flatten() * norm / norm_std
        return z_1, z_2
