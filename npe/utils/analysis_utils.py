import warnings
warnings.filterwarnings('ignore', message='You are using `torch.load` with `weights_only=False`*', category=FutureWarning)

import os
import sys
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import torch
from scipy.interpolate import interp1d

import lal
import bilby
from bilby.gw.conversion import (
    component_masses_to_chirp_mass, 
    component_masses_to_symmetric_mass_ratio,
    chirp_mass_and_mass_ratio_to_component_masses,
    generate_all_bbh_parameters, generate_all_bns_parameters,
    generate_mass_parameters, generate_spin_parameters, generate_tidal_parameters,
    lambda_1_lambda_2_to_lambda_tilde, lambda_1_lambda_2_to_delta_lambda_tilde,
    lambda_tilde_to_lambda_1_lambda_2, lambda_tilde_delta_lambda_tilde_to_lambda_1_lambda_2
)
from bilby.core.result import Result

from param_estimation_test.npe_wf_analysis import VAE, PhaseModificationAnalysis

import lalsimulation
from astropy import units as u, constants as c

MSUN_KM = lal.MSUN_SI * lal.G_SI / lal.C_SI ** 2 / 1e3
MSUN_S  = MSUN_KM / lal.C_SI * 1e3

from workflow import initialize

project_base_path = os.path.abspath('.')
if os.path.abspath('..') not in sys.path:
    sys.path.append(os.path.abspath('..'))

# Import PyTorch network
def import_network(network_path, base_path=project_base_path, network_type='BBH', data_dim=None, cond_dim=None):

    network_fullpath = os.path.expanduser(os.path.join(base_path, network_path))

    device = torch.device('cpu')
    optimizer_kwargs = dict(lr=1e-4, weight_decay=1e-4)
    scheduler_kwargs = dict(gamma=0.9)

    if network_type in ['BNS']:
        model_kwargs = dict(depth=4, width=512, 
                            data_dim=640 if data_dim is None else data_dim,
                            cond_dim=6 if cond_dim is None else cond_dim,
                            grid_dim=2, freeze_shape=True)
    else:
        model_kwargs = dict(depth=4, width=512, 
                            data_dim=640 if data_dim is None else data_dim,
                            cond_dim=4 if cond_dim is None else cond_dim,
                            grid_dim=2, freeze_shape=True)
    
    model_type = VAE
    optimizer_type = torch.optim.AdamW
    scheduler_type = torch.optim.lr_scheduler.ExponentialLR

    model, optimizer, scheduler = initialize(
                                    model_type, optimizer_type, scheduler_type,
                                    model_kwargs=model_kwargs,
                                    optimizer_kwargs=optimizer_kwargs,
                                    scheduler_kwargs=scheduler_kwargs, device=device)
    
    state_dict = torch.load(network_fullpath, map_location=device, weights_only=True)

    #print('Model dimensions: \n  |  data_dim: {}\n  |  grid_dim: {}'.format(model.data_dim, model.grid_dim))
    #print(state_dict['model'].keys())

    model = model_type(**model_kwargs).to(device)
    optimizer = optimizer_type(model.parameters(), **optimizer_kwargs)
    scheduler = scheduler_type(optimizer, **scheduler_kwargs)

    model.load_state_dict(state_dict['model'])
    if optimizer is not None: optimizer.load_state_dict(state_dict['optimizer'])
    if scheduler is not None: scheduler.load_state_dict(state_dict['scheduler'])

    model.eval()
    model.train(False)

    return model, device, optimizer, scheduler



# ------------------------------------------------------------------
#
#                    ppE FUNCTIONS AND UTILITIES
#
# ------------------------------------------------------------------



def get_phi_ppe(frequency_array, mass_1, mass_2, chi_1, chi_2, b, beta):
    mtot = mass_1 + mass_2
    mc = component_masses_to_chirp_mass(mass_1, mass_2)
    freqs = np.asarray(frequency_array, dtype=np.float64)
    #freqs = np.asarray(frequency_array, dtype=np.float32)
    fcut = 1.8e-2 / mtot / MSUN_S
    freqs_geom = np.pi * freqs * mc * MSUN_S
    fcut_geom = np.pi * fcut * mc * MSUN_S
    mask_low = freqs <= 0.
    mask_high = freqs > fcut
    mask_mid = ~(mask_low|mask_high)
    phi = np.zeros_like(freqs)
    phi[mask_mid] = beta * freqs_geom[mask_mid] ** (b/3.)
    phicut = beta * fcut_geom ** (b/3.)
    dphicut = phicut / fcut_geom * (b/3.)
    phi[mask_high] = phicut + dphicut * (freqs_geom[mask_high] - fcut_geom)
    return phi



def get_ppe_bound(b_ppe, m1, m2, chi1z, chi2z, l1=0., l2=0., cq1=0., cq2=0., ppe_ref=10):
    ppe_ref = ppe_ref * u.Hz * (m1 + m2) * u.solMass
    ppe_ref = ppe_ref * c.G / c.c**3
    ppe_ref = ppe_ref.to('').value
    ppe_ref_v = (np.pi * ppe_ref) ** (1/3)
    b_ref = max(-5, int(np.floor(b_ppe)))
    # if b_ref <= -4:
    #     # GR coeff at 0.5PN is 0, merge with the b <= -5 branch
    #     b_ref = -5
    eta = bilby.gw.conversion.component_masses_to_symmetric_mass_ratio(m1, m2)
    num_samples = 1
    param_vecs = [lal.CreateREAL8Vector(num_samples) for _ in range(8)]
    param_vecs[0].data = np.atleast_1d(m1)
    param_vecs[1].data = np.atleast_1d(m2)
    param_vecs[2].data = np.atleast_1d(chi1z)
    param_vecs[3].data = np.atleast_1d(chi2z)
    param_vecs[4].data = np.atleast_1d(l1)
    param_vecs[5].data = np.atleast_1d(l2)
    param_vecs[6].data = np.atleast_1d(cq1)
    param_vecs[7].data = np.atleast_1d(cq2)

    pn_coeff = lalsimulation.SimInspiralTaylorF2AlignedPhasingArray(*param_vecs).data
    pn_coeff = pn_coeff[:len(pn_coeff)//3] # remove coeffs for vlogv and vlogvsq
    if b_ref not in [-4, 3, 4] and b_ref < 6:
        pn_coeff = pn_coeff.reshape(-1, num_samples)[b_ref+5] # take only the pn coeff of the ppE order
        beta_ppe_bound = np.abs(pn_coeff) * ppe_ref_v**(b_ref-b_ppe)
    else:
        if b_ref == -4:
            pn_coeff1 = pn_coeff.reshape(-1, num_samples)[0] # 0PN coeff
            pn_coeff2 = pn_coeff.reshape(-1, num_samples)[2] # 1PN coeff
            beta_ppe_bound = np.sqrt(np.abs(pn_coeff1) * np.abs(pn_coeff2))
        elif b_ref == 3:
            pn_coeff1 = pn_coeff.reshape(-1, num_samples)[7] # 3.5PN coeff
            pn_coeff2 = pn_coeff.reshape(-1, num_samples)[10] # 5PN coeff
            beta_ppe_bound = np.cbrt(np.abs(pn_coeff1) * np.abs(pn_coeff2))
        elif b_ref == 4:
            pn_coeff1 = pn_coeff.reshape(-1, num_samples)[7] # 3.5PN coeff
            pn_coeff2 = pn_coeff.reshape(-1, num_samples)[10] # 5PN coeff
            beta_ppe_bound = np.cbrt(np.abs(pn_coeff1) * np.abs(pn_coeff2))**2.

    beta_ppe_bound /= eta**(b_ppe/5.)
    return beta_ppe_bound[0]

def get_ppe_bound_fn_safe(m1, m2, chi1z, chi2z, l1=0., l2=0., cq1=0., cq2=0., ppe_ref=10, b_min=-13, b_max=-1):
    b_ppe_interp_valid = np.array([b_i for b_i in np.linspace(b_min, b_max, int(b_max - b_min + 1))])
    beta_max_interp_valid = np.array([get_ppe_bound(b_ppe_val, m1, m2, chi1z, chi2z, l1, l2, cq1, cq2, ppe_ref) for b_ppe_val in b_ppe_interp_valid])
    
    beta_interp_fn_log_q = interp1d(b_ppe_interp_valid, np.log10(beta_max_interp_valid), kind='quadratic')
    beta_interp_fn_q = lambda b: 10.**beta_interp_fn_log_q(b)

    return beta_interp_fn_q

def get_ppe_bound_safe(b_ppe, func_in, m1=None, m2=None, chi1z=None, chi2z=None, l1=0., l2=0., cq1=0., cq2=0., ppe_ref=10):
    if func_in is None:
        b_ppe_max = min(max(-1, int(np.floor(b_ppe))), 5)
        beta_interp_fn = get_ppe_bound_fn_safe(m1, m2, chi1z, chi2z, l1, l2, cq1, cq2, ppe_ref, b_max=b_ppe_max)
    else:
        beta_interp_fn = func_in
    
    return beta_interp_fn(b_ppe)




# ------------------------------------------------------------------
#
#                    npE FUNCTIONS AND UTILITIES
#
# ------------------------------------------------------------------




# Given ppE theory params and intrinsic binary params, calculate VAE latent space representation
def get_vae_latent_ppe(vae_analyzer: PhaseModificationAnalysis, b, beta, m1, m2, chi1, chi2, l1=0., l2=0., cq1=0., cq2=0.):
    phi_func = lambda freqs: get_phi_ppe(freqs, m1, m2, chi1, chi2, b, beta)
    z1, z2 = vae_analyzer.extract_latent(phi_func, m1, m2, chi1, chi2, l1, l2)
    return z1, z2




# Interpolate theory angle mappings for injected binary parameters, at a given beta_ppE (for plotting purposes)
# b_ppe <--> z_theta
def map_latent_space_angles(vae_analyzer, n_points=100, b_min=-13, b_max=-1, beta_ppe_rel=1.,
                            m1=20., m2=15., chi1=0., chi2=0., l1=0., l2=0., cq1=0., cq2=0.,
                            ppe_ref=10):
    b_ppe_vals = np.linspace(b_min, b_max, n_points)

    bound_fn = get_ppe_bound_fn_safe(m1, m2, chi1, chi2, l1, l2, cq1, cq2, ppe_ref)
    get_ppe_bound = lambda b_ppe_in: get_ppe_bound_safe(b_ppe_in, bound_fn)

    beta_ppe_units = np.array([get_ppe_bound(b_ppe_val) for b_ppe_val in b_ppe_vals])
    beta_ppe_vals = beta_ppe_rel*beta_ppe_units

    latent_space_vals = np.array([get_vae_latent_ppe(vae_analyzer, b_ppe, beta_ppe, m1, m2, chi1, chi2, l1, l2, cq1, cq2) for b_ppe, beta_ppe in zip(b_ppe_vals, beta_ppe_vals)])
    
    z1_vals = latent_space_vals[:,0]
    z2_vals = latent_space_vals[:,1]
    z_abs_vals = np.sqrt(np.abs(z1_vals**2 + z2_vals**2))
    z_theta_vals = np.mod(np.arctan2(z2_vals, z1_vals), 2*np.pi)

    # f[b_ppe] -> z_theta
    theta_func = interp1d(b_ppe_vals, z_theta_vals, fill_value='extrapolate')
    # f[z_theta] -> b_ppe
    b_ppe_func = interp1d(z_theta_vals, b_ppe_vals, fill_value='extrapolate')
    return theta_func, b_ppe_func
    #vae_analyzer.extract_latent()

# Interpolate theory scale mappings for injected binary parameters, at a given b_ppE (for plotting purposes)
# beta_ppe <--> z_abs
def map_latent_space_scales(vae_analyzer, n_points=100, beta_min=-1., beta_max=1., b_ppe=-5.,
                            m1=20., m2=15., chi1=0., chi2=0., l1=0., l2=0., cq1=0., cq2=0.,
                            ppe_ref=10):
    
    bound_fn = get_ppe_bound_fn_safe(m1, m2, chi1, chi2, l1, l2, cq1, cq2, ppe_ref)
    get_ppe_bound = lambda b_ppe_in: get_ppe_bound_safe(b_ppe_in, bound_fn)

    beta_ppe_unit = get_ppe_bound(b_ppe)
    beta_ppe_vals = np.linspace(beta_min, beta_max, n_points) * beta_ppe_unit
    b_ppe_vals = np.array([b_ppe for beta_ppe in beta_ppe_vals])

    latent_space_vals = np.array([get_vae_latent_ppe(vae_analyzer, b_ppe, beta_ppe, m1, m2, chi1, chi2, l1, l2, cq1, cq2) for b_ppe, beta_ppe in zip(b_ppe_vals, beta_ppe_vals)])
    
    z1_vals = latent_space_vals[:,0]
    z2_vals = latent_space_vals[:,1]
    z_abs_vals = np.sqrt(np.abs(z1_vals**2 + z2_vals**2))
    z_theta_vals = np.mod(np.arctan2(z2_vals, z1_vals), 2*np.pi)

    # beta_ppe -> z_abs
    zabs_func = interp1d(beta_ppe_vals, z_abs_vals, fill_value='extrapolate')
    # z_abs -> beta_ppe
    beta_func = interp1d(z_abs_vals, beta_ppe_vals, fill_value='extrapolate')
    return zabs_func, beta_func
    #vae_analyzer.extract_latent()


# ------------------------------------------------------------------
#
#                           PLOTTING FUNCTIONS
#
# ------------------------------------------------------------------


def plot_dephasing(network_path, base_path=project_base_path, root_path='.', network_type='BBH', sample_zmags=False, show_plot=True, title='',
                   z_ang_min=0., z_ang_max=2*np.pi, verbosity=0, f_num=None, cond_dim=None,
                   m1_ref=None, m2_ref=None, chi1_ref=None, chi2_ref=None, l1_ref=None, l2_ref=None):
    if network_type not in ['BBH', 'BNS']:
        print('Error: Unsupported network_type. Defaulting to network_type=\'BBH\'')
        network_type = 'BBH'

    # Get dephasing values from shape function
    if network_type in ['BNS']:
        f_min = 0.00004
        f_max = 0.018
        f_num = 640 if f_num is None else f_num
        cond_dim = 6 if cond_dim is None else cond_dim

        m_min = 0.5
        m_max = 3.0

        if m1_ref is None:
            m1_ref = 1.4
        if m2_ref is None:
            m2_ref = 1.4
        if chi1_ref is None:
            chi1_ref = 0.001
        if chi2_ref is None:
            chi2_ref = 0.001
        if l1_ref is None:
            l1_ref = 300.
        if l2_ref is None:
            l2_ref = 300.
    elif network_type in ['NSBH']:
        f_min = 0.00004
        f_max = 0.018
        f_num = 640 if f_num is None else f_num
        cond_dim = 6 if cond_dim is None else cond_dim

        m_min = 0.6
        m_max = 6.0

        if m1_ref is None:
            m1_ref = 3.
        if m2_ref is None:
            m2_ref = 3.
        if chi1_ref is None:
            chi1_ref = 0.001
        if chi2_ref is None:
            chi2_ref = 0.001
        if l1_ref is None:
            l1_ref = 0.
        if l2_ref is None:
            l2_ref = 300.
    elif network_type in ['BBH']:
        f_min = 0.0004
        f_max = 0.018
        f_num = 640 if f_num is None else f_num
        cond_dim = 4 if cond_dim is None else cond_dim

        m_min = 5.0
        m_max = 30.0
        if m1_ref is None:
            m1_ref = 15.
        if m2_ref is None:
            m2_ref = 15.
        if chi1_ref is None:
            chi1_ref = 0.1
        if chi2_ref is None:
            chi2_ref = 0.1
        l1_ref = 0.
        l2_ref = 0.
    else:
        print(f'Error: Invalid network type: {network_type}')
        return False

    if verbosity >= 0:
        mtot = m1_ref + m2_ref
        if network_type in ['BNS', 'NSBH']:
            tidal_ref_str = f', lambda1={l1_ref:.1f}, lambda2={l2_ref:.1f}'
        else:
            tidal_ref_str = ''
        print(f'Using reference parameters: m1={m1_ref}, m2={m2_ref}, chi1={chi1_ref}, chi2={chi2_ref}{tidal_ref_str}')
        print(f'Chirp mass: {component_masses_to_chirp_mass(m1_ref, m2_ref):.2f}')
        #print(f'Frequency range (geometric): {f_min*(2*m_min*MSUN_S)} - {f_max*(2*m_max*MSUN_S)} km^-1')
        print(f'Frequency range (unitless): {f_min} - {f_max}')
        print(f'Frequency range (Hz): {f_min / mtot / MSUN_S:.2f} - {f_max / mtot / MSUN_S:.2f} Hz')
        print(f'Frequency num: {f_num}')
        #print(f'Latent space angle range: {z_ang_min} - {z_ang_max} rad')
        #print(f'Latent space angle samples: {360}')
        print(f'CVAE label dimension: {cond_dim}')
        
    # Import specified network
    #model, device = import_network(network_path, base_path)
    model, device, _, _ = import_network(network_path, base_path=os.path.join(base_path, root_path), 
                                         network_type=network_type, data_dim=f_num, cond_dim=cond_dim)

    #z_mag = np.linspace(0, 1, 100)
    #z_mag = 0.5
    z_ang_arr = np.linspace(z_ang_min, z_ang_max, 360)

    if sample_zmags:
        z1_arr = np.array([np.random.uniform(0,1) * np.cos(z_ang) for z_ang in z_ang_arr])
        z2_arr = np.array([np.random.uniform(0,1) * np.sin(z_ang) for z_ang in z_ang_arr])
    else:
        z1_arr = np.array([0.5 * np.cos(z_ang) for z_ang in z_ang_arr])
        z2_arr = np.array([0.5 * np.sin(z_ang) for z_ang in z_ang_arr])

    norm_fac = 1.

    if verbosity >= 5:
        print('z1 shape: ', z1_arr.shape)
        print('z2 shape: ', z2_arr.shape)

    def gw_params_to_vae_labels(mass_1, mass_2, chi_1, chi_2, lambda_1=0., lambda_2=0.):
        mc = bilby.gw.conversion.component_masses_to_chirp_mass(mass_1, mass_2)
        q = mass_2 / mass_1
        chi_sym = 0.5 * (chi_1 + chi_2)
        chi_asym = 0.5 * (chi_1 - chi_2)
        lambda_sym = 0.5 * (lambda_1 + lambda_2)
        lambda_asym = 0.5 * (lambda_1 - lambda_2)
        labels = [np.log(mc), q, chi_sym, chi_asym, lambda_sym, lambda_asym]
        return labels

    # Loop the below over all sampled z_theta values
    for z1, z2 in zip(z1_arr, z2_arr):
        z_mag = np.sqrt(z1**2 + z2**2)
        ## TODO: Finish adapting all of this
        f_min_geom = f_min*(2*m_min*MSUN_S)
        f_max_geom = f_max*(2*m_max*MSUN_S)
        # TODO: Should we provide unitless or geometric frequencies?
        freqs = np.linspace(f_min, f_max, f_num)
        geom_freqs = np.linspace(f_min_geom, f_max_geom, f_num)
        model_loggeom_freqs = np.linspace(np.log10(f_min_geom), np.log10(f_max_geom), f_num)
        geom_freqs_cutoff = np.exp(model_loggeom_freqs[-1])
        mask_low = geom_freqs <= 0
        mask_high = geom_freqs > geom_freqs_cutoff
        mask_mid = ~(mask_low|mask_high)
        #loggeom_freqs = np.log10(np.append(geom_freqs[mask_mid], geom_freqs_cutoff))
        loggeom_freqs = np.log10(np.append(geom_freqs[mask_mid], geom_freqs_cutoff))
        loggeom_freq_range = model_loggeom_freqs[-1] - model_loggeom_freqs[0]
        xin = (loggeom_freqs - model_loggeom_freqs[0]) / loggeom_freq_range
        #xin = torch.tensor(xin, dtype=torch.float32, device=device).view(1, -1)
        xin = torch.tensor(xin, device=device).view(1, -1)

        z = torch.tensor([z1/z_mag, z2/z_mag], device=device).view(1, -1)
        #z = torch.tensor([z1/z_mag, z2/z_mag], dtype=torch.float32, device=device).view(1, -1)

        # returns: [log(Mc), eta, chi_sym, chi_asym, lambda_sym, lambda_asym]
        cond = gw_params_to_vae_labels(m1_ref, m2_ref, chi1_ref, chi2_ref, l1_ref, l2_ref)
        #cond = torch.tensor(cond, dtype=torch.float32, device=device).view(1, -1)
        cond = torch.tensor(cond, device=device).view(1, -1)
        #print('z: ', z.shape, z.dtype)
        #print('cond: ', cond.shape, cond.dtype)

        Sout, dSout = model.extended_decoder(z, cond, xin)

        #print('Sout: ', Sout.shape)

        xout, dext = Sout.view(-1).cpu().detach().numpy().flatten(), dSout.item()

        #print('xout: ', xout.shape)
        #print('dext: ', dext)
        #print('Sout: ', Sout.shape)

        Smag = np.sqrt(np.sum([xout_val**2 for xout_val in xout]))
        #print('Smag: ', Smag)

        phases_mod = np.zeros_like(geom_freqs)
        phases_mod[mask_mid] = xout[:-1]
        phases_mod[mask_high] = xout[-1] + dext / loggeom_freq_range / np.log(10) * (geom_freqs[mask_high] / geom_freqs_cutoff - 1.)
        phases_mod *= z_mag * norm_fac / Smag

        #print('phases_mod: ', phases_mod.shape)

        if show_plot:
            # Plot dephasing
            x_vals = freqs
            y_vals = phases_mod

            #print(Sout)
            #print(dext)
            #print(xout)
            #print(phases_mod)

            plt.plot(x_vals, y_vals, color='b', alpha=0.7, lw=0.15)

    if show_plot:
        plt.xlabel(r'$\bar{f}$')
        plt.ylabel(r'$\hat{S}(f)$')
        plt.xscale('log')
        #plt.yscale('log')
        plt.grid()
        #plt.legend()
        #plt.xlim(f_min_geom, 1e-7)

        plt.title(title)

        plt.tight_layout
        plt.show()
        
    if show_plot and False:
        # Plot in latent-space plane
        x_vals_2, y_vals_2 = np.zip([z1_arr, z2_arr])
        plt.plot(x_vals_2, y_vals_2)
        plt.xlabel(r'$z_1$')
        plt.ylabel(r'$z_2$')
        plt.tight_layout()
        plt.show()



# Get and plot maximal value for dS/dtheta
# TODO: Finish this functionality, and adapt to BNS case
def get_dSdtheta(network_path, base_path=project_base_path, network_type='BBH', f_num=None, sample_zmags=False, show_plot=True, title=''):
    if network_type not in ['BBH', 'BNS']:
        print('Error: Unsupported network_type. Defaulting to network_type=\'BBH\'')
        network_type = 'BBH'

    if network_type in ['BNS']:
        f_min = 0.00004
        f_max = 0.018
        f_num = 640 if f_num is None else f_num

        m_min = 0.5
        m_max = 3.0
    elif network_type in ['BBH']:
        f_min = 0.0004
        f_max = 0.018
        f_num = 640 if f_num is None else f_num

        m_min = 5.0
        m_max = 30.0
    else:
        print('Error')
        return False
    
    # Import specified network
    #model, device = import_network(network_path, base_path)
    model, device, _, _ = import_network(network_path, base_path, network_type, 
                                         data_dim=f_num)

    #z_mag = np.linspace(0, 1, 100)
    #z_mag = 0.5
    z_theta_arr = np.linspace(0, np.pi, 360)

    if sample_zmags:
        z1_arr = np.array([np.random.uniform(0,1) * np.cos(z_theta) for z_theta in z_theta_arr])
        z2_arr = np.array([np.random.uniform(0,1) * np.sin(z_theta) for z_theta in z_theta_arr])
    else:
        z1_arr = np.array([0.5 * np.cos(z_theta) for z_theta in z_theta_arr])
        z2_arr = np.array([0.5 * np.sin(z_theta) for z_theta in z_theta_arr])

    # Loop the below over all sampled z_theta values
    for z1, z2 in zip(z1_arr, z2_arr):
        z_mag = np.sqrt(z1**2 + z2**2)
        ## TODO: Finish adapting all of this
        f_min_geom = f_min*(2*m_min*MSUN_S)
        f_max_geom = f_max*(2*m_max*MSUN_S)
        # TODO: Should we provide unitless or geometric frequencies?
        freqs = np.linspace(f_min, f_max, f_num)
        geom_freqs = np.linspace(f_min_geom, f_max_geom, f_num)
        model_loggeom_freqs = np.linspace(np.log10(f_min_geom), np.log10(f_max_geom), f_num)
        geom_freqs_cutoff = np.exp(model_loggeom_freqs[-1])
        mask_low = geom_freqs <= 0
        mask_high = geom_freqs > geom_freqs_cutoff
        mask_mid = ~(mask_low|mask_high)
        #loggeom_freqs = np.log10(np.append(geom_freqs[mask_mid], geom_freqs_cutoff))
        loggeom_freqs = np.log10(np.append(geom_freqs[mask_mid], geom_freqs_cutoff))
        loggeom_freq_range = model_loggeom_freqs[-1] - model_loggeom_freqs[0]
        xin = (loggeom_freqs - model_loggeom_freqs[0]) / loggeom_freq_range
        xin = torch.tensor(xin, device=device).view(1, -1)

        z = torch.tensor([z1/z_mag, z2/z_mag], device=device).view(1, -1)
        #cond = self.gw_params_to_vae_labels(mass_1, mass_2, chi_1, chi_2) # returns: [Mc, eta, chi_sym, chi_asym]
        cond = [25, 1, 0, 0]
        cond = torch.tensor(cond, device=device).view(1, -1)
        #print('z: ', z.shape)
        #print('cond: ', cond.shape)
        norm_fac = 1.

        Sout, dSout = model.extended_decoder(z, cond, xin)
        # TODO: Finish adapting all of this
        xout, dext = Sout.view(-1).cpu().detach().numpy().flatten(), dSout.item()
        Smag = np.sqrt(np.sum([xout_val**2 for xout_val in xout]))

        phases_mod = np.zeros_like(geom_freqs)
        phases_mod[mask_mid] = xout[:-1]
        phases_mod[mask_high] = xout[-1] + dext / loggeom_freq_range / np.log(10) * (geom_freqs[mask_high] / geom_freqs_cutoff - 1.)
        phases_mod *= z_mag * norm_fac / Smag

    if show_plot:
        y_vals = np.gradient(phases_mod)
        plt.plot(z_theta_arr, dext)

# Dataset loading imports
from data import DatasetManager, PhasingDataset
from train_network import DataLoader, vae_loss_fn, vae_diagnosis_fn # type: ignore
# Latent space plotting imports
from train import VAEEvaluationAgent, plot_latent_distrib
from workflow import get_current_epoch_value


# Validation dataset
def get_val_dataloader(base_path=project_base_path, dataset_name=None, network_type='BBH', seed=1234, use_tidal_data=False):
    return get_dataloader(base_path, dataset_name, network_type, subset='val', dataset_seed=seed, use_tidal_data=use_tidal_data)

def get_val_dataset(base_path=project_base_path, dataset_name=None, network_type='BBH', seed=1234, use_tidal_data=False):
    return get_val_dataloader(base_path, dataset_name, network_type, seed, use_tidal_data=use_tidal_data).dataset

# Train dataset
def get_train_dataloader(base_path=project_base_path, dataset_name=None, network_type='BBH', seed=1234, use_tidal_data=False):
    return get_dataloader(base_path, dataset_name, network_type, subset='train', dataset_seed=seed, use_tidal_data=use_tidal_data)

def get_train_dataset(base_path=project_base_path, dataset_name=None, network_type='BBH', seed=1234, use_tidal_data=False):
    return get_train_dataloader(base_path, dataset_name, network_type, seed, use_tidal_data=use_tidal_data).dataset

# Test dataset
def get_test_dataloader(base_path=project_base_path, dataset_name=None, network_type='BBH', seed=1234, use_tidal_data=False):
    return get_dataloader(base_path, dataset_name, network_type, subset='test', dataset_seed=seed, use_tidal_data=use_tidal_data)

def get_test_dataset(base_path=project_base_path, dataset_name=None, network_type='BBH', seed=1234, use_tidal_data=False):
    return get_test_dataloader(base_path, dataset_name, network_type, seed, use_tidal_data=use_tidal_data).dataset

# Function to import a dataset from a file
# TODO: More robust support for different network or dataset structure (partial/full tidal data, etc.)
def get_dataloader(base_path=project_base_path, dataset_name=None, network_type='BBH', dataset_seed=1234, subset=None, use_tidal_data=False):
    if network_type == 'BNS':
        network_type_short = 'NS'
        dataset_filenames = [
            "ppe-minus13.pkl",
            "ppe-minus12.pkl",
            "ppe-minus11.pkl",
            "ppe-minus10.pkl",
            "ppe-minus9.pkl",
            "ppe-minus8.pkl",
            "ppe-minus7.pkl",
            "ppe-minus6.pkl",
            "ppe-minus5.pkl",
            "ppe-minus4.pkl",
            "ppe-minus3.pkl",
            "ppe-minus2.pkl",
            "ppe-minus1.pkl",
        ]
        if use_tidal_data:
            dataset_filenames += [
                "ppe-minus0.pkl",
                "ppe-plus1.pkl",
                "ppe-plus2.pkl",
                "ppe-plus3.pkl",
                "ppe-plus4.pkl",
                "ppe-plus5.pkl",
            ]
    else:
        network_type_short = 'BH'
        dataset_filenames = [
            "ppe-minus13.pkl",
            "ppe-minus12.pkl",
            "ppe-minus11.pkl",
            "ppe-minus10.pkl",
            "ppe-minus9.pkl",
            "ppe-minus8.pkl",
            "ppe-minus7.pkl",
            "ppe-minus6.pkl",
            "ppe-minus5.pkl",
            "ppe-minus4.pkl",
            "ppe-minus3.pkl",
            "ppe-minus2.pkl",
            "ppe-minus1.pkl",
        ]

    if dataset_name is None:
        dataset_rootdir = os.path.expanduser(os.path.join(base_path, 'dataset_'+network_type_short))
    else:
        dataset_rootdir = os.path.expanduser(os.path.join(base_path, dataset_name))
    #print('Loading dataset from: ', dataset_rootdir)

    dataset_type = PhasingDataset
    dataset_n_ppe = 1
    dataset_norm_fac = {}
    dataset_sample_size = 0.25
    dataset_subset_split = [0.8, 0.1, 0.1]

    # TODO: Make this more robust (use_tidal vs. use_tidal_full, etc.)
    dataset_and_split = DatasetManager(
                dataset_filenames, root_dir=dataset_rootdir, 
                sample_size=dataset_sample_size, subset_split=dataset_subset_split, random_state=dataset_seed, 
                dataset_type=dataset_type, dataset_kwargs=dict(n_ppe=dataset_n_ppe, norm_fac=dataset_norm_fac, 
                                                               use_tidal=use_tidal_data))
        
    batch_size_train = 64
    batch_size_val = 1024
    batch_size_test = 64    # Assumption/guess
    batches_per_summary = 0.1

    dataset_train, dataset_val, dataset_test = dataset_and_split.subsets
    
    if subset is None:
        return (dataset_train, dataset_val, dataset_test)
    if subset in ['train', 'training']:
        data_loader_train = DataLoader(dataset_train, batch_size=batch_size_train, shuffle=True)
        return data_loader_train
    if subset in ['val', 'validation']:
        data_loader_val = DataLoader(dataset_val, batch_size=batch_size_val, shuffle=False)
        return data_loader_val
    if subset in ['test', 'testing', 'tune', 'tuning']:
        data_loader_test = DataLoader(dataset_test, batch_size=batch_size_test, shuffle=False)
        return data_loader_test

def plot_latent_space_distribution(network_path, base_path=project_base_path, root_path='.', network_type='BBH', dataset_seed=1234, dataset_name=None,
                                   use_final_kwargs=True, z_theta_inj=None, z_theta_rec=None, z_abs_inj=None, z_abs_rec=None,
                                   data_dim=640, cond_dim=4, num_epochs=50, rescaled=False, use_tidal_data=False):

    # Import network from file
    model, device, optimizer, scheduler = import_network(network_path, os.path.abspath(os.path.join(base_path, root_path)), 
                                                         network_type, data_dim, cond_dim)

    # State loss function and diagnosis function of interest
    loss_fn = vae_loss_fn
    diagnosis_fn = vae_diagnosis_fn

    # Use kwargs from shape function training
    if not use_final_kwargs:
        loss_kwargs = dict(
            kl_coeff=1e-6, 
            shape_coeff=1., scale_coeff=0., 
            recon_coeff=0., recon_scale_coeff=0.,
            with_mu=False, recon_use_mse=False, 
        )
        diagnosis_kwargs = dict(with_mu=False)
    else:
        # Use kwargs from scale function training
        loss_kwargs = dict(
            kl_coeff=0., 
            shape_coeff=0., scale_coeff=0., 
            recon_coeff=0., recon_scale_coeff=1.,
            with_mu=False, recon_use_mse=False, 
        )
        diagnosis_kwargs = dict(with_mu=False)

    # Define total epochs and current epoch (final)
    epochs_per_latent_plot = [(10, 1), (None, 10)]
    epochs_per_checkpoint = num_epochs
    if rescaled:
        final_epoch = 3 * num_epochs
    else:
        final_epoch = 2 * num_epochs
        
    current_epochs_per_latent_plot = get_current_epoch_value(epochs_per_latent_plot, final_epoch)
    current_epochs_per_checkpoint = get_current_epoch_value(epochs_per_checkpoint, final_epoch)

    current_loss_kwargs = dict()
    for k,v in loss_kwargs.items():
        current_loss_kwargs[k] = get_current_epoch_value(v, final_epoch)

    print('Importing Datasets...')
    # Get training and validation sets used for network training
    data_loader_train = get_train_dataloader(base_path, dataset_name, network_type, dataset_seed, use_tidal_data)
    data_loader_val = get_val_dataloader(base_path, dataset_name, network_type, dataset_seed, use_tidal_data)

    # Print some dataset info
    if True:
        #print(data_loader_val.dataset)
        print(f'Dataset type: {type(data_loader_val.dataset)}')
        print(f'Number of samples: {len(data_loader_val.dataset)}')
        print(f'Example data shape: {data_loader_val.dataset[0][0].shape}')
        print(f'Example labels:')
        print(f'    m1: {data_loader_val.dataset[0][1][0]}')
        print(f'    m2: {data_loader_val.dataset[0][1][1]}')
        print(f'    s1: {data_loader_val.dataset[0][1][2]}')
        print(f'    s2: {data_loader_val.dataset[0][1][3]}')
        if cond_dim >= 6:
            print(f'    lambda1: {data_loader_val.dataset[0][1][4]}')
            print(f'    lambda2: {data_loader_val.dataset[0][1][5]}')

    #training_seed = None
    #torch.manual_seed(training_seed)

    # Latent space plot parameters
    npoints_for_latent_plot = int(1e2)
    npoints_for_generation = 16
 
    print('Constructing Agent...')
    # Establish VAE Evaluation Agent from loaded model and dataset
    agent = VAEEvaluationAgent(
                            model, loss_fn, diagnosis_fn, current_loss_kwargs, diagnosis_kwargs,
                            distrib_thin_fac=npoints_for_latent_plot/len(data_loader_val.dataset),
                            sample_size=npoints_for_generation)

    # Get hyperparameters at this epoch
    hparams = dict(
            lr=scheduler.get_last_lr()[0], 
            weight_decay=optimizer.defaults['weight_decay'],
            batch_size=data_loader_train.batch_size,
        )
    # hparams.update(agent.loss_fn_kwargs)
    for k, v in agent.loss_fn_kwargs.items():
        try: float(v)
        except TypeError: pass
        else: hparams[k] = v
    #log_str = "{} Epoch {}".format(datetime.now().strftime('%H:%M:%S'), final_epoch)
    #log_str += "\nHparams = {}".format(hparams)
    #print(log_str); sys.stdout.flush()

    print('Training agent...')
    # Agent train step
    if True:
        agent.train()
        for i_batch, data_batch in enumerate(data_loader_train):
            optimizer.zero_grad()
            loss = agent.add_batch(data_batch)
            if torch.isfinite(loss):
                loss.backward()
                optimizer.step()
            #if batches_per_summary > 0 and (i_batch+1) % batches_per_summary == 0:
                #loss_recent = agent.pop_recent_loss()
                #log_str = "  {:0.0f}%, interval loss = {:0.3e}".format(
                #                    100. * (i_batch+1) / len(data_loader_train), loss_recent)
                #print(log_str); sys.stdout.flush()
        scheduler.step()

        #loss_train = agent.report_loss()

    print('Evaluating Agent...')
    # Agent evaluation step
    agent.eval()
    for i_batch, data_batch in enumerate(data_loader_val):
        agent.add_batch(data_batch)
    #loss_val = agent.report_loss()
    #metrics = agent.report_metrics()

    print('Plotting Latent Space distribution...')
    # Plot latent space distribution

    if False:
        #print('Agent plot')
        fig_latent = agent.report_latent_distrib()
    else:
        #print('Manual plot')
        distrib = {k: np.concatenate(agent._acc_distrib[k], axis=0) for k in agent._distrib_keys}
        fig_latent = plot_latent_distrib(distrib, plot_masked=False)

    if False:
        # WIP: Plot sample values as projected onto latent space
        from waveform_analysis import PhaseModificationAnalysis
        model_kwargs = dict(depth=4, width=512, data_dim=640, cond_dim=4, grid_dim=2, freeze_shape=True)
        network_fullpath = os.path.expanduser(os.path.join(base_path, network_path))
        val_dataset_analysis = PhaseModificationAnalysis(network_fullpath, model_kwargs=model_kwargs)

        def extract_latent_from_data(dataset_item):
            phases = dataset_item[0]
            labels = dataset_item[1]
            b_ppe, gamma_bar = dataset_item[2]
            #mtot = mass_1 + mass_2
            #freqs = 10**val_dataset_analysis.model_loggeom_freqs / mtot / MSUN_S
            #phases = phase_func(freqs) / self.norm_fac
            norm = np.sqrt(np.mean(phases*phases, axis=-1))
            if norm == 0.:
                return 0., 0.
            phases = torch.tensor(phases, dtype=torch.double, device=val_dataset_analysis.device).view(1, 1, -1)
            labels = val_dataset_analysis.gw_params_to_vae_labels(labels[0], labels[1], labels[2], labels[3])
            labels = torch.tensor(labels, dtype=torch.double, device=val_dataset_analysis.device).view(1, -1)
            z, _ = val_dataset_analysis.model.encoder(phases, labels)
            norm_std = val_dataset_analysis.model.decoder(z, labels)
            norm_std = torch.sqrt(torch.mean(norm_std*norm_std, dim=-1)).squeeze().item()
            z_1, z_2 = z.cpu().detach().numpy().flatten() * norm / norm_std
            return z_1, z_2, b_ppe

        plot_values = np.array([extract_latent_from_data(dataset_item) for dataset_item in data_loader_val.dataset])
        ax_latent = fig_latent.get_axes()
        ax_latent[0].scatter(x=plot_values[:,0], y=plot_values[:,1], figure=fig_latent)
        #  labels=plot_values[:,2]

    if z_abs_inj is not None and z_theta_inj is not None:
        z_1_inj = z_abs_inj * np.cos(z_theta_inj)
        z_2_inj = z_abs_inj * np.sin(z_theta_inj)
        ax_latent = fig_latent.get_axes()
        ax_latent[0].scatter(x=[z_1_inj], y=[z_2_inj], figure=fig_latent, marker='h', label='Injected')

    if z_abs_rec is not None and z_theta_rec is not None:
        z_1_rec = z_abs_rec * np.cos(z_theta_rec)
        z_2_rec = z_abs_rec * np.sin(z_theta_rec)
        ax_latent = fig_latent.get_axes()
        ax_latent[0].scatter(x=[z_1_rec], y=[z_2_rec], figure=fig_latent, marker='x', label='Recovered')

    #plt.legend()
    #plt.tight_layout()
    #plt.show()


def compare_reconstructed_phasing(network_path, base_path=project_base_path, root_path='.', b_inj=-5, beta_rel_inj=0.5, f_max=None,
                                  plot_difference=False, network_type='BBH', injection_parameters=None,
                                  data_dim=None, cond_dim=None, plot_scale='linear'):
    # Import Network
    network_file = os.path.expanduser(os.path.join(base_path, root_path, network_path))
    if network_type in ['BNS', 'NSBH']:
        fmin = 0.00004
        network_kwargs = dict(depth=4, width=512,
                              data_dim=640 if data_dim is None else data_dim, 
                              grid_dim=2,
                              cond_dim=6 if cond_dim is None else cond_dim)
    else:
        fmin = 0.0004
        network_kwargs = dict(depth=4, width=512,
                              data_dim=640 if data_dim is None else data_dim, 
                              grid_dim=2,
                              cond_dim=4 if cond_dim is None else cond_dim)
        
    vae_analyzer = PhaseModificationAnalysis(network_file, network_kwargs, min_fgeom=fmin)

    #def get_vae_latent_ppe(b, beta, m1, m2, chi1, chi2, l1=0., l2=0., cq1=0., cq2=0.):
    #    phi_func = lambda freqs: get_phi_ppe(freqs, m1, m2, chi1, chi2, b, beta)
    #    z1, z2 = vae_analyzer.extract_latent(phi_func, m1, m2, chi1, chi2)
    #    return z1, z2
    
    if injection_parameters is None:
        # Example injected waveform
        injection_parameters = dict(
            # # for mtot = 15
            # mass_1=9.,
            # mass_2=6.,
            # luminosity_distance=534.2871037194635,
            # for mtot = 35
            mass_1=21.,
            mass_2=14.,
            luminosity_distance=862.3762176154274,
            chi_1=0.,
            chi_2=0.,
            theta_jn=0.5235987755982988, 
            psi=2.659, 
            phase=1.3, 
            geocent_time=1126259642.413,
            ra=1.375, 
            dec=-1.2108,
        )
        if network_type in ['BNS']:
            injection_parameters['mass_1'] = 2.1
            injection_parameters['mass_2'] = 1.4
            injection_parameters['luminosity_distance'] = 40.7
            injection_parameters['chi_1'] = 0.01
            injection_parameters['chi_2'] = 0.012
            injection_parameters['lambda_1'] = 312
            injection_parameters['lambda_2'] = 1043
        if network_type in ['NSBH']:
            injection_parameters['mass_1'] = 3.0
            injection_parameters['lambda_1'] = 0.

    # Populate missing values which can be constructed from existing ones
    injection_parameters = generate_all_bbh_parameters(injection_parameters) if network_type in ['BBH'] else generate_all_bns_parameters(injection_parameters)

    injection_parameters['b'] = b_inj
    ppe_unit = get_ppe_bound(
        injection_parameters['b'],
        injection_parameters['mass_1'], 
        injection_parameters['mass_2'],
        injection_parameters['chi_1'],
        injection_parameters['chi_2'],
        injection_parameters.get('lambda_1', 0.),
        injection_parameters.get('lambda_2', 0.),
        injection_parameters.get('cq1', 0.),
        injection_parameters.get('cq2', 0.),
        ppe_ref=10)
    injection_parameters['beta'] = beta_rel_inj * ppe_unit
    z1, z2 = get_vae_latent_ppe(vae_analyzer,
            injection_parameters['b'],
            injection_parameters['beta'],
            injection_parameters['mass_1'], 
            injection_parameters['mass_2'],
            injection_parameters['chi_1'],
            injection_parameters['chi_2'],
            injection_parameters.get('lambda_1', 0.),
            injection_parameters.get('lambda_2', 0.),
            injection_parameters.get('cq1', 0.),
            injection_parameters.get('cq2', 0.))
    #injection_parameters['z_1'] = z1
    #injection_parameters['z_2'] = z2
    #injection_parameters['z_abs'] = np.sqrt(z1*z1 + z2*z2)
    #injection_parameters['z_theta'] = np.mod(np.arctan2(z2, z1), 2*np.pi)
    if 'chirp_mass' not in injection_parameters:
        mc = component_masses_to_chirp_mass(injection_parameters['mass_1'], injection_parameters['mass_2'])
    else:
        mc = injection_parameters['chirp_mass']
    if network_type not in ['BNS', 'NSBH']:
        lt = 0.
    elif 'lambda_tilde' not in injection_parameters:
        lt = lambda_1_lambda_2_to_lambda_tilde(injection_parameters.get('lambda_1', 0.), injection_parameters.get('lambda_2', 0.))
    else:
        lt = injection_parameters['lambda_tilde']
    plot_freqs = 10**(vae_analyzer.model_loggeom_freqs) / (np.pi * mc * MSUN_S)
    plot_freqs_geom = 10**(vae_analyzer.model_loggeom_freqs)

    print('Injection Parameters:')
    print('  |  b', b_inj)
    print('  |  beta', beta_rel_inj*ppe_unit)
    print('  |  z1', z1)
    print('  |  z2', z2)
    print('  |  z_abs', np.sqrt(z1**2 + z2**2))
    print('  |  z_theta', np.mod(np.arctan2(z2, z1), 2*np.pi))
    print('  |  chirp_mass', mc)
    if network_type in ['BNS', 'NSBH']:
        print('  |  lambda_tilde', lt)


    # Get injected phi_ppe
    inject_phi = get_phi_ppe(plot_freqs,
                             injection_parameters['mass_1'], injection_parameters['mass_2'],
                             injection_parameters['chi_1'], injection_parameters['chi_2'],
                             injection_parameters['b'], injection_parameters['beta'])

    # Get reconstructed phi_ppe
    recon_phi = vae_analyzer.phase_mod(plot_freqs,
                                       injection_parameters['mass_1'], injection_parameters['mass_2'],
                                       injection_parameters['chi_1'], injection_parameters['chi_2'],
                                       injection_parameters.get('lambda_1', 0.), injection_parameters.get('lambda_2', 0.),
                                       z1, z2)
    
    # Set up figure with main and secondary axes
    if plot_difference:
        fig, (ax_main, ax_secondary) = plt.subplots(2, 1, sharex=True, figsize=(8, 6),
                                                gridspec_kw={'height_ratios': [3, 1]})
    else:
        fig, ax_main = plt.subplots(1, 1, figsize=(8, 4))
        ax_secondary = None

    # Plot ppE injected phase mod vs. npE reconstruction
    ax_main.plot(plot_freqs, inject_phi, label='Injected')
    ax_main.plot(plot_freqs, recon_phi, label='Reconstructed', linestyle=':')
    if plot_difference:
        phi_rel_diff = np.abs(inject_phi-recon_phi)/np.abs(inject_phi)
        ax_secondary.plot(plot_freqs, phi_rel_diff, label='_Difference', linestyle='--', color='g')
        ax_secondary.set_yscale('log' if np.all(phi_rel_diff > 0.) else 'symlog' if np.any(phi_rel_diff > 0.) else 'linear')
        ax_secondary.set_ylabel('Rel. Difference')
        ax_secondary.set_xscale(plot_scale)
        ax_secondary.set_xlabel(r'$f$ [Hz]')
        ax_secondary.grid()
    else:
        ax_main.set_xlabel(r'$f$ [Hz]')
    

    if f_max is not None:
        ax_main.set_xlim(fmin, f_max)
        if plot_difference:
            ax_secondary.set_xlim(fmin, f_max)
    
    ax_main.set_ylabel(r'$\Delta \Phi$')
    ax_main.grid()
    ax_main.set_xscale(plot_scale)
    #ax_main.set_yscale('log')
    ax_main.set_title('ppE vs. npE:   ' + (r'{ $b_{ppE}$ = %.2f  |  $\beta_{ppE}$ = %.2e }' % (injection_parameters['b'], injection_parameters['beta'])))
    ax_main.legend()

    def freqs_to_geom(x):
        return x * (np.pi * mc * MSUN_S)
    def geom_to_freqs(x):
        return x / (np.pi * mc * MSUN_S)
    
    #ax2 = plt.gca().secondary_xaxis('top', functions=(geom_to_freqs, freqs_to_geom))
    #ax2.set_xlabel(r'$\bar{f}$')

    #plt.grid()
    plt.tight_layout()
    plt.show()


def plot_beta_max(b_ppe_min=-13., b_ppe_max=-1., b_ppe_num=1000,
                  m1=20., m2=35., chi1z=0.1, chi2z=0.1, 
                  l1=0., l2=0., cq1=0., cq2=0., ppe_ref=10):
    b_ppe_range = np.linspace(b_ppe_min, b_ppe_max, b_ppe_num)
    b_ppe_interp_range = np.linspace(-13., -1., 1000)
    b_ppe_valid = np.array([b_i for b_i in np.linspace(-13., -1., 13) if b_i <= b_ppe_max and b_i >= b_ppe_min])
    b_ppe_interp_valid = np.array([b_i for b_i in np.linspace(-13., -1., 13)])
    beta_max_vals = np.array([get_ppe_bound(b_ppe_val, m1, m2, chi1z, chi2z, l1, l2, cq1, cq2, ppe_ref) for b_ppe_val in b_ppe_range])
    beta_max_valid = np.array([get_ppe_bound(b_ppe_val, m1, m2, chi1z, chi2z, l1, l2, cq1, cq2, ppe_ref) for b_ppe_val in b_ppe_valid])
    beta_max_interp_valid = np.array([get_ppe_bound(b_ppe_val, m1, m2, chi1z, chi2z, l1, l2, cq1, cq2, ppe_ref) for b_ppe_val in b_ppe_interp_valid])

    # Plot original function and valid limits for defined integer values of b_ppE
    plt.plot(b_ppe_range, beta_max_vals, label='get_ppe_bound')
    plt.plot(b_ppe_valid, beta_max_valid, linestyle='None', marker='D', label='Valid')

    # Plot new interpolation function
    if False:
        # Separately interpolate b_ppe < -5 (linear) and b_ppe > -5 (nonlinear) sectors (in log-space)
        beta_interp_fn_1 = interp1d(b_ppe_interp_valid[:-4], np.log10(beta_max_interp_valid[:-4]), kind='linear', assume_sorted=True)
        beta_interp_fn_2 = interp1d(b_ppe_interp_valid[-5:], np.log10(beta_max_interp_valid[-5:]), kind='slinear', assume_sorted=True)
        beta_interp_fn = lambda b_ppe_in: 10**(beta_interp_fn_1(b_ppe_in)) if b_ppe_in <= -5. else 10**(beta_interp_fn_2(b_ppe_in))
    elif False:
        # Interpolate all b_ppe --> beta_ppe_max points directly
        beta_interp_fn_log = interp1d(b_ppe_interp_valid, np.log10(beta_max_interp_valid), kind='slinear')
        beta_interp_fn = lambda b: 10.**beta_interp_fn_log(b)
        beta_interp_fn_log_q = interp1d(b_ppe_interp_valid, np.log10(beta_max_interp_valid), kind='quadratic')
        beta_interp_fn_q = lambda b: 10.**beta_interp_fn_log_q(b)

        beta_max_interp = np.array([beta_interp_fn(b_ppe_val) for b_ppe_val in b_ppe_range])
        beta_max_interp_q = np.array([beta_interp_fn_q(b_ppe_val) for b_ppe_val in b_ppe_range])
        plt.plot(b_ppe_range, beta_max_interp, linestyle='--', label='linear')
        plt.plot(b_ppe_range, beta_max_interp_q, linestyle='--', label='quadratic')
    elif True:
        bound_fn = get_ppe_bound_fn_safe(m1, m2, chi1z, chi2z, l1, l2, cq1, cq2, ppe_ref)
        beta_max_interp_q3 = np.array([get_ppe_bound_safe(b_ppe_val, bound_fn) for b_ppe_val in b_ppe_range])

        plt.plot(b_ppe_range, beta_max_interp_q3, linestyle='--', label='quadratic')
    else:
        bound_fn = get_ppe_bound_fn_safe(m1, m2, chi1z, chi2z, l1, l2, cq1, cq2, ppe_ref)
        beta_max_interp_q2 = np.array([bound_fn(b_ppe_val) for b_ppe_val in b_ppe_range])

        plt.plot(b_ppe_range, beta_max_interp_q2, linestyle='--', label='quadratic')

    plt.grid()
    plt.xlabel(r'$b_{ppE}$')
    plt.ylabel(r'$\beta_{ppE,max}$')
    plt.yscale('log')
    plt.legend()

    plt.tight_layout()
    plt.show()






# ------------------------------------------------------------------
#
#                    RESULT PARAMETER COMPARISON UTILS
#
# ------------------------------------------------------------------


def get_missing_inj_params(inj_params):
    if inj_params is not None:
        inj_params = generate_mass_parameters(inj_params)
        inj_params = generate_spin_parameters(inj_params)
        if np.any([key in inj_params for key in ['lambda_1', 'lambda_2', 'lambda_tilde', 'lambda_delta_tilde']]):
            if 'lambda_tilde' in inj_params and not(np.any([key in inj_params for key in ['lambda_1', 'lambda_2']])):
                if 'delta_lambda_tilde' in inj_params:
                    inj_params[['lambda_1', 'lambda_2']] = lambda_tilde_delta_lambda_tilde_to_lambda_1_lambda_2(inj_params['lambda_tilde'], inj_params['delta_lambda_tilde'], inj_params['mass_1'], inj_params['mass_2'])
                else:
                    inj_params[['lambda_1', 'lambda_2']] = lambda_tilde_to_lambda_1_lambda_2(inj_params['lambda_tilde'], inj_params['mass_1'], inj_params['mass_2'])
            else:
                inj_params = generate_tidal_parameters(inj_params)

    return inj_params


# Get injected parameters for a run (including ppE to npE latent space parameter conversion if necessary)
def get_inj_params(pe_result: Result, vae_analyzer: PhaseModificationAnalysis=None, verbosity=0):

    inj_params = get_missing_inj_params(pe_result.injection_parameters)
    
    b_inj = inj_params.get('b', None)
    beta_inj = inj_params.get('beta', None)
    z_abs_inj = inj_params.get('z_abs', None)
    z_theta_inj = inj_params.get('z_theta', None)
    
    if np.all([var is None for var in [b_inj, beta_inj, z_abs_inj, z_theta_inj]]):
        return None, None, None, None
    
    if np.all([var is not None for var in [b_inj, beta_inj, z_abs_inj, z_theta_inj]]):
        return b_inj, beta_inj, z_abs_inj, z_theta_inj
    
    if vae_analyzer is None:
        return b_inj, beta_inj, z_abs_inj, z_theta_inj
    
    # We need intrinsic binary parameters if we are going to swap between ppE and npE bases
    if np.any([key in inj_params for key in ['mass_1','mass_2']]):
        m1_inj = inj_params['mass_1']
        m2_inj = inj_params['mass_2']
    else:
        mc = inj_params['chirp_mass']
        q  = inj_params['mass_ratio']
        m1_inj, m2_inj = chirp_mass_and_mass_ratio_to_component_masses(mc, q)
    chi1_inj = inj_params['chi_1']
    chi2_inj = inj_params['chi_2']

    # For futureproofing, we also get the tidal and spin deformation parameters
    if np.any([key in inj_params for key in ['lambda_1', 'lambda_2', 'lambda_tilde', 'lambda_delta_tilde']]):
        l1_inj = inj_params.get('lambda_1', 0.)
        l2_inj = inj_params.get('lambda_2', 0.)
    else:
        l1_inj = inj_params.get('lambda_1', None)
        l2_inj = inj_params.get('lambda_2', None)
    cq1_inj = inj_params.get('cq_1', None)
    cq2_inj = inj_params.get('cq_2', None)
    
    # Mask missing tidal parameters to 0 for other functions that expect them
    if np.all([tidal_param is None for tidal_param in [l1_inj, l2_inj, cq1_inj, cq2_inj]]):
        tidal_params = dict()
    else:
        tidal_params = {f'{p_name}': p_val if p_val is not None else 0. for p_name, p_val in zip(['l1', 'l2', 'cq1', 'cq2'], [l1_inj, l2_inj, cq1_inj, cq2_inj])}
    if verbosity >= 3:
        print('tidal params: ', tidal_params)

    # Get interpolation function to map npE latent space angles to ppE theory angles
    bppe_to_ztheta, ztheta_to_bppe = map_latent_space_angles(vae_analyzer, beta_ppe_rel=(beta_inj if beta_inj is not None else z_abs_inj),
                                                             m1=m1_inj, m2=m2_inj, chi1=chi1_inj, chi2=chi2_inj, 
                                                             **tidal_params)


    if z_theta_inj is None or z_abs_inj is None:
        if b_inj is not None and beta_inj is not None:
            z1_inj, z2_inj = get_vae_latent_ppe(vae_analyzer,
                    b_inj,
                    beta_inj,
                    m1_inj,
                    m2_inj,
                    chi1_inj,
                    chi2_inj,
                    **tidal_params)
            z_abs_inj = np.sqrt(z1_inj*z1_inj + z2_inj*z2_inj)
            z_theta_inj = np.mod(np.arctan2(z2_inj, z1_inj), 2*np.pi)
    else:
        if b_inj is None:
            #z1_inj, z2_inj = get_vae_latent(vae_analyzer, m1_inj, m2_inj, chi1_inj, chi2_inj)
            b_inj = ztheta_to_bppe(z_theta_inj)
        if beta_inj is None:
            #beta_ppe_max = get_ppe_bound(m1_inj, m2_inj, chi1_inj, chi2_inj, b_inj)
            #beta_inj = beta_inj_rel * beta_ppe_max
            
            beta_to_zabs, zabs_to_beta = map_latent_space_scales(vae_analyzer, b_ppe=b_inj,
                                                                 m1=m1_inj, m2=m2_inj, chi1=chi1_inj, chi2=chi2_inj,
                                                                 **tidal_params)
            beta_inj = zabs_to_beta(z_abs_inj)
    
    return b_inj, beta_inj, z_abs_inj, z_theta_inj

# Get recovered posteriors for a run (including ppE to npE latent space parameter conversion if necessary)
def get_rec_params(result: Result, vae_analyzer: PhaseModificationAnalysis, use_median=False, verbosity=0):

    inj_params = get_missing_inj_params(result.injection_parameters)

    if inj_params is not None:
        b_inj = inj_params.get('b', None)
        beta_inj = inj_params.get('beta', None)
        z_abs_inj = inj_params.get('z_abs', None)
        z_theta_inj = inj_params.get('z_theta', None)
    else:
        b_inj = None
        beta_inj = None
        z_abs_inj = None
        z_theta_inj = None

    result_posterior = pd.DataFrame(result.posterior.copy())
    if verbosity >= 3:
        print('Posterior columns:', result_posterior.columns)

    # Populate additional parameters of interest from sample posteriors
    if not(np.all(['mass_1' in result_posterior.columns, 'mass_2' in result_posterior.columns])):
        result_posterior[['mass_1', 'mass_2']] = result_posterior[['chirp_mass', 'mass_ratio']].apply(chirp_mass_and_mass_ratio_to_component_masses, axis=1)
    result_posterior = result_posterior.apply(generate_mass_parameters, axis=1)
    result_posterior = result_posterior.apply(generate_spin_parameters, axis=1)
    if np.any(['lambda_1' in result_posterior.columns, 'lambda_2' in result_posterior.columns, 'lambda_tilde' in result_posterior.columns]):
        result_posterior = result_posterior.apply(generate_tidal_parameters, axis=1)
    
    mle_idx = np.argmax(result_posterior.log_likelihood)
    if 'b' in result_posterior:
        if use_median:
            b_rec = result_posterior.b.median()
        else:
            b_rec = result_posterior.iloc[mle_idx]['b']
    else:
        b_rec = None
    if 'beta' in result_posterior:
        if use_median:
            beta_rec = result_posterior.beta.median()
        else:
            beta_rec = result_posterior.iloc[mle_idx]['beta']
    else: 
        beta_rec = None
    if 'z_abs' in result_posterior:
        if use_median:
            z_abs_rec = result_posterior.z_abs.median()
        else:
            z_abs_rec = result_posterior.iloc[mle_idx]['z_abs']
    else:
        z_abs_rec = None
    if 'z_theta' in result_posterior:
        if use_median:
            z_theta_rec = result_posterior.z_theta.median()
        else:
            z_theta_rec = result_posterior.iloc[mle_idx]['z_theta']
    else:
        z_theta_rec = None
    
    # If all parameters are None, we can't proceed.
    if np.all([var is not None for var in [b_rec, beta_rec, z_abs_rec, z_theta_rec]]):
        return b_rec, beta_rec, z_abs_rec, z_theta_rec

    # If we aren't capable of doing further npE VAE latent space analysis, we are done here.
    if vae_analyzer is None:
        return b_rec, beta_rec, z_abs_rec, z_theta_rec
    

    # Get posterior MLE/median values for m1, m2, chi1, chi2
    if np.any([key in result_posterior for key in ['mass_1','mass_2']]):
        if use_median:
            m1_rec = result_posterior['mass_1'].median()
            m2_rec = result_posterior['mass_2'].median()
        else:
            m1_rec = result_posterior.iloc[mle_idx]['mass_1']
            m2_rec = result_posterior.iloc[mle_idx]['mass_2']
    else:
        if use_median:
            mc = result_posterior['chirp_mass'].median()
            q  = result_posterior['mass_ratio'].median()
        else:
            mc = result_posterior.iloc[mle_idx]['chirp_mass']
            q = result_posterior.iloc[mle_idx]['mass_ratio']
        m1_rec, m2_rec = chirp_mass_and_mass_ratio_to_component_masses(mc, q)
    if use_median:
        chi1_rec = result_posterior['chi_1'].median()
        chi2_rec = result_posterior['chi_2'].median()
    else:
        chi1_rec = result_posterior.iloc[mle_idx]['chi_1']
        chi2_rec = result_posterior.iloc[mle_idx]['chi_2']

    # Get posterior median values for lambda1, lambda2
    if np.all([key in result.posterior for key in ['lambda_1', 'lambda_2']]):
        if use_median:
            l1_rec = result_posterior['lambda_1'].median()
            l2_rec = result_posterior['lambda_2'].median()
        else:
            l1_rec = result_posterior.iloc[mle_idx]['lambda_1']
            l2_rec = result_posterior.iloc[mle_idx]['lambda_2']
    else:
        l1_rec_post = result_posterior.get('lambda_1', None) if use_median else result_posterior.iloc[mle_idx].get('lambda_1', None)
        l1_rec = l1_rec_post.median() if use_median else l1_rec_post
        l2_rec_post = result_posterior.get('lambda_2', None) if use_median else result_posterior.iloc[mle_idx].get('lambda_2', None)
        l2_rec = l2_rec_post.median() if use_median else l2_rec_post
    cq1_rec_post = result_posterior.get('cq_1', None) if use_median else result_posterior.iloc[mle_idx].get('cq_1', None)
    cq1_rec = cq1_rec_post.median() if use_median else cq1_rec_post
    cq2_rec_post = result_posterior.get('cq_2', None) if use_median else result_posterior.iloc[mle_idx].get('cq_2', None)
    cq2_rec = cq2_rec_post.median() if use_median else cq2_rec_post
    
    # Mask missing tidal parameters to 0 for other functions that expect them
    if np.all([tidal_param is None for tidal_param in [l1_rec, l2_rec, cq1_rec, cq2_rec]]):
        tidal_params_rec = dict()
    else:
        tidal_params_rec = {f'{p_name}': p_val if p_val is not None else 0. for p_name, p_val in zip(['l1', 'l2', 'cq1', 'cq2'], [l1_rec, l2_rec, cq1_rec, cq2_rec])}
    if verbosity >= 3:
        print('recovered tidal params: ', tidal_params_rec)


    # Get injected values for m1, m2, chi1, chi2
    if np.any([key in inj_params for key in ['mass_1','mass_2']]):
        m1_inj = inj_params['mass_1']
        m2_inj = inj_params['mass_2']
    else:
        mc = inj_params['chirp_mass']
        q  = inj_params['mass_ratio']
        m1_inj, m2_inj = chirp_mass_and_mass_ratio_to_component_masses(mc, q)
    chi1_inj = inj_params['chi_1']
    chi2_inj = inj_params['chi_2']

    # Get injected values for lambda1, lambda2
    if np.any([key in inj_params for key in ['lambda_1', 'lambda_2']]):
        l1_inj = inj_params.get('lambda_1', 0.)
        l2_inj = inj_params.get('lambda_2', 0.)
    #elif np.any([key in inj_params for key in ['lambda_1', 'lambda_2', 'lambda_tilde']]):
    #    l1_inj, l2_inj = generate_tidal_parameters(inj_params)[['lambda_1', 'lambda_2']]
    else:
        l1_inj = inj_params.get('lambda_1', None)
        l2_inj = inj_params.get('lambda_2', None)
    cq1_inj = inj_params.get('cq_1', None)
    cq2_inj = inj_params.get('cq_2', None)

    if np.all([tidal_param is None for tidal_param in [l1_inj, l2_inj, cq1_inj, cq2_inj]]):
        tidal_params_inj = dict()
    else:
        tidal_params_inj = {f'{p_name}': p_val if p_val is not None else 0. for p_name, p_val in zip(['l1', 'l2', 'cq1', 'cq2'], [l1_inj, l2_inj, cq1_inj, cq2_inj])}


    # TODO: Should we set this based on injected parameters, or recovered ones?
    # (Which is more helpful for the purpose of diagnosing errored runs?)
    if b_rec is None:
        beta_ppe_max = get_ppe_bound_safe(b_inj, None, m1_rec, m2_rec, chi1_rec, chi2_rec, **tidal_params_rec)
    else:
        beta_ppe_max = get_ppe_bound_safe(b_rec, None, m1_rec, m2_rec, chi1_rec, chi2_rec, **tidal_params_rec)
    #beta_ppe_max = get_ppe_bound(b_inj, m1_inj, m2_inj, chi1_inj, chi2_inj, **tidal_params_rec)
    beta_inj_rel = np.abs(beta_inj)/beta_ppe_max if beta_inj is not None else None
    beta_rec_rel = np.abs(beta_rec)/beta_ppe_max if beta_rec is not None else None

    # Train theory mapping according to "true" (injected) parameter relations
    bppe_to_ztheta, ztheta_to_bppe = map_latent_space_angles(vae_analyzer, beta_ppe_rel=(beta_inj_rel if beta_inj_rel is not None else z_abs_inj),
                                                             m1=m1_inj, m2=m2_inj, chi1=chi1_inj, chi2=chi2_inj,
                                                             **tidal_params_inj)
    # Alternative (unused) version which uses recovered parameter relations to estimate theory angle mapping
    #bppe_to_ztheta, ztheta_to_bppe = map_latent_space_angles(vae_analyzer, beta_ppe_rel=(beta_rec_rel if beta_rec_rel is not None else z_abs_rec),
    #                                                         m1=m1_rec, m2=m2_rec, chi1=chi1_rec, chi2=chi2_rec,
    #                                                         **tidal_params_rec)

    if z_theta_rec is None or z_abs_rec is None:
        if b_rec is not None and beta_rec is not None:
            z1_rec, z2_rec = get_vae_latent_ppe(vae_analyzer,
                    b_rec,
                    beta_rec,
                    m1_rec,
                    m2_rec,
                    chi1_rec,
                    chi2_rec,
                    **tidal_params_rec)
            z_abs_rec = np.sqrt(z1_rec*z1_rec + z2_rec*z2_rec)
            z_theta_rec = np.mod(np.arctan2(z2_rec, z1_rec), 2*np.pi)
    else:
        if b_rec is None:
            #z1_rec, z2_rec = get_vae_latent(vae_analyzer, m1_inj, m2_inj, chi1_inj, chi2_inj)
            b_rec = ztheta_to_bppe(z_theta_rec)
        if beta_rec is None:
            #beta_ppe_max = get_ppe_bound(m1_inj, m2_inj, chi1_inj, chi2_inj, b_rec)
            #beta_rec = beta_rec_rel * beta_ppe_max
            
            # Train mapping according to "true" (injected) parameter relations
            beta_to_zabs, zabs_to_beta = map_latent_space_scales(vae_analyzer, b_ppe=b_inj,
                                                                 m1=m1_inj, m2=m2_inj, chi1=chi1_inj, chi2=chi2_inj,
                                                                 **tidal_params_inj)
            #beta_to_zabs, zabs_to_beta = map_latent_space_scales(vae_analyzer, b_ppe=b_rec,
            #                                                     m1=m1_rec, m2=m2_rec, chi1=chi1_rec, chi2=chi2_rec,
            #                                                     **tidal_params_rec)
            beta_rec = zabs_to_beta(z_abs_rec)
    
    return b_rec, beta_rec, z_abs_rec, z_theta_rec

def compare_params(result_name, network_name=None, verbosity=1, root_path=project_base_path, network_type='BBH', data_dim=None):
    if network_name is None:
        vae_analyzer = None
    else:
        network_file = os.path.abspath(os.path.expanduser(os.path.join(root_path, '.', network_name)))
        if verbosity >= 2:
            print('network_file: ', network_file)
        if network_type in ['BNS']:
            fmin = 0.00004
            network_kwargs = dict(depth=4, width=512, 
                                  data_dim=640 if data_dim is None else data_dim,
                                  cond_dim=4, grid_dim=2)
        else:
            fmin = 0.0004
            network_kwargs = dict(depth=4, width=512, 
                                  data_dim=640 if data_dim is None else data_dim,
                                  cond_dim=4, grid_dim=2)
        vae_analyzer = PhaseModificationAnalysis(network_file, network_kwargs, min_fgeom=fmin)

    result_file = os.path.abspath(os.path.expanduser(os.path.join(root_path, 'logs', 'pe', result_name, 'npe-pe_result.json')))
    if verbosity >= 2:
        print('result_file: ', result_file)
    pe_result = Result.from_json(result_file)

    b_ppe_inj, beta_ppe_inj, z_abs_inj, z_theta_inj = get_inj_params(pe_result, vae_analyzer, verbosity=verbosity)
    if verbosity >= 2:
        print('b_ppe_inj: ', b_ppe_inj)
        print(type(b_ppe_inj))

    b_ppe_rec, beta_ppe_rec, z_abs_rec, z_theta_rec = get_rec_params(pe_result, vae_analyzer, verbosity=verbosity)
    if verbosity >= 2:
        print('b_ppe_rec: ', b_ppe_rec)
        print(type(b_ppe_rec))

    if verbosity >= 1:
        if verbosity >= 2:
            print('------------------------------------------------------')
        print('b_ppe:')
        print('   inj: %.2f     rec: %.2f' % (b_ppe_inj, b_ppe_rec))
        print('beta_ppe:')
        print('   inj: %.2e     rec: %.2e' % (beta_ppe_inj, beta_ppe_rec))
        print('z_theta:')
        print('   inj: %.2f     rec: %.2f' % (z_theta_inj, z_theta_rec))
        print('z_abs:')
        print('   inj: %.2f     rec: %.2f' % (z_abs_inj, z_abs_rec))

    result_dict = {'b_ppe_inj': b_ppe_inj, 'b_ppe_rec': b_ppe_rec, 'beta_ppe_inj': beta_ppe_inj, 'beta_ppe_rec': beta_ppe_rec,
                   'z_theta_inj': z_theta_inj, 'z_theta_rec': z_theta_rec, 'z_abs_inj': z_abs_inj, 'z_abs_rec': z_abs_rec}
    
    return result_dict


