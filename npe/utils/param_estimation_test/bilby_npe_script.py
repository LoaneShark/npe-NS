#!/usr/bin/env python

import argparse
import os
import time
from datetime import datetime
import pickle
import numpy as np
import pandas as pd

import lal
import lalsimulation
import bilby
from bilby.gw.conversion import (
    component_masses_to_chirp_mass, 
    component_masses_to_symmetric_mass_ratio,
)
from bilby.core.utils import logger
from astropy import units as u
from astropy import constants as c
from importlib import import_module

from npe_wf_analysis import PhaseModificationAnalysis

parser = argparse.ArgumentParser()
parser.add_argument("-l", "--label", type=str, default="bilby",
                    help="Label of the bilby run.")
parser.add_argument("-n", "--npool", type=int, default=1, 
                    help="Number of CPUs.")
parser.add_argument("-t", "--check-point-delta-t", type=int, default=3600, 
                    help="Seconds between checkpoints.")
parser.add_argument("--b-inj", type=float, default=-5,
                    help="PPE b for injection.")
parser.add_argument("--beta-rel-inj", type=float, default=-0.,
                    help="PPE beta for injection, relative to the post-Einsteinian boundary.")
parser.add_argument("--rootdir", type=str, default="~/npe",
                    help="Path to npe project root directory.")
parser.add_argument("--network", type=str, default="npe_network.pt",
                    help="Name of the VAE to load.")
parser.add_argument("--save_results", type=argparse.BooleanOptionalAction, default=True, 
                    help="Toggle whether or not all data files are saved manually")
args = parser.parse_args()

NPOOL = args.npool
CHECKPOINT_DELTAT = args.check_point_delta_t

ROOTDIR = os.path.expanduser(args.rootdir)
PE_DIR  = os.path.join(ROOTDIR, 'npe', 'utils', 'param_estimation_test')

outdir = os.path.join(ROOTDIR, 'logs', 'pe')
label = args.label
save_results = args.save_results
network_name = args.network
bilby.core.utils.setup_logger(outdir=outdir, label=label)

np.random.seed(1234)

MSUN_KM = lal.MSUN_SI * lal.G_SI / lal.C_SI ** 2 / 1e3
MSUN_S  = MSUN_KM / lal.C_SI * 1e3

def get_phi_ppe(frequency_array, mass_1, mass_2, chi_1, chi_2, b, beta):
    mtot = mass_1 + mass_2
    mc = component_masses_to_chirp_mass(mass_1, mass_2)
    freqs = np.asarray(frequency_array, dtype=np.float64)
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

def source_model_inj(frequency_array, mass_1, mass_2, luminosity_distance, 
                     a_1, tilt_1, phi_12, a_2, tilt_2, phi_jl, theta_jn, phase, 
                     b, beta, **kwargs):
    chi_1 = a_1 * np.cos(tilt_1)
    chi_2 = a_2 * np.cos(tilt_2)
    freqs = np.append(frequency_array, kwargs['reference_frequency'])
    phi = get_phi_ppe(freqs, mass_1, mass_2, chi_1, chi_2, b, beta)
    phi = phi[:-1] - phi[-1]
    polarizations = bilby.gw.source.lal_binary_black_hole(
        frequency_array, mass_1, mass_2, luminosity_distance, 
        a_1, tilt_1, phi_12, a_2, tilt_2, phi_jl, theta_jn, phase, **kwargs)
    for k in polarizations:
        polarizations[k] *= np.exp(-1j * phi)
    return polarizations

logger.info('Loading VAE: %s', network_name)
#network_file = os.path.join(PE_DIR, "npe_network.pt")
network_file = os.path.join(PE_DIR, network_name)
network_kwargs = dict(depth=4, width=512,
                      data_dim=640, grid_dim=2)
vae_analyzer = PhaseModificationAnalysis(network_file, network_kwargs)

def source_model_rec(frequency_array, mass_1, mass_2, luminosity_distance, 
                     a_1, tilt_1, phi_12, a_2, tilt_2, phi_jl, theta_jn, phase, 
                     z_abs, z_theta, **kwargs):
    chi_1 = a_1 * np.cos(tilt_1)
    chi_2 = a_2 * np.cos(tilt_2)
    z_1 = z_abs * np.cos(z_theta)
    z_2 = z_abs * np.sin(z_theta)
    freqs = np.append(frequency_array, kwargs['reference_frequency'])
    phi = vae_analyzer.phase_mod(freqs, mass_1, mass_2, chi_1, chi_2, z_1, z_2)
    phi = phi[:-1] - phi[-1]
    polarizations = bilby.gw.source.lal_binary_black_hole(
        frequency_array, mass_1, mass_2, luminosity_distance, 
        a_1, tilt_1, phi_12, a_2, tilt_2, phi_jl, theta_jn, phase, **kwargs)
    for k in polarizations:
        polarizations[k] *= np.exp(-1j * phi)
    return polarizations

def get_vae_latent(m1, m2, chi1, chi2, b, beta):
    phi_func = lambda freqs: get_phi_ppe(freqs, m1, m2, chi1, chi2, b, beta)
    z1, z2 = vae_analyzer.extract_latent(phi_func, m1, m2, chi1, chi2)
    return z1, z2

def get_ppe_bound(m1, m2, chi1z, chi2z, b_ppe, ppe_ref=10):
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
    for i in range(4, 8):
        # tidal deformation and spin deformation
        param_vecs[i].data = np.zeros_like(param_vecs[i].data)
    pn_coeff = lalsimulation.SimInspiralTaylorF2AlignedPhasingArray(*param_vecs).data
    pn_coeff = pn_coeff[:len(pn_coeff)//3] # remove coeffs for vlogv and vlogvsq
    if b_ref != -4:
        pn_coeff = pn_coeff.reshape(-1, num_samples)[b_ref+5] # take only the pn coeff of the ppE order
        beta_ppe_bound = np.abs(pn_coeff) * ppe_ref_v**(b_ref-b_ppe)
    else:
        pn_coeff1 = pn_coeff.reshape(-1, num_samples)[0]
        pn_coeff2 = pn_coeff.reshape(-1, num_samples)[2]
        beta_ppe_bound = np.sqrt(np.abs(pn_coeff1) * np.abs(pn_coeff2))
    beta_ppe_bound /= eta**(b_ppe/5.)
    return beta_ppe_bound[0]

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
injection_parameters['b'] = args.b_inj
ppe_unit = get_ppe_bound(
    injection_parameters['mass_1'], 
    injection_parameters['mass_2'],
    injection_parameters['chi_1'],
    injection_parameters['chi_2'],
    injection_parameters['b'],
    ppe_ref=10)
injection_parameters['beta'] = args.beta_rel_inj * ppe_unit
z1, z2 = get_vae_latent(
        injection_parameters['mass_1'], 
        injection_parameters['mass_2'],
        injection_parameters['chi_1'],
        injection_parameters['chi_2'],
        injection_parameters['b'],
        injection_parameters['beta'])
injection_parameters['z_1'] = z1
injection_parameters['z_2'] = z2
injection_parameters['z_abs'] = np.sqrt(z1*z1 + z2*z2)
injection_parameters['z_theta'] = np.mod(np.arctan2(z2, z1), 2*np.pi)

# Fix to 128s for BNS
duration = bilby.gw.detector.get_safe_signal_duration(
        injection_parameters['mass_1'],
        injection_parameters['mass_2'],
        injection_parameters['chi_1'],
        injection_parameters['chi_2'],
        0.,0.,flow=10)
start_time = injection_parameters['geocent_time'] + 2 - duration
mtot = injection_parameters['mass_1'] + injection_parameters['mass_2']
fcut = 0.018 / (mtot * MSUN_S) # inspiral cutoff in phenomd
sampling_frequency = min(4096, int(2 * fcut))
reference_frequency = 20
minimum_frequency = 10

waveform_arguments = dict(waveform_approximant='IMRPhenomD',
                          reference_frequency=reference_frequency, 
                          minimum_frequency=minimum_frequency)
waveform_generator_inj = bilby.gw.WaveformGenerator(
    duration=duration, 
    sampling_frequency=sampling_frequency,
    frequency_domain_source_model=source_model_inj,
    parameter_conversion=bilby.gw.conversion.convert_to_lal_binary_black_hole_parameters,
    waveform_arguments=waveform_arguments)
waveform_generator_rec = bilby.gw.WaveformGenerator(
    duration=duration, 
    sampling_frequency=sampling_frequency,
    frequency_domain_source_model=source_model_rec,
    parameter_conversion=bilby.gw.conversion.convert_to_lal_binary_black_hole_parameters,
    waveform_arguments=waveform_arguments)

interferometers = bilby.gw.detector.InterferometerList(['H1', 'L1', 'V1'])
for interferometer in interferometers:
    interferometer.power_spectral_density = bilby.gw.detector.PowerSpectralDensity(asd_file='Aplus_asd.txt')
    interferometer.minimum_frequency = minimum_frequency
interferometers.set_strain_data_from_zero_noise(
    sampling_frequency=sampling_frequency, 
    duration=duration, start_time=start_time)
interferometers.inject_signal(parameters=injection_parameters,
                              waveform_generator=waveform_generator_inj)
interferometers.plot_data(outdir=outdir, label=label)
if save_results:
    interferometers.save_data(outdir=outdir, label=label)

priors = bilby.gw.prior.BBHPriorDict(aligned_spin=True, conversion_function=bilby.gw.conversion.convert_to_lal_binary_black_hole_parameters)
for key in ['psi', 'ra', 'dec', 'theta_jn', 'luminosity_distance']:
    priors[key] = injection_parameters[key]
priors['geocent_time'] = bilby.core.prior.Uniform(
    minimum=injection_parameters['geocent_time'] - 0.1, 
    maximum=injection_parameters['geocent_time'] + 0.1, name='geocent_time')
priors['chirp_mass'] = bilby.core.prior.Uniform(
    minimum=1., maximum=80., name='chirp_mass', unit='$M_{\\odot}$')
priors['mass_ratio'] = bilby.core.prior.Uniform(
    minimum=0.125, maximum=1., name='mass_ratio')
priors.pop('mass_1')
priors.pop('mass_2')
priors['z_abs'] = bilby.core.prior.Uniform(minimum=0., maximum=1., name='z_abs')
# priors['z_abs'] = bilby.core.prior.PowerLaw(minimum=0., maximum=1., alpha=1., name='z_abs') # this will be uniform in z1 and z2
priors['z_theta'] = bilby.core.prior.Uniform(minimum=0., maximum=2*np.pi, name='z_theta', boundary='periodic')

likelihood = bilby.gw.GravitationalWaveTransient(
    interferometers=interferometers, 
    waveform_generator=waveform_generator_rec,
    time_marginalization=False, 
    phase_marginalization=True, 
    distance_marginalization=False, 
    priors=priors)

result = bilby.run_sampler(
    likelihood=likelihood, priors=priors, 
    sampler='dynesty',
    sample='acceptance-walk',
    nlive=1000,
    naccept=60,
    dlogz=0.1,
    npool=NPOOL,
    check_point_delta_t=CHECKPOINT_DELTAT,
    injection_parameters=injection_parameters, outdir=outdir, label=label,
    conversion_function=bilby.gw.conversion.generate_all_bbh_parameters)

if save_results:
    result.save_posterior_samples()
    #result.save_to_file()

result.plot_corner()

# Plot reconstructed waveform posterior over detector noise/ASD
from bilby.core.result import result_file_name
from bilby.gw.result import CBCResult

# Reload results as CBCResult class
output_file = result_file_name(outdir=outdir, label=label)
cbc_result = CBCResult.from_json(output_file, outdir=outdir, label=label)

for ifo in interferometers:
    cbc_result.plot_interferometer_waveform_posterior(interferometer=ifo, n_samples=1000, save=True)