import os
import sys
import shutil
from datetime import datetime
import multiprocessing
import argparse

import numpy as np
import pandas as pd
from tqdm import tqdm

from astropy import units as u
from astropy import constants as c
import lal, lalsimulation
import bilby

from bilby.gw.conversion import (chirp_mass_and_mass_ratio_to_component_masses, total_mass_and_mass_ratio_to_component_masses, 
                                 component_masses_to_chirp_mass, component_masses_to_mass_ratio, component_masses_to_total_mass,
                                 lambda_tilde_to_lambda_1_lambda_2, lambda_1_lambda_2_to_lambda_tilde)


def get_cli():
    parser = argparse.ArgumentParser(
        "Create an npy file with phase rescaling factors for the original dataset")

    parser.add_argument("--psd-path", type=str, nargs='+',
                        help="Path(s) to the PSD estimate.")
    parser.add_argument("--asd", action="store_true", default=False,
                        help="Whether the noise curve is given in ASD.")
    parser.add_argument("--dataset-path", type=str,
                        help="Path to the original dataset.")
    parser.add_argument("--use-tidal-data", action="store_true", default=False,
                        help="Toggle whether or not tidal data labels are expected (cond_dim == 6).")
    parser.add_argument("--use-tidal-data-full", action="store_true", default=False,
                        help="Toggle whether or not tidal data labels are expected (cond_dim == 8).")
    parser.add_argument("--min-over-tphi", action="store_true", default=False)
    parser.add_argument("--min-over-m1m2", action="store_true", default=False)
    parser.add_argument("--min-over-q", action="store_true", default=False)
    parser.add_argument("--min-over-mc", action="store_true", default=False)
    parser.add_argument("--min-over-lambda", action="store_true", default=False)
    parser.add_argument("--use-1p5pn-ref-phase", action="store_true", default=False)
    parser.add_argument("--use-2p5pn-ref-phase", action="store_true", default=False)
    parser.add_argument("--pool", default=1, type=int, help="Pool size.")
    # output
    parser.add_argument("--output-rootdir", required=True,
                        help="Output directory.")

    args = parser.parse_args()
    return args

# 0PN phase in GR
def get_0pn_phases(freqs, m1, m2):
    mc = bilby.gw.conversion.component_masses_to_chirp_mass(m1, m2)
    return 3./128. * (np.pi*mc*freqs*lal.MTSUN_SI)**(-5./3.)

# 0PN + 1PN phase in GR
def get_0to1p5pn_phases(freqs, m1, m2):
    mtot = m1 + m2
    eta = m1 * m2 / mtot / mtot
    v3 = np.pi * mtot * freqs * lal.MTSUN_SI
    v5 = np.power(v3, 5./3.)
    cbase = 3. / 128. / eta
    c0 = 1.
    c2 = 3715. / 756. + 55. / 9. * eta
    phases = cbase * (c0 / v5 + c2 / v3)
    return phases

# 0PN, 1PN, and 1.5PN phase in GR (with aligned spins)
def get_0to2pn_phases(freqs, m1, m2, chi1, chi2):
    mtot = m1 + m2
    eta = m1 * m2 / mtot / mtot
    delta_m = (m1 - m2) / mtot
    chi_s = (chi1 + chi2) / 2.
    chi_a = (chi1 - chi2) / 2.

    v3 = np.pi * mtot * freqs * lal.MTSUN_SI
    v5 = np.power(v3, 5./3.)
    v2 = np.power(v3, 2./3.)

    cbase = 3. / 128. / eta

    c0 = 1.
    c2 = 3715. / 756. + 55. / 9. * eta
    c3 = -16. * np.pi + 113. / 3. * delta_m * chi_a + (113. / 3. - 76. / 3. * eta) * chi_s
    phases = cbase * (c0/v5 + c2/v3 + c3/v2)
    return phases

# 0PN, 1PN, 1.5PN, and 2PN phase in GR (with aligned spins)
def get_0to2p5pn_phases(freqs, m1, m2, chi1, chi2):
    mtot = m1 + m2
    eta = m1 * m2 / mtot / mtot
    delta_m = (m1 - m2) / mtot
    chi_s = (chi1 + chi2) / 2.
    chi_a = (chi1 - chi2) / 2.

    v3 = np.pi * mtot * freqs * lal.MTSUN_SI
    v5 = np.power(v3, 5./3.)
    v2 = np.power(v3, 2./3.)
    v1 = np.power(v3, 1./3.)

    cbase = 3. / 128. / eta

    c0 = 1.
    c2 = 3715. / 756. + 55. / 9. * eta
    c3 = -16. * np.pi + 113. / 3. * delta_m * chi_a + (113. / 3. - 76. / 3. * eta) * chi_s
    c4 = (15293365 / 508032) + (27145 / 504) * eta + \
         3085/72 * eta**2 + (-405/8 + 200 * eta) * chi_a**2 - \
         405/4 * delta_m * chi_a * chi_s + (-405/8 + 5/2 * eta) * chi_s**2
    phases = cbase * (c0/v5 + c2/v3 + c3/v2 + c4/v1)
    return phases

# Get N_eff ^ 2 without minimization
# TODO: Do we want to include a more robust expression for the amplitude?
def get_neff2_logf_nomin_over(freqs, phases, *psd_interp):
    a2 = freqs ** (-7./3.)
    sn_inv = 0.
    for interp in psd_interp:
        sn_inv += 1. / interp(freqs)
    return np.sum(freqs * a2 * sn_inv * phases * phases)

# Get N_eff ^ 2 with minimization over time shift and phase shift
def get_neff2_logf_min_over_tphi(freqs, phases, *psd_interp):
    from scipy.optimize import minimize
    def loss_func(x):
        dphi, dt = x
        phases_shifted = phases + 2. * np.pi * freqs * dt - dphi
        return get_neff2_logf_nomin_over(freqs, phases_shifted, *psd_interp)
    vals = phases
    dvals = (vals[2:]-vals[:-2]) / (freqs[1:-1] * (np.log(freqs[2:])-np.log(freqs[:-2]))) / 2. / np.pi
    vmax = np.max(np.abs(vals))
    dvmax = np.max(np.abs(dvals))
    res = minimize(loss_func, x0=(0.,0.), bounds=((-vmax,vmax),(-dvmax,dvmax)))
    return res.fun

# Get N_eff ^ 2 with minimization over time shift, phase shift, and mass ratio
# NOTE: Total mass is assumed to be fixed to the reference total mass
def get_neff2_logf_min_over_tphiq(freqs, phases, m1ref, m2ref, *psd_interp):
    from scipy.optimize import minimize
    qref = component_masses_to_mass_ratio(m1ref, m2ref)
    mtotref = component_masses_to_total_mass(m1ref, m2ref)
    def loss_func(x):
        dphi, dt, q = x
        phases_shifted = phases + 2. * np.pi * freqs * dt - dphi
        m1, m2 = total_mass_and_mass_ratio_to_component_masses(mtotref, q)
        #phases_shifted += get_0pn_phases(freqs, m1ref, m2ref) - get_0pn_phases(freqs, m1, m2)
        phases_shifted += get_0to1p5pn_phases(freqs, m1ref, m2ref) - get_0to1p5pn_phases(freqs, m1, m2)
        return get_neff2_logf_nomin_over(freqs, phases_shifted, *psd_interp)
    vals = phases
    dvals = (vals[2:]-vals[:-2]) / (freqs[1:-1] * (np.log(freqs[2:])-np.log(freqs[:-2]))) / 2. / np.pi
    vmax = np.max(np.abs(vals))
    dvmax = np.max(np.abs(dvals))
    res = minimize(loss_func, 
                   x0=(0., 0., qref), 
                   bounds=((-vmax,vmax), (-dvmax,dvmax), (0.1, 1.)))
    return res.fun

# Get N_eff ^ 2 with minimization over time shift, phase shift, mass ratio, and chirp mass
# NOTE: Chirp mass range is assumed to be an NS mass range for now
def get_neff2_logf_min_over_tphiqmc(freqs, phases, m1ref, m2ref, *psd_interp):
    from scipy.optimize import minimize
    qref = component_masses_to_mass_ratio(m1ref, m2ref)
    mcref = component_masses_to_chirp_mass(m1ref, m2ref)
    mcref_min = min(0.2, 0.1*mcref)
    mcref_max = max(5., 10.*mcref)
    def loss_func(x):
        dphi, dt, q, mc = x
        phases_shifted = phases + 2. * np.pi * freqs * dt - dphi
        m1, m2 = chirp_mass_and_mass_ratio_to_component_masses(mc, q)
        #phases_shifted += get_0pn_phases(freqs, m1ref, m2ref) - get_0pn_phases(freqs, m1, m2)
        phases_shifted += get_0to1p5pn_phases(freqs, m1ref, m2ref) - get_0to1p5pn_phases(freqs, m1, m2)
        return get_neff2_logf_nomin_over(freqs, phases_shifted, *psd_interp)
    vals = phases
    dvals = (vals[2:]-vals[:-2]) / (freqs[1:-1] * (np.log(freqs[2:])-np.log(freqs[:-2]))) / 2. / np.pi
    vmax = np.max(np.abs(vals))
    dvmax = np.max(np.abs(dvals))
    res = minimize(loss_func, 
                   x0=(0.,0., qref, mcref), 
                   bounds=((-vmax,vmax), (-dvmax,dvmax), (0.1, 1.), (mcref_min, mcref_max)))
    return res.fun

# Get N_eff ^ 2 with minimization over time shift, phase shift, mass ratio, and tidal deformability
# TODO: WIP, we do not yet have an analytic expression for the tidal parameter phase corrections implemented
def get_neff2_logf_min_over_tphiqlambda(freqs, phases, m1ref, m2ref, chi1ref, chi2ref, lambdaref, *psd_interp):
    from scipy.optimize import minimize
    qref = component_masses_to_mass_ratio(m1ref, m2ref)
    #mcref = component_masses_to_chirp_mass(m1ref, m2ref)
    mtotref = component_masses_to_total_mass(m1ref, m2ref)
    def loss_func(x):
        dphi, dt, q, lambda_tilde = x
        m1, m2 = total_mass_and_mass_ratio_to_component_masses(mtotref, q)
        phases_shifted = phases + 2. * np.pi * freqs * dt - dphi
        phases_shifted += get_0to2p5pn_phases(freqs, m1ref, m2ref, chi1ref, chi2ref) - get_0to2p5pn_phases(freqs, m1, m2, chi1ref, chi2ref)
        # TODO: Implement tidal phase correction function
        #phases_shifted += get_tidal_phase_correction(freqs, m1ref, m2ref, chi1ref, chi2ref, lambdaref) - get_tidal_phase_correction(freqs, m1, m2, chi1ref, chi2ref, lambda_tilde)
        return get_neff2_logf_nomin_over(freqs, phases_shifted, *psd_interp)
    vals = phases
    dvals = (vals[2:]-vals[:-2]) / (freqs[1:-1] * (np.log(freqs[2:])-np.log(freqs[:-2]))) / 2. / np.pi
    vmax = np.max(np.abs(vals))
    dvmax = np.max(np.abs(dvals))
    res = minimize(loss_func, x0=(0., 0., qref, lambdaref), 
                   bounds=((-vmax,vmax),(-dvmax,dvmax),
                           (0.1, 1.),(0., 10000.)))
    return res.fun

# Get N_eff ^ 2 with minimization over time shift, phase shift, and component masses
def get_neff2_logf_min_over_tphim1m2(freqs, phases, m1ref, m2ref, *psd_interp):
    from scipy.optimize import minimize
    def loss_func(x):
        dphi, dt, m1, m2 = x
        phases_shifted = phases + 2. * np.pi * freqs * dt - dphi
        phases_shifted += get_0to1p5pn_phases(freqs, m1ref, m2ref) - get_0to1p5pn_phases(freqs, m1, m2)
        return get_neff2_logf_nomin_over(freqs, phases_shifted, *psd_interp)
    vals = phases
    dvals = (vals[2:]-vals[:-2]) / (freqs[1:-1] * (np.log(freqs[2:])-np.log(freqs[:-2]))) / 2. / np.pi
    vmax = np.max(np.abs(vals))
    dvmax = np.max(np.abs(dvals))
    res = minimize(loss_func, x0=(0.,0.,m1ref,m2ref), 
                   bounds=((-vmax,vmax),(-dvmax,dvmax),
                           (0.1*m1ref,10.*m1ref),(0.1*m2ref,10.*m2ref)))
    print("min_over_tphim1m2: res for m1={}, m2={} is {}".format(m1ref, m2ref, res))
    print("res.m1: {}".format(res.m1))
    print("res.m2: {}".format(res.m2))
    return res.fun

# Parent function to call the appropriate N_eff ^ 2 calculation
# TODO: We could minimize over lambda1 lambda2 as well
def get_neff2_logf(freqs, phases, *psd_interp, 
                   min_over_tphi=False, 
                   min_over_m1m2=False, 
                   min_over_q=False,
                   min_over_mc=False,
                   min_over_lambda=False,
                   m1ref=None, m2ref=None, chi1ref=None, chi2ref=None, lambdaref=None):
    if (not min_over_tphi) and (not min_over_m1m2) and (not min_over_q) and (not min_over_mc) and (not min_over_lambda):
        return get_neff2_logf_nomin_over(freqs, phases, *psd_interp)
    elif min_over_tphi and (not min_over_m1m2) and (not min_over_q) and (not min_over_mc) and (not min_over_lambda):
        return get_neff2_logf_min_over_tphi(freqs, phases, *psd_interp)
    elif min_over_tphi and min_over_q and (not min_over_m1m2) and (not min_over_mc):
        return get_neff2_logf_min_over_tphiq(freqs, phases, m1ref, m2ref, *psd_interp)
    elif min_over_tphi and min_over_q and min_over_mc:
        return get_neff2_logf_min_over_tphiqmc(freqs, phases, m1ref, m2ref, *psd_interp)
    elif min_over_tphi and min_over_lambda:
        if lambdaref is None:
            raise ValueError("lambdaref must be provided when min_over_lambda is True")
        if min_over_q and not min_over_mc:
            return get_neff2_logf_min_over_tphiqlambda(freqs, phases, m1ref, m2ref, chi1ref, chi2ref, lambdaref, *psd_interp)
        # TODO: Implement minimization over tphi, q, chirp mass, and lambda
        #elif min_over_q and min_over_mc:
        #    return get_neff2_logf_min_over_tphiqmclambda(freqs, phases, m1ref, m2ref, chi1ref, chi2ref, lambdaref, *psd_interp)
    elif min_over_tphi and min_over_m1m2:
        return get_neff2_logf_min_over_tphim1m2(freqs, phases, m1ref, m2ref, *psd_interp)
    else:
        raise ValueError("Invalid combination of minimization options: min_over_tphi={}, min_over_m1m2={}, min_over_q={}, min_over_mc={}, min_over_lambda={}".format(
            min_over_tphi, min_over_m1m2, min_over_q, min_over_mc, min_over_lambda))

# Get the rescaling factor for the phases
# This is equal to (N_eff,0PN)^2 / ((N_eff,phase_diff)^2)
def get_rescale_fac_logf(freqs, phases, m1, m2, chi1, chi2, ltilde, *psd_interp, 
                         min_over_tphi=False, min_over_m1m2=False, min_over_q=False, min_over_mc=False, min_over_lambda=False, 
                         use_1p5pn_ref_phase=False, use_2p5pn_ref_phase=False):
    # Get N_eff^2 for reference phase (default to just 0PN phase)
    if use_1p5pn_ref_phase:
        ref_phases = get_0to1p5pn_phases(freqs, m1, m2, chi1, chi2)
    elif use_2p5pn_ref_phase:
        ref_phases = get_0to2p5pn_phases(freqs, m1, m2, chi1, chi2)
    else:
        ref_phases = get_0pn_phases(freqs, m1, m2)
    neff20 = get_neff2_logf(freqs, ref_phases, *psd_interp, 
                            min_over_tphi=min_over_tphi, 
                            min_over_m1m2=min_over_m1m2, 
                            min_over_q=min_over_q,
                            min_over_mc=min_over_mc,
                            min_over_lambda=min_over_lambda,
                            m1ref=m1, m2ref=m2, chi1ref=chi1, chi2ref=chi2, lambdaref=ltilde)
    # Get N_eff^2 for phase shift
    neff2 = get_neff2_logf(freqs, phases, *psd_interp, 
                           min_over_tphi=min_over_tphi,
                           min_over_m1m2=min_over_m1m2, 
                           min_over_q=min_over_q,
                           min_over_mc=min_over_mc,
                           min_over_lambda=min_over_lambda,
                           m1ref=m1, m2ref=m2, chi1ref=chi1, chi2ref=chi2, lambdaref=ltilde)
    if min_over_m1m2 or min_over_q or min_over_mc or min_over_lambda:
        # If we are minimizing over mass parameters, we need to symmetrize the phase shifts
        neff2n = get_neff2_logf(freqs, -phases, *psd_interp, 
                                min_over_tphi=min_over_tphi,
                                min_over_m1m2=min_over_m1m2, 
                                min_over_q=min_over_q,
                                min_over_mc=min_over_mc,
                                min_over_lambda=min_over_lambda,
                                m1ref=m1, m2ref=m2, chi1ref=chi1, chi2ref=chi2, lambdaref=ltilde)
        neff2 = np.sqrt(neff2 * neff2n)
    return np.sqrt(neff20 / neff2)


args = get_cli()
if not args.asd:
    psd = [bilby.gw.detector.psd.PowerSpectralDensity(psd_file=fpath) for fpath in args.psd_path]
else:
    psd = [bilby.gw.detector.psd.PowerSpectralDensity(asd_file=fpath) for fpath in args.psd_path]
psd_interp = [p.power_spectral_density_interpolated for p in psd]
def worker(dfrow):
    i, row = dfrow
    m1, m2 = row['m1'], row['m2']
    chi1, chi2 = row['s1z'], row['s2z']
    l1 = row['l1'] if args.use_tidal_data or args.use_tidal_data_full else 0.
    l2 = row['l2'] if args.use_tidal_data or args.use_tidal_data_full else 0.
    ltilde = (16./13.) * ((m1 + 12.*m2)*m1**4*l1 + (m2 + 12.*m1)*m2**4*l2) / (m1 + m2)**5
    freqs = row['freqs'] / (m1+m2) / lal.MTSUN_SI
    phases = row['phases']
    return get_rescale_fac_logf(freqs, phases, m1, m2, chi1, chi2, ltilde, *psd_interp, 
                                min_over_tphi=args.min_over_tphi,
                                min_over_m1m2=args.min_over_m1m2,
                                min_over_q=args.min_over_q,
                                min_over_mc=args.min_over_mc,
                                min_over_lambda=args.min_over_lambda,
                                use_1p5pn_ref_phase=args.use_1p5pn_ref_phase,
                                use_2p5pn_ref_phase=args.use_2p5pn_ref_phase)

if __name__ == '__main__':

    log_str = "{} Rescaling dataset {} with noise curve(s) {}, as {}".format(
        datetime.now().strftime('%H:%M:%S'), args.dataset_path, args.psd_path, "PSD(s)" if not args.asd else "ASD(s)")
    log_str += ", min_over_tphi={}, min_over_m1m2={}, min_over_q={}, min_over_mc={}, min_over_lambda={}...".format(
        args.min_over_tphi, args.min_over_m1m2, args.min_over_q, args.min_over_mc, args.min_over_lambda)
    print(log_str); print(); sys.stdout.flush()
    df = pd.read_pickle(args.dataset_path)

    log_str = "{} Using a pool of {} workers...".format(
        datetime.now().strftime('%H:%M:%S'), args.pool)
    print(log_str); print(); sys.stdout.flush()
    with multiprocessing.Pool(args.pool) as p:
        z = p.map(worker, tqdm(df.iterrows(), total=len(df)))
    z = np.asarray(z)

    dataset_filename = os.path.basename(args.dataset_path)
    output_filename = dataset_filename.rpartition('.')[0] + "-phase-rescale-fac.npy"
    output_filepath = os.path.join(args.output_rootdir, output_filename)
    if not os.path.exists(args.output_rootdir):
        os.makedirs(args.output_rootdir, exist_ok=True)
    np.save(output_filepath, z)

    log_str = "{} Result saved to {}...".format(
        datetime.now().strftime('%H:%M:%S'), output_filepath)
    print(log_str); print(); sys.stdout.flush()

