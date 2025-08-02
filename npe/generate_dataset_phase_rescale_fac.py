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


def get_cli():
    parser = argparse.ArgumentParser(
        "Create an npy file with phase rescaling factors for the original dataset")

    parser.add_argument("--psd-path", type=str, nargs='+',
                        help="Path(s) to the PSD estimate.")
    parser.add_argument("--asd", action="store_true", default=False,
                        help="Whether the noise curve is given in ASD.")
    parser.add_argument("--dataset-path", type=str,
                        help="Path to the original dataset.")
    parser.add_argument("--min-over-tphi", action="store_true", default=False)
    parser.add_argument("--min-over-m1m2", action="store_true", default=False)
    parser.add_argument("--pool", default=1, type=int, help="Pool size.")
    # output
    parser.add_argument("--output-rootdir", required=True,
                        help="Output directory.")

    args = parser.parse_args()
    return args


def get_0pn_phases(freqs, m1, m2):
    mc = bilby.gw.conversion.component_masses_to_chirp_mass(m1, m2)
    return 3./128. * (np.pi*mc*freqs*lal.MTSUN_SI)**(-5./3.)

# TODO: Verify origin/validity of these values for BNS case
def get_0to1pn_phases(freqs, m1, m2):
    mtot = m1 + m2
    eta = m1 * m2 / mtot / mtot
    v3 = np.pi * mtot * freqs * lal.MTSUN_SI
    v5 = np.power(v3, 5./3.)
    cbase = 3. / 128. / eta
    c0 = 1.
    c2 = 3715. / 756. + 55. / 9. * eta
    phases = cbase * (c0 / v5 + c2 / v3)
    return phases

# Get N_eff ^ 2 without minimization
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

# Get N_eff ^ 2 with minimization over time shift, phase shift, and component masses
def get_neff2_logf_min_over_tphim1m2(freqs, phases, m1ref, m2ref, *psd_interp):
    from scipy.optimize import minimize
    def loss_func(x):
        dphi, dt, m1, m2 = x
        phases_shifted = phases + 2. * np.pi * freqs * dt - dphi
        phases_shifted += get_0to1pn_phases(freqs, m1ref, m2ref) - get_0to1pn_phases(freqs, m1, m2)
        return get_neff2_logf_nomin_over(freqs, phases_shifted, *psd_interp)
    vals = phases
    dvals = (vals[2:]-vals[:-2]) / (freqs[1:-1] * (np.log(freqs[2:])-np.log(freqs[:-2]))) / 2. / np.pi
    vmax = np.max(np.abs(vals))
    dvmax = np.max(np.abs(dvals))
    res = minimize(loss_func, x0=(0.,0.,m1ref,m2ref), 
                   bounds=((-vmax,vmax),(-dvmax,dvmax),
                           (0.1*m1ref,10.*m1ref),(0.1*m2ref,10.*m2ref)))
    return res.fun

# Parent function to call the appropriate N_eff ^ 2 calculation
def get_neff2_logf(freqs, phases, *psd_interp, 
                   min_over_tphi=False, 
                   min_over_m1m2=False, m1ref=None, m2ref=None):
    if (not min_over_tphi) and (not min_over_m1m2):
        return get_neff2_logf_nomin_over(freqs, phases, *psd_interp)
    elif min_over_tphi and (not min_over_m1m2):
        return get_neff2_logf_min_over_tphi(freqs, phases, *psd_interp)
    elif min_over_tphi and min_over_m1m2:
        return get_neff2_logf_min_over_tphim1m2(freqs, phases, m1ref, m2ref, *psd_interp)
    else:
        raise ValueError("min_over_tphi={} while min_over_m1m2={}".format(min_over_tphi, min_over_tphi))

# Get the rescaling factor for the phases
# This is equal to (N_eff,0PN)^2 / ((N_eff,phase_diff)^2)
def get_rescale_fac_logf(freqs, phases, m1, m2, *psd_interp, min_over_tphi=False, min_over_m1m2=False):
    # Get N_eff^2 for 0PN phase
    neff20 = get_neff2_logf(freqs, get_0pn_phases(freqs, m1, m2), *psd_interp, 
                            min_over_tphi=min_over_tphi, 
                            min_over_m1m2=min_over_m1m2, m1ref=m1, m2ref=m2)
    # Get N_eff^2 for phase shift
    neff2 = get_neff2_logf(freqs, phases, *psd_interp, 
                           min_over_tphi=min_over_tphi,
                           min_over_m1m2=min_over_m1m2, m1ref=m1, m2ref=m2)
    if min_over_m1m2:
        # If we are minimizing over m1 and m2, we need to symmetrize the phase shifts
        neff2n = get_neff2_logf(freqs, -phases, *psd_interp, 
                                min_over_tphi=min_over_tphi,
                                min_over_m1m2=min_over_m1m2, m1ref=m1, m2ref=m2)
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
    freqs = row['freqs'] / (m1+m2) / lal.MTSUN_SI
    phases = row['phases']
    return get_rescale_fac_logf(freqs, phases, m1, m2, *psd_interp, 
                                min_over_tphi=args.min_over_tphi,
                                min_over_m1m2=args.min_over_m1m2)

if __name__ == '__main__':

    log_str = "{} Rescaling dataset {} with noise curve(s) {}, as {}".format(
        datetime.now().strftime('%H:%M:%S'), args.dataset_path, args.psd_path, "PSD(s)" if not args.asd else "ASD(s)")
    log_str += ", min_over_tphi={}, min_over_m1m2={}...".format(
        args.min_over_tphi, args.min_over_m1m2)
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

