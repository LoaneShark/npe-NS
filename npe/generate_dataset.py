import argparse
import multiprocessing
import os

import numpy as np
import pandas as pd
from tqdm import tqdm

from astropy import units as u
from astropy import constants as c
import lal, lalsimulation


def get_cli():
    parser = argparse.ArgumentParser(
        "Create a pickle file with GR/non-GR IMRPhenomD waveform phasing")

    # point-mass source parameters
    parser.add_argument("--m1-min", default=10.0, type=float,
                        help="Minimum value of m1 in solmass.")
    parser.add_argument("--m2-min", default=10.0, type=float,
                        help="Minimum value of m2 in solmass.")
    parser.add_argument("--m1-max", default=80.0, type=float,
                        help="Maximum value of m1 in solmass.")
    parser.add_argument("--m2-max", default=80.0, type=float,
                        help="Maximum value of m2 in solmass.")
    parser.add_argument("--chi1z-min", default=0.0, type=float,
                        help="Minimum value of dimensionless aligned primary spin.")
    parser.add_argument("--chi1z-max", default=0.0, type=float,
                        help="Maximum value of dimensionless aligned primary spin.")
    parser.add_argument("--chi2z-min", default=0.0, type=float,
                        help="Minimum value of dimensionless aligned secondary spin.")
    parser.add_argument("--chi2z-max", default=0.0, type=float,
                        help="Maximum value of dimensionless aligned secondary spin.")
    
    # tidal source parameters
    parser.add_argument("--l1-min", default=0.0, type=float,
                        help="Minimum value of dimensionless Lambda_1.")
    parser.add_argument("--l2-min", default=0.0, type=float,
                        help="Minimum value of dimensionless Lambda_2.")
    parser.add_argument("--l1-max", default=0.0, type=float,
                        help="Maximum value of dimensionless Lambda_1.")
    parser.add_argument("--l2-max", default=0.0, type=float,
                        help="Maximum value of dimensionless Lambda_2.")
    parser.add_argument("--cq1-min", default=0.0, type=float,
                        help="Minimum value of quadrupolar spin deformability CQ_1.")
    parser.add_argument("--cq2-min", default=0.0, type=float,
                        help="Minimum value of quadrupolar spin deformability CQ_2.")
    parser.add_argument("--cq1-max", default=0.0, type=float,
                        help="Maximum value of quadrupolar spin deformability CQ_1.")
    parser.add_argument("--cq2-max", default=0.0, type=float,
                        help="Maximum value of quadrupolar spin deformability CQ_2.")

    # modified gr parameters
    parser.add_argument("--b-ppe", type=int, 
                        help="Leading ppE index.")
    parser.add_argument("--n-ppe", type=int, 
                        help="Length of the ppE series.")
    parser.add_argument("--ppe-ref-min", default=0., type=float,
                        help="Lower frequency bound considered for the ppE expansion."
                             "If non-positive, use fmin.")
    parser.add_argument("--ppe-ref-max", default=0., type=float,
                        help="Upper frequency bound considered for the ppE expansion."
                             "If non-positive, use fmax.")
    parser.add_argument("--ppe-ref-min-in-geometric-units", action='store_true', default=False)
    parser.add_argument("--ppe-ref-max-in-geometric-units", action='store_true', default=False)

    # phasing representation
    parser.add_argument("--labels-only", action='store_true', default=False,
                        help="Output only the labels, and no freqs or phasing values.")
    parser.add_argument("--fmin", default=10.0, type=float,
                        help="Minimum frequency value.")
    parser.add_argument("--fmax", default=500.0, type=float,
                        help="Maximum frequency value.")
    parser.add_argument("--num-freqs", default=500, type=int,
                        help="Number of frequency points.")
    parser.add_argument("--logspace-freqs", action='store_true', default=False, 
                        help="Supply for log-spaced freqs.")
    parser.add_argument("--freq-in-geometric-units", action='store_true', default=False, 
                        help="Frequency is in natural units.")
    parser.add_argument("--minus-gr", action='store_true', default=False,
                        help="Output only correction to GR.")

    # sampling specs
    parser.add_argument("--num-samples", default=100, type=int,
                        help="Number of samples.")
    parser.add_argument("--seed", default=1, type=int,
                        help="Random seed.")
    parser.add_argument("--pool", default=1, type=int, help="Pool size.")

    # output
    parser.add_argument("-o", "--output-file", required=True,
                        help="Name of the output file. Stored as pickle.")

    args = parser.parse_args()
    return args


def _get_masses_and_spins(
        m1_min, m2_min, m1_max, m2_max,
        chi1z_min, chi2z_min, chi1z_max, chi2z_max,
        num_samples):
    """Get uniformly sampled masses and aligned spins"""
    m1 = np.random.uniform(m1_min, m1_max, num_samples)
    m2 = np.random.uniform(m2_min, m2_max, num_samples)
    chi1z = np.random.uniform(chi1z_min, chi1z_max, num_samples)
    chi2z = np.random.uniform(chi2z_min, chi2z_max, num_samples)
    return m1, m2, chi1z, chi2z

def _get_deformation_terms(
        L1_min, L2_min, L1_max, L2_max,
        CQ1_min, CQ2_min, CQ1_max, CQ2_max,
        num_samples):
    """Get uniformly sampled masses and aligned spins"""
    L1 = np.random.uniform(L1_min, L1_max, num_samples)
    L2 = np.random.uniform(L2_min, L2_max, num_samples)
    CQ1 = np.random.uniform(CQ1_min, CQ1_max, num_samples)
    CQ2 = np.random.uniform(CQ2_min, CQ2_max, num_samples)
    return L1, L2, CQ1, CQ2

def _get_pn_coeffs(m1, m2, chi1z, chi2z, l1, l2, cq1, cq2):
    num_samples = len(m1)
    param_vecs = [lal.CreateREAL8Vector(num_samples) for _ in range(8)]
    param_vecs[0].data = m1
    param_vecs[1].data = m2
    param_vecs[2].data = chi1z
    param_vecs[3].data = chi2z
    # tidal deformation and spin deformation
    # TODO: What units/ranges are expected for self-spin quadrupole deformabilities?
    param_vecs[4].data = l1
    param_vecs[5].data = l2
    param_vecs[6].data = cq1
    param_vecs[7].data = cq2

    '''
    for i in range(4, 8):
        # tidal deformation and spin deformation
        param_vecs[i].data = np.zeros_like(param_vecs[i].data)
    '''

    coeffs = lalsimulation.SimInspiralTaylorF2AlignedPhasingArray(*param_vecs).data
    coeffs_v, coeffs_vlogv, coeffs_vlogvsq = coeffs.reshape(3,-1,num_samples)
    return coeffs_v.T, coeffs_vlogv.T, coeffs_vlogvsq.T

# TODO: does this need to be mofied further for BNS case?
def _get_coeff_bound(b, coeffs_v, v_min, v_max):
    bound = np.zeros_like(b, dtype=float)
    mask_fbd = (b == 0) | (b >= 3)
    mask_pos = (b >= -3) & (~mask_fbd)
    mask_neg = (b <= -5) & (~mask_fbd)
    mask_miss = ~(mask_fbd|mask_pos|mask_neg)
    bound[mask_pos] = np.abs(coeffs_v[mask_pos,b[mask_pos]+5])
    bound[mask_neg] = np.abs(coeffs_v[mask_neg,0]) * v_min[mask_neg] ** (-5-b[mask_neg])
    bound_miss_pos = np.abs(coeffs_v[mask_miss,2]) * v_min[mask_miss]
    bound_miss_neg = np.abs(coeffs_v[mask_miss,0]) / v_max[mask_miss]
    bound[mask_miss] = np.min([bound_miss_pos, bound_miss_neg], axis=0)
    return bound


def _get_gamma_bar_bound(b, coeffs_v, v_min, v_max):
    return _get_coeff_bound(b, coeffs_v, v_min, v_max)


def _get_dpsi_bar_bound(k, coeffs_v, v_min, v_max, gamma_bar_bound):
    mask = gamma_bar_bound == 0.
    dpsi_bar_bound = np.zeros_like(k, dtype=float)
    delta_bar_bound = _get_coeff_bound(k[~mask], coeffs_v[~mask], v_min[~mask], v_max[~mask])
    dpsi_bar_bound[~mask] = delta_bar_bound / gamma_bar_bound[~mask]
    return dpsi_bar_bound


def _convert_f_to_fgeom(f, mtot):
    fgeom = f * u.Hz * mtot * u.solMass
    fgeom = fgeom * c.G / c.c**3
    fgeom = fgeom.to('').value
    return fgeom


def _generate_meta_data_chunks(
        m1_min, m2_min, m1_max, m2_max,
        chi1z_min, chi2z_min, chi1z_max, chi2z_max,
        l1_min, l2_min, l1_max, l2_max,
        cq1_min, cq2_min, cq1_max, cq2_max,
        b_ppe, n_ppe, 
        ppe_ref_min, ppe_ref_min_in_geometric_units,
        ppe_ref_max, ppe_ref_max_in_geometric_units,
        num_samples, seed, num_chunks):
    """Get meta data in chunks"""
    np.random.seed(seed)
    m1, m2, chi1z, chi2z = _get_masses_and_spins(
        m1_min, m2_min, m1_max, m2_max,
        chi1z_min, chi2z_min, chi1z_max, chi2z_max,
        num_samples
    )
    l1, l2, cq1, cq2 = _get_deformation_terms(
        l1_min, l2_min, l1_max, l2_max,
        cq1_min, cq2_min, cq1_max, cq2_max,
        num_samples
    )

    b = np.repeat(b_ppe, num_samples)
    ref_min = np.repeat(ppe_ref_min, num_samples)
    ref_max = np.repeat(ppe_ref_max, num_samples)
    if not ppe_ref_min_in_geometric_units:
        ref_min = _convert_f_to_fgeom(ref_min, m1 + m2)
    if not ppe_ref_max_in_geometric_units:
        ref_max = _convert_f_to_fgeom(ref_max, m1 + m2)
    v_min = (np.pi * ref_min) ** (1/3)
    v_max = (np.pi * ref_max) ** (1/3)

    pn_coeffs_v, _, _ = _get_pn_coeffs(m1, m2, chi1z, chi2z, l1, l2, cq1, cq2)
    #print('pn_coeffs_v: ', pn_coeffs_v.shape)
    #print(pn_coeffs_v)
    gamma_bar_bound = _get_gamma_bar_bound(b, pn_coeffs_v, v_min, v_max)
    dpsi_bar_bounds = [_get_dpsi_bar_bound(b+i, pn_coeffs_v, v_min, v_max, gamma_bar_bound) \
                        for i in range(2, n_ppe)]
    dpsi_bar_bounds = np.asarray(dpsi_bar_bounds).reshape(-1,num_samples).T
    ppe_bounds = np.concatenate([gamma_bar_bound[:, None], dpsi_bar_bounds], axis=-1)

    gamma_bar = gamma_bar_bound * (-1. + 2. * np.random.randint(0, 2, num_samples))
    dpsi_bars = np.random.uniform(-dpsi_bar_bounds, dpsi_bar_bounds)
    ppe_coeffs = np.concatenate([gamma_bar[:, None], dpsi_bars], axis=-1)

    labels = np.vstack([m1, m2, chi1z, chi2z, l1, l2, b]).T
    labels = np.concatenate([labels, ppe_bounds, ppe_coeffs], axis=-1)
    return np.array_split(labels, num_chunks)


def _populate_chunk(metadata_array, 
                    fmin=10., fmax=1000., 
                    num_freqs=1000, log_spacing=False,
                    freq_in_natural_units=False,
                    minus_gr=False, labels_only=False):
    """Generate and populate the phasing
    
    Parameters
    ----------
    metadata_array : array_like
        array containing masses, aligned spins, length-scale
    fmin : float
        minimum frequency for phasing array
    fmax : float
        maximum frequency for phasing array
    num_freqs : int
        number of frequency points
    log_spacing : bool
        log/linear spacing of frequency points
    freq_in_natural_units : bool
        frequency is in geometric/SI units i.e. Hz
    insert_func : callable
        lalsimulation function to insert modified GR parameters
        into a LALDict. Required parameter.
    beta_parameter_func : callable
        Function in :module:`modified_gr_utils` that returns the
        beta parameter corresponding to the modified theory.
        Required parameter.
    """
    # FIXME
    assert minus_gr

    n_ppe = (metadata_array.shape[-1] - 9) // 2 + 2
    ppe_keys = [f'dpsi_bar_{i}' for i in range(2, n_ppe)]
    ppe_bound_keys = [k+'_bound' for k in ppe_keys]
    r = pd.DataFrame(
        data=metadata_array,
        columns=['m1', 'm2', 's1z', 's2z', 'L1', 'L2', 'b_ppe'] \
                + ['gamma_bar_bound'] + ppe_bound_keys \
                + ['gamma_bar'] + ppe_keys)
    if labels_only:
        return r

    mtot = np.sum(metadata_array[:,:2], axis=-1, keepdims=True)
    b = metadata_array[:,[6]]
    k = b + np.arange(n_ppe)[None,:]
    gamma_bar = metadata_array[:,[6+n_ppe]]
    dpsi_bars = metadata_array[:,7+n_ppe:]
    delta_bars = np.concatenate([gamma_bar,
                                 np.zeros_like(gamma_bar),
                                 gamma_bar * dpsi_bars], axis=-1)

    if not log_spacing:
        freqs = np.linspace(fmin, fmax, num_freqs)
    else:
        freqs = np.logspace(np.log10(fmin), np.log10(fmax), num_freqs)
    freqs = np.tile(freqs, (metadata_array.shape[0], 1))
    if not freq_in_natural_units:
        freqs = _convert_f_to_fgeom(freqs, mtot)
    v = (np.pi * freqs) ** (1/3)
    phases = delta_bars[:,None,:] * v[:,:,None] ** k[:,None,:]
    phases = np.sum(phases, axis=-1)
    r['freqs'] = list(freqs)
    r['phases'] = list(phases)
    return r


def main():
    args = get_cli()
    if args.ppe_ref_min <= 0:
        args.ppe_ref_min = args.fmin
        args.ppe_ref_min_in_geometric_units = args.freq_in_geometric_units
    if args.ppe_ref_max <= 0:
        args.ppe_ref_max = args.fmax
        args.ppe_ref_max_in_geometric_units = args.freq_in_geometric_units
    chunks = _generate_meta_data_chunks(
        args.m1_min, args.m2_min,
        args.m1_max, args.m2_max,
        args.chi1z_min, args.chi2z_min,
        args.chi1z_max, args.chi2z_max,
        args.l1_min, args.l2_min, 
        args.l1_max, args.l2_max,
        args.cq1_min, args.cq2_min, 
        args.cq1_max, args.cq2_max,
        args.b_ppe, args.n_ppe,
        args.ppe_ref_min, args.ppe_ref_min_in_geometric_units,
        args.ppe_ref_max, args.ppe_ref_max_in_geometric_units,
        args.num_samples, args.seed,
        args.pool
    )
    
    with multiprocessing.Pool(args.pool) as p:
        rs = [
            p.apply_async(
                _populate_chunk,
                args=(chunk,),
                kwds=dict(
                    fmin=args.fmin,
                    fmax=args.fmax,
                    num_freqs=args.num_freqs,
                    log_spacing=args.logspace_freqs,
                    freq_in_natural_units=args.freq_in_geometric_units,
                    minus_gr=args.minus_gr,
                    labels_only=args.labels_only,
                )
            ) for chunk in chunks
        ]
        result = [r.get() for r in tqdm(rs)]

    output_directory = os.path.dirname(args.output_file)
    if not os.path.exists(output_directory):
        os.makedirs(output_directory)
    pd.concat(result).to_pickle(
        args.output_file,
        protocol=3  # compatible with python < 3.8
    )


if __name__ == '__main__':
    main()
