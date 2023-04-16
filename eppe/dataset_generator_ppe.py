import argparse
import multiprocessing

import numpy as np
import pandas as pd
from tqdm import tqdm

from astropy import units as u
import lal, lalsimulation

from eppe.utils import modified_gr_utils, lal_utils


def get_cli():
    parser = argparse.ArgumentParser(
        "Create a pickle file with GR/non-GR IMRPhenomD waveform phasing")
    parser.add_argument("--m1-min", default=10.0, type=float,
                        help="Minimum value of m1 in solmass")
    parser.add_argument("--m2-min", default=10.0, type=float,
                        help="Minimum value of m2 in solmass")
    parser.add_argument("--m1-max", default=80.0, type=float,
                        help="Maximum value of m1 in solmass")
    parser.add_argument("--m2-max", default=80.0, type=float,
                        help="Maximum value of m2 in solmass")
    parser.add_argument("--fmin", default=10.0, type=float,
                        help="Minimum frequency value")
    parser.add_argument("--fmax", default=500.0, type=float,
                        help="Maximum frequency value")
    parser.add_argument("--freq-in-geometric-units", action='store_true',
                        default=False, help="Frequency is in natural units.")
    parser.add_argument("--num-freqs", default=500, type=int,
                        help="Number of frequency points")
    parser.add_argument("--logspace-freqs", action='store_true',
                        default=False, help="Supply for log-spaced freqs")
    parser.add_argument(
        "--chi1z-min", default=0.0, type=float,
        help="Minimum value of dimensionless aligned primary spin."
    )
    parser.add_argument(
        "--chi1z-max", default=0.0, type=float,
        help="Maximum value of dimensionless aligned primary spin."
    )
    parser.add_argument(
        "--chi2z-min", default=0.0, type=float,
        help="Minimum value of dimensionless aligned secondary spin."
    )
    parser.add_argument(
        "--chi2z-max", default=0.0, type=float,
        help="Maximum value of dimensionless aligned secondary spin."
    )
    parser.add_argument("--num-samples", default=100, type=int,
                        help="Number of samples")
    parser.add_argument("--seed", default=1, type=int,
                        help="Random seed")
    parser.add_argument("--pool", default=1, type=int, help="Pool size")
    parser.add_argument("-o", "--output-file", required=True,
                        help="Name of the output file. Stored as pickle.")

    parser.add_argument("--minus-gr", action='store_true', default=False,
                        help="Output only deviation from GR")
    parser.add_argument("--b-ppe", default=0, type=int, 
                        help="Leading ppE index")
    parser.add_argument("--beta-ppe-abs-max", default=0.0, type=float, 
                        help="Maximum absolute value of the leading ppE beta, ignored if given --ppe-saturate-perturbation")
    parser.add_argument("--ppe-saturate-perturbation", action='store_true', default=False,
                        help="Whether to set leading ppE beta by saturating the purturbative condition. "
                             "If b >= -5, this sets max abs beta s.t. ppE correction has the same magnitude as the GR phasing at the same PN order. "
                             "If b < -5, this sets max abs beta s.t. ppE correction has the same magnitude as the GR phasing at the leading order, "
                             "evaluated at the frequency given by ppe-perturbation-v.")
    parser.add_argument("--ppe-perturbation-v", default=0.1, type=float,
                        help="Frequency at which ppE correction must be smaller than the GR leading phase "
                             "in order to be considered as perturbation when b < -5, given in terms of v/c.")

    parser.add_argument("--labels-only", action='store_true', default=False,
                        help="Output only the labels, and no freqs or phasing values")
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


def _generate_meta_data_chunks(
        m1_min, m2_min, m1_max, m2_max,
        chi1z_min, chi2z_min, chi1z_max, chi2z_max,
        b_ppe, beta_ppe_abs_max, 
        ppe_saturate_perturbation, 
        ppe_perturbation_v,
        num_samples, seed, num_chunks):
    """Get meta data in chunks"""
    np.random.seed(seed)
    m1, m2, chi1z, chi2z = _get_masses_and_spins(
        m1_min, m2_min, m1_max, m2_max,
        chi1z_min, chi2z_min, chi1z_max, chi2z_max,
        num_samples
    )
    if ppe_saturate_perturbation:
        eta = modified_gr_utils._symmetric_mass_ratio(m1, m2)
        if b_ppe <= -4: # GR coeff at 0.5PN is 0, merge with the b <= -5 branch
            pn_coeff = 3. / 128. / eta
            beta_ppe_abs_max = pn_coeff / eta ** (b_ppe/5.)
            beta_ppe_abs_max /= ppe_perturbation_v ** (5 + b_ppe)
            beta_ppe_abs_max /= -b_ppe / 5 # compare to GR by dphi/df, not phi
        else:
            param_vecs = [lal.CreateREAL8Vector(num_samples) for _ in range(8)]
            param_vecs[0].data = m1
            param_vecs[1].data = m2
            param_vecs[2].data = chi1z
            param_vecs[3].data = chi2z
            for i in range(4, 8): # tidal deformation and spin deformation
                param_vecs[i].data = np.zeros_like(param_vecs[i].data)
            pn_coeff = lalsimulation.SimInspiralTaylorF2AlignedPhasingArray(*param_vecs).data
            pn_coeff = pn_coeff[:len(pn_coeff)//3] # remove coeffs for vlogv and vlogvsq
            pn_coeff = pn_coeff.reshape(-1, num_samples)[b_ppe+5] # take only the pn coeff of the ppE order
            beta_ppe_abs_max = np.abs(pn_coeff) / eta ** (b_ppe/5.)
        beta_ppe = np.random.uniform(-beta_ppe_abs_max, beta_ppe_abs_max, num_samples)
    elif beta_ppe_abs_max == 0.0:
        beta_ppe = np.zeros(num_samples)
    else:
        beta_ppe = np.random.uniform(-beta_ppe_abs_max, beta_ppe_abs_max, num_samples)
    b_ppe = np.full_like(beta_ppe, b_ppe)

    return np.array_split(
        np.vstack((m1, m2, chi1z, chi2z, b_ppe, beta_ppe)).T, num_chunks)


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
    r = pd.DataFrame(
        data=metadata_array,
        columns=('m1', 'm2', 's1z', 's2z', 'b_ppe', 'beta_ppe'))

    if not labels_only:
        freqs = []
        phases = []
        for m1, m2, s1z, s2z, b_ppe, beta_ppe in metadata_array:
            # create and insert Non-GR ppE coefficient in LAL Dict
            params = lal.CreateDict()
            b_ppe = int(np.rint(b_ppe))
            twice_pn_order = b_ppe + 5
            eta = modified_gr_utils._symmetric_mass_ratio(m1, m2)
            lalsim_beta_ppe = modified_gr_utils.non_gr_parameter_lalsimulation(beta_ppe, eta, twice_pn_order/2.)
            insert_func = "SimInspiralWaveformParamsInsertNonGRBetaPPE"
            insert_func += f"Minus{-twice_pn_order}" if twice_pn_order < 0 else f"{twice_pn_order}"
            getattr(lalsimulation, insert_func)(params, lalsim_beta_ppe)

            freq, phase = lal_utils.get_phenomD_phasing(
                m1, m2, s1z, s2z, fmin, fmax,
                num_freqs, params, log_spacing,
                freq_in_natural_units
            )
            if minus_gr:
                _, phase_gr = lal_utils.get_phenomD_phasing(
                    m1, m2, s1z, s2z, fmin, fmax,
                    num_freqs, lal.CreateDict(), log_spacing,
                    freq_in_natural_units
                )
                phase -= phase_gr
            freqs.append(freq)
            phases.append(phase)
        
        r['freqs'] = freqs
        r['phases'] = phases

    return r


def main():
    args = get_cli()
    chunks = _generate_meta_data_chunks(
        args.m1_min, args.m2_min,
        args.m1_max, args.m2_max,
        args.chi1z_min, args.chi2z_min,
        args.chi1z_max, args.chi2z_max,
        args.b_ppe, args.beta_ppe_abs_max,
        args.ppe_saturate_perturbation, 
        args.ppe_perturbation_v,
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

    pd.concat(result).to_pickle(
        args.output_file,
        protocol=3  # compatible with python < 3.8
    )


if __name__ == '__main__':
    main()
