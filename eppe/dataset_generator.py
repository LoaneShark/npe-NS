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

    coupling_constant_group = parser.add_mutually_exclusive_group()
    coupling_constant_group.add_argument(
        "--edgb-max-length-scale", help="Upper limit of EDGB length scale in km",
        type=float, default=0.
    )
    coupling_constant_group.add_argument(
        "--dcs-max-length-scale", help="Upper limit of dCS length scale in km",
        type=float, default=0.
    )
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
        max_length_scale, num_samples, seed, num_chunks):
    """Get meta data in chunks"""
    np.random.seed(seed)
    m1, m2, chi1z, chi2z = _get_masses_and_spins(
        m1_min, m2_min, m1_max, m2_max,
        chi1z_min, chi2z_min, chi1z_max, chi2z_max,
        num_samples
    )
    length_scale = np.zeros(num_samples) if max_length_scale == 0. \
        else np.random.uniform(0., max_length_scale, num_samples)

    return np.array_split(
        np.vstack((m1, m2, chi1z, chi2z, length_scale)).T, num_chunks)


def _populate_chunk(metadata_array, fmin=10., fmax=1000.,
                    num_freqs=1000, log_spacing=False,
                    freq_in_natural_units=False, insert_func=None,
                    beta_parameter_func=None):
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
    freqs = []
    phases = []
    for m1, m2, s1z, s2z, length_scale in metadata_array:
        # create and insert Non-GR ppE coefficient in LAL Dict
        params = lal.CreateDict()
        if length_scale > 0.:
            assert beta_parameter_func is not None
            assert insert_func is not None  # sanity check
            # get beta in Nico's notation
            beta_ppe = getattr(modified_gr_utils, beta_parameter_func)(
                m1 * u.solMass, m2 * u.solMass,  # units needed
                s1z, s2z, length_scale * u.km
            )
            # multiply lalsim prefactors
            eta = modified_gr_utils._symmetric_mass_ratio(m1, m2)
            pn_order = modified_gr_utils.PN_ORDER_ASSOC[beta_parameter_func]
            lalsim_beta_ppe = modified_gr_utils.non_gr_parameter_lalsimulation(
                beta_ppe, eta, pn_order
            )
            # insert into LAL Dict
            getattr(lalsimulation, insert_func)(params, lalsim_beta_ppe)

        freq, phase = lal_utils.get_phenomD_phasing(
            m1, m2, s1z, s2z, fmin, fmax,
            num_freqs, params, log_spacing,
            freq_in_natural_units
        )
        freqs.append(freq)
        phases.append(phase)
    
    r = pd.DataFrame(
        data=metadata_array,
        columns=('m1', 'm2', 's1z', 's2z', 'length_scale'))
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
        args.edgb_max_length_scale or args.dcs_max_length_scale,
        args.num_samples, args.seed,
        args.pool
    )

    beta_parameter_setter = 'non_gr_beta_edbg' if args.edgb_max_length_scale \
        else 'non_gr_beta_dcs' if args.dcs_max_length_scale else None
    
    laldict_setter = 'SimInspiralWaveformParamsInsertNonGRBetaPPEMinus2' if \
        args.edgb_max_length_scale else 'SimInspiralWaveformParamsInsertNonGRBetaPPE4' \
            if args.dcs_max_length_scale else None
    
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
                    insert_func=laldict_setter,
                    beta_parameter_func=beta_parameter_setter
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
