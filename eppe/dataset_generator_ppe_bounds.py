import argparse
import multiprocessing

import numpy as np
import pandas as pd
from tqdm import tqdm

from astropy import units as u
from astropy import constants as c
import lal, lalsimulation

from eppe.utils import modified_gr_utils, lal_utils


def get_cli():
    parser = argparse.ArgumentParser(
        "Create a pickle file with GR/non-GR IMRPhenomD waveform phasing")

    # source parameters
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

    # modified gr parameters
    parser.add_argument("--b-ppe", type=float, 
                        help="Leading ppE index.")
    parser.add_argument("--ppe-ref", default=0.1, type=float,
                        help="Frequency at which ppE correction must be smaller than GR. "
                             "The number is given in terms of v/c. "
                             "However, if ppe-ref-in-f is set to True, it is in Hz. "
                             "This does not take any effect if b is an interger and b = -5 or b >= -4.")
    parser.add_argument("--ppe-ref-in-hz", action='store_true', default=False)

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


def _generate_meta_data_chunks(
        m1_min, m2_min, m1_max, m2_max,
        chi1z_min, chi2z_min, chi1z_max, chi2z_max,
        b_ppe, ppe_ref, ppe_ref_in_hz,
        num_samples, seed, num_chunks):
    """Get meta data in chunks"""
    np.random.seed(seed)
    m1, m2, chi1z, chi2z = _get_masses_and_spins(
        m1_min, m2_min, m1_max, m2_max,
        chi1z_min, chi2z_min, chi1z_max, chi2z_max,
        num_samples
    )

    eta = modified_gr_utils._symmetric_mass_ratio(m1, m2)
    if not ppe_ref_in_hz:
        ppe_ref_v = ppe_ref
    else:
        ppe_ref = ppe_ref * u.Hz * (m1 + m2) * u.solMass
        ppe_ref = ppe_ref * c.G / c.c**3
        ppe_ref = ppe_ref.to('').value
        ppe_ref_v = (np.pi * ppe_ref) ** (1/3)

    b_ref = int(np.floor(b_ppe))
    if b_ref <= -4:
        # GR coeff at 0.5PN is 0, merge with the b <= -5 branch
        b_ref = -5

    param_vecs = [lal.CreateREAL8Vector(num_samples) for _ in range(8)]
    param_vecs[0].data = m1
    param_vecs[1].data = m2
    param_vecs[2].data = chi1z
    param_vecs[3].data = chi2z
    for i in range(4, 8):
        # tidal deformation and spin deformation
        param_vecs[i].data = np.zeros_like(param_vecs[i].data)
    pn_coeff = lalsimulation.SimInspiralTaylorF2AlignedPhasingArray(*param_vecs).data
    pn_coeff = pn_coeff[:len(pn_coeff)//3] # remove coeffs for vlogv and vlogvsq
    pn_coeff = pn_coeff.reshape(-1, num_samples)[b_ref+5] # take only the pn coeff of the ppE order
    beta_ppe_bound = np.abs(pn_coeff) * ppe_ref_v**(b_ref-b_ppe)
    beta_ppe_bound /= eta**(b_ppe/5.)

    beta_ppe = -1. + 2. * np.random.randint(0, 2, num_samples)
    beta_ppe *= beta_ppe_bound
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
    # FIXME
    assert minus_gr

    r = pd.DataFrame(
        data=metadata_array,
        columns=('m1', 'm2', 's1z', 's2z', 'b_ppe', 'beta_ppe'))

    if labels_only:
        return r

    freqs = []
    phases = []
    for m1, m2, s1z, s2z, b_ppe, beta_ppe in metadata_array:

        if not log_spacing:
            freq = np.linspace(fmin, fmax, num_freqs)
        else:
            freq = np.logspace(np.log10(fmin), np.log10(fmax), num_freqs)
        if not freq_in_natural_units:
            freq = freq * u.Hz * (m1 + m2) * u.solMass
            freq = freq * c.G / c.c**3
            freq = freq.to('').value

        eta = modified_gr_utils._symmetric_mass_ratio(m1, m2)
        phase = beta_ppe * eta**(b_ppe/5) * (np.pi*freq)**(b_ppe/3)
        
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
        args.b_ppe, args.ppe_ref, args.ppe_ref_in_hz,
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
