from fractions import Fraction
from functools import reduce
import operator

from astropy import units as u, constants as c
from gwpy.timeseries import TimeSeries
from gwpy.frequencyseries import FrequencySeries
import lal
import lalsimulation
import numpy as np


_strain_unit = u.IrreducibleUnit('strain')
_lal_unit_map = [u.m, u.kg, u.s, u.A, u.K, _strain_unit, u.count]
_lal_unit_ind = {unit: ind for ind, unit in enumerate(_lal_unit_map)}

lal_unit_bases = _lal_unit_map


# adopted from https://git.ligo.org/waveforms/new-waveforms-interface/-/blob/master/prototype/prototype_interface.py#L14-189
def from_lal_unit(lunit):
    aunit = reduce(
        operator.mul,
        (
            _lal_unit_map[i] ** Fraction(num ,den+1)
            for i, (num, den) in enumerate(
                zip(lunit.unitNumerator, lunit.unitDenominatorMinusOne)
            )
        )
    )
    return aunit * 10**lunit.powerOfTen


def from_lal_frequencyseries(lalfs):
    unit = from_lal_unit(lalfs.sampleUnits)
    return FrequencySeries(
        lalfs.data.data, name=lalfs.name,
        epoch=lalfs.epoch, f0=lalfs.f0,
        df=lalfs.deltaF, unit=unit
    )


def to_timeseries(series):
    if isinstance(series, TimeSeries):
        return series  # noop
    elif isinstance(series, FrequencySeries):
        # note: undo gwpy's normalization and apply correct measure
        # FIXME not sure the factors are right...
        # FIXME: do something to avoid wrap-around and modify epoch
        return series.ifft() * series.df
    else:
        raise ValueError('series must be a TimeSeries')


def get_phenomD_phasing(m1, m2, s1z, s2z, fmin, fmax, num_freqs, lal_params,
                        log_spacing=False, freq_in_natural_units=False):
    """Get PhenomD phasing, phi(f).
    
    Parameters
    ----------
    m1 : float
        primary mass in solar mass
    m2 : float
        secondary mass in solar mass
    s1z : float
        z-component of dimensionless primary spin
    s2z : float
        z-component of dimensionless primary spin
    fmin : float
        minimum frequency
    fmax : float
        maximum frequency
    num_freqs : int
        number of frequency points
    lal_params : LALDict
        LAL dictionary storing modified gravity parameters
    log_spacing : bool
        whether log-spaced as opposed to lin-spaced. Default False.
    freq_in_natural_units : bool
        whether the frequency values are in dimensionless units.

    Returns
    -------
    freqs, phases : numpy.array
        f, phi(f)
    
    Notes
    -----
    Frequencies are interpreted in dimensionless units when ``freq_in_natural_units``
    is supplied i.e. create Mf array, obtain f, then supply to phasing function.
    Note that for PhenomD all system ringdown at the same geometric frequency.
    """
    phases = lal.CreateREAL8Sequence(num_freqs)
    freqs = lal.CreateREAL8Sequence(num_freqs)
    spacing_func = np.logspace if log_spacing else np.linspace
    fmin = np.log10(fmin) if log_spacing else fmin
    fmax = np.log10(fmax) if log_spacing else fmax
    freq_vals = spacing_func(fmin, fmax, num_freqs)
    freqs.data = freq_vals
    if freq_in_natural_units:
        m_tot_sec = m1 + m2
        m_tot_sec *= u.solMass * c.G / c.c**3
        freqs.data /= m_tot_sec.to('s').value
    lalsimulation.IMRPhenomDPhaseFrequencySequence(
        phases, freqs,
        0, num_freqs,  # min and max index to be populated, we want all.
        m1, m2,  # these have to be in solar masses
        0., 0., s1z,  # x, y, z of s1
        0., 0., s2z,  # x, y, z of s2
        lal_params
    )
    return freq_vals, phases.data
