import pytest

from astropy import units as u, constants as c
import lal

from eppe.utils import lal_utils


@pytest.mark.parametrize(
    'm1,m2,s1z,s2z',
    [[10, 20, 0.0, 0.0],
     [20., 20., 0.1, -0.1]]
)
def test_phenomD_si_vs_natural_units(m1, m2, s1z, s2z):
    """Check that waveform phasing is consistent whether
    supplying frequency values in SI units or geometric units.
    """
    
    mtot = m1 + m2
    m_tot_sec = mtot * u.solMass * c.G / c.c**3
    m_tot_sec = m_tot_sec.to('s')
    
    f_min = 10 * u.Hz
    f_max = 100 * u.Hz
    
    Mf_min = (m_tot_sec * f_min).decompose().value
    Mf_max = (m_tot_sec * f_max).decompose().value
    f_min = f_min.value
    f_max = f_max.value

    f_si, phasing_si = lal_utils.get_phenomD_phasing(
        m1, m2, s1z, s2z, f_min, f_max, 100, lal.CreateDict(),
        freq_in_natural_units=False
    )
    f_geom, phasing_geom = lal_utils.get_phenomD_phasing(
        m1, m2, s1z, s2z, Mf_min, Mf_max, 100, lal.CreateDict(),
        freq_in_natural_units=True
    )
    # assert equality of start and end frequency/phasing values
    assert f_si[0] == pytest.approx(f_geom[0] / m_tot_sec.value)
    assert f_si[-1] == pytest.approx(f_geom[-1] / m_tot_sec.value)
    assert phasing_geom[0] == pytest.approx(phasing_si[0], rel=1e-9)
    assert phasing_geom[-1] == pytest.approx(phasing_si[-1], rel=1e-9)
