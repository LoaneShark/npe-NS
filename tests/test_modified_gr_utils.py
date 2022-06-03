import pytest

from astropy import units as u, constants as c
import lal, lalsimulation
import numpy as np

from eppe import __version__
from eppe.utils import modified_gr_utils, lal_utils

def test_version():
    assert __version__ == '0.1.0'


def test_edgb_parameter_sanity():
    """EDGB parameter sanity check when passing
    masses in natural vs SI units."""
    mass1 = 20 * u.solMass
    mass2 = 10 * u.solMass
    chi1 = 0.1
    chi2 = 0.1

    mass1_in_m = mass1 * c.G / c.c**2
    mass1_in_m = mass1_in_m.to('m').value
    mass2_in_m = mass2 * c.G / c.c**2
    mass2_in_m = mass2_in_m.to('m').value

    assert modified_gr_utils.non_gr_beta_edbg(
        mass1, mass2, chi1, chi2, sqrt_alpha_edgb=3*u.km
    ) == pytest.approx(
        modified_gr_utils.non_gr_beta_edbg(
            mass1_in_m, mass2_in_m, chi1, chi2,
            sqrt_alpha_edgb=3*u.km
        )
    )


def test_dcs_parameter_sanity():
    """dCS parameter sanity check when passing
    masses in natural vs SI units."""
    mass1 = 20 * u.solMass
    mass2 = 10 * u.solMass
    chi1 = 0.1
    chi2 = 0.1

    mass1_in_m = mass1 * c.G / c.c**2
    mass1_in_m = mass1_in_m.to('m').value
    mass2_in_m = mass2 * c.G / c.c**2
    mass2_in_m = mass2_in_m.to('m').value

    assert modified_gr_utils.non_gr_beta_dcs(
        mass1, mass2, chi1, chi2, sqrt_alpha_dcs=5*u.km
    ) == pytest.approx(
        modified_gr_utils.non_gr_beta_dcs(
            mass1_in_m, mass2_in_m, chi1, chi2,
            sqrt_alpha_dcs=5*u.km
        )
    )


@pytest.mark.parametrize(
    'm1,m2,s1z,s2z,z,beta_dcs,beta_edgb',
    [[20 * u.solMass, 20 * u.solMass, 0.2, 0.3, 0.1,
      1.98560523277534e+15, -55954111690.6503],
     [20 * u.solMass, 10 * u.solMass, 0.2, -0.3, 0.4,
      2.42452158852979e+17 , -6.66139426872417e+15],
     [30 * u.solMass, 20 * u.solMass, -0.6, 0.3, 0.2,
      3.34052475935505e+16 , -147469980447714],
     [30 * u.solMass, 10 * u.solMass, -0.1, 0.05, 0.3,
      4.96946796687268e+15 , -6.87481734971683e+15]]
)
def test_dCS_EDGB_beta_value_sanity(m1, m2, s1z, s2z, z, beta_dcs, beta_edgb):
    """Test known values of beta dCS and beta EDGB.
    Reference values from Scott Perkins."""
    source_m1 = m1 / (1 + z)
    source_m2 = m2 / (1 + z)
    my_beta_dcs = modified_gr_utils.non_gr_beta_dcs(
        source_m1, source_m2, s1z, s2z,
        sqrt_alpha_dcs=1*u.s  # Ref values use alpha = 1 s2
    )
    my_beta_edgb = modified_gr_utils.non_gr_beta_edbg(
        source_m1, source_m2, s1z, s2z,
        sqrt_alpha_edgb=1*u.s
    )
    # minor differences due to difference in numbers in astropy vs
    # that used by Scott to generate reference values
    assert my_beta_dcs == pytest.approx(beta_dcs, rel=1e-3)
    assert my_beta_edgb == pytest.approx(beta_edgb, rel=1e-3)


@pytest.mark.parametrize(
    'insert_attr,param_value',
    [['SimInspiralWaveformParamsInsertNonGRBetaPPE4', 1.0],
     ['SimInspiralWaveformParamsInsertNonGRBetaPPEMinus2', 3.0]]
)
def test_phemonD_phasing_sanity(insert_attr, param_value):
    "Test phenomD phasing array"
    m1, m2 = 20., 10.
    s1z, s2z = 0.1, 0.1
    fmin, fmax = 10., 500.,
    num_freqs = 100
    lal_params = lal.CreateDict()
    getattr(lalsimulation, insert_attr)(lal_params, param_value)

    freq1, phase1 = lal_utils.get_phenomD_phasing(
        m1, m2, s1z, s2z, fmin, fmax, num_freqs, lal_params
    )
    freq2, phase2 = lal_utils.get_phenomD_phasing(
        m2, m1, s2z, s1z, fmin, fmax, num_freqs, lal_params
    )  # swap primary and secondary as a basic sanity check
    assert np.allclose(phase1, phase2)
