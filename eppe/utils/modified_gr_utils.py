import numpy as np

from astropy import units as u

from . import natural_units


PN_ORDER_ASSOC = {
    'non_gr_beta_edbg': -1,
    'non_gr_beta_dcs': 2
}


def _symmetric_mass_ratio(mass1, mass2):
    return mass1 * mass2 / (mass1 + mass2)**2


def non_gr_parameter_lalsimulation(beta, eta, pn_order):
    """
    Multiply beta in Nico's notation with correct pre-factors,
    based on PN order, to add to lalsimulation PN phasing series.

    Parameters
    ----------
    beta : float
        Beta ppE coefficient.
    eta : float
        Symmetric mass ratio
    pn_order : int
        PN order corresponding to beta coefficient: Ex. -1 for EDGB, 2 for dCS.
    Notes
    -----
        Basically scaling out the newtonian prefactor as,
        
        :math:`\\beta \rightarrow (128\eta/3) \times
        \beta \eta^{(-5 + 2\text{PN Ord.})/5}`
    """
    r = 128 * eta / 3
    r *= beta
    r *= eta**((-5. + 2 * pn_order) / 5.)
    return r


@natural_units
def non_gr_beta_scalar_tensor(mass1, mass2, chi1, chi2,
                              phi_dot=1.0 * u.Hz):
    """ppE coefficient for scalar-tensor theories of
    gravity.

    Parameters
    ----------
    mass1 : float, array_like
        source frame primary mass
    mass2 : float, array_like
        source frame secondary mass
    chi1 : float, array_like
        dimensionless primary spin magnitude, between 0 and 1
    chi2 : float, array_like
        dimensionless secondary spin magnitude, between 0 and 1
    phi_dot : float
        growth rate of scalar field in cosmological background.
        Units of 1/time. Defaults to 1 Hz in natural units.

    Notes
    -----
    Supply astropy quantities to all values, or provide natural units.
    """
    eta = _symmetric_mass_ratio(mass1, mass2)
    s_st = lambda x: 0.5 * (1 + np.sqrt(1 - x**2))
    s1_st = s_st(chi1)
    s2_st = s_st(chi2)
    # see eq. 17 of arXiv:1603.08955
    beta_st = -5./1792. * phi_dot**2 * eta**(2./5.)
    beta_st *= (mass1 * s1_st - mass2 * s2_st)**2
    return beta_st


@natural_units
def non_gr_beta_edbg(mass1, mass2, chi1, chi2,
                     sqrt_alpha_edgb=3.0*u.km):
    """ppE coefficient for EDGB gravity.

    Parameters
    ----------
    mass1 : float, array_like
        source frame primary mass
    mass2 : float, array_like
        source frame secondary mass
    chi1 : float, array_like
        dimensionless primary spin magnitude along direction of
        angular momentum, between 0 and 1.
    chi2 : float, array_like
        dimensionless secondary spin magnitude along direction of
        angular momentum, between 0 and 1.
    sqrt_alpha_edgb : astropy.quantity.Quantity
        EDGB length scale

    Notes
    -----
    Supply astropy quantities to all values, or provide natural units.
    """
    eta = _symmetric_mass_ratio(mass1, mass2)
    m = mass1 + mass2
    sqrt_alpha_edgb_val = sqrt_alpha_edgb.to('m').value
    xeta_edgb = 16 * np.pi * (sqrt_alpha_edgb_val / m)**4
    s1 = 2./chi1**2 * (np.sqrt(1 - chi1**2) - 1 + chi1**2)
    s2 = 2./chi2**2 * (np.sqrt(1 - chi2**2) - 1 + chi2**2)
    beta_edgb = -5./7168.
    beta_edgb *= xeta_edgb
    beta_edgb /= eta**(18./5.)
    beta_edgb /= m**4
    beta_edgb *= (mass1**2 * s2 - mass2**2 * s1)**2
    return beta_edgb


@natural_units
def non_gr_beta_dcs(mass1, mass2, chi1_perp, chi2_perp,
                    sqrt_alpha_dcs=5.0*u.km):
    """ppE coefficient for dCS gravity.

    Parameters
    ----------
    mass1 : float, array_like
        source frame primary mass
    mass2 : float, array_like
        source frame secondary mass
    chi1_perp : float, array_like
        dimensionless primary spin magnitude along direction of
        angular momentum, between 0 and 1.
    chi2_perp : float, array_like
        dimensionless secondary spin magnitude along direction of
        angular momentum, between 0 and 1.
    sqrt_alpha_dcs : astropy.quantity.Quantity
        dCS length scale.

    Notes
    -----
    Supply astropy quantities to all values, or provide natural units.
    """
    eta = _symmetric_mass_ratio(mass1, mass2)
    m = mass1 + mass2
    sqrt_alpha_dcs_val = sqrt_alpha_dcs.to('m').value
    xeta_dcs = 16 * np.pi * (sqrt_alpha_dcs_val / m)**4
    chi_s = 0.5 * (chi1_perp + chi2_perp)
    chi_a = 0.5 * (chi1_perp - chi2_perp)
    delta_m = mass1 - mass2
    delta_m /= mass1 + mass2
    # see eq. 16 of arXiv:1603.08955
    _term_1 = 1 - 231808. / 61969. * eta
    _term_1 *= chi_s**2
    _term_2 = 1 - 16068. / 61969. * eta
    _term_2 *= chi_a**2
    _term_3 = -2 * delta_m * chi_s * chi_a
    beta_dcs = 1549225. / 11812864. * xeta_dcs / eta**(14./5.)
    beta_dcs *= _term_1 + _term_2 + _term_3

    return beta_dcs
