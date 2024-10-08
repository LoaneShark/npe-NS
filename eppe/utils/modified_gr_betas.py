import numpy as np
from astropy import units as u
from astropy import constants as c
from importlib import import_module

from .conversion import (
    component_masses_to_chirp_mass, 
    component_masses_to_symmetric_mass_ratio,
)

def get_ppe_beta(mass1, mass2, chi1, chi2, theory, *coupl_args, **coupl_kwargs):
    theory = str(theory).lower()
    func = eval('get_ppe_beta_' + theory)
    return func(mass1, mass2, chi1, chi2, *coupl_args, **coupl_kwargs)

# ============= Theories with b = -1 =============

def get_ppe_beta_dcs(mass1, mass2, chi1, chi2, sqrt_alpha=0):
    """Assume sqrt_alpha given in km"""
    m = mass1 + mass2
    q1, q2 = mass1 / m, mass2 / m
    eta = component_masses_to_symmetric_mass_ratio(mass1, mass2)
    coupl = (sqrt_alpha * u.km) / (m * u.solMass)
    coupl = (c.c**2 / c.G * coupl).to('').value
    xeta_dcs = 16 * np.pi * coupl**4
    chi_s = 0.5 * (chi1 + chi2)
    chi_a = 0.5 * (chi1 - chi2)
    delta_m = q1 - q2
    # see eq. 16 of arXiv:1603.08955
    _term_1 = (1 - 231808. / 61969. * eta) * chi_s**2
    _term_2 = (1 - 16068. / 61969. * eta) * chi_a**2
    _term_3 = -2 * delta_m * chi_s * chi_a
    beta_dcs = 1549225. / 11812864. * xeta_dcs / eta**(14./5.)
    beta_dcs = beta_dcs * (_term_1 + _term_2 + _term_3)
    return beta_dcs

def get_ppe_beta_non_commutative(mass1, mass2, chi1, chi2, sqrt_lambda=0):
    """sqrt_lambda is dimensionless"""
    eta = component_masses_to_symmetric_mass_ratio(mass1, mass2)
    # see arxiv:1809.00259
    beta = -75/256 * eta**(-4/5) * (2*eta-1) * sqrt_lambda**4
    return beta


# ============= Theories with b = -3 =============

class MDRDAlpha(object):
    
    def __init__(self, alpha=2, cosmology='Planck18'):
        self.alpha = alpha
        if type(cosmology) == str:
            cosmology = getattr(import_module('astropy.cosmology'), cosmology)
        self.cosmology = cosmology
        self.z_max = None
        
    def __call__(self, z):
        return self._interp(z) * u.Mpc

    def prepare_interpolation(self, z_max=10, n_interp=1000):
        from scipy.interpolate import interp1d
        zs = np.linspace(0, z_max, n_interp)
        ds = [self.integral_seg(
                    0, z, self.alpha, 
                    self.cosmology.Om0,
                    self.cosmology.Ode0) for z in zs]
        ds = np.asarray(ds) * (1+zs)**(1-self.alpha)
        ds = ds / self.cosmology.H0 * c.c
        ds = ds.to('Mpc').value
        self._interp = interp1d(zs, ds)
        self.z_max = z_max
    
    @staticmethod
    def integral_seg(z1, z2, alpha, Om0, Ode0):
        from scipy.integrate import quadrature
        func = lambda x: (1+x)**(alpha-2) / np.sqrt(Om0*(1+x)**3+Ode0)
        res, _ = quadrature(func, z1, z2)
        return res

_MDRD0 = MDRDAlpha(0)
def get_ppe_beta_massive_gravity(mass1, mass2, chi1, chi2, mg=0, z=0):
    """Assume mg given in eV
        z is dimensionless"""
    # see arxiv:1110.2720 Eq (28)
    if _MDRD0.z_max is None:
        _MDRD0.prepare_interpolation(z_max=10, n_interp=1000)
    mc = component_masses_to_chirp_mass(mass1, mass2)
    lamg = c.c * c.h / (mg * u.eV)
    fac = 1/lamg**2 * _MDRD0(z) * (mc*u.solMass*c.G/c.c**2)
    fac = fac.to('').value
    beta = np.pi**2 / (1+z) * fac
    return beta


# ============= Theories with b = -5 =============

def get_ppe_beta_einstein_aether(mass1, mass2, chi1, chi2, alpha1=0, alpha2=0):
    """Alpha1,2 are dimensionless
        This also works for khronometric gravity"""
    beta = 1/256 * alpha1 / (8 + alpha1)
    beta = beta * (1 - 8*alpha2/alpha1) ** 2.5
    return beta

def get_ppe_beta_khronometric(mass1, mass2, chi1, chi2, alpha1=0, alpha2=0):
    return get_ppe_beta_einstein_eather(mass1, mass2, chi1, chi2, alpha1, alpha2)


# ============= Theories with b = -7 =============

def get_ppe_beta_edgb(mass1, mass2, chi1, chi2, sqrt_alpha=0):
    """Assume sqrt_alpha given in km"""
    def get_s(chi):
        low = 1e-4 # empirically determined, to avoid numerical instability
        chi = np.asarray(chi, dtype=float)
        chi[np.abs(chi) < low] = low
        s = 2./chi**2 * (np.sqrt(1 - chi**2) - 1 + chi**2)
        return s
    s1, s2 = get_s(chi1), get_s(chi2)
    m = mass1 + mass2
    q1, q2 = mass1 / m, mass2 / m
    eta = component_masses_to_symmetric_mass_ratio(mass1, mass2)
    coupl = (sqrt_alpha * u.km) / (m * u.solMass)
    coupl = (c.c**2 / c.G * coupl).to('').value
    xeta_edgb = 16 * np.pi * coupl**4
    # s1 = 2./chi1**2 * (np.sqrt(1 - chi1**2) - 1 + chi1**2)
    # s2 = 2./chi2**2 * (np.sqrt(1 - chi2**2) - 1 + chi2**2)
    beta_edgb = -5. / 7168. / eta**(18./5.) * xeta_edgb
    beta_edgb = beta_edgb * (q1**2 * s2 - q2**2 * s1)**2
    return beta_edgb


def get_ppe_beta_scalar_tensor(mass1, mass2, chi1, chi2, phi_dot=0):
    """Assume phi_dot given in 1 / s"""
    m = mass1 + mass2
    q1, q2 = mass1 / m, mass2 / m
    eta = component_masses_to_symmetric_mass_ratio(mass1, mass2)
    s1_st = 0.5 * (1 + np.sqrt(1 - chi1**2))
    s2_st = 0.5 * (1 + np.sqrt(1 - chi2**2))
    # see eq. 17 of arXiv:1603.08955
    coupl = (phi_dot * u.Hz) * (m * u.solMass)
    coupl = (c.G / c.c**3 * coupl).to('').value
    beta_st = -5./1792. * eta**(2./5.) * coupl**2
    beta_st = beta_st * (q1 * s1_st - q2 * s2_st)**2
    return beta_st


# ============= Theories with b = -13 =============

def get_ppe_beta_extra_dim(mass1, mass2, chi1, chi2, ell=0):
    """Assume ell given in um"""
    m = mass1 + mass2
    eta = component_masses_to_symmetric_mass_ratio(mass1, mass2)
    # see arxiv:1603.08955 Eq (18) and arxiv:1801.03208 Eq (17)
    dmdt = -2.8e-7 / m**2 * (ell/10)**2
    dmdt = (dmdt * c.G * u.solMass / u.year / c.c**3).to('').value
    beta = 25/851968 * dmdt * (3-26*eta+34*eta**2) / eta**0.4 / (1-2*eta)
    return beta

def get_ppe_beta_varying_g(mass1, mass2, chi1, chi2, g_dot=0):
    """Assume g_dot given in 10^(-12) / year"""
    mc = component_masses_to_chirp_mass(mass1, mass2)
    # see eq 65 of arxiv:1809.00259, taking s_A = 0 and delta_Gdot = 0
    g_dot = g_dot * 1e-12 / u.year
    g_dot = g_dot * c.G * u.solMass / c.c**3
    g_dot = g_dot.to('').value
    beta = -25/851968 * g_dot * 11 * mc
    return beta


# ============= EdGB with higher PN orders =============

def get_ppe_beta_edgb_to_nnlo_nospin(mass1, mass2, sqrt_alpha=0):
    """Assume sqrt_alpha given in km"""
    m = mass1 + mass2
    eta = component_masses_to_symmetric_mass_ratio(mass1, mass2)
    coupl = (sqrt_alpha * u.km) / (m * u.solMass)
    coupl = (c.c**2 / c.G * coupl).to('').value
    xeta = 16 * np.pi * coupl**4
    fac = xeta / eta**(18./5.)
    beta_n7 = 5. / 7168. * fac * (4 * eta - 1)
    beta_n5 = 5. / 688128. * fac * (685 - 3916 * eta + 2016 * eta**2)
    beta_n3 = -5. / 387072. * fac * (1 - 2 * eta)**2 * (995 + 952 * eta)
    return beta_n7, beta_n5, beta_n3

