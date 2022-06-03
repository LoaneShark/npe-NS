from functools import wraps

import astropy.constants as c
import astropy.units as u

def _natural():
    """
    Returns a list of equivalence pairs that handle the conversion
    between mass and energy.
    """
    schwarzchild_factor = c.G * c.M_sun / c.c**2  # roughly M_sun -> 1.5 km

    sol_mass_to_meters = lambda _: _ * schwarzchild_factor.to('m').value
    meters_to_sol_mass = lambda _: _ / schwarzchild_factor.to('m').value
    sol_mass_to_seconds = lambda _: (
        sol_mass_to_meters(_) * u.m / c.c
    ).to('s').value
    seconds_to_sol_mass = lambda _: meters_to_sol_mass(
            (_ * u.s * c.c).to('m').value
    )

    return u.Equivalency(
        [(u.M_sun, u.m, sol_mass_to_meters, meters_to_sol_mass),
         (u.M_sun, u.s, sol_mass_to_seconds, seconds_to_sol_mass),
         (u.m, u.s, lambda _: (_ / c.c).value, lambda _: (_ * c.c).value)]
    )


def natural_units(func):
    @wraps(func)
    def res(*args, **kwargs):
        mass1, mass2, *other_args = args
        if hasattr(mass1, 'to'):
            mass1 = mass1.to('m', equivalencies=_natural()).value
        if hasattr(mass2, 'to'):
            mass2 = mass2.to('m', equivalencies=_natural()).value
        new_args = (mass1, mass2, *other_args)
        new_kwargs = kwargs.copy()
        for kwarg, val in kwargs.items():
            if kwarg.startswith("sqrt_alpha"):  # FIXME: could be buggy
                new_kwargs.update(
                    {kwarg : val.to('m', equivalencies=_natural())}
                )
        return func(*new_args, **new_kwargs)
    return res


__all__ = ('modified_gr_utils', 'lal_utils')
