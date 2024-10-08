import numpy as np

def component_masses_to_chirp_mass(mass_1, mass_2):
    return (mass_1 * mass_2) ** 0.6 / (mass_1 + mass_2) ** 0.2

def component_masses_to_mass_ratio(mass_1, mass_2):
    return mass_2 / mass_1

def component_masses_to_symmetric_mass_ratio(mass_1, mass_2):
    symmetric_mass_ratio = (mass_1 * mass_2) / (mass_1 + mass_2) ** 2
    return np.minimum(symmetric_mass_ratio, 0.25)

def component_masses_to_total_mass(mass_1, mass_2):
    return mass_1 + mass_2

def chirp_mass_and_mass_ratio_to_component_masses(chirp_mass, mass_ratio):
    total_mass = chirp_mass_and_mass_ratio_to_total_mass(
                    chirp_mass=chirp_mass, mass_ratio=mass_ratio)
    mass_1, mass_2 = total_mass_and_mass_ratio_to_component_masses(
                    total_mass=total_mass, mass_ratio=mass_ratio)
    return mass_1, mass_2

def chirp_mass_and_mass_ratio_to_total_mass(chirp_mass, mass_ratio):
    with np.errstate(invalid="ignore"):
        return chirp_mass * (1 + mass_ratio) ** 1.2 / mass_ratio ** 0.6

def chirp_mass_and_total_mass_to_symmetric_mass_ratio(chirp_mass, total_mass):
    return (chirp_mass / total_mass) ** (5 / 3)

def total_mass_and_mass_ratio_to_component_masses(mass_ratio, total_mass):
    mass_1 = total_mass / (1 + mass_ratio)
    mass_2 = mass_1 * mass_ratio
    return mass_1, mass_2

def symmetric_mass_ratio_to_mass_ratio(symmetric_mass_ratio):
    temp = (1 / symmetric_mass_ratio / 2 - 1)
    return temp - (temp ** 2 - 1) ** 0.5
