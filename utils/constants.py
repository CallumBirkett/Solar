# Fundamental constants

G = 6.67430e-11  # Graviational constant - [m^3 kg^-1 s^-2]
K_B = 1.380649e-23  # Boltzmann constant - [J K^-1]
MASS_PROTON = 0.6726219e-27  # Proton mass - [kg]


# Solar constants
MASS_SUN = 1.98847e30  # Solar mass - [kg]
RADIUS_SUN = 6.9634e8  # Solar radius - [km]

# Plasma constants
MMW = 0.6  # # Mean Molecular Weight - dimensionless
TEMP_CORONA = 1.5e6  # Coronal Temperature - [K]

# Optional useful distances
AU = 1.496e11  # astronomical unit [m]


# Calculated constants
def sound_speed(T=TEMP_CORONA, mmw=MMW):
    """
    Sound speed is how strongly pressure responds to compression.
    Derived from the deivative of pressure wrt density.
    """
    return (K_B * T / (mmw * MASS_PROTON)) ** 0.5
