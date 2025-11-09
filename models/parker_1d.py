# --- ensure project root is on the import path ---
import os, sys

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)
# -------------------------------------------------

from utils.constants import *

# AIM: Develop a model for the 1D (radially symmetric Parker Solar Wind) with isothermal flow.

# Mass conservation

# Euler Equations

# Isothermal Closure

# Non-dimensionalize units

# Derive a critical point

# Solve time-dependent equations until steady-state is reached.

# Model validation

"""
Extensions:

- Polytropic equation of state.
- Add heating terms.
- Add rotation.
- Move to 1D time-dependent HD solver.
"""


# Calculated values
def sound_speed(T=TEMP_CORONA, mmw=MMW):
    """
    Isothermal sound speed.
    Sound speed is how strongly pressure responds to compression.
    Derived from the deivative of pressure wrt density.
    """
    return (K_B * T / (mmw * MASS_PROTON)) ** 0.5


def parker_critical_radius(G=G, M=MASS_SUN, cs=sound_speed()):
    """
    Radius at which singularities occur in the Parker ODE.
    Integrate outwards from here.
    """
    return [G * M / 2 * cs**2]


def parker_critical_slope(G=G, M=MASS_SUN, cs=sound_speed()):
    """
    Gradient at the critical radius must be cs/rc in order to get a smooth solution.
    Follows by applying L'Hoptial's rule at the critical radius.
    """
    rc = parker_critical_radius(G, M, cs)
    return cs / rc


# ODE Definition
def parker_rhs(r, u, G=G, M=MASS_SUN, cs=sound_speed):
    numerator = u * ([2 * cs**2 / r] - [G * M / r**2])
    denominator = u**2 - cs**2  # blow-up when u = cs.
    return numerator / denominator


# ODE Solver


if __name__ == "__main__":

    print("G: ", G)
    print("MASS_SUN: ", MASS_SUN)
    print("Sound Speed: ", cs)
