import numpy as np
from matplotlib import pyplot as plt

# --- ensure project root is on the import path ---
import os, sys

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)
# -------------------------------------------------

from utils.constants import *

# AIM: Develop a model for the 1D (radially symmetric Parker Solar Wind) with isothermal flow.

# Non-dimensionalize units

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
    return G * M / 2 * cs**2


def parker_critical_slope(G=G, M=MASS_SUN, cs=sound_speed()):
    """
    Gradient at the critical radius must be cs/rc in order to get a smooth solution.
    Follows by applying L'Hoptial's rule at the critical radius.
    """
    rc = parker_critical_radius(G, M, cs)
    return cs / rc


# ODE Definition
def parker_rhs(r, u, G=G, M=MASS_SUN, cs=sound_speed):
    numerator = u * (2 * cs**2 / r - G * M / r**2)
    denominator = u**2 - cs**2  # blow-up when u = cs.
    return numerator / denominator


# RK4 integrator step function
def rk4_step(f, r, u, h, args):
    """
    Calculation of slope based on a degree 4 Runge-Kutta method.
    """
    k1 = f(r, u, *args)
    k2 = f(r + 0.5 * h, u + 0.5 * h * k1, *args)
    k3 = f(r + 0.5 * h, u + 0.5 * h * k2, *args)
    k4 = f(r + h, u + h * k3, *args)
    return u + (h / 6) * (
        k1 + 2 * k2 + 2 * k3 + k4
    )  # return averaged contribution from slopes


# RK4 integration wrapper
def integrate_rk4(f, r0, u0, r_end, h, args):
    r, u = r0, u0
    rs, us = [r], [u]  # initialise arrays to hold radius data and resulting u calcs
    while (h > 0 and r < r_end) or (h < 0 and r > r_end):
        u = rk4_step(f, r, u, h, args)
        r += h
        rs.append(r)
        us.append(u)
    return np.array(rs), np.array(us)


if __name__ == "__main__":

    # critical radius data.
    rc = parker_critical_radius()
    slope_c = parker_critical_slope()

    # 　Set starting points for the integration
    eps = 1e-3
    r0_out = rc * (1 + eps)  # Away from Sun
    u0_out = sound_speed() + slope_c * (r0_out - rc)
    r0_in = rc * (1 - eps)  # Towards Sun
    u0_in = sound_speed() + slope_c * (r0_in - rc)

    # system spans many orders of magnitude in radius, set dimensionless step-size
    h = 1e-3 * rc

    r_out, u_out = integrate_rk4(
        parker_rhs, r0_out, u0_out, rc * 20, h, (G, MASS_SUN, sound_speed())
    )
    r_in, u_in = integrate_rk4(
        parker_rhs, r0_out, u0_out, rc * 0.3, -h, (G, MASS_SUN, sound_speed())
    )

    # join together values into one array
    r_all = np.concatenate([r_in[::-1], r_out])
    u_all = np.concatenate([u_in[::-1], u_out])

    # plotting
    plt.figure(figsize=(7, 5))
    plt.plot(r_all / rc, u_all / sound_speed(), lw=2)
    plt.axvline(1.0, color="k", ls="--", label="Critical Point")
    plt.xlabel(r"$r/rc$")
    plt.ylabel(r"$u/cs$")
    plt.title("Isothermal Parker Wind - RK4 Integration")
    plt.grid(True)
    plt.legend()
    plt.show()
