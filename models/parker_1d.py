import numpy as np
from matplotlib import pyplot as plt
from scipy.integrate import solve_ivp

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


# --- Calculated values ---
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
    return G * M / (2 * cs**2)


def parker_critical_slope(G=G, M=MASS_SUN, cs=sound_speed()):
    """
    Gradient at the critical radius must be cs/rc in order to get a smooth solution.
    Follows by applying L'Hoptial's rule at the critical radius.
    """
    rc = parker_critical_radius(G, M, cs)
    return cs / rc


# --- ODE Definition ---
def parker_rhs(r, u, G=G, M=MASS_SUN, cs=sound_speed()):
    numerator = u * (2 * cs**2 / r - G * M / r**2)
    denominator = u**2 - cs**2  # blow-up when u = cs.
    return numerator / denominator


# --- RK4 Integrator Leads to unstable colution below critical radius. RK45 solver now in use ---
# # RK4 integrator step function
# def rk4_step(f, r, u, h, args):
#     """
#     Calculation of slope based on a degree 4 Runge-Kutta method.
#     """
#     k1 = f(r, u, *args)
#     k2 = f(r + 0.5 * h, u + 0.5 * h * k1, *args)
#     k3 = f(r + 0.5 * h, u + 0.5 * h * k2, *args)
#     k4 = f(r + h, u + h * k3, *args)
#     return u + (h / 6) * (
#         k1 + 2 * k2 + 2 * k3 + k4
#     )  # return averaged contribution from slopes


# # RK4 integration wrapper
# def integrate_rk4(f, r0, u0, r_end, h, args):
#     r, u = r0, u0
#     rs, us = [r], [u]  # initialise arrays to hold radius data and resulting u calcs
#     while (h > 0 and r < r_end) or (h < 0 and r > r_end):
#         u = rk4_step(f, r, u, h, args)
#         r += h
#         rs.append(r)
#         us.append(u)
#     return np.array(rs), np.array(us)


if __name__ == "__main__":

    # critical radius data.
    rc = parker_critical_radius()
    slope_c = parker_critical_slope()
    cs = sound_speed()

    # 　Set starting points for the integration
    eps = 1e-3  # small distance away from critical radius
    r0_out = rc * (1 + eps)  # Away from Sun
    u0_out = cs + slope_c * (r0_out - rc)
    r0_in = rc * (1 - eps)  # Towards Sun
    u0_in = cs + slope_c * (r0_in - rc)

    # system spans many orders of magnitude in radius, set dimensionless step-size
    h_out = 1e-3 * rc
    h_in = 1e-3 * rc

    # integrate numerical solution using rk45 solver
    sol_out = solve_ivp(
        parker_rhs,
        (r0_out, rc * 50),
        [u0_out],
        method="RK45",
        rtol=1e-8,
        atol=1e-10,
        dense_output=True,
    )
    sol_in = solve_ivp(
        parker_rhs,
        (r0_in, RADIUS_SUN * (1 + eps)),  # integrate backwards towards Sun's surface
        [u0_in],
        method="RK45",
        rtol=1e-8,
        atol=1e-10,
        dense_output=True,
    )

    # comparisons at astronomical unit
    au_over_rc = AU / rc
    u_at_au = sol_out.sol(AU)[0]
    u_norm_at_au = u_at_au / cs

    print("Sound Speed =", cs)
    print("Critical Radius  =", rc)
    print("Critical Radius / Solar Radius =", rc / RADIUS_SUN)
    print("Critical Radius / AU =", rc / AU)
    print("Teff =", MMW * MASS_PROTON * cs**2 / K_B)

    # plotting
    plt.figure(figsize=(8, 5))
    plt.plot(sol_in.t / rc, sol_in.y[0] / cs, "r", label="Subsonic (inward)")
    plt.plot(sol_out.t / rc, sol_out.y[0] / cs, "b", label="Supersonic (outward)")
    plt.axvline(1.0, color="k", ls="--", label="Critical radius")
    plt.text(
        au_over_rc * 1.01,
        u_norm_at_au * 0.87,
        f"{u_norm_at_au:.2f} $c_s$",
        va="center",
        ha="left",
        color="black",
    )
    plt.axvline(au_over_rc, color="mediumseagreen", ls="--", label="1 AU")
    plt.axhline(1.0, color="gray", ls=":")
    plt.xlabel(r"$r / r_c$")
    plt.ylabel(r"$v / c_s$")
    plt.title("Parker Solar Wind - Subsonic and Supersonic Branches")
    plt.legend()
    plt.grid(True)
    plt.show()
    plt.show()
