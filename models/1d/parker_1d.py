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

"""
Extensions:

--- Polytropic equation of state ---
Relation between gas pressure and density of the form P = K * rho ** gamma,
where K is constant along stream liens and gamm is the polytropic index.

This form of the relation is derived from the First Law of Thermodynamics
for a fluid element.

Isothermal -> dQ = 0 so no contribution from FLoT. 

In an isothermal flow gamma = 1. 
In an adiabatic flow gamme = 5 / 3.

In reality, the solar wind is neither adiabatic or isothermal, so we assume the gas
behaves with an effective polytropic index of 1 < gamma < 1.5.


- Add heating terms.
- Add rotation.
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
    """
    Isothermal ODE in du/dr. Combination of mass conservation and Euler equation.
    Isothermal -> no energy transfer so no need to include energy equaiton.
    """
    numerator = u * (2 * cs**2 / r - G * M / r**2)
    denominator = u**2 - cs**2  # blow-up when u = cs.
    return numerator / denominator


if __name__ == "__main__":

    # --- Critical radius data ---
    rc = parker_critical_radius()
    slope_c = parker_critical_slope()
    cs = sound_speed()

    # 　--- Integration starting points ---
    eps = 1e-3  # small distance away from critical radius
    r0_out = rc * (1 + eps)  # Away from Sun
    u0_out = cs + slope_c * (r0_out - rc)
    r0_in = rc * (1 - eps)  # Towards Sun
    u0_in = cs + slope_c * (r0_in - rc)

    # --- Set dimensionless step size ---
    h_out = 1e-3 * rc
    h_in = 1e-3 * rc

    # --- RK45 Solver ---
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

    # --- Comparisons at astronomical unit ---
    au_over_rc = AU / rc
    u_at_au = sol_out.sol(AU)[0]
    u_norm_at_au = u_at_au / cs

    # --- Validation statements ---
    # print("Sound Speed =", cs)
    # print("Critical Radius  =", rc)
    # print("Critical Radius / Solar Radius =", rc / RADIUS_SUN)
    # print("Critical Radius / AU =", rc / AU)
    # print("Teff =", MMW * MASS_PROTON * cs**2 / K_B)

    # --- Plotting ---
    plt.figure(figsize=(8, 5))
    plt.plot(sol_in.t / rc, sol_in.y[0] / cs, "r", label="Subsonic (inward)")
    plt.plot(sol_out.t / rc, sol_out.y[0] / cs, "b", label="Supersonic (outward)")

    # Lines to indicate transition around critical radius
    plt.axvline(1.0, color="k", ls="--", label="Critical radius")
    plt.axhline(1.0, color="gray", ls=":")

    # Comparisons to 1 AU
    plt.text(
        au_over_rc * 1.01,
        u_norm_at_au * 0.87,
        f"{u_norm_at_au:.2f} $c_s$",
        va="center",
        ha="left",
        color="black",
    )
    plt.axvline(au_over_rc, color="mediumseagreen", ls="--", label="1 AU")

    plt.xlabel(r"$r / r_c$")
    plt.ylabel(r"$v / c_s$")
    plt.title("Parker Solar Wind - Subsonic and Supersonic Branches")
    plt.legend()
    plt.grid(True)
    plt.show()
    plt.show()
