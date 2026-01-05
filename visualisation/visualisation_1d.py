from typing import Optional
from matplotlib import pyplot as plt

from utils.constants import AU, RADIUS_SUN
from models.solutions import Solution1D


def plot_parker_velocity_profile(
    solution: Solution1D,
    show_au: bool = True,
    show_critical: bool = True,
    show_sol: bool = True,
    ax: Optional[plt.Axes] = None
):
    if ax is None:
        _, ax = plt.subplots(figsize=(8, 5))

    # Normalization from critical point
    rc = solution.rc
    cs = solution.cs_crit

    # Normalized branches (dimensionless)
    (r_in_norm, u_in_norm), (r_out_norm, u_out_norm) = solution.normalized_branches()

    # main curves
    ax.plot(r_in_norm, u_in_norm, "r", label="Subsonic (inward)")
    ax.plot(r_out_norm, u_out_norm, "b", label="Supersonic (outward)")

    # critical radius lines
    if show_critical:
        ax.axvline(1.0, color="k", ls="--", label="Critical radius")
        ax.axhline(1.0, color="gray", ls=":")

    # AU comparison
    if show_au:
        au_over_rc = AU / rc

        # Only annotate u(AU) if dense output exists
        if hasattr(solution.sol_out, "sol") and solution.sol_out.sol is not None:
            u_at_au = solution.sol_out.sol(AU)[0]
            u_norm_at_au = u_at_au / cs

            ax.text(
                au_over_rc * 1.01,
                u_norm_at_au * 0.87,
                f"{u_norm_at_au:.2f} $c_{{s,crit}}$",
                va="center",
                ha="left",
                color="black",
            )

        ax.axvline(au_over_rc, color="mediumseagreen", ls="--", label="1 AU")

    # Solar surface marker at R_sun/rc
    if show_sol:
        ax.axvline(0.0, color="k", ls="-", label="Solar radius")

    ax.set_xlabel(r"$r / r_c$")
    ax.set_ylabel(r"$u / c_{s,\mathrm{crit}}$")
    ax.set_title("Parker Solar Wind - Subsonic and Supersonic Branches")
    ax.legend()
    ax.grid(True)

    return ax
