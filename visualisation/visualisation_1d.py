# abstract out visualisation code here. Ideally generalalized visualisation pipelines for 1d/2d/3d models 
from typing import Any, Optional
from matplotlib import pyplot as plt

from utils.constants import AU, RADIUS_SUN

def plot_parker_velocity_profile(
        model: Any,
        sol_out: Any,
        sol_in: Any,
        show_au: bool = True, # show au comparison and critical radius by default
        show_critical: bool = True,
        ax: Optional[plt.Axes] = None
):
    
    if ax is None:
        fig,  ax = plt.subplots(figsize=(8,5))

    # initialise model variables
    cs = model.cs 
    rc = model.critical_radius()

    # normalise values
    r_in_norm = sol_in.t / rc
    r_out_norm = sol_out.t / rc
    u_in_norm = sol_in.y[0] / cs 
    u_out_norm = sol_out.y[0] / cs

    # main curves - joined in/out velocity profiles
    ax.plot(r_in_norm, u_in_norm, "r", label="Subsonic (inward)")
    ax.plot(r_out_norm, u_out_norm, "b", label="Supersonic (outward)")

    # critical radius lines
    if show_critical:
        plt.axvline(1.0, color="k", ls="--", label="Critical radius")
        plt.axhline(1.0, color="gray", ls=":")
    
    # AU comparison
    if show_au:
        au_over_rc = AU / rc
        u_at_au = sol_out.sol(AU)[0]
        u_norm_at_au = u_at_au / cs

        plt.text(
        au_over_rc * 1.01,  # horizontal position of label
        u_norm_at_au * 0.87,  # vertical position of label
        f"{u_norm_at_au:.2f} $c_s$",
        va="center",
        ha="left",
        color="black",
    )
    plt.axvline(au_over_rc, color="mediumseagreen", ls="--", label="1 AU")


    plt.xlabel(r"$r / r_c$")
    plt.ylabel(r"$u / c_s$")
    plt.title("Parker Solar Wind - Subsonic and Supersonic Branches")
    plt.legend()
    plt.grid(True)

    return ax

