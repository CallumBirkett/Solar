from models.models_1d import ParkerIsothermal1D, ParkerPolytropic1D
from physics.equations_of_state import IsothermalEOS, PolytropicEOS
from numerics.solvers.rk import solve_ode 
from visualisation.visualisation_1d import plot_parker_velocity_profile

import matplotlib.pyplot as plt 


def isothermal1d():
    # define eos and model
    eos = IsothermalEOS()
    model = ParkerIsothermal1D(eos)

    # solve ODE
    solution = model.solve()

    # plotting
    plot_parker_velocity_profile(solution)
    plt.show()

def polytropic1d():
    # define eos and model
    eos = PolytropicEOS()
    model = ParkerPolytropic1D(eos)

    # solve ODE
    solution = model.solve()

    # plotting
    plot_parker_velocity_profile(solution)
    plt.show()

def main():
    polytropic1d()


if __name__ == "__main__":
    main()