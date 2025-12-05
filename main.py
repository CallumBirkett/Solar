from models.models_1d import ParkerIsothermal1D
from physics.equations_of_state import IsothermalEOS, PolytropicEOS
from numerics.solvers.rk import solve_ode 
from visualisation.visualisation_1d import plot_parker_velocity_profile

import matplotlib.pyplot as plt 


def isothermal1d():
    # define eos and model
    eos = IsothermalEOS()
    model = ParkerIsothermal1D(eos)

    # solve Parker ODE
    sol_in, sol_out = model.solve()

    # plotting
    plot_parker_velocity_profile(model, sol_in, sol_out)
    plt.show()

def polytropic1d():
    # give model params
    # define eos and model
    # solve parker ode
    # plotting
    pass
def main():
    isothermal1d()



if __name__ == "__main__":
    main()