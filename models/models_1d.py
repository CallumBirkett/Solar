from utils.constants import G, MASS_SUN, RADIUS_SUN
from physics.equations_of_state import IsothermalEOS
from numerics.solvers.rk import solve_ode

class BaseModel1D:
    def rhs(self, r, u):
        raise NotImplementedError
    
    def solve(self, solver):
        raise NotImplementedError

class ParkerIsothermal1D(BaseModel1D):
    def __init__(self, eos):
        """
        Expects isothermal equation of state from physics/equation_of_state
        """
        self.eos = eos
        self.cs = eos.sound_speed()

    def critical_radius(self):
        return G * MASS_SUN / (2 * self.cs ** 2)
    
    def critical_slope(self):
        rc = self.critical_radius()
        return self.cs / rc 
    
    def rhs(self, r, u):
        cs = self.cs
        numerator = u * (2 * cs ** 2 / r - G * MASS_SUN / r ** 2)
        denominator = u ** 2 - cs ** 2
        return numerator / denominator 
    
    def solve(self):
        """
        Run solve_ode inwards and outwards from the critical radius
        """

        rc = self.critical_radius()
        slope_c = self.critical_slope()

        # 　--- Integration starting points ---
        eps = 1e-3  # small distance away from critical radius
        r0_out = rc * (1 + eps)  # Away from Sun
        u0_out = self.cs + slope_c * (r0_out - rc)
        r0_in = rc * (1 - eps)  # Towards Sun
        u0_in = self.cs + slope_c * (r0_in - rc)


        # --- RK45 Solver ---
        sol_out = solve_ode(
            self.rhs,
            (r0_out, rc * 50),
            [u0_out]
        )
        sol_in = solve_ode(
            self.rhs,
            (r0_in, RADIUS_SUN * (1 + eps)),  # integrate backwards towards Sun's surface
            [u0_in]
        )

        return sol_out, sol_in