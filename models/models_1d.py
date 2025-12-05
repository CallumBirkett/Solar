from utils.constants import G, MASS_SUN, RADIUS_SUN
from physics.equations_of_state import IsothermalEOS, PolytropicEOS
from numerics.solvers.rk import solve_ode

import numpy as np

class BaseModel1D:
    def rhs(self, r, u):
        raise NotImplementedError
    
    def solve(self):
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

        cs = self.cs
        rc = self.critical_radius()
        slope_c = self.critical_slope()

        # 　--- Integration starting points ---
        eps = 1e-3  # small distance away from critical radius
        r0_out = rc * (1 + eps)  # Away from Sun
        u0_out = cs + slope_c * (r0_out - rc)
        r0_in = rc * (1 - eps)  # Towards Sun
        u0_in = cs + slope_c * (r0_in - rc)


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

        return sol_in, sol_out
    

class ParkerPolytropic1D(BaseModel1D):
    def __init__(self, eos, rho0 = 10e-13, T0 = 1.5e6, u0=1e3, r0=RADIUS_SUN):
        self.eos = eos
        self.gamma = eos.gamma
        self.K = eos.K
        self.rho0 = rho0
        self.T0 = T0
        self.u0 = u0
        self.r0 = r0
        self.cs0 = self.eos.sound_speed(self.rho0)
        self.mass_flux = self.rho0 * self.u0 * self.r0**2
    
    # critical radius expression
    def critical_radius(self, csc): # csc defined in solve()
        return G * MASS_SUN / (2 * csc ** 2)
    
    def density(self, r, u):
        return self.mass_flux / (u * r ** 2)
    
    def sound_speed(self, r, u):
        rho = self.density(r, u)
        return self.eos.sound_speed(rho)

    def critical_sound_speed(self, gamma, cs0, u0, r0, G=G, M=MASS_SUN):
        """For non-isothermal solution. Compute critical sound speed from coronal 
            base by solving Bernoulli equation.
            Bernoulli energy is constant on streamlines."""
        numerator = 0.5 * u0 ** 2 + (1 / gamma) * cs0 ** 2 - (G * M / r0)
        denominator = 1 / (gamma - 1.0) - 1.5

        # check for blow-up
        if denominator == 0:
            raise ValueError("Denominator vanishes at gamma = 5/3, no regular transonic solution")
        
        # proceed only if no blow-up
        csc_squared = numerator / denominator
        
        # check for physical solution
        if csc_squared <= 0:
            raise ValueError("Computed critical sound speed squared and found it to be <= 0. Base conditions" \
            "incompatibile with transonic solution")
        
        return np.sqrt(csc_squared)

    def critical_slope(self, gamma, csc, rc
    ):
        disc_condition = 5.0 - 3.0 * gamma
        disc = 2.0 * (5.0 - 3.0 * gamma)
        numerator = 2.0 * (1.0 - gamma) + np.sqrt(disc)
        denominator = gamma + 1.0

        if disc_condition < 0:
            raise ValueError("No real solutions for critical slope, gamma must be <= 5/3")
        
        uc = csc

        return (uc / rc) * (numerator/denominator)

    
    def rhs(self, r, u):
        cs = self.sound_speed(r, u)
        numerator = u * (2 * cs ** 2 / r - G * MASS_SUN / r ** 2)
        denominator = u ** 2 - cs ** 2
        return numerator / denominator 


    def solve(self):
        """
        Run solve_ode inwards and outwards from the critical radius
        """

        csc = self.critical_sound_speed(self.gamma, self.cs0, self.u0, self.r0)
        rc = self.critical_radius(csc)
        slope_c = self.critical_slope(self.gamma, csc, rc)

        # 　--- Integration starting points ---
        eps = 1e-3  # small distance away from critical radius
        r0_out = rc * (1 + eps)  # Away from Sun
        u0_out = csc + slope_c * (r0_out - rc)
        r0_in = rc * (1 - eps)  # Towards Sun
        u0_in = csc + slope_c * (r0_in - rc)


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

        return sol_in, sol_out