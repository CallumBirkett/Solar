from utils.constants import G, MASS_SUN, RADIUS_SUN
from physics.equations_of_state import IsothermalEOS, PolytropicEOS
from numerics.solvers.rk import solve_ode
from models.types import CriticalPoint
from models.solutions import Solution1D

import numpy as np

class BaseModel1D:

    def critical_point(self):
        raise NotImplementedError # any wind model must have a critical point
    
    @property # expose derived physics as read-only observables
    def rc(self):
        return self.critical_point().rc
    
    @property
    def cs_crit(self):
        return self.critical_point().cs_crit
    
    @property
    def cs(self):
        return self.cs_crit 
    
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
    
    def critical_slope(self, rc, cs):
        """
        A helper function for calculating critical point data
        """
        return cs / rc 
    
    def critical_point(self):
        cs = self.eos.sound_speed()
        rc = G * MASS_SUN / (2 * cs ** 2)
        slope = self.critical_slope(rc=rc, cs=cs)
        uc = cs # universal critical speed
        return CriticalPoint(rc=rc, uc=uc, cs_crit=cs, slope=slope) # return dataclass object (with builtin constructor)
    
    def rhs(self, r, u):
        cs = self.cs_crit
        numerator = u * (2 * cs ** 2 / r - G * MASS_SUN / r ** 2)
        denominator = u ** 2 - cs ** 2
        return numerator / denominator 
    
    def solve(self, *, eps: float = 1e-3, r_max_factor: float = 50.0) -> Solution1D:

        cp = self.critical_point()
        rc, cs, slope_c = cp.rc, cp.cs_crit, cp.slope 

        r0_out = rc * (1.0 + eps)
        u0_out = cs + slope_c * (r0_out - rc)

        r0_in = rc * (1 - eps)
        u0_in = cs + slope_c * (r0_in - rc)

        sol_out = solve_ode(self.rhs, (r0_out, rc * r_max_factor), [u0_out])
        sol_in = solve_ode(self.rhs, (r0_in, RADIUS_SUN * (1.0 + eps)), [u0_in])

        return Solution1D(
            model = self,
            critical = cp,
            sol_in=sol_in,
            sol_out=sol_out,
            meta = {
                "eps": eps,
                "r_max_factor": r_max_factor,
                "r0_out": r0_out,
                "r0_in": r0_in,
                "u0_out": u0_out,
                "u0_in": u0_in
            }
        )
    

class ParkerPolytropic1D(BaseModel1D):
    def __init__(self, eos, rho0 = 10e-13, T0 = 1.5e6, u0=1e3, r0=2.5*RADIUS_SUN):
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
        numerator = 0.5 * u0 ** 2 +  cs0 ** 2 / (gamma - 1.0) - (G * M / r0)
        denominator = 1 / (gamma - 1.0) - 1.5

        # check for blow-up
        if denominator == 0:
            raise ValueError("Denominator vanishes at gamma = 5/3, no regular transonic solution")
        
        # proceed only if no blow-up
        csc_squared = numerator / denominator
        
        # check for physical solution
        if csc_squared <= 0:
            raise ValueError("Computed critical sound speed squared and found it to be <= 0. Base conditions " \
            "incompatibile with transonic solution")
        
        return np.sqrt(csc_squared)

    def critical_slope(self, gamma, csc, rc):
        disc_condition = 5.0 - 3.0 * gamma

        if disc_condition < 0:
            raise ValueError("No real solutions for critical slope, gamma must be <= 5/3")
    
        disc = 2.0 * (5.0 - 3.0 * gamma)
        numerator = -2.0 * (1.0 - gamma) + np.sqrt(disc)
        denominator = gamma + 1.0
        
        uc = csc

        return (uc / rc) * (numerator/denominator)

    def critical_point(self) -> CriticalPoint:
        csc = self.critical_sound_speed(self.gamma, self.cs0, self.u0, self.r0)
        rc = self.critical_radius(csc)
        slope = self.critical_slope(self.gamma, csc, rc)
        uc = csc
        return CriticalPoint(rc=rc, uc=uc, cs_crit=csc, slope=slope)
    
    def rhs(self, r, u):
        cs = self.sound_speed(r, u)
        numerator = u * (2 * cs ** 2 / r - G * MASS_SUN / r ** 2)
        denominator = u ** 2 - cs ** 2
        return numerator / denominator 


    def solve(self, *, eps: float = 1e-3, r_max_factor: float = 50.0) -> Solution1D:

        cp = self.critical_point()
        rc, cs, slope_c = cp.rc, cp.cs_crit, cp.slope 

        r0_out = rc * (1.0 + eps)
        u0_out = cs + slope_c * (r0_out - rc)

        r0_in = rc *(1 - eps)
        u0_in = cs + slope_c * (r0_in - rc)

        sol_out = solve_ode(self.rhs, (r0_out, rc * r_max_factor), [u0_out])
        sol_in = solve_ode(self.rhs, (r0_in, RADIUS_SUN * (1.0 + eps)), [u0_in])

        return Solution1D(
            model = self,
            critical = cp,
            sol_in=sol_in,
            sol_out=sol_out,
            meta = {
                "eps": eps,
                "r_max_factor": r_max_factor,
                "r0_out": r0_out,
                "r0_in": r0_in,
                "u0_out": u0_out,
                "u0_in": u0_in
            }
        )