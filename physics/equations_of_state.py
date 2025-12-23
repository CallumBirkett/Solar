import numpy as np
from utils.constants import K_B, MASS_PROTON, MMW, TEMP_CORONA, G, MASS_SUN


class EquationOfState:
    """Base class."""

    def sound_speed(self, *args, **kwargs):
        raise NotImplementedError


class IsothermalEOS(EquationOfState):
    def __init__(self, T=TEMP_CORONA, mmw=MMW):
        self.T = T
        self.mmw = mmw

    def sound_speed(self):
        return np.sqrt(K_B * self.T / (self.mmw * MASS_PROTON))

class PolytropicEOS(EquationOfState):
    def __init__(self, gamma=1.3, rho0=10e-13, T0 = 1.5e6, mmw=MMW):
        self.gamma = gamma
        self.rho0 = rho0
        self.T0 = T0
        self.P0 = rho0*K_B*T0 / (mmw * MASS_PROTON)
        self.K = self.P0 / (rho0 ** gamma)
        self.mmw = mmw
    
    def pressure(self, rho):
        rho = np.asarray(rho)
        return self.K * rho ** self.gamma # polytropic pressure
    
    def temperature(self, rho):
        rho = np.asarray(rho)
        return (self.mmw * MASS_PROTON / K_B) * self.K * rho**(self.gamma - 1.0) # P = K * rho ** gamma
    
    def sound_speed(self, rho):
        rho = np.asarray(rho)
        return np.sqrt(self.gamma * self.K * rho ** (self.gamma - 1.0)) # definition of sound speed
    
    