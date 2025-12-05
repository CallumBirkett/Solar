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
    def __init__(self, gamma, K, mmw=MMW):
        self.gamma = gamma
        self.K = K 
        self.mmw = mmw

    @classmethod 
    def from_base_params(
        cls, # use a class method for extensibility (e.g. ModifiedPolytropicEOS(PolytropicEOS) produces instances of ModifiedPolytropicEOS)
        gamma,
        rho0,
        T0,
        mmw=MMW
    ):
        P0 = rho0*K_B*T0 / (mmw * MASS_PROTON)
        K = P0 / (rho0 ** gamma)
        return cls(gamma=gamma, K=K, mmw=mmw) # return an instance from base parameters
    
    def pressure(self, rho):
        rho = np.asarrary(rho)
        return self.K * rho ** self.gamma # polytropic pressure
    
    def temperature(self, rho):
        rho = np.asarray(rho)
        return (self.mmw * MASS_PROTON / K_B) * self.K * self.rho**(self.gamma - 1.0) # P = K * rho ** gamma
    
    def sound_speed(self, rho):
        rho = np.asarray(rho)
        return np.sqrt(self.gamma * self.K * rho ** (self.gamma - 1.0)) # definition of sound speed
    
    