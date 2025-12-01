import numpy as np
from utils.constants import K_B, MASS_PROTON, MMW, TEMP_CORONA


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

