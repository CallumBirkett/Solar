from dataclasses import dataclass

@dataclass(frozen=True) # derived values should be immutable
class CriticalPoint:
    critical_radius: float
    critical_speed: float
    sound_speed_crit: float
    slope_crit: float 