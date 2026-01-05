from dataclasses import dataclass


@dataclass(frozen=True) # derived values should be immutable
class CriticalPoint:
    rc: float
    uc: float
    cs_crit: float
    slope: float 

