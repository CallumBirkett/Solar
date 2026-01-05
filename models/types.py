from dataclasses import dataclass

@dataclass(frozen=True) # derived values should be immutable
class CriticalPoint:
    crc: float
    uc: float
    cs_crit: float
    slope: float 