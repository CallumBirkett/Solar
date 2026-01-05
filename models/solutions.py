from dataclasses import dataclass, field
from typing import Any, Dict, Optional, Tuple
from models.types import CriticalPoint

@dataclass(frozen=True)
class Solution1D:
    model: Any
    critical: CriticalPoint
    sol_in: Any
    sol_out: Any
    meta: Dict[str, Any] = field(default_factory=dict) # optional additional info

    @property
    def rc(self) -> float:
        return self.critical.rc
    
    @property
    def cs_crit(self) -> float:
        return self.critical.cs_crit
    
    def normalized_branches(self) -> Tuple[Tuple[Any, Any], Tuple[Any, Any]]:
        r_in = self.sol_in.t / self.rc 
        r_out = self.sol_out.t / self.rc 

        u_in = self.sol_in.y[0] / self.cs_crit 
        u_out = self.sol_out.y[0] / self.cs_crit

        return (r_in, u_in), (r_out, u_out)