# --- ensure project root is on the import path ---
import os, sys

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)
# -------------------------------------------------

from utils.constants import G, MASS_SUN, sound_speed

# AIM: Develop a model for the 1D (radially symmetric Parker Solar Wind) with isothermal flow.

# Mass conservation

# Euler Equations

# Isothermal Closure

# Non-dimensionalize units

# Derive a critical point

# Solve time-dependent equations until steady-state is reached.

# Model validation

"""
Extensions:

- Polytropic equation of state.
- Add heating terms.
- Add rotation.
- Move to 1D time-dependent HD solver.
"""

if __name__ == "__main__":
    cs = sound_speed()
    print("G: ", G)
    print("MASS_SUN: ", MASS_SUN)
    print("Sound Speed: ", cs)
