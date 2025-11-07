# --- ensure project root is on the import path ---
import os, sys

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)
# -------------------------------------------------

import utils.constants

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
