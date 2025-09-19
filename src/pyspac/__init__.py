# In src/pyspac/__init__.py

__version__ = "0.1.0"

# Expose the main class of the package
from .pyspac import PhaseCurve

# Expose the most useful constants for the user to inspect
from .constants import (
    ALLOWED_PHASE_CURVE_MODELS,
    GENERAL_CONSTRAINT_METHODS,
    ALL_FITTING_METHODS,
)
