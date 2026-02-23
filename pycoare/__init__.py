"""PyCOARE - A Python implementation of the COARE algorithm.

PyCOARE is a Python implementation of the COARE (Coupled Ocean-Atmosphere Response Experiment) algorithm, which is used to calculate air-sea fluxes of momentum, heat, and moisture. This package provides functions for both the 3.5 and 3.6 versions of the COARE algorithm.
"""

from .coare_35 import coare_35
from .coare_36 import coare_36

__all__ = [
    "coare_35",
    "coare_36",
]
