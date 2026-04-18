"""Compatibility wrapper for legacy imports.

The canonical production implementation now lives in the canonical modules.
"""

from modules.Classes import *  # noqa: F401,F403
from modules.default_params import *  # noqa: F401,F403
from modules.geometry import *  # noqa: F401,F403
from modules.match_utils import (
    find_optimized_coalescence_params,
    mismatch_from_params,
    mismatch_from_strains,
    optimize_mismatch_gammaP,
    optimize_mismatch_mcz,
    optimized_match_bounded,
)
from modules.numerics import *  # noqa: F401,F403
from modules.runtime_helpers import *  # noqa: F401,F403
from modules.snr import *  # noqa: F401,F403
from modules.waveform import *  # noqa: F401,F403

__all__ = [name for name in globals() if not name.startswith("_")]
