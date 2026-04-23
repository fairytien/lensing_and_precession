"""Facade exports for waveform utility helpers.

Source-of-truth implementations are split into specialized modules:
- modules/waveform.py
- modules/numerics.py
- modules/geometry.py
- modules/snr.py

Matching logic lives in modules/match_utils.py as the single source of truth.

This facade preserves historical import paths (`modules.functions`).
"""

from modules.Classes import LensingGeo, Precessing
from modules.default_params import *  # noqa: F401,F403

from modules.waveform import *  # noqa: F401,F403
from modules.numerics import *  # noqa: F401,F403
from modules.geometry import *  # noqa: F401,F403
from modules.snr import *  # noqa: F401,F403
from modules.runtime_helpers import *  # noqa: F401,F403
from modules.match_utils import (
    find_optimized_coalescence_params,
    mismatch_from_params,
    mismatch_from_strains,
    optimize_mismatch_gammaP,
    optimize_mismatch_mcz,
    optimized_match_bounded,
)
