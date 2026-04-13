"""Data-cleaning helpers for waveform utilities.

This module is a source-of-truth implementation for numeric cleanup helpers
formerly kept in `functions_v3.py`.
"""

from typing import List, Tuple, Union

import numpy as np


def omit_numerical_errors(
    arr: Union[np.ndarray, List, Tuple], n: int = 16, order: float = 1.5
) -> np.ndarray:
    """Replace large outliers relative to a rolling median with NaN."""
    if n % 2 == 0:
        raise ValueError("n must be odd for symmetric neighbor selection")
    if n < 3:
        raise ValueError("n must be at least 3")

    arr_copy = np.asarray(arr, dtype=float)
    if len(arr_copy) < n:
        raise ValueError(f"Array length {len(arr_copy)} must be >= n={n}")

    try:
        from scipy.signal import medfilt

        median_values = medfilt(arr_copy, kernel_size=n)
    except ImportError:
        median_values = _rolling_median_fallback(arr_copy, n)

    threshold = order * median_values
    arr_copy[arr_copy > threshold] = np.nan

    return arr_copy


def _rolling_median_fallback(arr: np.ndarray, n: int) -> np.ndarray:
    """Fallback rolling median implementation when scipy is unavailable."""
    half_window = n // 2
    result = np.empty_like(arr)

    for i in range(len(arr)):
        start = max(0, i - half_window)
        end = min(len(arr), i + half_window + 1)
        window = arr[start:end]
        result[i] = np.nanmedian(window)

    return result
