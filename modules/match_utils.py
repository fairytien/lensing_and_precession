import numpy as np


def cast_to_match_precision(arr: np.ndarray) -> np.ndarray:
    """Cast arrays to complex128 for stable matched filtering with PyCBC."""
    return np.asarray(arr, dtype=np.complex128)


def ensure_same_length(t: np.ndarray, s: np.ndarray) -> tuple:
    """Pad or truncate template to match source length; returns (t_fixed, s)."""
    if t.shape[0] == s.shape[0]:
        return t, s
    if t.shape[0] < s.shape[0]:
        pad = np.zeros((s.shape[0] - t.shape[0],), dtype=t.dtype)
        return np.concatenate([t, pad], axis=0), s
    return t[: s.shape[0]], s
