"""NP-specific match utilities for the scripts/np_fast sandbox.

Contains mismatch_gamma_block_serial — a serial version of mismatch_gamma_job
designed for single-grid-cell (NP) banks where the multiprocessing pool
overhead is unnecessary.
"""

from typing import Optional, Tuple, Union

import numpy as np
from pycbc.types import FrequencySeries

from modules.match_utils import (
    MatchMethod,
    PsdLike,
    cast_to_match_precision,
    ensure_same_length,
    mismatch_from_strains,
)


def mismatch_gamma_block_serial(
    template_block: np.ndarray,
    gamma_arr: np.ndarray,
    s_strain: Union[np.ndarray, FrequencySeries],
    psd: Optional[PsdLike],
    f_min: float,
    delta_f: float,
    match_method: MatchMethod = MatchMethod.OPTIMIZED_BOUNDED,
) -> Tuple[np.ndarray, float, float]:
    """Serial equivalent of mismatch_gamma_job for one (theta, omega) bank slice."""

    templates = cast_to_match_precision(np.asarray(template_block))
    if templates.ndim == 1:
        templates = templates.reshape(1, -1)

    gamma_vals = np.asarray(gamma_arr, dtype=float).reshape(-1)
    if templates.shape[0] != gamma_vals.shape[0]:
        raise ValueError(
            "Template gamma axis and gamma_arr must have the same length. "
            f"Got templates={templates.shape[0]} gamma={gamma_vals.shape[0]}."
        )

    source = cast_to_match_precision(np.asarray(s_strain))
    ep_vec = np.empty(templates.shape[0], dtype=np.float32)
    best_ep = np.inf
    best_gamma = float(gamma_vals[0]) if gamma_vals.size else 0.0

    for idx, template in enumerate(templates):
        t_arr, s_arr = ensure_same_length(template, source)
        res = mismatch_from_strains(
            t_arr,
            s_arr,
            f_min=f_min,
            delta_f=delta_f,
            psd=psd,
            match_method=match_method,
        )
        ep = float(res["mismatch"])
        ep_vec[idx] = ep
        if ep < best_ep:
            best_ep = ep
            best_gamma = float(gamma_vals[idx])

    return ep_vec, float(best_ep), float(best_gamma)
