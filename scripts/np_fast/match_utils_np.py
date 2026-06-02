"""NP-specific match utilities for the scripts/np_fast sandbox.

Contains mismatch_block_serial — a serial template sweep function
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


def mismatch_block_serial(
    template_block: np.ndarray,
    labels: np.ndarray,
    s_strain: Union[np.ndarray, FrequencySeries],
    psd: Optional[PsdLike],
    f_min: float,
    delta_f: float,
    match_method: MatchMethod = MatchMethod.OPTIMIZED_BOUNDED,
) -> Tuple[np.ndarray, float, float]:
    """Serial evaluation of mismatches for a block of templates against a source.

    Args:
        template_block: 2D array of templates (shape: n_templates x n_freq).
        labels: 1D array of labels corresponding to the templates (e.g. gamma_P, template mcz).
        s_strain: Source strain.
        psd: Detector noise PSD.
        f_min: Minimum frequency cutoff.
        delta_f: Frequency spacing in Hz.
        match_method: Matching method to use.

    Returns:
        Tuple (ep_vec, best_ep, best_label)
    """
    templates = cast_to_match_precision(np.asarray(template_block))
    if templates.ndim == 1:
        templates = templates.reshape(1, -1)

    label_vals = np.asarray(labels, dtype=float).reshape(-1)
    if templates.shape[0] != label_vals.shape[0]:
        raise ValueError(
            "Template block row count and label array must have the same length. "
            f"Got templates={templates.shape[0]}, labels={label_vals.shape[0]}."
        )

    # Pre-convert source strain to FrequencySeries once outside the loop
    if not isinstance(s_strain, FrequencySeries):
        s_series = FrequencySeries(cast_to_match_precision(np.asarray(s_strain)), delta_f)
    else:
        s_series = s_strain

    # Pre-resolve and convert PSD to FrequencySeries once outside the loop
    if psd is not None and not isinstance(psd, FrequencySeries):
        psd_series = FrequencySeries(psd, delta_f)
    else:
        psd_series = psd

    ep_vec = np.empty(templates.shape[0], dtype=np.float32)
    best_ep = np.inf
    best_label = float(label_vals[0]) if label_vals.size else 0.0

    for idx, template in enumerate(templates):
        # ensure_same_length handles length mismatch on raw numpy arrays
        t_arr, _ = ensure_same_length(template, s_series.numpy())
        t_series = FrequencySeries(t_arr, delta_f)

        res = mismatch_from_strains(
            t_series,
            s_series,
            f_min=f_min,
            delta_f=delta_f,
            psd=psd_series,
            match_method=match_method,
        )
        ep = float(res["mismatch"])
        ep_vec[idx] = ep
        if ep < best_ep:
            best_ep = ep
            best_label = float(label_vals[idx])

    return ep_vec, float(best_ep), float(best_label)


def precompute_lensing_factors(
    lens_params: dict,
    y: float,
    f_array: np.ndarray,
) -> Tuple[np.ndarray, float, float]:
    """Precompute the unlensed waveform and magnification factors for a source.

    Args:
        lens_params: Dictionary of lens parameters.
        y: Source position.
        f_array: Frequency array.

    Returns:
        Tuple (h_I, sqrt_mu_p, sqrt_mu_m) where:
            - h_I: Unlensed waveform strain array.
            - sqrt_mu_p: Square root of plus magnification.
            - sqrt_mu_m: Square root of minus magnification.
    """
    from modules.waveform import LensingGeo, SOLMASS2SEC

    params_copy = dict(lens_params)
    params_copy["y"] = y
    params_copy["MLz"] = 1.0 * SOLMASS2SEC  # dummy MLz to satisfy initialization
    unlensed_wf = LensingGeo(params_copy)
    h_I = unlensed_wf.hI(f_array)

    mu_plus = unlensed_wf.mu_plus()
    mu_minus = unlensed_wf.mu_minus()
    sqrt_mu_p = np.sqrt(np.abs(mu_plus))
    sqrt_mu_m = np.sqrt(np.abs(mu_minus))

    return h_I, sqrt_mu_p, sqrt_mu_m


def build_lensed_source_strain(
    h_I: np.ndarray,
    sqrt_mu_p: float,
    sqrt_mu_m: float,
    f_array: np.ndarray,
    td: float,
    delta_f: float,
) -> FrequencySeries:
    """Algebraically construct lensed source strain from unlensed waveform.

    Args:
        h_I: Unlensed waveform strain array.
        sqrt_mu_p: Square root of plus magnification.
        sqrt_mu_m: Square root of minus magnification.
        f_array: Frequency array in Hz.
        td: Time delay in seconds.
        delta_f: Frequency spacing in Hz.

    Returns:
        FrequencySeries representing the lensed strain.
    """
    F_val = sqrt_mu_p - 1j * sqrt_mu_m * np.exp(2j * np.pi * f_array * td)
    return FrequencySeries(h_I * F_val, delta_f)



