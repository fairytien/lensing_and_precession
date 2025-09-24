"""Utilities for matched filtering and worker processes."""

import numpy as np
import h5py
from typing import Tuple

from modules.functions_v3 import get_fcut_from_mcz, Sn, mismatch_from_strains


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


def build_psd_for_mcz(
    f_min: float,
    delta_f: float,
    mcz_msun: float,
) -> Tuple[np.ndarray, np.ndarray, float]:
    """
    Build frequency grid and PSD for a given mcz using provided helpers.
    Returns (s_farr, psd, f_cut).
    """
    f_cut = get_fcut_from_mcz(mcz_msun)
    s_farr = np.arange(f_min, f_cut, delta_f)
    psd = Sn(s_farr, f_min=f_min, delta_f=delta_f)
    return s_farr, psd, f_cut


def build_source_strain_for_td(
    get_gw_func, lens_params: dict, f_min: float, delta_f: float
) -> np.ndarray:
    """
    Compute source strain for given lens params and sampling settings.
    Returns complex strain array.
    """
    s = get_gw_func(lens_params, f_min=f_min, delta_f=delta_f)
    return s["strain"]


# Globals for worker processes (used by init_mismatch_worker/mismatch_gamma_job)
_S_STRAIN = None
_PSD = None
_DELTA_F = None
_COMPARE_BOTH = False
_USE_OPT_MATCH = True
_BANK_H5 = None
_BANK_DSET = None
_GAMMA_ARR = None
_GAMMA_CHUNK = None


def init_mismatch_worker(
    s_strain,
    psd,
    delta_f,
    compare_both,
    use_opt_match,
    bank_path,
    gamma_arr,
    gamma_chunk,
):
    """Initializer for multiprocessing workers used in mismatch computation.

    Sets module-level globals for the worker process, opens the HDF5 bank
    read-only, and registers an atexit handler to close the file. The source
    strain is cast to complex128 to harmonize precision with PSD and templates.

    Args:
        s_strain: Complex frequency-domain source strain array.
        psd: Frequency-domain PSD array (compatible with delta_f and frequency grid).
        delta_f: Frequency spacing in Hz.
        compare_both: If True, compare both lensed parities when computing mismatch.
        use_opt_match: If True, use optimal match configuration in mismatch routine.
        bank_path: Path to the HDF5 bank file to read templates from.
        gamma_arr: 1D array of gamma values corresponding to bank axis 2.
        gamma_chunk: Optional int for gamma tiling size per worker iteration.
    """
    import atexit

    global _S_STRAIN, _PSD, _DELTA_F, _COMPARE_BOTH, _USE_OPT_MATCH, _BANK_H5, _BANK_DSET, _GAMMA_ARR, _GAMMA_CHUNK
    _S_STRAIN = cast_to_match_precision(s_strain)
    _PSD = psd
    _DELTA_F = delta_f
    _COMPARE_BOTH = bool(compare_both)
    _USE_OPT_MATCH = bool(use_opt_match)
    _BANK_H5 = h5py.File(bank_path, "r")
    _BANK_DSET = _BANK_H5["bank"]
    _GAMMA_ARR = gamma_arr
    _GAMMA_CHUNK = int(gamma_chunk) if gamma_chunk is not None else None
    atexit.register(lambda: _BANK_H5.close())


def mismatch_gamma_job(args: tuple) -> tuple:
    """Compute mismatches for a single (theta=row=r, omega=col=c) across gamma.

    Expects that `init_mismatch_worker` was called in this worker to populate
    globals: source strain, PSD, delta_f, bank handles, gamma chunk size, etc.

    Args:
        args: Tuple (r, c) of integer indices into (theta, omega) axes.

    Returns:
        Tuple (r, c, ep_vec, best_ep, best_gamma) where:
            - ep_vec: 1D float32 array of mismatches for all gamma at (r, c)
            - best_ep: minimal mismatch value over gamma (float)
            - best_gamma: gamma value achieving minimal mismatch (float)
    """
    r, c = args
    n_gamma = _BANK_DSET.shape[2]
    ep_vec = np.empty(n_gamma, dtype=np.float32)
    best_ep = np.inf
    best_gamma = 0.0
    chunk = _GAMMA_CHUNK or max(1, min(32, n_gamma))
    for k0 in range(0, n_gamma, chunk):
        k1 = min(n_gamma, k0 + chunk)
        gamma_block = _BANK_DSET[int(r), int(c), k0:k1, :]  # shape (g, n_freq)
        gamma_block = cast_to_match_precision(gamma_block)
        for local_idx in range(gamma_block.shape[0]):
            k = k0 + local_idx
            t_arr = gamma_block[local_idx]
            t_arr, _ = ensure_same_length(t_arr, _S_STRAIN)
            res = mismatch_from_strains(
                t_arr,
                _S_STRAIN,
                f_min=20.0,
                delta_f=_DELTA_F,
                psd=_PSD,
                use_opt_match=_USE_OPT_MATCH,
                compare_both=_COMPARE_BOTH,
            )
            ep = float(res["mismatch"])
            ep_vec[k] = ep
            if ep < best_ep:
                best_ep = ep
                best_gamma = float(_GAMMA_ARR[k])
    return int(r), int(c), ep_vec, float(best_ep), float(best_gamma)
