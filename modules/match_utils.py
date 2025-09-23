import numpy as np
import h5py
from typing import Callable, Tuple

from modules.functions_v3 import mismatch_from_strains


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
    Sn_func: Callable,
    f_min: float,
    delta_f: float,
    mcz_msun: float,
    fcut_func: Callable[[float], float],
) -> Tuple[np.ndarray, np.ndarray, float]:
    """
    Build frequency grid and PSD for a given mcz using provided helpers.
    Returns (s_f, psd, f_cut).
    """
    f_cut = float(fcut_func(mcz_msun))
    s_f = np.arange(f_min, f_cut, delta_f)
    psd = Sn_func(s_f, f_min=f_min, delta_f=delta_f)
    return s_f, psd, f_cut


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
