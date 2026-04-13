"""Utilities for matched filtering and worker processes."""

import numpy as np
import h5py
from typing import Tuple, Union, List

from scipy.optimize import minimize_scalar
from pycbc.filter import match, optimized_match
from pycbc.filter.matchedfilter import (
    make_frequency_series,
    get_cutoff_indices,
    sigmasq,
)
from pycbc.types import FrequencySeries


#############################
# Section 1: Core Match API #
#############################


def optimized_match_bounded(
    vec1,
    vec2,
    psd=None,
    low_frequency_cutoff=None,
    high_frequency_cutoff=None,
    v1_norm=None,
    v2_norm=None,
    return_phase=False,
):
    """Optimized match wrapper using SciPy bounded minimization.

    Uses bounds=(-delta_t, delta_t) for sub-sample time-shift refinement.
    If bounded minimization fails, falls back to local PyCBC optimized_match
    (typically brent in unmodified installs).
    """

    htilde = make_frequency_series(vec1)
    stilde = make_frequency_series(vec2)

    assert np.isclose(htilde.delta_f, stilde.delta_f)
    delta_f = stilde.delta_f

    assert np.isclose(htilde.delta_t, stilde.delta_t)
    delta_t = stilde.delta_t

    try:
        _, max_id, _ = match(
            htilde,
            stilde,
            psd=psd,
            low_frequency_cutoff=low_frequency_cutoff,
            high_frequency_cutoff=high_frequency_cutoff,
            return_phase=True,
        )

        stilde_shifted = stilde.cyclic_time_shift(-max_id * delta_t)

        frequencies = stilde_shifted.sample_frequencies.numpy()
        waveform_1 = htilde.numpy()
        waveform_2 = stilde_shifted.numpy()

        N = (len(stilde_shifted) - 1) * 2
        kmin, kmax = get_cutoff_indices(
            low_frequency_cutoff, high_frequency_cutoff, delta_f, N
        )
        mask = slice(kmin, kmax)

        waveform_1 = waveform_1[mask]
        waveform_2 = waveform_2[mask]
        frequencies = frequencies[mask]

        if psd is not None:
            psd_arr = psd.numpy()[mask]
        else:
            psd_arr = np.ones_like(waveform_1)

        def product(a, b):
            integral = np.sum(np.conj(a) * b / psd_arr) * delta_f
            return 4 * abs(integral), np.angle(integral)

        def product_offset(dt):
            offset = np.exp(2j * np.pi * frequencies * dt)
            return product(waveform_1, waveform_2 * offset)

        def to_minimize(dt):
            return -product_offset(dt)[0]

        norm_1 = (
            sigmasq(htilde, psd, low_frequency_cutoff, high_frequency_cutoff)
            if v1_norm is None
            else v1_norm
        )
        norm_2 = (
            sigmasq(stilde_shifted, psd, low_frequency_cutoff, high_frequency_cutoff)
            if v2_norm is None
            else v2_norm
        )
        norm = np.sqrt(norm_1 * norm_2)

        res = minimize_scalar(
            to_minimize,
            method="bounded",
            bounds=(-delta_t, delta_t),
        )
        m, angle = product_offset(res.x)

        if return_phase:
            return m / norm, res.x / delta_t + max_id, -angle
        return m / norm, res.x / delta_t + max_id

    except Exception:
        return optimized_match(
            vec1,
            vec2,
            psd=psd,
            low_frequency_cutoff=low_frequency_cutoff,
            high_frequency_cutoff=high_frequency_cutoff,
            v1_norm=v1_norm,
            v2_norm=v2_norm,
            return_phase=return_phase,
        )


#################################
# Section 2: Mismatch Utilities #
#################################


def mismatch_from_strains(
    t_strain: Union[np.ndarray, FrequencySeries],
    s_strain: Union[np.ndarray, FrequencySeries],
    f_min: float = 20,
    delta_f: float = 0.25,
    psd: FrequencySeries = None,
    use_opt_match=True,
    compare_both=False,
    sn_func=None,
) -> dict:
    """Compute mismatch between two strains with optional bounded optimization."""

    if sn_func is None:
        from modules.functions import Sn as sn_func

    if not isinstance(t_strain, FrequencySeries):
        t_strain = FrequencySeries(t_strain, delta_f)
    if not isinstance(s_strain, FrequencySeries):
        s_strain = FrequencySeries(s_strain, delta_f)

    if psd is None:
        f_arr = s_strain.sample_frequencies + f_min
        psd = sn_func(f_arr, f_min=f_min, delta_f=delta_f)

    if compare_both:
        results = []
        for func, name in zip([match, optimized_match], ["match", "optimized_match"]):
            try:
                match_val, index, phi = func(t_strain, s_strain, psd, return_phase=True)  # type: ignore
                results.append(
                    {
                        "mismatch": 1 - match_val,
                        "index": index,
                        "phi": phi,
                        "match_val": match_val,
                        "match_method": name,
                    }
                )
            except Exception:
                continue
        if not results:
            raise RuntimeError("Both match and optimized_match failed.")
        best_result = max(results, key=lambda x: x["match_val"])
        return {k: v for k, v in best_result.items() if k != "match_val"}

    if use_opt_match:
        match_val, index, phi = optimized_match_bounded(  # type: ignore
            t_strain, s_strain, psd=psd, return_phase=True
        )
    else:
        match_val, index, phi = match(  # type: ignore
            t_strain, s_strain, psd, return_phase=True
        )

    return {"mismatch": 1 - match_val, "index": index, "phi": phi}


def mismatch_from_params(
    t_params: dict,
    s_params: dict,
    f_min: float = 20,
    delta_f: float = 0.25,
    psd: FrequencySeries = None,
    lens_Class=None,
    prec_Class=None,
    use_opt_match=True,
    compare_both=False,
    get_gw_func=None,
    sn_func=None,
) -> dict:
    """Compute mismatch between two parameter dictionaries."""

    if (
        get_gw_func is None
        or sn_func is None
        or lens_Class is None
        or prec_Class is None
    ):
        from modules.functions import (
            get_gw as _get_gw,
            Sn as _Sn,
            LensingGeo,
            Precessing,
        )

        if get_gw_func is None:
            get_gw_func = _get_gw
        if sn_func is None:
            sn_func = _Sn
        if lens_Class is None:
            lens_Class = LensingGeo
        if prec_Class is None:
            prec_Class = Precessing

    t_gw = get_gw_func(t_params, f_min, delta_f, lens_Class, prec_Class)
    t_h = t_gw["strain"]
    s_gw = get_gw_func(s_params, f_min, delta_f, lens_Class, prec_Class)
    s_h = s_gw["strain"]
    t_h.resize(len(s_h))

    if psd is None:
        f_arr = s_gw["f_array"]
        psd = sn_func(f_arr, f_min=f_min, delta_f=delta_f)

    return mismatch_from_strains(
        t_h,
        s_h,
        f_min=f_min,
        delta_f=delta_f,
        psd=psd,
        use_opt_match=use_opt_match,
        compare_both=compare_both,
        sn_func=sn_func,
    )


#####################################
# Section 3: Mismatch Optimizations #
#####################################


def optimize_mismatch_mcz(
    t_params: dict,
    s_params: dict,
    f_min: float = 20,
    delta_f: float = 0.25,
    psd: FrequencySeries = None,
    lens_Class=None,
    prec_Class=None,
    use_opt_match=True,
    compare_both=False,
    set_to_params_func=None,
    get_gw_func=None,
    sn_func=None,
    mismatch_from_strains_func=None,
    solmass2sec=None,
) -> dict:
    """Optimize mismatch over template chirp mass around source chirp mass."""

    if (
        set_to_params_func is None
        or get_gw_func is None
        or sn_func is None
        or mismatch_from_strains_func is None
        or solmass2sec is None
        or lens_Class is None
        or prec_Class is None
    ):
        from modules.functions import (
            set_to_params as _set_to_params,
            get_gw as _get_gw,
            Sn as _Sn,
            mismatch_from_strains as _mismatch_from_strains,
            SOLMASS2SEC as _SOLMASS2SEC,
            LensingGeo,
            Precessing,
        )

        if set_to_params_func is None:
            set_to_params_func = _set_to_params
        if get_gw_func is None:
            get_gw_func = _get_gw
        if sn_func is None:
            sn_func = _Sn
        if mismatch_from_strains_func is None:
            mismatch_from_strains_func = _mismatch_from_strains
        if solmass2sec is None:
            solmass2sec = _SOLMASS2SEC
        if lens_Class is None:
            lens_Class = LensingGeo
        if prec_Class is None:
            prec_Class = Precessing

    t_params_copy, s_params_copy = set_to_params_func(t_params, s_params)

    n_pts = 101
    mcz_src_msun = s_params_copy["mcz"] / solmass2sec
    mcz_arr_msun = np.linspace(mcz_src_msun - 1, mcz_src_msun + 1, n_pts)

    s_gw = get_gw_func(s_params_copy, f_min, delta_f, lens_Class, prec_Class)
    s_h = s_gw["strain"]
    f_arr = s_gw["f_array"]
    if psd is None:
        psd = sn_func(f_arr, f_min=f_min, delta_f=delta_f)

    ep_arr = np.empty(n_pts, dtype=float)
    idx_arr = np.empty(n_pts, dtype=int)
    phi_arr = np.empty(n_pts, dtype=float)

    for i, mcz in enumerate(mcz_arr_msun):
        t_params_i = {**t_params_copy, "mcz": float(mcz) * solmass2sec}
        t_h = get_gw_func(t_params_i, f_min, delta_f, lens_Class, prec_Class)["strain"]
        if len(t_h) != len(s_h):
            t_h.resize(len(s_h))

        res = mismatch_from_strains_func(
            t_h,
            s_h,
            f_min=f_min,
            delta_f=delta_f,
            psd=psd,
            use_opt_match=use_opt_match,
            compare_both=compare_both,
        )
        ep_arr[i] = res["mismatch"]
        idx_arr[i] = res["index"]
        phi_arr[i] = res["phi"]

    ep_min_idx = np.argmin(ep_arr)
    return {
        "ep_min": ep_arr[ep_min_idx],
        "ep_min_mcz": mcz_arr_msun[ep_min_idx],
        "ep_min_idx": idx_arr[ep_min_idx],
        "ep_min_phi": phi_arr[ep_min_idx],
    }


def optimize_mismatch_gammaP(
    t_params: dict,
    s_params: dict = None,
    s_strain: FrequencySeries = None,
    f_min: float = 20,
    delta_f: float = 0.25,
    psd: FrequencySeries = None,
    lens_Class=None,
    prec_Class=None,
    use_opt_match=True,
    compare_both=False,
    grid_points: int = 101,
    gamma_grid: np.ndarray = None,
    two_stage=False,
    coarse_points: int = 17,
    xatol: float = 1e-3,
    maxiter: int = 50,
    set_to_params_func=None,
    get_gw_func=None,
    sn_func=None,
    mismatch_from_strains_func=None,
) -> dict:
    """Optimize mismatch over template precession phase gamma_P."""

    if (
        set_to_params_func is None
        or get_gw_func is None
        or sn_func is None
        or mismatch_from_strains_func is None
        or lens_Class is None
        or prec_Class is None
    ):
        from modules.functions import (
            set_to_params as _set_to_params,
            get_gw as _get_gw,
            Sn as _Sn,
            mismatch_from_strains as _mismatch_from_strains,
            LensingGeo,
            Precessing,
        )

        if set_to_params_func is None:
            set_to_params_func = _set_to_params
        if get_gw_func is None:
            get_gw_func = _get_gw
        if sn_func is None:
            sn_func = _Sn
        if mismatch_from_strains_func is None:
            mismatch_from_strains_func = _mismatch_from_strains
        if lens_Class is None:
            lens_Class = LensingGeo
        if prec_Class is None:
            prec_Class = Precessing

    if s_strain is None and s_params is None:
        raise ValueError("Either s_params or s_strain must be provided")

    if "gamma_P" not in t_params:
        raise ValueError("t_params must contain gamma_P")

    t_params_copy = set_to_params_func(t_params)[0]

    if s_strain is None:
        s_params_copy = set_to_params_func(s_params)[0]
        s_gw = get_gw_func(s_params_copy, f_min, delta_f, lens_Class, prec_Class)
        s_strain_local = s_gw["strain"]
        f_arr = s_gw["f_array"]
        psd_local = (
            psd if psd is not None else sn_func(f_arr, f_min=f_min, delta_f=delta_f)
        )
    else:
        s_strain_local = s_strain
        if psd is not None:
            psd_local = psd
        else:
            f_arr = s_strain_local.sample_frequencies + f_min
            psd_local = sn_func(f_arr, f_min=f_min, delta_f=s_strain_local.delta_f)

    if not two_stage:
        if gamma_grid is not None:
            gamma_arr = np.asarray(gamma_grid, dtype=float)
            gamma_arr = np.mod(gamma_arr, 2 * np.pi)
        else:
            gamma_arr = np.linspace(0, 2 * np.pi, int(grid_points), endpoint=False)

        n_gam = len(gamma_arr)
        ep_arr = np.empty(n_gam, dtype=float)
        idx_arr = np.empty(n_gam, dtype=int)
        phi_arr = np.empty(n_gam, dtype=float)

        for i, gamma_P in enumerate(gamma_arr):
            t_params_i = {**t_params_copy, "gamma_P": float(gamma_P)}
            t_strain = get_gw_func(t_params_i, f_min, delta_f, lens_Class, prec_Class)[
                "strain"
            ]
            if len(t_strain) != len(s_strain_local):
                t_strain.resize(len(s_strain_local))

            res = mismatch_from_strains_func(
                t_strain,
                s_strain_local,
                f_min=f_min,
                delta_f=delta_f,
                psd=psd_local,
                use_opt_match=use_opt_match,
                compare_both=compare_both,
            )
            ep_arr[i] = res["mismatch"]
            idx_arr[i] = res["index"]
            phi_arr[i] = res["phi"]

        ep_min_idx = np.argmin(ep_arr)
        return {
            "ep_min": float(ep_arr[ep_min_idx]),
            "ep_min_gammaP": float(gamma_arr[ep_min_idx]),
            "ep_min_idx": int(idx_arr[ep_min_idx]),
            "ep_min_phi": float(phi_arr[ep_min_idx]),
        }

    if coarse_points < 3:
        raise ValueError("coarse_points must be >= 3 for two-stage search")

    gamma_coarse = np.linspace(0, 2 * np.pi, int(coarse_points), endpoint=False)
    ep_coarse = np.empty_like(gamma_coarse)

    def objective(gamma_val: float) -> float:
        gamma = float(np.mod(gamma_val, 2 * np.pi))
        t_params_i = {**t_params_copy, "gamma_P": gamma}
        t_strain_local = get_gw_func(
            t_params_i, f_min, delta_f, lens_Class, prec_Class
        )["strain"]
        if len(t_strain_local) != len(s_strain_local):
            t_strain_local.resize(len(s_strain_local))
        res_local = mismatch_from_strains_func(
            t_strain_local,
            s_strain_local,
            f_min=f_min,
            delta_f=delta_f,
            psd=psd_local,
            use_opt_match=use_opt_match,
            compare_both=compare_both,
        )
        return float(res_local["mismatch"])

    for i, g in enumerate(gamma_coarse):
        ep_coarse[i] = objective(float(g))

    best_idx = int(np.argmin(ep_coarse))
    g0 = float(gamma_coarse[best_idx])
    half_width = (2 * np.pi) / float(coarse_points)
    a = g0 - half_width
    b = g0 + half_width

    segments: List[Tuple[float, float]]
    if a < 0:
        segments = [(0.0, b), (2 * np.pi + a, 2 * np.pi)]
    elif b > 2 * np.pi:
        segments = [(0.0, b - 2 * np.pi), (a, 2 * np.pi)]
    else:
        segments = [(a, b)]

    best_fun = np.inf
    best_x = g0
    for lo, hi in segments:
        res = minimize_scalar(
            objective,
            method="bounded",
            bounds=(lo, hi),
            options={"xatol": float(xatol), "maxiter": int(maxiter)},
        )
        if res.fun < best_fun:
            best_fun = float(res.fun)
            best_x = float(res.x)

    gamma_star = float(np.mod(best_x, 2 * np.pi))
    t_params_star = {**t_params_copy, "gamma_P": gamma_star}
    t_strain_star = get_gw_func(t_params_star, f_min, delta_f, lens_Class, prec_Class)[
        "strain"
    ]
    if len(t_strain_star) != len(s_strain_local):
        t_strain_star.resize(len(s_strain_local))
    res_star = mismatch_from_strains_func(
        t_strain_star,
        s_strain_local,
        f_min=f_min,
        delta_f=delta_f,
        psd=psd_local,
        use_opt_match=use_opt_match,
        compare_both=compare_both,
    )

    return {
        "ep_min": float(res_star["mismatch"]),
        "ep_min_gammaP": gamma_star,
        "ep_min_idx": int(res_star["index"]),
        "ep_min_phi": float(res_star["phi"]),
    }


def find_optimized_coalescence_params(
    t_params: dict,
    s_params: dict,
    f_min: float = 20,
    delta_f: float = 0.25,
    psd: FrequencySeries = None,
    lens_Class=None,
    prec_Class=None,
    use_opt_match=True,
    compare_both=False,
    optimize_gammaP=True,
    verify_optimization=False,
    set_to_params_func=None,
    get_fcut_from_mcz_func=None,
    sn_func=None,
    optimize_mismatch_gammaP_func=None,
    mismatch_from_params_func=None,
    get_gw_func=None,
    solmass2sec=None,
    **kwargs,
) -> dict:
    """Find optimal t_c/phi_c (and optionally gamma_P) for template params."""

    if (
        set_to_params_func is None
        or get_fcut_from_mcz_func is None
        or sn_func is None
        or optimize_mismatch_gammaP_func is None
        or mismatch_from_params_func is None
        or get_gw_func is None
        or solmass2sec is None
        or lens_Class is None
        or prec_Class is None
    ):
        from modules.functions import (
            set_to_params as _set_to_params,
            get_fcut_from_mcz as _get_fcut_from_mcz,
            Sn as _Sn,
            optimize_mismatch_gammaP as _optimize_mismatch_gammaP,
            mismatch_from_params as _mismatch_from_params,
            get_gw as _get_gw,
            SOLMASS2SEC as _SOLMASS2SEC,
            LensingGeo,
            Precessing,
        )

        if set_to_params_func is None:
            set_to_params_func = _set_to_params
        if get_fcut_from_mcz_func is None:
            get_fcut_from_mcz_func = _get_fcut_from_mcz
        if sn_func is None:
            sn_func = _Sn
        if optimize_mismatch_gammaP_func is None:
            optimize_mismatch_gammaP_func = _optimize_mismatch_gammaP
        if mismatch_from_params_func is None:
            mismatch_from_params_func = _mismatch_from_params
        if get_gw_func is None:
            get_gw_func = _get_gw
        if solmass2sec is None:
            solmass2sec = _SOLMASS2SEC
        if lens_Class is None:
            lens_Class = LensingGeo
        if prec_Class is None:
            prec_Class = Precessing

    t_params_copy, s_params_copy = set_to_params_func(t_params, s_params)

    if psd is None:
        f_cut = get_fcut_from_mcz_func(s_params_copy["mcz"] / solmass2sec)
        f_arr = np.arange(f_min, f_cut, delta_f)
        psd = sn_func(f_arr, f_min=f_min, delta_f=delta_f)

    if optimize_gammaP:
        if "gamma_P" not in t_params:
            raise ValueError("t_params must contain gamma_P")

        gammaP_results = optimize_mismatch_gammaP_func(
            t_params_copy,
            s_params_copy,
            f_min=f_min,
            delta_f=delta_f,
            psd=psd,
            lens_Class=lens_Class,
            prec_Class=prec_Class,
            use_opt_match=use_opt_match,
            compare_both=compare_both,
            **kwargs,
        )
        ep_min_gammaP = gammaP_results["ep_min_gammaP"]
        t_params_copy["gamma_P"] = ep_min_gammaP

        ep_min_idx = gammaP_results["ep_min_idx"]
        src_strain = get_gw_func(s_params_copy, f_min, delta_f, lens_Class, prec_Class)[
            "strain"
        ]
        delta_t = src_strain.delta_t
        t_params_copy["t_c"] = t_params_copy["t_c"] - ep_min_idx * delta_t
    else:
        ep_min_gammaP = None
        initial_mismatch = mismatch_from_params_func(
            t_params_copy,
            s_params_copy,
            f_min,
            delta_f,
            psd,
            lens_Class,
            prec_Class,
            use_opt_match,
            compare_both,
        )
        ep_min_idx = initial_mismatch["index"]
        src_strain = get_gw_func(s_params_copy, f_min, delta_f, lens_Class, prec_Class)[
            "strain"
        ]
        delta_t = src_strain.delta_t
        t_params_copy["t_c"] = t_params_copy["t_c"] - ep_min_idx * delta_t

    mismatch_results = mismatch_from_params_func(
        t_params_copy,
        s_params_copy,
        f_min,
        delta_f,
        psd,
        lens_Class,
        prec_Class,
        use_opt_match,
        compare_both,
    )
    phi = mismatch_results["phi"]
    t_params_copy["phi_c"] = phi

    if verify_optimization:
        mismatch_results = mismatch_from_params_func(
            t_params_copy,
            s_params_copy,
            f_min,
            delta_f,
            psd,
            lens_Class,
            prec_Class,
            use_opt_match,
            compare_both,
        )
        print(
            f"Verification results: index = {mismatch_results['index']:.3g}, phi = {mismatch_results['phi']:.3g}, both should be ~0 if optimization was successful"
        )

    return {
        "opt_t_params": t_params_copy,
        "ep_min": mismatch_results["mismatch"],
        "ep_min_idx": mismatch_results["index"],
        "ep_min_phi": mismatch_results["phi"],
        "ep_min_gammaP": ep_min_gammaP,
    }


##############################
# Section 4: Helper Routines #
##############################


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
    from modules.functions import get_fcut_from_mcz, Sn

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


###########################################
# Section 5: Multiprocessing Worker State #
###########################################


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
