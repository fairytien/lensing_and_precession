"""Utilities for matched filtering and worker processes.

Sections:
- Dependency resolution and override wiring
- Shared frequency, PSD, and strain helpers
- Core match API wrappers
- Mismatch APIs (strain- and parameter-level)
- Mismatch optimization helpers
- Convenience helpers for arrays and setup
- Multiprocessing worker state and jobs

Design goals: preserve numerical behavior, keep call-sites explicit, and reduce
repeated boilerplate in optimization loops.
"""

import numpy as np
import h5py
from typing import Tuple, Union, List, Optional, cast

from scipy.optimize import OptimizeResult, minimize_scalar
from pycbc.filter import match, optimized_match
from pycbc.filter.matchedfilter import (
    make_frequency_series,
    get_cutoff_indices,
    sigmasq,
)

from pycbc.types import FrequencySeries


PsdLike = Union[np.ndarray, FrequencySeries]


# ============================================================================
# Dependency Resolution And Override Wiring
# ============================================================================


def _resolve_deps(**overrides):
    """Fill None-valued overrides with lazy imports from source-of-truth modules."""
    if all(v is not None for v in overrides.values()):
        return overrides
    from modules.Classes import LensingGeo, Precessing
    from modules.default_params import SOLMASS2SEC
    from modules.snr import Sn
    from modules.waveform import set_to_params, get_fcut_from_mcz, get_gw

    _defaults = {
        "set_to_params_func": set_to_params,
        "get_fcut_from_mcz_func": get_fcut_from_mcz,
        "sn_func": Sn,
        "optimize_mismatch_gammaP_func": optimize_mismatch_gammaP,
        "mismatch_from_params_func": mismatch_from_params,
        "mismatch_from_strains_func": mismatch_from_strains,
        "get_gw_func": get_gw,
        "solmass2sec": SOLMASS2SEC,
        "lens_Class": LensingGeo,
        "prec_Class": Precessing,
    }
    return {k: v if v is not None else _defaults[k] for k, v in overrides.items()}


# ============================================================================
# Shared Frequency, PSD, And Strain Helpers
# ============================================================================


def _build_frequency_grid(f_min: float, delta_f: float, f_cut: float) -> np.ndarray:
    """Build the standard half-open frequency grid used by match helpers."""
    return np.arange(f_min, f_cut, delta_f)


def _resolve_psd_from_frequency_array(
    psd: Optional[PsdLike],
    f_arr: np.ndarray,
    f_min: float,
    delta_f: float,
    sn_func,
) -> PsdLike:
    """Reuse a provided PSD or build one from a supplied frequency grid."""
    if psd is not None:
        return psd
    return sn_func(f_arr, f_min=f_min, delta_f=delta_f)


def _resolve_psd_from_strain(
    psd: Optional[PsdLike],
    strain: FrequencySeries,
    f_min: float,
    sn_func,
    delta_f: Optional[float] = None,
) -> PsdLike:
    """Reuse a provided PSD or derive one from a strain's sample grid."""
    if psd is not None:
        return psd
    f_arr = strain.sample_frequencies + f_min
    psd_delta_f = strain.delta_f if delta_f is None else delta_f
    return sn_func(f_arr, f_min=f_min, delta_f=psd_delta_f)


def _resize_frequency_series_like(
    strain: FrequencySeries, reference: FrequencySeries
) -> FrequencySeries:
    """Resize a FrequencySeries in place to match a reference length.

    This helper is for PyCBC FrequencySeries objects produced by waveform
    generation. It preserves that code path's object type and mutates the
    series only when its length differs from the reference.
    """
    if len(strain) != len(reference):
        strain.resize(len(reference))
    return strain


def _build_template_strain_like_source(
    get_gw_func,
    t_params: dict,
    f_min: float,
    delta_f: float,
    lens_Class,
    prec_Class,
    s_strain: FrequencySeries,
) -> FrequencySeries:
    """Build template strain and resize to match a source strain length."""
    t_strain = get_gw_func(t_params, f_min, delta_f, lens_Class, prec_Class)["strain"]
    return _resize_frequency_series_like(t_strain, s_strain)


# ============================================================================
# Core Match API Wrappers
# ============================================================================


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
        _, max_id, _ = cast(
            Tuple[float, float, float],
            match(
                htilde,
                stilde,
                psd=psd,
                low_frequency_cutoff=low_frequency_cutoff,
                high_frequency_cutoff=high_frequency_cutoff,
                return_phase=True,
            ),
        )

        stilde_shifted = cast(
            FrequencySeries, stilde.cyclic_time_shift(-max_id * delta_t)
        )

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

        res = cast(
            OptimizeResult,
            minimize_scalar(
                to_minimize,
                method="bounded",
                bounds=(-delta_t, delta_t),
            ),
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


# ============================================================================
# Mismatch APIs
# ============================================================================


def mismatch_from_strains(
    t_strain: Union[np.ndarray, FrequencySeries],
    s_strain: Union[np.ndarray, FrequencySeries],
    f_min: float = 20,
    delta_f: float = 0.25,
    psd: Optional[PsdLike] = None,
    use_opt_match=True,
    compare_both=False,
    sn_func=None,
) -> dict:
    """Compute mismatch between two strains with optional bounded optimization."""

    if sn_func is None:
        from modules.snr import Sn as sn_func

    if not isinstance(t_strain, FrequencySeries):
        t_strain = FrequencySeries(t_strain, delta_f)
    if not isinstance(s_strain, FrequencySeries):
        s_strain = FrequencySeries(s_strain, delta_f)

    psd = _resolve_psd_from_strain(
        psd,
        s_strain,
        f_min=f_min,
        sn_func=sn_func,
        delta_f=delta_f,
    )

    if compare_both:
        results = []
        for func, name in zip(
            [match, optimized_match_bounded],
            ["match", "optimized_match_bounded"],
        ):
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
            raise RuntimeError("Both match and optimized_match_bounded failed.")
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
    psd: Optional[PsdLike] = None,
    lens_Class=None,
    prec_Class=None,
    use_opt_match=True,
    compare_both=False,
    get_gw_func=None,
    sn_func=None,
) -> dict:
    """Compute mismatch between two parameter dictionaries."""

    d = _resolve_deps(
        get_gw_func=get_gw_func,
        sn_func=sn_func,
        lens_Class=lens_Class,
        prec_Class=prec_Class,
    )
    get_gw_func = d["get_gw_func"]
    sn_func = d["sn_func"]
    lens_Class = d["lens_Class"]
    prec_Class = d["prec_Class"]

    t_gw = get_gw_func(t_params, f_min, delta_f, lens_Class, prec_Class)
    t_h = t_gw["strain"]
    s_gw = get_gw_func(s_params, f_min, delta_f, lens_Class, prec_Class)
    s_h = s_gw["strain"]
    t_h = _resize_frequency_series_like(t_h, s_h)

    psd = _resolve_psd_from_frequency_array(
        psd,
        s_gw["f_array"],
        f_min=f_min,
        delta_f=delta_f,
        sn_func=sn_func,
    )

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


# ============================================================================
# Mismatch Optimization Helpers
# ============================================================================


def optimize_mismatch_mcz(
    t_params: dict,
    s_params: dict,
    f_min: float = 20,
    delta_f: float = 0.25,
    psd: Optional[PsdLike] = None,
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

    d = _resolve_deps(
        set_to_params_func=set_to_params_func,
        get_gw_func=get_gw_func,
        sn_func=sn_func,
        mismatch_from_strains_func=mismatch_from_strains_func,
        solmass2sec=solmass2sec,
        lens_Class=lens_Class,
        prec_Class=prec_Class,
    )
    set_to_params_func = d["set_to_params_func"]
    get_gw_func = d["get_gw_func"]
    sn_func = d["sn_func"]
    mismatch_from_strains_func = d["mismatch_from_strains_func"]
    solmass2sec = d["solmass2sec"]
    lens_Class = d["lens_Class"]
    prec_Class = d["prec_Class"]

    t_params_copy, s_params_copy = set_to_params_func(t_params, s_params)

    n_pts = 101
    mcz_src_msun = s_params_copy["mcz"] / solmass2sec
    mcz_arr_msun = np.linspace(mcz_src_msun - 1, mcz_src_msun + 1, n_pts)

    s_gw = get_gw_func(s_params_copy, f_min, delta_f, lens_Class, prec_Class)
    s_h = s_gw["strain"]
    f_arr = s_gw["f_array"]
    psd = _resolve_psd_from_frequency_array(
        psd,
        f_arr,
        f_min=f_min,
        delta_f=delta_f,
        sn_func=sn_func,
    )

    ep_arr = np.empty(n_pts, dtype=float)
    idx_arr = np.empty(n_pts, dtype=int)
    phi_arr = np.empty(n_pts, dtype=float)

    for i, mcz in enumerate(mcz_arr_msun):
        t_params_i = {**t_params_copy, "mcz": float(mcz) * solmass2sec}
        t_h = _build_template_strain_like_source(
            get_gw_func,
            t_params_i,
            f_min,
            delta_f,
            lens_Class,
            prec_Class,
            s_h,
        )

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
    s_params: Optional[dict] = None,
    s_strain: Optional[FrequencySeries] = None,
    f_min: float = 20,
    delta_f: float = 0.25,
    psd: Optional[PsdLike] = None,
    lens_Class=None,
    prec_Class=None,
    use_opt_match=True,
    compare_both=False,
    grid_points: int = 101,
    gamma_grid: Optional[np.ndarray] = None,
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

    d = _resolve_deps(
        set_to_params_func=set_to_params_func,
        get_gw_func=get_gw_func,
        sn_func=sn_func,
        mismatch_from_strains_func=mismatch_from_strains_func,
        lens_Class=lens_Class,
        prec_Class=prec_Class,
    )
    set_to_params_func = d["set_to_params_func"]
    get_gw_func = d["get_gw_func"]
    sn_func = d["sn_func"]
    mismatch_from_strains_func = d["mismatch_from_strains_func"]
    lens_Class = d["lens_Class"]
    prec_Class = d["prec_Class"]

    if s_strain is None and s_params is None:
        raise ValueError("Either s_params or s_strain must be provided")

    if "gamma_P" not in t_params:
        raise ValueError("t_params must contain gamma_P")

    t_params_copy = set_to_params_func(t_params)[0]

    if s_strain is None:
        s_params_copy = set_to_params_func(s_params)[0]
        s_gw = get_gw_func(s_params_copy, f_min, delta_f, lens_Class, prec_Class)
        s_strain_local = s_gw["strain"]
        psd_local = _resolve_psd_from_frequency_array(
            psd,
            s_gw["f_array"],
            f_min=f_min,
            delta_f=delta_f,
            sn_func=sn_func,
        )
    else:
        s_strain_local = s_strain
        psd_local = _resolve_psd_from_strain(
            psd,
            s_strain_local,
            f_min=f_min,
            sn_func=sn_func,
        )

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
            t_strain = _build_template_strain_like_source(
                get_gw_func,
                t_params_i,
                f_min,
                delta_f,
                lens_Class,
                prec_Class,
                s_strain_local,
            )

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
        t_strain_local = _build_template_strain_like_source(
            get_gw_func,
            t_params_i,
            f_min,
            delta_f,
            lens_Class,
            prec_Class,
            s_strain_local,
        )
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
        res = cast(
            OptimizeResult,
            minimize_scalar(
                objective,
                method="bounded",
                bounds=(lo, hi),
                options={"xatol": float(xatol), "maxiter": int(maxiter)},
            ),
        )
        if res.fun < best_fun:
            best_fun = float(res.fun)
            best_x = float(res.x)

    gamma_star = float(np.mod(best_x, 2 * np.pi))
    t_params_star = {**t_params_copy, "gamma_P": gamma_star}
    t_strain_star = _build_template_strain_like_source(
        get_gw_func,
        t_params_star,
        f_min,
        delta_f,
        lens_Class,
        prec_Class,
        s_strain_local,
    )
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
    psd: Optional[PsdLike] = None,
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

    d = _resolve_deps(
        set_to_params_func=set_to_params_func,
        get_fcut_from_mcz_func=get_fcut_from_mcz_func,
        sn_func=sn_func,
        optimize_mismatch_gammaP_func=optimize_mismatch_gammaP_func,
        mismatch_from_params_func=mismatch_from_params_func,
        get_gw_func=get_gw_func,
        solmass2sec=solmass2sec,
        lens_Class=lens_Class,
        prec_Class=prec_Class,
    )
    set_to_params_func = d["set_to_params_func"]
    get_fcut_from_mcz_func = d["get_fcut_from_mcz_func"]
    sn_func = d["sn_func"]
    optimize_mismatch_gammaP_func = d["optimize_mismatch_gammaP_func"]
    mismatch_from_params_func = d["mismatch_from_params_func"]
    get_gw_func = d["get_gw_func"]
    solmass2sec = d["solmass2sec"]
    lens_Class = d["lens_Class"]
    prec_Class = d["prec_Class"]

    t_params_copy, s_params_copy = set_to_params_func(t_params, s_params)

    def _evaluate_current_mismatch() -> dict:
        return mismatch_from_params_func(
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

    if psd is None:
        f_cut = get_fcut_from_mcz_func(s_params_copy["mcz"] / solmass2sec)
        f_arr = _build_frequency_grid(f_min, delta_f, f_cut)
        psd = _resolve_psd_from_frequency_array(
            psd,
            f_arr,
            f_min=f_min,
            delta_f=delta_f,
            sn_func=sn_func,
        )

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
    else:
        ep_min_gammaP = None
        initial_mismatch = _evaluate_current_mismatch()
        ep_min_idx = initial_mismatch["index"]

    src_strain = get_gw_func(s_params_copy, f_min, delta_f, lens_Class, prec_Class)[
        "strain"
    ]
    delta_t = src_strain.delta_t
    ep_min_idx_wrapped = _wrap_match_index(ep_min_idx, len(src_strain))
    t_params_copy["t_c"] = t_params_copy["t_c"] - ep_min_idx_wrapped * delta_t

    mismatch_results = _evaluate_current_mismatch()
    phi = mismatch_results["phi"]
    t_params_copy["phi_c"] = phi

    if verify_optimization:
        mismatch_results = _evaluate_current_mismatch()
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


# ============================================================================
# Convenience Helpers For Arrays And Setup
# ============================================================================


# Array preparation helpers.


def cast_to_match_precision(arr: np.ndarray) -> np.ndarray:
    """Cast arrays to complex128 for stable matched filtering with PyCBC."""
    return np.asarray(arr, dtype=np.complex128)


def ensure_same_length(t: np.ndarray, s: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """Pad or truncate ndarray inputs to match the source length.

    This helper is for raw NumPy arrays, such as template-bank rows loaded
    from HDF5 in worker processes. Unlike _resize_frequency_series_like, it
    returns adjusted arrays rather than mutating a FrequencySeries in place.
    """
    if t.shape[0] == s.shape[0]:
        return t, s
    if t.shape[0] < s.shape[0]:
        pad = np.zeros((s.shape[0] - t.shape[0],), dtype=t.dtype)
        return np.concatenate([t, pad], axis=0), s
    return t[: s.shape[0]], s


def _wrap_match_index(index: float, n_freq: int) -> float:
    """Wrap cyclic match index to the minimal centered shift in samples."""
    n_time = max(1, 2 * (int(n_freq) - 1))
    return float((float(index) + n_time / 2.0) % n_time - n_time / 2.0)


# Match setup helpers.


def build_psd_for_mcz(
    f_min: float,
    delta_f: float,
    mcz_msun: float,
) -> Tuple[np.ndarray, PsdLike, float]:
    """
    Build the match-setup frequency grid and PSD for a given chirp mass.
    Returns (s_farr, psd, f_cut).
    """
    from modules.snr import Sn
    from modules.waveform import get_fcut_from_mcz

    f_cut = float(get_fcut_from_mcz(mcz_msun))
    s_farr = _build_frequency_grid(f_min, delta_f, f_cut)
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


# ============================================================================
# Multiprocessing Worker State And Jobs
# ============================================================================


# Globals for worker processes (used by init_mismatch_worker/mismatch_gamma_job)
_S_STRAIN: Optional[np.ndarray] = None
_PSD: Optional[PsdLike] = None
_DELTA_F: Optional[float] = None
_COMPARE_BOTH = False
_USE_OPT_MATCH = True
_BANK_H5: Optional[h5py.File] = None
_BANK_DSET: Optional[h5py.Dataset] = None
_GAMMA_ARR: Optional[np.ndarray] = None
_GAMMA_CHUNK: Optional[int] = None


def _require_worker_state() -> tuple:
    """Validate worker state and return initialized globals for job functions."""
    if (
        _S_STRAIN is None
        or _DELTA_F is None
        or _BANK_DSET is None
        or _GAMMA_ARR is None
    ):
        raise RuntimeError(
            "Worker state is not initialized. Call init_mismatch_worker first."
        )
    return (
        _S_STRAIN,
        _PSD,
        _DELTA_F,
        _COMPARE_BOTH,
        _USE_OPT_MATCH,
        _BANK_DSET,
        _GAMMA_ARR,
        _GAMMA_CHUNK,
    )


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
    bank_h5 = h5py.File(bank_path, "r")
    bank_obj = bank_h5["bank"]
    if not isinstance(bank_obj, h5py.Dataset):
        bank_h5.close()
        raise TypeError("Expected 'bank' to be an HDF5 dataset.")

    _S_STRAIN = cast_to_match_precision(s_strain)
    _PSD = psd
    _DELTA_F = float(delta_f)
    _COMPARE_BOTH = bool(compare_both)
    _USE_OPT_MATCH = bool(use_opt_match)
    _BANK_H5 = bank_h5
    _BANK_DSET = cast(h5py.Dataset, bank_obj)
    _GAMMA_ARR = np.asarray(gamma_arr)
    _GAMMA_CHUNK = int(gamma_chunk) if gamma_chunk is not None else None
    atexit.register(lambda h5=bank_h5: h5.close())


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
    (
        s_strain,
        psd,
        delta_f,
        compare_both,
        use_opt_match,
        bank_dset,
        gamma_arr,
        gamma_chunk,
    ) = _require_worker_state()
    r, c = args
    n_gamma = bank_dset.shape[2]
    ep_vec = np.empty(n_gamma, dtype=np.float32)
    best_ep = np.inf
    best_gamma = 0.0
    chunk = gamma_chunk or max(1, min(32, n_gamma))
    for k0 in range(0, n_gamma, chunk):
        k1 = min(n_gamma, k0 + chunk)
        gamma_block = cast(
            np.ndarray, bank_dset[int(r), int(c), k0:k1, :]
        )  # shape (g, n_freq)
        gamma_block = cast_to_match_precision(gamma_block)
        for local_idx in range(gamma_block.shape[0]):
            k = k0 + local_idx
            t_arr = gamma_block[local_idx]
            t_arr, _ = ensure_same_length(t_arr, s_strain)
            res = mismatch_from_strains(
                t_arr,
                s_strain,
                f_min=20.0,
                delta_f=delta_f,
                psd=psd,
                use_opt_match=use_opt_match,
                compare_both=compare_both,
            )
            ep = float(res["mismatch"])
            ep_vec[k] = ep
            if ep < best_ep:
                best_ep = ep
                best_gamma = float(gamma_arr[k])
    return int(r), int(c), ep_vec, float(best_ep), float(best_gamma)
