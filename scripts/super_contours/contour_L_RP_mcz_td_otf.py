import sys, os, argparse
from typing import Tuple, Dict, Any, List, Optional
from multiprocessing import Pool, cpu_count

import numpy as np
import copy

# Ensure project root is on path
sys.path.insert(
    0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
)

# Reuse utilities and defaults
from modules.functions_v3 import *
from modules.default_params_v3 import *
from modules.Classes_v2 import Precessing as P2
from pycbc.types import FrequencySeries


def _ensure_dirs(base_dir: str) -> Tuple[str, str]:
    fig_dir = os.path.join(base_dir, "figures", "super_contours")
    data_dir = os.path.join(base_dir, "data", "super_contours")
    os.makedirs(fig_dir, exist_ok=True)
    os.makedirs(data_dir, exist_ok=True)
    return fig_dir, data_dir


def _build_params_for_orientation(
    theta_J: Optional[float] = None,
    phi_J: Optional[float] = None,
    theta_S: Optional[float] = None,
    phi_S: Optional[float] = None,
) -> Tuple[dict, dict]:
    # Default to Taman edge-on, then apply any user overrides
    lens_p, rp_p = set_orientation(
        orient_params["Taman"]["edgeon"], lens_params_1, RP_params_1
    )

    if theta_J is not None:
        lens_p["theta_J"] = theta_J
        rp_p["theta_J"] = theta_J
    if phi_J is not None:
        lens_p["phi_J"] = phi_J
        rp_p["phi_J"] = phi_J
    if theta_S is not None:
        lens_p["theta_S"] = theta_S
        rp_p["theta_S"] = theta_S
    if phi_S is not None:
        lens_p["phi_S"] = phi_S
        rp_p["phi_S"] = phi_S

    return lens_p, rp_p


def _compute_cell_min_ep(
    args: tuple,
) -> tuple:
    (
        omega_val,
        theta_val,
        t_params_base,
        s_params_base,
        f_min,
        delta_f,
        psd,
        compare_both,
        use_opt_match,
    ) = args

    # Set template parameters for this grid cell
    t_params = copy.deepcopy(t_params_base)
    t_params["omega_tilde"] = omega_val
    t_params["theta_tilde"] = theta_val

    # If compare_both is enabled, use library to choose best match mode internally
    if compare_both:
        res = optimize_mismatch_gammaP(
            t_params,
            s_params_base,
            f_min=f_min,
            delta_f=delta_f,
            psd=psd,
            prec_Class=P2,
            compare_both=True,
        )
        return res["ep_min"], res["ep_min_gammaP"]

    # Otherwise, evaluate once using the chosen match mode
    res = optimize_mismatch_gammaP(
        t_params,
        s_params_base,
        f_min=f_min,
        delta_f=delta_f,
        psd=psd,
        prec_Class=P2,
        use_opt_match=use_opt_match,
        compare_both=False,
    )
    return res["ep_min"], res["ep_min_gammaP"]


def _gen_template_entry(args: tuple) -> tuple:
    (
        r,
        c,
        k,
        omega_val,
        theta_val,
        gamma_val,
        t_params_base,
        f_min,
        delta_f,
    ) = args

    t_params = copy.deepcopy(t_params_base)
    t_params["omega_tilde"] = omega_val
    t_params["theta_tilde"] = theta_val
    t_params["gamma_P"] = gamma_val

    t_strain = get_gw(
        t_params,
        f_min=f_min,
        delta_f=delta_f,
        prec_Class=P2,
        frequencySeries=False,
    )["strain"]
    return (r, c, k), t_strain


def _build_template_bank_for_mcz(
    RP_params_base: dict,
    omega_min: float,
    omega_max: float,
    omega_points: int,
    theta_min: float,
    theta_max: float,
    theta_points: int,
    gamma_points: int,
    f_min: float,
    delta_f: float,
    n_workers: int,
) -> tuple:
    omega_arr = np.linspace(omega_min, omega_max, omega_points)
    theta_arr = np.linspace(theta_min, theta_max, theta_points)
    gamma_arr = np.linspace(0, 2 * np.pi, gamma_points, endpoint=False)

    template_bank: np.ndarray = np.empty(
        (theta_points, omega_points, gamma_points), dtype=object
    )

    jobs: List[tuple] = []
    for r in range(theta_points):
        for c in range(omega_points):
            for k in range(gamma_points):
                jobs.append(
                    (
                        r,
                        c,
                        k,
                        omega_arr[c],
                        theta_arr[r],
                        gamma_arr[k],
                        RP_params_base,
                        f_min,
                        delta_f,
                    )
                )

    if n_workers is None:
        n_workers = min(cpu_count(), len(jobs))

    print(
        f"Building template bank: theta={theta_points}, omega={omega_points}, gamma={gamma_points} → {len(jobs)} templates"
    )
    with Pool(n_workers) as pool:
        for (r, c, k), t_strain in pool.map(_gen_template_entry, jobs):
            template_bank[r, c, k] = t_strain

    return omega_arr, theta_arr, gamma_arr, template_bank


def _min_mismatch_over_gamma(args: tuple) -> tuple:
    (
        r,
        c,
        t_strains_gamma,
        gamma_arr,
    ) = args

    best_ep = np.inf
    best_gamma = 0.0
    for k, t_arr in enumerate(t_strains_gamma):
        t_fs = FrequencySeries(t_arr, _DELTA_F)
        if len(t_fs) != len(_S_STRAIN):
            t_fs.resize(len(_S_STRAIN))
        res = mismatch_from_strains(
            t_fs,
            _S_STRAIN,
            f_min=0.0,
            delta_f=_DELTA_F,
            psd=_PSD,
            use_opt_match=True,
            compare_both=bool(_COMPARE_BOTH),
        )
        ep = float(res["mismatch"])
        if ep < best_ep:
            best_ep = ep
            best_gamma = float(gamma_arr[k])

    return r, c, best_ep, best_gamma


def _epsilon_grid_from_bank(
    template_bank: np.ndarray,
    gamma_arr: np.ndarray,
    s_strain: FrequencySeries,
    f_min: float,
    delta_f: float,
    psd: FrequencySeries,
    n_workers: int,
    compare_both: bool,
) -> tuple:
    theta_points, omega_points, gamma_points = template_bank.shape
    Z = np.zeros((theta_points, omega_points), dtype=float)
    G = np.zeros_like(Z)

    jobs: List[tuple] = []
    for r in range(theta_points):
        for c in range(omega_points):
            jobs.append(
                (
                    r,
                    c,
                    template_bank[r, c, :],
                    gamma_arr,
                )
            )

    if n_workers is None:
        n_workers = min(cpu_count(), len(jobs))

    with Pool(
        n_workers,
        initializer=_init_mismatch_worker,
        initargs=(s_strain, psd, delta_f, compare_both),
    ) as pool:
        for r, c, ep_min, g_best in pool.map(_min_mismatch_over_gamma, jobs):
            Z[r, c] = ep_min
            G[r, c] = g_best

    return Z, G


def _init_mismatch_worker(s_strain, psd, delta_f, compare_both):
    # Share read-only objects across jobs in this worker
    global _S_STRAIN, _PSD, _DELTA_F, _COMPARE_BOTH
    _S_STRAIN = s_strain
    _PSD = psd
    _DELTA_F = delta_f
    _COMPARE_BOTH = compare_both


def _orientation_tag(
    theta_J: Optional[float],
    phi_J: Optional[float],
    theta_S: Optional[float],
    phi_S: Optional[float],
) -> str:
    if any(v is not None for v in (theta_J, phi_J, theta_S, phi_S)):
        tj = "nan" if theta_J is None else f"{theta_J:.3f}"
        pj = "nan" if phi_J is None else f"{phi_J:.3f}"
        ts = "nan" if theta_S is None else f"{theta_S:.3f}"
        ps = "nan" if phi_S is None else f"{phi_S:.3f}"
        return f"custom_thetaJ{tj}_phiJ{pj}_thetaS{ts}_phiS{ps}"
    return "Taman_edgeon"


def _bank_cache_path(
    mcz_msun: float,
    omega_points: int,
    theta_points: int,
    gamma_points: int,
    f_min: float,
    delta_f: float,
    tag: str,
    bank_dir: str,
    bank_prefix: str,
) -> str:
    os.makedirs(bank_dir, exist_ok=True)
    return os.path.join(
        bank_dir,
        f"{bank_prefix}_mcz{mcz_msun:.0f}_o{omega_points}_t{theta_points}_g{gamma_points}_f{int(f_min)}_df{delta_f:.2f}_{tag}.npz",
    )


def _estimate_bank_bytes(
    t_params_base: dict,
    omega_min: float,
    omega_max: float,
    theta_min: float,
    theta_max: float,
    f_min: float,
    delta_f: float,
    gamma_points: int,
    omega_points: int,
    theta_points: int,
) -> int:
    # Estimate memory footprint by sampling one template strain
    omega_mid = 0.5 * (omega_min + omega_max)
    theta_mid = 0.5 * (theta_min + theta_max)
    t_params = copy.deepcopy(t_params_base)
    t_params["omega_tilde"] = omega_mid
    t_params["theta_tilde"] = theta_mid
    t_params["gamma_P"] = 0.0
    t_strain = get_gw(
        t_params,
        f_min=f_min,
        delta_f=delta_f,
        prec_Class=P2,
        frequencySeries=False,
    )["strain"]
    per_template_bytes = int(getattr(t_strain, "nbytes", np.asarray(t_strain).nbytes))
    total_templates = int(omega_points) * int(theta_points) * int(gamma_points)
    return per_template_bytes * total_templates


def _get_template_bank_for_mcz(
    rp_base: dict,
    omega_min: float,
    omega_max: float,
    omega_points: int,
    theta_min: float,
    theta_max: float,
    theta_points: int,
    gamma_points: int,
    f_min: float,
    delta_f: float,
    n_workers: int,
    cache_banks: bool,
    bank_dir: str,
    bank_prefix: str,
    bank_mem_limit_gb: float,
    force_bank: bool,
) -> tuple:
    mcz_msun = rp_base["mcz"] / SOLMASS2SEC
    tag = _orientation_tag(
        rp_base.get("theta_J"),
        rp_base.get("phi_J"),
        rp_base.get("theta_S"),
        rp_base.get("phi_S"),
    )
    npz_path = _bank_cache_path(
        mcz_msun,
        omega_points,
        theta_points,
        gamma_points,
        f_min,
        delta_f,
        tag,
        bank_dir,
        bank_prefix,
    )

    if cache_banks and os.path.isfile(npz_path):
        print(f"Loading cached template bank: {npz_path}")
        data = np.load(npz_path, allow_pickle=True)
        return (
            data["omega_arr"],
            data["theta_arr"],
            data["gamma_arr"],
            data["template_bank"],
        )

    # Memory safety check
    est_bytes = _estimate_bank_bytes(
        rp_base,
        omega_min,
        omega_max,
        theta_min,
        theta_max,
        f_min,
        delta_f,
        gamma_points,
        omega_points,
        theta_points,
    )
    est_gb = est_bytes / (1024**3)
    if est_gb > bank_mem_limit_gb and not force_bank:
        print(
            f"[WARN] Estimated bank size ~{est_gb:.2f} GB exceeds limit {bank_mem_limit_gb:.2f} GB. "
            "Falling back to legacy per-cell evaluation for this mcz. Use --force_bank to override or reduce grid sizes."
        )
        return None, None, None, None

    # Build bank
    omega_arr, theta_arr, gamma_arr, bank = _build_template_bank_for_mcz(
        rp_base,
        omega_min,
        omega_max,
        omega_points,
        theta_min,
        theta_max,
        theta_points,
        gamma_points,
        f_min,
        delta_f,
        n_workers,
    )

    if cache_banks:
        print(f"Saving template bank: {npz_path}")
        np.savez_compressed(
            npz_path,
            omega_arr=omega_arr,
            theta_arr=theta_arr,
            gamma_arr=gamma_arr,
            template_bank=bank,
        )

    return omega_arr, theta_arr, gamma_arr, bank


def _compute_contour_for_mcz_td(
    mcz_val: float,
    td_val: float,
    I: float,
    theta_J: Optional[float],
    phi_J: Optional[float],
    theta_S: Optional[float],
    phi_S: Optional[float],
    omega_min: float,
    omega_max: float,
    omega_points: int,
    theta_min: float,
    theta_max: float,
    theta_points: int,
    f_min: float,
    delta_f: float,
    n_workers: int,
    compare_both: bool,
    use_opt_match: bool,
) -> Dict[str, Any]:
    """
    For a single (mcz, td), compute the RP mismatch contour over
    (omega_tilde, theta_tilde), optimizing over gamma_P and taking the
    min between use_opt_match=False/True. Orientation angles may be overridden; if
    not provided, defaults to Taman edge-on.
    """

    # Build fresh parameter dictionaries for this task
    lens_params, RP_params = _build_params_for_orientation(
        theta_J=theta_J, phi_J=phi_J, theta_S=theta_S, phi_S=phi_S
    )

    # Set chirp mass for both source and template (convert Msun -> sec)
    lens_params["mcz"] = RP_params["mcz"] = mcz_val * SOLMASS2SEC

    # Set lensing parameters from I and td
    y = get_y_from_I(I)
    lens_params["y"] = y
    lens_params["MLz"] = get_MLz_from_td(td_val, y) * SOLMASS2SEC

    # Precompute PSD based on source strain once
    s_gw = get_gw(lens_params, f_min=f_min, delta_f=delta_f)
    f_arr = s_gw["f_array"]
    if len(f_arr) < 2:
        # Not enough samples; return NaN grids
        Z = np.full((theta_points, omega_points), np.nan, dtype=float)
        G = np.full((theta_points, omega_points), np.nan, dtype=float)
        return {
            "omega_matrix": np.meshgrid(
                np.linspace(omega_min, omega_max, omega_points),
                np.linspace(theta_min, theta_max, theta_points),
            )[0],
            "theta_matrix": np.meshgrid(
                np.linspace(omega_min, omega_max, omega_points),
                np.linspace(theta_min, theta_max, theta_points),
            )[1],
            "epsilon_matrix": Z,
            "gammaP_min_matrix": G,
            "source_params": lens_params,
            "template_params": RP_params,
        }
    psd = Sn(f_arr, f_min=f_min, delta_f=delta_f)

    # Build parameter grids
    omega_arr = np.linspace(omega_min, omega_max, omega_points)
    theta_arr = np.linspace(theta_min, theta_max, theta_points)
    X, Y = np.meshgrid(omega_arr, theta_arr)

    # Legacy per-cell computation retained for outer mode
    jobs: List[tuple] = []
    for r in range(theta_points):
        for c in range(omega_points):
            jobs.append(
                (
                    X[r, c],
                    Y[r, c],
                    RP_params,
                    lens_params,
                    f_min,
                    delta_f,
                    psd,
                    compare_both,
                    use_opt_match,
                )
            )

    if n_workers is None:
        n_workers = min(cpu_count(), len(jobs))

    Z = np.zeros_like(X, dtype=float)
    G = np.zeros_like(X, dtype=float)

    with Pool(n_workers) as pool:
        results = pool.map(_compute_cell_min_ep, jobs)

    k = 0
    for r in range(theta_points):
        for c in range(omega_points):
            Z[r, c], G[r, c] = results[k]
            k += 1

    return {
        "omega_matrix": X,
        "theta_matrix": Y,
        "epsilon_matrix": Z,
        "gammaP_min_matrix": G,
        "source_params": lens_params,
        "template_params": RP_params,
    }


def _compute_scalar_min_for_mcz_td(
    mcz_val: float,
    td_val: float,
    I: float,
    theta_J: Optional[float],
    phi_J: Optional[float],
    theta_S: Optional[float],
    phi_S: Optional[float],
    omega_min: float,
    omega_max: float,
    omega_points: int,
    theta_min: float,
    theta_max: float,
    theta_points: int,
    f_min: float,
    delta_f: float,
    n_workers: int,
    compare_both: bool,
    use_opt_match: bool,
) -> tuple:
    """
    Compute the minimal mismatch across the RP grid for a single (mcz, td).
    Returns (ep_min, omega_best, theta_best, gammaP_best).
    """
    contour = _compute_contour_for_mcz_td(
        mcz_val,
        td_val,
        I,
        theta_J,
        phi_J,
        theta_S,
        phi_S,
        omega_min,
        omega_max,
        omega_points,
        theta_min,
        theta_max,
        theta_points,
        f_min,
        delta_f,
        n_workers,
        compare_both,
        use_opt_match,
    )

    Z = contour["epsilon_matrix"]
    G = contour["gammaP_min_matrix"]
    X = contour["omega_matrix"]
    Y = contour["theta_matrix"]

    idx = np.unravel_index(np.argmin(Z, axis=None), Z.shape)
    ep_min = float(Z[idx])
    omega_best = float(X[idx])
    theta_best = float(Y[idx])
    gamma_best = float(G[idx])

    return ep_min, omega_best, theta_best, gamma_best


@timer_decorator
def main(
    I: float = 0.5,
    theta_J: Optional[float] = None,
    phi_J: Optional[float] = None,
    theta_S: Optional[float] = None,
    phi_S: Optional[float] = None,
    mcz_min: float = 10.0,
    mcz_max: float = 90.0,
    mcz_points: int = 81,
    td_min_ms: float = 20.0,
    td_max_ms: float = 60.0,
    td_points: int = 41,
    omega_min: float = 0.0,
    omega_max: float = 5.0,
    omega_points: int = 31,
    theta_min: float = 0.0,
    theta_max: float = 15.0,
    theta_points: int = 61,
    f_min: float = 20.0,
    delta_f: float = 0.25,
    no_plot: bool = True,
    n_workers: int = None,
    parallel_mode: str = "grid",  # "grid" or "outer"
    outer_workers: int = None,
    compare_both: bool = False,
    use_opt_match: bool = True,
    gamma_points: int = 51,
):
    base_dir = os.path.dirname(
        os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    )
    fig_dir, data_dir = _ensure_dirs(base_dir)

    # Arrays (units: mcz in Msun; td in seconds, but plot/save in ms)
    mcz_arr = np.linspace(mcz_min, mcz_max, mcz_points)
    td_arr_ms = np.linspace(td_min_ms, td_max_ms, td_points)
    td_arr = td_arr_ms / 1e3

    # Allocate output matrices: rows=mcz, cols=td
    Zmap = np.zeros((len(mcz_arr), len(td_arr)), dtype=float)
    Omap = np.zeros_like(Zmap)
    Tmap = np.zeros_like(Zmap)
    Gmap = np.zeros_like(Zmap)

    # Compute per-(mcz, td) minimal mismatch across the RP grid
    if parallel_mode == "outer":
        # Parallelize across (mcz, td) pairs, and compute each grid serially inside the worker
        jobs: List[tuple] = []
        for i, mcz in enumerate(mcz_arr):
            for j, td in enumerate(td_arr):
                jobs.append(
                    (
                        mcz,
                        td,
                        I,
                        theta_J,
                        phi_J,
                        theta_S,
                        phi_S,
                        omega_min,
                        omega_max,
                        omega_points,
                        theta_min,
                        theta_max,
                        theta_points,
                        f_min,
                        delta_f,
                        1,  # avoid nested pools; do grid serially in worker
                        compare_both,
                        use_opt_match,
                    )
                )
        if outer_workers is None:
            outer_workers = min(cpu_count(), len(jobs))
        print(f"Parallelizing over (mcz, td) with {outer_workers} workers ...")
        with Pool(outer_workers) as pool:
            results_list = pool.starmap(_compute_scalar_min_for_mcz_td, jobs)

        k = 0
        for i in range(len(mcz_arr)):
            for j in range(len(td_arr)):
                ep_min, omega_best, theta_best, gamma_best = results_list[k]
                Zmap[i, j] = ep_min
                Omap[i, j] = omega_best
                Tmap[i, j] = theta_best
                Gmap[i, j] = gamma_best
                k += 1
    else:
        # Default: build template bank once per mcz, reuse across all td values
        for i, mcz in enumerate(mcz_arr):
            print(
                f"Precomputing template bank for mcz={mcz:.2f} Msun ({i+1}/{len(mcz_arr)})"
            )
            # Base params for this mass and orientation
            lens_base, rp_base = _build_params_for_orientation(
                theta_J=theta_J, phi_J=phi_J, theta_S=theta_S, phi_S=phi_S
            )
            lens_base["mcz"] = rp_base["mcz"] = mcz * SOLMASS2SEC

            # Optionally load/build bank with cache and memory safety
            omega_arr, theta_arr, gamma_arr, bank = _get_template_bank_for_mcz(
                rp_base,
                omega_min,
                omega_max,
                omega_points,
                theta_min,
                theta_max,
                theta_points,
                gamma_points,
                f_min,
                delta_f,
                n_workers,
                cache_banks=cache_banks,
                bank_dir=bank_dir,
                bank_prefix=bank_prefix,
                bank_mem_limit_gb=bank_mem_limit_gb,
                force_bank=force_bank,
            )

            # For each td, compute mismatch grid by scanning gamma and take minima over (omega, theta)
            for j, td in enumerate(td_arr):
                lens_params = copy.deepcopy(lens_base)
                y = get_y_from_I(I)
                lens_params["y"] = y
                lens_params["MLz"] = get_MLz_from_td(td, y) * SOLMASS2SEC

                s_gw = get_gw(lens_params, f_min=f_min, delta_f=delta_f)
                f_arr = s_gw["f_array"]
                if len(f_arr) < 2:
                    Zgrid = np.full((theta_points, omega_points), np.nan)
                    Ggrid = np.full((theta_points, omega_points), np.nan)
                else:
                    psd = Sn(f_arr, f_min=f_min, delta_f=delta_f)
                    if bank is None:
                        # Fall back to legacy per-cell path if bank was refused by memory guard
                        contour = _compute_contour_for_mcz_td(
                            mcz,
                            td,
                            I,
                            theta_J,
                            phi_J,
                            theta_S,
                            phi_S,
                            omega_min,
                            omega_max,
                            omega_points,
                            theta_min,
                            theta_max,
                            theta_points,
                            f_min,
                            delta_f,
                            n_workers,
                            compare_both,
                            use_opt_match,
                        )
                        Zgrid = contour["epsilon_matrix"]
                        Ggrid = contour["gammaP_min_matrix"]
                        omega_arr = contour["omega_matrix"][0]
                        theta_arr = contour["theta_matrix"][:, 0]
                    else:
                        Zgrid, Ggrid = _epsilon_grid_from_bank(
                            bank,
                            gamma_arr,
                            s_gw["strain"],
                            f_min,
                            delta_f,
                            psd,
                            n_workers,
                            compare_both,
                        )
                # Extract minima and record
                idx = np.unravel_index(np.nanargmin(Zgrid, axis=None), Zgrid.shape)
                Zmap[i, j] = float(Zgrid[idx])
                # Mesh coordinate arrays match bank
                Omap[i, j] = float(omega_arr[idx[1]])
                Tmap[i, j] = float(theta_arr[idx[0]])
                Gmap[i, j] = float(Ggrid[idx])

    # Package results and save
    custom_orient = any(v is not None for v in (theta_J, phi_J, theta_S, phi_S))
    results: Dict[str, Any] = {
        "mcz_arr": mcz_arr,
        "td_arr": td_arr,
        "I": I,
        "location": "custom" if custom_orient else "Taman_edgeon",
        "template": "RP",
        "omega_range": (omega_min, omega_max, omega_points),
        "theta_range": (theta_min, theta_max, theta_points),
        "epsilon_matrix": Zmap,
        "omega_best": Omap,
        "theta_best": Tmap,
        "gammaP_best": Gmap,
    }

    # Persist results
    pkl_path = pickle_data(
        results,
        data_dir,
        "contour_L_RP_mcz_td_Taman_edgeon",
    )

    # Plot mcz (y) vs td (x) contour of minimal mismatch
    if not no_plot:
        import matplotlib.pyplot as plt

        TD, MCZ = np.meshgrid(td_arr_ms, mcz_arr)
        plt.figure(figsize=(8, 6))
        cf = plt.contourf(TD, MCZ, Zmap, levels=100, cmap="jet")
        cbar = plt.colorbar(cf)
        cbar.set_label(r"$\epsilon(\tilde{h}_\mathrm{L}, \tilde{h}_\mathrm{RP})$")
        plt.xlabel(r"$\Delta t_d$ [ms]")
        plt.ylabel(r"$\mathcal{M}_s\ [M_\odot]$")
        plt.tight_layout()
        fig_path = os.path.join(
            fig_dir,
            f"contour_L_RP_mcz_td_I{I}.pdf",
        )
        plt.savefig(fig_path, dpi=200)
        print("Figure saved as", fig_path)

    print("Pickle saved as", pkl_path)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description=(
            "RP (Taman edge-on) mismatch contours over (omega_tilde, theta_tilde) for each (mcz, td). "
            "Gamma_P is optimized; mismatch is the minimum of use_opt_match=False/True."
        )
    )
    parser.add_argument(
        "--I", type=float, default=0.5, help="Flux ratio I (default: 0.5)"
    )
    parser.add_argument(
        "--theta_J",
        type=float,
        default=None,
        help=("Override J polar angle [rad]. If omitted, use Taman edge-on."),
    )
    parser.add_argument(
        "--phi_J",
        type=float,
        default=None,
        help=("Override J azimuthal angle [rad]. If omitted, use Taman edge-on."),
    )
    parser.add_argument(
        "--theta_S",
        type=float,
        default=None,
        help=("Override sky polar angle [rad]. If omitted, use Taman edge-on."),
    )
    parser.add_argument(
        "--phi_S",
        type=float,
        default=None,
        help=("Override sky azimuthal angle [rad]. If omitted, use Taman edge-on."),
    )
    parser.add_argument("--mcz_min", type=float, default=10.0)
    parser.add_argument("--mcz_max", type=float, default=90.0)
    parser.add_argument("--mcz_points", type=int, default=81)
    parser.add_argument("--td_min_ms", type=float, default=20.0)
    parser.add_argument("--td_max_ms", type=float, default=60.0)
    parser.add_argument("--td_points", type=int, default=41)
    parser.add_argument("--omega_min", type=float, default=0.0)
    parser.add_argument("--omega_max", type=float, default=6.0)
    parser.add_argument("--omega_points", type=int, default=61)
    parser.add_argument("--theta_min", type=float, default=0.0)
    parser.add_argument("--theta_max", type=float, default=15.0)
    parser.add_argument("--theta_points", type=int, default=151)
    parser.add_argument("--no_plot", action="store_true")
    parser.add_argument(
        "--n_workers",
        type=int,
        default=None,
        help="Number of worker processes for grid evaluation (default: auto-detect)",
    )
    parser.add_argument(
        "--parallel_mode",
        type=str,
        default="grid",
        choices=["grid", "outer"],
        help="Parallelize over the (omega, theta) grid per (mcz, td) [grid], or across (mcz, td) pairs [outer] while computing each grid serially.",
    )
    parser.add_argument(
        "--outer_workers",
        type=int,
        default=None,
        help="Number of worker processes when parallel_mode=outer (default: auto-detect)",
    )
    parser.add_argument(
        "--compare_both",
        action="store_true",
        help="Use both match and optimized_match internally and take the best.",
    )
    parser.add_argument(
        "--use_opt_match",
        action="store_true",
        help="When compare_both is False, choose optimized_match (True) or match (False).",
    )
    parser.add_argument(
        "--gamma_points",
        type=int,
        default=51,
        help="Number of gamma_P grid points for template bank (default: 51)",
    )
    parser.add_argument(
        "--cache_banks",
        action="store_true",
        help="Cache/load template banks as NPZ in bank_dir",
    )
    parser.add_argument(
        "--bank_dir",
        type=str,
        default=os.path.join(
            os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
            "data",
            "template_banks",
        ),
        help="Directory to save/load NPZ template banks",
    )
    parser.add_argument(
        "--bank_prefix",
        type=str,
        default="rp_bank",
        help="Filename prefix for NPZ template banks",
    )
    parser.add_argument(
        "--bank_mem_limit_gb",
        type=float,
        default=2.0,
        help="Safety threshold for estimated bank size in GB (default: 2.0)",
    )
    parser.add_argument(
        "--force_bank",
        action="store_true",
        help="Force building bank even if estimated size exceeds threshold",
    )

    args = parser.parse_args()
    main(
        I=args.I,
        theta_J=args.theta_J,
        phi_J=args.phi_J,
        theta_S=args.theta_S,
        phi_S=args.phi_S,
        mcz_min=args.mcz_min,
        mcz_max=args.mcz_max,
        mcz_points=args.mcz_points,
        td_min_ms=args.td_min_ms,
        td_max_ms=args.td_max_ms,
        td_points=args.td_points,
        omega_min=args.omega_min,
        omega_max=args.omega_max,
        omega_points=args.omega_points,
        theta_min=args.theta_min,
        theta_max=args.theta_max,
        theta_points=args.theta_points,
        no_plot=args.no_plot,
        n_workers=args.n_workers,
        parallel_mode=args.parallel_mode,
        outer_workers=args.outer_workers,
        compare_both=args.compare_both,
        use_opt_match=args.use_opt_match,
        gamma_points=args.gamma_points,
        cache_banks=args.cache_banks,
        bank_dir=args.bank_dir,
        bank_prefix=args.bank_prefix,
        bank_mem_limit_gb=args.bank_mem_limit_gb,
        force_bank=args.force_bank,
    )
