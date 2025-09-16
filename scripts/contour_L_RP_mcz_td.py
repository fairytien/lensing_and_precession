import sys, os, argparse
from typing import Tuple, Dict, Any, List, Optional
from multiprocessing import Pool, cpu_count

import numpy as np
import copy

# Ensure project root is on path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Reuse utilities and defaults
from modules.functions_v3 import *
from modules.default_params_v3 import *
from modules.Classes_v2 import Precessing as P2


def _ensure_dirs(base_dir: str) -> Tuple[str, str]:
    fig_dir = os.path.join(base_dir, "figures")
    data_dir = os.path.join(base_dir, "data")
    os.makedirs(fig_dir, exist_ok=True)
    os.makedirs(data_dir, exist_ok=True)
    return fig_dir, data_dir


def _build_params_for_location(
    theta_J: Optional[float] = None,
    phi_J: Optional[float] = None,
    theta_S: Optional[float] = None,
    phi_S: Optional[float] = None,
) -> Tuple[dict, dict]:
    # Default to Taman edge-on, then apply any user overrides
    lens_p, rp_p = set_to_location(
        loc_params["Taman"]["edgeon"], lens_params_1, RP_params_1
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
    lens_params, RP_params = _build_params_for_location(
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

    # Prepare jobs for parallel execution across grid cells
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

    # Compute mismatch grid in parallel
    if n_workers is None:
        n_workers = min(cpu_count(), len(jobs))

    Z = np.zeros_like(X, dtype=float)
    G = np.zeros_like(X, dtype=float)

    with Pool(n_workers) as pool:
        results = pool.map(_compute_cell_min_ep, jobs)

    # Fill results back into matrices
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
):
    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
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
        # Default: parallelize within each (mcz, td) over the (omega, theta) grid
        for i, mcz in enumerate(mcz_arr):
            print(f"Computing minima for mcz={mcz:.2f} Msun ({i+1}/{len(mcz_arr)})")
            for j, td in enumerate(td_arr):
                print(
                    f"  td={td*1e3:.2f} ms ({j+1}/{len(td_arr)}) -> optimizing over (omega, theta, gamma_P)"
                )
                ep_min, omega_best, theta_best, gamma_best = (
                    _compute_scalar_min_for_mcz_td(
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
                )
                Zmap[i, j] = ep_min
                Omap[i, j] = omega_best
                Tmap[i, j] = theta_best
                Gmap[i, j] = gamma_best

    # Package results and save
    custom_orient = any(v is not None for v in (theta_J, phi_J, theta_S, phi_S))
    results: Dict[str, Any] = {
        "mcz_arr": mcz_arr,
        "td_arr": td_arr,
        "I": I,
        "location": "custom" if custom_orient else "Taman.edgeon",
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
    )
