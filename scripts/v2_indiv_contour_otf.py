import sys, os, argparse
from typing import Tuple, Dict, Any, List
from multiprocessing import Pool, cpu_count

import numpy as np

# Ensure project root is on path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from modules.contours_v2 import *  # noqa: F401,F403


def _ensure_dirs(base_dir: str) -> Tuple[str, str]:
    fig_dir = os.path.join(base_dir, "figures")
    data_dir = os.path.join(base_dir, "data")
    os.makedirs(fig_dir, exist_ok=True)
    os.makedirs(data_dir, exist_ok=True)
    return fig_dir, data_dir


def _build_params(
    mcz_msun: float,
    td_s: float,
    I: float,
    theta_S: float,
    phi_S: float,
    theta_J: float,
    phi_J: float,
) -> Tuple[dict, dict]:
    lens_p = copy.deepcopy(lens_params_1)
    rp_p = copy.deepcopy(RP_params_1)

    # Set sky/orientation explicitly for both source and template
    for p in (lens_p, rp_p):
        p["theta_S"], p["phi_S"], p["theta_J"], p["phi_J"] = (
            theta_S,
            phi_S,
            theta_J,
            phi_J,
        )

    # Set chirp mass (sec)
    lens_p["mcz"] = rp_p["mcz"] = mcz_msun * solar_mass

    # Set lensing params from I, td
    y = get_y_from_I(I)
    lens_p["y"] = y
    lens_p["MLz"] = get_MLz_from_td(td_s, y) * solar_mass

    return lens_p, rp_p


def _compute_cell_min_ep(args: tuple) -> tuple:
    (
        omega_val,
        theta_val,
        t_params_base,
        s_params,
        f_min,
        delta_f,
        psd,
        compute_both_modes,
        use_opt_match,
    ) = args

    t_params = copy.deepcopy(t_params_base)
    t_params["omega_tilde"] = omega_val
    t_params["theta_tilde"] = theta_val

    if compute_both_modes:
        # Evaluate both matching modes and take the minimum
        res_no_opt = optimize_mismatch_gammaP(
            t_params,
            s_params,
            f_min=f_min,
            delta_f=delta_f,
            psd=psd,
            use_opt_match=False,
        )
        res_opt = optimize_mismatch_gammaP(
            t_params,
            s_params,
            f_min=f_min,
            delta_f=delta_f,
            psd=psd,
            use_opt_match=True,
        )

        if res_no_opt["ep_min"] <= res_opt["ep_min"]:
            return float(res_no_opt["ep_min"]), float(res_no_opt["ep_min_gammaP"])
        else:
            return float(res_opt["ep_min"]), float(res_opt["ep_min_gammaP"])
    else:
        res = optimize_mismatch_gammaP(
            t_params,
            s_params,
            f_min=f_min,
            delta_f=delta_f,
            psd=psd,
            use_opt_match=use_opt_match,
        )
        return float(res["ep_min"]), float(res["ep_min_gammaP"])


@timer_decorator
def main(
    mcz_msun: float = 20.0,
    td_ms: float = 22.0,
    I: float = 0.6,
    theta_S: float = np.pi / 3,
    phi_S: float = np.pi / 4,
    theta_J: float = np.pi / 6,
    phi_J: float = np.pi / 3,
    omega_min: float = 3.5,
    omega_max: float = 4.0,
    omega_points: int = 51,
    theta_min: float = 7.5,
    theta_max: float = 8.5,
    theta_points: int = 101,
    f_min: float = 20.0,
    delta_f: float = 0.25,
    use_opt_match: bool = True,
    compute_both_modes: bool = True,
    n_workers: int = None,
    no_plot: bool = False,
    tag: str = "",
):
    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    fig_dir, data_dir = _ensure_dirs(base_dir)

    td_s = td_ms / 1e3

    # Build params and PSD once
    s_params, t_params_base = _build_params(
        mcz_msun, td_s, I, theta_S, phi_S, theta_J, phi_J
    )
    s_gw = get_gw(s_params, f_min=f_min, delta_f=delta_f)
    f_arr = s_gw["f_array"]
    psd = Sn(f_arr)

    # Build grids
    omega_arr = np.linspace(omega_min, omega_max, omega_points)
    theta_arr = np.linspace(theta_min, theta_max, theta_points)
    X, Y = np.meshgrid(omega_arr, theta_arr)

    # Prepare jobs
    jobs: List[tuple] = []
    for r in range(theta_points):
        for c in range(omega_points):
            jobs.append(
                (
                    X[r, c],
                    Y[r, c],
                    t_params_base,
                    s_params,
                    f_min,
                    delta_f,
                    psd,
                    compute_both_modes,
                    use_opt_match,
                )
            )

    if n_workers is None:
        n_workers = min(cpu_count(), len(jobs))
    print(f"Using {n_workers} workers over a {theta_points}x{omega_points} grid")

    # Parallel compute
    with Pool(n_workers) as pool:
        results = pool.map(_compute_cell_min_ep, jobs)

    # Fill arrays
    Z = np.zeros_like(X, dtype=float)
    G = np.zeros_like(X, dtype=float)
    k = 0
    for r in range(theta_points):
        for c in range(omega_points):
            Z[r, c], G[r, c] = results[k]
            k += 1

    # Package results and save
    out = {
        "omega_matrix": X,
        "theta_matrix": Y,
        "epsilon_matrix": Z,
        "gammaP_min_matrix": G,
        "source_params": s_params,
        "template_params": t_params_base,
        "mcz_msun": mcz_msun,
        "td_ms": td_ms,
        "I": I,
        "angles": {
            "theta_S": theta_S,
            "phi_S": phi_S,
            "theta_J": theta_J,
            "phi_J": phi_J,
        },
        "min_over_use_opt_match": compute_both_modes,
    }

    base_name = (
        f"v2_indiv_mismatch_L_RP_mcz{int(mcz_msun)}_td{int(td_ms)}ms_I{I}_"
        f"thetaS{round(theta_S,3)}_phiS{round(phi_S,3)}_thetaJ{round(theta_J,3)}_phiJ{round(phi_J,3)}"
    )
    if tag:
        base_name = f"{base_name}_{tag}"

    pkl_path = pickle_data(out, data_dir, base_name)

    if not no_plot:
        import matplotlib.pyplot as plt

        plt.figure(figsize=(7.5, 6))
        cf = plt.contourf(X, Y, Z, levels=100, cmap="jet")
        cbar = plt.colorbar(cf)
        cbar.set_label(r"$\epsilon(\tilde{h}_\mathrm{L}, \tilde{h}_\mathrm{RP})$")
        plt.xlabel(r"$\tilde{\Omega}$")
        plt.ylabel(r"$\tilde{\theta}$")
        plt.tight_layout()
        fig_path = os.path.join(fig_dir, f"{base_name}.pdf")
        plt.savefig(fig_path, dpi=200)
        print("Figure saved as", fig_path)

    print("Pickle saved as", pkl_path)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description=(
            "Individual mismatch contour (L vs RP) over (omega_tilde, theta_tilde) with fixed mcz, td, I, and angles."
        )
    )
    parser.add_argument("--mcz_msun", type=float, default=20.0)
    parser.add_argument("--td_ms", type=float, default=22.0)
    parser.add_argument("--I", type=float, default=0.6)
    parser.add_argument("--theta_S", type=float, default=float(np.pi / 3))
    parser.add_argument("--phi_S", type=float, default=float(np.pi / 4))
    parser.add_argument("--theta_J", type=float, default=float(np.pi / 6))
    parser.add_argument("--phi_J", type=float, default=float(np.pi / 3))
    parser.add_argument("--omega_min", type=float, default=3.5)
    parser.add_argument("--omega_max", type=float, default=4.0)
    parser.add_argument("--omega_points", type=int, default=51)
    parser.add_argument("--theta_min", type=float, default=7.5)
    parser.add_argument("--theta_max", type=float, default=8.5)
    parser.add_argument("--theta_points", type=int, default=101)
    parser.add_argument("--f_min", type=float, default=20.0)
    parser.add_argument("--delta_f", type=float, default=0.25)
    parser.add_argument("--use_opt_match", action="store_true")
    parser.add_argument(
        "--single_mode",
        action="store_true",
        help="If set, compute only one match mode controlled by --use_opt_match. Otherwise compute both modes and take the min.",
    )
    parser.add_argument("--n_workers", type=int, default=None)
    parser.add_argument("--no_plot", action="store_true")
    parser.add_argument("--tag", type=str, default="")

    args = parser.parse_args()
    main(
        mcz_msun=args.mcz_msun,
        td_ms=args.td_ms,
        I=args.I,
        theta_S=args.theta_S,
        phi_S=args.phi_S,
        theta_J=args.theta_J,
        phi_J=args.phi_J,
        omega_min=args.omega_min,
        omega_max=args.omega_max,
        omega_points=args.omega_points,
        theta_min=args.theta_min,
        theta_max=args.theta_max,
        theta_points=args.theta_points,
        f_min=args.f_min,
        delta_f=args.delta_f,
        use_opt_match=args.use_opt_match,
        compute_both_modes=not args.single_mode,
        n_workers=args.n_workers,
        no_plot=args.no_plot,
        tag=args.tag,
    )
