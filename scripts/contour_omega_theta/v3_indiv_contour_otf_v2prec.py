import sys, os, argparse
from typing import Tuple, Dict, Any, List
from multiprocessing import Pool, cpu_count

import numpy as np
import copy

# Ensure project root is on path
sys.path.insert(
    0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
)

from modules.functions import (
    get_gw,
    Sn,
    optimize_mismatch_gammaP,
    get_y_from_I,
    get_MLz_from_td,
    pickle_data,
    timer_decorator,
)
from modules.default_params import (
    lens_params_1,
    RP_params_1,
    SOLMASS2SEC,
)
from modules.Classes import Precessing as P2


def _ensure_dirs(base_dir: str) -> Tuple[str, str]:
    fig_dir = os.path.join(base_dir, "figures", "contour_omega_theta")
    data_dir = os.path.join(base_dir, "data", "contour_omega_theta")
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
    """
    Build and return (source_params_lensed, template_params_precessing) for v3 APIs.
    - Lensed source params must contain keys "MLz" and "y" to be routed to Lensing classes
    - Precessing template params use Regular Precession model without lensing keys
    """
    s_params = copy.deepcopy(lens_params_1)
    t_params = copy.deepcopy(RP_params_1)

    # Set sky/orientation for both
    for p in (s_params, t_params):
        p["theta_S"], p["phi_S"], p["theta_J"], p["phi_J"] = (
            theta_S,
            phi_S,
            theta_J,
            phi_J,
        )

    # Set chirp mass (in seconds)
    s_params["mcz"] = t_params["mcz"] = float(mcz_msun) * SOLMASS2SEC

    # Set lensing params from I, td (MLz expected in seconds)
    y = get_y_from_I(I)
    s_params["y"] = y
    s_params["MLz"] = float(get_MLz_from_td(td_s, y)) * SOLMASS2SEC

    # Ensure template has an initial gamma_P (required by optimize_mismatch_gammaP)
    if "gamma_P" not in t_params:
        t_params["gamma_P"] = 0.0

    return s_params, t_params


def _compute_cell_min_ep(args: tuple) -> tuple:
    (
        omega_val,
        theta_val,
        t_params_base,
        s_params,
        f_min,
        delta_f,
        psd,
        compare_both,
        use_opt_match,
        two_stage,
        coarse_points,
        xatol,
        maxiter,
    ) = args

    t_params = copy.deepcopy(t_params_base)
    t_params["omega_tilde"] = float(omega_val)
    t_params["theta_tilde"] = float(theta_val)

    if compare_both:
        # Use new API to compare match and optimized_match_bounded internally
        res = optimize_mismatch_gammaP(
            t_params,
            s_params,
            f_min=f_min,
            delta_f=delta_f,
            psd=psd,
            compare_both=True,
            two_stage=two_stage,
            coarse_points=coarse_points,
            xatol=xatol,
            maxiter=maxiter,
            prec_Class=P2,
        )
        return float(res["ep_min"]), float(res["ep_min_gammaP"])  # epsilon, gammaP
    else:
        res = optimize_mismatch_gammaP(
            t_params,
            s_params,
            f_min=f_min,
            delta_f=delta_f,
            psd=psd,
            use_opt_match=use_opt_match,
            compare_both=False,
            two_stage=two_stage,
            coarse_points=coarse_points,
            xatol=xatol,
            maxiter=maxiter,
            prec_Class=P2,
        )
        return float(res["ep_min"]), float(res["ep_min_gammaP"])  # epsilon, gammaP


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
    compare_both: bool = False,
    n_workers: int = None,
    no_plot: bool = False,
    tag: str = "",
    two_stage: bool = False,
    coarse_points: int = 17,
    xatol: float = 1e-3,
    maxiter: int = 50,
):
    base_dir = os.path.dirname(
        os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    )
    fig_dir, data_dir = _ensure_dirs(base_dir)

    td_s = td_ms / 1e3

    # Build params and PSD once
    s_params, t_params_base = _build_params(
        mcz_msun, td_s, I, theta_S, phi_S, theta_J, phi_J
    )
    s_gw = get_gw(s_params, f_min=f_min, delta_f=delta_f)
    f_arr = s_gw["f_array"]
    psd = Sn(f_arr, f_min=f_min, delta_f=delta_f)

    # Build grids
    omega_arr = np.linspace(omega_min, omega_max, int(omega_points))
    theta_arr = np.linspace(theta_min, theta_max, int(theta_points))
    X, Y = np.meshgrid(omega_arr, theta_arr)

    # Prepare jobs
    jobs: List[tuple] = []
    for r in range(int(theta_points)):
        for c in range(int(omega_points)):
            jobs.append(
                (
                    X[r, c],
                    Y[r, c],
                    t_params_base,
                    s_params,
                    f_min,
                    delta_f,
                    psd,
                    compare_both,
                    use_opt_match,
                    two_stage,
                    int(coarse_points),
                    float(xatol),
                    int(maxiter),
                )
            )

    if n_workers is None:
        n_workers = min(cpu_count(), len(jobs))
    print(f"Using {n_workers} workers over a {omega_points}x{theta_points} grid")

    # Parallel compute
    with Pool(n_workers) as pool:
        results = pool.map(_compute_cell_min_ep, jobs)

    # Fill arrays
    Z = np.zeros_like(X, dtype=float)
    G = np.zeros_like(X, dtype=float)
    k = 0
    for r in range(int(theta_points)):
        for c in range(int(omega_points)):
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
        "compare_both": compare_both,
    }

    base_name = (
        f"v2_indiv_contour_mcz{int(mcz_msun)}_td{int(td_ms)}ms_I{I}_"
        f"thetaS{round(theta_S,3)}_phiS{round(phi_S,3)}_thetaJ{round(theta_J,3)}_phiJ{round(phi_J,3)}"
    )
    if tag:
        base_name = f"{base_name}_{tag}"

    pkl_path = pickle_data(out, data_dir, base_name)

    if not no_plot:
        import matplotlib.pyplot as plt

        cf = plt.contourf(X, Y, Z, levels=100, cmap="jet")
        cbar = plt.colorbar(cf)
        cbar.set_label(r"$\epsilon(\tilde{h}_{\mathrm{L}}, \tilde{h}_{\mathrm{RP}})$")

        # Find minimum mismatch and mark it with a green dot
        min_idx = np.unravel_index(np.argmin(Z), Z.shape)
        min_omega = X[min_idx]
        min_theta = Y[min_idx]
        min_epsilon = Z[min_idx]

        plt.plot(
            min_omega,
            min_theta,
            "go",
            markersize=5,
            markeredgecolor="darkgreen",
            markeredgewidth=1,
            label=r"min $\epsilon$",
        )
        plt.legend()

        plt.xlabel(r"$\tilde{\Omega}$")
        plt.ylabel(r"$\tilde{\theta}$")
        plt.tight_layout()
        fig_path = os.path.join(fig_dir, f"{base_name}.pdf")
        plt.savefig(fig_path, dpi=200)
        print("Figure saved as", fig_path)
        print(
            f"Minimum mismatch: {min_epsilon:.6f} at (omega={min_omega:.4f}, theta={min_theta:.4f})"
        )

    print("Pickle saved as", pkl_path)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description=(
            "Individual mismatch contour (L vs RP) over (omega_tilde, theta_tilde) with fixed mcz, td, I, and angles (v3, Precessing from Classes_v2)."
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
        "--compare_both",
        action="store_true",
        help="Use both match and optimized_match_bounded internally and take the best.",
    )
    parser.add_argument("--n_workers", type=int, default=None)
    parser.add_argument("--no_plot", action="store_true")
    parser.add_argument("--tag", type=str, default="")
    parser.add_argument("--two_stage", action="store_true")
    parser.add_argument("--coarse_points", type=int, default=17)
    parser.add_argument("--xatol", type=float, default=1e-3)
    parser.add_argument("--maxiter", type=int, default=50)

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
        compare_both=args.compare_both,
        n_workers=args.n_workers,
        no_plot=args.no_plot,
        tag=args.tag,
        two_stage=args.two_stage,
        coarse_points=args.coarse_points,
        xatol=args.xatol,
        maxiter=args.maxiter,
    )
