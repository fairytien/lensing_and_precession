import sys, os, argparse
from typing import Tuple
from multiprocessing import Pool, cpu_count
import functools

import numpy as np
import matplotlib.pyplot as plt

# Ensure project root is on path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Reuse utilities and defaults
from modules.contours_ver2 import *  # noqa: F401,F403


def _ensure_dirs(base_dir: str) -> Tuple[str, str]:
    fig_dir = os.path.join(base_dir, "figures")
    data_dir = os.path.join(base_dir, "data")
    os.makedirs(fig_dir, exist_ok=True)
    os.makedirs(data_dir, exist_ok=True)
    return fig_dir, data_dir


def _build_params_for_location():
    # Set sky location to Taman edge-on for both lensed source and NP templates
    # Returns deep copies that we can safely mutate
    lens_params, NP_params = set_to_location(
        loc_params["Taman"]["edgeon"], lens_params_1, NP_params_1
    )
    return lens_params, NP_params


def _compute_mismatch_for_mcz(args):
    """
    Compute mismatch for a single mcz value across all time delays.
    This function is designed to be used with multiprocessing.
    """
    mcz, td_arr, y, f_min, delta_f = args

    # Build fresh parameter dictionaries for this process
    lens_params, NP_params = _build_params_for_location()

    # Set chirp mass for both source and template (convert Msun -> sec)
    lens_params["mcz"] = NP_params["mcz"] = mcz * solar_mass

    # Precompute PSD for this mcz once (depends on mcz via f_cut)
    f_cut = get_fcut_from_mcz(mcz, lens_params["eta"])  # mcz in Msun
    f_array = np.arange(f_min, f_cut, delta_f)
    psd = Sn(f_array)

    # Compute mismatch for all time delays for this mcz
    mismatch_row = np.zeros(len(td_arr))

    for j, td in enumerate(td_arr):
        lens_params["y"] = y
        lens_params["MLz"] = get_MLz_from_td(td, y) * solar_mass

        # Mismatch: NP template vs Lensed source
        res_no_opt = mismatch(
            NP_params,
            lens_params,
            f_min=f_min,
            delta_f=delta_f,
            psd=psd,
            use_opt_match=False,
        )
        res_opt = mismatch(
            NP_params,
            lens_params,
            f_min=f_min,
            delta_f=delta_f,
            psd=psd,
            use_opt_match=True,
        )
        res = {"mismatch": min(res_no_opt["mismatch"], res_opt["mismatch"])}
        mismatch_row[j] = float(res["mismatch"])  # ensure JSON/pickle friendly

    return mismatch_row


def _compute_mismatch_for_mcz_optimized(args):
    """
    Compute optimized mismatch for a single mcz value across all time delays.
    This function optimizes over template mcz for each (source_mcz, td) pair.
    """
    mcz, td_arr, y, f_min, delta_f = args

    # Build fresh parameter dictionaries for this process
    lens_params, NP_params = _build_params_for_location()

    # Set source chirp mass (convert Msun -> sec)
    lens_params["mcz"] = mcz * solar_mass

    # Precompute PSD for this mcz once (depends on mcz via f_cut)
    f_cut = get_fcut_from_mcz(mcz, lens_params["eta"])  # mcz in Msun
    f_array = np.arange(f_min, f_cut, delta_f)
    psd = Sn(f_array)

    # Compute mismatch for all time delays for this mcz
    mismatch_row = np.zeros(len(td_arr))

    for j, td in enumerate(td_arr):
        lens_params["y"] = y
        lens_params["MLz"] = get_MLz_from_td(td, y) * solar_mass

        # Optimize mismatch over template mcz: NP template vs Lensed source
        opt_ep_results = optimize_mismatch_mcz(
            NP_params, lens_params, f_min=f_min, delta_f=delta_f, psd=psd
        )
        mismatch_row[j] = float(opt_ep_results["ep_min"])  # ensure JSON/pickle friendly

    return mismatch_row


@timer_decorator
def main(
    I: float = 0.5,
    mcz_min: float = 10.0,
    mcz_max: float = 90.0,
    mcz_points: int = 81,
    td_min_ms: float = 20.0,
    td_max_ms: float = 70.0,
    td_points: int = 51,
    f_min: float = 20.0,
    delta_f: float = 0.25,
    no_plot: bool = False,
    n_processes: int = None,
    optimize_mcz: bool = False,
    tag: str = "",
):
    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    fig_dir, data_dir = _ensure_dirs(base_dir)

    # Arrays (units: mcz in Msun; td in seconds, but plot in ms)
    mcz_arr = np.linspace(mcz_min, mcz_max, mcz_points)
    td_arr_ms = np.linspace(td_min_ms, td_max_ms, td_points)
    td_arr = td_arr_ms / 1e3

    # Get y parameter
    y = get_y_from_I(I)

    # Determine number of processes
    if n_processes is None:
        n_processes = min(cpu_count(), len(mcz_arr))

    print(f"Using {n_processes} processes for computation")

    # Prepare arguments for parallel computation
    args_list = [(mcz, td_arr, y, f_min, delta_f) for mcz in mcz_arr]

    # Choose computation function based on optimization option
    if optimize_mcz:
        compute_func = _compute_mismatch_for_mcz_optimized
        print("Computing mismatch with mcz optimization")
    else:
        compute_func = _compute_mismatch_for_mcz
        print("Computing mismatch without mcz optimization")

    # Compute epsilon grid in parallel
    with Pool(n_processes) as pool:
        results = pool.map(compute_func, args_list)

    # Convert results to numpy array
    Z = np.array(results)

    # Package results and save
    results = {
        "mcz_arr": mcz_arr,
        "td_arr": td_arr,
        "epsilon_matrix": Z,
        "I": I,
        "location": "Taman.edgeon",
        "template": "NP",
        "optimize_mcz": optimize_mcz,
    }

    filename_suffix = f"I{I}_opt_mcz" if optimize_mcz else f"I{I}"
    base_name = f"mismatch_contour_L_NP_mcz_td_{filename_suffix}"
    if tag:
        base_name = f"{base_name}_{tag}"
    pkl_path = pickle_data(results, data_dir, base_name)

    if not no_plot:
        TD, MCZ = np.meshgrid(td_arr_ms, mcz_arr)
        plt.figure(figsize=(8, 6))
        cf = plt.contourf(TD, MCZ, Z, levels=100, cmap="jet")
        cbar = plt.colorbar(cf)

        if optimize_mcz:
            cbar.set_label(
                r"$\min_{\mathcal{M}_{\rm t}}$ $\epsilon(\tilde{h}_{\rm L}, \tilde{h}_{\rm NP})$"
            )
        else:
            cbar.set_label(r"$\epsilon(\tilde{h}_\mathrm{L}, \tilde{h}_\mathrm{NP})$")

        plt.xlabel(r"$\Delta t_d$ [ms]")
        plt.ylabel(r"$\mathcal{M}_s\ [M_\odot]$")
        plt.tight_layout()

        fig_filename = f"{base_name}.pdf"
        fig_path = os.path.join(fig_dir, fig_filename)
        plt.savefig(fig_path, dpi=200)
        print("Figure saved as", fig_path)

    print("Pickle saved as", pkl_path)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Generate a mismatch contour (Lensed vs NP templates) over mcz and time delay."
    )
    parser.add_argument(
        "--I", type=float, default=0.5, help="Flux ratio I (default: 0.5)"
    )
    parser.add_argument("--mcz_min", type=float, default=10.0)
    parser.add_argument("--mcz_max", type=float, default=90.0)
    parser.add_argument("--mcz_points", type=int, default=81)
    parser.add_argument("--td_min_ms", type=float, default=20.0)
    parser.add_argument("--td_max_ms", type=float, default=70.0)
    parser.add_argument("--td_points", type=int, default=51)
    parser.add_argument("--no_plot", action="store_true")
    parser.add_argument(
        "--n_processes",
        type=int,
        default=None,
        help="Number of processes to use (default: auto-detect)",
    )
    parser.add_argument(
        "--optimize_mcz",
        action="store_true",
        help="Optimize mismatch over template chirp mass",
    )
    parser.add_argument(
        "--tag",
        type=str,
        default="",
        help="Optional suffix to append to dataset/figure names to avoid overwriting",
    )

    args = parser.parse_args()
    main(
        I=args.I,
        mcz_min=args.mcz_min,
        mcz_max=args.mcz_max,
        mcz_points=args.mcz_points,
        td_min_ms=args.td_min_ms,
        td_max_ms=args.td_max_ms,
        td_points=args.td_points,
        no_plot=args.no_plot,
        n_processes=args.n_processes,
        optimize_mcz=args.optimize_mcz,
        tag=args.tag,
    )
