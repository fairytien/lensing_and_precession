import sys, os, argparse
from typing import Any, Optional, Tuple, cast
from multiprocessing import Pool, cpu_count
from datetime import datetime

import numpy as np
import matplotlib.pyplot as plt
import h5py

# Ensure project root is on path
sys.path.insert(
    0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
)

# Reuse utilities and defaults (explicit imports only)
from modules.functions_v3 import (
    set_orientation,
    get_fcut_from_mcz,
    Sn,
    get_MLz_from_td,
    mismatch_from_params,
    optimize_mismatch_mcz,
    timer_decorator,
    get_y_from_I,
)
from modules.default_params_v3 import (
    SOLMASS2SEC,
    lens_params_1,
    NP_params_1,
    orient_params,
)
from modules.cosmology import apply_z, mcz_src_to_det
from modules.filenames import format_z_tag


def _ensure_dirs(base_dir: str) -> Tuple[str, str]:
    fig_dir = os.path.join(base_dir, "figures")
    data_dir = os.path.join(base_dir, "data")
    os.makedirs(fig_dir, exist_ok=True)
    os.makedirs(data_dir, exist_ok=True)
    return fig_dir, data_dir


def _save_contour_hdf5(
    filepath: str,
    mcz_arr: np.ndarray,
    td_arr: np.ndarray,
    epsilon_matrix: np.ndarray,
    I: float,
    z: Optional[float],
    location: str,
    template: str,
    optimize_mcz: bool,
) -> str:
    """
    Save contour results to HDF5 file with compression.

    Returns the filepath of the saved file.
    """
    os.makedirs(os.path.dirname(filepath), exist_ok=True)
    with h5py.File(filepath, "w") as h5:
        # Store arrays as datasets
        h5.create_dataset("mcz_arr", data=mcz_arr.astype(np.float64))
        h5.create_dataset("td_arr", data=td_arr.astype(np.float64))
        h5.create_dataset(
            "epsilon_matrix",
            data=epsilon_matrix.astype(np.float32),
            compression="gzip",
            compression_opts=4,
            shuffle=True,
        )
        # Store scalar metadata as attributes
        h5.attrs["I"] = float(I)
        h5.attrs["z"] = np.nan if z is None else float(z)
        h5.attrs["location"] = location
        h5.attrs["template"] = template
        h5.attrs["optimize_mcz"] = optimize_mcz
        h5.attrs["created"] = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    return filepath


def _compute_mismatch_for_mcz(args):
    """
    Compute mismatch for a single mcz value across all time delays.
    This function is designed to be used with multiprocessing.
    """
    return _compute_mismatch_row(args, optimize_mcz=False)


def _compute_mismatch_for_mcz_optimized(args):
    """
    Compute optimized mismatch for a single mcz value across all time delays.
    This function optimizes over template mcz for each (source_mcz, td) pair.
    """
    return _compute_mismatch_row(args, optimize_mcz=True)


def _compute_mismatch_row(args, optimize_mcz: bool):
    mcz, td_arr, y, f_min, delta_f, compare_both, z = args

    # Build fresh parameter dictionaries for this process
    lens_params, NP_params = set_orientation(
        orient_params["Taman"]["edgeon"], lens_params_1, NP_params_1
    )  # Location shouldn't matter for lensed and unlensed waveforms

    # Set chirp mass for both source and template (convert Msun -> sec)
    lens_params["mcz"] = NP_params["mcz"] = mcz * SOLMASS2SEC

    # Apply redshift if provided (updates mcz to detector-frame and sets dist)
    if z is not None:
        apply_z(lens_params, z)
        apply_z(NP_params, z)

    # Precompute PSD for this mcz once (depends on mcz via f_cut)
    mcz_for_fcut = mcz if z is None else float(mcz_src_to_det(mcz, z))
    f_cut = get_fcut_from_mcz(mcz_for_fcut, lens_params["eta"])  # mcz in Msun
    if f_cut <= f_min + delta_f:
        # Not enough bandwidth above f_min; return NaNs for this row
        return np.full(len(td_arr), np.nan, dtype=float)
    f_array = np.arange(f_min, f_cut, delta_f)
    if f_array.size < 2:
        return np.full(len(td_arr), np.nan, dtype=float)
    psd = cast(Any, Sn(f_array, f_min=f_min, delta_f=delta_f))

    # Compute mismatch for all time delays for this mcz
    mismatch_row = np.zeros(len(td_arr))

    for j, td in enumerate(td_arr):
        lens_params["y"] = y
        lens_params["MLz"] = get_MLz_from_td(td, y) * SOLMASS2SEC

        # Mismatch: NP template vs Lensed source
        try:
            if optimize_mcz:
                opt_ep_results = optimize_mismatch_mcz(
                    NP_params,
                    lens_params,
                    f_min=f_min,
                    delta_f=delta_f,
                    psd=psd,
                    use_opt_match=True,
                    compare_both=compare_both,
                )
                mismatch_row[j] = float(
                    opt_ep_results["ep_min"]
                )  # ensure JSON/pickle friendly
            else:
                res = mismatch_from_params(
                    NP_params,
                    lens_params,
                    f_min=f_min,
                    delta_f=delta_f,
                    psd=psd,
                    use_opt_match=True,
                    compare_both=compare_both,
                )
                mismatch_row[j] = float(res["mismatch"])  # ensure JSON/pickle friendly
        except Exception:
            mismatch_row[j] = np.nan

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
    n_processes: Optional[int] = None,
    optimize_mcz: bool = False,
    tag: str = "",
    compare_both: bool = False,
    z: Optional[float] = None,
):
    if I <= 0:
        raise ValueError("I must be > 0")
    if mcz_points < 2 or td_points < 2:
        raise ValueError("mcz_points and td_points must both be >= 2")
    if mcz_min >= mcz_max:
        raise ValueError("mcz_min must be smaller than mcz_max")
    if td_min_ms >= td_max_ms:
        raise ValueError("td_min_ms must be smaller than td_max_ms")
    if f_min <= 0:
        raise ValueError("f_min must be > 0")
    if delta_f <= 0:
        raise ValueError("delta_f must be > 0")
    if n_processes is not None and n_processes <= 0:
        raise ValueError("n_processes must be positive when provided")
    if z is not None and z < 0:
        raise ValueError("redshift z must be non-negative")

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
    else:
        n_processes = min(n_processes, len(mcz_arr))

    print(f"Using {n_processes} processes for computation")
    if z is not None:
        print(f"Applying redshift z={z} (mcz treated as source-frame)")

    # Prepare arguments for parallel computation
    args_list = [(mcz, td_arr, y, f_min, delta_f, compare_both, z) for mcz in mcz_arr]

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

    # Build output filename
    filename_suffix = f"I{I}_opt_mcz" if optimize_mcz else f"I{I}"
    base_name = f"contour_L_NP_mcz_td_{filename_suffix}{format_z_tag(z)}"
    if tag:
        base_name = f"{base_name}_{tag}"

    # Save results to HDF5
    h5_path = os.path.join(data_dir, f"{base_name}.h5")
    _save_contour_hdf5(
        filepath=h5_path,
        mcz_arr=mcz_arr,
        td_arr=td_arr,
        epsilon_matrix=Z,
        I=I,
        z=z,
        location="Taman.edgeon",
        template="NP",
        optimize_mcz=optimize_mcz,
    )

    if not no_plot:
        finite_values = Z[np.isfinite(Z)]
        if finite_values.size == 0:
            print("Skipping plot: epsilon grid has no finite values.")
            print("HDF5 saved as", h5_path)
            return

        TD, MCZ = np.meshgrid(td_arr_ms, mcz_arr)
        plt.figure(figsize=(8, 6))
        z_min = float(np.min(finite_values))
        z_max = float(np.max(finite_values))
        if z_min == z_max:
            eps = max(abs(z_min), 1.0) * 1e-12
            levels = np.linspace(z_min - eps, z_max + eps, 3)
        else:
            levels = 100

        cf = plt.contourf(TD, MCZ, Z, levels=levels, cmap="jet")
        cbar = plt.colorbar(cf)

        if optimize_mcz:
            cbar.set_label(
                r"$\min_{\mathcal{M}_{\rm t}}$ $\epsilon(\tilde{h}_{\rm L}, \tilde{h}_{\rm NP})$"
            )
        else:
            cbar.set_label(r"$\epsilon(\tilde{h}_\mathrm{L}, \tilde{h}_\mathrm{NP})$")

        plt.xlabel(r"$\Delta t_d$ [ms]")
        plt.ylabel(r"$\mathcal{M}_s\ [M_\odot]$")
        if z is not None:
            plt.title(rf"$z={z:.3g}$")
        plt.tight_layout()

        fig_filename = f"{base_name}.pdf"
        fig_path = os.path.join(fig_dir, fig_filename)
        plt.savefig(fig_path, dpi=200)
        plt.close()
        print("Figure saved as", fig_path)

    print("HDF5 saved as", h5_path)


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
    parser.add_argument(
        "--redshift",
        "-z",
        type=float,
        default=None,
        help="Redshift. If provided, mcz values are treated as source-frame and dist is computed from z.",
    )
    parser.add_argument("--no_plot", action="store_true")
    parser.add_argument("--f_min", type=float, default=20.0)
    parser.add_argument("--delta_f", type=float, default=0.25)
    parser.add_argument(
        "--compare_both",
        action="store_true",
        help="Use both match and optimized_match internally and take the best.",
    )
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
        z=args.redshift,
        no_plot=args.no_plot,
        n_processes=args.n_processes,
        optimize_mcz=args.optimize_mcz,
        tag=args.tag,
        compare_both=args.compare_both,
    )
