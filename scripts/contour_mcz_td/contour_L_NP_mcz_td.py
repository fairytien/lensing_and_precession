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
from modules.waveform import (
    set_orientation,
    get_fcut_from_mcz,
    get_MLz_from_td,
    get_y_from_I,
)
from modules.snr import Sn
from modules.match_utils import mismatch_from_params, optimize_mismatch_mcz
from modules.runtime_helpers import timer_decorator
from modules.default_params import (
    SOLMASS2SEC,
    lens_params_1,
    NP_params_1,
    orient_params,
)
from modules.cosmology import apply_z
from modules.filenames import _format_min_precision, contour_mcz_td_filename
from modules.plot_utils import apply_physics_paper_style, LBL_EPS_LNP, LBL_MCZ, LBL_TD
from modules.cli_utils import resolve_grid_array

apply_physics_paper_style()


ORIENTATION_TAG = "Taman_edgeon"


def _source_mcz_threshold_for_band(
    f_min: float, delta_f: float, eta: float, z: Optional[float]
) -> float:
    """Approx source-frame mcz threshold (Msun) where f_cut == f_min + delta_f.

    Above this source mass, there is insufficient frequency bandwidth and rows may
    be rejected by the f_cut guard.
    """
    f_guard = f_min + delta_f
    mcz_det_threshold = (
        eta ** (3 / 5) / (6 ** (3 / 2) * np.pi * f_guard) / SOLMASS2SEC
    )  # Msun
    if z is None:
        return float(mcz_det_threshold)
    return float(mcz_det_threshold / (1 + z))


def _ensure_dirs(base_dir: str) -> Tuple[str, str]:
    fig_dir = os.path.join(base_dir, "figures", "contour_mcz_td")
    data_dir = os.path.join(base_dir, "data", "contour_mcz_td")
    os.makedirs(fig_dir, exist_ok=True)
    os.makedirs(data_dir, exist_ok=True)
    return fig_dir, data_dir


def _with_l_np_contour_name(filepath: str, tag: str = "") -> str:
    dirname, filename = os.path.split(filepath)
    filename = filename.replace("contour_", "contour_L_NP_", 1)
    if tag:
        root, ext_with_dot = os.path.splitext(filename)
        filename = f"{root}_{tag}{ext_with_dot}"
    return os.path.join(dirname, filename)


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


def _compute_mismatch_row(args):
    mcz, td_arr, y, f_min, delta_f, compare_both, z, optimize_mcz = args

    # Build fresh parameter dictionaries for this process
    lens_params, NP_params = set_orientation(
        orient_params["Taman"]["edgeon"], lens_params_1, NP_params_1
    )  # Location shouldn't matter for lensed and unlensed waveforms

    # Set chirp mass for both source and template (convert Msun -> sec)
    lens_params["mcz"] = NP_params["mcz"] = mcz * SOLMASS2SEC

    # Apply redshift if provided (updates mcz to detector-frame and sets dist)
    if z is not None:
        lens_params = apply_z(lens_params, z)
        NP_params = apply_z(NP_params, z)

    # Precompute PSD for this mcz once (depends on detector-frame mcz via f_cut)
    mcz_for_fcut = float(lens_params["mcz"] / SOLMASS2SEC)
    f_cut = get_fcut_from_mcz(mcz_for_fcut, lens_params["eta"])  # mcz in Msun
    if f_cut <= f_min + delta_f:
        # Not enough bandwidth above f_min; return NaNs for this row
        mcz_src = float(mcz)
        print(
            "Dropping mcz row due to insufficient bandwidth: "
            f"mcz_src={mcz_src:.6g} Msun, "
            f"mcz_det={mcz_for_fcut:.6g} Msun, "
            f"f_cut={f_cut:.6g} Hz <= f_min+delta_f={f_min + delta_f:.6g} Hz",
            flush=True,
        )
        return np.full(len(td_arr), np.nan, dtype=float)
    f_array = np.arange(f_min, f_cut, delta_f)
    if f_array.size < 2:
        return np.full(len(td_arr), np.nan, dtype=float)
    psd = cast(Any, Sn(f_array, f_min=f_min, delta_f=delta_f))

    # Compute mismatch for all time delays for this mcz
    mismatch_row = np.zeros(len(td_arr))
    mismatch_func = optimize_mismatch_mcz if optimize_mcz else mismatch_from_params

    for j, td in enumerate(td_arr):
        lens_params["y"] = y
        lens_params["MLz"] = get_MLz_from_td(td, y) * SOLMASS2SEC

        # Mismatch: NP template vs Lensed source
        try:
            if optimize_mcz:
                opt_ep_results = mismatch_func(
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
                res = mismatch_func(
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
    mcz_points: Optional[int] = 81,
    mcz_step: Optional[float] = None,
    td_min_ms: float = 20.0,
    td_max_ms: float = 70.0,
    td_points: Optional[int] = 51,
    td_step_ms: Optional[float] = None,
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

    base_dir = os.path.dirname(
        os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    )
    fig_dir, data_dir = _ensure_dirs(base_dir)

    # Arrays (units: mcz in Msun; td in seconds, but plot in ms)
    mcz_arr = resolve_grid_array(
        mcz_min, mcz_max, pts=mcz_points, step=mcz_step, label="mcz"
    )
    td_arr_ms = resolve_grid_array(
        td_min_ms, td_max_ms, pts=td_points, step=td_step_ms, label="td_ms"
    )
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

    mcz_src_threshold = _source_mcz_threshold_for_band(
        f_min=f_min,
        delta_f=delta_f,
        eta=lens_params_1["eta"],
        z=z,
    )
    print(
        "Approx source-frame mcz threshold for f_cut guard "
        f"(f_cut <= f_min+delta_f): {mcz_src_threshold:.6g} Msun",
    )

    # Prepare arguments for parallel computation
    args_list = [
        (mcz, td_arr, y, f_min, delta_f, compare_both, z, optimize_mcz)
        for mcz in mcz_arr
    ]

    if optimize_mcz:
        print("Computing mismatch with mcz optimization")
    else:
        print("Computing mismatch without mcz optimization")

    # Compute epsilon grid in parallel
    with Pool(n_processes) as pool:
        results = pool.map(_compute_mismatch_row, args_list)

    # Convert results to numpy array
    Z = np.array(results)

    dropped_mask = np.all(~np.isfinite(Z), axis=1)
    dropped_count = int(np.sum(dropped_mask))
    if dropped_count > 0:
        dropped_mcz = mcz_arr[dropped_mask]
        print(
            f"Dropped {dropped_count}/{len(mcz_arr)} mcz rows (all-NaN). "
            f"Range: {float(dropped_mcz[0]):.6g} to {float(dropped_mcz[-1]):.6g} Msun"
        )

    # Save results to HDF5
    h5_path = _with_l_np_contour_name(
        contour_mcz_td_filename(
            fig_dir=data_dir,
            I=I,
            mcz_min=mcz_min,
            mcz_max=mcz_max,
            mcz_pts=int(len(mcz_arr)),
            td_min_ms=td_min_ms,
            td_max_ms=td_max_ms,
            td_pts=int(len(td_arr_ms)),
            orientation_tag=ORIENTATION_TAG,
            z=z,
            ext="h5",
        ),
        tag=tag,
    )
    _save_contour_hdf5(
        filepath=h5_path,
        mcz_arr=mcz_arr,
        td_arr=td_arr,
        epsilon_matrix=Z,
        I=I,
        z=z,
        location=ORIENTATION_TAG,
        template="NP",
        optimize_mcz=optimize_mcz,
    )

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
            cbar.set_label(LBL_EPS_LNP)

        plt.xlabel(LBL_TD)
        plt.ylabel(LBL_MCZ)
        if z is not None:
            z_label = _format_min_precision(z, prefix="z = ")
            plt.plot([], [], " ", label=z_label)
            plt.legend(loc="best", framealpha=0.6)
        plt.tight_layout()

        fig_path = _with_l_np_contour_name(
            contour_mcz_td_filename(
                fig_dir=fig_dir,
                I=I,
                mcz_min=mcz_min,
                mcz_max=mcz_max,
                mcz_pts=int(len(mcz_arr)),
                td_min_ms=td_min_ms,
                td_max_ms=td_max_ms,
                td_pts=int(len(td_arr_ms)),
                orientation_tag=ORIENTATION_TAG,
                z=z,
                ext="pdf",
            ),
            tag=tag,
        )
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
    parser.add_argument(
        "--mcz_step",
        type=float,
        default=None,
        help="Step size for mcz grid (arange-style). Mutually exclusive with --mcz_points.",
    )
    parser.add_argument("--td_min_ms", type=float, default=20.0)
    parser.add_argument("--td_max_ms", type=float, default=70.0)
    parser.add_argument("--td_points", type=int, default=51)
    parser.add_argument(
        "--td_step_ms",
        type=float,
        default=None,
        help="Step size for td grid in ms (arange-style). Mutually exclusive with --td_points.",
    )
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
        help="Use both match and optimized_match_bounded internally and take the best.",
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
        mcz_step=args.mcz_step,
        td_min_ms=args.td_min_ms,
        td_max_ms=args.td_max_ms,
        td_points=args.td_points,
        td_step_ms=args.td_step_ms,
        f_min=args.f_min,
        delta_f=args.delta_f,
        z=args.redshift,
        no_plot=args.no_plot,
        n_processes=args.n_processes,
        optimize_mcz=args.optimize_mcz,
        tag=args.tag,
        compare_both=args.compare_both,
    )
