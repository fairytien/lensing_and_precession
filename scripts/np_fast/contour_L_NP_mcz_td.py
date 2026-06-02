import sys, os, argparse
from typing import Optional, Tuple
from multiprocessing import Pool, cpu_count

import numpy as np
import h5py

# Ensure project root is on path
sys.path.insert(
    0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
)

from modules.waveform import (
    set_orientation,
    get_fcut_from_mcz,
    get_MLz_from_td,
    get_y_from_I,
    get_gw,
)
from modules.snr import Sn
from modules.match_utils import MatchMethod, ensure_same_length
from scripts.np_fast.match_utils_np import (
    mismatch_block_serial,
    precompute_lensing_factors,
    build_lensed_source_strain,
)
from modules.runtime_helpers import timer_decorator
from modules.default_params import (
    SOLMASS2SEC,
    lens_params_1,
    NP_params_1,
    orient_params,
)
from modules.cosmology import apply_z
from modules.filenames import contour_run_dir, best_match_mcz_td_filename
from modules.cli_utils import resolve_grid_array
from modules.bank_io import (
    write_dataset_units,
    write_orientation_attr,
    write_scalar_attr_with_unit,
    write_missing_mcz_td_metadata,
)

ORIENTATION_TAG = "Taman_edgeon"


def _compute_mismatch_row(args) -> Tuple[np.ndarray, np.ndarray]:
    mcz, td_arr, y, f_min, delta_f, z = args

    # Build fresh parameter dictionaries for this process
    lens_params, NP_params = set_orientation(
        orient_params["Taman"]["edgeon"], lens_params_1, NP_params_1
    )

    # Set chirp mass for both source and template (convert Msun -> sec)
    lens_params["mcz"] = NP_params["mcz"] = mcz * SOLMASS2SEC

    # Apply redshift if provided (updates mcz to detector-frame and sets dist)
    if z is not None:
        lens_params = apply_z(lens_params, z)
        NP_params = apply_z(NP_params, z)

    # Precompute PSD for this mcz once (depends on detector-frame mcz via f_cut)
    mcz_for_fcut = float(lens_params["mcz"] / SOLMASS2SEC)
    f_cut = get_fcut_from_mcz(mcz_for_fcut, lens_params["eta"])
    nan_row = (
        np.full(len(td_arr), np.nan, dtype=np.float32),
        np.full(len(td_arr), np.nan, dtype=np.float32),
    )
    if f_cut <= f_min + delta_f:
        print(
            "Dropping mcz row due to insufficient bandwidth: "
            f"mcz_src={float(mcz):.6g} Msun, "
            f"mcz_det={mcz_for_fcut:.6g} Msun, "
            f"f_cut={f_cut:.6g} Hz <= f_min+delta_f={f_min + delta_f:.6g} Hz",
            flush=True,
        )
        return nan_row
    f_array = np.arange(f_min, f_cut, delta_f)
    if f_array.size < 2:
        return nan_row
    psd = Sn(f_array, f_min=f_min, delta_f=delta_f)

    # Build NP template once for this mcz
    np_strain = get_gw(NP_params, f_min, delta_f)["strain"]
    template_block = np.array([np.asarray(np_strain)])  # shape (1, n_freq)
    labels = np.array([0.0])

    # Precompute unlensed source waveform and magnification factors
    h_I, sqrt_mu_p, sqrt_mu_m = precompute_lensing_factors(lens_params, y, f_array)

    # Pre-resize/ensure_same_length for template_block to match source length f_array.size
    template_block_resized, _ = ensure_same_length(template_block[0], h_I)
    template_block_resized = template_block_resized.reshape(1, -1)

    ep_min_arr = np.zeros(len(td_arr), dtype=np.float32)
    gamma_best_arr = np.zeros(len(td_arr), dtype=np.float32)

    for j, td in enumerate(td_arr):
        try:
            s_strain = build_lensed_source_strain(
                h_I, sqrt_mu_p, sqrt_mu_m, f_array, td, delta_f
            )

            _, ep_min, label_best = mismatch_block_serial(
                template_block_resized,
                labels,
                s_strain,
                psd,
                f_min,
                delta_f,
                MatchMethod.OPTIMIZED_BRENT,
            )
            ep_min_arr[j] = ep_min
            gamma_best_arr[j] = label_best
        except Exception:
            ep_min_arr[j] = np.nan
            gamma_best_arr[j] = np.nan

    return ep_min_arr, gamma_best_arr


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
    n_processes: Optional[int] = None,
    z: Optional[float] = None,
    run_dir: str = "data/mismatch_L_NP",
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

    # Arrays (units: mcz in Msun; td in seconds)
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

    # Prepare arguments for parallel computation
    args_list = [(mcz, td_arr, y, f_min, delta_f, z) for mcz in mcz_arr]

    # Compute mismatch grid in parallel
    with Pool(n_processes) as pool:
        results = pool.map(_compute_mismatch_row, args_list)

    # Unpack (ep_min_arr, gamma_best_arr) pairs
    Zmap = np.array([r[0] for r in results], dtype=np.float32)  # (mcz, td)
    Gmap = np.array([r[1] for r in results], dtype=np.float32)  # (mcz, td)
    Omap = np.zeros_like(Zmap)  # omega_best: all 0 for NP
    Tmap = np.zeros_like(Zmap)  # theta_best: all 0 for NP

    dropped_mask = np.all(~np.isfinite(Zmap), axis=1)
    dropped_count = int(np.sum(dropped_mask))
    if dropped_count > 0:
        dropped_mcz = mcz_arr[dropped_mask]
        print(
            f"Dropped {dropped_count}/{len(mcz_arr)} mcz rows (all-NaN). "
            f"Range: {float(dropped_mcz[0]):.6g} to {float(dropped_mcz[-1]):.6g} Msun"
        )

    # MLz array in seconds, one value per td
    mlz_arr = get_MLz_from_td(td_arr, y) * SOLMASS2SEC

    # Resolve output path via pipeline naming helpers
    run_dir_abs = contour_run_dir(
        os.path.join(base_dir, run_dir),
        I=I,
        mcz_min=mcz_min,
        mcz_max=mcz_max,
        td_min_ms=td_min_ms,
        td_max_ms=td_max_ms,
        z=z,
        orientation_tag=ORIENTATION_TAG,
    )
    h5_path = best_match_mcz_td_filename(
        run_dir_abs,
        I=I,
        mcz_min=float(mcz_arr[0]),
        mcz_max=float(mcz_arr[-1]),
        mcz_pts=int(len(mcz_arr)),
        td_min_ms=td_min_ms,
        td_max_ms=td_max_ms,
        td_pts=int(len(td_arr)),
        omega_min=0.0,
        omega_max=0.0,
        omega_pts=1,
        theta_min=0.0,
        theta_max=0.0,
        theta_pts=1,
        gamma_pts=1,
        orientation_tag=ORIENTATION_TAG,
        z=z,
    )

    with h5py.File(h5_path, "w") as h5:
        h5.create_dataset("mcz", data=mcz_arr.astype(np.float64))
        h5.create_dataset("td", data=td_arr.astype(np.float64))
        h5.create_dataset("MLz", data=mlz_arr.astype(np.float64))
        h5.create_dataset("epsilon_min", data=Zmap)
        h5.create_dataset("omega_best", data=Omap)
        h5.create_dataset("theta_best", data=Tmap)
        h5.create_dataset("gamma_best", data=Gmap)
        write_dataset_units(
            h5,
            {
                "mcz": "Msun",
                "td": "s",
                "MLz": "s",
                "omega_best": "dimensionless",
                "theta_best": "dimensionless",
                "gamma_best": "rad",
            },
        )
        h5["epsilon_min"].attrs["axis_order"] = "mcz,td"
        h5["omega_best"].attrs["axis_order"] = "mcz,td"
        h5["theta_best"].attrs["axis_order"] = "mcz,td"
        h5["gamma_best"].attrs["axis_order"] = "mcz,td"
        write_missing_mcz_td_metadata(
            h5,
            expected_mcz=mcz_arr.astype(np.float64),
            missing_mcz=mcz_arr[dropped_mask].astype(np.float64),
        )
        h5.attrs["I"] = float(I)
        h5.attrs["template_family"] = "NP"
        write_scalar_attr_with_unit(h5, "z", z, none_as_nan=True)
        write_orientation_attr(h5, ORIENTATION_TAG)

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
    parser.add_argument("--f_min", type=float, default=20.0)
    parser.add_argument("--delta_f", type=float, default=0.25)
    parser.add_argument(
        "--n_processes",
        type=int,
        default=None,
        help="Number of processes to use (default: auto-detect)",
    )
    parser.add_argument(
        "--run_dir",
        type=str,
        default="data/mismatch_L_NP",
        help="Base output directory (relative to project root). Default: data/mismatch_L_NP",
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
        n_processes=args.n_processes,
        run_dir=args.run_dir,
    )
