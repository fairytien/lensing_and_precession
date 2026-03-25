"""Compute optimal SNR (SNR of the source waveform itself) for lensed sources.

This script computes the SNR of the lensed source waveform for each (mcz, td) point.
The optimal SNR is: SNR^2 = ⟨h|h⟩ = 4 ∫ [|h(f)|^2 / S_n(f)] df
"""

import os, sys, argparse
import numpy as np
from scipy.integrate import simpson
import h5py

# Ensure project root is on path
sys.path.insert(
    0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
)

from modules.functions_v3 import (
    get_gw,
    get_y_from_I,
    get_MLz_from_td,
    Sn,
    get_fcut_from_mcz,
    timer_decorator,
)
from modules.default_params_v3 import SOLMASS2SEC, lens_params_1, orient_params
from modules.orientation import resolve_orientation
from modules.functions_v3 import set_orientation
from modules.filenames import best_match_mcz_td_filename

import logging
from multiprocessing import Pool, cpu_count

logging.basicConfig(level=logging.INFO, format="[%(levelname)s] %(message)s")


def compute_source_snr(args):
    """Compute optimal SNR for a single (mcz, td) point."""
    (
        mcz_idx,
        td_idx,
        mcz,
        td,
        I,
        f_min,
        delta_f,
        lens_base,
    ) = args

    try:
        # Reconstruct source waveform (lensed)
        lens_params_td = dict(lens_base)
        lens_params_td["MLz"] = float(get_MLz_from_td(td, lens_base["y"]) * SOLMASS2SEC)
        source_gw = get_gw(lens_params_td, f_min=f_min, delta_f=delta_f)
        h_s = np.asarray(source_gw["strain"])
        f_array = source_gw["f_array"]

        # Compute PSD
        psd = Sn(f_array, f_min=f_min, delta_f=delta_f)

        # Compute optimal SNR: SNR^2 = 4 ∫ [|h(f)|^2 / S_n(f)] df
        h_squared = np.abs(h_s) ** 2
        integrand = h_squared / psd
        snr_squared = 4 * simpson(integrand, x=f_array)

        if snr_squared > 0:
            snr_value = np.sqrt(snr_squared)
        else:
            snr_value = np.nan

        return mcz_idx, td_idx, snr_value

    except Exception as e:
        logging.warning(
            f"Error computing source SNR for mcz={mcz:.1f}, td={td*1e3:.1f}ms: {e}"
        )
        return mcz_idx, td_idx, np.nan


@timer_decorator
def main(
    results_dir: str,
    td_min_ms: float,
    td_max_ms: float,
    mcz_min: float,
    mcz_max: float,
    I: float,
    orientation_tag: str,
    f_min: float = 20.0,
    delta_f: float = 0.25,
    n_workers: int = None,
):
    """Compute optimal SNR for lensed sources.

    Parameters
    ----------
    results_dir : str
        Directory containing best-match files
    td_min_ms : float
        Minimum time delay in ms
    td_max_ms : float
        Maximum time delay in ms
    mcz_min : float
        Minimum chirp mass in Msun
    mcz_max : float
        Maximum chirp mass in Msun
    I : float
        Flux ratio
    orientation_tag : str
        Orientation tag
    f_min : float
        Minimum frequency in Hz
    delta_f : float
        Frequency spacing in Hz
    n_workers : int | None
        Number of parallel workers
    """
    # Find best-match file to get mcz and td arrays
    # Try to find any best-match file with matching parameters
    from glob import glob

    pattern = os.path.join(
        results_dir,
        "best_match",
        f"best_match*I{_format_min_precision(I)}*mcz{_format_min_precision(mcz_min)}-{_format_min_precision(mcz_max)}Msun*td{_format_min_precision(td_min_ms)}-{_format_min_precision(td_max_ms)}ms*{orientation_tag}.h5",
    )
    matches = glob(pattern)
    if not matches:
        # Try alternative pattern
        pattern2 = os.path.join(
            results_dir,
            "best_match",
            f"*I{_format_min_precision(I)}*mcz{_format_min_precision(mcz_min)}-{_format_min_precision(mcz_max)}Msun*td{_format_min_precision(td_min_ms)}-{_format_min_precision(td_max_ms)}ms*{orientation_tag}.h5",
        )
        matches = glob(pattern2)

    if not matches:
        raise FileNotFoundError(
            f"Could not find best-match file matching pattern: {pattern}\n"
            f"Please ensure a best-match file exists with matching parameters."
        )

    best_match_path = matches[0]  # Use first match

    if not os.path.isfile(best_match_path):
        raise FileNotFoundError(
            f"Best-match file not found: {best_match_path}\n"
            f"Please ensure the best-match file exists."
        )

    # Load mcz and td arrays from best-match file
    with h5py.File(best_match_path, "r") as h5:
        mcz_arr = np.array(h5["mcz"])
        td_arr = np.array(h5["td"])
        if "I" in h5.attrs:
            I_value = float(h5.attrs["I"])
        else:
            I_value = I

    logging.info(f"Loaded data: mcz={len(mcz_arr)} points, td={len(td_arr)} points")
    logging.info(f"I={I_value}")

    # Parse orientation tag.
    # Canonical format is underscore-separated; legacy dot separators are normalized.
    orientation_tag_norm = str(orientation_tag).replace(".", "_")
    if "_" in orientation_tag_norm:
        location_parts = orientation_tag_norm.split("_", 1)
        location_name = location_parts[0]
        orientation_name = location_parts[1] if len(location_parts) > 1 else "edgeon"
    else:
        location_name = "Taman"
        orientation_name = "edgeon"

    # Get orientation parameters
    if (
        location_name in orient_params
        and orientation_name in orient_params[location_name]
    ):
        orient_dict = orient_params[location_name][orientation_name]
    else:
        logging.warning(
            f"Unknown location/orientation: {orientation_tag}, using Taman_edgeon"
        )
        orient_dict = orient_params["Taman"]["edgeon"]

    # Set up source (lensed) parameters
    y = get_y_from_I(I_value)
    lens_base = set_orientation(orient_dict, lens_params_1)[0]
    lens_base["y"] = float(y)

    # Initialize output array
    n_mcz = len(mcz_arr)
    n_td = len(td_arr)
    snr_matrix = np.full((n_mcz, n_td), np.nan, dtype=np.float32)

    # Prepare jobs
    if n_workers is None:
        n_workers = min(cpu_count(), n_mcz * n_td)

    jobs = []
    for mcz_idx, mcz in enumerate(mcz_arr):
        # Set mcz for source
        lens_params_mcz = dict(lens_base)
        lens_params_mcz["mcz"] = float(mcz) * SOLMASS2SEC

        for td_idx, td in enumerate(td_arr):
            jobs.append(
                (
                    mcz_idx,
                    td_idx,
                    mcz,
                    td,
                    I_value,
                    f_min,
                    delta_f,
                    lens_params_mcz,
                )
            )

    # Compute in parallel
    logging.info(f"Computing source SNR using {n_workers} workers...")
    with Pool(n_workers) as pool:
        results = pool.map(compute_source_snr, jobs)

    # Fill output array
    for mcz_idx, td_idx, snr_val in results:
        snr_matrix[mcz_idx, td_idx] = snr_val

    # Save results
    output_path = os.path.join(
        results_dir,
        "best_match",
        f"source_snr_I{_format_min_precision(I_value)}_mcz{_format_min_precision(mcz_min)}-{_format_min_precision(mcz_max)}Msun_td{_format_min_precision(td_min_ms)}-{_format_min_precision(td_max_ms)}ms_{orientation_tag}.h5",
    )
    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    with h5py.File(output_path, "w") as h5:
        h5.create_dataset("mcz", data=mcz_arr.astype(np.float64))
        h5.create_dataset("td", data=td_arr.astype(np.float64))
        h5.create_dataset("source_snr", data=snr_matrix.astype(np.float32))
        h5.attrs["I"] = I_value

    logging.info(f"Results saved to: {output_path}")

    # Print statistics
    valid_snr = snr_matrix[~np.isnan(snr_matrix)]
    if len(valid_snr) > 0:
        logging.info(
            f"Source SNR range: {valid_snr.min():.6f} to {valid_snr.max():.6f}"
        )

    return output_path


if __name__ == "__main__":
    from modules.filenames import _format_min_precision

    p = argparse.ArgumentParser(
        description="Compute optimal SNR (SNR of source waveform) for lensed sources."
    )
    p.add_argument(
        "--results_dir",
        type=str,
        default="data/contours_td_mcz",
        help="Directory containing best-match files",
    )
    p.add_argument(
        "--td_min_ms",
        type=float,
        required=True,
        help="Minimum time delay in ms",
    )
    p.add_argument(
        "--td_max_ms",
        type=float,
        required=True,
        help="Maximum time delay in ms",
    )
    p.add_argument(
        "--mcz_min",
        type=float,
        required=True,
        help="Minimum chirp mass in Msun",
    )
    p.add_argument(
        "--mcz_max",
        type=float,
        required=True,
        help="Maximum chirp mass in Msun",
    )
    p.add_argument(
        "--I",
        type=float,
        default=0.5,
        help="Flux ratio (default: 0.5)",
    )
    p.add_argument(
        "--orientation_tag",
        type=str,
        required=True,
        help="Orientation tag (e.g., Taman_edgeon)",
    )
    p.add_argument(
        "--f_min",
        type=float,
        default=20.0,
        help="Minimum frequency in Hz (default: 20.0)",
    )
    p.add_argument(
        "--delta_f",
        type=float,
        default=0.25,
        help="Frequency spacing in Hz (default: 0.25)",
    )
    p.add_argument(
        "--n_workers",
        type=int,
        default=None,
        help="Number of parallel workers (default: cpu_count())",
    )

    args = p.parse_args()
    main(
        results_dir=args.results_dir,
        td_min_ms=args.td_min_ms,
        td_max_ms=args.td_max_ms,
        mcz_min=args.mcz_min,
        mcz_max=args.mcz_max,
        I=args.I,
        orientation_tag=args.orientation_tag,
        f_min=args.f_min,
        delta_f=args.delta_f,
        n_workers=args.n_workers,
    )
