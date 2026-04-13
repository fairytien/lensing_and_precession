"""Compute Lindblom criterion using source SNR instead of template SNR.

This script computes the Lindblom criterion using the formula:
Lindblom = ε - (1 - √(1 - 1/ρ_s²))

where:
- ε is the mismatch between source and best-matching template
- ρ_s is the SNR of the source waveform itself (not the template)
"""

import os, sys, argparse
import numpy as np
import h5py

# Ensure project root is on path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from modules.filenames import _format_min_precision
from modules.functions import timer_decorator

import logging

logging.basicConfig(level=logging.INFO, format="[%(levelname)s] %(message)s")


@timer_decorator
def main(
    results_dir: str,
    td_min_ms: float,
    td_max_ms: float,
    mcz_min: float,
    mcz_max: float,
    I: float,
    orientation_tag: str,
    output_dir: str = None,
):
    """Compute Lindblom using source SNR.

    Parameters
    ----------
    results_dir : str
        Directory containing best-match and source SNR files
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
    output_dir : str | None
        Directory to save output file (default: results_dir/best_match)
    """
    if output_dir is None:
        output_dir = os.path.join(results_dir, "best_match")

    os.makedirs(output_dir, exist_ok=True)

    # Load mismatch data from best-match file (for RP templates)
    from glob import glob
    pattern = os.path.join(
        results_dir,
        "best_match",
        f"best_match*I{_format_min_precision(I)}*mcz{_format_min_precision(mcz_min)}-{_format_min_precision(mcz_max)}Msun*td{_format_min_precision(td_min_ms)}-{_format_min_precision(td_max_ms)}ms*{orientation_tag}.h5",
    )
    matches = glob(pattern)
    if not matches:
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

    best_match_path = matches[0]
    logging.info(f"Loading mismatch data from: {best_match_path}")

    # Load source SNR
    source_snr_path = os.path.join(
        results_dir,
        "best_match",
        f"source_snr_I{_format_min_precision(I)}_mcz{_format_min_precision(mcz_min)}-{_format_min_precision(mcz_max)}Msun_td{_format_min_precision(td_min_ms)}-{_format_min_precision(td_max_ms)}ms_{orientation_tag}.h5",
    )

    if not os.path.isfile(source_snr_path):
        raise FileNotFoundError(
            f"Source SNR file not found: {source_snr_path}\n"
            f"Please run scripts/compute_source_snr_contour.py first."
        )

    logging.info(f"Loading source SNR from: {source_snr_path}")

    # Load data
    with h5py.File(best_match_path, "r") as h5_best:
        mcz_arr = np.array(h5_best["mcz"])
        td_arr = np.array(h5_best["td"])
        if "I" in h5_best.attrs:
            I_value = float(h5_best.attrs["I"])
        else:
            I_value = I

        # Get mismatch from best-match file
        # We need to reconstruct mismatch from lindblom and snr
        if "lindblom_at_best_match" in h5_best and "snr_at_best_match" in h5_best:
            lindblom_old = np.array(h5_best["lindblom_at_best_match"])
            snr_template = np.array(h5_best["snr_at_best_match"])
            # Reconstruct mismatch: mismatch = lindblom + 1/(2*SNR^2)
            mismatch = lindblom_old + 1.0 / (2.0 * snr_template**2)
        elif "epsilon_min" in h5_best:
            mismatch = np.array(h5_best["epsilon_min"])
        else:
            raise KeyError(
                f"Could not find mismatch data in {best_match_path}. "
                f"Available datasets: {list(h5_best.keys())}"
            )

    with h5py.File(source_snr_path, "r") as h5_snr:
        source_snr = np.array(h5_snr["source_snr"])

    # Verify arrays match
    if mismatch.shape != source_snr.shape:
        raise ValueError(
            f"Shape mismatch: mismatch={mismatch.shape}, source_snr={source_snr.shape}"
        )

    # Compute Lindblom using new formula: ε - (1 - √(1 - 1/ρ_s²))
    # For values where ρ_s² < 1, the square root would be imaginary, so we set those to NaN
    rho_s_squared = source_snr**2
    valid_mask = rho_s_squared > 1.0

    # Initialize output array with NaNs
    lindblom_new = np.full_like(mismatch, np.nan, dtype=np.float32)

    # Compute 1 - √(1 - 1/ρ_s²) only for valid values
    if np.any(valid_mask):
        sqrt_arg = 1.0 - 1.0 / rho_s_squared
        # Additional check: sqrt_arg should be >= 0 (and rho_s² > 1)
        sqrt_valid = (sqrt_arg >= 0.0) & valid_mask
        
        if np.any(sqrt_valid):
            sqrt_term = np.sqrt(sqrt_arg[sqrt_valid])
            threshold = 1.0 - sqrt_term
            # Compute Lindblom = mismatch - threshold for valid values
            lindblom_new[sqrt_valid] = mismatch[sqrt_valid] - threshold

    # Save results
    output_path = os.path.join(
        output_dir,
        f"lindblom_from_source_snr_I{_format_min_precision(I_value)}_mcz{_format_min_precision(mcz_min)}-{_format_min_precision(mcz_max)}Msun_td{_format_min_precision(td_min_ms)}-{_format_min_precision(td_max_ms)}ms_{orientation_tag}.h5",
    )

    with h5py.File(output_path, "w") as h5:
        h5.create_dataset("mcz", data=mcz_arr.astype(np.float64))
        h5.create_dataset("td", data=td_arr.astype(np.float64))
        h5.create_dataset("lindblom_from_source_snr", data=lindblom_new.astype(np.float32))
        h5.create_dataset("source_snr", data=source_snr.astype(np.float32))
        h5.create_dataset("mismatch", data=mismatch.astype(np.float32))
        h5.attrs["I"] = I_value

    logging.info(f"Results saved to: {output_path}")

    # Print statistics
    valid_lindblom = lindblom_new[~np.isnan(lindblom_new)]
    if len(valid_lindblom) > 0:
        logging.info(
            f"Lindblom value range: {valid_lindblom.min():.6f} to {valid_lindblom.max():.6f}"
        )

    return output_path


if __name__ == "__main__":
    p = argparse.ArgumentParser(
        description="Compute Lindblom criterion using source SNR instead of template SNR."
    )
    p.add_argument(
        "--results_dir",
        type=str,
        default="data/contours_td_mcz",
        help="Directory containing best-match and source SNR files",
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
        "--output_dir",
        type=str,
        default=None,
        help="Directory to save output file (default: results_dir/best_match)",
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
        output_dir=args.output_dir,
    )

