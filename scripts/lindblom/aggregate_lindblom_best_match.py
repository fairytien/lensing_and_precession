"""Aggregate per-mcz Lindblom cubes into one best-match HDF5 file.

Scans results_dir for per-mcz Lindblom cubes, reduces each across (theta, omega),
stacks over mcz, and writes a combined best_match_lindblom_*.h5.

Use scripts/create_contour_mcz_td_from_lindblom.py to plot the aggregated results.
"""

import os, sys, argparse, glob
import h5py
import numpy as np

# Ensure project root is on path for local invocation
sys.path.insert(
    0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
)

from modules.filenames import (
    _format_min_precision,
    get_mismatch_cube_resolution,
    mismatch_mcz_cube_filename,
)
from modules.runtime_helpers import timer_decorator


@timer_decorator
def main(
    results_dir: str,
    td_min_ms: float,
    td_max_ms: float,
    mcz_min: float,
    mcz_max: float,
    orientation_tag: str,
):
    """Aggregate Lindblom cubes into best-match file.

    Parameters
    ----------
    results_dir : str
        Directory containing Lindblom cube files
    td_min_ms : float
        Minimum time delay in ms (for filename matching)
    td_max_ms : float
        Maximum time delay in ms (for filename matching)
    mcz_min : float
        Minimum chirp mass in Msun
    mcz_max : float
        Maximum chirp mass in Msun
    orientation_tag : str
        Orientation tag (for filename matching)
    """
    # Find all Lindblom cube files
    pattern = os.path.join(
        results_dir,
        "mismatch_cubes",
        f"*_lindblom.h5",
    )
    cube_paths_all = sorted(glob.glob(pattern))

    # Filter by mcz range and orientation
    cube_paths = []
    for p in cube_paths_all:
        base = os.path.basename(p)
        # Try to extract mcz from filename (e.g., mismatch_cubes_mcz47Msun_..._lindblom.h5)
        try:
            if "mcz" in base:
                token = base.split("_mcz", 1)[1]
                mcz_str = token.split("Msun", 1)[0]
                mcz_val = float(mcz_str)
            else:
                # Try to get mcz from file contents
                with h5py.File(p, "r") as h5:
                    if "mcz" in h5:
                        mcz_val = float(np.array(h5["mcz"]).item())
                    else:
                        continue
        except Exception:
            continue

        # Check mcz range
        if mcz_min <= mcz_val <= mcz_max:
            # Check orientation tag in filename
            if orientation_tag in base:
                cube_paths.append(p)

    if not cube_paths:
        raise FileNotFoundError(
            f"No Lindblom cube files found matching pattern: {pattern}\n"
            f"with mcz in [{mcz_min}, {mcz_max}] and orientation '{orientation_tag}'"
        )

    print(f"Found {len(cube_paths)} Lindblom cube files")

    mcz_vals = []
    L_rows = []  # Lindblom values
    S_rows = []  # SNR values
    td_arr = None
    source_attrs = {}
    has_snr = False

    for p in cube_paths:
        try:
            h5 = h5py.File(p, "r")
        except Exception as e:
            print(f"Warning: Skipping unreadable file: {p} ({e})")
            continue

        with h5:
            # Get mcz from file (may be in dataset or filename)
            if "mcz" in h5:
                mcz = float(np.array(h5["mcz"]).item())
            else:
                # Fallback: try to extract from filename
                base = os.path.basename(p)
                try:
                    token = base.split("_mcz", 1)[1]
                    mcz_str = token.split("Msun", 1)[0]
                    mcz = float(mcz_str)
                except Exception:
                    print(f"Warning: Could not extract mcz from {p}, skipping")
                    continue

            if td_arr is None:
                td_arr = np.array(h5["td"])  # (td,)
                # Extract source parameters from first cube file
                for key in ["I", "theta_J", "phi_J", "theta_S", "phi_S"]:
                    if key in h5.attrs:
                        source_attrs[key] = h5.attrs[key]

            # Check if SNR cube exists (for backward compatibility)
            if "snr_cube" in h5:
                snr_cube = np.array(h5["snr_cube"])  # (td, theta, omega)
            else:
                snr_cube = None
            theta_arr = np.array(h5["theta"])  # (theta,)
            omega_arr = np.array(h5["omega"])  # (omega,)

            # Load mismatch cube to get minimum mismatch (best-matching template)
            # Find the corresponding mismatch cube file
            base = os.path.basename(p)
            # Replace _lindblom.h5 with .h5 to get mismatch cube filename
            mismatch_cube_path = p.replace("_lindblom.h5", ".h5")
            if not os.path.isfile(mismatch_cube_path):
                raise FileNotFoundError(
                    f"Mismatch cube not found: {mismatch_cube_path}\n"
                    f"Required to compute Lindblom from best-matching template."
                )

            # Load mismatch data
            with h5py.File(mismatch_cube_path, "r") as h5_mm:
                epsilon_min_grid = np.array(
                    h5_mm["epsilon_min_grid"]
                )  # (td, theta, omega)

            # For each td, find the best-matching template (minimum mismatch)
            # and compute Lindblom = mismatch - 1/(2*SNR^2)
            L = np.zeros(td_arr.shape[0], dtype=np.float32)
            S = None
            if snr_cube is not None:
                S = np.zeros(td_arr.shape[0], dtype=np.float32)
            for j in range(td_arr.shape[0]):
                # Find the (theta, omega) that gives minimum mismatch (best-matching template)
                mismatch_grid = epsilon_min_grid[j]  # (theta, omega)
                best_match_idx = np.unravel_index(
                    int(np.nanargmin(mismatch_grid)), mismatch_grid.shape
                )
                mismatch_best = float(mismatch_grid[best_match_idx])

                if snr_cube is not None:
                    Sgrid = snr_cube[j]
                    snr_best = float(Sgrid[best_match_idx])
                    # Compute Lindblom = mismatch - 1/(2*SNR^2)
                    if snr_best > 0 and not np.isnan(snr_best):
                        L[j] = mismatch_best - 1.0 / (2.0 * snr_best**2)
                        S[j] = snr_best
                    else:
                        L[j] = np.nan
                        S[j] = np.nan
                else:
                    # If no SNR data, cannot compute Lindblom
                    L[j] = np.nan

            mcz_vals.append(mcz)
            L_rows.append(L)
            if snr_cube is not None and S is not None:
                S_rows.append(S)
                has_snr = True
            elif has_snr:
                # If we've seen SNR data before but this cube doesn't have it, add NaN row
                S_rows.append(np.full(td_arr.shape[0], np.nan, dtype=np.float32))

    if not mcz_vals:
        raise ValueError("No valid Lindblom cubes found")

    # Build full mcz grid [mcz_min, ..., mcz_max] with blanks (NaNs) for missing entries
    desired_mcz = np.arange(float(mcz_min), float(mcz_max) + 1.0, 1.0, dtype=np.float64)
    td_len = td_arr.shape[0]
    Lmap = np.full((desired_mcz.shape[0], td_len), np.nan, dtype=np.float32)
    if has_snr:
        Smap = np.full((desired_mcz.shape[0], td_len), np.nan, dtype=np.float32)

    # Place available rows at the correct indices
    present_mcz = np.array(mcz_vals, dtype=np.float64)
    order = np.argsort(present_mcz)
    present_mcz_sorted = present_mcz[order]
    L_rows_sorted = [L_rows[i] for i in order]
    if has_snr and len(S_rows) == len(L_rows):
        S_rows_sorted = [S_rows[i] for i in order]
    elif has_snr:
        # Handle case where some cubes don't have SNR - pad with NaN
        S_rows_sorted = []
        for i in order:
            if i < len(S_rows):
                S_rows_sorted.append(S_rows[i])
            else:
                S_rows_sorted.append(np.full(td_len, np.nan, dtype=np.float32))

    index_map = {val: idx for idx, val in enumerate(desired_mcz)}
    for val, Lr in zip(present_mcz_sorted, L_rows_sorted):
        if val in index_map:
            j = index_map[val]
            Lmap[j, :] = Lr
    if has_snr and len(S_rows_sorted) == len(L_rows_sorted):
        for val, Sr in zip(present_mcz_sorted, S_rows_sorted):
            if val in index_map:
                j = index_map[val]
                Smap[j, :] = Sr

    # Determine resolution from first cube
    td_pts = omega_pts = theta_pts = None
    I_value = None
    if cube_paths:
        try:
            with h5py.File(cube_paths[0], "r") as h5:
                td_pts = len(h5["td"])
                omega_pts = len(h5["omega"])
                theta_pts = len(h5["theta"])
                if "I" in h5.attrs:
                    I_value = float(h5.attrs["I"])
        except Exception:
            td_pts = omega_pts = theta_pts = None
            I_value = None

    # Save combined best-match file
    if I_value is None:
        I_value = source_attrs.get("I", 0.5)

    # Ensure I is in source_attrs for the output file
    if "I" not in source_attrs:
        source_attrs["I"] = I_value

    os.makedirs(os.path.join(results_dir, "best_match"), exist_ok=True)
    summary_path = os.path.join(
        results_dir,
        "best_match",
        f"best_match_lindblom_I{_format_min_precision(I_value)}_mcz{_format_min_precision(mcz_min)}-{_format_min_precision(mcz_max)}Msun_td{_format_min_precision(td_min_ms)}-{_format_min_precision(td_max_ms)}ms_td{td_pts}-o{omega_pts}-t{theta_pts}_{orientation_tag}.h5",
    )

    with h5py.File(summary_path, "w") as h5:
        h5.create_dataset("mcz", data=desired_mcz.astype(np.float64))
        h5.create_dataset("td", data=td_arr.astype(np.float64))
        h5.create_dataset("lindblom_at_best_match", data=Lmap.astype(np.float32))
        if has_snr:
            h5.create_dataset("snr_at_best_match", data=Smap.astype(np.float32))

        # Copy source attributes
        for key, value in source_attrs.items():
            h5.attrs[key] = value

    print(f"Lindblom best-match file saved to: {summary_path}")
    print(f"Shape: mcz={len(desired_mcz)}, td={len(td_arr)}")
    print(f"Lindblom value range: {np.nanmin(Lmap):.6f} to {np.nanmax(Lmap):.6f}")
    if has_snr:
        print(f"SNR value range: {np.nanmin(Smap):.6f} to {np.nanmax(Smap):.6f}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Aggregate per-mcz Lindblom cubes into best-match file"
    )
    parser.add_argument(
        "--results_dir",
        type=str,
        required=True,
        help="Directory containing Lindblom cube files",
    )
    parser.add_argument(
        "--td_min_ms",
        type=float,
        required=True,
        help="Minimum time delay in ms",
    )
    parser.add_argument(
        "--td_max_ms",
        type=float,
        required=True,
        help="Maximum time delay in ms",
    )
    parser.add_argument(
        "--mcz_min",
        type=float,
        required=True,
        help="Minimum chirp mass in Msun",
    )
    parser.add_argument(
        "--mcz_max",
        type=float,
        required=True,
        help="Maximum chirp mass in Msun",
    )
    parser.add_argument(
        "--orientation_tag",
        type=str,
        required=True,
        help="Orientation tag (e.g., Taman_edgeon)",
    )

    args = parser.parse_args()
    main(
        results_dir=args.results_dir,
        td_min_ms=args.td_min_ms,
        td_max_ms=args.td_max_ms,
        mcz_min=args.mcz_min,
        mcz_max=args.mcz_max,
        orientation_tag=args.orientation_tag,
    )
