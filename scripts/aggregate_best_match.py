"""Aggregate per-mcz mismatch cubes into one best-match HDF5 file.

Scans results_dir/mismatch_cubes for per-mcz cubes, reduces each across
(theta, omega), stacks over mcz, and writes a combined best_match_*.h5.

Use scripts/create_contour_mcz_td_from_best_match.py to plot the aggregated results.
"""

import os, sys, argparse, glob
import h5py
import numpy as np

# Ensure project root is on path for local invocation
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from modules.filenames import (
    best_match_mcz_td_filename,
    _format_min_precision,
    get_mismatch_cube_resolution,
)
from modules.functions_v3 import timer_decorator


@timer_decorator
def main(
    results_dir: str,
    td_min_ms: float,
    td_max_ms: float,
    mcz_min: float,
    mcz_max: float,
    orientation_tag: str,
):
    cube_paths_all = sorted(
        glob.glob(
            os.path.join(
                results_dir,
                "mismatch_cubes",
                f"mismatch_cubes_mcz*Msun_I*_td{_format_min_precision(td_min_ms)}-{_format_min_precision(td_max_ms)}ms_td*-o*-t*-g*_{orientation_tag}.h5",
            )
        )
    )
    # Filter by requested mcz range by parsing the filename token 'mcz<val>Msun'
    cube_paths = []
    for p in cube_paths_all:
        base = os.path.basename(p)
        try:
            # Example: mismatch_cubes_mcz47Msun_td20-70ms_TAG.h5
            token = base.split("_mcz", 1)[1]
            mcz_str = token.split("Msun", 1)[0]
            mcz_val = float(mcz_str)
        except Exception:
            continue
        if mcz_min <= mcz_val <= mcz_max:
            cube_paths.append(p)
    if not cube_paths:
        raise FileNotFoundError("No mismatch cube files found")

    mcz_vals = []
    Z_rows = []
    O_rows = []
    T_rows = []
    G_rows = []
    td_arr = None
    # Store source parameters from first cube (should be same across all)
    source_attrs = {}

    for p in cube_paths:
        # Skip unreadable/corrupted files gracefully
        try:
            h5 = h5py.File(p, "r")
        except Exception as e:
            print(f"Warning: Skipping unreadable file: {p} ({e})")
            continue
        with h5:
            mcz = float(np.array(h5["mcz"]).item())
            if td_arr is None:
                td_arr = np.array(h5["td"])  # (td,)
                # Extract source parameters from first cube file if available
                if "I" in h5.attrs:
                    source_attrs["I"] = h5.attrs["I"]
                if "theta_J" in h5.attrs:
                    source_attrs["theta_J"] = h5.attrs["theta_J"]
                if "phi_J" in h5.attrs:
                    source_attrs["phi_J"] = h5.attrs["phi_J"]
                if "theta_S" in h5.attrs:
                    source_attrs["theta_S"] = h5.attrs["theta_S"]
                if "phi_S" in h5.attrs:
                    source_attrs["phi_S"] = h5.attrs["phi_S"]
            ep_min_grid = np.array(h5["epsilon_min_grid"])  # (td, theta, omega)
            g_best_grid = np.array(h5["gamma_best_grid"])  # (td, theta, omega)
            theta_arr = np.array(h5["theta"])  # (theta,)
            omega_arr = np.array(h5["omega"])  # (omega,)

            # For each td, find global min over (theta, omega)
            Z = np.zeros(td_arr.shape[0], dtype=np.float32)
            O = np.zeros_like(Z)
            T = np.zeros_like(Z)
            G = np.zeros_like(Z)
            for j in range(td_arr.shape[0]):
                Zgrid = ep_min_grid[j]
                idx = np.unravel_index(int(np.nanargmin(Zgrid)), Zgrid.shape)
                Z[j] = float(Zgrid[idx])
                T[j] = float(theta_arr[idx[0]])
                O[j] = float(omega_arr[idx[1]])
                G[j] = float(g_best_grid[j, idx[0], idx[1]])

            mcz_vals.append(mcz)
            Z_rows.append(Z)
            O_rows.append(O)
            T_rows.append(T)
            G_rows.append(G)

    # Build full mcz grid [mcz_min, ..., mcz_max] with blanks (NaNs) for missing entries
    desired_mcz = np.arange(float(mcz_min), float(mcz_max) + 1.0, 1.0, dtype=np.float64)
    td_len = td_arr.shape[0]
    Zmap = np.full((desired_mcz.shape[0], td_len), np.nan, dtype=np.float32)
    Omap = np.full_like(Zmap, np.nan)
    Tmap = np.full_like(Zmap, np.nan)
    Gmap = np.full_like(Zmap, np.nan)

    # Place available rows at the correct indices
    present_mcz = np.array(mcz_vals, dtype=np.float64)
    order = np.argsort(present_mcz)
    present_mcz_sorted = present_mcz[order]
    Z_rows_sorted = [Z_rows[i] for i in order]
    O_rows_sorted = [O_rows[i] for i in order]
    T_rows_sorted = [T_rows[i] for i in order]
    G_rows_sorted = [G_rows[i] for i in order]
    index_map = {val: idx for idx, val in enumerate(desired_mcz)}
    for val, Zr, Or, Tr, Gr in zip(
        present_mcz_sorted, Z_rows_sorted, O_rows_sorted, T_rows_sorted, G_rows_sorted
    ):
        if val in index_map:
            j = index_map[val]
            Zmap[j, :] = Zr
            Omap[j, :] = Or
            Tmap[j, :] = Tr
            Gmap[j, :] = Gr

    # Determine mcz resolution from number of cubes within range
    mcz_pts = len(cube_paths)
    # Infer td/o/t/g resolution directly from HDF5 contents of the first cube
    # Also extract I from attributes for filename
    td_pts = omega_pts = theta_pts = gamma_pts = None
    I_value = None
    if cube_paths:
        try:
            with h5py.File(cube_paths[0], "r") as h5:
                td_i, omega_i, theta_i, gamma_i = get_mismatch_cube_resolution(h5)
                td_pts = int(td_i) if td_i > 0 else None
                omega_pts = int(omega_i) if omega_i > 0 else None
                theta_pts = int(theta_i) if theta_i > 0 else None
                gamma_pts = int(gamma_i) if gamma_i > 0 else None
                if "I" in h5.attrs:
                    I_value = float(h5.attrs["I"])
        except Exception:
            td_pts = omega_pts = theta_pts = gamma_pts = None
            I_value = None

    # Save combined best-match file with resolution encoded
    if I_value is None:
        raise ValueError("Could not infer I value from mismatch cube files")
    summary_path = best_match_mcz_td_filename(
        results_dir,
        I=I_value,
        mcz_min=mcz_min,
        mcz_max=mcz_max,
        mcz_pts=mcz_pts,
        td_min_ms=td_min_ms,
        td_max_ms=td_max_ms,
        td_pts=td_pts,
        omega_pts=omega_pts,
        theta_pts=theta_pts,
        gamma_pts=gamma_pts,
        orientation_tag=orientation_tag,
    )
    with h5py.File(summary_path, "w") as h5:
        h5.create_dataset("mcz", data=desired_mcz)
        h5.create_dataset("td", data=td_arr.astype(np.float64))
        h5.create_dataset("epsilon_min", data=Zmap.astype(np.float32))
        h5.create_dataset("omega_best", data=Omap.astype(np.float32))
        h5.create_dataset("theta_best", data=Tmap.astype(np.float32))
        h5.create_dataset("gamma_best", data=Gmap.astype(np.float32))
        # Save source parameters as attributes if available
        for key, val in source_attrs.items():
            h5.attrs[key] = val
    print(f"Saved aggregated best-match results: {summary_path}")
    print(f"Use scripts/create_contour_mcz_td_from_best_match.py to plot the results.")


if __name__ == "__main__":
    p = argparse.ArgumentParser(
        description="Aggregate per-mcz mismatch cubes into a combined best-match file."
    )
    p.add_argument("--results_dir", type=str, required=True)
    p.add_argument("--td_min_ms", type=float, required=True)
    p.add_argument("--td_max_ms", type=float, required=True)
    p.add_argument("--mcz_min", type=float, required=True)
    p.add_argument("--mcz_max", type=float, required=True)
    p.add_argument("--orientation_tag", type=str, required=True)
    args = p.parse_args()

    main(
        results_dir=args.results_dir,
        td_min_ms=args.td_min_ms,
        td_max_ms=args.td_max_ms,
        mcz_min=args.mcz_min,
        mcz_max=args.mcz_max,
        orientation_tag=args.orientation_tag,
    )
