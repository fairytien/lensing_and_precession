"""Aggregate per-mcz mismatch cubes into one best-match HDF5 file.

Scans run_dir/mismatch_cubes for per-mcz cubes, reduces each across
(theta, omega), stacks over mcz, and writes a combined best_match_*.h5.

Use python -m scripts.mismatch_mcz_td.plot_contour_mcz_td_from_best_match to plot
the aggregated results.
"""

import os
import argparse
import h5py
import numpy as np
import logging
from typing import Optional

from modules.filenames import (
    best_match_mcz_td_filename,
    contour_run_dir,
    find_mismatch_cube_files,
    get_mismatch_cube_resolution,
)
from modules.functions_v3 import timer_decorator
from modules.bank_io import (
    read_source_attrs,
    read_mcz_grid_attrs,
    read_mismatch_cube_shape,
    mcz_grid_meta_consistent,
    write_missing_mcz_metadata,
    write_orientation_attr,
    write_scalar_attr_with_unit,
    write_dataset_units,
)
from modules.cli_utils import add_mcz_grid_args, add_td_grid_args, add_redshift_arg

logging.basicConfig(level=logging.INFO, format="[%(levelname)s] %(message)s")


@timer_decorator
def main(
    run_dir: str,
    I: float,
    td_min_ms: float,
    td_max_ms: float,
    mcz_min: float,
    mcz_max: float,
    orientation_tag: str,
    z: Optional[float],
):
    tol = 1e-6
    z_val = None if z is None else float(z)
    run_dir = contour_run_dir(
        run_dir,
        I=I,
        mcz_min=mcz_min,
        mcz_max=mcz_max,
        td_min_ms=td_min_ms,
        td_max_ms=td_max_ms,
        z=z_val,
        orientation_tag=orientation_tag,
    )
    logging.info(f"Resolved aggregation run directory: {run_dir}")
    logging.info(
        f"Resolved best-match output directory: {os.path.join(run_dir, 'best_match')}"
    )

    cube_paths = find_mismatch_cube_files(
        results_dir=run_dir,
        td_min_ms=td_min_ms,
        td_max_ms=td_max_ms,
        orientation_tag=orientation_tag,
        z=z_val,
        mcz_min=mcz_min,
        mcz_max=mcz_max,
    )
    if not cube_paths:
        raise FileNotFoundError("No mismatch cube files found")

    mcz_msun_vals = []
    Z_rows = []
    O_rows = []
    T_rows = []
    G_rows = []
    td_arr = None
    ref_shape = None
    warned_shape_mismatch = False
    # Store source parameters from first cube (should be same across all)
    source_attrs = {}
    mcz_grid_meta = None

    for p in cube_paths:
        # Skip unreadable/corrupted files gracefully
        try:
            h5 = h5py.File(p, "r")
        except Exception as e:
            print(f"Warning: Skipping unreadable file: {p} ({e})")
            continue
        with h5:
            mcz_msun = float(np.array(h5["mcz"]).item())
            if td_arr is None:
                td_arr = np.array(h5["td"])  # (td,)
                ref_shape = read_mismatch_cube_shape(h5)
                # Extract source parameters from first cube file if available
                source_attrs = read_source_attrs(h5)
                mcz_grid_meta = read_mcz_grid_attrs(h5)
            else:
                # Light authenticity check: metadata should be consistent across cubes.
                meta_i = read_mcz_grid_attrs(h5)
                if not mcz_grid_meta_consistent(mcz_grid_meta, meta_i, tol=tol):
                    print(
                        "Warning: Inconsistent mcz grid metadata across mismatch cubes; "
                        "falling back to discovered mcz values where needed."
                    )
                    mcz_grid_meta = {}
                if not warned_shape_mismatch:
                    shape_i = read_mismatch_cube_shape(h5)
                    if ref_shape is not None and shape_i != ref_shape:
                        print(
                            "Warning: Inconsistent axis sizes across mismatch cubes "
                            "(td/theta/omega/gamma). Results may be partial for mismatched files."
                        )
                        warned_shape_mismatch = True
            # (td, theta, omega)
            ep_min_grid = np.array(h5["epsilon_min_grid"])
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

            mcz_msun_vals.append(mcz_msun)
            Z_rows.append(Z)
            O_rows.append(O)
            T_rows.append(T)
            G_rows.append(G)

    if td_arr is None or not mcz_msun_vals:
        raise ValueError("No readable mismatch cubes found for aggregation")

    # Use the mcz values discovered from mismatch cubes directly.
    # Canonicalize tiny floating noise to keep logically identical points together.
    mcz_msun_vals_arr = np.round(np.array(mcz_msun_vals, dtype=np.float64), decimals=10)
    desired_mcz_msun = np.sort(np.unique(mcz_msun_vals_arr))

    # Warning-only completeness check for internal missing mcz rows.
    # Prefer the exact compute grid saved by Stage 1; fallback to discovered rows.
    expected_mcz_msun = desired_mcz_msun.copy()
    if mcz_grid_meta:
        try:
            mcz_min_msun_attr = float(mcz_grid_meta["mcz_min"])
            mcz_max_msun_attr = float(mcz_grid_meta["mcz_max"])
            mcz_pts_attr = int(mcz_grid_meta["mcz_pts"])
            if mcz_pts_attr > 0:
                expected_mcz_msun = np.linspace(
                    mcz_min_msun_attr,
                    mcz_max_msun_attr,
                    mcz_pts_attr,
                    dtype=np.float64,
                )
        except Exception:
            expected_mcz_msun = desired_mcz_msun.copy()

    expected_mcz_msun = np.round(expected_mcz_msun, decimals=10)

    missing_mcz_msun = []
    for x in expected_mcz_msun:
        if np.min(np.abs(desired_mcz_msun - x)) > max(tol, 1e-3 * (abs(x) + 1.0)):
            missing_mcz_msun.append(float(x))

    if missing_mcz_msun:
        preview = ", ".join(f"{v:g}" for v in missing_mcz_msun[:10])
        suffix = "..." if len(missing_mcz_msun) > 10 else ""
        print(
            f"Warning: Detected {len(missing_mcz_msun)} missing mcz rows within requested range: {preview}{suffix}"
        )

    # Use expected mcz grid as output axis so missing rows remain explicit NaNs for plotting.
    desired_mcz_msun = expected_mcz_msun

    td_len = td_arr.shape[0]
    Zmap = np.full((desired_mcz_msun.shape[0], td_len), np.nan, dtype=np.float32)
    Omap = np.full_like(Zmap, np.nan)
    Tmap = np.full_like(Zmap, np.nan)
    Gmap = np.full_like(Zmap, np.nan)

    # Place available rows by nearest expected mcz index (with small tolerance).
    present_mcz_msun = np.round(np.array(mcz_msun_vals, dtype=np.float64), decimals=10)
    order = np.argsort(present_mcz_msun)
    present_mcz_msun_sorted = present_mcz_msun[order]
    Z_rows_sorted = [Z_rows[i] for i in order]
    O_rows_sorted = [O_rows[i] for i in order]
    T_rows_sorted = [T_rows[i] for i in order]
    G_rows_sorted = [G_rows[i] for i in order]
    if desired_mcz_msun.shape[0] > 1:
        pos_diffs = np.diff(desired_mcz_msun)
        pos_diffs = pos_diffs[pos_diffs > tol]
        row_tol = (
            max(tol, 0.25 * float(np.min(pos_diffs))) if pos_diffs.size > 0 else tol
        )
    else:
        row_tol = tol

    for val, Zr, Or, Tr, Gr in zip(
        present_mcz_msun_sorted,
        Z_rows_sorted,
        O_rows_sorted,
        T_rows_sorted,
        G_rows_sorted,
    ):
        j = int(np.argmin(np.abs(desired_mcz_msun - val)))
        if abs(float(desired_mcz_msun[j]) - float(val)) <= row_tol:
            Zmap[j, :] = Zr
            Omap[j, :] = Or
            Tmap[j, :] = Tr
            Gmap[j, :] = Gr

    # Determine mcz resolution from discovered unique mcz values
    mcz_pts = int(desired_mcz_msun.shape[0])
    # Infer td/o/t/g resolution directly from HDF5 contents of the first cube
    # Also extract I from attributes for filename
    td_pts = omega_pts = theta_pts = gamma_pts = None
    omega_min = omega_max = None
    theta_min = theta_max = None
    I_value = None
    mlz_arr = None
    if cube_paths:
        try:
            with h5py.File(cube_paths[0], "r") as h5:
                td_i, omega_i, theta_i, gamma_i = get_mismatch_cube_resolution(h5)
                td_pts = int(td_i) if td_i > 0 else None
                omega_pts = int(omega_i) if omega_i > 0 else None
                theta_pts = int(theta_i) if theta_i > 0 else None
                gamma_pts = int(gamma_i) if gamma_i > 0 else None
                if "omega" in h5:
                    omega_arr_i = np.array(h5["omega"], dtype=np.float64)
                    if omega_arr_i.size > 0:
                        omega_min = float(np.nanmin(omega_arr_i))
                        omega_max = float(np.nanmax(omega_arr_i))
                if "theta" in h5:
                    theta_arr_i = np.array(h5["theta"], dtype=np.float64)
                    if theta_arr_i.size > 0:
                        theta_min = float(np.nanmin(theta_arr_i))
                        theta_max = float(np.nanmax(theta_arr_i))
                if "I" in h5.attrs:
                    I_value = float(h5.attrs["I"])
                if "MLz" in h5:
                    mlz_arr = np.array(h5["MLz"], dtype=np.float64)
        except Exception:
            td_pts = omega_pts = theta_pts = gamma_pts = None
            omega_min = omega_max = None
            theta_min = theta_max = None
            I_value = None
            mlz_arr = None

    # Save combined best-match file with resolution encoded
    if I_value is None:
        raise ValueError("Could not infer I value from mismatch cube files")
    mcz_min_msun_out = float(np.min(desired_mcz_msun))
    mcz_max_msun_out = float(np.max(desired_mcz_msun))
    summary_path = best_match_mcz_td_filename(
        run_dir,
        I=I_value,
        mcz_min=mcz_min_msun_out,
        mcz_max=mcz_max_msun_out,
        mcz_pts=mcz_pts,
        td_min_ms=td_min_ms,
        td_max_ms=td_max_ms,
        td_pts=td_pts,
        omega_min=omega_min,
        omega_max=omega_max,
        omega_pts=omega_pts,
        theta_min=theta_min,
        theta_max=theta_max,
        theta_pts=theta_pts,
        gamma_pts=gamma_pts,
        orientation_tag=orientation_tag,
        z=z_val,
    )
    with h5py.File(summary_path, "w") as h5:
        h5.create_dataset("mcz", data=desired_mcz_msun)
        h5.create_dataset("td", data=td_arr.astype(np.float64))
        if mlz_arr is not None:
            h5.create_dataset("MLz", data=mlz_arr.astype(np.float64))
        h5.create_dataset("epsilon_min", data=Zmap.astype(np.float32))
        h5.create_dataset("omega_best", data=Omap.astype(np.float32))
        h5.create_dataset("theta_best", data=Tmap.astype(np.float32))
        h5.create_dataset("gamma_best", data=Gmap.astype(np.float32))
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
        write_missing_mcz_metadata(
            h5,
            expected_mcz=np.array(expected_mcz_msun, dtype=np.float64),
            missing_mcz=np.array(missing_mcz_msun, dtype=np.float64),
        )
        # Save source parameters as attributes if available
        for key, val in source_attrs.items():
            h5.attrs[key] = val
        write_scalar_attr_with_unit(h5, "z", z_val, none_as_nan=True)
        write_orientation_attr(h5, orientation_tag)
    print(f"Saved aggregated best-match results: {summary_path}")
    print(
        "Use python -m scripts.mismatch_mcz_td.plot_contour_mcz_td_from_best_match to plot the results."
    )


if __name__ == "__main__":
    p = argparse.ArgumentParser(
        description="Aggregate per-mcz mismatch cubes into a combined best-match file."
    )
    p.add_argument(
        "--run_dir",
        type=str,
        required=True,
        help=(
            "Base contour run directory used to read mismatch_cubes/ and write best_match/. "
            "Final tagged run directory is auto-derived if needed."
        ),
    )
    p.add_argument(
        "--I",
        type=float,
        required=True,
        help="Flux ratio used to resolve the run directory (I token in path).",
    )
    add_td_grid_args(
        p,
        default_min_ms=None,
        default_max_ms=None,
        default_pts=None,
        required=True,
    )
    add_mcz_grid_args(
        p,
        default_min=None,
        default_max=None,
        default_pts=None,
        required=True,
    )
    add_redshift_arg(p, default_z=None)
    p.add_argument("--orientation_tag", type=str, required=True)
    args = p.parse_args()

    main(
        run_dir=args.run_dir,
        I=args.I,
        td_min_ms=args.td_min_ms,
        td_max_ms=args.td_max_ms,
        mcz_min=args.mcz_min,
        mcz_max=args.mcz_max,
        orientation_tag=args.orientation_tag,
        z=args.z,
    )
