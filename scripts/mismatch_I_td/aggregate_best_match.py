"""Aggregate per-I mismatch cubes into one best-match HDF5 file.

Scans run_dir/mismatch_cubes for per-I cubes, reduces each across
(theta, omega), stacks over I, and writes a combined best_match_*.h5.

Use python -m scripts.mismatch_I_td.plot_contour_I_td_from_best_match to plot
the aggregated results.
"""

import os
import argparse
import h5py
import numpy as np
import logging
from typing import Optional

from modules.filenames import (
    best_match_I_td_filename,
    contour_I_td_run_dir,
    default_mismatch_base_dir,
    find_mismatch_I_cube_files,
    get_mismatch_cube_resolution,
    parse_I_from_mismatch_I_cube_path,
)
from modules.functions import timer_decorator
from modules.bank_io import (
    read_source_attrs,
    read_I_td_grid_attrs,
    read_mismatch_cube_shape,
    I_td_grid_meta_consistent,
    write_missing_I_td_metadata,
    write_orientation_attr,
    write_scalar_attr_with_unit,
    write_dataset_units,
)
from modules.cli_utils import add_I_grid_args, add_td_grid_args, add_redshift_arg

logging.basicConfig(level=logging.INFO, format="[%(levelname)s] %(message)s")


@timer_decorator
def main(
    run_dir: str,
    mcz: float,
    td_min_ms: float,
    td_max_ms: float,
    I_min: float,
    I_max: float,
    orientation_tag: str,
    z: Optional[float],
):
    tol = 1e-6
    z_val = None if z is None else float(z)
    mcz_msun = float(mcz)

    run_dir = contour_I_td_run_dir(
        run_dir,
        mcz=mcz_msun,
        I_min=I_min,
        I_max=I_max,
        td_min_ms=td_min_ms,
        td_max_ms=td_max_ms,
        z=z_val,
        orientation_tag=orientation_tag,
    )
    logging.info(f"Resolved aggregation run directory: {run_dir}")
    logging.info(
        f"Resolved best-match output directory: {os.path.join(run_dir, 'best_match')}"
    )

    cube_paths = find_mismatch_I_cube_files(
        results_dir=run_dir,
        td_min_ms=td_min_ms,
        td_max_ms=td_max_ms,
        mcz_msun=mcz_msun,
        orientation_tag=orientation_tag,
        z=z_val,
        I_min=I_min,
        I_max=I_max,
    )
    if not cube_paths:
        raise FileNotFoundError("No mismatch I-cube files found")

    I_vals = []
    Z_rows = []
    O_rows = []
    T_rows = []
    G_rows = []
    td_arr = None
    ref_shape = None
    warned_shape_mismatch = False
    # Store source parameters from first cube (should be same across all)
    source_attrs = {}
    I_grid_meta = None

    for p in cube_paths:
        # Skip unreadable/corrupted files gracefully
        try:
            h5 = h5py.File(p, "r")
        except Exception as e:
            print(f"Warning: Skipping unreadable file: {p} ({e})")
            continue
        with h5:
            I_val = float(np.array(h5["I"]).item())
            if td_arr is None:
                td_arr = np.array(h5["td"])  # (td,)
                ref_shape = read_mismatch_cube_shape(h5)
                # Extract source parameters from first cube file if available
                source_attrs = read_source_attrs(h5)
                I_grid_meta = read_I_td_grid_attrs(h5)
            else:
                # Light authenticity check: metadata should be consistent across cubes.
                meta_i = read_I_td_grid_attrs(h5)
                if not I_td_grid_meta_consistent(I_grid_meta, meta_i, tol=tol):
                    print(
                        "Warning: Inconsistent I grid metadata across mismatch cubes; "
                        "falling back to discovered I values where needed."
                    )
                    I_grid_meta = {}
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

            I_vals.append(I_val)
            Z_rows.append(Z)
            O_rows.append(O)
            T_rows.append(T)
            G_rows.append(G)

    if td_arr is None or not I_vals:
        raise ValueError("No readable mismatch I-cubes found for aggregation")

    # Canonicalize I values and build expected grid from metadata or discovered files.
    I_vals_arr = np.round(np.array(I_vals, dtype=np.float64), decimals=10)
    discovered_I = np.sort(np.unique(I_vals_arr))

    # Prefer metadata grid if consistent; fallback to discovered values.
    expected_I = discovered_I
    if I_grid_meta and I_grid_meta.get("I_pts", 0) > 0:
        try:
            expected_I = np.round(
                np.linspace(
                    float(I_grid_meta["I_min"]),
                    float(I_grid_meta["I_max"]),
                    int(I_grid_meta["I_pts"]),
                ),
                decimals=10,
            )
        except Exception:
            pass

    # Warn about missing rows (discovered vs expected).
    tol_base = max(tol, 1e-3)
    missing_I = [x for x in expected_I if np.min(np.abs(discovered_I - x)) > tol_base]
    if missing_I:
        print(
            f"Warning: {len(missing_I)} missing I rows: {missing_I[:5]}{'...' if len(missing_I) > 5 else ''}"
        )

    # Build output grids with NaN placeholders for missing rows.
    desired_I = expected_I
    td_len = td_arr.shape[0]
    Zmap = np.full((len(desired_I), td_len), np.nan, dtype=np.float32)
    Omap, Tmap, Gmap = np.copy(Zmap), np.copy(Zmap), np.copy(Zmap)

    # Place rows using tolerance-based index matching.
    row_tol = max(tol, 0.25 * np.min(np.diff(desired_I))) if len(desired_I) > 1 else tol
    for i, (I_val, Zr, Or, Tr, Gr) in enumerate(
        zip(I_vals, Z_rows, O_rows, T_rows, G_rows)
    ):
        j = int(np.argmin(np.abs(desired_I - I_val)))
        if abs(desired_I[j] - I_val) <= row_tol:
            Zmap[j], Omap[j], Tmap[j], Gmap[j] = Zr, Or, Tr, Gr

    I_pts = len(desired_I)
    # Infer td/o/t/g resolution directly from HDF5 contents of the first cube
    td_pts = omega_pts = theta_pts = gamma_pts = None
    omega_min = omega_max = None
    theta_min = theta_max = None
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
                if "MLz" in h5:
                    mlz_arr = np.array(h5["MLz"], dtype=np.float64)
        except Exception:
            td_pts = omega_pts = theta_pts = gamma_pts = None
            omega_min = omega_max = None
            theta_min = theta_max = None
            mlz_arr = None

    # Save combined best-match file with resolution encoded
    I_min_out = float(np.min(desired_I))
    I_max_out = float(np.max(desired_I))
    summary_path = best_match_I_td_filename(
        run_dir,
        mcz_msun=mcz_msun,
        I_min=I_min_out,
        I_max=I_max_out,
        I_pts=I_pts,
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
        h5.create_dataset("I", data=desired_I)
        h5.create_dataset("td", data=td_arr.astype(np.float64))
        h5.create_dataset("mcz", data=np.array([mcz_msun], dtype=np.float64))
        if mlz_arr is not None:
            h5.create_dataset("MLz", data=mlz_arr.astype(np.float64))
        h5.create_dataset("epsilon_min", data=Zmap.astype(np.float32))
        h5.create_dataset("omega_best", data=Omap.astype(np.float32))
        h5.create_dataset("theta_best", data=Tmap.astype(np.float32))
        h5.create_dataset("gamma_best", data=Gmap.astype(np.float32))
        write_dataset_units(
            h5,
            {
                "I": "dimensionless",
                "mcz": "Msun",
                "td": "s",
                "MLz": "s",
                "omega_best": "dimensionless",
                "theta_best": "dimensionless",
                "gamma_best": "rad",
            },
        )
        h5["epsilon_min"].attrs["axis_order"] = "I,td"
        h5["omega_best"].attrs["axis_order"] = "I,td"
        h5["theta_best"].attrs["axis_order"] = "I,td"
        h5["gamma_best"].attrs["axis_order"] = "I,td"
        write_missing_I_td_metadata(
            h5,
            expected_I=np.array(expected_I, dtype=np.float64),
            missing_I=np.array(missing_I, dtype=np.float64),
        )
        # Save source parameters as attributes if available
        for key, val in source_attrs.items():
            h5.attrs[key] = val
        write_scalar_attr_with_unit(h5, "z", z_val, none_as_nan=True)
        write_scalar_attr_with_unit(h5, "mcz_source_msun", mcz_msun)
        write_orientation_attr(h5, orientation_tag)
    print(f"Saved aggregated best-match results: {summary_path}")
    print(
        "Use python -m scripts.mismatch_I_td.plot_contour_I_td_from_best_match to plot the results."
    )


if __name__ == "__main__":
    p = argparse.ArgumentParser(
        description="Aggregate per-I mismatch cubes into a combined best-match file."
    )
    p.add_argument(
        "--run_dir",
        type=str,
        default=default_mismatch_base_dir(),
        help=(
            "Base contour run directory used to read mismatch_cubes/ and write best_match/. "
            "Final tagged run directory is auto-derived if needed."
        ),
    )
    p.add_argument(
        "--mcz",
        type=float,
        required=True,
        help="Source-frame chirp mass in Msun (fixed for the I-td pipeline).",
    )
    add_td_grid_args(
        p,
        default_min_ms=None,
        default_max_ms=None,
        default_pts=None,
        required=True,
    )
    add_I_grid_args(
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
        mcz=args.mcz,
        td_min_ms=args.td_min_ms,
        td_max_ms=args.td_max_ms,
        I_min=args.I_min,
        I_max=args.I_max,
        orientation_tag=args.orientation_tag,
        z=args.z,
    )
