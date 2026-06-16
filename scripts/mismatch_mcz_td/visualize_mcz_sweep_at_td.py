"""Visualize epsilon contours over (theta, omega) while sweeping chirp mass at fixed td.

Loads multiple per-mcz mismatch cube HDF5 files (created by create_mcz_mismatch_cube),
then:
- Extracts epsilon_min_grid slice at the requested time delay td (ms)
- Builds a movie (MP4 if ffmpeg available, else GIF) sweeping over chirp mass
- Optionally writes an interactive HTML slider (Plotly) to scrub chirp mass

Input directory must point to a canonical mismatch_cubes/ directory for one
mcz_td run; run-level tokens are inferred from its parent path.
"""

import glob
import os
import argparse
from typing import List, Optional

import numpy as np
import h5py
from modules.filenames import (
    mismatch_sweep_mcz_td_filename,
    parse_mcz_td_run_dir_metadata,
    parse_template_grid_tokens,
    parse_mcz_from_mismatch_mcz_cube_path,
)
import matplotlib.pyplot as plt
from modules.plot_utils import (
    LBL_EPS_LP,
    apply_physics_paper_style,
    draw_omega_theta_contour_panel,
    format_colorbar_ticks,
    resolve_contour_vlim,
    save_figure,
)
from scripts.utils._cube_viz import (
    find_td_index,
    save_contour_movie,
    save_html_slider,
)

apply_physics_paper_style(base_font=16, label_font=20, tick_font=16, legend_font=14)


def _infer_mcz_msun_numeric(h5_file) -> float:
    """Read numeric mcz from HDF5 dataset 'mcz'."""
    if "mcz" in h5_file:
        val = np.array(h5_file["mcz"]).astype(float).ravel()
        if val.size >= 1:
            return float(val[0])
    # Fallback to filename parser for canonical naming.
    parsed = parse_mcz_from_mismatch_mcz_cube_path(h5_file.filename)
    if parsed is not None:
        return float(parsed)
    return float("nan")


def save_grid_with_individual_colorbars(
    omega: np.ndarray,
    theta: np.ndarray,
    eps_grid_over_mcz: np.ndarray,
    gamma_grid_over_mcz: np.ndarray,
    mcz_msun_list: np.ndarray,
    out_path: str,
    *,
    vmin: float,
    vmax: float,
    saturated: bool,
    cmap: str = "jet",
    levels: int = 100,
    cols: int = 4,
    max_panels: int = 12,
) -> str:
    """Create a static grid figure with each panel having its own colorbar.

    Slices a subset of mcz frames (evenly spaced up to max_panels).
    """
    os.makedirs(os.path.dirname(out_path), exist_ok=True)

    n_frames = eps_grid_over_mcz.shape[0]
    if n_frames == 0:
        raise ValueError("No frames available to plot")

    panels = min(int(max_panels), n_frames)
    sel_idx = np.linspace(0, n_frames - 1, panels).round().astype(int)
    rows = int(np.ceil(panels / max(int(cols), 1)))

    X, Y = np.meshgrid(omega, theta)
    fig, axes = plt.subplots(
        rows, int(cols), figsize=(3.8 * cols, 3.4 * rows), squeeze=False
    )

    for k, ax in enumerate(axes.flat):
        if k >= panels:
            ax.axis("off")
            continue
        i = int(sel_idx[k])
        cf = draw_omega_theta_contour_panel(
            ax,
            X,
            Y,
            eps_grid_over_mcz[i],
            gamma_grid_over_mcz[i],
            mcz_msun=float(mcz_msun_list[i]),
            vmin=vmin,
            vmax=vmax,
            saturated=saturated,
            levels=levels,
            cmap=cmap,
        )
        cbar = fig.colorbar(cf, ax=ax)
        cbar.set_label(LBL_EPS_LP)
        format_colorbar_ticks(cbar, vmin, vmax, decimals=2)

    fig.tight_layout()
    save_figure(fig, out_path)
    return out_path


def main():
    p = argparse.ArgumentParser(
        description="Make movie and/or slider sweeping mcz at fixed td"
    )
    p.add_argument(
        "--input_dir",
        required=True,
        help=(
            "Canonical mismatch_cubes/ directory for one mcz_td run, e.g. "
            "data/mismatch_I0p5_z1_mcz5-45_td20-70_Taman_faceon/mismatch_cubes"
        ),
    )
    p.add_argument(
        "--td_ms",
        type=float,
        default=40.0,
        help="Time delay in milliseconds to slice each cube",
    )
    p.add_argument(
        "--output_dir",
        default=os.path.join(
            os.path.dirname(
                os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
            ),
            "figures/mismatch_cubes_mcz_sweep",
        ),
        help="Directory for outputs (movie + slider)",
    )
    p.add_argument("--cmap", type=str, default="jet")
    p.add_argument("--levels", type=int, default=100)
    p.add_argument("--fps", type=int, default=5)
    p.add_argument(
        "--vmax",
        type=float,
        default=0.5,
        help="Saturate the colorbar at this epsilon value (default: 0.5)",
    )
    p.add_argument(
        "--mp4",
        action="store_true",
        help="Force MP4; fallback to GIF if ffmpeg missing",
    )
    p.add_argument("--gif", action="store_true", help="Force GIF output")
    p.add_argument("--html", action="store_true", help="Generate HTML slider output")
    p.add_argument(
        "--grid",
        action="store_true",
        help="Also create a multi-panel static figure with individual colorbars",
    )
    p.add_argument("--grid_cols", type=int, default=4)
    p.add_argument("--grid_max_panels", type=int, default=12)
    args = p.parse_args()

    cube_dir = os.path.normpath(args.input_dir)
    if os.path.basename(cube_dir) != "mismatch_cubes":
        raise ValueError(
            f"--input_dir must point to a mismatch_cubes/ directory, got: {args.input_dir}"
        )

    run_meta = parse_mcz_td_run_dir_metadata(cube_dir)
    if run_meta is None:
        raise ValueError(
            f"Could not parse canonical mcz_td run metadata from input_dir: {args.input_dir}"
        )

    files = sorted(glob.glob(os.path.join(cube_dir, "*.h5")))
    if not files:
        raise FileNotFoundError(
            f"No mismatch cube files found in {args.input_dir}. Expected .h5 files directly inside mismatch_cubes/."
        )

    mcz_msun_vals: List[float] = []
    eps_slices: List[np.ndarray] = []
    gamma_slices: List[np.ndarray] = []
    theta_ref: Optional[np.ndarray] = None
    omega_ref: Optional[np.ndarray] = None
    gamma_ref: Optional[np.ndarray] = None
    grid_params: Optional[dict] = None

    for fp in files:
        with h5py.File(fp, "r") as h5:
            for ds in (
                "td",
                "theta",
                "omega",
                "gamma",
                "epsilon_min_grid",
                "gamma_best_grid",
            ):
                if ds not in h5:
                    raise KeyError(
                        f"Dataset '{ds}' missing in {fp}; found keys: {list(h5.keys())}"
                    )

            theta = np.array(h5["theta"], dtype=float)
            omega = np.array(h5["omega"], dtype=float)
            gamma = np.array(h5["gamma"], dtype=float)
            td = np.array(h5["td"], dtype=float)
            eps = np.array(h5["epsilon_min_grid"], dtype=float)
            gamma_best = np.array(h5["gamma_best_grid"], dtype=float)

            raw_I = h5.attrs.get("I")
            if raw_I is None or not np.isclose(float(raw_I), float(run_meta["I"])):
                raise ValueError(
                    f"Cube I metadata does not match input_dir run metadata: {fp}"
                )

            raw_z = h5.attrs.get("z")
            if raw_z is None or not np.isclose(
                float(raw_z), float(run_meta["z"]), equal_nan=True
            ):
                raise ValueError(
                    f"Cube z metadata does not match input_dir run metadata: {fp}"
                )

            raw_orientation = h5.attrs.get("orientation_tag")
            if raw_orientation is None or str(raw_orientation).strip() != str(
                run_meta["orientation_tag"]
            ):
                raise ValueError(
                    f"Cube orientation_tag does not match input_dir run metadata: {fp}"
                )

            if theta_ref is None:
                theta_ref = theta
                omega_ref = omega
                gamma_ref = gamma

                grid_params = parse_template_grid_tokens(fp)
                if grid_params is None:
                    raise ValueError(
                        f"Could not parse template-grid token from canonical cube filename: {fp}"
                    )
            else:
                if theta.shape != theta_ref.shape or not np.allclose(theta, theta_ref):
                    raise ValueError(
                        "Theta grids differ across cubes; cannot sweep mcz coherently"
                    )
                if omega.shape != omega_ref.shape or not np.allclose(omega, omega_ref):
                    raise ValueError(
                        "Omega grids differ across cubes; cannot sweep mcz coherently"
                    )
                if gamma_ref is not None and (
                    gamma.shape != gamma_ref.shape or not np.allclose(gamma, gamma_ref)
                ):
                    raise ValueError(
                        "Gamma grids differ across cubes; cannot sweep mcz coherently"
                    )

            td_idx = find_td_index(td, args.td_ms)
            if eps.ndim != 3 or eps.shape[1:] != (theta.size, omega.size):
                raise ValueError(
                    f"Unexpected epsilon_min_grid shape {eps.shape}; expected (n_td, {theta.size}, {omega.size})"
                )
            eps_slices.append(eps[td_idx])
            gamma_slices.append(gamma_best[td_idx])
            mcz_msun_vals.append(_infer_mcz_msun_numeric(h5))

    mcz_msun_arr = np.array(mcz_msun_vals, dtype=float)
    order = np.argsort(mcz_msun_arr)
    mcz_msun_arr = mcz_msun_arr[order]
    eps_stack = np.stack([eps_slices[i] for i in order], axis=0)
    gamma_stack = np.stack([gamma_slices[i] for i in order], axis=0)

    assert theta_ref is not None and omega_ref is not None
    theta_ref = np.asarray(theta_ref)
    omega_ref = np.asarray(omega_ref)
    assert grid_params is not None

    vmin, vmax, saturated = resolve_contour_vlim(eps_stack, vmax_cap=args.vmax)

    os.makedirs(args.output_dir, exist_ok=True)
    movie_ext = "mp4" if (args.mp4 and not args.gif) else "gif"
    movie_path = mismatch_sweep_mcz_td_filename(
        fig_dir=args.output_dir,
        I=float(run_meta["I"]),
        td_ms=float(args.td_ms),
        z=float(run_meta["z"]),
        mcz_min=float(run_meta["mcz_min"]),
        mcz_max=float(run_meta["mcz_max"]),
        omega_min=float(grid_params["omega_min"]),
        omega_max=float(grid_params["omega_max"]),
        omega_pts=int(grid_params["omega_pts"]),
        theta_min=float(grid_params["theta_min"]),
        theta_max=float(grid_params["theta_max"]),
        theta_pts=int(grid_params["theta_pts"]),
        gamma_pts=int(grid_params["gamma_pts"]),
        orientation_tag=str(run_meta["orientation_tag"]),
        ext=movie_ext,
    )
    base, _ = os.path.splitext(movie_path)

    save_contour_movie(
        omega=omega_ref,
        theta=theta_ref,
        eps_grid=eps_stack,
        gamma_grid=gamma_stack,
        sweep_values=mcz_msun_arr,
        out_path=movie_path,
        sweep_label="mcz",
        sweep_fmt="{:.2f} Msun",
        cmap=args.cmap,
        levels=args.levels,
        fps=args.fps,
        vmax_cap=args.vmax,
    )

    if args.html:
        html_path = base + ".html"
        save_html_slider(
            omega=omega_ref,
            theta=theta_ref,
            eps_grid=eps_stack,
            gamma_grid=gamma_stack,
            sweep_values=mcz_msun_arr,
            out_path=html_path,
            sweep_label="mcz",
            sweep_fmt="{:.2f} Msun",
            cmap="Jet",
            levels=args.levels,
            vmax_cap=args.vmax,
        )

    if args.grid:
        grid_path = base + "_grid.png"
        save_grid_with_individual_colorbars(
            omega=omega_ref,
            theta=theta_ref,
            eps_grid_over_mcz=eps_stack,
            gamma_grid_over_mcz=gamma_stack,
            mcz_msun_list=mcz_msun_arr,
            out_path=grid_path,
            vmin=vmin,
            vmax=vmax,
            saturated=saturated,
            cmap=args.cmap,
            levels=args.levels,
            cols=args.grid_cols,
            max_panels=args.grid_max_panels,
        )


if __name__ == "__main__":
    main()
