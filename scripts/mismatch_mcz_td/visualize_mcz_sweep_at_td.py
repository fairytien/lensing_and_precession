"""Visualize epsilon contours over (theta, omega) while sweeping chirp mass at fixed td.

Loads multiple per-mcz mismatch cube HDF5 files (created by create_mismatch_mcz_cube),
then:
- Extracts epsilon_min_grid slice at the requested time delay td (ms)
- Builds a movie (MP4 if ffmpeg available, else GIF) sweeping over chirp mass
- Optionally writes an interactive HTML slider (Plotly) to scrub chirp mass

Default input directory points to the same location used by visualize_mismatch_cube if present.
"""

import os
import argparse
from typing import List, Optional

import numpy as np
import h5py
from modules.filenames import (
    find_mismatch_mcz_cube_files,
    parse_mcz_from_mismatch_mcz_cube_path,
)
import matplotlib.pyplot as plt
from modules.plot_utils import apply_physics_paper_style
from scripts.mismatch_mcz_td._viz_utils import (
    infer_orientation_tag_from_filename,
    format_resolution_suffix,
    global_min_max,
    find_td_index,
    save_contour_movie,
    save_html_slider,
)

apply_physics_paper_style()


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
    mcz_msun_list: np.ndarray,
    out_path: str,
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

    # Choose indices evenly spaced across available frames
    panels = min(int(max_panels), n_frames)
    sel_idx = np.linspace(0, n_frames - 1, panels).round().astype(int)
    rows = int(np.ceil(panels / max(int(cols), 1)))

    X, Y = np.meshgrid(omega, theta)
    fig, axes = plt.subplots(
        rows, int(cols), figsize=(3.2 * cols, 2.8 * rows), squeeze=False
    )

    for k, ax in enumerate(axes.flat):
        if k >= panels:
            ax.axis("off")
            continue
        i = int(sel_idx[k])
        Z = eps_grid_over_mcz[i]
        cf = ax.contourf(X, Y, Z, levels=levels, cmap=cmap)
        cbar = fig.colorbar(cf, ax=ax)
        cbar.set_label(r"$\epsilon$")
        ax.set_xlabel(r"$\tilde{\Omega}$")
        ax.set_ylabel(r"$\tilde{\theta}$")
        ax.set_title(f"mcz = {mcz_msun_list[i]:.2f} Msun", fontsize=10)

    fig.tight_layout()
    fig.savefig(out_path, dpi=160, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved grid figure: {out_path}")
    return out_path


def main():
    repo_root = os.path.dirname(
        os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    )

    p = argparse.ArgumentParser(
        description="Make movie and/or slider sweeping mcz at fixed td"
    )
    p.add_argument(
        "--input_dir",
        default=os.path.join(repo_root, "data/mismatch"),
        help=(
            "Run directory that contains mismatch_cubes/ subfolder with per-mcz "
            "mismatch cube HDF5 files"
        ),
    )
    p.add_argument(
        "--orientation_tag",
        type=str,
        default=None,
        help="Filter files by orientation tag (e.g., Taman_edgeon). If omitted, infer from files",
    )
    p.add_argument(
        "--td_ms",
        type=float,
        default=40.0,
        help="Time delay in milliseconds to slice each cube",
    )
    p.add_argument(
        "--output_dir",
        default=os.path.join(repo_root, "figures/mismatch_cubes_mcz_sweep"),
        help="Directory for outputs (movie + slider)",
    )
    p.add_argument("--cmap", type=str, default="jet")
    p.add_argument("--levels", type=int, default=100)
    p.add_argument("--fps", type=int, default=5)
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

    files = find_mismatch_mcz_cube_files(
        results_dir=args.input_dir,
        td_min_ms=None,
        td_max_ms=None,
        orientation_tag=args.orientation_tag or "*",
    )
    if not files:
        raise FileNotFoundError(
            "No mismatch cube files found in "
            f"{args.input_dir} (tag={args.orientation_tag}). "
            "Expected cube .h5 files under input_dir/mismatch_cubes/."
        )

    # Load axes consistency and collect per-mcz slices at fixed td
    mcz_msun_vals: List[float] = []
    eps_slices: List[np.ndarray] = []
    theta_ref: Optional[np.ndarray] = None
    omega_ref: Optional[np.ndarray] = None
    res_suffix: Optional[str] = None
    orient_tag: Optional[str] = args.orientation_tag

    for fp in files:
        with h5py.File(fp, "r") as h5:
            for ds in ("td", "theta", "omega", "epsilon_min_grid"):
                if ds not in h5:
                    raise KeyError(
                        f"Dataset '{ds}' missing in {fp}; found keys: {list(h5.keys())}"
                    )

            theta = np.array(h5["theta"], dtype=float)
            omega = np.array(h5["omega"], dtype=float)
            td = np.array(h5["td"], dtype=float)
            eps = np.array(h5["epsilon_min_grid"], dtype=float)

            if theta_ref is None:
                theta_ref = theta
                omega_ref = omega
                res_suffix = format_resolution_suffix(h5)
                if orient_tag is None:
                    orient_tag = infer_orientation_tag_from_filename(fp)
            else:
                if theta.shape != theta_ref.shape or not np.allclose(theta, theta_ref):
                    raise ValueError(
                        "Theta grids differ across cubes; cannot sweep mcz coherently"
                    )
                if omega.shape != omega_ref.shape or not np.allclose(omega, omega_ref):
                    raise ValueError(
                        "Omega grids differ across cubes; cannot sweep mcz coherently"
                    )

            td_idx = find_td_index(td, args.td_ms)
            if eps.ndim != 3 or eps.shape[1:] != (theta.size, omega.size):
                raise ValueError(
                    f"Unexpected epsilon_min_grid shape {eps.shape}; expected (n_td, {theta.size}, {omega.size})"
                )

            slice_eps = eps[td_idx]
            eps_slices.append(slice_eps)
            mcz_msun_vals.append(_infer_mcz_msun_numeric(h5))

    # Sort by mcz
    mcz_msun_arr = np.array(mcz_msun_vals, dtype=float)
    order = np.argsort(mcz_msun_arr)
    mcz_msun_arr = mcz_msun_arr[order]
    eps_stack = np.stack(
        [eps_slices[i] for i in order], axis=0
    )  # (n_mcz, n_theta, n_omega)

    assert theta_ref is not None and omega_ref is not None
    theta_ref = np.asarray(theta_ref)
    omega_ref = np.asarray(omega_ref)

    os.makedirs(args.output_dir, exist_ok=True)
    tag = orient_tag or "unknown"
    td_tag = f"td{args.td_ms:.1f}".replace(".", "p")
    mcz_msun_min = f"{float(np.nanmin(mcz_msun_arr)):g}".replace(".", "p")
    mcz_msun_max = f"{float(np.nanmax(mcz_msun_arr)):g}".replace(".", "p")
    base = f"epsilon_cube_mcz_sweep_{td_tag}_mcz{mcz_msun_min}-{mcz_msun_max}_{res_suffix}_{tag}"
    movie_ext = ".mp4" if (args.mp4 and not args.gif) else ".gif"
    movie_path = os.path.join(args.output_dir, base + movie_ext)

    # Movie over mcz
    save_contour_movie(
        omega=omega_ref,
        theta=theta_ref,
        eps_grid=eps_stack,
        sweep_values=mcz_msun_arr,
        out_path=movie_path,
        sweep_label="mcz",
        sweep_fmt="{:.2f} Msun",
        cmap=args.cmap,
        levels=args.levels,
        fps=args.fps,
    )

    # HTML slider over mcz
    if args.html:
        html_path = os.path.join(args.output_dir, base + ".html")
        save_html_slider(
            omega=omega_ref,
            theta=theta_ref,
            eps_grid=eps_stack,
            sweep_values=mcz_msun_arr,
            out_path=html_path,
            sweep_label="mcz",
            sweep_fmt="{:.2f} Msun",
            cmap="Jet",
            levels=args.levels,
        )

    # Grid with individual colorbars
    if args.grid:
        grid_path = os.path.join(args.output_dir, base + "_grid.png")
        save_grid_with_individual_colorbars(
            omega=omega_ref,
            theta=theta_ref,
            eps_grid_over_mcz=eps_stack,
            mcz_msun_list=mcz_msun_arr,
            out_path=grid_path,
            cmap=args.cmap,
            levels=args.levels,
            cols=args.grid_cols,
            max_panels=args.grid_max_panels,
        )


if __name__ == "__main__":
    main()
