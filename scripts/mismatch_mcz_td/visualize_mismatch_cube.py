"""Visualize epsilon contours over (theta, omega) from a mismatch cube.

Loads a per-mcz mismatch cube HDF5 (with datasets created by create_mismatch_cube),
then:
- Builds a movie (MP4 if ffmpeg available, else GIF) sweeping all available td values
- Optionally writes an interactive HTML slider (Plotly) to scrub td

"""

import os
import argparse

import numpy as np
import h5py
from modules.filenames import parse_mcz_from_mismatch_cube_path
from modules.plot_utils import apply_physics_paper_style
from scripts.mismatch_mcz_td._viz_utils import (
    infer_orientation_tag_from_filename,
    format_resolution_suffix,
    save_contour_movie,
    save_html_slider,
)

apply_physics_paper_style()


def _infer_mcz_from_filename(path: str) -> str:
    """Extract mcz value token from a cube filename.

    Returns values like "70" or "70p5".
    Returns "unknown" if the mcz cannot be inferred.
    """
    val = parse_mcz_from_mismatch_cube_path(path)
    if val is not None:
        return f"{float(val):g}".replace(".", "p")
    return "unknown"


def _format_td_range_tag(td_s: np.ndarray) -> str:
    """Build td range token from td dataset values in seconds."""
    td_ms = np.asarray(td_s, dtype=float) * 1e3
    if td_ms.size == 0:
        return "td-unknown"
    td_min = f"{float(np.nanmin(td_ms)):g}".replace(".", "p")
    td_max = f"{float(np.nanmax(td_ms)):g}".replace(".", "p")
    return f"td{td_min}-{td_max}"


def main():
    repo_root = os.path.dirname(
        os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    )

    p = argparse.ArgumentParser(
        description="Make movie and/or slider from mismatch cube"
    )
    p.add_argument(
        "--input_path",
        required=True,
        help=("Path to mismatch cube HDF5."),
    )
    p.add_argument(
        "--output_dir",
        default=os.path.join(repo_root, "figures/mismatch_cubes"),
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
    args = p.parse_args()

    if not os.path.isfile(args.input_path):
        raise FileNotFoundError(f"Input cube not found: {args.input_path}")

    with h5py.File(args.input_path, "r") as h5:
        for ds in ("td", "theta", "omega", "epsilon_min_grid"):
            if ds not in h5:
                raise KeyError(
                    f"Dataset '{ds}' missing in {args.input_path}; found keys: {list(h5.keys())}"
                )

        td = np.array(h5["td"], dtype=float)  # seconds
        theta = np.array(h5["theta"], dtype=float)
        omega = np.array(h5["omega"], dtype=float)
        eps = np.array(h5["epsilon_min_grid"], dtype=float)  # (td, theta, omega)

        # Get resolution suffix before orientation tag
        res_suffix = format_resolution_suffix(h5)

    if eps.ndim != 3 or eps.shape[1:] != (theta.size, omega.size):
        raise ValueError(
            f"Unexpected epsilon_min_grid shape {eps.shape}; expected (n_td, {theta.size}, {omega.size})"
        )

    os.makedirs(args.output_dir, exist_ok=True)
    tag = infer_orientation_tag_from_filename(args.input_path)
    mcz_msun_tag = _infer_mcz_from_filename(args.input_path)
    td_range_tag = _format_td_range_tag(td)

    base = f"epsilon_cube_td_sweep_mcz{mcz_msun_tag}_{td_range_tag}_{res_suffix}_{tag}"
    movie_ext = ".mp4" if (args.mp4 and not args.gif) else ".gif"
    movie_path = os.path.join(args.output_dir, base + movie_ext)

    td_ms = td * 1e3

    # Movie
    save_contour_movie(
        omega=omega,
        theta=theta,
        eps_grid=eps,
        sweep_values=td_ms,
        out_path=movie_path,
        sweep_label="td",
        sweep_fmt="{:.1f} ms",
        cmap=args.cmap,
        levels=args.levels,
        fps=args.fps,
    )

    # HTML slider
    if args.html:
        html_path = os.path.join(args.output_dir, base + ".html")
        save_html_slider(
            omega=omega,
            theta=theta,
            eps_grid=eps,
            sweep_values=td_ms,
            out_path=html_path,
            sweep_label="td",
            sweep_fmt="{:.1f} ms",
            cmap="Jet",  # Plotly colorscale name
            levels=args.levels,
        )


if __name__ == "__main__":
    main()


"""
Example CLI Usage on TACC:
    conda activate fairytien_gw 
    && python -m scripts.mismatch_mcz_td.visualize_mismatch_cube 
    --input_path /work/10000/fairytien33/ls6/lensing_and_precession/data/mismatch/mismatch_cubes/mismatch_cubes_mcz30_I0p5_td20-70x51_omega0-6x61_theta0-15x151_gamma0-2pix51_Taman_edgeon.h5 --gif
"""
