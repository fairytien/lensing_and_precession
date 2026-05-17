"""Visualize epsilon contours over (theta, omega) from a mismatch cube.

Loads a per-mcz or per-I mismatch cube HDF5 (same schema: td, theta, omega,
epsilon_min_grid), then:
- Builds a movie (MP4 if ffmpeg available, else GIF) sweeping all available td values
- Optionally writes an interactive HTML slider (Plotly) to scrub td
- Reuses the input cube basename for output files to preserve canonical tokens

"""

import argparse
import os

import h5py
import numpy as np
from modules.plot_utils import apply_physics_paper_style
from scripts.utils._cube_viz import save_contour_movie, save_html_slider

apply_physics_paper_style()


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

        if eps.ndim != 3 or eps.shape[1:] != (theta.size, omega.size):
            raise ValueError(
                f"Unexpected epsilon_min_grid shape {eps.shape}; expected (n_td, {theta.size}, {omega.size})"
            )

    os.makedirs(args.output_dir, exist_ok=True)
    base = os.path.splitext(os.path.basename(args.input_path))[0]
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
    && python -m scripts.utils.visualize_mismatch_cube
    --input_path /work/10000/fairytien33/ls6/lensing_and_precession/data/mismatch/mismatch_cubes/mismatch_cubes_mcz30_I0p5_td20-70x51_omega0-6x61_theta0-15x151_gamma0-2pix51_Taman_edgeon.h5 --gif
"""
