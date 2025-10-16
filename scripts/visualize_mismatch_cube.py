"""Visualize epsilon contours over (theta, omega) from a mismatch cube.

Loads a per-mcz mismatch cube HDF5 (with datasets created by create_mismatch_cube),
then:
- Builds a movie (MP4 if ffmpeg available, else GIF) sweeping td=20–70 ms
- Optionally writes an interactive HTML slider (Plotly) to scrub td

Default input points to the mcz=50 Msun, td=20–70 ms cube if present.
"""

import os
import sys
import argparse
import re
from typing import Optional, Tuple

import numpy as np
import h5py
import matplotlib.pyplot as plt
from matplotlib import animation


def _infer_orientation_tag_from_filename(path: str) -> str:
    """Extract orientation tag from a cube filename, e.g., *_Taman_edgeon.h5 -> Taman_edgeon.

    Returns "unknown" if the tag cannot be inferred.
    """
    base = os.path.basename(path)
    m = re.match(r".*_([A-Za-z0-9]+_[A-Za-z0-9]+)\.h5$", base)
    if m:
        return m.group(1)
    return "unknown"


def _infer_mcz_from_filename(path: str) -> str:
    """Extract mcz value from a cube filename, e.g., *mcz70Msun* -> 70Msun.

    Returns "unknown" if the mcz cannot be inferred.
    """
    base = os.path.basename(path)
    m = re.match(r".*mcz(\d+Msun).*", base)
    if m:
        return m.group(1)
    return "unknown"


def _global_min_max(zcube: np.ndarray) -> Tuple[float, float]:
    """Compute global finite min/max across all frames of epsilon_min_grid.

    Ignores NaNs/inf to stabilize color scale across frames.
    """
    z = np.asarray(zcube, dtype=float)
    z = z[np.isfinite(z)]
    if z.size == 0:
        return 0.0, 1.0
    return float(np.min(z)), float(np.max(z))


def save_movie(
    omega: np.ndarray,
    theta: np.ndarray,
    eps_grid: np.ndarray,
    td_s: np.ndarray,
    out_path: str,
    cmap: str = "jet",
    levels: int = 100,
    fps: int = 5,
) -> str:
    """Create a contour movie over td from epsilon_min_grid.

    Parameters
    ----------
    omega : (n_omega,) array
    theta : (n_theta,) array
    eps_grid : (n_td, n_theta, n_omega)
    td_s : (n_td,) array in seconds
    out_path : output movie path (.mp4 or .gif)
    """
    os.makedirs(os.path.dirname(out_path), exist_ok=True)

    # Build a static figure/axes and update contour per frame
    fig, ax = plt.subplots(figsize=(6.8, 5.2))

    td_ms = td_s * 1e3
    zmin, zmax = _global_min_max(eps_grid)

    # First frame
    X, Y = np.meshgrid(omega, theta)
    cf = [
        ax.contourf(X, Y, eps_grid[0], levels=levels, cmap=cmap, vmin=zmin, vmax=zmax)
    ]
    cbar = fig.colorbar(cf[0])
    cbar.set_label(r"$\epsilon(\tilde{h}_L, \tilde{h}_P)$")
    ax.set_xlabel(r"$\tilde{\Omega}$")
    ax.set_ylabel(r"$\tilde{\theta}$")
    ttl = ax.set_title(f"td = {td_ms[0]:.1f} ms")
    fig.tight_layout()

    def _update(i):
        # Remove previous collections
        for coll in cf[0].collections:
            coll.remove()
        cf[0] = ax.contourf(
            X, Y, eps_grid[i], levels=levels, cmap=cmap, vmin=zmin, vmax=zmax
        )
        ttl.set_text(f"td = {td_ms[i]:.1f} ms")
        return cf[0].collections + [ttl]

    ani = animation.FuncAnimation(
        fig, _update, frames=len(td_ms), blit=False, interval=1000 / max(fps, 1)
    )

    # Try MP4 first, fallback to GIF via Pillow
    ext = os.path.splitext(out_path)[1].lower()
    writer_used = None
    try:
        if ext == ".mp4":
            writer = animation.FFMpegWriter(fps=fps, bitrate=1800)
            ani.save(out_path, writer=writer, dpi=160)
            writer_used = "ffmpeg"
        else:
            raise RuntimeError("force_gif")
    except Exception:
        from matplotlib.animation import PillowWriter

        gif_path = (
            out_path if ext == ".gif" else (os.path.splitext(out_path)[0] + ".gif")
        )
        ani.save(gif_path, writer=PillowWriter(fps=fps))
        writer_used = "pillow"
        out_path = gif_path

    plt.close(fig)
    print(f"Saved movie: {out_path} (writer={writer_used})")
    return out_path


def save_html_slider(
    omega: np.ndarray,
    theta: np.ndarray,
    eps_grid: np.ndarray,
    td_s: np.ndarray,
    out_path: str,
    cmap: str = "Jet",
    levels: int = 100,
) -> Optional[str]:
    """Write an interactive HTML slider using Plotly (if available).

    Returns output path on success, or None if Plotly is unavailable.
    """
    try:
        import plotly.graph_objs as go
        from plotly.offline import plot as plot_html
    except Exception:
        print("Plotly not available; skipping HTML slider.")
        return None

    os.makedirs(os.path.dirname(out_path), exist_ok=True)

    td_ms = td_s * 1e3
    zmin, zmax = _global_min_max(eps_grid)

    # Initial frame
    fig = go.Figure()
    fig.add_trace(
        go.Contour(
            x=omega,
            y=theta,
            z=eps_grid[0],
            colorscale=cmap,
            contours=dict(coloring="fill", showlabels=False),
            ncontours=levels,
            colorbar=dict(title=r"$\epsilon$"),
            zmin=zmin,
            zmax=zmax,
        )
    )

    # Frames
    frames = []
    for i in range(eps_grid.shape[0]):
        frames.append(
            go.Frame(
                data=[
                    go.Contour(
                        x=omega,
                        y=theta,
                        z=eps_grid[i],
                        colorscale=cmap,
                        contours=dict(coloring="fill", showlabels=False),
                        ncontours=levels,
                        zmin=zmin,
                        zmax=zmax,
                    )
                ],
                name=f"td={td_ms[i]:.1f}ms",
            )
        )

    fig.frames = frames

    # Slider + play/pause
    steps = [
        dict(
            method="animate",
            label=f"{td_ms[i]:.1f} ms",
            args=[
                [f"td={td_ms[i]:.1f}ms"],
                {"frame": {"duration": 0, "redraw": True}, "mode": "immediate"},
            ],
        )
        for i in range(len(td_ms))
    ]

    sliders = [
        dict(
            active=0,
            currentvalue={"prefix": "td: ", "suffix": " ms"},
            pad={"t": 50},
            steps=steps,
        )
    ]

    fig.update_layout(
        title="Epsilon contours over (theta, omega)",
        xaxis_title=r"$\\tilde{\\Omega}$",
        yaxis_title=r"$\\tilde{\\theta}$",
        sliders=sliders,
        updatemenus=[
            dict(
                type="buttons",
                showactive=False,
                y=1.15,
                x=1.05,
                xanchor="right",
                yanchor="top",
                buttons=[
                    dict(
                        label="Play",
                        method="animate",
                        args=[
                            None,
                            {
                                "frame": {"duration": 100, "redraw": True},
                                "fromcurrent": True,
                            },
                        ],
                    ),
                    dict(
                        label="Pause",
                        method="animate",
                        args=[
                            [None],
                            {
                                "frame": {"duration": 0, "redraw": False},
                                "mode": "immediate",
                            },
                        ],
                    ),
                ],
            )
        ],
    )

    plot_html(fig, filename=out_path, auto_open=False)
    print(f"Saved HTML slider: {out_path}")
    return out_path


def main():
    repo_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

    p = argparse.ArgumentParser(
        description="Make movie and/or slider from mismatch cube"
    )
    p.add_argument(
        "--input",
        default=os.path.join(
            repo_root,
            "data/contours_td_mcz/mismatch_cubes/mismatch_cubes_mcz50Msun_td20-70ms_Taman_edgeon.h5",
        ),
        help="Path to mismatch cube HDF5",
    )
    p.add_argument(
        "--outdir",
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
    p.add_argument("--no_html", action="store_true", help="Skip HTML slider output")
    args = p.parse_args()

    if not os.path.isfile(args.input):
        raise FileNotFoundError(f"Input cube not found: {args.input}")

    with h5py.File(args.input, "r") as h5:
        for ds in ("td", "theta", "omega", "epsilon_min_grid"):
            if ds not in h5:
                raise KeyError(
                    f"Dataset '{ds}' missing in {args.input}; found keys: {list(h5.keys())}"
                )

        td = np.array(h5["td"], dtype=float)  # seconds
        theta = np.array(h5["theta"], dtype=float)
        omega = np.array(h5["omega"], dtype=float)
        eps = np.array(h5["epsilon_min_grid"], dtype=float)  # (td, theta, omega)

    if eps.ndim != 3 or eps.shape[1:] != (theta.size, omega.size):
        raise ValueError(
            f"Unexpected epsilon_min_grid shape {eps.shape}; expected (n_td, {theta.size}, {omega.size})"
        )

    os.makedirs(args.outdir, exist_ok=True)
    tag = _infer_orientation_tag_from_filename(args.input)
    mcz_tag = _infer_mcz_from_filename(args.input)

    base = f"epsilon_contours_mcz{mcz_tag}_td20-70ms_{tag}"
    movie_ext = ".mp4" if (args.mp4 and not args.gif) else ".gif"
    movie_path = os.path.join(args.outdir, base + movie_ext)

    # Movie
    save_movie(
        omega=omega,
        theta=theta,
        eps_grid=eps,
        td_s=td,
        out_path=movie_path,
        cmap=args.cmap,
        levels=args.levels,
        fps=args.fps,
    )

    # HTML slider
    if not args.no_html:
        html_path = os.path.join(args.outdir, base + ".html")
        save_html_slider(
            omega=omega,
            theta=theta,
            eps_grid=eps,
            td_s=td,
            out_path=html_path,
            cmap="Jet",  # Plotly colorscale name
            levels=args.levels,
        )


if __name__ == "__main__":
    # Ensure project root is on PYTHONPATH
    sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    main()
