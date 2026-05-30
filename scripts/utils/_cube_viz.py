"""Shared visualization helpers for mismatch cube contour scripts."""

import os
import re
from typing import Optional, Tuple

import numpy as np
import matplotlib.pyplot as plt
from matplotlib import animation
from modules.filenames import get_mismatch_cube_resolution
from modules.plot_utils import LBL_EPS_LP, LBL_OMEGA, LBL_THETA


def infer_orientation_tag_from_filename(path: str) -> str:
    """Extract orientation tag from a cube filename, e.g., *_Taman_edgeon.h5 -> Taman_edgeon.

    Returns "unknown" if the tag cannot be inferred.
    """
    base = os.path.basename(path)
    m = re.match(r".*_([A-Za-z0-9]+_[A-Za-z0-9]+)\.h5$", base)
    if m:
        return m.group(1)
    return "unknown"


def format_resolution_suffix(h5_file) -> str:
    """Build resolution suffix using get_mismatch_cube_resolution (td-o-t-g)."""
    td_pts, omega_pts, theta_pts, gamma_pts = get_mismatch_cube_resolution(h5_file)
    return f"td{td_pts}-o{omega_pts}-t{theta_pts}-g{gamma_pts}"


def global_min_max(zcube: np.ndarray) -> Tuple[float, float]:
    """Compute global finite min/max across all frames.

    Ignores NaNs/inf to stabilize color scale across frames.
    """
    z = np.asarray(zcube, dtype=float)
    z = z[np.isfinite(z)]
    if z.size == 0:
        return 0.0, 1.0
    return float(np.min(z)), float(np.max(z))


def find_td_index(td_seconds: np.ndarray, target_td_ms: float) -> int:
    """Find nearest index in td array (in seconds) to target milliseconds."""
    target_s = float(target_td_ms) * 1e-3
    return int(np.argmin(np.abs(td_seconds - target_s)))


def save_contour_movie(
    omega: np.ndarray,
    theta: np.ndarray,
    eps_grid: np.ndarray,
    sweep_values: np.ndarray,
    out_path: str,
    sweep_label: str,
    sweep_fmt: str,
    cmap: str = "jet",
    levels: int = 100,
    fps: int = 5,
) -> str:
    """Create a contour movie over a sweep parameter (td or mcz).

    Parameters
    ----------
    omega : (n_omega,) array
    theta : (n_theta,) array
    eps_grid : (n_frames, n_theta, n_omega)
    sweep_values : (n_frames,) array of parameter values for title
    out_path : output movie path (.mp4 or .gif)
    sweep_label : label prefix for title, e.g. "td" or "mcz"
    sweep_fmt : format string for each value, e.g. "{:.1f} ms" or "{:.2f} Msun"
    """
    os.makedirs(os.path.dirname(out_path), exist_ok=True)

    fig, ax = plt.subplots(figsize=(6.8, 5.2))
    zmin, zmax = global_min_max(eps_grid)

    X, Y = np.meshgrid(omega, theta)
    cf = ax.contourf(X, Y, eps_grid[0], levels=levels, cmap=cmap, vmin=zmin, vmax=zmax)
    cbar = fig.colorbar(cf)
    cbar.set_label(LBL_EPS_LP)
    ax.set_xlabel(LBL_OMEGA)
    ax.set_ylabel(LBL_THETA)
    ttl = ax.set_title(f"{sweep_label} = {sweep_fmt.format(sweep_values[0])}")
    fig.tight_layout()

    def _update(i):
        for coll in ax.collections:
            coll.remove()
        ax.contourf(X, Y, eps_grid[i], levels=levels, cmap=cmap, vmin=zmin, vmax=zmax)
        ttl.set_text(f"{sweep_label} = {sweep_fmt.format(sweep_values[i])}")
        return ax.collections + [ttl]

    ani = animation.FuncAnimation(
        fig, _update, frames=len(sweep_values), blit=False, interval=1000 / max(fps, 1)
    )

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
    sweep_values: np.ndarray,
    out_path: str,
    sweep_label: str,
    sweep_fmt: str,
    cmap: str = "Jet",
    levels: int = 100,
) -> Optional[str]:
    """Write an interactive HTML slider using Plotly (if available).

    Parameters
    ----------
    sweep_label : label prefix for slider, e.g. "td" or "mcz"
    sweep_fmt : format string for each value, e.g. "{:.1f} ms" or "{:.2f} Msun"

    Returns output path on success, or None if Plotly is unavailable.
    """
    try:
        import plotly.graph_objs as go
        from plotly.offline import plot as plot_html
    except Exception:
        print("Plotly not available; skipping HTML slider.")
        return None

    os.makedirs(os.path.dirname(out_path), exist_ok=True)

    zmin, zmax = global_min_max(eps_grid)

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

    frames = []
    for i in range(eps_grid.shape[0]):
        label = sweep_fmt.format(sweep_values[i])
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
                name=f"{sweep_label}={label}",
            )
        )

    fig.frames = frames

    steps = [
        dict(
            method="animate",
            label=sweep_fmt.format(sweep_values[i]),
            args=[
                [f"{sweep_label}={sweep_fmt.format(sweep_values[i])}"],
                {"frame": {"duration": 0, "redraw": True}, "mode": "immediate"},
            ],
        )
        for i in range(len(sweep_values))
    ]

    # Extract unit suffix from sweep_fmt (e.g. "{:.1f} ms" -> " ms")
    unit_suffix = (
        sweep_fmt.replace("{:.1f}", "")
        .replace("{:.2f}", "")
        .replace("{:g}", "")
        .strip()
    )

    sliders = [
        dict(
            active=0,
            currentvalue={
                "prefix": f"{sweep_label}: ",
                "suffix": f" {unit_suffix}" if unit_suffix else "",
            },
            pad={"t": 50},
            steps=steps,
        )
    ]

    fig.update_layout(
        title="Epsilon contours over (theta, omega)",
        xaxis_title=r"$\tilde{\Omega}$",
        yaxis_title=r"$\tilde{\theta}$",
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
