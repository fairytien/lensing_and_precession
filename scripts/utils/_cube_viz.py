"""Shared visualization helpers for mismatch cube contour scripts."""

import os
import re
from typing import Optional, Tuple

import numpy as np
import matplotlib.pyplot as plt
from matplotlib import animation
from modules.filenames import get_mismatch_cube_resolution
from modules.plot_utils import (
    LBL_EPS_LP,
    LBL_OMEGA,
    LBL_THETA,
    add_colorbar_axes,
    best_match_contour_legend_label,
    compute_best_match_point,
    draw_omega_theta_contour_panel,
    format_colorbar_ticks,
    mcz_contour_panel_text,
    resolve_contour_vlim,
)


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


def _plotly_frame_annotations(point: dict, mcz_msun: float) -> list:
    return [
        dict(
            x=0.03,
            y=0.96,
            xref="paper",
            yref="paper",
            text=mcz_contour_panel_text(mcz_msun),
            showarrow=False,
            xanchor="left",
            yanchor="top",
            bgcolor="rgba(255,255,255,0.7)",
            borderwidth=0,
        ),
        dict(
            x=0.97,
            y=0.03,
            xref="paper",
            yref="paper",
            text=best_match_contour_legend_label(point, line_break="<br>"),
            showarrow=False,
            xanchor="right",
            yanchor="bottom",
            bgcolor="rgba(255,255,255,0.7)",
            borderwidth=0,
        ),
    ]


def save_contour_movie(
    omega: np.ndarray,
    theta: np.ndarray,
    eps_grid: np.ndarray,
    gamma_grid: np.ndarray,
    sweep_values: np.ndarray,
    out_path: str,
    sweep_label: str,
    sweep_fmt: str,
    cmap: str = "jet",
    levels: int = 100,
    fps: int = 5,
    vmax_cap: Optional[float] = None,
) -> str:
    """Create a styled contour movie over a sweep parameter (td or mcz).

    Parameters
    ----------
    omega : (n_omega,) array
    theta : (n_theta,) array
    eps_grid : (n_frames, n_theta, n_omega)
    gamma_grid : (n_frames, n_theta, n_omega) — best-fit gamma_P at each grid point
    sweep_values : (n_frames,) array of mcz values (Msun) used for the chirp-mass box
    out_path : output movie path (.mp4 or .gif)
    sweep_label : label prefix for the sweep parameter, e.g. "mcz" or "td"
    sweep_fmt : format string for each value, e.g. "{:.2f} Msun" or "{:.1f} ms"
    vmax_cap : optional upper color limit; data max used when None
    """
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    X, Y = np.meshgrid(omega, theta)
    vmin, vmax, saturated = resolve_contour_vlim(eps_grid, vmax_cap=vmax_cap)

    fig, ax = plt.subplots(figsize=(7.0, 7.0))
    cf0 = draw_omega_theta_contour_panel(
        ax, X, Y, eps_grid[0], gamma_grid[0],
        mcz_msun=float(sweep_values[0]),
        vmin=vmin, vmax=vmax, saturated=saturated, levels=levels, cmap=cmap,
    )
    fig.tight_layout()
    fig.canvas.draw()
    cax = add_colorbar_axes(fig, ax, pad=0.02, width=0.02)
    cbar = fig.colorbar(cf0, cax=cax)
    cbar.set_label(LBL_EPS_LP)
    format_colorbar_ticks(cbar, vmin, vmax, decimals=2)

    def _update(i):
        ax.clear()
        draw_omega_theta_contour_panel(
            ax, X, Y, eps_grid[i], gamma_grid[i],
            mcz_msun=float(sweep_values[i]),
            vmin=vmin, vmax=vmax, saturated=saturated, levels=levels, cmap=cmap,
        )
        return ax.collections

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
    gamma_grid: np.ndarray,
    sweep_values: np.ndarray,
    out_path: str,
    sweep_label: str,
    sweep_fmt: str,
    cmap: str = "Jet",
    levels: int = 100,
    vmax_cap: Optional[float] = None,
) -> Optional[str]:
    """Write an interactive HTML slider using Plotly (if available).

    Parameters
    ----------
    gamma_grid : (n_frames, n_theta, n_omega) — best-fit gamma_P at each grid point
    sweep_values : (n_frames,) array of mcz values (Msun) for the chirp-mass box
    vmax_cap : optional upper color limit; data max used when None

    Returns output path on success, or None if Plotly is unavailable.
    """
    try:
        import plotly.graph_objs as go
        from plotly.offline import plot as plot_html
    except Exception:
        print("Plotly not available; skipping HTML slider.")
        return None

    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    X, Y = np.meshgrid(omega, theta)
    zmin, zmax, _ = resolve_contour_vlim(eps_grid, vmax_cap=vmax_cap)

    point0 = compute_best_match_point(X, Y, eps_grid[0], gamma_grid[0])
    mcz0 = float(sweep_values[0])

    fig = go.Figure()
    fig.add_trace(
        go.Contour(
            x=omega, y=theta, z=eps_grid[0],
            colorscale=cmap,
            contours=dict(coloring="fill", showlabels=False),
            ncontours=levels,
            colorbar=dict(title=LBL_EPS_LP),
            zmin=zmin, zmax=zmax,
        )
    )
    fig.add_trace(
        go.Scatter(
            x=[point0["min_omega"]], y=[point0["min_theta"]],
            mode="markers",
            marker=dict(
                symbol="star", size=16, color="white",
                line=dict(color="rgba(80,80,80,0.8)", width=1),
            ),
            showlegend=False,
        )
    )

    frames = []
    for i in range(eps_grid.shape[0]):
        point = compute_best_match_point(X, Y, eps_grid[i], gamma_grid[i])
        frames.append(
            go.Frame(
                data=[
                    go.Contour(
                        x=omega, y=theta, z=eps_grid[i],
                        colorscale=cmap,
                        contours=dict(coloring="fill", showlabels=False),
                        ncontours=levels,
                        zmin=zmin, zmax=zmax,
                    ),
                    go.Scatter(
                        x=[point["min_omega"]], y=[point["min_theta"]],
                        mode="markers",
                        marker=dict(
                            symbol="star", size=16, color="white",
                            line=dict(color="rgba(80,80,80,0.8)", width=1),
                        ),
                        showlegend=False,
                    ),
                ],
                layout=dict(annotations=_plotly_frame_annotations(point, float(sweep_values[i]))),
                name=f"{sweep_label}={sweep_fmt.format(sweep_values[i])}",
            )
        )

    fig.frames = frames
    fig.update_layout(
        annotations=_plotly_frame_annotations(point0, mcz0),
        xaxis_title=LBL_OMEGA,
        yaxis_title=LBL_THETA,
        yaxis=dict(scaleanchor="x", scaleratio=1),
    )

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

    unit_suffix = (
        sweep_fmt.replace("{:.1f}", "").replace("{:.2f}", "").replace("{:g}", "").strip()
    )

    fig.update_layout(
        sliders=[
            dict(
                active=0,
                currentvalue={
                    "prefix": f"{sweep_label}: ",
                    "suffix": f" {unit_suffix}" if unit_suffix else "",
                },
                pad={"t": 50},
                steps=steps,
            )
        ],
        updatemenus=[
            dict(
                type="buttons",
                showactive=False,
                y=1.15, x=1.05, xanchor="right", yanchor="top",
                buttons=[
                    dict(
                        label="Play", method="animate",
                        args=[None, {"frame": {"duration": 100, "redraw": True}, "fromcurrent": True}],
                    ),
                    dict(
                        label="Pause", method="animate",
                        args=[[None], {"frame": {"duration": 0, "redraw": False}, "mode": "immediate"}],
                    ),
                ],
            )
        ],
    )

    plot_html(fig, filename=out_path, auto_open=False)
    print(f"Saved HTML slider: {out_path}")
    return out_path
