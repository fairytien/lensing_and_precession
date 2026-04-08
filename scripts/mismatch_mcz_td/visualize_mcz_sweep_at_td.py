"""Visualize epsilon contours over (theta, omega) while sweeping chirp mass at fixed td.

Loads multiple per-mcz mismatch cube HDF5 files (created by create_mismatch_cube),
then:
- Extracts epsilon_min_grid slice at the requested time delay td (ms)
- Builds a movie (MP4 if ffmpeg available, else GIF) sweeping over chirp mass
- Optionally writes an interactive HTML slider (Plotly) to scrub chirp mass

Default input directory points to the same location used by visualize_mismatch_cube if present.
"""

import os
import argparse
import re
from typing import List, Optional, Tuple

import numpy as np
import h5py
from modules.filenames import (
    get_mismatch_cube_resolution,
    parse_mcz_from_mismatch_cube_path,
)
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


def _infer_mcz_msun_numeric(h5_file) -> float:
    """Read numeric mcz from HDF5 dataset 'mcz'."""
    if "mcz" in h5_file:
        val = np.array(h5_file["mcz"]).astype(float).ravel()
        if val.size >= 1:
            return float(val[0])
    # Fallback to filename parser for canonical naming.
    parsed = parse_mcz_from_mismatch_cube_path(h5_file.filename)
    if parsed is not None:
        return float(parsed)
    return float("nan")


def _format_resolution_suffix(h5_file) -> str:
    """Build resolution suffix using get_mismatch_cube_resolution (td-o-t-g)."""
    td_pts, omega_pts, theta_pts, gamma_pts = get_mismatch_cube_resolution(h5_file)
    return f"td{td_pts}-o{omega_pts}-t{theta_pts}-g{gamma_pts}"


def _global_min_max(zcube: np.ndarray) -> Tuple[float, float]:
    """Compute global finite min/max across frames.

    Ignores NaNs/inf to stabilize color scale across frames.
    """
    z = np.asarray(zcube, dtype=float)
    z = z[np.isfinite(z)]
    if z.size == 0:
        return 0.0, 1.0
    return float(np.min(z)), float(np.max(z))


def _find_td_index(td_seconds: np.ndarray, target_td_ms: float) -> int:
    """Find nearest index in td array (in seconds) to target milliseconds."""
    target_s = float(target_td_ms) * 1e-3
    idx = int(np.argmin(np.abs(td_seconds - target_s)))
    return idx


def save_movie_over_mcz(
    omega: np.ndarray,
    theta: np.ndarray,
    eps_grid_over_mcz: np.ndarray,
    mcz_msun_list: np.ndarray,
    out_path: str,
    cmap: str = "jet",
    levels: int = 100,
    fps: int = 5,
) -> str:
    """Create a contour movie over chirp mass.

    Parameters
    ----------
    omega : (n_omega,) array
    theta : (n_theta,) array
    eps_grid_over_mcz : (n_mcz, n_theta, n_omega)
    mcz_msun_list : (n_mcz,) array in Msun
    out_path : output movie path (.mp4 or .gif)
    """
    os.makedirs(os.path.dirname(out_path), exist_ok=True)

    fig, ax = plt.subplots(figsize=(6.8, 5.2))

    zmin, zmax = _global_min_max(eps_grid_over_mcz)

    X, Y = np.meshgrid(omega, theta)
    cf = ax.contourf(
        X, Y, eps_grid_over_mcz[0], levels=levels, cmap=cmap, vmin=zmin, vmax=zmax
    )
    cbar = fig.colorbar(cf)
    cbar.set_label(r"$\epsilon(\tilde{h}_L, \tilde{h}_P)$")
    ax.set_xlabel(r"$\tilde{\Omega}$")
    ax.set_ylabel(r"$\tilde{\theta}$")
    ttl = ax.set_title(f"mcz = {mcz_msun_list[0]:.2f} Msun")
    fig.tight_layout()

    def _update(i):
        for coll in ax.collections:
            coll.remove()
        ax.contourf(
            X, Y, eps_grid_over_mcz[i], levels=levels, cmap=cmap, vmin=zmin, vmax=zmax
        )
        ttl.set_text(f"mcz = {mcz_msun_list[i]:.2f} Msun")
        return ax.collections + [ttl]

    ani = animation.FuncAnimation(
        fig,
        _update,
        frames=len(mcz_msun_list),
        blit=False,
        interval=1000 / max(fps, 1),
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


def save_html_slider_over_mcz(
    omega: np.ndarray,
    theta: np.ndarray,
    eps_grid_over_mcz: np.ndarray,
    mcz_msun_list: np.ndarray,
    out_path: str,
    cmap: str = "Jet",
    levels: int = 100,
) -> Optional[str]:
    """Write an interactive HTML slider over chirp mass using Plotly (if available)."""
    try:
        import plotly.graph_objs as go
        from plotly.offline import plot as plot_html
    except Exception:
        print("Plotly not available; skipping HTML slider.")
        return None

    os.makedirs(os.path.dirname(out_path), exist_ok=True)

    zmin, zmax = _global_min_max(eps_grid_over_mcz)

    fig = go.Figure()
    fig.add_trace(
        go.Contour(
            x=omega,
            y=theta,
            z=eps_grid_over_mcz[0],
            colorscale=cmap,
            contours=dict(coloring="fill", showlabels=False),
            ncontours=levels,
            colorbar=dict(title=r"$\epsilon$"),
            zmin=zmin,
            zmax=zmax,
        )
    )

    frames = []
    for i in range(eps_grid_over_mcz.shape[0]):
        frames.append(
            go.Frame(
                data=[
                    go.Contour(
                        x=omega,
                        y=theta,
                        z=eps_grid_over_mcz[i],
                        colorscale=cmap,
                        contours=dict(coloring="fill", showlabels=False),
                        ncontours=levels,
                        zmin=zmin,
                        zmax=zmax,
                    )
                ],
                name=f"mcz={mcz_msun_list[i]:.2f}Msun",
            )
        )

    fig.frames = frames

    steps = [
        dict(
            method="animate",
            label=f"{mcz_msun_list[i]:.2f} Msun",
            args=[
                [f"mcz={mcz_msun_list[i]:.2f}Msun"],
                {"frame": {"duration": 0, "redraw": True}, "mode": "immediate"},
            ],
        )
        for i in range(len(mcz_msun_list))
    ]

    sliders = [
        dict(
            active=0,
            currentvalue={"prefix": "mcz: ", "suffix": " Msun"},
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


def _discover_cube_files(input_dir: str, orientation_tag: Optional[str]) -> List[str]:
    """Find valid mismatch cube files under input_dir.

    Supports both:
    - input_dir as a folder that directly contains cube .h5 files
    - input_dir as a run directory that contains a mismatch_cubes/ subfolder
    """
    if not os.path.isdir(input_dir):
        return []

    candidate_dirs = [input_dir]
    nested = os.path.join(input_dir, "mismatch_cubes")
    if os.path.isdir(nested):
        candidate_dirs.insert(0, nested)

    files = []
    seen = set()
    for cube_dir in candidate_dirs:
        for name in os.listdir(cube_dir):
            if not name.endswith(".h5"):
                continue
            if orientation_tag and not name.endswith(f"_{orientation_tag}.h5"):
                continue
            path = os.path.join(cube_dir, name)
            if path in seen:
                continue
            if parse_mcz_from_mismatch_cube_path(path) is None:
                continue
            files.append(path)
            seen.add(path)
    return sorted(files)


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
        default=os.path.join(repo_root, "data/mismatch/mismatch_cubes"),
        help=(
            "Directory containing per-mcz mismatch cube HDF5 files, or a run "
            "directory that contains mismatch_cubes/"
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

    files = _discover_cube_files(args.input_dir, args.orientation_tag)
    if not files:
        raise FileNotFoundError(
            "No mismatch cube files found in "
            f"{args.input_dir} (tag={args.orientation_tag}). "
            "Expected cube .h5 files directly under input_dir or under "
            "input_dir/mismatch_cubes/."
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
                res_suffix = _format_resolution_suffix(h5)
                if orient_tag is None:
                    orient_tag = _infer_orientation_tag_from_filename(fp)
            else:
                if theta.shape != theta_ref.shape or not np.allclose(theta, theta_ref):
                    raise ValueError(
                        "Theta grids differ across cubes; cannot sweep mcz coherently"
                    )
                if omega.shape != omega_ref.shape or not np.allclose(omega, omega_ref):
                    raise ValueError(
                        "Omega grids differ across cubes; cannot sweep mcz coherently"
                    )

            td_idx = _find_td_index(td, args.td_ms)
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
    save_movie_over_mcz(
        omega=omega_ref,
        theta=theta_ref,
        eps_grid_over_mcz=eps_stack,
        mcz_msun_list=mcz_msun_arr,
        out_path=movie_path,
        cmap=args.cmap,
        levels=args.levels,
        fps=args.fps,
    )

    # HTML slider over mcz
    if args.html:
        html_path = os.path.join(args.output_dir, base + ".html")
        save_html_slider_over_mcz(
            omega=omega_ref,
            theta=theta_ref,
            eps_grid_over_mcz=eps_stack,
            mcz_msun_list=mcz_msun_arr,
            out_path=html_path,
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
