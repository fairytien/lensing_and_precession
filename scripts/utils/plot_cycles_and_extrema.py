import os, argparse, pickle
from typing import Optional, Tuple

import numpy as np
import matplotlib.pyplot as plt

from modules.cosmology import source_mass_redshift_scale
from modules.waveform import mcz_for_n_lens_cycles, number_of_lens_cycles
from modules.filenames import contour_mcz_td_filename
from modules.plot_utils import apply_physics_paper_style, save_figure
from modules.cli_utils import add_cycle_extrema_overlay_args
from modules.lens_cycle_extrema import (
    find_mcz_troughs,
    find_mcz_peaks,
    fixed_mcz_cycle_positions_ms,
    fixed_mcz_peak_positions_ms,
    fixed_mcz_trough_positions_ms,
)

apply_physics_paper_style()


def _optional_positive_float(container, key: str) -> Optional[float]:
    """Read optional scalar metadata and keep only finite positive values."""
    if key not in container:
        return None
    value = float(container[key])
    if not np.isfinite(value) or value <= 0:
        return None
    return value


def plot_mcz_extrema(
    td_arr: np.ndarray,
    mcz_min: float,
    mcz_max: float,
    eta: float = 0.25,
    plot_troughs: bool = True,
    plot_peaks: bool = True,
    mcz_scale: float = 1.0,
    ax=None,
) -> None:
    """Overlay mcz trough and/or peak points on specified matplotlib axes.

    Parameters
    ----------
    td_arr : np.ndarray
        Array of time delays in seconds
    mcz_min, mcz_max : float
        Chirp mass range boundaries in solar masses
    eta : float
        Symmetric mass ratio (default: 0.25)
    plot_troughs : bool
        If True, plot mcz trough points (default: True)
    plot_peaks : bool
        If True, plot mcz peak points (default: True)
    mcz_scale : float
        Multiplicative factor applied to overlay mcz values before plotting.
    ax : matplotlib.axes.Axes, optional
        Axes to plot on. If None, uses current axes.
    """
    if ax is None:
        ax = plt.gca()

    if not plot_troughs and not plot_peaks:
        return

    # Generate overlays in unscaled coordinates, then map onto displayed mcz axis.
    mcz_min_unscaled = float(mcz_min) / mcz_scale
    mcz_max_unscaled = float(mcz_max) / mcz_scale

    if plot_troughs:
        td_trough_pts, mcz_trough_pts = find_mcz_troughs(
            td_arr, eta=eta, mcz_min=mcz_min_unscaled, mcz_max=mcz_max_unscaled
        )
        mcz_trough_pts = mcz_trough_pts * mcz_scale
        if td_trough_pts.size > 0:
            ax.scatter(
                td_trough_pts * 1e3,  # Convert to ms
                mcz_trough_pts,
                c="white",
                marker=".",
                s=5,
                alpha=0.8,
                label="troughs",
                zorder=5,
            )

    if plot_peaks:
        td_peak_pts, mcz_peak_pts = find_mcz_peaks(
            td_arr, eta=eta, mcz_min=mcz_min_unscaled, mcz_max=mcz_max_unscaled
        )
        mcz_peak_pts = mcz_peak_pts * mcz_scale
        if td_peak_pts.size > 0:
            ax.scatter(
                td_peak_pts * 1e3,  # Convert to ms
                mcz_peak_pts,
                c="red",
                marker=".",
                s=5,
                alpha=0.8,
                label="peaks",
                zorder=5,
            )


def plot_cycle_lines(
    td_arr: np.ndarray,
    td_arr_ms: np.ndarray,
    eta: float = 0.25,
    f_min: float = 20.0,
    mcz_scale: float = 1.0,
    ax=None,
) -> None:
    """Overlay 1/2/3 lensing cycle lines on specified matplotlib axes.

    Parameters
    ----------
    td_arr : np.ndarray
        Array of time delays in seconds
    td_arr_ms : np.ndarray
        Array of time delays in milliseconds (for plotting)
    eta : float
        Symmetric mass ratio (default: 0.25)
    f_min : float
        Minimum frequency in Hz (default: 20.0)
    mcz_scale : float
        Multiplicative factor applied to cycle-line mcz values before plotting.
    ax : matplotlib.axes.Axes, optional
        Axes to plot on. If None, uses current axes.
    """
    if ax is None:
        ax = plt.gca()

    for n_cyc, ls_style in FIXED_MCZ_CYCLE_STYLES.items():
        mcz_cyc = mcz_for_n_lens_cycles(n_cyc, td_arr, f_min=f_min, eta=eta) * mcz_scale
        label = f"{n_cyc} cycle" if n_cyc == 1 else f"{n_cyc} cycles"
        ax.plot(td_arr_ms, mcz_cyc, color="black", ls=ls_style, lw=2, label=label)


# ==============================================================================
# Fixed-mcz vertical-line overlay drawing
# ==============================================================================

FIXED_MCZ_CYCLE_STYLES: dict[int, str] = {1: "-", 2: "--", 3: ":"}
_PEAK_COLOR = "magenta"
_TROUGH_COLOR = "white"
_AXVLINE_KW: dict = {"lw": 1.0, "alpha": 0.9, "zorder": 6}


def draw_fixed_mcz_cycle_overlay(
    ax,
    mcz_det_msun: float,
    td_min_ms: float,
    td_max_ms: float,
    *,
    eta: float = 0.25,
    f_min: float = 20.0,
) -> dict:
    """Draw N=1/2/3 lensing-cycle vertical lines on *ax* for a fixed detector-frame chirp mass.

    Returns the positions dict ``{n_cycles: td_ms}`` of lines actually drawn.
    """
    positions = fixed_mcz_cycle_positions_ms(
        mcz_det_msun,
        td_min_ms,
        td_max_ms,
        eta=eta,
        f_min=f_min,
        cycle_counts=tuple(FIXED_MCZ_CYCLE_STYLES),
    )
    for n_cycles, td_ms in positions.items():
        ax.axvline(
            td_ms, color="black", ls=FIXED_MCZ_CYCLE_STYLES[n_cycles], **_AXVLINE_KW
        )
    return positions


def draw_fixed_mcz_extrema_overlay(
    ax,
    mcz_det_msun: float,
    td_min_ms: float,
    td_max_ms: float,
    *,
    eta: float = 0.25,
    plot_peaks: bool = False,
    plot_troughs: bool = False,
) -> None:
    """Draw peak/trough vertical lines on *ax* for a fixed detector-frame chirp mass."""
    if plot_peaks:
        for td_ms in fixed_mcz_peak_positions_ms(
            mcz_det_msun, td_min_ms, td_max_ms, eta=eta
        ):
            ax.axvline(td_ms, color=_PEAK_COLOR, ls=":", **_AXVLINE_KW)
    if plot_troughs:
        for td_ms in fixed_mcz_trough_positions_ms(
            mcz_det_msun, td_min_ms, td_max_ms, eta=eta
        ):
            ax.axvline(td_ms, color=_TROUGH_COLOR, ls=":", **_AXVLINE_KW)


def draw_fixed_mcz_overlays(
    ax,
    mcz_det_msun: float,
    td_min_ms: float,
    td_max_ms: float,
    *,
    overlay_cycles: bool = False,
    overlay_peaks: bool = False,
    overlay_troughs: bool = False,
    eta: float = 0.25,
    f_min: float = 20.0,
) -> dict:
    """Draw cycle and/or extrema overlays on *ax* for a fixed detector-frame chirp mass.

    Returns the positions dict ``{n_cycles: td_ms}`` from the cycle overlay,
    or an empty dict when *overlay_cycles* is False.
    """
    positions = {}
    if overlay_cycles:
        positions = draw_fixed_mcz_cycle_overlay(
            ax, mcz_det_msun, td_min_ms, td_max_ms, eta=eta, f_min=f_min
        )
    if overlay_peaks or overlay_troughs:
        draw_fixed_mcz_extrema_overlay(
            ax,
            mcz_det_msun,
            td_min_ms,
            td_max_ms,
            eta=eta,
            plot_peaks=overlay_peaks,
            plot_troughs=overlay_troughs,
        )
    return positions


def make_fixed_mcz_overlay_legend_handles(
    *,
    cycle_n_list=None,
    include_peaks: bool = False,
    include_troughs: bool = False,
) -> list:
    """Return ``Line2D`` handles for a fixed-mcz overlay legend.

    Parameters
    ----------
    cycle_n_list : sequence of int or None
        Cycle counts to include handles for (e.g. ``[1, 2, 3]`` or
        ``list(positions)``).  Pass ``None`` to omit cycle handles.
    include_peaks : bool
        Include the peak handle.
    include_troughs : bool
        Include the trough handle.
    """
    from matplotlib.lines import Line2D

    handles = []
    if cycle_n_list is not None:
        for n in cycle_n_list:
            ls = FIXED_MCZ_CYCLE_STYLES.get(n)
            if ls is not None:
                handles.append(
                    Line2D(
                        [0],
                        [0],
                        color="black",
                        lw=2,
                        ls=ls,
                        label=rf"$N_{{\mathrm{{lensed}}}}={n}$",
                    )
                )
    if include_peaks:
        handles.append(
            Line2D(
                [0],
                [0],
                linestyle="None",
                marker="o",
                markersize=6,
                markerfacecolor=_PEAK_COLOR,
                markeredgecolor=_PEAK_COLOR,
                label="peak",
            )
        )
    if include_troughs:
        handles.append(
            Line2D(
                [0],
                [0],
                linestyle="None",
                marker="o",
                markersize=6,
                markerfacecolor=_TROUGH_COLOR,
                markeredgecolor="black",
                label="trough",
            )
        )
    return handles


def draw_nlens_isocontours(
    ax,
    td_ms: np.ndarray,
    y_arr: np.ndarray,
    nlens_grid: np.ndarray,
    *,
    label_style: str = "inline",
    fontsize: int = 8,
) -> list:
    """Overlay N_lensed=1/2/3 isocontours on *ax* from a precomputed nlens grid.

    label_style="inline"  — ax.clabel() places labels directly on the contour lines.
    label_style="legend"  — no inline labels; returns Line2D handles for a legend.
    """
    from matplotlib.lines import Line2D

    handles = []
    for n, ls in FIXED_MCZ_CYCLE_STYLES.items():
        cs = ax.contour(
            td_ms,
            y_arr,
            nlens_grid,
            levels=[n],
            colors=["black"],
            linestyles=[ls],
            linewidths=2.0,
        )
        if label_style == "inline":
            ax.clabel(cs, fmt=rf"$N_{{\mathrm{{lensed}}}}={n}$", fontsize=fontsize)
        else:
            handles.append(
                Line2D(
                    [0],
                    [0],
                    color="black",
                    lw=2,
                    ls=ls,
                    label=rf"$N_{{\mathrm{{lensed}}}}={n}$",
                )
            )
    return handles


def _load_data(
    input_path: str,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, Optional[float], Optional[float]]:
    """Load (mcz_arr [Msun], td_arr [s], epsilon_matrix, I, z) from .pkl or .h5 file.

    - Pickle must contain keys: 'mcz_arr' (Msun), 'td_arr' (seconds), 'epsilon_matrix'.
    - HDF5 (best_match) must contain datasets: 'mcz', 'td', 'epsilon_min', and 'I' attr.
    """
    if not os.path.exists(input_path):
        raise FileNotFoundError(f"Input file not found: {input_path}")

    _, ext = os.path.splitext(input_path)
    ext = ext.lower()
    if ext == ".pkl":
        with open(input_path, "rb") as f:
            data = pickle.load(f)
        mcz_arr = np.asarray(data["mcz_arr"], dtype=float)
        td_arr = np.asarray(data["td_arr"], dtype=float)
        Z = np.asarray(data["epsilon_matrix"], dtype=float)
        # Old pickle format may not have I; leave as None if missing
        I_value = float(data["I"]) if "I" in data else None
        z_value = _optional_positive_float(data, "z")
        return mcz_arr, td_arr, Z, I_value, z_value
    elif ext == ".h5":
        import h5py  # local import to avoid hard dep if unused

        with h5py.File(input_path, "r") as h5:
            if not all(k in h5 for k in ("mcz", "td", "epsilon_min")):
                raise ValueError(
                    "HDF5 must be a best_match file with datasets: 'mcz', 'td', 'epsilon_min'"
                )
            mcz_arr = np.asarray(h5["mcz"], dtype=float)
            td_arr = np.asarray(h5["td"], dtype=float)
            Z = np.asarray(h5["epsilon_min"], dtype=float)
            # Extract I from attributes (may be absent)
            I_value = float(h5.attrs["I"]) if "I" in h5.attrs else None
            z_value = _optional_positive_float(h5.attrs, "z")
            return mcz_arr, td_arr, Z, I_value, z_value
    else:
        raise ValueError(f"Unsupported input extension '{ext}'. Use .pkl or .h5")


def _validate_redshift(name: str, z_value: Optional[float]) -> Optional[float]:
    if z_value is None:
        return None
    z = float(z_value)
    if not np.isfinite(z) or z <= 0:
        raise ValueError(f"{name} must be finite and > 0, got {z_value}")
    return z


def _apply_redshift_conversion(
    mcz_arr: np.ndarray,
    input_z: Optional[float],
    z_from: Optional[float],
    z_to: Optional[float],
) -> Tuple[np.ndarray, Optional[float], Optional[float], Optional[float], float]:
    """Optionally remap mcz axis from z_from to z_to.

    Returns converted mcz_arr, z_from_used, z_to_used, z_for_output_token, mcz_scale.
    """
    z_from_val = _validate_redshift("z_from", z_from)
    z_to_val = _validate_redshift("z_to", z_to)
    input_z_val = _validate_redshift("input z", input_z)

    if z_to_val is None:
        if z_from_val is not None:
            raise ValueError("--z_from requires --z_to")
        return mcz_arr, None, None, input_z_val, 1.0

    if z_from_val is None:
        if input_z_val is None:
            raise ValueError(
                "Cannot infer input redshift. Provide --z_from when using --z_to."
            )
        z_from_val = input_z_val

    scale = source_mass_redshift_scale(z_from_val, z_to_val)
    return mcz_arr * scale, z_from_val, z_to_val, z_to_val, scale


def _clean_axis_endpoint(value: float, tol: float = 1e-6) -> float:
    """Snap near-integer endpoints to clean tokens and trim float noise."""
    nearest_int = round(float(value))
    if abs(float(value) - float(nearest_int)) <= tol:
        return float(nearest_int)
    return float(np.round(value, 8))


def _infer_orientation_tag(input_path: str) -> str:
    """Infer orientation_tag from the input filename.

    Expects names like:
      - best_match_td20-70ms_mcz30-46Msun_Taman_edgeon.h5
      - mismatch_cubes_mcz30Msun_td20-70ms_Taman_edgeon.h5
      - contour_td20-70ms_mcz30-46Msun_Taman_edgeon.pkl
    Falls back to 'unknown' if no tag segment found.
    """
    base = os.path.basename(input_path)
    name, _ = os.path.splitext(base)
    parts = name.split("_")
    if len(parts) >= 2:
        # Orientation tag is usually the last token
        return parts[-1]
    return "unknown"


def main():
    parser = argparse.ArgumentParser(
        description="Overlay mcz_1cyc, mcz_2cyc, mcz_3cyc, and mcz_extrema lines on a mismatch (L vs P) contour of td vs mcz."
    )
    parser.add_argument(
        "--input_path",
        type=str,
        required=True,
        help="Path to input file (.pkl with mcz_arr, td_arr, epsilon_matrix) or best_match .h5",
    )
    parser.add_argument(
        "--optimize_mcz",
        action="store_true",
        help="Use optimized mismatch over template chirp mass (affects filename)",
    )
    parser.add_argument(
        "--tag",
        type=str,
        default="",
        help="Optional suffix to append to figure filename",
    )
    add_cycle_extrema_overlay_args(parser)
    parser.add_argument(
        "--z_from",
        type=float,
        default=None,
        help="Input redshift override used for mcz-axis conversion.",
    )
    parser.add_argument(
        "--z_to",
        type=float,
        default=None,
        help="If provided, remap mcz axis to this redshift via (1+z_from)/(1+z_to).",
    )
    parser.add_argument(
        "--nlens-label-style",
        choices=["legend", "inline"],
        default="legend",
        help=(
            "How to label N_lensed cycle overlays with --overlay-cycles: "
            "'legend' (default) adds cycle handles to the legend box; "
            "'inline' places labels directly on the lines with ax.clabel."
        ),
    )
    args = parser.parse_args()

    base_dir = os.path.dirname(
        os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    )
    fig_dir = os.path.join(base_dir, "figures", "utils")
    os.makedirs(fig_dir, exist_ok=True)

    input_path = args.input_path

    mcz_arr, td_arr, Z, I_value, input_z = _load_data(input_path)
    mcz_arr, z_from_used, z_to_used, z_for_output, overlay_mcz_scale = (
        _apply_redshift_conversion(
            mcz_arr,
            input_z=input_z,
            z_from=args.z_from,
            z_to=args.z_to,
        )
    )

    # Validate data
    if mcz_arr.size == 0 or td_arr.size == 0:
        raise ValueError("Loaded arrays are empty")

    # Build grid for plotting
    td_arr_ms = td_arr * 1e3
    mcz_min, mcz_max = mcz_arr.min(), mcz_arr.max()

    # Plot
    plt.figure(figsize=(8, 6))
    cf = plt.contourf(td_arr_ms, mcz_arr, Z, levels=100, cmap="jet")
    cbar = plt.colorbar(cf)
    if args.optimize_mcz:
        cbar.set_label(
            r"$\min_{\mathcal{M}_{\rm t}}$ $\epsilon(\tilde{h}_{\rm L}, \tilde{h}_{\rm P})$"
        )
    else:
        cbar.set_label(
            r"$\min_{\tilde{\Omega}, \, \tilde{\theta}, \, \gamma_P}\; \epsilon(\tilde{h}_L,\tilde{h}_P)$"
        )
    plt.xlabel(r"$\Delta t_d$ [ms]")
    plt.ylabel(r"$\mathcal{M}_s\ [M_\odot]$")

    # Overlay cycle lines if requested
    cycle_handles: list = []
    if args.overlay_cycles:
        MCZ_DET_GRID, TD_GRID = np.meshgrid(
            mcz_arr / overlay_mcz_scale, td_arr, indexing="ij"
        )
        nlens_grid = number_of_lens_cycles(
            MCZ_DET_GRID, TD_GRID, f_min=args.f_min, eta=args.eta
        )
        cycle_handles = draw_nlens_isocontours(
            plt.gca(),
            td_arr_ms,
            mcz_arr,
            nlens_grid,
            label_style=args.nlens_label_style,
        )

    # Overlay mcz extrema points if requested
    if args.overlay_troughs or args.overlay_peaks:
        plot_mcz_extrema(
            td_arr,
            mcz_min,
            mcz_max,
            eta=args.eta,
            plot_troughs=args.overlay_troughs,
            plot_peaks=args.overlay_peaks,
            mcz_scale=overlay_mcz_scale,
        )

    # Optionally show legend if there are labeled artists
    if args.show_legend:
        ax = plt.gca()
        handles, labels = ax.get_legend_handles_labels()
        all_handles = handles + cycle_handles
        if all_handles:
            plt.legend(handles=all_handles, loc="best")
    plt.tight_layout()

    # Generate output filename derived from source data + orientation tag, with 'overlayed' suffix
    td_min_ms = td_arr.min() * 1e3
    td_max_ms = td_arr.max() * 1e3
    orientation_tag = _infer_orientation_tag(input_path)

    # Build filename with overlayed suffix
    base_fig = contour_mcz_td_filename(
        fig_dir,
        I=I_value,
        mcz_min=_clean_axis_endpoint(mcz_min),
        mcz_max=_clean_axis_endpoint(mcz_max),
        mcz_pts=int(len(mcz_arr)),
        td_min_ms=td_min_ms,
        td_max_ms=td_max_ms,
        td_pts=int(len(td_arr)),
        orientation_tag=orientation_tag,
        z=z_for_output,
        ext="pdf",
    )
    base_name, base_ext = os.path.splitext(base_fig)

    # Add suffixes: _overlayed and optional user tag
    suffixes = ["overlayed"]
    if args.tag:
        suffixes.append(args.tag)
    out_path = f"{base_name}_{'_'.join(suffixes)}{base_ext}"
    if z_to_used is not None:
        print(
            f"Applied mcz-axis redshift conversion: z_from={z_from_used:g}, z_to={z_to_used:g}, scale={overlay_mcz_scale:.12g}"
        )
    save_figure(plt.gcf(), out_path)


if __name__ == "__main__":
    main()
