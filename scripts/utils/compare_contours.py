import sys, os, argparse, pickle, re
import numpy as np
import numpy.ma as ma
import matplotlib.pyplot as plt

# Ensure project root is on path
sys.path.insert(
    0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
)

from modules.plot_utils import (
    LBL_EPS_LP,
    LBL_MCZ,
    LBL_MIN_EPS_LP,
    LBL_TD,
    apply_physics_paper_style,
    format_colorbar_ticks,
    save_figure,
    set_square_axes,
)
from modules.cli_utils import add_cycle_extrema_overlay_args

apply_physics_paper_style()

from scripts.utils.plot_cycles_and_extrema import plot_cycle_lines, plot_mcz_extrema


def _td_mcz_dataset(td_s, mcz, Z):
    X, Y = np.meshgrid(td_s * 1e3, mcz)
    return X, Y, Z, LBL_TD, LBL_MCZ, "td_mcz"


def _apply_colorbar_ticks(cbar, vmin, vmax, round_ticks, n_ticks, decimals):
    if not round_ticks:
        return
    nbins = (
        None
        if isinstance(n_ticks, str) and n_ticks.strip().lower() == "auto"
        else max(2, int(n_ticks))
    )
    format_colorbar_ticks(
        cbar,
        vmin,
        vmax,
        nbins=nbins,
        decimals=int(decimals) if decimals is not None else 2,
    )


def _resize_colorbar(fig, cbar, factor):
    if factor == 1.0:
        return
    fig.canvas.draw()
    if hasattr(fig, "get_constrained_layout") and fig.get_constrained_layout():
        fig.set_constrained_layout(False)
    cpos = cbar.ax.get_position()
    factor = max(0.05, min(1.5, factor))
    new_height = cpos.height * factor
    new_y0 = cpos.y0 + 0.5 * (cpos.height - new_height)
    cbar.ax.set_position([cpos.x0, new_y0, cpos.width, new_height])
    fig.canvas.draw_idle()


def load_generic_dataset(path):
    """Load a dataset (omega/theta or td/mcz) and return X, Y, Z, axis labels, and data type.

    Supports:
    - Pickle with keys: 'omega_matrix', 'theta_matrix', 'epsilon_matrix'
    - Pickle with keys: 'td_arr' (s), 'mcz_arr' (Msun), 'epsilon_matrix' (mcz x td)
    - HDF5 best_match: datasets 'mcz', 'td', 'epsilon_min'
    - HDF5 contour_mcz_td: datasets 'mcz_arr', 'td_arr', 'epsilon_matrix'
    - HDF5 mismatch_cube: datasets 'td', 'theta', 'omega', 'epsilon_min_grid' (uses first td slice)

    Returns
    -------
    tuple
        (X, Y, Z, xlabel, ylabel, data_type) where data_type is 'omega_theta' or 'td_mcz'
    """
    _, ext = os.path.splitext(path)
    ext = ext.lower()

    if ext == ".pkl":
        with open(path, "rb") as fh:
            d = pickle.load(fh)
        # Case 1: omega-theta grid
        if all(k in d for k in ("omega_matrix", "theta_matrix", "epsilon_matrix")):
            X = np.asarray(d["omega_matrix"])
            Y = np.asarray(d["theta_matrix"])
            Z = np.asarray(d["epsilon_matrix"], dtype=float)
            return X, Y, Z, r"$\tilde{\Omega}$", r"$\tilde{\theta}$", "omega_theta"
        # Case 2: td-mcz map
        if all(k in d for k in ("td_arr", "mcz_arr", "epsilon_matrix")):
            td = np.asarray(d["td_arr"], dtype=float)  # seconds
            mcz = np.asarray(d["mcz_arr"], dtype=float)
            Z = np.asarray(d["epsilon_matrix"], dtype=float)
            return _td_mcz_dataset(td, mcz, Z)
        raise ValueError("Unsupported pickle structure: missing expected keys")

    elif ext == ".h5":
        import h5py

        with h5py.File(path, "r") as h5:
            # Case 1: best_match file
            if all(k in h5 for k in ("mcz", "td", "epsilon_min")):
                mcz = np.asarray(h5["mcz"], dtype=float)
                td = np.asarray(h5["td"], dtype=float)
                Z = np.asarray(h5["epsilon_min"], dtype=float)
                return _td_mcz_dataset(td, mcz, Z)
            # Case 2: contour_mcz_td file
            elif all(k in h5 for k in ("mcz_arr", "td_arr", "epsilon_matrix")):
                mcz = np.asarray(h5["mcz_arr"], dtype=float)
                td = np.asarray(h5["td_arr"], dtype=float)
                Z = np.asarray(h5["epsilon_matrix"], dtype=float)
                return _td_mcz_dataset(td, mcz, Z)
            # Case 3: mismatch cube file (td, theta, omega)
            elif all(k in h5 for k in ("td", "theta", "omega", "epsilon_min_grid")):
                td = np.asarray(h5["td"], dtype=float)  # (n_td,)
                theta = np.asarray(h5["theta"], dtype=float)  # (n_theta,)
                omega = np.asarray(h5["omega"], dtype=float)  # (n_omega,)
                eps_grid = np.asarray(
                    h5["epsilon_min_grid"], dtype=float
                )  # (n_td, n_theta, n_omega)

                Z = eps_grid[0]  # (n_theta, n_omega) — first td slice

                # Create meshgrid: X (omega), Y (theta)
                # For Z[i,j], i indexes theta, j indexes omega
                X, Y = np.meshgrid(omega, theta)
                # Z shape is (n_theta, n_omega) which matches meshgrid output
                return (
                    X,
                    Y,
                    Z,
                    r"$\tilde{\Omega}$",
                    r"$\tilde{\theta}$",
                    "omega_theta",
                )
        raise ValueError(
            "Unsupported HDF5 structure: expecting best_match (mcz, td, epsilon_min) "
            "or contour_mcz_td (mcz_arr, td_arr, epsilon_matrix) "
            "or mismatch_cube (td, theta, omega, epsilon_min_grid)"
        )
    else:
        raise ValueError(f"Unsupported extension: {ext}")


def compute_color_scale(epsilons, scale_from="auto"):
    """Compute unified color scale for multiple epsilon arrays.

    Parameters
    ----------
    epsilons : list of ndarray
        List of epsilon matrices
    scale_from : str or int
        'auto' for global min/max, or 1-based index to use specific dataset's range

    Returns
    -------
    tuple
        (list of masked arrays, vmin, vmax)
    """
    eps_masked = [ma.masked_invalid(ep) for ep in epsilons]
    idx_choice = None

    # Parse scale_from parameter
    if isinstance(scale_from, str):
        sf = scale_from.strip().lower()
        if sf.isdigit():
            idx_choice = int(sf) - 1  # Convert to 0-based index
    elif isinstance(scale_from, int):
        idx_choice = scale_from - 1  # Convert to 0-based index

    # Validate index choice
    if idx_choice is not None:
        if idx_choice < 0 or idx_choice >= len(eps_masked):
            print(
                f"Warning: scale_from index {idx_choice + 1} out of range [1, {len(eps_masked)}]. Using 'auto' instead."
            )
            idx_choice = None
        elif eps_masked[idx_choice].count() == 0:
            print(
                f"Warning: Dataset {idx_choice + 1} has no valid data. Using 'auto' instead."
            )
            idx_choice = None

    # Use specific dataset's range if valid index provided
    if idx_choice is not None:
        vmin = float(eps_masked[idx_choice].min())
        vmax = float(eps_masked[idx_choice].max())
        return eps_masked, vmin, vmax

    # Compute global range across all datasets
    mins = [float(ep.min()) for ep in eps_masked if ep.count() > 0]
    maxs = [float(ep.max()) for ep in eps_masked if ep.count() > 0]
    if not mins or not maxs:
        raise ValueError(
            "All input epsilon matrices are empty or NaN-only; cannot set color scale."
        )
    return eps_masked, min(mins), max(maxs)


def sanitize_filename(name):
    """Convert a label to a safe filename component."""
    # Remove or replace problematic characters
    safe = re.sub(r"[^\w\s\-.]", "", name)  # Keep alphanumeric, spaces, hyphens, dots
    safe = safe.replace(" ", "")  # Remove spaces
    return safe


def create_ratio_contour(
    num_path,
    den_path,
    tag=None,
    outdir="figures/utils",
    n_levels=100,
    cmap="jet",
    cbar_round_ticks=False,
    cbar_n_ticks="auto",
    cbar_decimals=None,
    cbar_resize_factor=1.0,
    eta=0.25,
    f_min=20.0,
    overlay_cycles=False,
    overlay_peaks=False,
    overlay_troughs=False,
    title=None,
):
    """Create a single contour plot of the ratio of epsilon matrices (num/den).

    Both inputs must be td–mcz datasets with identical grids.

    Parameters
    ----------
    num_path : str
        Path to numerator dataset file.
    den_path : str
        Path to denominator dataset file.
    tag : str | None
        Optional tag appended to output filename.
    outdir : str
        Directory to save output figure.
    n_levels : int
        Number of contour levels.
    cmap : str
        Matplotlib colormap name.
    cbar_round_ticks : bool
        Use rounded tick locations on colorbar.
    cbar_n_ticks : str | int
        Number of colorbar ticks when rounding.
    cbar_decimals : int | None
        Format colorbar tick labels to this many decimals.
    cbar_resize_factor : float
        Scale colorbar height by this factor (0.3-1.2 typical range).
    eta : float
        Symmetric mass ratio for cycle line calculations. Default is 0.25.
    f_min : float
        Minimum frequency in Hz for cycle line calculations. Default is 20.0.
    overlay_cycles : bool
        If True, overlay 1/2/3 lensing cycle lines. Default is False.
    overlay_peaks : bool
        If True, overlay mcz peak points. Default is False.
    overlay_troughs : bool
        If True, overlay mcz trough points. Default is False.
    title : str | None
        Custom title for the plot. If None, uses default title.
    """
    if not os.path.exists(num_path):
        raise FileNotFoundError(f"Dataset file not found: {num_path}")
    if not os.path.exists(den_path):
        raise FileNotFoundError(f"Dataset file not found: {den_path}")

    Xn, Yn, Zn, xlab_n, ylab_n, type_n = load_generic_dataset(num_path)
    Xd, Yd, Zd, xlab_d, ylab_d, type_d = load_generic_dataset(den_path)

    if type_n != "td_mcz" or type_d != "td_mcz":
        raise ValueError("Both datasets must be td–mcz grids for ratio plotting.")

    # Validate grids match
    if Xn.shape != Xd.shape or Yn.shape != Yd.shape:
        raise ValueError(
            f"Grid shapes differ: numerator {Xn.shape} vs denominator {Xd.shape}"
        )
    if not (np.allclose(Xn, Xd) and np.allclose(Yn, Yd)):
        raise ValueError("Grid coordinates (td, mcz) differ between datasets.")

    # Compute ratio with masking to avoid division by zero/invalids
    Zn_mask = ma.masked_invalid(Zn)
    Zd_mask = ma.masked_invalid(Zd)
    denom_safe = ma.masked_less_equal(Zd_mask, 0)
    ratio = ma.divide(Zn_mask, denom_safe)

    fig, ax = plt.subplots(1, 1, figsize=(5, 5), constrained_layout=True)
    cf = ax.contourf(
        Xn,
        Yn,
        ratio,
        levels=n_levels if isinstance(n_levels, (list, np.ndarray)) else n_levels,
        cmap=cmap,
    )

    # Set title (custom or default)
    if title is None:
        title = "Ratio of Mismatch Between RP and NP Templates With Lensed Sources"
    ax.set_title(title, pad=15)

    ax.set_xlabel(xlab_n)
    ax.set_ylabel(ylab_n)

    # Overlay cycle lines and extrema for td–mcz contours
    # Extract td array from the X grid (first row, convert ms to s)
    td_arr_ms = Xn[0, :]
    td_arr = td_arr_ms / 1e3  # Convert from ms to s

    # Get mcz range for extrema overlays
    mcz_data_min = Yn.min()
    mcz_data_max = Yn.max()

    # Overlay cycle lines if requested
    if overlay_cycles:
        plot_cycle_lines(td_arr, td_arr_ms, eta=eta, f_min=f_min, ax=ax)

    # Overlay mcz extrema points if requested
    if overlay_peaks or overlay_troughs:
        plot_mcz_extrema(
            td_arr,
            mcz_data_min,
            mcz_data_max,
            eta=eta,
            plot_troughs=overlay_troughs,
            plot_peaks=overlay_peaks,
            ax=ax,
        )

    set_square_axes(ax)

    # Colorbar
    cbar = fig.colorbar(
        cf,
        ax=ax,
        location="right",
        use_gridspec=True,
        shrink=1.0,
        fraction=0.046,
        pad=0.04,
    )
    cbar.set_label(
        r"$\min_{\tilde{\Omega},\,\tilde{\theta},\,\gamma_{\mathrm{P}}}\;\epsilon(\tilde{h}_{\mathrm{L}},\tilde{h}_{\mathrm{P}}) / \epsilon(\tilde{h}_{\mathrm{L}},\tilde{h}_{\mathrm{NP}})$"
    )

    vmin_r = float(ratio.min()) if ratio.count() > 0 else 0.0
    vmax_r = float(ratio.max()) if ratio.count() > 0 else 1.0
    _apply_colorbar_ticks(
        cbar, vmin_r, vmax_r, cbar_round_ticks, cbar_n_ticks, cbar_decimals
    )
    _resize_colorbar(fig, cbar, cbar_resize_factor)

    # Output path
    num_name = sanitize_filename(os.path.splitext(os.path.basename(num_path))[0])
    den_name = sanitize_filename(os.path.splitext(os.path.basename(den_path))[0])
    tag_str = f"_{tag}" if tag else ""
    out_path = os.path.join(outdir, f"ratio_{num_name}_OVER_{den_name}{tag_str}.pdf")
    save_figure(fig, out_path)


def create_comparison_contours(
    paths,
    labels=None,
    tag=None,
    outdir="figures/utils",
    scale_from="auto",
    n_levels=100,
    cmap="jet",
    cbar_round_ticks=False,
    cbar_n_ticks="auto",
    cbar_decimals=None,
    cbar_resize_factor=1.0,
    eta=0.25,
    f_min=20.0,
    overlay_cycles=False,
    overlay_peaks=False,
    overlay_troughs=False,
    overlay_cycles_for=None,
    overlay_peaks_for=None,
    overlay_troughs_for=None,
):
    """Create comparison contours (2+ datasets) with a unified color scale.

    Parameters
    ----------
    paths : list[str]
        List of dataset file paths (.pkl or .h5). Length must be >= 2.
    labels : list[str] | None
        Optional list of panel titles, same length as paths. Defaults to filenames.
    tag : str | None
        Optional tag appended to output filename (preceded by underscore).
    outdir : str
        Directory to save output figure.
    scale_from : str | int
        Which dataset to take color scale from: 'auto' (global), or a 1-based index (1,2,3...).
    n_levels : int
        Number of contour levels for each subplot.
    cmap : str
        Matplotlib colormap name.
    cbar_round_ticks : bool
        Use rounded tick locations on colorbar.
    cbar_n_ticks : str | int
        Number of colorbar ticks when rounding ('auto' or integer).
    cbar_decimals : int | None
        Format colorbar tick labels to this many decimals.
    cbar_resize_factor : float
        Scale colorbar height by this factor (0.3-1.2 typical range).
    eta : float
        Symmetric mass ratio for cycle line calculations. Default is 0.25.
    f_min : float
        Minimum frequency in Hz for cycle line calculations. Default is 20.0.
    overlay_cycles : bool
        If True, overlay 1/2/3 lensing cycle lines on td-mcz plots. Default is False.
    overlay_peaks : bool
        If True, overlay mcz peak points on td-mcz plots. Default is False.
    overlay_troughs : bool
        If True, overlay mcz trough points on td-mcz plots. Default is False.
    """
    if not isinstance(paths, (list, tuple)) or len(paths) < 2:
        raise ValueError("paths must be a list of at least 2 dataset file paths")
    # Validate all paths exist
    for p in paths:
        if not os.path.exists(p):
            raise FileNotFoundError(f"Dataset file not found: {p}")

    # If no user-provided labels, derive from filenames (basename without extension)
    if labels is None:
        labels = [os.path.splitext(os.path.basename(p))[0] for p in paths]
    if len(labels) != len(paths):
        raise ValueError(
            f"labels length ({len(labels)}) must match paths length ({len(paths)})"
        )

    # Load datasets and collect fields, with explicit error context per path
    loaded = []
    for p in paths:
        try:
            loaded.append(load_generic_dataset(p))
        except Exception as e:
            raise RuntimeError(f"Failed to load dataset: {p}: {e}") from e

    # Unpack loaded data
    Xs = [t[0] for t in loaded]
    Ys = [t[1] for t in loaded]
    epsilons = [t[2] for t in loaded]
    xlabels = [t[3] for t in loaded]
    ylabels = [t[4] for t in loaded]
    data_types = [t[5] for t in loaded]

    # Validate all datasets are the same type
    if len(set(data_types)) > 1:
        raise ValueError(
            f"All datasets must be the same type. Found: {set(data_types)}. "
            "Cannot mix omega-theta and td-mcz grids."
        )

    # Mask NaNs and compute color scale via helper
    eps_masked, global_min, global_max = compute_color_scale(epsilons, scale_from)

    print(f"Global epsilon range: {global_min:.6f} to {global_max:.6f}")
    for lab, ep in zip(labels, eps_masked):
        if ep.count() > 0:
            print(
                f"{lab} epsilon range: {float(ep.min()):.6f} to {float(ep.max()):.6f}"
            )
        else:
            print(f"{lab} epsilon range: NaN-only")

    # Determine subplot grid.
    # Use a true 2x2 layout for four panels to avoid empty columns.
    n = len(paths)
    if n == 4:
        rows, cols = 2, 2
    else:
        cols = min(3, n)
        rows = int(np.ceil(n / float(cols)))
    figsize = (5 * cols, 5 * rows)

    fig, axes = plt.subplots(rows, cols, figsize=figsize, constrained_layout=True)
    # Ensure axes is always an array, even for single subplot
    axes = np.atleast_1d(axes).reshape(-1)

    contour_handles = []
    for i in range(n):
        ax = axes[i]
        cf = ax.contourf(
            Xs[i],
            Ys[i],
            eps_masked[i],
            levels=np.linspace(global_min, global_max, n_levels),
            cmap=cmap,
        )
        ax.set_title(labels[i], pad=15)
        ax.set_xlabel(xlabels[i])
        ax.set_ylabel(ylabels[i])

        # Overlay cycle lines and extrema for td–mcz contours
        if data_types[i] == "td_mcz":
            # Extract td array from the X grid (first row, convert ms to s)
            td_arr_ms = Xs[i][0, :]
            td_arr = td_arr_ms / 1e3  # Convert from ms to s

            # Get mcz range for extrema overlays
            mcz_data_min = Ys[i].min()
            mcz_data_max = Ys[i].max()

            panel_idx = i + 1
            if overlay_cycles and (
                overlay_cycles_for is None or panel_idx in overlay_cycles_for
            ):
                plot_cycle_lines(td_arr, td_arr_ms, eta=eta, f_min=f_min, ax=ax)
            do_peaks = overlay_peaks and (
                overlay_peaks_for is None or panel_idx in overlay_peaks_for
            )
            do_troughs = overlay_troughs and (
                overlay_troughs_for is None or panel_idx in overlay_troughs_for
            )
            if do_peaks or do_troughs:
                plot_mcz_extrema(
                    td_arr,
                    mcz_data_min,
                    mcz_data_max,
                    eta=eta,
                    plot_troughs=do_troughs,
                    plot_peaks=do_peaks,
                    ax=ax,
                )

        # Set square aspect ratio if supported
        set_square_axes(ax)
        contour_handles.append(cf)

    # Hide any unused axes
    for j in range(n, len(axes)):
        axes[j].axis("off")

    # Colorbar for all panels
    cbar = fig.colorbar(
        contour_handles[-1],
        ax=list(axes[:n]),
        location="right",
        use_gridspec=True,
        shrink=1.0,
        fraction=0.046,
        pad=0.04,
    )

    _resize_colorbar(fig, cbar, cbar_resize_factor)
    if data_types[0] == "td_mcz":
        cbar.set_label(LBL_MIN_EPS_LP)
    else:
        cbar.set_label(LBL_EPS_LP)
    _apply_colorbar_ticks(
        cbar, global_min, global_max, cbar_round_ticks, cbar_n_ticks, cbar_decimals
    )

    # Prepare output path with sanitized filename
    tag_str = f"_{tag}" if tag else ""
    safe_labels = [sanitize_filename(lab) for lab in labels]
    joined = "_".join(safe_labels)
    fig_path = os.path.join(outdir, f"compare_{joined}_same_scale{tag_str}.pdf")

    # Print statistics
    print("\nGrid Information:")
    for lab, X, Y in zip(labels, Xs, Ys):
        print(f"{lab} grid shape: {X.shape}")
        print(f"{lab} X range: {float(X.min()):.1f} to {float(X.max()):.1f}")
        print(f"{lab} Y range: {float(Y.min()):.1f} to {float(Y.max()):.1f}")

    save_figure(fig, fig_path)


def _parse_args():
    parser = argparse.ArgumentParser(
        description="Compare 2+ mismatch contour datasets with a unified color scale."
    )
    parser.add_argument(
        "--paths",
        nargs="+",
        default=None,
        help="List of 2+ dataset files to compare (.pkl or best_match .h5)",
    )
    parser.add_argument(
        "--labels",
        nargs="+",
        help="Optional list of labels (same length as --paths). Defaults to 1..N",
    )
    parser.add_argument(
        "--tag",
        dest="tag",
        default=None,
        help="Optional tag appended to output filename",
    )
    parser.add_argument(
        "--outdir", dest="outdir", default="figures/utils", help="Output directory"
    )
    parser.add_argument(
        "--n_levels",
        type=int,
        default=100,
        help="Number of contour levels for each subplot (default: 100)",
    )
    parser.add_argument(
        "--cmap",
        type=str,
        default="jet",
        help="Matplotlib colormap name to use (default: jet)",
    )
    parser.add_argument(
        "--scale_from",
        default="auto",
        help=(
            "Which dataset to take color scale from: 'auto' (global) or a 1-based index (1,2,3...). "
            "Default: auto."
        ),
    )
    parser.add_argument(
        "--cbar_round_ticks",
        action="store_true",
        help="Use rounded tick locations on the colorbar (nice numbers)",
    )
    parser.add_argument(
        "--cbar_n_ticks",
        type=str,
        default="auto",
        help=(
            "Approximate number of colorbar ticks when rounding; "
            "use an integer or 'auto' (default) to choose automatically based on range"
        ),
    )
    parser.add_argument(
        "--cbar_decimals",
        type=int,
        help="If set, format colorbar tick labels to this many decimals",
    )
    parser.add_argument(
        "--cbar_resize_factor",
        type=float,
        default=1.0,
        help=(
            "Scale the colorbar height by this factor (e.g., 0.6 shrinks to 60%). "
            "Typical range 0.3–1.2."
        ),
    )
    add_cycle_extrema_overlay_args(parser, include_show_legend=False)
    parser.add_argument(
        "--overlay-peaks-for",
        nargs="+",
        type=int,
        help="Specify which panels (1-based indices) should get peak overlays. Default: all panels if --overlay-peaks is used.",
    )
    parser.add_argument(
        "--overlay-troughs-for",
        nargs="+",
        type=int,
        help="Specify which panels (1-based indices) should get trough overlays. Default: all panels if --overlay-troughs is used.",
    )
    parser.add_argument(
        "--overlay-cycles-for",
        nargs="+",
        type=int,
        help="Specify which panels (1-based indices) should get cycle overlays. Default: all panels if --overlay-cycles is used.",
    )
    parser.add_argument(
        "--ratio_of",
        nargs=2,
        metavar=("NUM", "DEN"),
        help=(
            "If provided, plot a single contour of the ratio of epsilon matrices "
            "(NUM/DEN). Both must be td–mcz datasets with identical grids."
        ),
    )
    parser.add_argument(
        "--title",
        type=str,
        default=None,
        help="Custom title for ratio plot (only used with --ratio_of)",
    )
    # removed: --cbar_use_inset (manual resize is used instead)
    args = parser.parse_args()
    return args


if __name__ == "__main__":
    args = _parse_args()
    if args.ratio_of:
        num, den = args.ratio_of
        create_ratio_contour(
            num_path=num,
            den_path=den,
            tag=args.tag,
            outdir=args.outdir,
            n_levels=args.n_levels,
            cmap=args.cmap,
            cbar_round_ticks=args.cbar_round_ticks,
            cbar_n_ticks=args.cbar_n_ticks,
            cbar_decimals=args.cbar_decimals,
            cbar_resize_factor=args.cbar_resize_factor,
            eta=args.eta,
            f_min=args.f_min,
            overlay_cycles=args.overlay_cycles,
            overlay_peaks=args.overlay_peaks,
            overlay_troughs=args.overlay_troughs,
            title=args.title,
        )
    else:
        if not args.paths or len(args.paths) < 2:
            raise SystemExit("--paths must include at least 2 files to compare")
        create_comparison_contours(
            args.paths,
            labels=args.labels,
            tag=args.tag,
            outdir=args.outdir,
            scale_from=args.scale_from,
            n_levels=args.n_levels,
            cmap=args.cmap,
            cbar_round_ticks=args.cbar_round_ticks,
            cbar_n_ticks=args.cbar_n_ticks,
            cbar_decimals=args.cbar_decimals,
            cbar_resize_factor=args.cbar_resize_factor,
            eta=args.eta,
            f_min=args.f_min,
            overlay_cycles=args.overlay_cycles,
            overlay_peaks=args.overlay_peaks,
            overlay_troughs=args.overlay_troughs,
            overlay_cycles_for=args.overlay_cycles_for,
            overlay_peaks_for=args.overlay_peaks_for,
            overlay_troughs_for=args.overlay_troughs_for,
        )

"""Example CLI Usage:
python /work/10000/fairytien33/ls6/lensing_and_precession/scripts/utils/compare_contours.py
--paths /work/10000/fairytien33/ls6/lensing_and_precession/data/contour_mcz_td/mismatch_contour_L_NP_mcz_td_I0.5_2025-08-18_12-57-22.pkl /work/10000/fairytien33/ls6/lensing_and_precession/data/contours_td_mcz/best_match/best_match_td20-70ms_mcz10-90Msun_Taman_edgeon.h5 
--labels "Lensed Sources vs Non-Precessing Templates" "Lensed Sources vs Regularly Precessing Templates" 
--cbar_round_ticks 
--cbar_resize_factor 0.85
--overlay-cycles
--overlay-peaks
--overlay-troughs


python /work/10000/fairytien33/ls6/lensing_and_precession/scripts/utils/compare_contours.py 
--paths /work/10000/fairytien33/ls6/lensing_and_precession/data/contour_mcz_td/mismatch_contour_L_NP_mcz_td_I0.5_2025-08-18_12-57-22.pkl /work/10000/fairytien33/ls6/lensing_and_precession/data/contours_td_mcz/best_match/best_match_td20-70ms_mcz10-90Msun_Taman_edgeon.h5 
--labels 'Lensed Sources vs Non-Precessing Templates' 'Lensed Sources vs Regularly Precessing Templates' 
--cbar_round_ticks 
--cbar_resize_factor 0.85 
--overlay-cycles 
--overlay-troughs 
--overlay-cycles-for 1 2 
--overlay-troughs-for 1


python /work/10000/fairytien33/ls6/lensing_and_precession/scripts/utils/compare_contours.py
--ratio_of /work/10000/fairytien33/ls6/lensing_and_precession/data/contours_td_mcz/best_match/best_match_td20-70ms_mcz10-90Msun_Taman_edgeon.h5 /work/10000/fairytien33/ls6/lensing_and_precession/data/contour_mcz_td/mismatch_contour_L_NP_mcz_td_I0.5_2025-08-18_12-57-22.pkl
--overlay-cycles
"""
