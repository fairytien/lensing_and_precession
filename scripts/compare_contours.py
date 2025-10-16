import sys
import os
import argparse
import pickle
import re
import numpy as np
import numpy.ma as ma
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker

# Ensure project root is on path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from modules.functions_v3 import mcz_for_n_lens_cycles


def load_pickle_data(filepath):
    """Load pickle data and return the loaded object"""
    with open(filepath, "rb") as f:
        return pickle.load(f)


def load_generic_dataset(path):
    """Load a dataset (omega/theta or td/mcz) and return X, Y, Z, axis labels, and data type.

    Supports:
    - Pickle with keys: 'omega_matrix', 'theta_matrix', 'epsilon_matrix'
    - Pickle with keys: 'td_arr' (s), 'mcz_arr' (Msun), 'epsilon_matrix' (mcz x td)
    - HDF5 best_match: datasets 'mcz', 'td', 'epsilon_min'

    Returns
    -------
    tuple
        (X, Y, Z, xlabel, ylabel, data_type) where data_type is 'omega_theta' or 'td_mcz'
    """
    _, ext = os.path.splitext(path)
    ext = ext.lower()

    if ext == ".pkl":
        d = load_pickle_data(path)
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
            X, Y = np.meshgrid(td * 1e3, mcz)  # x in ms
            return (
                X,
                Y,
                Z,
                r"$\Delta t_d$ [ms]",
                r"$\mathcal{M}_s\ [M_\odot]$",
                "td_mcz",
            )
        raise ValueError("Unsupported pickle structure: missing expected keys")

    elif ext == ".h5":
        import h5py

        with h5py.File(path, "r") as h5:
            if all(k in h5 for k in ("mcz", "td", "epsilon_min")):
                mcz = np.asarray(h5["mcz"], dtype=float)
                td = np.asarray(h5["td"], dtype=float)
                Z = np.asarray(h5["epsilon_min"], dtype=float)
                X, Y = np.meshgrid(td * 1e3, mcz)  # x in ms
                return (
                    X,
                    Y,
                    Z,
                    r"$\Delta t_d$ [ms]",
                    r"$\mathcal{M}_s\ [M_\odot]$",
                    "td_mcz",
                )
        raise ValueError(
            "Unsupported HDF5 structure: expecting best_match with mcz, td, epsilon_min"
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
    outdir="figures",
    n_levels=100,
    cmap="jet",
    cbar_round_ticks=False,
    cbar_n_ticks="auto",
    cbar_decimals=None,
    cbar_resize_factor=0.9,
):
    """Create a single contour plot of the ratio of epsilon matrices (num/den).

    Both inputs must be td–mcz datasets with identical grids.
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
        extend="both",
    )
    ax.set_title(
        "Ratio of Mismatch Between RP and NP Templates With Lensed Sources", pad=15
    )
    ax.set_xlabel(xlab_n)
    ax.set_ylabel(ylab_n)
    if hasattr(ax, "set_box_aspect"):
        ax.set_box_aspect(1)

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
        r"$\min_{\tilde{\Omega},\,\tilde{\theta},\,\gamma_P}\;\epsilon(\tilde{h}_{\rm L},\tilde{h}_{\rm P}) / \epsilon(\tilde{h}_{\rm L},\tilde{h}_{\rm NP})$"
    )

    # Rounded ticks on colorbar
    if cbar_round_ticks:
        if isinstance(cbar_n_ticks, str) and cbar_n_ticks.strip().lower() == "auto":
            nbins_val = "auto"
        else:
            nbins_val = max(2, int(cbar_n_ticks))
        locator = mticker.MaxNLocator(nbins=nbins_val, steps=[1, 2, 2.5, 5, 10])
        cbar.locator = locator
        if cbar_decimals is not None:
            cbar.formatter = mticker.FormatStrFormatter(f"%.{int(cbar_decimals)}f")
        cbar.update_ticks()

    # Resize colorbar height if requested
    if cbar_resize_factor != 1.0:
        fig.canvas.draw()
        if hasattr(fig, "get_constrained_layout") and fig.get_constrained_layout():
            fig.set_constrained_layout(False)
        cpos = cbar.ax.get_position()
        factor = max(0.05, min(1.5, float(cbar_resize_factor)))
        new_height = cpos.height * factor
        new_y0 = cpos.y0 + 0.5 * (cpos.height - new_height)
        cbar.ax.set_position([cpos.x0, new_y0, cpos.width, new_height])
        fig.canvas.draw_idle()

    # Output path
    os.makedirs(outdir, exist_ok=True)
    num_name = sanitize_filename(os.path.splitext(os.path.basename(num_path))[0])
    den_name = sanitize_filename(os.path.splitext(os.path.basename(den_path))[0])
    tag_str = f"_{tag}" if tag else ""
    out_path = os.path.join(outdir, f"ratio_{num_name}_OVER_{den_name}{tag_str}.pdf")
    plt.savefig(out_path, dpi=200, bbox_inches="tight")
    print(f"Ratio figure saved as {out_path}")
    plt.show()


def create_comparison_contours(
    paths,
    labels=None,
    tag=None,
    outdir="figures",
    scale_from="auto",
    n_levels=100,
    cmap="jet",
    cbar_round_ticks=False,
    cbar_n_ticks="auto",
    cbar_decimals=None,
    cbar_resize_factor=0.9,
    eta=0.25,
    f_min=20.0,
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

    # Determine subplot grid: up to 3 columns per row
    n = len(paths)
    cols = min(3, n)
    rows = int(np.ceil(n / 3.0))
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
            extend="both",
        )
        ax.set_title(labels[i], pad=15)
        ax.set_xlabel(xlabels[i])
        ax.set_ylabel(ylabels[i])

        # Overlay cycle lines for td–mcz contours (use data_type instead of fragile string detection)
        is_td_mcz_i = data_types[i] == "td_mcz"
        if is_td_mcz_i:
            # Extract td array from the X grid (first row, convert ms to s)
            td_arr_ms = Xs[i][0, :]
            td_arr = td_arr_ms / 1e3  # Convert from ms to s

            # Compute 1/2/3-cycle lines
            mcz_1cyc = mcz_for_n_lens_cycles(1.0, td_arr, f_min=f_min, eta=eta)
            mcz_2cyc = mcz_for_n_lens_cycles(2.0, td_arr, f_min=f_min, eta=eta)
            mcz_3cyc = mcz_for_n_lens_cycles(3.0, td_arr, f_min=f_min, eta=eta)

            # Plot cycle lines
            ax.plot(td_arr_ms, mcz_1cyc, color="black", ls="-", lw=2, label="1 cycle")
            ax.plot(td_arr_ms, mcz_2cyc, color="black", ls="--", lw=2, label="2 cycles")
            ax.plot(td_arr_ms, mcz_3cyc, color="black", ls=":", lw=2, label="3 cycles")

        # Set square aspect ratio if supported
        if hasattr(ax, "set_box_aspect"):
            ax.set_box_aspect(1)
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

    # Manually resize colorbar height by provided factor, keeping it centered vertically
    if cbar_resize_factor != 1.0:
        # Force a draw so positions are finalized
        fig.canvas.draw()
        # Disable constrained_layout to prevent it from resetting positions
        if hasattr(fig, "get_constrained_layout") and fig.get_constrained_layout():
            fig.set_constrained_layout(False)

        cpos = cbar.ax.get_position()
        factor = float(cbar_resize_factor)
        # Clamp factor to reasonable range
        factor = max(0.05, min(1.5, factor))
        new_height = cpos.height * factor
        new_y0 = cpos.y0 + 0.5 * (cpos.height - new_height)
        cbar.ax.set_position([cpos.x0, new_y0, cpos.width, new_height])
        fig.canvas.draw_idle()
    # Set colorbar label based on data type
    is_td_mcz = data_types[0] == "td_mcz"
    if is_td_mcz:
        cbar.set_label(
            r"$\min_{\tilde{\Omega},\,\tilde{\theta},\,\gamma_P}\;\epsilon(\tilde{h}_L,\tilde{h}_P)$"
        )
    else:
        cbar.set_label(r"$\epsilon(\tilde{h}_{\mathrm{L}}, \tilde{h}_{\mathrm{P}})$")

    # Optional: rounded ticks on colorbar
    if cbar_round_ticks:
        # Allow automatic tick count when cbar_n_ticks == 'auto'
        if isinstance(cbar_n_ticks, str) and cbar_n_ticks.strip().lower() == "auto":
            nbins_val = "auto"
        else:
            nbins_val = max(2, int(cbar_n_ticks))

        locator = mticker.MaxNLocator(nbins=nbins_val, steps=[1, 2, 2.5, 5, 10])
        cbar.locator = locator
        if cbar_decimals is not None:
            cbar.formatter = mticker.FormatStrFormatter(f"%.{int(cbar_decimals)}f")
        cbar.update_ticks()

    # Prepare output path with sanitized filename
    os.makedirs(outdir, exist_ok=True)
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

    # Save figure first (before showing to avoid potential resource issues)
    plt.savefig(fig_path, dpi=200, bbox_inches="tight")
    print(f"\nComparison figure saved as {fig_path}")

    # Show the plot
    plt.show()


def _parse_args():
    parser = argparse.ArgumentParser(
        description="Compare 2+ mismatch contour datasets with a unified color scale."
    )
    parser.add_argument(
        "--paths",
        nargs="+",
        required=True,
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
        "--outdir", dest="outdir", default="figures", help="Output directory"
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
        default=0.9,
        help=(
            "Scale the colorbar height by this factor (e.g., 0.6 shrinks to 60%). "
            "Typical range 0.3–1.2."
        ),
    )
    parser.add_argument(
        "--eta",
        type=float,
        default=0.25,
        help="Symmetric mass ratio for cycle line calculations (default: 0.25)",
    )
    parser.add_argument(
        "--f_min",
        type=float,
        default=20.0,
        help="Minimum frequency in Hz for cycle line calculations (default: 20.0)",
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
        )
    else:
        if len(args.paths) < 2:
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
        )

"""Example CLI Usage:
python /work/10000/fairytien33/ls6/lensing_and_precession/scripts/compare_contours.py
--paths /work/10000/fairytien33/ls6/lensing_and_precession/data/super_contours/mismatch_contour_L_NP_mcz_td_I0.5_2025-08-18_12-57-22.pkl /work/10000/fairytien33/ls6/lensing_and_precession/data/contours_td_mcz/best_match/best_match_td20-70ms_mcz10-90Msun_Taman_edgeon.h5 
--labels "Lensed Sources vs Non-Precessing Templates" "Lensed Sources vs Regularly Precessing Templates" 
--cbar_round_ticks 
--cbar_resize_factor 0.85


python /work/10000/fairytien33/ls6/lensing_and_precession/scripts/compare_contours.py
--ratio_of /work/10000/fairytien33/ls6/lensing_and_precession/data/contours_td_mcz/best_match/best_match_td20-70ms_mcz10-90Msun_Taman_edgeon.h5 /work/10000/fairytien33/ls6/lensing_and_precession/data/super_contours/mismatch_contour_L_NP_mcz_td_I0.5_2025-08-18_12-57-22.pkl
--cbar_round_ticks
"""
