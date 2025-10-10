import sys, os, argparse, pickle
import numpy as np
import numpy.ma as ma
import matplotlib.pyplot as plt

# Ensure project root is on path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from modules.functions_v3 import mcz_for_n_lens_cycles


def load_pickle_data(filepath):
    """Load pickle data and return the loaded object"""
    with open(filepath, "rb") as f:
        return pickle.load(f)


def load_generic_dataset(path):
    """Load a dataset (omega/theta or td/mcz) and return X, Y, Z and axis labels.

    Supports:
    - Pickle with keys: 'omega_matrix', 'theta_matrix', 'epsilon_matrix'
    - Pickle with keys: 'td_arr' (s), 'mcz_arr' (Msun), 'epsilon_matrix' (mcz x td)
    - HDF5 best_match: datasets 'mcz', 'td', 'epsilon_min'
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
            return X, Y, Z, r"$\tilde{\Omega}$", r"$\tilde{\theta}$"
        # Case 2: td-mcz map
        if all(k in d for k in ("td_arr", "mcz_arr", "epsilon_matrix")):
            td = np.asarray(d["td_arr"], dtype=float)  # seconds
            mcz = np.asarray(d["mcz_arr"], dtype=float)
            Z = np.asarray(d["epsilon_matrix"], dtype=float)
            X, Y = np.meshgrid(td * 1e3, mcz)  # x in ms
            return X, Y, Z, r"$\Delta t_d$ [ms]", r"$\mathcal{M}_s\ [M_\odot]$"
        raise ValueError("Unsupported pickle structure: missing expected keys")
    elif ext == ".h5":
        import h5py

        with h5py.File(path, "r") as h5:
            if all(k in h5 for k in ("mcz", "td", "epsilon_min")):
                mcz = np.asarray(h5["mcz"], dtype=float)
                td = np.asarray(h5["td"], dtype=float)
                Z = np.asarray(h5["epsilon_min"], dtype=float)
                X, Y = np.meshgrid(td * 1e3, mcz)  # x in ms
                return X, Y, Z, r"$\Delta t_d$ [ms]", r"$\mathcal{M}_s\ [M_\odot]$"
        raise ValueError(
            "Unsupported HDF5 structure: expecting best_match with mcz, td, epsilon_min"
        )
    else:
        raise ValueError(f"Unsupported extension: {ext}")


def compute_color_scale(epsilons, scale_from="auto"):
    eps_masked = [ma.masked_invalid(ep) for ep in epsilons]
    idx_choice = None
    if isinstance(scale_from, str):
        sf = scale_from.strip().lower()
        if sf.isdigit():
            idx_choice = max(0, int(sf) - 1)
    elif isinstance(scale_from, int):
        idx_choice = max(0, scale_from - 1)

    if (
        idx_choice is not None
        and idx_choice < len(eps_masked)
        and eps_masked[idx_choice].count() > 0
    ):
        vmin = float(eps_masked[idx_choice].min())
        vmax = float(eps_masked[idx_choice].max())
        return eps_masked, vmin, vmax

    mins = [float(ep.min()) for ep in eps_masked if ep.count() > 0]
    maxs = [float(ep.max()) for ep in eps_masked if ep.count() > 0]
    if not mins or not maxs:
        raise ValueError(
            "All input epsilon matrices are empty or NaN-only; cannot set color scale."
        )
    return eps_masked, min(mins), max(maxs)


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
        List of pickle file paths. Length must be >= 2.
    labels : list[str] | None
        Optional list of panel titles, same length as paths. Defaults to 1..N.
    tag : str | None
        Optional tag appended to output filename (preceded by underscore).
    outdir : str
        Directory to save output figure.
    scale_from : str | int | None
        Which dataset to take color scale from: 'auto' (global), a 1-based index (1,2,3...), or None.
    n_levels : int
        Number of contour levels for each subplot.
    eta : float
        Symmetric mass ratio for cycle line calculations. Default is 0.25.
    f_min : float
        Minimum frequency in Hz for cycle line calculations. Default is 20.0.
    """

    if not isinstance(paths, (list, tuple)) or len(paths) < 2:
        raise ValueError("paths must be a list of at least 2 pickle file paths")
    for p in paths:
        if not os.path.exists(p):
            raise FileNotFoundError(f"Pickle not found: {p}")

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
    # Unpack
    Xs = [t[0] for t in loaded]
    Ys = [t[1] for t in loaded]
    epsilons = [t[2] for t in loaded]
    xlabels = [t[3] for t in loaded]
    ylabels = [t[4] for t in loaded]

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
    axes = np.array(axes).reshape(-1)

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
        ax.set_title(labels[i])
        ax.set_xlabel(xlabels[i])
        ax.set_ylabel(ylabels[i])

        # Overlay cycle lines for td-mcz contours
        if is_td_mcz:
            # Extract td array from the X grid (first row, convert ms to s)
            td_arr_ms = Xs[i][0, :]
            td_arr = td_arr_ms / 1e3  # Convert from ms to s

            # Compute 1/2/3-cycle lines
            mcz_1cyc = mcz_for_n_lens_cycles(1.0, td_arr, f_min=f_min, eta=eta)
            mcz_2cyc = mcz_for_n_lens_cycles(2.0, td_arr, f_min=f_min, eta=eta)
            mcz_3cyc = mcz_for_n_lens_cycles(3.0, td_arr, f_min=f_min, eta=eta)

            # Plot cycle lines
            ax.plot(
                td_arr_ms,
                mcz_1cyc,
                color="black",
                ls="-",
                lw=2,
                label="1 lensing modulation",
            )
            ax.plot(
                td_arr_ms,
                mcz_2cyc,
                color="black",
                ls="--",
                lw=2,
                label="2 lensing modulations",
            )
            ax.plot(
                td_arr_ms,
                mcz_3cyc,
                color="black",
                ls=":",
                lw=2,
                label="3 lensing modulations",
            )

        try:
            ax.set_box_aspect(1)
        except Exception:
            pass
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
    # Important: perform after layout; disable constrained_layout to avoid it overriding manual set_position
    try:
        try:
            import matplotlib as mpl  # noqa: F401
        except Exception:
            pass
        # Force a draw so positions are finalized
        try:
            fig.canvas.draw()
        except Exception:
            pass
        # Disable constrained_layout if active to prevent it from resetting positions
        try:
            if hasattr(fig, "get_constrained_layout") and fig.get_constrained_layout():
                fig.set_constrained_layout(False)
        except Exception:
            pass
        cpos = cbar.ax.get_position()
        factor = float(cbar_resize_factor)
        if not (0.05 <= factor <= 1.5):
            factor = 0.9
        new_height = cpos.height * factor
        new_y0 = cpos.y0 + 0.5 * (cpos.height - new_height)
        cbar.ax.set_position([cpos.x0, new_y0, cpos.width, new_height])
        try:
            fig.canvas.draw_idle()
        except Exception:
            pass
    except Exception:
        pass
    # If this is a td–mcz contour, use the minimized-over-(Omega, theta, gamma) label
    is_td_mcz = any(
        isinstance(xlab, str)
        and (
            ("Δ" in xlab)  # unicode delta
            or ("Delta" in xlab)  # plain text
            or ("t_d" in xlab and "ms" in xlab)
        )
        for xlab in xlabels
    )
    if is_td_mcz:
        cbar.set_label(
            r"$\min_{\tilde{\Omega},\,\tilde{\theta},\,\gamma_P}\;\epsilon(\tilde{h}_L,\tilde{h}_P)$"
        )
    else:
        cbar.set_label(r"$\epsilon(\tilde{h}_{\mathrm{L}}, \tilde{h}_{\mathrm{P}})$")

    # Optional: rounded ticks on colorbar
    if cbar_round_ticks:
        try:
            import matplotlib.ticker as mticker

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
        except Exception:
            pass

    # Output
    os.makedirs(outdir, exist_ok=True)
    tag_str = f"_{tag}" if tag else ""
    safe_labels = [lab.replace(" ", "") for lab in labels]
    joined = "_".join(safe_labels)
    fig_path = os.path.join(outdir, f"compare_{joined}_same_scale{tag_str}.pdf")

    # Stats
    print("\nGrid Information:")
    for lab, X, Y in zip(labels, Xs, Ys):
        print(f"{lab} grid shape: {X.shape}")
        print(f"{lab} X range: {float(X.min()):.1f} to {float(X.max()):.1f}")
        print(f"{lab} Y range: {float(Y.min()):.1f} to {float(Y.max()):.1f}")

    # Show the plot immediately, then save after window is closed
    try:
        plt.show()
    finally:
        plt.savefig(fig_path, dpi=200, bbox_inches="tight")
        print(f"Comparison figure saved as {fig_path}")


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
    # removed: --cbar_use_inset (manual resize is used instead)
    args = parser.parse_args()
    return args


if __name__ == "__main__":
    args = _parse_args()
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
