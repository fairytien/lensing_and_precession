import sys, os
import pickle
import argparse
import numpy as np
import matplotlib.pyplot as plt

# Ensure project root is on path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


def load_pickle_data(filepath):
    """Load pickle data and return the loaded object"""
    with open(filepath, "rb") as f:
        return pickle.load(f)


def create_comparison_contours(paths, labels=None, tag=None, outdir="figures"):
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
    """

    if not isinstance(paths, (list, tuple)) or len(paths) < 2:
        raise ValueError("paths must be a list of at least 2 pickle file paths")
    for p in paths:
        if not os.path.exists(p):
            raise FileNotFoundError(f"Pickle not found: {p}")

    if labels is None:
        labels = [f"{i+1}" for i in range(len(paths))]
    if len(labels) != len(paths):
        raise ValueError("labels length must match paths length")

    # Load datasets and collect fields
    datasets = [load_pickle_data(p) for p in paths]
    omegas = [d["omega_matrix"] for d in datasets]
    thetas = [d["theta_matrix"] for d in datasets]
    epsilons = [d["epsilon_matrix"] for d in datasets]

    # Global color scale
    global_min = min(float(ep.min()) for ep in epsilons)
    global_max = max(float(ep.max()) for ep in epsilons)

    print(f"Global epsilon range: {global_min:.6f} to {global_max:.6f}")
    for lab, ep in zip(labels, epsilons):
        print(f"{lab} epsilon range: {float(ep.min()):.6f} to {float(ep.max()):.6f}")

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
            omegas[i],
            thetas[i],
            epsilons[i],
            levels=np.linspace(global_min, global_max, 100),
            cmap="jet",
            extend="both",
        )
        ax.set_title(labels[i])
        ax.set_xlabel(r"$\tilde{\Omega}$")
        ax.set_ylabel(r"$\tilde{\theta}$")
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
        contour_handles[-1], ax=list(axes[:n]), location="right", shrink=0.9, pad=0.02
    )
    cbar.set_label(r"$\epsilon(\tilde{h}_{\mathrm{L}}, \tilde{h}_{\mathrm{RP}})$")

    # Output
    os.makedirs(outdir, exist_ok=True)
    tag_str = f"_{tag}" if tag else ""
    safe_labels = [lab.replace(" ", "") for lab in labels]
    joined = "_".join(safe_labels)
    fig_path = os.path.join(outdir, f"compare_{joined}_same_scale{tag_str}.pdf")
    plt.savefig(fig_path, dpi=200, bbox_inches="tight")
    print(f"Comparison figure saved as {fig_path}")

    # Stats
    print("\nGrid Information:")
    for lab, om, th in zip(labels, omegas, thetas):
        print(f"{lab} grid shape: {om.shape}")
        print(f"{lab} omega range: {float(om.min()):.3f} to {float(om.max()):.3f}")
        print(f"{lab} theta range: {float(th.min()):.3f} to {float(th.max()):.3f}")

    plt.show()


def _parse_args():
    parser = argparse.ArgumentParser(
        description="Compare 2+ mismatch contour datasets with a unified color scale."
    )
    parser.add_argument(
        "--paths",
        nargs="+",
        help="List of 2+ pickle files to compare (overrides --a/--b)",
    )
    # Backwards-compat flags for exactly two inputs
    parser.add_argument(
        "--a",
        dest="path_a",
        help="Path to first dataset (A) pickle file",
    )
    parser.add_argument(
        "--b",
        dest="path_b",
        help="Path to second dataset (B) pickle file",
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
    args = parser.parse_args()
    return args


if __name__ == "__main__":
    args = _parse_args()
    if args.paths and len(args.paths) >= 2:
        create_comparison_contours(
            args.paths, labels=args.labels, tag=args.tag, outdir=args.outdir
        )
    else:
        # Fallback to legacy two-input mode
        if not args.path_a or not args.path_b:
            raise SystemExit("Provide either --paths (2+ files) or both --a and --b")
        labels = None
        if args.labels:
            labels = args.labels[:2]
        else:
            labels = ["1", "2"]
        create_comparison_contours(
            [args.path_a, args.path_b], labels=labels, tag=args.tag, outdir=args.outdir
        )
