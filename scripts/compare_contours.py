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


def create_comparison_contours(
    path_a, path_b, tag=None, outdir="figures", label_a="A", label_b="B"
):
    """Create comparison contours (two datasets) with a unified color scale.

    Parameters
    ----------
    path_a : str
        Path to first dataset (dataset A) pickle file.
    path_b : str
        Path to second dataset (dataset B) pickle file.
    tag : str | None
        Optional tag appended to output filename (preceded by underscore).
    outdir : str
        Directory to save output figure.
    label_a : str
        Title/label for dataset A panel.
    label_b : str
        Title/label for dataset B panel.
    """

    if not os.path.exists(path_a):
        raise FileNotFoundError(f"Dataset A pickle not found: {path_a}")
    if not os.path.exists(path_b):
        raise FileNotFoundError(f"Dataset B pickle not found: {path_b}")

    # Load both datasets
    data_a = load_pickle_data(path_a)
    data_b = load_pickle_data(path_b)

    # Extract matrices
    omega_a = data_a["omega_matrix"]
    theta_a = data_a["theta_matrix"]
    epsilon_a = data_a["epsilon_matrix"]

    omega_b = data_b["omega_matrix"]
    theta_b = data_b["theta_matrix"]
    epsilon_b = data_b["epsilon_matrix"]

    # Find global min/max for consistent color scaling
    global_min = min(epsilon_a.min(), epsilon_b.min())
    global_max = max(epsilon_a.max(), epsilon_b.max())

    print(f"Global epsilon range: {global_min:.6f} to {global_max:.6f}")
    print(f"{label_a} epsilon range: {epsilon_a.min():.6f} to {epsilon_a.max():.6f}")
    print(f"{label_b} epsilon range: {epsilon_b.min():.6f} to {epsilon_b.max():.6f}")

    # Create figure with two subplots; use constrained_layout to manage colorbar space
    fig, (ax1, ax2) = plt.subplots(
        1, 2, figsize=(10, 5), sharey=True, constrained_layout=True
    )

    # Plot dataset A
    cf1 = ax1.contourf(
        omega_a,
        theta_a,
        epsilon_a,
        levels=np.linspace(global_min, global_max, 100),
        cmap="jet",
        extend="both",
    )
    ax1.set_title(label_a)
    ax1.set_xlabel(r"$\tilde{\Omega}$")
    ax1.set_ylabel(r"$\tilde{\theta}$")

    # Plot dataset B
    cf2 = ax2.contourf(
        omega_b,
        theta_b,
        epsilon_b,
        levels=np.linspace(global_min, global_max, 100),
        cmap="jet",
        extend="both",
    )
    ax2.set_title(label_b)
    ax2.set_xlabel(r"$\tilde{\Omega}$")
    ax2.set_ylabel(r"$\tilde{\theta}$")

    # Make axes less horizontally stretched by enforcing equal box aspect (square)
    for ax in (ax1, ax2):
        try:
            ax.set_box_aspect(1)
        except Exception:
            pass  # Fallback for very old matplotlib versions

    # Add colorbar on the right of both subplots (not between/over them)
    cbar = fig.colorbar(cf2, ax=[ax1, ax2], location="right", shrink=0.9, pad=0.02)
    cbar.set_label(r"$\epsilon(\tilde{h}_{\mathrm{L}}, \tilde{h}_{\mathrm{RP}})$")

    # Prepare output path
    os.makedirs(outdir, exist_ok=True)
    tag_str = f"_{tag}" if tag else ""
    safe_a = label_a.replace(" ", "")
    safe_b = label_b.replace(" ", "")
    fig_path = os.path.join(
        outdir, f"compare_{safe_a}_vs_{safe_b}_same_scale{tag_str}.pdf"
    )
    plt.savefig(fig_path, dpi=200, bbox_inches="tight")
    print(f"Comparison figure saved as {fig_path}")

    # Print some statistics for comparison
    print("\nGrid Information:")
    print(f"{label_a} grid shape: {omega_a.shape}")
    print(f"{label_b} grid shape: {omega_b.shape}")
    print(f"{label_a} omega range: {omega_a.min():.3f} to {omega_a.max():.3f}")
    print(f"{label_b} omega range: {omega_b.min():.3f} to {omega_b.max():.3f}")
    print(f"{label_a} theta range: {theta_a.min():.3f} to {theta_a.max():.3f}")
    print(f"{label_b} theta range: {theta_b.min():.3f} to {theta_b.max():.3f}")

    plt.show()


def _parse_args():
    parser = argparse.ArgumentParser(
        description="Compare two mismatch contour datasets with a unified color scale."
    )
    parser.add_argument(
        "--a",
        dest="path_a",
        default="data/v2_indiv_mismatch_L_RP_mcz20_td22ms_I0.6_thetaS1.047_phiS0.785_thetaJ0.524_phiJ1.047_2_2025-08-22_13-08-25.pkl",
        help="Path to first dataset (A) pickle file",
    )
    parser.add_argument(
        "--b",
        dest="path_b",
        default="data/v3_indiv_mismatch_L_RP_mcz20_td22ms_I0.6_thetaS1.047_phiS0.785_thetaJ0.524_phiJ1.047_3_2025-08-22_11-42-35.pkl",
        help="Path to second dataset (B) pickle file",
    )
    # Backwards compatibility flags (deprecated)
    parser.add_argument(
        "--v2",
        dest="deprecated_v2",
        help=argparse.SUPPRESS,
    )
    parser.add_argument(
        "--v3",
        dest="deprecated_v3",
        help=argparse.SUPPRESS,
    )
    parser.add_argument(
        "--label-a", dest="label_a", default="A", help="Label for dataset A panel"
    )
    parser.add_argument(
        "--label-b", dest="label_b", default="B", help="Label for dataset B panel"
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
    # Apply deprecated flags if provided
    if getattr(args, "deprecated_v2", None):
        args.path_a = args.deprecated_v2
    if getattr(args, "deprecated_v3", None):
        args.path_b = args.deprecated_v3
    return args


if __name__ == "__main__":
    args = _parse_args()
    create_comparison_contours(
        args.path_a,
        args.path_b,
        tag=args.tag,
        outdir=args.outdir,
        label_a=args.label_a,
        label_b=args.label_b,
    )
