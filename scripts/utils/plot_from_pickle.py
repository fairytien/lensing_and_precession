import os
import re
import argparse
import pickle
from typing import Optional

import numpy as np

# Use non-interactive backend for cluster environments
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt


def derive_fig_path_from_pickle(pickle_path: str, tag: Optional[str] = None) -> str:
    data_dir = os.path.dirname(pickle_path)
    root_dir = os.path.dirname(data_dir)
    figures_dir = os.path.join(root_dir, "figures")
    os.makedirs(figures_dir, exist_ok=True)

    filename = os.path.basename(pickle_path)
    # Strip the trailing timestamp _YYYY-MM-DD_HH-MM-SS if present
    name_wo_ext = os.path.splitext(filename)[0]
    m = re.match(r"^(.*)_\d{4}-\d{2}-\d{2}_\d{2}-\d{2}-\d{2}$", name_wo_ext)
    base = m.group(1) if m else name_wo_ext
    if tag:
        base = f"{base}_{tag}"
    return os.path.join(figures_dir, f"{base}.pdf")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Render contour plot from saved pickle output.",
    )
    parser.add_argument(
        "--pkl",
        required=True,
        help="Path to saved pickle file produced by script.",
    )
    parser.add_argument(
        "--fig",
        default=None,
        help="Optional output figure path (.pdf). If omitted, derived from pickle name.",
    )
    parser.add_argument(
        "--tag",
        default="",
        help="Optional tag appended to derived figure basename (default: empty).",
    )
    parser.add_argument(
        "--theta_points",
        type=int,
        default=None,
        help=(
            "If set and smaller than the stored grid, subsample the theta dimension (rows) "
            "to this many points before plotting."
        ),
    )
    parser.add_argument(
        "--levels",
        type=int,
        default=100,
        help="Number of contour levels for plotting (default: 100).",
    )
    args = parser.parse_args()

    pickle_path = os.path.abspath(args.pkl)
    if not os.path.isfile(pickle_path):
        raise FileNotFoundError(f"Pickle not found: {pickle_path}")

    with open(pickle_path, "rb") as f:
        out = pickle.load(f)

    X = out["omega_matrix"]
    Y = out["theta_matrix"]
    Z = out["epsilon_matrix"]

    # Validate shapes
    if not (
        isinstance(X, np.ndarray)
        and isinstance(Y, np.ndarray)
        and isinstance(Z, np.ndarray)
    ):
        raise ValueError(
            "omega_matrix, theta_matrix, epsilon_matrix must be numpy arrays"
        )
    if X.shape != Y.shape or X.shape != Z.shape:
        raise ValueError(f"Mismatched shapes: X{X.shape}, Y{Y.shape}, Z{Z.shape}")

    # Optional theta subsampling (reduce number of rows)
    tag = args.tag
    if (
        args.theta_points is not None
        and isinstance(X, np.ndarray)
        and args.theta_points > 0
        and args.theta_points < X.shape[0]
    ):
        n_theta_orig = X.shape[0]
        idx = np.linspace(0, n_theta_orig - 1, args.theta_points, dtype=int)
        idx = np.unique(idx)
        if idx.size < args.theta_points:
            # In rare cases of duplicates due to rounding, pad by adding missing indices
            missing = args.theta_points - idx.size
            extra = np.setdiff1d(np.arange(n_theta_orig), idx)[:missing]
            idx = np.sort(np.concatenate([idx, extra]))
        print(
            f"Subsampling theta dimension from {n_theta_orig} to {idx.size} rows for plotting."
        )
        X = X[idx, :]
        Y = Y[idx, :]
        Z = Z[idx, :]
        if not tag:
            tag = f"theta{idx.size}"

    # Add levels tag if not default
    if args.levels != 100:
        levels_tag = f"levels{args.levels}"
        tag = f"{tag}_{levels_tag}" if tag else levels_tag

    fig_path = (
        args.fig
        if args.fig
        else derive_fig_path_from_pickle(pickle_path, tag=tag if tag else None)
    )

    cf = plt.contourf(X, Y, Z, levels=args.levels, cmap="jet")
    cbar = plt.colorbar(cf)
    cbar.set_label(r"$\epsilon(\tilde{h}_{\mathrm{L}}, \tilde{h}_{\mathrm{RP}})$")
    plt.xlabel(r"$\tilde{\Omega}$")
    plt.ylabel(r"$\tilde{\theta}$")

    # Set aspect ratio to be rectangular based on the number of omega vs theta points
    omega_range = X.max() - X.min()
    theta_range = Y.max() - Y.min()
    aspect_ratio = (omega_range / X.shape[1]) / (theta_range / X.shape[0])
    plt.gca().set_aspect(aspect_ratio)

    plt.tight_layout()
    plt.savefig(fig_path, dpi=200)
    print("Figure saved as", fig_path)


if __name__ == "__main__":
    main()
