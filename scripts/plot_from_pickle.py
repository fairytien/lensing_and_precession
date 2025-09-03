import os
import re
import argparse
import pickle

import numpy as np

# Use non-interactive backend for cluster environments
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt


def derive_fig_path_from_pickle(pickle_path: str, tag: str | None = None) -> str:
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

    fig_path = (
        args.fig if args.fig else derive_fig_path_from_pickle(pickle_path, tag=args.tag)
    )

    cf = plt.contourf(X, Y, Z, levels=100, cmap="jet")
    cbar = plt.colorbar(cf)
    cbar.set_label(r"$\epsilon(\tilde{h}_{\mathrm{L}}, \tilde{h}_{\mathrm{RP}})$")
    plt.xlabel(r"$\tilde{\Omega}$")
    plt.ylabel(r"$\tilde{\theta}$")
    plt.tight_layout()
    plt.savefig(fig_path, dpi=200)
    print("Figure saved as", fig_path)


if __name__ == "__main__":
    main()
