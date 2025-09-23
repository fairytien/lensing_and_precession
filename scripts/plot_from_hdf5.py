import os
import argparse
from typing import Optional

import numpy as np
import h5py


def _default_fig_path(repo_root: str, td_ms: np.ndarray, mcz: np.ndarray) -> str:
    td_min_ms = float(np.min(td_ms))
    td_max_ms = float(np.max(td_ms))
    mcz_min = float(np.min(mcz))
    mcz_max = float(np.max(mcz))
    fig_dir = os.path.join(repo_root, "figures")
    os.makedirs(fig_dir, exist_ok=True)
    return os.path.join(
        fig_dir,
        f"contour_td{td_min_ms:.0f}-{td_max_ms:.0f}ms_mcz{mcz_min:.0f}-{mcz_max:.0f}Msun_min_mismatch.pdf",
    )


def main(
    input_h5: str,
    output_path: Optional[str],
    dpi: int,
    cmap: str,
    levels: int,
    show: bool,
    td_decimals: int,
) -> None:
    # Input validation
    if not os.path.isfile(input_h5):
        raise FileNotFoundError(f"Input file not found: {input_h5}")
    if levels <= 0:
        raise ValueError(f"levels must be positive, got {levels}")
    if dpi <= 0:
        raise ValueError(f"dpi must be positive, got {dpi}")

    repo_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

    with h5py.File(input_h5, "r") as h5:
        # Check required datasets exist
        required_datasets = ["mcz", "td", "epsilon_min"]
        missing = [ds for ds in required_datasets if ds not in h5]
        if missing:
            raise KeyError(f"Missing required datasets in {input_h5}: {missing}")

        mcz = np.array(h5["mcz"], dtype=float)  # (mcz_pts,)
        td_s = np.array(h5["td"], dtype=float)  # (td_pts,) in seconds
        Z = np.array(h5["epsilon_min"], dtype=float)  # (mcz_pts, td_pts)

    td_ms = td_s * 1e3

    if Z.shape != (mcz.shape[0], td_ms.shape[0]):
        raise ValueError(
            f"epsilon_min shape {Z.shape} incompatible with mcz {mcz.shape} and td {td_ms.shape}"
        )

    # Check for problematic data
    if np.any(np.isnan(Z)):
        print("Warning: epsilon_min contains NaN values")
    if np.any(np.isinf(Z)):
        print("Warning: epsilon_min contains infinite values")

    if output_path is None:
        output_path = _default_fig_path(repo_root, td_ms, mcz)
    else:
        # Only create directory if output_path has one
        output_dir = os.path.dirname(output_path)
        if output_dir:
            os.makedirs(output_dir, exist_ok=True)

    import matplotlib.pyplot as plt

    TD, MCZ = np.meshgrid(td_ms, mcz)
    fig = plt.figure(figsize=(8, 6))
    ax = fig.add_subplot(111)
    cf = ax.contourf(TD, MCZ, Z, levels=levels, cmap=cmap)
    cbar = fig.colorbar(cf, ax=ax)
    cbar.set_label(r"$\epsilon(\tilde{h}_L, \tilde{h}_P)$")
    ax.set_xlabel(r"$\Delta t_d$ [ms]")
    ax.set_ylabel(r"$\mathcal{M}_s\ [M_\odot]$")

    # Format x-axis ticks with specified decimal precision
    if td_decimals >= 0:
        import matplotlib.ticker as ticker

        ax.xaxis.set_major_formatter(
            ticker.FuncFormatter(lambda x, p: f"{x:.{td_decimals}f}")
        )
    fig.tight_layout()
    fig.savefig(output_path, dpi=dpi)

    print(f"Saved plot: {output_path}")

    if show:
        plt.show()


if __name__ == "__main__":
    p = argparse.ArgumentParser(
        description="Plot mismatch contour from best_match HDF5 file."
    )
    p.add_argument(
        "--input",
        required=True,
        help="Path to best_match_*.h5 produced by contourTd script",
    )
    p.add_argument(
        "--out",
        default=None,
        help="Output figure path (PDF/PNG). Defaults to repo figures path",
    )
    p.add_argument("--dpi", type=int, default=200)
    p.add_argument("--cmap", type=str, default="jet")
    p.add_argument("--levels", type=int, default=100)
    p.add_argument("--show", action="store_true")
    p.add_argument(
        "--td_decimals",
        type=int,
        default=0,
        help="Number of decimal places for time-delay x-axis ticks (default: 0, i.e., integers)",
    )
    args = p.parse_args()

    main(
        input_h5=args.input,
        output_path=args.out,
        dpi=args.dpi,
        cmap=args.cmap,
        levels=args.levels,
        show=args.show,
        td_decimals=args.td_decimals,
    )
