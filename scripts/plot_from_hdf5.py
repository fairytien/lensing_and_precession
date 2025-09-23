import os, argparse, sys
from typing import Optional

import numpy as np
import h5py
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker

# Ensure project root is on path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from modules.filenames import contour_td_mcz_filename


def main(
    input_h5: str,
    output_path: Optional[str],
    dpi: int,
    cmap: str,
    levels: int,
    show: bool,
    td_decimals: int,
    mcz_decimals: int,
    end_tag: Optional[str] = None,
) -> None:
    # Input validation
    if not os.path.isfile(input_h5):
        raise FileNotFoundError(f"Input file not found: {input_h5}")
    if levels <= 0:
        raise ValueError(f"levels must be positive, got {levels}")
    if dpi <= 0:
        raise ValueError(f"dpi must be positive, got {dpi}")
    if td_decimals < 0:
        raise ValueError(f"td_decimals must be non-negative, got {td_decimals}")
    if mcz_decimals < 0:
        raise ValueError(f"mcz_decimals must be non-negative, got {mcz_decimals}")

    repo_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

    with h5py.File(input_h5, "r") as h5:
        # Check required datasets exist
        required_datasets = ["mcz", "td", "epsilon_min"]
        missing = [ds for ds in required_datasets if ds not in h5]
        if missing:
            raise KeyError(f"Missing required datasets in {input_h5}: {missing}")

        mcz = np.array(h5["mcz"], dtype=np.float64)  # (mcz_pts,)
        td_s = np.array(h5["td"], dtype=np.float64)  # (td_pts,) in seconds
        Z = np.array(h5["epsilon_min"], dtype=np.float64)  # (mcz_pts, td_pts)

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
    if np.all(Z == Z.flat[0]):
        print("Warning: epsilon_min appears to be constant")

    if output_path is None:
        # Use centralized filename builder with orientation tag
        td_min_ms = float(np.min(td_ms))
        td_max_ms = float(np.max(td_ms))
        mcz_min = float(np.min(mcz))
        mcz_max = float(np.max(mcz))
        fig_dir = os.path.join(repo_root, "figures")

        # Extract orientation tag from filename
        input_basename = os.path.basename(input_h5)
        # Try to extract tag from filename like "best_match_td20-22ms_mcz10-12Msun_f20_df1.00_Taman_faceon.h5"
        if "_" in input_basename:
            parts = input_basename.replace(".h5", "").split("_")
            if len(parts) >= 2:
                # Last two parts should be author_orientation (e.g., "Taman_faceon")
                orientation_tag = "_".join(parts[-2:])
            else:
                orientation_tag = "unknown"
        else:
            orientation_tag = "unknown"

        # Append end_tag if provided
        if end_tag is not None:
            orientation_tag = f"{orientation_tag}_{end_tag}"

        output_path = contour_td_mcz_filename(
            fig_dir=fig_dir,
            td_min_ms=td_min_ms,
            td_max_ms=td_max_ms,
            mcz_min=mcz_min,
            mcz_max=mcz_max,
            orientation_tag=orientation_tag,
            ext="pdf",
        )
    else:
        # Only create directory if output_path has one
        output_dir = os.path.dirname(output_path)
        if output_dir:
            os.makedirs(output_dir, exist_ok=True)

    # Create meshgrid for contour plotting
    TD, MCZ = np.meshgrid(td_ms, mcz)

    # Create figure and plot
    fig, ax = plt.subplots(figsize=(8, 6))
    cf = ax.contourf(TD, MCZ, Z, levels=levels, cmap=cmap)
    cbar = fig.colorbar(cf, ax=ax)
    cbar.set_label(
        r"$\min_{\~\Omega, \~\theta, \gamma_P}$ $\epsilon(\tilde{h}_L, \tilde{h}_P)$"
    )
    ax.set_xlabel(r"$\Delta t_d$ [ms]")
    ax.set_ylabel(r"$\mathcal{M}_s\ [M_\odot]$")

    # TODO: Ensure square aspect ratio for the contour plot (excluding colorbar)

    # Format axis ticks with specified decimal precision
    if td_decimals >= 0:
        ax.xaxis.set_major_formatter(
            ticker.FuncFormatter(lambda x, p: f"{x:.{td_decimals}f}")
        )

    if mcz_decimals >= 0:
        ax.yaxis.set_major_formatter(
            ticker.FuncFormatter(lambda x, p: f"{x:.{mcz_decimals}f}")
        )
    # Save and display
    fig.tight_layout()
    fig.savefig(output_path, dpi=dpi, bbox_inches="tight")
    print(f"Saved plot: {output_path}")

    if show:
        plt.show()
    else:
        plt.close(fig)  # Clean up memory


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
        default=1,
        help="Number of decimal places for time-delay x-axis ticks (default: 1, e.g., 1.0ms)",
    )
    p.add_argument(
        "--mcz_decimals",
        type=int,
        default=1,
        help="Number of decimal places for chirp mass y-axis ticks (default: 1, e.g., 20.0 Msun)",
    )
    p.add_argument(
        "--end_tag",
        type=str,
        default=None,
        help="Optional suffix to append to the orientation tag extracted from input filename.",
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
        mcz_decimals=args.mcz_decimals,
        end_tag=args.end_tag,
    )
