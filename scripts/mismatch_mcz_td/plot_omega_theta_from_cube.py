"""Plot epsilon contours over (theta, omega) at a specific td from a per-mcz cube."""

import os
import sys
import argparse
from typing import Optional

import numpy as np
import h5py
import matplotlib.pyplot as plt

# Ensure project root is on PYTHONPATH
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))


def _find_td_index(td_seconds: np.ndarray, target_td_ms: float) -> int:
    target_s = float(target_td_ms) * 1e-3
    return int(np.argmin(np.abs(td_seconds - target_s)))


def main(
    input_h5: str,
    td_ms: float,
    output_path: Optional[str],
    levels: int,
    cmap: str,
    dpi: int,
) -> None:
    if not os.path.isfile(input_h5):
        raise FileNotFoundError(f"Input cube not found: {input_h5}")

    with h5py.File(input_h5, "r") as h5:
        for ds in ("td", "theta", "omega", "epsilon_min_grid"):
            if ds not in h5:
                raise KeyError(
                    f"Dataset '{ds}' missing in {input_h5}; found keys: {list(h5.keys())}"
                )
        td = np.array(h5["td"], dtype=float)
        theta = np.array(h5["theta"], dtype=float)
        omega = np.array(h5["omega"], dtype=float)
        eps = np.array(h5["epsilon_min_grid"], dtype=float)

    td_idx = _find_td_index(td, td_ms)
    Z = eps[td_idx]

    X, Y = np.meshgrid(omega, theta)
    fig, ax = plt.subplots(figsize=(6.8, 5.2))
    cf = ax.contourf(X, Y, Z, levels=levels, cmap=cmap)
    cbar = fig.colorbar(cf)
    cbar.set_label(r"$\epsilon(\tilde{h}_L, \tilde{h}_P)$")
    ax.set_xlabel(r"$\tilde{\Omega}$")
    ax.set_ylabel(r"$\tilde{\theta}$")
    ax.set_title(f"td = {td[td_idx]*1e3:.1f} ms")
    fig.tight_layout()

    if output_path is None:
        base = os.path.splitext(os.path.basename(input_h5))[0]
        output_dir = os.path.join(
            os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
            "figures",
            "indiv",
        )
        os.makedirs(output_dir, exist_ok=True)
        td_tag = f"td{float(td_ms):.1f}ms".replace(".", "p")
        output_path = os.path.join(output_dir, f"{base}_{td_tag}.png")
    else:
        out_dir = os.path.dirname(output_path)
        if out_dir:
            os.makedirs(out_dir, exist_ok=True)

    fig.savefig(output_path, dpi=dpi, bbox_inches="tight")
    print(f"Saved plot: {output_path}")
    plt.close(fig)


if __name__ == "__main__":
    p = argparse.ArgumentParser(
        description="Plot (theta, omega) epsilon contours at fixed td from a mismatch cube"
    )
    p.add_argument("--input", required=True, help="Path to per-mcz mismatch cube .h5")
    p.add_argument(
        "--td_ms", type=float, required=True, help="Time delay in milliseconds"
    )
    p.add_argument("--out", default=None, help="Output image path (PNG/PDF)")
    p.add_argument("--levels", type=int, default=100)
    p.add_argument("--cmap", type=str, default="jet")
    p.add_argument("--dpi", type=int, default=160)
    args = p.parse_args()

    main(args.input, args.td_ms, args.out, args.levels, args.cmap, args.dpi)
