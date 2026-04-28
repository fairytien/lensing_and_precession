"""
Validate the hypothesis that the unimodal region of the mismatch surface
in the (omega_tilde, theta_tilde) plane corresponds to ~2-3 lensing
oscillations in band.

Produces:
  Figure 1: (mcz, td) map of min-mismatch with n_lens contour overlays
  Figure 2: Gradient of best-fit (omega, theta) revealing multimodality
            boundaries, with n_lens contours
  Figure 3: Panel of individual (omega, theta) mismatch surfaces at
            representative (mcz, td) points spanning different n_lens values,
            with local minima marked
  Figure 4: Best-fit omega and theta vs td within the ~unimodal band
  Figure 5: Scatter plot of n_minima vs n_lens from all available individual
            contour data
"""

import sys, os, glob, pickle, argparse

import numpy as np
import h5py
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib import ticker
from scipy.ndimage import gaussian_filter, sobel

sys.path.insert(
    0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
)

from modules.waveform import (
    number_of_lens_cycles,
    mcz_for_n_lens_cycles,
    get_fcut_from_mcz,
)
from modules.plot_utils import apply_physics_paper_style

from scipy.optimize import minimize
from scipy.interpolate import RegularGridInterpolator


def _is_near(p1, p2, threshold=0.5):
    return np.linalg.norm(np.array(p1) - np.array(p2)) < threshold


def find_local_minima(Z, x, y, print_results=False):
    """Find local minima in a 2D surface Z(y, x) via multi-start Nelder-Mead.

    Standalone reimplementation avoiding the legacy import chain in
    modules.multimodality.
    """
    Zt = Z.T  # (n_x, n_y) for RegularGridInterpolator
    interp = RegularGridInterpolator((x, y), Zt, bounds_error=False, fill_value=np.inf)

    def obj(xy):
        if xy[0] < x[0] or xy[0] > x[-1] or xy[1] < y[0] or xy[1] > y[-1]:
            return np.inf
        return float(interp(xy))

    starts = [
        (x[0], y[0]),
        (x[-1], y[0]),
        (x[0], y[-1]),
        (x[-1], y[-1]),
        (x[len(x) // 2], y[len(y) // 2]),
        (x[0], y[len(y) // 2]),
        (x[-1], y[len(y) // 2]),
        (x[len(x) // 2], y[0]),
        (x[len(x) // 2], y[-1]),
    ]

    results = []
    for pt in starts:
        res = minimize(obj, pt, method="Nelder-Mead")
        results.append((np.round(res.x, 1), res.fun))

    # filter near-duplicates
    filtered = []
    for coord, zv in results:
        ct = tuple(coord)
        if not any(_is_near(ct, tuple(ec), 0.5) for ec, _ in filtered):
            filtered.append((coord, zv))

    Z_mean = np.mean(Zt)
    filtered = [(c, zv) for c, zv in filtered if zv < Z_mean]

    if print_results:
        for c, zv in filtered:
            print(f"Local minimum at {c}: {zv}")
    return filtered


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
SOLMASS2SEC = 4.92624076e-6


def load_best_match(path):
    """Load aggregated best-match HDF5 and return arrays in physical units."""
    with h5py.File(path, "r") as f:
        mcz_msun = f["mcz"][:]  # Msun
        td_s = f["td"][:]  # seconds
        epsilon_min = f["epsilon_min"][:]  # (n_mcz, n_td)
        omega_best = f["omega_best"][:]
        theta_best = f["theta_best"][:]
        gamma_best = f["gamma_best"][:]
        if "expected_I" in f:
            raise ValueError(
                "modality_nlens expects an mcz_td best-match file with an mcz sweep and fixed scalar I; aggregated I_td best-match files are not supported."
            )
        if "I" in f.attrs:
            I = float(f.attrs["I"])
        elif "source_param_I" in f.attrs:
            I = float(f.attrs["source_param_I"])
        else:
            raise KeyError(
                "Missing fixed source flux ratio metadata on best-match file."
            )
        z = float(f.attrs.get("z", f.attrs.get("source_param_z", 0)))
    return dict(
        mcz=mcz_msun,
        td=td_s,
        epsilon_min=epsilon_min,
        omega_best=omega_best,
        theta_best=theta_best,
        gamma_best=gamma_best,
        I=I,
        z=z,
    )


def compute_nlens_grid(mcz_msun, td_s, z=0.0):
    """Return n_lens on (mcz, td) meshgrid.

    Parameters
    ----------
    mcz_msun : array
        Source-frame chirp masses in Msun.
    td_s : array
        Time delays in seconds.
    z : float
        Source redshift.  The stored mcz values are source-frame;
        number_of_lens_cycles needs detector-frame mcz to obtain the
        detector-frame f_cut, so we multiply by (1+z).
    """
    TD, MCZ = np.meshgrid(td_s, mcz_msun)
    return number_of_lens_cycles(MCZ * (1 + z), TD)


def gradient_magnitude(arr, dx, dy):
    """Finite-difference gradient magnitude of 2D array."""
    gy, gx = np.gradient(arr, dx, dy)
    return np.sqrt(gx**2 + gy**2)


def gradient_magnitude_normalized(arr):
    """Index-based gradient magnitude (avoids axis-scale bias)."""
    gy, gx = np.gradient(arr)
    return np.sqrt(gx**2 + gy**2)


def count_minima_from_contour(
    epsilon_matrix, omega_matrix, theta_matrix, significance_factor=0.5
):
    """Count local minima in a (theta, omega) mismatch surface.

    Only counts minima whose depth is within significance_factor * range
    of the global minimum. This filters shallow/spurious local minima.
    """
    omega_1d = omega_matrix[0, :]  # omega varies along columns
    theta_1d = theta_matrix[:, 0]  # theta varies along rows
    # find_local_minima expects Z[i,j] indexed by (x=omega, y=theta)
    # and epsilon_matrix is (n_theta, n_omega), so we need to transpose
    minima = find_local_minima(
        epsilon_matrix,
        x=omega_1d,
        y=theta_1d,
        print_results=False,
    )
    if not minima:
        return minima

    # Apply significance filter
    ep_min = min(z for _, z in minima)
    ep_range = epsilon_matrix.max() - ep_min
    threshold = ep_min + significance_factor * ep_range
    significant = [(c, z) for c, z in minima if z <= threshold]
    return significant if significant else minima[:1]


def load_indiv_contours(data_dir, z_filter=None):
    """Load all v3 individual contour pickles, optionally filtering by z."""
    pattern = os.path.join(data_dir, "v3_indiv_contour_*.pkl")
    files = sorted(glob.glob(pattern))
    contours = []
    for fpath in files:
        try:
            with open(fpath, "rb") as f:
                d = pickle.load(f)
            basename = os.path.basename(fpath)
            # Parse z from filename
            z_val = None
            if "_z" in basename:
                z_part = basename.split("_z")[1].split("_")[0]
                try:
                    z_val = float(z_part)
                except ValueError:
                    z_val = None
            if z_filter is not None and z_val != z_filter:
                continue
            d["z"] = z_val
            d["filepath"] = fpath
            contours.append(d)
        except Exception:
            continue
    return contours


# ---------------------------------------------------------------------------
# Figure 1: Min-mismatch map with n_lens contours
# ---------------------------------------------------------------------------
def fig1_mismatch_nlens(data, outdir):
    mcz, td = data["mcz"], data["td"]
    td_ms = td * 1e3
    nlens = compute_nlens_grid(mcz, td, z=data["z"])

    fig, ax = plt.subplots(figsize=(8, 6))
    cf = ax.contourf(
        td_ms,
        mcz,
        np.log10(data["epsilon_min"]),
        levels=50,
        cmap="jet",
    )
    cbar = fig.colorbar(cf, ax=ax)
    cbar.set_label(r"$\log_{10}\,\epsilon_{\min}$")

    # n_lens contour lines
    for n, ls, col in [
        (1, "--", "white"),
        (2, "-", "red"),
        (3, "-", "red"),
        (4, "--", "white"),
        (5, ":", "white"),
    ]:
        cs = ax.contour(
            td_ms,
            mcz,
            nlens,
            levels=[n],
            colors=[col],
            linestyles=[ls],
            linewidths=1.5,
        )
        ax.clabel(cs, fmt=rf"$N_\mathrm{{lens}}={n}$", fontsize=9)

    # Shade the 2-3 band
    ax.contourf(
        td_ms,
        mcz,
        nlens,
        levels=[2, 3],
        colors=["red"],
        alpha=0.12,
    )

    ax.set_xlabel(r"$\Delta t_d$ [ms]")
    ax.set_ylabel(r"$\mathcal{M}_z$ [$M_\odot$]")
    ax.set_title(
        r"Minimum mismatch with $N_\mathrm{lens}$ contours"
        f"\n($I={data['I']:.1f}$, $z={data['z']}$, edge-on)"
    )
    fig.tight_layout()
    fig.savefig(os.path.join(outdir, "fig1_mismatch_nlens.pdf"), dpi=200)
    plt.close(fig)
    print("  Saved fig1_mismatch_nlens")


# ---------------------------------------------------------------------------
# Figure 2: Gradient of best-fit parameters → multimodality boundaries
# ---------------------------------------------------------------------------
def fig2_gradient_multimodality(data, outdir):
    mcz, td = data["mcz"], data["td"]
    td_ms = td * 1e3
    nlens = compute_nlens_grid(mcz, td, z=data["z"])

    # Index-based gradient to avoid mixing mcz (Msun) and td (ms) scales
    omega_grad = gradient_magnitude_normalized(data["omega_best"])
    theta_grad = gradient_magnitude_normalized(data["theta_best"])
    combined_grad = np.sqrt(omega_grad**2 + theta_grad**2)

    # Smooth slightly for visual clarity
    combined_smooth = gaussian_filter(combined_grad, sigma=1.0)

    fig, axes = plt.subplots(1, 3, figsize=(18, 5.5))

    # Panel a: omega gradient
    cf0 = axes[0].contourf(td_ms, mcz, omega_grad, levels=50, cmap="jet")
    fig.colorbar(cf0, ax=axes[0], label=r"$|\nabla \tilde{\Omega}_\mathrm{best}|$")
    axes[0].set_title(r"$|\nabla \tilde{\Omega}_\mathrm{best}|$")

    # Panel b: theta gradient
    cf1 = axes[1].contourf(td_ms, mcz, theta_grad, levels=50, cmap="jet")
    fig.colorbar(cf1, ax=axes[1], label=r"$|\nabla \tilde{\theta}_\mathrm{best}|$")
    axes[1].set_title(r"$|\nabla \tilde{\theta}_\mathrm{best}|$")

    # Panel c: combined gradient
    cf2 = axes[2].contourf(td_ms, mcz, combined_smooth, levels=50, cmap="jet")
    fig.colorbar(cf2, ax=axes[2], label=r"$|\nabla|$ combined")
    axes[2].set_title("Combined gradient (smoothed)")

    for ax in axes:
        for n, ls, col in [
            (1, ":", "cyan"),
            (2, "-", "cyan"),
            (3, "-", "cyan"),
            (4, ":", "cyan"),
        ]:
            cs = ax.contour(
                td_ms,
                mcz,
                nlens,
                levels=[n],
                colors=[col],
                linestyles=[ls],
                linewidths=1.5,
            )
            ax.clabel(cs, fmt=rf"$N_\mathrm{{lens}}={n}$", fontsize=8)
        ax.set_xlabel(r"$\Delta t_d$ [ms]")
        ax.set_ylabel(r"$\mathcal{M}_z$ [$M_\odot$]")

    fig.suptitle(
        "Gradient of best-fit template parameters (high gradient → multimodal boundary)\n"
        f"$I={data['I']:.1f}$, $z={data['z']}$, edge-on",
        fontsize=13,
    )
    fig.tight_layout(rect=[0, 0, 1, 0.92])
    fig.savefig(os.path.join(outdir, "fig2_gradient_multimodality.pdf"), dpi=200)
    plt.close(fig)
    print("  Saved fig2_gradient_multimodality")


# ---------------------------------------------------------------------------
# Figure 3: Example mismatch surfaces at different n_lens values
# ---------------------------------------------------------------------------
def fig3_example_surfaces(contours, outdir):
    """Plot (omega, theta) surfaces for contours at different n_lens values."""
    if not contours:
        print("  No individual contours available for fig3 — skipping")
        return

    # Compute n_lens for each contour
    for c in contours:
        c_z = c.get("z", 0.0) or 0.0
        c["n_lens"] = float(
            number_of_lens_cycles(c["mcz_msun"] * (1 + c_z), c["td_ms"] * 1e-3)
        )

    # Sort by n_lens
    contours_sorted = sorted(contours, key=lambda c: c["n_lens"])

    # Select up to 6 representative contours spanning the n_lens range
    n_total = len(contours_sorted)
    if n_total <= 6:
        selected = contours_sorted
    else:
        indices = np.linspace(0, n_total - 1, 6, dtype=int)
        selected = [contours_sorted[i] for i in indices]

    n_panels = len(selected)
    ncols = min(3, n_panels)
    nrows = (n_panels + ncols - 1) // ncols

    fig, axes = plt.subplots(nrows, ncols, figsize=(6 * ncols, 5 * nrows))
    if n_panels == 1:
        axes = np.array([axes])
    axes = np.atleast_2d(axes)

    for idx, c in enumerate(selected):
        row, col = divmod(idx, ncols)
        ax = axes[row, col]

        omega_1d = c["omega_matrix"][0, :]
        theta_1d = c["theta_matrix"][:, 0]
        ep = c["epsilon_matrix"]
        n_lens_val = c["n_lens"]

        cf = ax.contourf(omega_1d, theta_1d, ep, levels=50, cmap="jet")
        fig.colorbar(cf, ax=ax, shrink=0.8)

        # Find and mark local minima
        minima = count_minima_from_contour(ep, c["omega_matrix"], c["theta_matrix"])
        n_min = len(minima)
        for coord, z_val in minima:
            ax.plot(
                coord[0],
                coord[1],
                "r*",
                markersize=12,
                markeredgecolor="white",
                markeredgewidth=0.5,
            )

        ax.set_xlabel(r"$\tilde{\Omega}$")
        ax.set_ylabel(r"$\tilde{\theta}$")
        ax.set_title(
            rf"$\mathcal{{M}}_z={c['mcz_msun']:.0f}\,M_\odot$, "
            rf"$\Delta t_d={c['td_ms']:.0f}$ ms"
            f"\n$N_\\mathrm{{lens}}={n_lens_val:.1f}$, "
            f"$N_\\mathrm{{min}}={n_min}$",
            fontsize=10,
        )

    # Hide extra axes
    for idx in range(n_panels, nrows * ncols):
        row, col = divmod(idx, ncols)
        axes[row, col].set_visible(False)

    fig.suptitle(
        r"Mismatch $\epsilon(\tilde{\Omega}, \tilde{\theta})$ surfaces"
        " at different $N_\\mathrm{lens}$\n"
        "(red stars = local minima)",
        fontsize=13,
    )
    fig.tight_layout(rect=[0, 0, 1, 0.93])
    fig.savefig(os.path.join(outdir, "fig3_example_surfaces.pdf"), dpi=200)
    plt.close(fig)
    print("  Saved fig3_example_surfaces")


# ---------------------------------------------------------------------------
# Figure 4: Best-fit omega, theta vs td in the unimodal band
# ---------------------------------------------------------------------------
def fig4_bestfit_vs_td(data, outdir):
    mcz, td = data["mcz"], data["td"]
    td_ms = td * 1e3
    nlens = compute_nlens_grid(mcz, td, z=data["z"])

    fig, axes = plt.subplots(2, 2, figsize=(13, 10))

    # Select a few mcz values spanning the range
    mcz_samples = [15, 20, 30, 40, 50, 60]
    mcz_samples = [m for m in mcz_samples if m >= mcz.min() and m <= mcz.max()]
    cmap = plt.cm.coolwarm
    colors = [cmap(i / max(1, len(mcz_samples) - 1)) for i in range(len(mcz_samples))]

    for i_m, mcz_val in enumerate(mcz_samples):
        # Find closest mcz index
        idx_mcz = np.argmin(np.abs(mcz - mcz_val))
        actual_mcz = mcz[idx_mcz]

        omega_row = data["omega_best"][idx_mcz, :]
        theta_row = data["theta_best"][idx_mcz, :]
        eps_row = data["epsilon_min"][idx_mcz, :]
        nlens_row = nlens[idx_mcz, :]

        label = rf"$\mathcal{{M}}_z={actual_mcz:.0f}\,M_\odot$"

        axes[0, 0].plot(td_ms, omega_row, "-o", ms=3, label=label, color=colors[i_m])
        axes[0, 1].plot(td_ms, theta_row, "-o", ms=3, label=label, color=colors[i_m])
        axes[1, 0].plot(td_ms, eps_row, "-o", ms=3, label=label, color=colors[i_m])
        axes[1, 1].plot(
            nlens_row, omega_row, "o", ms=4, label=label, color=colors[i_m], alpha=0.6
        )

    axes[0, 0].set_ylabel(r"$\tilde{\Omega}_\mathrm{best}$")
    axes[0, 0].set_xlabel(r"$\Delta t_d$ [ms]")
    axes[0, 0].legend(fontsize=8, ncol=2)
    axes[0, 0].set_title(r"Best-fit $\tilde{\Omega}$ vs $\Delta t_d$")

    axes[0, 1].set_ylabel(r"$\tilde{\theta}_\mathrm{best}$")
    axes[0, 1].set_xlabel(r"$\Delta t_d$ [ms]")
    axes[0, 1].set_title(r"Best-fit $\tilde{\theta}$ vs $\Delta t_d$")

    axes[1, 0].set_ylabel(r"$\epsilon_\mathrm{min}$")
    axes[1, 0].set_xlabel(r"$\Delta t_d$ [ms]")
    axes[1, 0].set_title(r"$\epsilon_\mathrm{min}$ vs $\Delta t_d$")

    axes[1, 1].set_ylabel(r"$\tilde{\Omega}_\mathrm{best}$")
    axes[1, 1].set_xlabel(r"$N_\mathrm{lens}$")
    axes[1, 1].set_title(r"$\tilde{\Omega}_\mathrm{best}$ vs $N_\mathrm{lens}$")
    for n_val in [2, 3]:
        axes[1, 1].axvline(
            n_val,
            color="red",
            ls="--",
            alpha=0.5,
            label=rf"$N_\mathrm{{lens}}={n_val}$" if n_val == 2 else "",
        )
    axes[1, 1].legend(fontsize=8, ncol=2)

    # Shade unimodal region in the td-based panels
    for ax_row in [axes[0, 0], axes[0, 1], axes[1, 0]]:
        ax_row.grid(True, alpha=0.3)

    fig.suptitle(
        f"Best-fit parameters vs time delay ($I={data['I']:.1f}$, $z={data['z']}$, edge-on)",
        fontsize=13,
    )
    fig.tight_layout(rect=[0, 0, 1, 0.95])
    fig.savefig(os.path.join(outdir, "fig4_bestfit_vs_td.pdf"), dpi=200)
    plt.close(fig)
    print("  Saved fig4_bestfit_vs_td")


# ---------------------------------------------------------------------------
# Figure 5: n_minima vs n_lens scatter from individual contours
# ---------------------------------------------------------------------------
def fig5_nminima_vs_nlens(contours, outdir):
    """Scatter plot: number of local minima vs n_lens from individual surfaces."""
    if not contours:
        print("  No individual contours available for fig5 — skipping")
        return

    n_lens_arr = []
    n_min_arr = []
    mcz_arr = []
    labels = []

    for c in contours:
        c_z = c.get("z", 0.0) or 0.0
        n_lens_val = float(
            number_of_lens_cycles(c["mcz_msun"] * (1 + c_z), c["td_ms"] * 1e-3)
        )
        minima = count_minima_from_contour(
            c["epsilon_matrix"], c["omega_matrix"], c["theta_matrix"]
        )
        n_min = len(minima)
        n_lens_arr.append(n_lens_val)
        n_min_arr.append(n_min)
        mcz_arr.append(c["mcz_msun"])
        labels.append(
            rf"$\mathcal{{M}}_z={c['mcz_msun']:.0f}$, " rf"$t_d={c['td_ms']:.0f}$ms"
        )

    n_lens_arr = np.array(n_lens_arr)
    n_min_arr = np.array(n_min_arr)
    mcz_arr = np.array(mcz_arr)

    fig, ax = plt.subplots(figsize=(8, 6))
    sc = ax.scatter(
        n_lens_arr,
        n_min_arr,
        c=mcz_arr,
        cmap="coolwarm",
        s=80,
        edgecolors="black",
        linewidths=0.5,
        zorder=5,
    )
    cbar = fig.colorbar(sc, ax=ax)
    cbar.set_label(r"$\mathcal{M}_z$ [$M_\odot$]")

    # Annotate each point
    for i, lbl in enumerate(labels):
        ax.annotate(
            lbl,
            (n_lens_arr[i], n_min_arr[i]),
            fontsize=6,
            ha="left",
            va="bottom",
            xytext=(4, 4),
            textcoords="offset points",
        )

    # Shade the hypothesized unimodal band
    ax.axvspan(
        2, 3, alpha=0.15, color="green", label=r"$2 \leq N_\mathrm{lens} \leq 3$"
    )
    ax.axhline(1, color="gray", ls=":", alpha=0.5)

    ax.set_xlabel(
        r"$N_\mathrm{lens} = (f_\mathrm{cut} - f_\mathrm{min}) \times \Delta t_d$"
    )
    ax.set_ylabel(
        r"Number of local minima in $\epsilon(\tilde{\Omega}, \tilde{\theta})$"
    )
    ax.set_title("Multimodality vs. lensing oscillation count")
    ax.legend(fontsize=10)
    ax.set_ylim(0, max(n_min_arr) + 1)
    ax.yaxis.set_major_locator(ticker.MaxNLocator(integer=True))
    ax.grid(True, alpha=0.3)

    fig.tight_layout()
    fig.savefig(os.path.join(outdir, "fig5_nminima_vs_nlens.pdf"), dpi=200)
    plt.close(fig)
    print("  Saved fig5_nminima_vs_nlens")


# ---------------------------------------------------------------------------
# Figure 6: Unimodality classification on (mcz, td) grid
# ---------------------------------------------------------------------------
def fig6_unimodal_map(data, outdir, grad_threshold=0.5):
    """Classify each (mcz, td) point as unimodal (low gradient) or multimodal."""
    mcz, td = data["mcz"], data["td"]
    td_ms = td * 1e3
    nlens = compute_nlens_grid(mcz, td, z=data["z"])

    omega_grad = gradient_magnitude_normalized(data["omega_best"])
    theta_grad = gradient_magnitude_normalized(data["theta_best"])
    combined_grad = gaussian_filter(np.sqrt(omega_grad**2 + theta_grad**2), sigma=1.5)

    # Classify: low combined gradient → likely unimodal
    # Use adaptive threshold: median + 1*MAD
    med = np.median(combined_grad)
    mad = np.median(np.abs(combined_grad - med))
    threshold = med + 1.0 * mad
    unimodal = (combined_grad < threshold).astype(float)

    fig, axes = plt.subplots(1, 2, figsize=(14, 5.5))

    # Left: unimodality classification
    im0 = axes[0].pcolormesh(td_ms, mcz, unimodal, cmap="RdYlGn", vmin=0, vmax=1)
    cbar0 = fig.colorbar(im0, ax=axes[0], ticks=[0, 1])
    cbar0.set_ticklabels(["Multimodal", "Unimodal"])
    for n, ls in [(1, ":"), (2, "-"), (3, "-"), (4, ":")]:
        cs = axes[0].contour(
            td_ms,
            mcz,
            nlens,
            levels=[n],
            colors=["blue"],
            linestyles=[ls],
            linewidths=2,
        )
        axes[0].clabel(cs, fmt=rf"$N_\mathrm{{lens}}={n}$", fontsize=9)
    axes[0].set_xlabel(r"$\Delta t_d$ [ms]")
    axes[0].set_ylabel(r"$\mathcal{M}_z$ [$M_\odot$]")
    axes[0].set_title("Unimodal classification (gradient-based)")

    # Right: n_lens with unimodal overlay
    cf = axes[1].contourf(td_ms, mcz, nlens, levels=20, cmap="jet")
    fig.colorbar(cf, ax=axes[1], label=r"$N_\mathrm{lens}$")
    # Overlay unimodal boundary
    axes[1].contour(
        td_ms,
        mcz,
        unimodal,
        levels=[0.5],
        colors=["black"],
        linewidths=2,
        linestyles=["-"],
    )
    axes[1].set_xlabel(r"$\Delta t_d$ [ms]")
    axes[1].set_ylabel(r"$\mathcal{M}_z$ [$M_\odot$]")
    axes[1].set_title(r"$N_\mathrm{lens}$ with unimodal boundary (black)")

    fig.suptitle(
        f"Unimodal region vs $N_\\mathrm{{lens}}$ ($I={data['I']:.1f}$, $z={data['z']}$, edge-on)\n"
        f"Gradient threshold = median + MAD = {threshold:.3f}",
        fontsize=12,
    )
    fig.tight_layout(rect=[0, 0, 1, 0.90])
    fig.savefig(os.path.join(outdir, "fig6_unimodal_map.pdf"), dpi=200)
    plt.close(fig)
    print("  Saved fig6_unimodal_map")

    return unimodal, nlens, threshold


# ---------------------------------------------------------------------------
# Figure 7: Quantitative summary — histogram of n_lens in unimodal vs multimodal
# ---------------------------------------------------------------------------
def fig7_nlens_histogram(unimodal, nlens, data, outdir):
    """Histogram of n_lens values separated by unimodal/multimodal classification."""
    uni_vals = nlens[unimodal > 0.5].ravel()
    multi_vals = nlens[unimodal <= 0.5].ravel()

    fig, ax = plt.subplots(figsize=(8, 5))
    bins = np.linspace(0, nlens.max(), 40)
    ax.hist(
        uni_vals,
        bins=bins,
        alpha=0.6,
        label="Unimodal region",
        color="green",
        edgecolor="black",
        linewidth=0.5,
    )
    ax.hist(
        multi_vals,
        bins=bins,
        alpha=0.6,
        label="Multimodal region",
        color="red",
        edgecolor="black",
        linewidth=0.5,
    )
    ax.axvline(2, color="blue", ls="--", lw=2, label=r"$N_\mathrm{lens}=2$")
    ax.axvline(3, color="blue", ls="-.", lw=2, label=r"$N_\mathrm{lens}=3$")
    ax.set_xlabel(r"$N_\mathrm{lens}$")
    ax.set_ylabel("Count (grid points)")
    ax.set_title(
        f"Distribution of $N_\\mathrm{{lens}}$ by modality classification\n"
        f"($I={data['I']:.1f}$, $z={data['z']}$, edge-on)"
    )
    ax.legend()
    ax.grid(True, alpha=0.3)

    # Print summary statistics
    print(
        f"  Unimodal region: n_lens mean={uni_vals.mean():.2f}, "
        f"median={np.median(uni_vals):.2f}, "
        f"std={uni_vals.std():.2f}, "
        f"[{uni_vals.min():.2f}, {uni_vals.max():.2f}]"
    )
    print(
        f"  Multimodal region: n_lens mean={multi_vals.mean():.2f}, "
        f"median={np.median(multi_vals):.2f}"
    )

    fig.tight_layout()
    fig.savefig(os.path.join(outdir, "fig7_nlens_histogram.pdf"), dpi=200)
    plt.close(fig)
    print("  Saved fig7_nlens_histogram")


# ---------------------------------------------------------------------------
# Figure 8: Gradient magnitude binned by N_lens (quantitative correlation)
# ---------------------------------------------------------------------------
def fig8_gradient_vs_nlens(data, outdir):
    """Binned plot of gradient magnitude vs N_lens for direct hypothesis test."""
    mcz, td = data["mcz"], data["td"]
    nlens = compute_nlens_grid(mcz, td, z=data["z"])

    omega_grad = gradient_magnitude_normalized(data["omega_best"])
    theta_grad = gradient_magnitude_normalized(data["theta_best"])
    combined_grad = np.sqrt(omega_grad**2 + theta_grad**2)

    # Flatten
    nlens_flat = nlens.ravel()
    grad_flat = combined_grad.ravel()

    # Bin by N_lens
    bin_edges = np.arange(0, nlens_flat.max() + 0.5, 0.5)
    bin_centers = 0.5 * (bin_edges[:-1] + bin_edges[1:])
    grad_medians = np.zeros(len(bin_centers))
    grad_q25 = np.zeros(len(bin_centers))
    grad_q75 = np.zeros(len(bin_centers))
    frac_high_grad = np.zeros(len(bin_centers))

    threshold = np.median(grad_flat) + np.median(
        np.abs(grad_flat - np.median(grad_flat))
    )

    for i in range(len(bin_centers)):
        mask = (nlens_flat >= bin_edges[i]) & (nlens_flat < bin_edges[i + 1])
        if mask.sum() > 0:
            vals = grad_flat[mask]
            grad_medians[i] = np.median(vals)
            grad_q25[i] = np.percentile(vals, 25)
            grad_q75[i] = np.percentile(vals, 75)
            frac_high_grad[i] = (vals > threshold).mean()

    fig, axes = plt.subplots(1, 2, figsize=(14, 5.5))

    # Left: Median gradient vs N_lens
    axes[0].fill_between(bin_centers, grad_q25, grad_q75, alpha=0.3, color="steelblue")
    axes[0].plot(
        bin_centers,
        grad_medians,
        "o-",
        color="steelblue",
        ms=4,
        label="Median (IQR shaded)",
    )
    axes[0].axvspan(
        2, 3, alpha=0.15, color="green", label=r"$2 \leq N_\mathrm{lens} \leq 3$"
    )
    axes[0].set_xlabel(r"$N_\mathrm{lens}$")
    axes[0].set_ylabel("Gradient magnitude (index-based)")
    axes[0].set_title("Parameter gradient vs lensing oscillation count")
    axes[0].legend(fontsize=9)
    axes[0].grid(True, alpha=0.3)

    # Right: Fraction of grid points with high gradient
    axes[1].bar(
        bin_centers,
        frac_high_grad,
        width=0.45,
        color="coral",
        edgecolor="black",
        linewidth=0.5,
    )
    axes[1].axvspan(
        2, 3, alpha=0.15, color="green", label=r"$2 \leq N_\mathrm{lens} \leq 3$"
    )
    axes[1].set_xlabel(r"$N_\mathrm{lens}$")
    axes[1].set_ylabel("Fraction of multimodal grid points")
    axes[1].set_title("Multimodality fraction vs $N_\\mathrm{lens}$")
    axes[1].legend(fontsize=9)
    axes[1].grid(True, alpha=0.3, axis="y")

    fig.suptitle(
        f"Quantitative gradient analysis ($I={data['I']:.1f}$, $z={data['z']}$, edge-on)",
        fontsize=13,
    )
    fig.tight_layout(rect=[0, 0, 1, 0.93])
    fig.savefig(os.path.join(outdir, "fig8_gradient_vs_nlens.pdf"), dpi=200)
    plt.close(fig)
    print("  Saved fig8_gradient_vs_nlens")


# ---------------------------------------------------------------------------
# Figure 9: Unimodality & mismatch breakdown by N_lens band
# ---------------------------------------------------------------------------
def fig9_nlens_band_breakdown(data, outdir):
    """Bar charts of unimodality fraction, avg gradient, mismatch, and
    parameter spread broken down by N_lens band."""
    mcz, td = data["mcz"], data["td"]
    nlens = compute_nlens_grid(mcz, td, z=data["z"])

    omega_grad = gradient_magnitude_normalized(data["omega_best"])
    theta_grad = gradient_magnitude_normalized(data["theta_best"])
    combined_grad = gaussian_filter(np.sqrt(omega_grad**2 + theta_grad**2), sigma=1.5)

    med = np.median(combined_grad)
    mad = np.median(np.abs(combined_grad - med))
    threshold = med + 1.0 * mad
    unimodal = combined_grad < threshold

    band_labels = [
        r"$\leq 1$",
        r"$(1,2]$",
        r"$(2,3]$",
        r"$(3,4]$",
        r"$>4$",
    ]
    band_masks = [
        nlens <= 1,
        (nlens > 1) & (nlens <= 2),
        (nlens > 2) & (nlens <= 3),
        (nlens > 3) & (nlens <= 4),
        nlens > 4,
    ]

    pct_uni = []
    avg_grad = []
    avg_eps = []
    std_omega = []
    std_theta = []
    for mask in band_masks:
        n = mask.sum()
        pct_uni.append(100 * (mask & unimodal).sum() / n if n > 0 else 0)
        avg_grad.append(combined_grad[mask].mean() if n > 0 else 0)
        avg_eps.append(data["epsilon_min"][mask].mean() if n > 0 else 0)
        std_omega.append(data["omega_best"][mask].std() if n > 0 else 0)
        std_theta.append(data["theta_best"][mask].std() if n > 0 else 0)

    x = np.arange(len(band_labels))
    fig, axes = plt.subplots(2, 2, figsize=(13, 9))

    # (a) % Unimodal
    bars_a = axes[0, 0].bar(x, pct_uni, color="seagreen", edgecolor="black", lw=0.5)
    axes[0, 0].set_ylabel("% Unimodal")
    axes[0, 0].set_title("(a) Unimodality fraction by $N_\\mathrm{lens}$ band")
    axes[0, 0].set_ylim(0, 105)
    axes[0, 0].axhline(50, color="gray", ls=":", alpha=0.5)
    for i, v in enumerate(pct_uni):
        axes[0, 0].text(i, v + 1.5, f"{v:.0f}%", ha="center", fontsize=9)

    # (b) Mean gradient
    bars_b = axes[0, 1].bar(x, avg_grad, color="steelblue", edgecolor="black", lw=0.5)
    axes[0, 1].set_ylabel("Mean combined gradient")
    axes[0, 1].set_title("(b) Average parameter gradient")
    axes[0, 1].axhline(
        threshold, color="red", ls="--", lw=1.5, label=f"Threshold = {threshold:.3f}"
    )
    axes[0, 1].legend(fontsize=9)

    # (c) Mean mismatch
    bars_c = axes[1, 0].bar(x, avg_eps, color="coral", edgecolor="black", lw=0.5)
    axes[1, 0].set_ylabel(r"Mean $\epsilon_\mathrm{min}$")
    axes[1, 0].set_title(r"(c) Average minimum mismatch $\epsilon_\mathrm{min}$")
    axes[1, 0].set_yscale("log")
    # Replace zero with tiny value for log scale
    for bar, val in zip(bars_c, avg_eps):
        if val > 0:
            axes[1, 0].text(
                bar.get_x() + bar.get_width() / 2,
                val * 1.3,
                f"{val:.1e}",
                ha="center",
                fontsize=8,
                rotation=45,
            )

    # (d) Parameter spread
    w = 0.35
    axes[1, 1].bar(
        x - w / 2,
        std_omega,
        w,
        color="mediumpurple",
        edgecolor="black",
        lw=0.5,
        label=r"$\sigma(\tilde{\Omega})$",
    )
    axes[1, 1].bar(
        x + w / 2,
        std_theta,
        w,
        color="goldenrod",
        edgecolor="black",
        lw=0.5,
        label=r"$\sigma(\tilde{\theta})$",
    )
    axes[1, 1].set_ylabel("Std. deviation of best-fit parameter")
    axes[1, 1].set_title("(d) Best-fit parameter spread")
    axes[1, 1].legend(fontsize=9)

    for ax in axes.ravel():
        ax.set_xticks(x)
        ax.set_xticklabels(band_labels)
        ax.set_xlabel(r"$N_\mathrm{lens}$ band")
        ax.grid(True, alpha=0.3, axis="y")

    fig.suptitle(
        f"$N_\\mathrm{{lens}}$ band breakdown ($I={data['I']:.1f}$, $z={data['z']}$, edge-on)",
        fontsize=13,
    )
    fig.tight_layout(rect=[0, 0, 1, 0.94])
    fig.savefig(os.path.join(outdir, "fig9_nlens_band_breakdown.pdf"), dpi=200)
    plt.close(fig)
    print("  Saved fig9_nlens_band_breakdown")


# ---------------------------------------------------------------------------
# Figure 10: Spatial map of N_lens regimes on (mcz, td) plane
# ---------------------------------------------------------------------------
def fig10_nlens_regime_map(data, outdir):
    """Color-code the (mcz, td) plane by N_lens regime, overlaid with
    unimodal/multimodal boundary and mismatch contours."""
    mcz, td = data["mcz"], data["td"]
    td_ms = td * 1e3
    nlens = compute_nlens_grid(mcz, td, z=data["z"])

    omega_grad = gradient_magnitude_normalized(data["omega_best"])
    theta_grad = gradient_magnitude_normalized(data["theta_best"])
    combined_grad = gaussian_filter(np.sqrt(omega_grad**2 + theta_grad**2), sigma=1.5)
    med = np.median(combined_grad)
    mad = np.median(np.abs(combined_grad - med))
    threshold = med + 1.0 * mad
    unimodal = combined_grad < threshold

    # Assign regime labels: 0=N<=1, 1=(1,2], 2=(2,3], 3=(3,4], 4=>4
    regime = np.zeros_like(nlens, dtype=int)
    regime[nlens <= 1] = 0
    regime[(nlens > 1) & (nlens <= 2)] = 1
    regime[(nlens > 2) & (nlens <= 3)] = 2
    regime[(nlens > 3) & (nlens <= 4)] = 3
    regime[nlens > 4] = 4

    from matplotlib.colors import ListedColormap, BoundaryNorm

    regime_colors = ["#d73027", "#fc8d59", "#91cf60", "#1a9850", "#4575b4"]
    regime_labels = [
        r"$N_\mathrm{lens}\leq 1$",
        r"$1<N_\mathrm{lens}\leq 2$",
        r"$2<N_\mathrm{lens}\leq 3$",
        r"$3<N_\mathrm{lens}\leq 4$",
        r"$N_\mathrm{lens}>4$",
    ]
    cmap_regime = ListedColormap(regime_colors)
    norm_regime = BoundaryNorm([-0.5, 0.5, 1.5, 2.5, 3.5, 4.5], cmap_regime.N)

    fig, axes = plt.subplots(1, 2, figsize=(15, 6))

    # Left: regime map with unimodal boundary
    im = axes[0].pcolormesh(td_ms, mcz, regime, cmap=cmap_regime, norm=norm_regime)
    cbar = fig.colorbar(im, ax=axes[0], ticks=[0, 1, 2, 3, 4])
    cbar.set_ticklabels(regime_labels)
    cbar.ax.tick_params(labelsize=8)

    # Unimodal boundary in black
    axes[0].contour(
        td_ms,
        mcz,
        unimodal.astype(float),
        levels=[0.5],
        colors=["black"],
        linewidths=2.5,
        linestyles=["-"],
    )
    axes[0].set_xlabel(r"$\Delta t_d$ [ms]")
    axes[0].set_ylabel(r"$\mathcal{M}_z$ [$M_\odot$]")
    axes[0].set_title(r"$N_\mathrm{lens}$ regime + unimodal boundary (black)")

    # Right: mismatch with N_lens contours and unimodal boundary
    cf = axes[1].contourf(td_ms, mcz, data["epsilon_min"], levels=50, cmap="jet")
    fig.colorbar(cf, ax=axes[1], label=r"$\epsilon_\mathrm{min}$")
    for n, ls, c in [
        (1, ":", "white"),
        (2, "-", "white"),
        (3, "-", "white"),
        (4, ":", "white"),
    ]:
        cs = axes[1].contour(
            td_ms, mcz, nlens, levels=[n], colors=[c], linestyles=[ls], linewidths=1.5
        )
        axes[1].clabel(cs, fmt=rf"$N={n}$", fontsize=8)
    # Unimodal boundary
    axes[1].contour(
        td_ms,
        mcz,
        unimodal.astype(float),
        levels=[0.5],
        colors=["black"],
        linewidths=2.5,
        linestyles=["--"],
    )
    axes[1].set_xlabel(r"$\Delta t_d$ [ms]")
    axes[1].set_ylabel(r"$\mathcal{M}_z$ [$M_\odot$]")
    axes[1].set_title(
        r"$\epsilon_\mathrm{min}$ + $N_\mathrm{lens}$ contours + unimodal boundary"
    )

    fig.suptitle(
        f"$N_\\mathrm{{lens}}$ regime map ($I={data['I']:.1f}$, $z={data['z']}$, edge-on)",
        fontsize=13,
    )
    fig.tight_layout(rect=[0, 0, 1, 0.93])
    fig.savefig(os.path.join(outdir, "fig10_nlens_regime_map.pdf"), dpi=200)
    plt.close(fig)
    print("  Saved fig10_nlens_regime_map")


# ===========================================================================
# Main
# ===========================================================================
def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--best_match",
        default="data/mismatch_I0p5_z1e-08_mcz10-90_td20-70_Taman_edgeon/"
        "best_match/best_match_I0p5_z1_mcz5-45x81_td20-70x51_"
        "omega0-6x61_theta0-15x151_gamma0-2pix51_Taman_edgeon.h5",
        help="Path to aggregated best-match HDF5 file",
    )
    parser.add_argument(
        "--indiv_dir",
        default="data/contour_omega_theta",
        help="Directory with v3_indiv_contour_*.pkl files",
    )
    parser.add_argument(
        "--outdir",
        default="figures/modality_nlens",
        help="Output directory for figures",
    )
    parser.add_argument(
        "--z_filter",
        default=None,
        type=float,
        help="Only use individual contours at this redshift",
    )
    args = parser.parse_args()

    os.makedirs(args.outdir, exist_ok=True)
    apply_physics_paper_style()

    # Load data
    print("Loading best-match data...")
    data = load_best_match(args.best_match)
    print(
        f"  mcz: {data['mcz'].min():.0f} - {data['mcz'].max():.0f} Msun "
        f"({len(data['mcz'])} pts)"
    )
    print(
        f"  td: {data['td'].min()*1e3:.0f} - {data['td'].max()*1e3:.0f} ms "
        f"({len(data['td'])} pts)"
    )
    print(f"  I={data['I']}, z={data['z']}")

    print("\nLoading individual contours...")
    contours = load_indiv_contours(args.indiv_dir, z_filter=args.z_filter)
    print(
        f"  Found {len(contours)} contours"
        + (f" at z={args.z_filter}" if args.z_filter else " (all z)")
    )

    # Generate figures
    print("\nGenerating figures...")
    fig1_mismatch_nlens(data, args.outdir)
    fig2_gradient_multimodality(data, args.outdir)
    fig3_example_surfaces(contours, args.outdir)
    fig4_bestfit_vs_td(data, args.outdir)
    fig5_nminima_vs_nlens(contours, args.outdir)
    unimodal, nlens, threshold = fig6_unimodal_map(data, args.outdir)
    fig7_nlens_histogram(unimodal, nlens, data, args.outdir)
    fig8_gradient_vs_nlens(data, args.outdir)
    fig9_nlens_band_breakdown(data, args.outdir)
    fig10_nlens_regime_map(data, args.outdir)

    print(f"\nAll figures saved to {args.outdir}/")


if __name__ == "__main__":
    main()
