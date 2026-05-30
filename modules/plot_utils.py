# ============================================================================
# Imports
# ============================================================================


# import py scripts
from modules.default_params import SOLMASS2SEC
from modules.geometry import calculate_cosJN
from modules.waveform import get_I_from_y, get_td_from_MLz

# import libraries
import os
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
from matplotlib import colors
from matplotlib.lines import Line2D
from fractions import Fraction
import numpy as np

# ============================================================================
# Style And Convenience
# ============================================================================


def set_default_plot_style():
    plt.rcParams["figure.dpi"] = 150
    plt.rcParams["axes.titlesize"] = 24
    plt.rcParams["axes.labelsize"] = 24
    plt.rcParams["xtick.labelsize"] = 18
    plt.rcParams["ytick.labelsize"] = 18
    plt.rcParams["legend.fontsize"] = 20
    plt.rcParams["figure.titlesize"] = 24


def apply_physics_paper_style(
    base_font: int = 12,
    label_font: int = 14,
    tick_font: int = 11,
    legend_font: int = 11,
) -> None:
    """Apply a consistent style suited for physics-paper figures."""
    plt.rcParams.update(
        {
            "font.family": "serif",
            "font.size": base_font,
            "axes.labelsize": label_font,
            "xtick.labelsize": tick_font,
            "ytick.labelsize": tick_font,
            "legend.fontsize": legend_font,
            "mathtext.fontset": "cm",
            "mathtext.default": "it",
            "axes.unicode_minus": False,
            "axes.formatter.use_mathtext": True,
            "axes.linewidth": 1.0,
            "xtick.direction": "in",
            "ytick.direction": "in",
            "xtick.top": True,
            "ytick.right": True,
        }
    )


COLORBAR_PAD = 0.015
COLORBAR_WIDTH = 0.02


# ============================================================================
# Label Constants
# ============================================================================
# Python analogues of the \newcommand shortcuts in paper_lens_prec/preamble.tex.
# Use these in set_xlabel/set_ylabel/set_label/legend label= calls to keep
# typography consistent across scripts (variables italic, identifiers upright
# roman, \epsilon throughout, no \mathit on h).

# -- Axis labels --------------------------------------------------------------
LBL_TD = r"$\Delta t_{\mathrm{d}}\,[\mathrm{ms}]$"
LBL_MCZ = r"$\mathcal{M}_{\mathrm{s}}\,[\mathrm{M}_\odot]$"
LBL_I = r"$I$"
LBL_OMEGA = r"$\tilde{\Omega}$"
LBL_THETA = r"$\tilde{\theta}$"
LBL_F = r"$f\,[\mathrm{Hz}]$"

# -- Waveform symbols ---------------------------------------------------------
LBL_H_L = r"$\tilde{h}_{\mathrm{L}}$"
LBL_H_UL = r"$\tilde{h}_{\mathrm{UL}}$"
LBL_H_NP = r"$\tilde{h}_{\mathrm{NP}}$"
LBL_H_RP = r"$\tilde{h}_{\mathrm{RP}}$"
LBL_H_P = r"$\tilde{h}_{\mathrm{P}}$"
LBL_H_S = r"$\tilde{h}_{\mathrm{s}}$"  # generic source waveform
LBL_H_T = r"$\tilde{h}_{\mathrm{t}}$"  # generic template waveform

# -- Mismatch labels ----------------------------------------------------------
LBL_EPS_LP = r"$\epsilon(\tilde{h}_{\mathrm{L}}, \tilde{h}_{\mathrm{P}})$"
LBL_EPS_LNP = r"$\epsilon(\tilde{h}_{\mathrm{L}}, \tilde{h}_{\mathrm{NP}})$"
LBL_EPS_ST = r"$\epsilon\left(\tilde{h}_{\mathrm{s}},\,\tilde{h}_{\mathrm{t}}\right)$"
LBL_MIN_EPS_LP = (
    r"$\min_{\tilde{\Omega},\,\tilde{\theta},\,\gamma_{\mathrm{P}}}\,"
    r"\epsilon(\tilde{h}_{\mathrm{L}}, \tilde{h}_{\mathrm{P}})$"
)
LBL_MIN_EPS_ST = (
    r"$\min_{\tilde{\Omega},\,\tilde{\theta},\,\gamma_{\mathrm{P}}}\,"
    r"\epsilon\left(\tilde{h}_{\mathrm{s}},\,\tilde{h}_{\mathrm{t}}\right)$"
)


def add_colorbar_axes(fig, target_axes, *, pad=COLORBAR_PAD, width=COLORBAR_WIDTH):
    """Add and return a colorbar axes (cax) aligned to the right of target_axes.

    The colorbar will span vertically from the bottom of the lowest axis to the
    top of the highest axis in target_axes, and will be placed to the right
    of the rightmost axis.
    """
    fig.canvas.draw()
    axes_list = list(np.asarray(target_axes, dtype=object).flat)

    positions = [ax.get_position() for ax in axes_list]
    x1 = max(pos.x1 for pos in positions)
    y0 = min(pos.y0 for pos in positions)
    y1 = max(pos.y1 for pos in positions)

    # Freeze the layout so it cannot re-run after the new axes is added,
    # which would shift the subplots and misalign the colorbar.
    fig.set_layout_engine("none")
    return fig.add_axes([x1 + pad, y0, width, y1 - y0])


def save_figure(fig, path, *, dpi=400):
    """Save *fig* to *path* with tight bbox, close it, and print the path."""
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    fig.savefig(path, dpi=dpi, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved figure: {path}")


def format_colorbar_ticks(
    cbar,
    vmin,
    vmax,
    *,
    n_ticks=15,
    decimals=2,
    use_locator=True,
    nbins=None,
    steps=(1, 1.5, 2, 2.5, 3, 5, 10),
    prune=None,
):
    """Set colorbar ticks to rounded values.

    Default mode uses a ``MaxNLocator`` to choose nice tick positions
    (pass *nbins* and/or *steps* to tune it).  Set *use_locator=False*
    to fall back to *n_ticks* linearly spaced between *vmin* and *vmax*.
    Pass *prune* (e.g. ``'both'``) to remove ticks at the colorbar extremes.
    """
    if use_locator:
        cbar.locator = mticker.MaxNLocator(
            nbins=max(2, nbins or n_ticks),
            steps=list(steps),
            prune=prune,
        )
    else:
        cbar.set_ticks(np.linspace(vmin, vmax, n_ticks))
    cbar.ax.yaxis.set_major_formatter(mticker.FormatStrFormatter(f"%.{decimals}f"))
    cbar.update_ticks()


def configure_I_axis(ax) -> None:
    """Apply the standard flux ratio (I) axis tick style.

    Sets major ticks every 0.2 (labeled) and minor ticks every 0.1 (unlabeled).
    Since axes are typically shared, calling this on one axis is sufficient.
    """
    ax.yaxis.set_major_locator(mticker.MultipleLocator(0.2))
    ax.yaxis.set_minor_locator(mticker.MultipleLocator(0.1))


def set_square_axes(*axes):
    """Force square box aspect on each axis (for contour panels)."""
    for ax in axes:
        ax.set_box_aspect(1)


def add_overlay_legend(
    fig,
    handles,
    *,
    ncol=None,
    alpha=0.35,
    fontsize=11,
    loc="lower center",
    bbox_to_anchor=(0.5, 0.012),
):
    """Add a translucent overlay legend at the figure bottom."""
    if not handles:
        return None
    legend = fig.legend(
        handles=handles,
        loc=loc,
        bbox_to_anchor=bbox_to_anchor,
        ncol=ncol if ncol is not None else len(handles),
        frameon=True,
        fontsize=fontsize,
    )
    legend.get_frame().set_alpha(alpha)
    return legend


def angle_to_pi_string(
    angle: float, denom_thres: int = 50, wrap_math: bool = True
) -> str:
    """
    Converts an angle in radians to a string in pi format.

    Args:
        angle (float): The angle in radians.
        denom_thres (int): The threshold for the denominator of the fraction. Default is 50.
        wrap_math (bool): If True, wrap non-zero pi expressions in `$...$`.

    Returns:
        str: The angle in pi format.
    """

    def _wrap(expr: str) -> str:
        return rf"${expr}$" if wrap_math else expr

    if np.isclose(angle, 0.0, atol=1e-10):
        return "0"
    if np.isclose(angle, np.pi, atol=1e-10):
        return _wrap(r"\pi")
    if np.isclose(angle, -np.pi, atol=1e-10):
        return _wrap(r"-\pi")

    fraction = Fraction(angle / np.pi).limit_denominator(1000)

    if fraction.denominator > denom_thres:
        return _wrap(rf"{angle / np.pi:.3f}\pi")

    numerator = fraction.numerator
    denominator = fraction.denominator

    if denominator == 1:
        if numerator == 1:
            return _wrap(r"\pi")
        if numerator == -1:
            return _wrap(r"-\pi")
        return _wrap(rf"{numerator}\pi")

    if numerator == 1:
        return _wrap(rf"\pi/{denominator}")
    if numerator == -1:
        return _wrap(rf"-\pi/{denominator}")
    return _wrap(rf"{numerator}\pi/{denominator}")


# ============================================================================
# Contour Plots
# ============================================================================


def _draw_mismatch_contour(X, Y, Z, n_levels):
    """Draw contourf with standard mismatch axes and colorbar."""
    plt.contourf(X, Y, Z, levels=n_levels, cmap="jet")
    plt.xlabel(r"$\~\Omega$", fontsize=14)
    plt.ylabel(r"$\~\theta$", fontsize=14)
    plt.colorbar(cmap="jet", norm=colors.Normalize(vmin=0, vmax=1)).set_label(
        label=r"$\epsilon(\~h_{\mathrm{L}}, \~h_{\mathrm{P}})$", size=14
    )


def _mark_contour_minima(X, Y, Z, n_minima):
    """Scatter white circles at the n lowest mismatch values."""
    if n_minima > 0:
        ep_min_indices = np.unravel_index(np.argsort(Z, axis=None)[:n_minima], Z.shape)
        plt.scatter(X[ep_min_indices], Y[ep_min_indices], color="white", marker="o")
        print(
            f"minima: {Z[ep_min_indices]}, omega: {X[ep_min_indices]}, theta: {Y[ep_min_indices]}"
        )


def _add_contour_title(src_params, td, I):
    """Add standard physics annotation title to the current contour plot."""
    plt.title(
        r"$\theta_{\mathrm{S}}$ = {}, $\phi_{\mathrm{S}}$ = {}, $\theta_{\mathrm{J}}$ = {}, $\phi_{\mathrm{J}}$ = {}, {} = {:.3g} {}, $\Delta t_{\mathrm{d}}$ = {:.3g} $\mathrm{{ms}}$, $I$ = {:.3g}".format(
            angle_to_pi_string(src_params["theta_S"]),
            angle_to_pi_string(src_params["phi_S"]),
            angle_to_pi_string(src_params["theta_J"]),
            angle_to_pi_string(src_params["phi_J"]),
            r"$\mathcal{M}_{\mathrm{s}}$",
            src_params["mcz"] / SOLMASS2SEC,
            r"$M_{\odot}$",
            td * 1e3,
            I,
        ),
        fontsize=12,
        y=1.021,
    )


def plot_indiv_contour(
    X: np.ndarray,
    Y: np.ndarray,
    Z: np.ndarray,
    src_params: dict,
    n_levels=100,
    n_minima=1,
    title=True,
    suptitle=True,
):
    """
    Plot a mismatch contour over `omega_tilde` and `theta_tilde` and optionally mark the lowest minima.

    Parameters
    ----------
    X, Y : np.ndarray
        2D grids (e.g., from `np.meshgrid`) for `omega_tilde` and `theta_tilde`.
    Z : np.ndarray
        2D matrix of mismatch values `epsilon(h_L, h_P)` in [0, 1]. Must match the shape of `X` and `Y`.
    src_params : dict
        Source parameters used for title annotations (e.g., angles, `mcz`).
    n_levels : int, optional
        Number of contour levels. Default is 100.
    n_minima : int, optional
        Number of lowest values in `Z` to highlight with markers. Default is 1.
    title : bool, optional
        If True, annotate the figure title with source and lensing info. Default is True.
    suptitle : bool, optional
        If True, add a descriptive suptitle. Default is True.

    Returns
    -------
    None

    Notes
    -----
    - Minima are shown as white circles and printed to stdout.
    - Uses `LensingGeo(src_params).td()` and `.I()` to annotate time delay and interference strength in the title when `title=True`.
    """

    _draw_mismatch_contour(X, Y, Z, n_levels)
    _mark_contour_minima(X, Y, Z, n_minima)

    if suptitle:
        plt.suptitle(
            "Mismatch Between RP Templates and a Lensed Source",
            fontsize=16,
            y=1.0215,
            x=0.435,
        )

    if title:
        I = get_I_from_y(src_params["y"])
        td = get_td_from_MLz(src_params["MLz"] / SOLMASS2SEC, src_params["y"])
        _add_contour_title(src_params, td, I)


def plot_indiv_contour_from_dict(
    d: dict, k: float, n_levels=100, n_minima=1, title=True, suptitle=True
):
    """
    Plot a mismatch contour from a precomputed dictionary entry at key `k`.

    Parameters
    ----------
    d : dict
        Dictionary with a 'contour' sub-dict for each key. Each `d[k]['contour']` must include 'omega_matrix', 'theta_matrix', 'epsilon_matrix', and 'source_params'.
        At the top level, one of 'td' or 'I' must be present to disambiguate `k` (if 'td' is present then `k` is interpreted as `I`; if 'I' is present then `k` is interpreted as `td`).
    k : float
        Value identifying the slice to plot (interpreted as `I` or `td`; see above).
    n_levels : int, optional
        Number of contour levels. Default is 100.
    n_minima : int, optional
        Number of lowest values in the mismatch matrix to highlight. Default is 1.
    title : bool, optional
        If True, include a title with angles, `mcz`, `td`, and `I`. Default is True.
    suptitle : bool, optional
        If True, add a descriptive suptitle. Default is True.

    Returns
    -------
    None

    Notes
    -----
    - Minima are marked with white circles and printed to stdout.
    - Leaves the current Figure/Axes active.
    """

    X = d[k]["contour"]["omega_matrix"]
    Y = d[k]["contour"]["theta_matrix"]
    Z = d[k]["contour"]["epsilon_matrix"]
    src_params = d[k]["contour"]["source_params"]
    if d.get("td") is not None:
        td = d["td"]
        I = k
    elif d.get("I") is not None:
        I = d["I"]
        td = k

    _draw_mismatch_contour(X, Y, Z, n_levels)
    _mark_contour_minima(X, Y, Z, n_minima)

    if suptitle:
        plt.suptitle(
            "Mismatch Between RP Templates and a Lensed Source",
            fontsize=16,
            y=1.0215,
            x=0.435,
        )

    if title:
        _add_contour_title(src_params, td, I)


def plot_special_coords(fix, fixed_phi, fixed_theta):
    """
    Plots the face-on and edge-on coordinates for a given fixed_phi and fixed_theta.

    Args:
        fix (str): Determines which variable is fixed ('S' for sky location, 'J' for binary orientation).
        fixed_phi (float): The fixed phi (in radians).
        fixed_theta (float): The fixed theta (in radians).

    Returns:
        None
    """

    n_pts = 151
    phi_arr = np.linspace(0, 2 * np.pi, n_pts)
    theta_arr = np.linspace(0, np.pi, n_pts)
    X, Y = np.meshgrid(phi_arr, theta_arr)

    if fix == "S":
        Z = calculate_cosJN(fixed_phi, fixed_theta, X, Y)
    else:  # fix == 'J'
        Z = calculate_cosJN(X, Y, fixed_phi, fixed_theta)

    # plot Z = 0 (edge-on)
    plt.contour(
        X, np.cos(Y), Z, levels=[0], linestyles="-", colors="black", labels="edge-on"
    )

    # plot |Z| = 1 (face-on) within error
    cond = np.isclose(np.abs(Z), 1, rtol=0, atol=1e-4)
    plt.scatter(X[cond], np.cos(Y[cond]), marker="x", color="white", label="face-on")

    # create custom legend handles
    legend = [
        Line2D([0], [0], c="black", lw=1, ls="-", label="edge-on"),
        Line2D([0], [0], c="white", marker="x", ms=5, label="face-on"),
    ]

    # plt.legend(handles=legend)


def create_cosJN_contour(fix, fixed_phi, fixed_theta):
    """
    Plots contours of the inclination angle between the J and N vectors.

    Args:
        fix (str): Determines which variable is fixed ('S' for sky location, 'J' for binary orientation).
        fixed_phi (float): The fixed phi (in radians).
        fixed_theta (float): The fixed theta (in radians).

    Returns:
        None
    """

    n_pts = 151
    phi_arr = np.linspace(0, 2 * np.pi, n_pts)
    theta_arr = np.linspace(0, np.pi, n_pts)
    X, Y = np.meshgrid(phi_arr, theta_arr)

    if fix == "S":
        Z = calculate_cosJN(fixed_phi, fixed_theta, X, Y)
    else:  # fix == 'J'
        Z = calculate_cosJN(X, Y, fixed_phi, fixed_theta)

    plt.contourf(X, np.cos(Y), Z, levels=60, cmap="jet")
    plt.colorbar(label=r"$\cos \iota_{JN}$")
    plt.xticks(
        np.arange(0, 2 * np.pi + np.pi / 4, np.pi / 4),
        [
            r"$0$",
            r"$\frac{\pi}{4}$",
            r"$\frac{\pi}{2}$",
            r"$\frac{3\pi}{4}$",
            r"$\pi$",
            r"$\frac{5\pi}{4}$",
            r"$\frac{3\pi}{2}$",
            r"$\frac{7\pi}{4}$",
            r"$2\pi$",
        ],
    )

    if fix == "S":
        plt.ylabel(r"$\cos \theta_J$")
        plt.xlabel(r"$\phi_J$")
        plt.title(
            r"$\phi_S$ = {:.3g}, $\theta_S$ = {:.3g}".format(fixed_phi, fixed_theta)
        )
    else:  # fix == 'J'
        plt.ylabel(r"$\cos \theta_S$")
        plt.xlabel(r"$\phi_S$")
        plt.title(
            r"$\phi_J$ = {:.3g}, $\theta_J$ = {:.3g}".format(fixed_phi, fixed_theta)
        )
