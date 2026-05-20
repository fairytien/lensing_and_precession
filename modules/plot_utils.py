# ============================================================================
# Imports
# ============================================================================


# import py scripts
from modules.default_params import SOLMASS2SEC
from modules.geometry import calculate_cosJN
from modules.match_utils import find_optimized_coalescence_params
from modules.waveform import set_to_params, get_gw, get_I_from_y, get_td_from_MLz

# import libraries
import os

import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
from matplotlib import colors
from matplotlib.figure import Figure
from matplotlib.lines import Line2D
from fractions import Fraction
import numpy as np

AxesArray = np.ndarray  # Array of matplotlib Axes returned by plt.subplots.

plt.rcParams["figure.dpi"] = 150

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

def save_figure(fig, path, *, dpi=300):
    """Save *fig* to *path* with tight bbox, close it, and print the path."""
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    fig.savefig(path, dpi=dpi, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved figure: {path}")


def format_colorbar_ticks(cbar, vmin, vmax, *, n_ticks=15, decimals=2,
                          use_locator=True, nbins=None,
                          steps=(1, 1.5, 2, 2.5, 3, 5, 10)):
    """Set colorbar ticks to rounded values.

    Default mode uses a ``MaxNLocator`` to choose nice tick positions
    (pass *nbins* and/or *steps* to tune it).  Set *use_locator=False*
    to fall back to *n_ticks* linearly spaced between *vmin* and *vmax*.
    """
    if use_locator:
        cbar.locator = mticker.MaxNLocator(
            nbins=max(2, nbins or n_ticks), steps=list(steps),
        )
    else:
        cbar.set_ticks(np.linspace(vmin, vmax, n_ticks))
    cbar.ax.yaxis.set_major_formatter(
        mticker.FormatStrFormatter(f"%.{decimals}f")
    )
    cbar.update_ticks()


def set_square_axes(*axes):
    """Force square box aspect on each axis (for contour panels)."""
    for ax in axes:
        ax.set_box_aspect(1)


def add_overlay_legend(fig, handles, *, ncol=None, alpha=0.35,
                       fontsize=11, loc="lower center",
                       bbox_to_anchor=(0.5, 0.012)):
    """Add a translucent overlay legend at the figure bottom."""
    if not handles:
        return None
    legend = fig.legend(
        handles=handles, loc=loc, bbox_to_anchor=bbox_to_anchor,
        ncol=ncol if ncol is not None else len(handles),
        frameon=True, fontsize=fontsize,
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
# Waveform Plots
# ============================================================================


def plot_standalone_waveform_comparison(
    t_params: dict,  # template parameters
    s_params: dict,  # source parameters
    phase_shift: float = 0,
    return_fig: bool = False,
    **kwargs,
) -> tuple[Figure, AxesArray] | None:
    """
    Plot a standalone, publication-ready comparison between a lensed source and its best-matching RP template.

    This function allocates a new matplotlib Figure with two panels:
    - left: |h~(f)| (strain amplitude), log-scale
    - right: unwrapped phase difference Phi_s(f) - Phi_t(f) + `phase_shift`

    Before plotting, it optimizes the RP template coalescence parameters against the source by calling `find_optimized_coalescence_params()`, which:
    1) if `optimize_gammaP=True`, scans the initial precession phase gamma_P to minimize mismatch,
    2) adjusts the template time and phase of coalescence (t_c, phi_c).

    Parameters
    ----------
    t_params : dict
        Template waveform parameters.
    s_params : dict
        Source waveform parameters.
    phase_shift : float, optional
        Constant phase offset added when plotting the phase-difference curve.
        Useful to visually align curves; does not affect optimized mismatch.
        Default is 0.
    **kwargs
        Additional keyword arguments passed to `find_optimized_coalescence_params()` and `get_gw()` functions.

    Returns
    -------
    None
        The function creates a Figure and Axes and leaves them active; it does not return them explicitly.

    Raises
    ------
    ValueError
        If the template dict is not an RP parameter set (e.g., missing gamma_P), or if the source dict is incompatible with the provided `lens_Class`.

    Notes
    -----
    - The suptitle prints key parameters: mcz, td, I, optimized (omega_tilde, theta_tilde), gamma_P, and epsilon (mismatch).
    - For overlaying multiple templates on existing Axes, use `plot_compared_waveform_on_axes()`.
    - This function prints debugging info (index, phase) that should be ~0 when `verify_optimization=True`.

    Examples
    --------
    >>> plot_standalone_waveform_comparison(t_params, s_params, phase_shift=0.5, use_opt_match=True, optimize_gammaP=True, verify_optimization=True)
    """

    fig, axes = plt.subplots(nrows=1, ncols=2, figsize=(18, 6))
    fig.subplots_adjust(wspace=0.25)

    t_params_copy, s_params_copy = set_to_params(t_params, s_params)

    # get optimized coalescence parameters (new return structure)
    opt_coal_results = find_optimized_coalescence_params(
        t_params_copy, s_params_copy, **kwargs
    )
    t_params_copy = opt_coal_results["opt_t_params"]

    # Filter kwargs for get_gw()
    get_gw_kwargs = {
        key: value
        for key, value in kwargs.items()
        if key in get_gw.__code__.co_varnames
    }

    # source waveform
    s_gw = get_gw(s_params_copy, **get_gw_kwargs)
    s_strain = np.abs(s_gw["strain"])
    s_phase = s_gw["phase"]
    s_farray = s_gw["f_array"]
    I = get_I_from_y(s_params_copy["y"])
    td = get_td_from_MLz(s_params_copy["MLz"] / SOLMASS2SEC, s_params_copy["y"])
    axes[0].plot(s_farray, s_strain, label="source", c="k", ls="-")

    # template waveform
    t_gw = get_gw(t_params_copy, **get_gw_kwargs)
    t_strain = np.abs(t_gw["strain"])
    t_phase = t_gw["phase"]
    t_farray = t_gw["f_array"]
    axes[0].plot(t_farray, t_strain, label="template", c="k", ls="--")

    # phase difference
    phase_diff = s_phase - t_phase
    phase_diff = np.unwrap(phase_diff + phase_shift)
    axes[1].plot(s_farray, phase_diff, c="k", ls="-")

    # customize strain plot
    axes[0].set_yscale("log")
    axes[0].legend(fontsize=20)
    axes[0].set_xlabel(r"$f\,[\mathrm{Hz}]$", fontsize=24)
    axes[0].set_ylabel(r"$|\~{h}|$", fontsize=24)
    axes[0].tick_params(axis="both", which="major", labelsize=18)
    axes[0].grid()
    axes[0].set_title("Strain", fontsize=24)

    # customize phase difference plot
    axes[1].set_xlabel(r"$f\,[\mathrm{Hz}]$", fontsize=24)
    axes[1].set_ylabel(
        r"$\Phi_{\mathrm{s}} - \Phi_{\mathrm{t}}\,[\mathrm{rad}]$",
        fontsize=24,
    )
    axes[1].tick_params(axis="both", which="major", labelsize=18)
    axes[1].grid()
    axes[1].set_title("Phase Difference", fontsize=24)

    # customize suptitle
    fig.suptitle(
        r"{} = {:.3g} {}, $\Delta t_{\mathrm{d}}$ = {:.3g} $\mathrm{{ms}}$, $I$ = {:.3g}, $\~\Omega$ = {:.3g}, $\~\theta$ = {:.3g}, $\gamma_{\mathrm{P}}$ = {:.3g}, $\epsilon = {:.3g}$".format(
            r"$\mathcal{M}_{\mathrm{s}}$",
            s_params_copy["mcz"] / SOLMASS2SEC,
            r"$M_{\odot}$",
            td * 1e3,
            I,
            t_params_copy["omega_tilde"],
            t_params_copy["theta_tilde"],
            t_params_copy["gamma_P"],
            opt_coal_results["ep_min"],
        ),
        fontsize=24,
        y=1.02,
    )

    if return_fig:
        return fig, axes


def plot_template_on_axes(
    t_params: dict,  # template parameters
    s_params: dict,  # source parameters
    axes: AxesArray,
    phase_shift: float = 0,
    label: str = "template",
    c: str = "k",
    ls: str = "--",
    **kwargs,
) -> None:
    """
    This function plots a template waveform and its phase difference with respect to a source waveform on pre-existing axes.
    It optimizes coalescence parameters to minimize mismatch and plots the template strain on `axes[0]` and phase difference on `axes[1]`.
    Designed to be called multiple times to build up complex plots with multiple waveform comparisons.

    Parameters
    ----------
    t_params : dict
        Template waveform parameters.
    s_params : dict
        Source waveform parameters (used for phase difference calculation).
    axes : matplotlib.axes._axes.Axes
        Array of matplotlib axes. `axes[0]` for strain, `axes[1]` for phase difference.
    phase_shift : float, optional
        Phase shift to apply to the phase difference calculation. Default is 0.
    label : str, optional
        Label for the plotted template waveform. Default is "template".
    c : str, optional
        Color of the plotted lines. Default is "k" (black).
    ls : str, optional
        Line style for the plotted lines. Default is "--" (dashed).
    **kwargs
        Additional keyword arguments passed to `find_optimized_coalescence_params()` and `get_gw()` functions.

    Returns
    -------
    None
        Adds plots to the provided matplotlib axes.

    Notes
    -----
    - This function requires pre-existing axes and is designed to be called multiple times on the same plot.
    - For a complete standalone comparison plot, use `plot_standalone_waveform_comparison()`.
    """

    t_params_copy, s_params_copy = set_to_params(t_params, s_params)

    # Get optimized coalescence parameters
    opt_coal_results = find_optimized_coalescence_params(
        t_params_copy, s_params_copy, **kwargs
    )
    t_params_copy = opt_coal_results["opt_t_params"]

    # Filter kwargs for get_gw()
    get_gw_kwargs = {
        key: value
        for key, value in kwargs.items()
        if key in get_gw.__code__.co_varnames
    }

    # source waveform
    s_gw = get_gw(s_params_copy, **get_gw_kwargs)
    s_phase = s_gw["phase"]

    # template waveform
    t_gw = get_gw(t_params_copy, **get_gw_kwargs)
    t_strain = np.abs(t_gw["strain"])
    t_phase = t_gw["phase"]
    axes[0].plot(t_gw["f_array"], t_strain, label=label, c=c, ls=ls)

    # phase difference
    phase_diff = s_phase - t_phase
    phase_diff = np.unwrap(phase_diff + phase_shift)
    axes[1].plot(s_gw["f_array"], phase_diff, label=label, c=c, ls=ls)

    print(
        r"mcz = {:.3g} solar masses, omega_tilde = {:.3g}, theta_tilde = {:.3g}, gamma_P = {:.3g}, epsilon = {:.3g}".format(
            s_params_copy["mcz"] / SOLMASS2SEC,
            t_params_copy["omega_tilde"],
            t_params_copy["theta_tilde"],
            t_params_copy["gamma_P"],
            opt_coal_results["ep_min"],
        )
    )


def plot_waveform_panels(
    data,
    axes: AxesArray,
    plot_local_min=False,
    local_min_omega=0.0,
    local_min_theta=0.0,
    **kwargs,
) -> None:
    """
    Add a publication-style two-panel comparison on existing axes:
    - axes[0] shows |h~(f)| for the lensed source (magenta, solid) with overlaid RP templates;
    - axes[1] shows unwrapped phase differences relative to the source for each template.

    Parameters
    ----------
    data : dict
        Must include 'source_params', 'template_params', and
        'stats' with 'ep_min_omega_tilde', 'ep_min_theta_tilde', 'ep_min_gammaP'.
        If `plot_local_min=True`, also requires 'omega_matrix', 'theta_matrix', and 'gammaP_min_matrix'.
    axes : matplotlib.axes._axes.Axes
        Length-2 array-like: `axes[0]` for strain, `axes[1]` for phase difference.
    plot_local_min : bool, optional
        If True, overlay a local-min RP at (local_min_omega, local_min_theta). Default is False.
    local_min_omega : float, optional
        Omega for the optional local minimum. Default is 0.0.
    local_min_theta : float, optional
        Theta for the optional local minimum. Default is 0.0.
    **kwargs
        Additional keyword arguments passed to `find_optimized_coalescence_params()` and `get_gw()` functions.

    Returns
    -------
    None

    Notes
    -----
    - Mutates `data['template_params']` in place to set (omega_tilde, theta_tilde, gamma_P).
    - Does not create a new Figure; styling can be added via `customize_2x1_axes()`.
    """

    # plot source waveform
    s_params = data["source_params"]
    s_gw = get_gw(s_params, **kwargs)
    s_strain = np.abs(s_gw["strain"])
    axes[0].plot(s_gw["f_array"], s_strain, label="lensed", c="magenta", ls="-")

    # plot template waveforms
    s_params = data["source_params"]
    t_params = data["template_params"]
    t_params["omega_tilde"] = 0
    t_params["theta_tilde"] = 0
    t_params["gamma_P"] = 0
    plot_template_on_axes(
        t_params,
        s_params,
        axes=axes,
        label="unlensed",
        c="k",
        ls="--",
        **kwargs,
    )

    t_params = data["template_params"]
    t_params["omega_tilde"] = data["stats"]["ep_min_omega_tilde"]
    t_params["theta_tilde"] = data["stats"]["ep_min_theta_tilde"]
    t_params["gamma_P"] = data["stats"]["ep_min_gammaP"]
    plot_template_on_axes(
        t_params,
        s_params,
        axes=axes,
        label="best" if plot_local_min else "RP",
        c="k",
        ls="-",
        **kwargs,
    )

    if plot_local_min:
        t_params = data["template_params"]
        t_params["omega_tilde"] = local_min_omega
        t_params["theta_tilde"] = local_min_theta
        t_params["gamma_P"] = data["gammaP_min_matrix"][
            np.where(
                (data["omega_matrix"] == local_min_omega)
                & (data["theta_matrix"] == local_min_theta)
            )
        ]
        plot_template_on_axes(
            t_params,
            s_params,
            axes=axes,
            label="local",
            c="blue",
            ls="-.",
            **kwargs,
        )


def customize_2x1_axes(axes: AxesArray) -> None:
    # customize strain plot
    axes[0].set_xlabel(r"$f\,[\mathrm{Hz}]$", fontsize=24)
    axes[0].set_ylabel(r"$|\~{h}|$", fontsize=24)
    axes[0].tick_params(axis="both", which="major", labelsize=18)
    axes[0].grid()
    axes[0].set_yscale("log")
    axes[0].legend(fontsize=20)

    # customize phase difference plot
    axes[1].set_xlabel(r"$f\,[\mathrm{Hz}]$", fontsize=24)
    axes[1].set_ylabel(
        r"$\Phi_{\mathrm{s}} - \Phi_{\mathrm{t}}\,[\mathrm{rad}]$",
        fontsize=24,
    )
    axes[1].tick_params(axis="both", which="major", labelsize=18)
    axes[1].grid()
    # handles, labels = axes[0].get_legend_handles_labels()
    # axes[1].legend(handles, labels, fontsize=20)


def customize_2x2_axes(axes: AxesArray) -> None:
    # top panel
    axes[0, 0].legend(
        bbox_to_anchor=(2.3, 1), loc="upper left", borderaxespad=0.0, fontsize=20
    )
    axes[0, 0].tick_params(axis="both", which="major", labelsize=18)
    axes[0, 0].grid()
    axes[0, 0].set_yscale("log")
    axes[0, 1].tick_params(axis="both", which="major", labelsize=18)
    axes[0, 1].grid()

    # bottom panel
    axes[1, 0].legend(
        bbox_to_anchor=(2.3, 1), loc="upper left", borderaxespad=0.0, fontsize=20
    )
    axes[1, 0].set_xlabel(r"$f\,[\mathrm{Hz}]$", fontsize=24)
    axes[1, 0].tick_params(axis="both", which="major", labelsize=18)
    axes[1, 0].grid()
    axes[1, 0].set_yscale("log")
    axes[1, 1].set_xlabel(r"$f\,[\mathrm{Hz}]$", fontsize=24)
    axes[1, 1].tick_params(axis="both", which="major", labelsize=18)
    axes[1, 1].grid()

    # set y limits for both axes[0, 0] and axes[1, 0] such that they are same
    y0_0 = axes[0, 0].get_ylim()
    y1_0 = axes[1, 0].get_ylim()
    y_max = max(y0_0[1], y1_0[1])
    y_max = 1e-23 if y_max < 1e-23 else y_max
    y_min = min(y0_0[0], y1_0[0])
    axes[0, 0].set_ylim(y_min, y_max)
    axes[1, 0].set_ylim(y_min, y_max)


def customize_2x2_axes_ratio(axes: AxesArray) -> None:
    # top panel
    axes[0, 0].legend(
        bbox_to_anchor=(2.3, 1), loc="upper left", borderaxespad=0.0, fontsize=20
    )
    axes[0, 0].tick_params(axis="both", which="major", labelsize=18)
    axes[0, 0].grid()
    axes[0, 1].tick_params(axis="both", which="major", labelsize=18)
    axes[0, 1].grid()

    # bottom panel
    axes[1, 0].legend(
        bbox_to_anchor=(2.3, 1), loc="upper left", borderaxespad=0.0, fontsize=20
    )
    axes[1, 0].set_xlabel(r"$f\,[\mathrm{Hz}]$", fontsize=24)
    axes[1, 0].tick_params(axis="both", which="major", labelsize=18)
    axes[1, 0].grid()
    axes[1, 1].set_xlabel(r"$f\,[\mathrm{Hz}]$", fontsize=24)
    axes[1, 1].tick_params(axis="both", which="major", labelsize=18)
    axes[1, 1].grid()

    for ax in (axes[0, 0], axes[1, 0]):
        ax.set_yscale("linear")
        ax.relim()
        ax.autoscale_view()


def customize_3x2_axes_abs(axes: AxesArray) -> None:
    for row in range(3):
        axes[row, 0].legend(
            bbox_to_anchor=(2.3, 1), loc="upper left", borderaxespad=0.0, fontsize=20
        )
        axes[row, 0].tick_params(axis="both", which="major", labelsize=18)
        axes[row, 0].grid()
        axes[row, 0].set_yscale("log")
        axes[row, 1].tick_params(axis="both", which="major", labelsize=18)
        axes[row, 1].grid()

    axes[2, 0].set_xlabel(r"$f\,[\mathrm{Hz}]$", fontsize=24)
    axes[2, 1].set_xlabel(r"$f\,[\mathrm{Hz}]$", fontsize=24)


def customize_3x2_axes_ratio(axes: AxesArray) -> None:
    for row in range(3):
        axes[row, 0].legend(
            bbox_to_anchor=(2.3, 1), loc="upper left", borderaxespad=0.0, fontsize=20
        )
        axes[row, 0].tick_params(axis="both", which="major", labelsize=18)
        axes[row, 0].grid()
        axes[row, 0].set_yscale("linear")
        axes[row, 1].tick_params(axis="both", which="major", labelsize=18)
        axes[row, 1].grid()

    axes[2, 0].set_xlabel(r"$f\,[\mathrm{Hz}]$", fontsize=24)
    axes[2, 1].set_xlabel(r"$f\,[\mathrm{Hz}]$", fontsize=24)


def customize_2x1_axes_ratio(axes: AxesArray) -> None:
    axes[0].set_xlabel(r"$f\,[\mathrm{Hz}]$", fontsize=24)
    axes[0].set_ylabel(r"$\left(B / B_{\mathrm{UL}}\right) - 1$", fontsize=24)
    axes[0].tick_params(axis="both", which="major", labelsize=18)
    axes[0].grid()
    axes[0].set_yscale("linear")
    axes[0].legend(fontsize=20)

    axes[1].set_xlabel(r"$f\,[\mathrm{Hz}]$", fontsize=24)
    axes[1].set_ylabel(
        r"$\Phi_{\mathrm{L}} - \Phi_{\mathrm{RP}}\,[\mathrm{rad}]$",
        fontsize=24,
    )
    axes[1].tick_params(axis="both", which="major", labelsize=18)
    axes[1].grid()


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
