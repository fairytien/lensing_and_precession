#############################
# Section 1: Import Modules #
#############################


# import py scripts
from modules.functions_v3 import (
    set_to_params,
    get_gw,
    find_optimized_coalescence_params,
    get_I_from_y,
    get_td_from_MLz,
)
from modules.default_params_v3 import SOLMASS2SEC

# import libraries
import matplotlib
import matplotlib.pyplot as plt
from matplotlib import colors
from fractions import Fraction
import numpy as np

plt.rcParams["figure.dpi"] = 150

##########################
# Section 2: Convenience #
##########################


def set_default_plot_style():
    plt.rcParams["figure.dpi"] = 150
    plt.rcParams["axes.titlesize"] = 24
    plt.rcParams["axes.labelsize"] = 24
    plt.rcParams["xtick.labelsize"] = 18
    plt.rcParams["ytick.labelsize"] = 18
    plt.rcParams["legend.fontsize"] = 20
    plt.rcParams["figure.titlesize"] = 24


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


##########################
# Section 3: Plotting 2D #
##########################


def plot_standalone_waveform_comparison(
    t_params: dict,  # template parameters
    s_params: dict,  # source parameters
    phase_shift: float = 0,
    return_fig: bool = False,
    **kwargs,
) -> None:
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
    axes[0].set_xlabel("f (Hz)", fontsize=24)
    axes[0].set_ylabel(r"$|\~{h}|$", fontsize=24)
    axes[0].tick_params(axis="both", which="major", labelsize=18)
    axes[0].grid()
    axes[0].set_title("Strain", fontsize=24)

    # customize phase difference plot
    axes[1].set_xlabel("f (Hz)", fontsize=24)
    axes[1].set_ylabel(r"$\Phi_{\rm s} - \Phi_{\rm t}$ (rad)", fontsize=24)
    axes[1].tick_params(axis="both", which="major", labelsize=18)
    axes[1].grid()
    axes[1].set_title("Phase Difference", fontsize=24)

    # customize suptitle
    fig.suptitle(
        r"{} = {:.3g} {}, $\Delta t_d$ = {:.3g} ms, $I$ = {:.3g}, $\~\Omega$ = {:.3g}, $\~\theta$ = {:.3g}, $\gamma_P$ = {:.3g}, $\epsilon = {:.3g}$".format(
            r"$\mathcal{M}_{\rm s}$",
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
    axes: matplotlib.axes._axes.Axes,
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
    axes: matplotlib.axes._axes.Axes,
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


def customize_2x1_axes(axes: matplotlib.axes._axes.Axes) -> None:
    # customize strain plot
    axes[0].set_xlabel("f (Hz)", fontsize=24)
    axes[0].set_ylabel(r"$|\~{h}|$", fontsize=24)
    axes[0].tick_params(axis="both", which="major", labelsize=18)
    axes[0].grid()
    axes[0].set_yscale("log")
    axes[0].legend(fontsize=20)

    # customize phase difference plot
    axes[1].set_xlabel("f (Hz)", fontsize=24)
    axes[1].set_ylabel(r"$\Phi_{\rm s} - \Phi_{\rm t}$ (rad)", fontsize=24)
    axes[1].tick_params(axis="both", which="major", labelsize=18)
    axes[1].grid()
    # handles, labels = axes[0].get_legend_handles_labels()
    # axes[1].legend(handles, labels, fontsize=20)


def customize_2x2_axes(axes: matplotlib.axes._axes.Axes) -> None:
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
    axes[1, 0].set_xlabel("f (Hz)", fontsize=24)
    axes[1, 0].tick_params(axis="both", which="major", labelsize=18)
    axes[1, 0].grid()
    axes[1, 0].set_yscale("log")
    axes[1, 1].set_xlabel("f (Hz)", fontsize=24)
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


def customize_2x2_axes_ratio(axes: matplotlib.axes._axes.Axes) -> None:
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
    axes[1, 0].set_xlabel("f (Hz)", fontsize=24)
    axes[1, 0].tick_params(axis="both", which="major", labelsize=18)
    axes[1, 0].grid()
    axes[1, 1].set_xlabel("f (Hz)", fontsize=24)
    axes[1, 1].tick_params(axis="both", which="major", labelsize=18)
    axes[1, 1].grid()

    for ax in (axes[0, 0], axes[1, 0]):
        ax.set_yscale("linear")
        ax.relim()
        ax.autoscale_view()


def customize_3x2_axes_abs(axes: matplotlib.axes._axes.Axes) -> None:
    for row in range(3):
        axes[row, 0].legend(
            bbox_to_anchor=(2.3, 1), loc="upper left", borderaxespad=0.0, fontsize=20
        )
        axes[row, 0].tick_params(axis="both", which="major", labelsize=18)
        axes[row, 0].grid()
        axes[row, 0].set_yscale("log")
        axes[row, 1].tick_params(axis="both", which="major", labelsize=18)
        axes[row, 1].grid()

    axes[2, 0].set_xlabel("f (Hz)", fontsize=24)
    axes[2, 1].set_xlabel("f (Hz)", fontsize=24)


def customize_3x2_axes_ratio(axes: matplotlib.axes._axes.Axes) -> None:
    for row in range(3):
        axes[row, 0].legend(
            bbox_to_anchor=(2.3, 1), loc="upper left", borderaxespad=0.0, fontsize=20
        )
        axes[row, 0].tick_params(axis="both", which="major", labelsize=18)
        axes[row, 0].grid()
        axes[row, 0].set_yscale("linear")
        axes[row, 1].tick_params(axis="both", which="major", labelsize=18)
        axes[row, 1].grid()

    axes[2, 0].set_xlabel("f (Hz)", fontsize=24)
    axes[2, 1].set_xlabel("f (Hz)", fontsize=24)


def customize_2x1_axes_ratio(axes: matplotlib.axes._axes.Axes) -> None:
    axes[0].set_xlabel("f (Hz)", fontsize=24)
    axes[0].set_ylabel(r"$\left(B/B_{\rm unlensed}\right) - 1$", fontsize=24)
    axes[0].tick_params(axis="both", which="major", labelsize=18)
    axes[0].grid()
    axes[0].set_yscale("linear")
    axes[0].legend(fontsize=20)

    axes[1].set_xlabel("f (Hz)", fontsize=24)
    axes[1].set_ylabel(r"$\Phi_{\mathrm{L}} - \Phi_{\mathrm{RP}}$ (rad)", fontsize=24)
    axes[1].tick_params(axis="both", which="major", labelsize=18)
    axes[1].grid()


##########################
# Section 4: Plotting 3D #
##########################


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

    plt.contourf(X, Y, Z, levels=n_levels, cmap="jet")
    plt.xlabel(r"$\~\Omega$", fontsize=14)
    plt.ylabel(r"$\~\theta$", fontsize=14)
    plt.colorbar(cmap="jet", norm=colors.Normalize(vmin=0, vmax=1)).set_label(
        label=r"$\epsilon(\~h_{\rm L}, \~h_{\rm P})$", size=14
    )

    if n_minima > 0:
        ep_min_indices = np.unravel_index(np.argsort(Z, axis=None)[:n_minima], Z.shape)
        plt.scatter(X[ep_min_indices], Y[ep_min_indices], color="white", marker="o")
        print(
            f"minima: {Z[ep_min_indices]}, omega: {X[ep_min_indices]}, theta: {Y[ep_min_indices]}"
        )

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
        plt.title(
            r"$\theta_S$ = {}, $\phi_S$ = {}, $\theta_J$ = {}, $\phi_J$ = {}, {} = {:.3g} {}, $\Delta t_d$ = {:.3g} ms, $I$ = {:.3g}".format(
                angle_to_pi_string(src_params["theta_S"]),
                angle_to_pi_string(src_params["phi_S"]),
                angle_to_pi_string(src_params["theta_J"]),
                angle_to_pi_string(src_params["phi_J"]),
                r"$\mathcal{M}_{\rm s}$",
                src_params["mcz"] / SOLMASS2SEC,
                r"$M_{\odot}$",
                td * 1e3,
                I,
            ),
            fontsize=12,
            y=1.021,
        )


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

    plt.contourf(X, Y, Z, levels=n_levels, cmap="jet")
    plt.xlabel(r"$\~\Omega$", fontsize=14)
    plt.ylabel(r"$\~\theta$", fontsize=14)
    plt.colorbar(cmap="jet", norm=colors.Normalize(vmin=0, vmax=1)).set_label(
        label=r"$\epsilon(\~h_{\rm L}, \~h_{\rm P})$", size=14
    )

    if n_minima > 0:
        ep_min_indices = np.unravel_index(np.argsort(Z, axis=None)[:n_minima], Z.shape)
        plt.scatter(X[ep_min_indices], Y[ep_min_indices], color="white", marker="o")
        print(
            f"minima: {Z[ep_min_indices]}, omega: {X[ep_min_indices]}, theta: {Y[ep_min_indices]}"
        )

    if suptitle:
        plt.suptitle(
            "Mismatch Between RP Templates and a Lensed Source",
            fontsize=16,
            y=1.0215,
            x=0.435,
        )

    if title:
        plt.title(
            r"$\theta_S$ = {}, $\phi_S$ = {}, $\theta_J$ = {}, $\phi_J$ = {}, {} = {:.3g} {}, $\Delta t_d$ = {:.3g} ms, $I$ = {:.3g}".format(
                angle_to_pi_string(src_params["theta_S"]),
                angle_to_pi_string(src_params["phi_S"]),
                angle_to_pi_string(src_params["theta_J"]),
                angle_to_pi_string(src_params["phi_J"]),
                r"$\mathcal{M}_{\rm s}$",
                src_params["mcz"] / SOLMASS2SEC,
                r"$M_{\odot}$",
                td * 1e3,
                I,
            ),
            fontsize=12,
            y=1.021,
        )
