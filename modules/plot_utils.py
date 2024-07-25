#############################
# Section 1: Import Modules #
#############################


# import py scripts
from modules.functions_ver2 import *
from modules.contours_ver2 import *


##########################
# Section 2: Convenience #
##########################


def default_plot_fontsizes():
    plt.rcParams["figure.dpi"] = 150
    plt.rcParams["axes.titlesize"] = 24
    plt.rcParams["axes.labelsize"] = 24
    plt.rcParams["xtick.labelsize"] = 18
    plt.rcParams["ytick.labelsize"] = 18
    plt.rcParams["legend.fontsize"] = 20
    plt.rcParams["figure.titlesize"] = 24


def angle_in_pi_format(angle: float, denom_thres=50) -> str:
    """
    Converts an angle in radians to a string in pi format.

    Args:
        angle (float): The angle in radians.
        denom_thres (int): The threshold for the denominator of the fraction. Default is 50.

    Returns:
        str: The angle in pi format.
    """

    # Handle special cases
    if angle == 0:
        return "0"
    elif angle == np.pi:
        return r"$\pi$"
    elif angle == -np.pi:
        return r"-$\pi$"
    elif angle % np.pi == 0:
        return rf"{int(angle / np.pi)}$\pi$"

    else:
        # Convert the angle to a fraction of pi
        fraction = Fraction(angle / np.pi).limit_denominator(1000)

        # If the denominator is above the threshold, return the decimal form
        if fraction.denominator > denom_thres:
            return rf"{angle/np.pi:.3f}$\pi$"

        # If the numerator is 1, we don't need to show it
        if fraction.numerator == 1:
            return rf"$\pi$/{fraction.denominator}"

        # Otherwise, return the fraction form
        return rf"{fraction.numerator}$\pi$/{fraction.denominator}"


##########################
# Section 3: Plotting 2D #
##########################


def plot_waveform_comparison(
    t_params: dict,  # template parameters
    s_params: dict,  # source parameters
    phase_shift: float = 0,
    f_min=20,
    delta_f=0.25,
    psd=None,
    lens_Class=LensingGeo,
    prec_Class=Precessing,
    use_opt_match=True,
    get_updated_mismatch_results=False,
) -> None:
    """
    Plots the source and optimized template waveforms in terms of strain and phase difference for comparison. The optimized coalescence parameters are used to get the optimized template waveform.

    Parameters
    ----------
    t_params : dict
        The parameters for the template waveform.
    s_params : dict
        The parameters for the source waveform.
    f_min : float, optional
        The minimum frequency for the waveform. Default is 20 Hz.
    delta_f : float, optional
        The frequency spacing between samples. Default is 0.25 Hz.
    psd : FrequencySeries, optional
        The power spectral density of the detector noise. If not provided, it will be calculated based on the aLIGO noise curve from arXiv:0903.0338, as a function of the source waveform's frequency range. Default is None.
    lens_Class : class, optional
        A class representing the lensed waveform. Default is LensingGeo.
    prec_Class : class, optional
        A class representing the precessing waveform. Default is Precessing.
    use_opt_match : bool, optional
        If True, uses the optimized_match function from pycbc.filter. Default is True.
    get_updated_mismatch_results : bool, optional
        If True, gets the updated mismatch results dictionary after the correct t_c and phi_c are updated in the template parameters. The t_c (index) and phi_c in the updated mismatch results should be ~0. This is useful for debugging but also slows down the function. Default is False.

    Returns
    -------
    None
    """

    fig, axes = plt.subplots(nrows=1, ncols=2, figsize=(18, 6))
    fig.subplots_adjust(wspace=0.25)

    t_params_copy, s_params_copy = set_to_params(t_params, s_params)

    # get optimized coalescence parameters
    updated_params = find_optimized_coalescence_params(
        t_params_copy,
        s_params_copy,
        f_min,
        delta_f,
        psd,
        lens_Class,
        prec_Class,
        use_opt_match,
        get_updated_mismatch_results,
    )
    s_params_copy = updated_params["updated_s_params"]
    t_params_copy = updated_params["updated_t_params"]
    epsilon = updated_params["updated_mismatch_results"]["mismatch"]
    idx = updated_params["updated_mismatch_results"]["index"]
    phi = updated_params["updated_mismatch_results"]["phi"]

    print(
        f"idx = {idx:.3g}, phi = {phi:.3g}, both should be ~0 if get_updated_mismatch_results is True"
    )  # FOR DEBUGGING

    # source waveform
    s_gw = get_gw(s_params_copy, f_min, delta_f, lens_Class, prec_Class)
    s_strain = np.abs(s_gw["strain"])
    s_phase = s_gw["phase"]
    s_farray = s_gw["f_array"]
    lens_inst = lens_Class(s_params_copy)
    td = lens_inst.td()
    I = lens_inst.I()
    axes[0].plot(s_farray, s_strain, label="source", c="k", ls="-")

    # template waveform
    t_gw = get_gw(t_params_copy, f_min, delta_f, lens_Class, prec_Class)
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
        r"{} = {:.3g} {}, $\Delta t_d$ = {:.3g} ms, $I$ = {:.3g}, $\~\Omega$ = {:.3g}, $\~\theta$ = {:.3g}, $\epsilon = {:.3g}$".format(
            r"$\mathcal{M}_{\rm s}$",
            s_params_copy["mcz"] / solar_mass,
            r"$M_{\odot}$",
            td * 1e3,
            I,
            t_params_copy["omega_tilde"],
            t_params_copy["theta_tilde"],
            epsilon,
        ),
        fontsize=24,
        y=1.02,
    )


def plot_template_waveform(
    t_params: dict,  # template parameters
    s_params: dict,  # source parameters
    axes: matplotlib.axes._axes.Axes,
    phase_shift: float = 0,
    label: str = "template",
    c: str = "k",
    ls: str = "--",
    **kwargs,
) -> None:

    t_params_copy, s_params_copy = set_to_params(t_params, s_params)

    # get optimized coalescence parameters
    updated_params = find_optimized_coalescence_params(
        t_params_copy, s_params_copy, **kwargs
    )
    s_params_copy = updated_params["updated_s_params"]
    t_params_copy = updated_params["updated_t_params"]
    epsilon = updated_params["updated_mismatch_results"]["mismatch"]
    idx = updated_params["updated_mismatch_results"]["index"]
    phi = updated_params["updated_mismatch_results"]["phi"]

    print(
        f"idx = {idx:.3g}, phi = {phi:.3g}, both should be ~0 if get_updated_mismatch_results is True"
    )  # FOR DEBUGGING

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

    # print out suptitle
    print(
        r"mcz = {:.3g} solar masses, omega_tilde = {:.3g}, theta_tilde = {:.3g}, epsilon = {:.3g}".format(
            s_params_copy["mcz"] / solar_mass,
            t_params_copy["omega_tilde"],
            t_params_copy["theta_tilde"],
            epsilon,
        )
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
):
    plt.contourf(X, Y, Z, levels=n_levels, cmap="jet")
    plt.xlabel(r"$\~\Omega$", fontsize=14)
    plt.ylabel(r"$\~\theta$", fontsize=14)
    plt.colorbar(cmap="jet", norm=colors.Normalize(vmin=0, vmax=1)).set_label(
        label=r"$\epsilon(\~h_{\rm P}, \~h_{\rm L})$", size=14
    )

    if n_minima > 0:
        ep_min_indices = np.unravel_index(np.argsort(Z, axis=None)[:n_minima], Z.shape)
        plt.scatter(X[ep_min_indices], Y[ep_min_indices], color="white", marker="o")

    plt.suptitle(
        "Mismatch Between RP Templates and a Lensed Source",
        fontsize=16,
        y=1.0215,
        x=0.435,
    )

    td = LensingGeo(src_params).td()
    I = LensingGeo(src_params).I()

    plt.title(
        r"$\theta_S$ = {}, $\phi_S$ = {}, $\theta_J$ = {}, $\phi_J$ = {}, {} = {:.3g} {}, $\Delta t_d$ = {:.3g} ms, $I$ = {:.3g}".format(
            angle_in_pi_format(src_params["theta_S"]),
            angle_in_pi_format(src_params["phi_S"]),
            angle_in_pi_format(src_params["theta_J"]),
            angle_in_pi_format(src_params["phi_J"]),
            r"$\mathcal{M}_{\rm s}$",
            src_params["mcz"] / solar_mass,
            r"$M_{\odot}$",
            td * 1e3,
            I,
        ),
        fontsize=12,
        y=1.021,
    )


def plot_indiv_contour_from_dict(d: dict, k: float, n_levels=100, n_minima=1):
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
        label=r"$\epsilon(\~h_{\rm P}, \~h_{\rm L})$", size=14
    )

    if n_minima > 0:
        ep_min_indices = np.unravel_index(np.argsort(Z, axis=None)[:n_minima], Z.shape)
        plt.scatter(X[ep_min_indices], Y[ep_min_indices], color="white", marker="o")

    plt.suptitle(
        "Mismatch Between RP Templates and a Lensed Source",
        fontsize=16,
        y=1.0215,
        x=0.435,
    )

    plt.title(
        r"$\theta_S$ = {}, $\phi_S$ = {}, $\theta_J$ = {}, $\phi_J$ = {}, {} = {:.3g} {}, $\Delta t_d$ = {:.3g} ms, $I$ = {:.3g}".format(
            angle_in_pi_format(src_params["theta_S"]),
            angle_in_pi_format(src_params["phi_S"]),
            angle_in_pi_format(src_params["theta_J"]),
            angle_in_pi_format(src_params["phi_J"]),
            r"$\mathcal{M}_{\rm s}$",
            src_params["mcz"] / solar_mass,
            r"$M_{\odot}$",
            td * 1e3,
            I,
        ),
        fontsize=12,
        y=1.021,
    )
