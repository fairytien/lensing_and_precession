#############################
# Section 1: Import Modules #
#############################


# if running on Google Colab, uncomment the following lines
# import sys
# !{sys.executable} -m pip install pycbc ligo-common --no-cache-dir

# import py scripts
from modules.functions_ver2 import *


################################################
# Section 7: Optimize Mismatch Over Parameters #
################################################


def find_optimized_coalescence_params_careful(
    t_params: dict,  # template parameters
    s_params: dict,  # source parameters
    f_min=20,
    delta_f=0.25,
    psd=None,
    lens_Class=LensingGeo,
    prec_Class=Precessing,
    use_opt_match=True,
    be_extra_careful=False,
    get_updated_mismatch_results=False,
) -> dict:

    # TODO: be_extra_careful=True, get_updated_mismatch_results=True returns phi similar to be_extra_careful=False, get_updated_mismatch_results=False

    """
    Find the optimized time and phase of coalescence in the template parameters for the template waveform to match with the source waveform.

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
    be_extra_careful : bool, optional
        If True, repeats the process one more time to be extra careful in updating the coalescence parameters. But this slows down the function. Default is False.
    get_updated_mismatch_results : bool, optional
        If True, gets the updated mismatch results dictionary after the correct t_c and phi_c are updated in the template parameters. The t_c (index) and phi_c in the updated mismatch results should be ~0. This is useful for debugging but also slows down the function. Default is False.

    Returns
    -------
    dict
        A dictionary containing the following keys:
        - "updated_t_params" (dict): The updated parameters for the template waveform.
        - "updated_s_params" (dict): The updated parameters for the source waveform.
        - "updated_mismatch_results" (dict): The updated mismatch results.
    """

    t_params_copy, s_params_copy = set_to_params(t_params, s_params)

    gammaP_results_dict = optimize_mismatch_gammaP(
        t_params_copy,
        s_params_copy,
        f_min,
        delta_f,
        psd,
        lens_Class,
        prec_Class,
        use_opt_match,
    )

    # get the optimized gamma_P to plot the optimized precessing template waveform
    t_params_copy["gamma_P"] = gammaP_results_dict["ep_min_gammaP"]

    # get the optimized t_c
    ep_min_idx = gammaP_results_dict["ep_min_idx"]
    src_strain = get_gw(s_params_copy, f_min, delta_f, lens_Class, prec_Class)["strain"]
    delta_t = src_strain.delta_t
    t_params_copy["t_c"] = t_params_copy["t_c"] - ep_min_idx * delta_t

    # the optimized phi_c can only be found AFTER the optimized t_c (or index for shifting the waveform) is found
    mismatch_results_dict = mismatch(
        t_params_copy,
        s_params_copy,
        f_min,
        delta_f,
        psd,
        lens_Class,
        prec_Class,
        use_opt_match,
    )
    phi = mismatch_results_dict["phi"]
    t_params_copy["phi_c"] = phi

    # if needed, repeat the process one more time to be extra careful in updating the optimized coalescence parameters, but this is slower
    if be_extra_careful:
        updated_idx = mismatch_results_dict["index"]
        t_params_copy["t_c"] = t_params_copy["t_c"] - updated_idx * delta_t
        mismatch_results_dict = mismatch(
            t_params_copy,
            s_params_copy,
            f_min,
            delta_f,
            psd,
            lens_Class,
            prec_Class,
            use_opt_match,
        )
        updated_phi = mismatch_results_dict["phi"]
        t_params_copy["phi_c"] = updated_phi

    # after the optimized t_c and phi_c are updated in t_params_copy, the t_c and phi_c in the updated mismatch_results_dict should be ~0
    if get_updated_mismatch_results:
        mismatch_results_dict = mismatch(
            t_params_copy,
            s_params_copy,
            f_min,
            delta_f,
            psd,
            lens_Class,
            prec_Class,
            use_opt_match,
        )

    return {
        "updated_t_params": t_params_copy,
        "updated_s_params": s_params_copy,
        "updated_mismatch_results": mismatch_results_dict,
    }


#######################################
# Section 8: Plot Waveform Comparison #
#######################################


def plot_waveform_comparison_careful(
    t_params: dict,  # template parameters
    s_params: dict,  # source parameters
    f_min=20,
    delta_f=0.25,
    psd=None,
    lens_Class=LensingGeo,
    prec_Class=Precessing,
    use_opt_match=True,
    be_extra_careful=False,
    get_updated_mismatch_results=False,
) -> None:
    """
    Plot the source and optimized template waveforms in terms of strain and phase difference for comparison. The optimized coalescence parameters are used to get the optimized template waveform.

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
    be_extra_careful : bool, optional
        If True, repeats the process one more time to be extra careful in updating the coalescence parameters. But this slows down the function. Default is False.
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
    updated_params = find_optimized_coalescence_params_careful(
        t_params_copy,
        s_params_copy,
        f_min,
        delta_f,
        psd,
        lens_Class,
        prec_Class,
        use_opt_match,
        be_extra_careful,
        get_updated_mismatch_results,
    )
    s_params_copy = updated_params["updated_s_params"]
    t_params_copy = updated_params["updated_t_params"]
    epsilon = updated_params["updated_mismatch_results"]["mismatch"]
    idx = updated_params["updated_mismatch_results"]["index"]
    phi = updated_params["updated_mismatch_results"]["phi"]

    print(
        f"idx = {idx:.6g}, phi = {phi:.6g}, both should be ~0 if get_updated_mismatch_results is True"
    )  # FOR DEBUGGING

    # source waveform
    s_gw = get_gw(s_params_copy, f_min, delta_f, lens_Class, prec_Class)
    s_strain = np.abs(s_gw["strain"])
    s_phase = s_gw["phase"]
    lens_inst = lens_Class(s_params_copy)
    td = lens_inst.td()
    I = lens_inst.I()
    axes[0].plot(s_gw["f_array"], s_strain, label="source", c="k", ls="-")

    # template waveform
    t_gw = get_gw(t_params_copy, f_min, delta_f, lens_Class, prec_Class)
    t_strain = np.abs(t_gw["strain"])
    t_phase = t_gw["phase"]
    axes[0].plot(t_gw["f_array"], t_strain, label="template", c="k", ls="--")

    # customize strain plot
    axes[0].legend(fontsize=20)
    axes[0].set_xlabel("f (Hz)", fontsize=24)
    axes[0].set_ylabel(r"$|\~{h}|$", fontsize=24)
    axes[0].tick_params(axis="both", which="major", labelsize=18)
    axes[0].set_title("Strain", fontsize=24)

    # phase difference
    phase_diff = s_phase - t_phase
    phase_diff = np.unwrap(phase_diff)
    axes[1].plot(s_gw["f_array"], phase_diff, c="k", ls="-")

    # customize phase difference plot
    axes[1].set_xlabel("f (Hz)", fontsize=24)
    axes[1].set_ylabel(r"$\Phi_{\rm s} - \Phi_{\rm t}$ (rad)", fontsize=24)
    axes[1].tick_params(axis="both", which="major", labelsize=18)
    axes[1].set_title("Phase Difference", fontsize=24)

    # customize suptitle
    fig.suptitle(
        r"$\Delta t_d$ = {:.3g} ms, $I$ = {:.3g}, $\~\Omega$ = {:.3g}, $\~\theta$ = {:.3g}, {} = {:.3g} {}, $\epsilon = {:.3g}$".format(
            td * 1e3,
            I,
            t_params_copy["omega_tilde"],
            t_params_copy["theta_tilde"],
            r"$\mathcal{M}_{\rm s}$",
            s_params_copy["mcz"] / solar_mass,
            r"$M_{\odot}$",
            epsilon,
        ),
        fontsize=24,
        y=1.02,
    )


####################
# Section 9: Draft #
####################


def get_MLz_limits_for_RP_L_draft(
    params: dict, lower="min", upper="max", y=0.25, f_min=20
) -> dict:
    """
    Calculate the lower and upper limits of the lens mass [solar mass] such that the number of modulation cycles in the lensed waveform is comparable to the number of precession cycles.

    Args:
        params (dict): A dictionary containing the precessing parameters.
        lower (str or float, optional): The lower limit of the number of modulation cycles in the lensed waveform. Default is "min" for boundary between wave optics and geometric optics.
        upper (str or float, optional): The upper limit of the number of modulation cycles in the lensed waveform. Default is "max" for matching the number of precession cycles.
        y (float, optional): The source position of the lens [dimensionless]. Default is 0.25.
        f_min (float, optional): The minimum frequency. Default is 20 Hz.

    Returns:
        dict: A dictionary containing the following keys:
        - "MLz_min" (float): The minimum lens mass [solar mass].
        - "MLz_max" (float): The maximum lens mass [solar mass].
        - "td_min" (float): The minimum time delay [second].
        - "td_max" (float): The maximum time delay [second].

    Raises:
        ValueError: If "gamma_P" is not present in params.
    """

    # condition that params must be precessing parameters and already contain gamma_P
    if "gamma_P" not in params:
        raise ValueError("params must be precessing parameters")

    mcz = params["mcz"] / solar_mass
    eta = params["eta"]
    f_cut = get_fcut_from_mcz(mcz, eta)

    if lower == "min":
        MLz_min = (1 / (8 * np.pi * f_min)) / solar_mass
        td_min = get_td_from_MLz(MLz_min, y)
    elif isinstance(lower, float):
        td_min = lower / (f_cut - f_min)
        MLz_min = get_MLz_from_td(td_min, y)

    if upper == "max":
        n_prec_cycles = number_of_prec_cycles(params, f_min=f_min)
        td_max = math.ceil(n_prec_cycles) / (f_cut - f_min)
        MLz_max = get_MLz_from_td(td_max, y)
    elif isinstance(upper, float):
        td_max = upper / (f_cut - f_min)
        MLz_max = get_MLz_from_td(td_max, y)

    results = {
        "MLz_min": MLz_min,
        "MLz_max": MLz_max,
        "td_min": td_min,
        "td_max": td_max,
    }

    return results
