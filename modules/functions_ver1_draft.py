#############################
# Section 1: Import Modules #
#############################

# if running on Google Colab, uncomment the following lines
# import sys
# !{sys.executable} -m pip install pycbc ligo-common --no-cache-dir

# import py scripts
from modules.Classes_ver1 import *
from modules.default_params_ver1 import *
from modules.functions_ver1 import *

# import modules
from pycbc.filter import (
    match,
    optimized_match,
    make_frequency_series,
    get_cutoff_indices,
    sigmasq,
)

########################
# Section 2: Shortcuts #
########################

# see functions_ver1.py

###########################################
# Section 3: Inclination & Special Coords #
###########################################

# see functions_ver1.py

#######################
# Section 4: Mismatch #
#######################


def mismatch(
    cmd,
    l_params,
    rp_params,
    np_params,
    lens_Class=LensingGeo,
    prec_Class=Precessing,
    return_index=True,
    return_phi=True,
    use_optimized_match=True,
) -> Union[float, Tuple]:
    """
    Calculates the mismatch between two waveforms using the given parameters.

    Args:
        cmd (str): A string indicating which waveforms to compare.
            Options: "L & RP", "RP & L", "L & NP", "NP & L", "RP & NP", "NP & RP".
        l_params (dict): A dictionary of lensing parameters.
        rp_params (dict): A dictionary of precessing parameters.
        np_params (dict): A dictionary of non-precessing parameters.
        lens_Class (class, optional): A class representing the lensing waveform. Default is LensingGeo.
        prec_Class (class, optional): A class representing the precessing waveform. Default is Precessing.
        return_index (bool, optional): If True, returns the number of samples to shift to get the match. Default is False.
        return_phi (bool, optional): If True, returns the phase to rotate complex waveform to get the match. Default is False.
        use_optimized_match (bool, optional): If True, uses the optimized_match function from pycbc.filter. Default is False.

    Returns:
        mismatch (float): The mismatch between the two waveforms.
        index (int): The number of samples to shift to get the match, if return_index is True.
        phi (float): The phase to rotate complex waveform to get the match, if return_phi is True.
    """

    epsilon, index, phi = -1, -1, -1

    if cmd in ["L & RP", "RP & L"]:
        lens_inst = lens_Class(l_params)
        RP_inst = prec_Class(rp_params)

        f_cut = lens_inst.f_cut()
        f_min = 20
        delta_f = 0.25
        f_range = np.arange(f_min, f_cut, delta_f)

        psd_n = Sn(f_range)

        h_L = lens_inst.strain(f_range, delta_f=delta_f)
        h_RP = RP_inst.strain(f_range, delta_f=delta_f)

        if use_optimized_match:
            match_val, index, phi = optimized_match(h_RP, h_L, psd_n, return_phase=True)  # type: ignore
        else:
            match_val, index, phi = match(h_RP, h_L, psd_n, return_phase=True)  # type: ignore

        epsilon = 1 - match_val

    elif cmd in ["L & NP", "NP & L"]:
        lens_inst = lens_Class(l_params)
        NP_inst = prec_Class(np_params)

        f_cut = lens_inst.f_cut()
        f_min = 20
        delta_f = 0.25
        f_range = np.arange(f_min, f_cut, delta_f)

        psd_n = Sn(f_range)

        h_L = lens_inst.strain(f_range, delta_f=delta_f)
        h_NP = NP_inst.strain(f_range, delta_f=delta_f)

        if use_optimized_match:
            match_val, index, phi = optimized_match(h_NP, h_L, psd_n, return_phase=True)  # type: ignore
        else:
            match_val, index, phi = match(h_NP, h_L, psd_n, return_phase=True)  # type: ignore

        epsilon = 1 - match_val

    elif cmd in ["RP & NP", "NP & RP"]:
        RP_inst = prec_Class(rp_params)
        NP_inst = prec_Class(np_params)

        f_cut = RP_inst.f_cut()
        f_min = 20
        delta_f = 0.25
        f_range = np.arange(f_min, f_cut, delta_f)

        psd_n = Sn(f_range)

        h_RP = RP_inst.strain(f_range, delta_f=delta_f)
        h_NP = NP_inst.strain(f_range, delta_f=delta_f)

        if use_optimized_match:
            match_val, index, phi = optimized_match(h_NP, h_RP, psd_n, return_phase=True)  # type: ignore
        else:
            match_val, index, phi = match(h_NP, h_RP, psd_n, return_phase=True)  # type: ignore

        epsilon = 1 - match_val

    if return_index and return_phi:
        return epsilon, index, phi
    elif return_index:
        return epsilon, index
    elif return_phi:
        return epsilon, phi
    return epsilon


def mismatch_concise(
    cmd,
    l_params,
    rp_params,
    np_params,
    lens_Class=LensingGeo,
    prec_Class=Precessing,
    use_optimized_match=False,
) -> dict:
    """
    Calculates the mismatch between two waveforms using the given parameters.

    Parameters:
    -----------
    cmd : str
        A string indicating which waveforms to compare.
        Options: "L & RP", "RP & L", "L & NP", "NP & L", "RP & NP", "NP & RP".
    l_params : dict
        A dictionary of lensing parameters.
    rp_params : dict
        A dictionary of precessing parameters.
    np_params : dict
        A dictionary of non-precessing parameters.
    lens_Class : class, optional
        A class representing the lensing waveform. Default is LensingGeo.
    prec_Class : class, optional
        A class representing the precessing waveform. Default is Precessing.
    use_optimized_match : bool, optional
        If True, uses the optimized_match function from pycbc.filter. Default is False.

    Returns:
    --------
    dict
        A dictionary containing the following keys:
        - "mismatch" (float): The mismatch between the two waveforms.
        - "index" (int): The number of samples to shift to get the match.
        - "phi" (float): The phase to rotate complex waveform to get the match.
    """

    mismatch, index, phi = -9, -9, -9

    waveform_params = {
        "L & RP": (l_params, rp_params),
        "RP & L": (rp_params, l_params),
        "L & NP": (l_params, np_params),
        "NP & L": (np_params, l_params),
        "RP & NP": (rp_params, np_params),
        "NP & RP": (np_params, rp_params),
    }

    lens_inst = lens_Class(l_params)
    prec_inst, np_inst = prec_Class(rp_params), prec_Class(np_params)

    waveform_objects = {l_params: lens_inst, rp_params: prec_inst, np_params: np_inst}

    f_cut = lens_inst.f_cut()
    f_min, delta_f = 20, 0.25
    f_range = np.arange(f_min, f_cut, delta_f)
    psd_n = Sn(f_range)

    if cmd in waveform_params:
        param1, param2 = waveform_params[cmd]
        h1 = waveform_objects[param1].strain(f_range, delta_f=delta_f)
        h2 = waveform_objects[param2].strain(f_range, delta_f=delta_f)

        match_func = optimized_match if use_optimized_match else match
        match_val, index, phi = match_func(h1, h2, psd_n, return_phase=True)  # type: ignore

        mismatch = 1 - match_val

    return {"mismatch": mismatch, "index": index, "phi": phi}


def mismatch_general(
    params1,
    params2,
    lens_Class=LensingGeo,
    prec_Class=Precessing,
    use_optimized_match=False,
) -> dict:
    """
    Calculates the mismatch between two waveforms using the given parameters.

    Parameters:
    -----------
    params1 : dict
        The parameters for the first waveform.
    params2 : dict
        The parameters for the second waveform.
    lens_Class : class, optional
        A class representing the lensed waveform. Default is LensingGeo.
    prec_Class : class, optional
        A class representing the precessing waveform. Default is Precessing.
    use_optimized_match : bool, optional
        If True, uses the optimized_match function from pycbc.filter. Default is False.

    Returns:
    --------
    dict
        A dictionary containing the following keys:
        - "mismatch" (float): The mismatch between the two waveforms.
        - "index" (int): The number of samples to shift to get the match.
        - "phi" (float): The phase to rotate complex waveform to get the match.
    """

    gw1 = get_gw(params1, lens_Class=lens_Class, prec_Class=prec_Class)["waveform"]
    gw2 = get_gw(params2, lens_Class=lens_Class, prec_Class=prec_Class)["waveform"]
    gw1.resize(len(gw2))

    f_arr = get_gw(params2, lens_Class=lens_Class, prec_Class=prec_Class)["f_array"]
    psd_n = Sn(f_arr)

    match_func = optimized_match if use_optimized_match else match
    match_val, index, phi = match_func(gw1, gw2, psd_n, return_phase=True)  # type: ignore

    mismatch = 1 - match_val

    return {"mismatch": mismatch, "index": index, "phi": phi}


def mismatch_general_psd(
    params1,
    params2,
    lens_Class=LensingGeo,
    prec_Class=Precessing,
    use_optimized_match=False,
) -> dict:
    """
    Calculates the mismatch between two waveforms using the given parameters.

    Parameters:
    -----------
    params1 : dict
        The parameters for the first waveform.
    params2 : dict
        The parameters for the second waveform.
    lens_Class : class, optional
        A class representing the lensed waveform. Default is LensingGeo.
    prec_Class : class, optional
        A class representing the precessing waveform. Default is Precessing.
    use_optimized_match : bool, optional
        If True, uses the optimized_match function from pycbc.filter. Default is False.

    Returns:
    --------
    dict
        A dictionary containing the following keys:
        - "mismatch" (float): The mismatch between the two waveforms.
        - "index" (int): The number of samples to shift to get the match.
        - "phi" (float): The phase to rotate complex waveform to get the match.
    """

    gw1 = get_gw(params1, lens_Class=lens_Class, prec_Class=prec_Class)["waveform"]
    gw2 = get_gw(params2, lens_Class=lens_Class, prec_Class=prec_Class)["waveform"]
    gw1.resize(len(gw2))

    f_range = get_gw(params2, lens_Class=lens_Class, prec_Class=prec_Class)["f_range"]
    # psd_n = Sn(f_range)
    psd_n = np.ones_like(f_range)
    psd_n = FrequencySeries(psd_n, delta_f=0.25)

    match_func = optimized_match if use_optimized_match else match
    match_val, index, phi = match_func(gw1, gw2, psd_n, return_phase=True)  # type: ignore

    mismatch = 1 - match_val

    return {"mismatch": mismatch, "index": index, "phi": phi}


def mismatch_general_slice(
    params1,
    params2,
    lens_Class=LensingGeo,
    prec_Class=Precessing,
    use_optimized_match=False,
) -> dict:
    """
    Calculates the mismatch between two waveforms using the given parameters.

    Parameters:
    -----------
    params1 : dict
        The parameters for the first waveform.
    params2 : dict
        The parameters for the second waveform.
    lens_Class : class, optional
        A class representing the lensed waveform. Default is LensingGeo.
    prec_Class : class, optional
        A class representing the precessing waveform. Default is Precessing.
    use_optimized_match : bool, optional
        If True, uses the optimized_match function from pycbc.filter. Default is False.

    Returns:
    --------
    dict
        A dictionary containing the following keys:
        - "mismatch" (float): The mismatch between the two waveforms.
        - "index" (int): The number of samples to shift to get the match.
        - "phi" (float): The phase to rotate complex waveform to get the match.
    """

    gw1 = get_gw(params1, lens_Class=lens_Class, prec_Class=prec_Class)["waveform"]
    gw2 = get_gw(params2, lens_Class=lens_Class, prec_Class=prec_Class)["waveform"]

    if len(gw1) < len(gw2):
        gw2 = gw2[: len(gw1)]
    else:
        gw1 = gw1[: len(gw2)]

    f_range1 = get_gw(params1, lens_Class=lens_Class, prec_Class=prec_Class)["f_range"]
    f_range2 = get_gw(params2, lens_Class=lens_Class, prec_Class=prec_Class)["f_range"]
    f_range = f_range1 if len(f_range1) < len(f_range2) else f_range2

    psd_n = Sn(f_range)

    match_func = optimized_match if use_optimized_match else match
    match_val, index, phi = match_func(gw1, gw2, psd_n, return_phase=True)  # type: ignore

    mismatch = 1 - match_val

    return {"mismatch": mismatch, "index": index, "phi": phi}


def my_optimized_match(
    vec1,
    vec2,
    psd=None,
    low_frequency_cutoff=None,
    high_frequency_cutoff=None,
    v1_norm=None,
    v2_norm=None,
    return_phase=False,
):
    """Given two waveforms (as numpy arrays),
    compute the optimized match between them, making use
    of scipy.minimize_scalar.

    This function computes the same quantities as "match";
    it is more accurate and slower.

    Parameters
    ----------
    vec1 : TimeSeries or FrequencySeries
        The input vector containing a waveform.
    vec2 : TimeSeries or FrequencySeries
        The input vector containing a waveform.
    psd : FrequencySeries
        A power spectral density to weight the overlap.
    low_frequency_cutoff : {None, float}, optional
        The frequency to begin the match.
    high_frequency_cutoff : {None, float}, optional
        The frequency to stop the match.
    v1_norm : {None, float}, optional
        The normalization of the first waveform. This is equivalent to its
        sigmasq value. If None, it is internally calculated.
    v2_norm : {None, float}, optional
        The normalization of the second waveform. This is equivalent to its
        sigmasq value. If None, it is internally calculated.
    return_phase : {False, bool}, optional
        If True, also return the phase shift that gives the match.

    Returns
    -------
    match: float
    index: int
        The number of samples to shift to get the match.
    phi: float
        Phase to rotate complex waveform to get the match, if desired.
    """

    from scipy.optimize import minimize_scalar

    htilde = make_frequency_series(vec1)
    stilde = make_frequency_series(vec2)

    assert np.isclose(htilde.delta_f, stilde.delta_f)
    delta_f = stilde.delta_f

    assert np.isclose(htilde.delta_t, stilde.delta_t)
    delta_t = stilde.delta_t

    # a first time shift to get in the nearby region;
    # then the optimization is only used to move to the
    # correct subsample-timeshift witin (-delta_t, delta_t)
    # of this
    _, max_id, _ = match(  # type: ignore
        htilde,
        stilde,
        psd=psd,
        low_frequency_cutoff=low_frequency_cutoff,
        high_frequency_cutoff=high_frequency_cutoff,
        return_phase=True,
    )

    stilde = stilde.cyclic_time_shift(-max_id * delta_t)

    frequencies = stilde.sample_frequencies.numpy()  # type: ignore
    waveform_1 = htilde.numpy()
    waveform_2 = stilde.numpy()  # type: ignore

    N = (len(stilde) - 1) * 2  # type: ignore
    kmin, kmax = get_cutoff_indices(
        low_frequency_cutoff, high_frequency_cutoff, delta_f, N
    )
    mask = slice(kmin, kmax)

    waveform_1 = waveform_1[mask]
    waveform_2 = waveform_2[mask]
    frequencies = frequencies[mask]

    if psd is not None:
        psd_arr = psd.numpy()[mask]
    else:
        psd_arr = np.ones_like(waveform_1)

    def product(a, b):
        integral = np.sum(np.conj(a) * b / psd_arr) * delta_f
        return 4 * abs(integral), np.angle(integral)

    def product_offset(dt):
        offset = np.exp(2j * np.pi * frequencies * dt)
        return product(waveform_1, waveform_2 * offset)

    def to_minimize(dt):
        return -product_offset(dt)[0]

    norm_1 = (
        sigmasq(htilde, psd, low_frequency_cutoff, high_frequency_cutoff)
        if v1_norm is None
        else v1_norm
    )
    norm_2 = (
        sigmasq(stilde, psd, low_frequency_cutoff, high_frequency_cutoff)
        if v2_norm is None
        else v2_norm
    )

    norm = np.sqrt(norm_1 * norm_2)

    res = minimize_scalar(to_minimize, method="brent", bracket=(-delta_t, delta_t))
    m, angle = product_offset(res.x)

    if return_phase:
        return m / norm, res.x / delta_t + max_id, -angle
    else:
        return m / norm, res.x / delta_t + max_id


def optimize_mismatch_gammaP_draft(
    cmd,
    l_params,
    rp_params,
    np_params,
    lens_Class=LensingGeo,
    prec_Class=Precessing,
    use_optimized_match=False,
) -> dict:
    """
    Optimize the mismatch between the precessing template waveform and the signal by varying the polarization angle gamma_P of the precessing template.

    Parameters:
    -----------
    cmd (str): A string indicating which waveforms to compare.
        Options: "L & RP", "RP & L", "L & NP", "NP & L", "RP & NP", "NP & RP".
    l_params (dict): A dictionary of lensing parameters.
    rp_params (dict): A dictionary of precessing parameters.
    np_params (dict): A dictionary of non-precessing parameters.
    lens_Class (class, optional): A class representing the lensing waveform. Default is LensingGeo.
    prec_Class (class, optional): A class representing the precessing waveform. Default is Precessing.

    Returns:
    --------
    dict
        Dictionary containing the following keys:
        - "ep_min": minimum mismatch value
        - "ep_min_gamma": gamma_P value corresponding to minimum mismatch
        - "ep_min_idx": number of samples to shift to get the minimum mismatch at gamma_P = g_min
        - "ep_min_phi": phase to rotate complex waveform to get the minimum mismatch at gamma_P = g_min
        - "ep_max": maximum mismatch value
        - "ep_max_gamma": gamma_P value corresponding to maximum mismatch
        - "ep_max_idx": number of samples to shift to get the maximum mismatch at gamma_P = g_max
        - "ep_max_phi": phase to rotate complex waveform to get the maximum mismatch at gamma_P = g_max
        - "ep_0": mismatch value at gamma_P = 0
        - "ep_0_ind": number of samples to shift to get the mismatch at gamma_P = 0
        - "ep_0_phi": phase to rotate complex waveform to get the mismatch at gamma_P = 0
    """

    gamma_arr = np.linspace(0, 2 * np.pi, 100)
    ep_arr = np.empty_like(gamma_arr)
    idx_arr = np.empty_like(gamma_arr)
    phi_arr = np.empty_like(gamma_arr)

    for i, gamma_P in enumerate(gamma_arr):
        rp_params["gamma_P"] = gamma_P
        epsilon, index, phi = mismatch(cmd, l_params, rp_params, np_params)  # type: ignore
        ep_arr[i] = epsilon
        idx_arr[i] = index
        phi_arr[i] = phi

    ep_min_idx = np.argmin(ep_arr)
    ep_max_idx = np.argmax(ep_arr)

    results = {
        "ep_min": np.min(ep_arr),
        "ep_min_gamma": gamma_arr[ep_min_idx],
        "ep_min_idx": idx_arr[ep_min_idx],
        "ep_min_phi": phi_arr[ep_min_idx],
        "ep_max": np.max(ep_arr),
        "ep_max_gamma": gamma_arr[ep_max_idx],
        "ep_max_idx": idx_arr[ep_max_idx],
        "ep_max_phi": phi_arr[ep_max_idx],
        "ep_0": ep_arr[0],
        "ep_0_idx": idx_arr[0],
        "ep_0_phi": phi_arr[0],
    }

    return results


def optimize_mismatch_gammaP_concise(
    cmd,
    l_params,
    rp_params,
    np_params,
    lens_Class=LensingGeo,
    prec_Class=Precessing,
    use_optimized_match=False,
) -> dict:
    """
    Optimize the mismatch between the precessing template waveform and the signal by varying the polarization angle gamma_P of the precessing template.

    Parameters
    ----------
    cmd : str
        A string indicating which waveforms to compare.
        Options: "L & RP", "RP & L", "L & NP", "NP & L", "RP & NP", "NP & RP".
    l_params : dict
        A dictionary of lensing parameters.
    rp_params : dict
        A dictionary of precessing parameters.
    np_params : dict
        A dictionary of non-precessing parameters.
    lens_Class : class, optional
        A class representing the lensing waveform. Default is LensingGeo.
    prec_Class : class, optional
        A class representing the precessing waveform. Default is Precessing.
    use_optimized_match : bool, optional
        If True, uses the optimized_match function from pycbc.filter. Default is False.

    Returns
    -------
    dict
        Dictionary containing the following keys:
        - "ep_min": minimum mismatch value
        - "ep_min_gamma": gamma_P value corresponding to minimum mismatch
        - "ep_min_idx": number of samples to shift to get the minimum mismatch at gamma_P = g_min
        - "ep_min_phi": phase to rotate complex waveform to get the minimum mismatch at gamma_P = g_min
        - "ep_max": maximum mismatch value
        - "ep_max_gamma": gamma_P value corresponding to maximum mismatch
        - "ep_max_idx": number of samples to shift to get the maximum mismatch at gamma_P = g_max
        - "ep_max_phi": phase to rotate complex waveform to get the maximum mismatch at gamma_P = g_max
        - "ep_0": mismatch value at gamma_P = 0
        - "ep_0_idx": number of samples to shift to get the mismatch at gamma_P = 0
        - "ep_0_phi": phase to rotate complex waveform to get the mismatch at gamma_P = 0
    """

    gamma_arr = np.linspace(0, 2 * np.pi, 100)

    mismatch_dict = {
        gamma_P: mismatch_concise(
            cmd,
            l_params,
            {**rp_params, "gamma_P": gamma_P},
            np_params,
            lens_Class,
            prec_Class,
            use_optimized_match,
        )
        for gamma_P in gamma_arr
    }

    ep_arr = np.array([mismatch_dict[gamma_P]["mismatch"] for gamma_P in gamma_arr])
    idx_arr = np.array([mismatch_dict[gamma_P]["index"] for gamma_P in gamma_arr])
    phi_arr = np.array([mismatch_dict[gamma_P]["phi"] for gamma_P in gamma_arr])

    ep_min_idx = np.argmin(ep_arr)
    ep_max_idx = np.argmax(ep_arr)

    results = {
        "ep_min": np.min(ep_arr),
        "ep_min_gamma": gamma_arr[ep_min_idx],
        "ep_min_idx": idx_arr[ep_min_idx],
        "ep_min_phi": phi_arr[ep_min_idx],
        "ep_max": np.max(ep_arr),
        "ep_max_gamma": gamma_arr[ep_max_idx],
        "ep_max_idx": idx_arr[ep_max_idx],
        "ep_max_phi": phi_arr[ep_max_idx],
        "ep_0": ep_arr[0],
        "ep_0_idx": idx_arr[0],
        "ep_0_phi": phi_arr[0],
    }

    return results


def optimize_mismatch_gammaP_general(
    params1,
    params2,
    lens_Class=LensingGeo,
    prec_Class=Precessing,
    use_optimized_match=False,
) -> dict:
    """
    Optimize the mismatch between the precessing template and the signal by varying the initial precessing phase gamma_P of the template.

    Parameters
    ----------
    params1 : dict
        The parameters for the first waveform.
    params2 : dict
        The parameters for the second waveform.
    lens_Class : class, optional
        A class representing the lensed waveform. Default is LensingGeo.
    prec_Class : class, optional
        A class representing the precessing waveform. Default is Precessing.
    use_optimized_match : bool, optional
        If True, uses the optimized_match function from pycbc.filter. Default is False.

    Returns
    -------
    dict
        Dictionary containing the following keys:
        - "ep_min": minimum mismatch value
        - "ep_min_gamma": gamma_P value corresponding to minimum mismatch
        - "ep_min_idx": number of samples to shift to get the minimum mismatch at gamma_P = g_min
        - "ep_min_phi": phase to rotate complex waveform to get the minimum mismatch at gamma_P = g_min
        - "ep_max": maximum mismatch value
        - "ep_max_gamma": gamma_P value corresponding to maximum mismatch
        - "ep_max_idx": number of samples to shift to get the maximum mismatch at gamma_P = g_max
        - "ep_max_phi": phase to rotate complex waveform to get the maximum mismatch at gamma_P = g_max
        - "ep_0": mismatch value at gamma_P = 0
        - "ep_0_idx": number of samples to shift to get the mismatch at gamma_P = 0
        - "ep_0_phi": phase to rotate complex waveform to get the mismatch at gamma_P = 0
    """

    gamma_arr = np.linspace(0, 2 * np.pi, 100)

    # condition that params1 must be precessing parameters and already contain gamma_P
    if "gamma_P" not in params1:
        raise ValueError("params1 must be precessing parameters")

    mismatch_dict = {
        gamma_P: mismatch_general(
            {**params1, "gamma_P": gamma_P},
            params2,
            lens_Class,
            prec_Class,
            use_optimized_match,
        )
        for gamma_P in gamma_arr
    }

    ep_arr = np.array([mismatch_dict[gamma_P]["mismatch"] for gamma_P in gamma_arr])
    idx_arr = np.array([mismatch_dict[gamma_P]["index"] for gamma_P in gamma_arr])
    phi_arr = np.array([mismatch_dict[gamma_P]["phi"] for gamma_P in gamma_arr])

    ep_min_idx = np.argmin(ep_arr)
    ep_max_idx = np.argmax(ep_arr)

    results = {
        "ep_min": np.min(ep_arr),
        "ep_min_gamma": gamma_arr[ep_min_idx],
        "ep_min_idx": idx_arr[ep_min_idx],
        "ep_min_phi": phi_arr[ep_min_idx],
        "ep_max": np.max(ep_arr),
        "ep_max_gamma": gamma_arr[ep_max_idx],
        "ep_max_idx": idx_arr[ep_max_idx],
        "ep_max_phi": phi_arr[ep_max_idx],
        "ep_0": ep_arr[0],
        "ep_0_idx": idx_arr[0],
        "ep_0_phi": phi_arr[0],
    }

    return results


####################
# Section 5: Draft #
####################


def optimize_mismatch_gammaP_mcz(
    cmd,
    l_params,
    rp_params,
    np_params,
    lens_Class=LensingGeo,
    prec_Class=Precessing,
    optimized_match_function=False,
) -> dict:
    n_pts = 101
    gamma_arr = np.linspace(0, 2 * np.pi, n_pts)
    mcz_arr = np.linspace(
        l_params["mcz"] / solar_mass - 1, l_params["mcz"] / solar_mass + 1, n_pts
    )
    X, Y = np.meshgrid(gamma_arr, mcz_arr)
    ep_matrix = np.empty_like(X)

    for y in range(n_pts):
        for x in range(n_pts):
            rp_params["gamma_P"] = X[y, x]
            rp_params["mcz"] = Y[y, x] * solar_mass
            epsilon = mismatch(
                cmd, l_params, rp_params, np_params, lens_Class, prec_Class
            )
            ep_matrix[y, x] = epsilon

    # find minimum mismatch in ep_matrix
    ep_min = np.min(ep_matrix)
    ep_min_idx = np.argmin(ep_matrix)
    ep_min_gamma = X.flatten()[ep_min_idx]
    ep_min_mcz = Y.flatten()[ep_min_idx]

    # find maximum mismatch in ep_matrix
    ep_max = np.max(ep_matrix)
    ep_max_idx = np.argmax(ep_matrix)
    ep_max_gamma = X.flatten()[ep_max_idx]
    ep_max_mcz = Y.flatten()[ep_max_idx]

    # find mismatch at X = 0 (gamma) and Y = l_params["mcz"] at row n_pts // 2 and column 0
    ep_0 = ep_matrix[n_pts // 2, 0]
    ep_0_gamma = X[n_pts // 2, 0]
    ep_0_mcz = Y[n_pts // 2, 0]

    results = {
        "ep_min": ep_min,
        "ep_min_gamma": ep_min_gamma,
        "ep_min_mcz": ep_min_mcz,
        "ep_max": ep_max,
        "ep_max_gamma": ep_max_gamma,
        "ep_max_mcz": ep_max_mcz,
        "ep_0": ep_0,
        "ep_0_gamma": ep_0_gamma,
        "ep_0_mcz": ep_0_mcz,
    }

    return results


# optimize over mcz only for NP & L
def optimize_mismatch_mcz(
    cmd,
    l_params,
    rp_params,
    np_params,
    lens_Class=LensingGeo,
    prec_Class=Precessing,
    use_optimized_match=False,
) -> dict:
    n_pts = 101
    mcz_arr = np.linspace(
        l_params["mcz"] / solar_mass - 0.5, l_params["mcz"] / solar_mass + 0.5, n_pts
    )
    ep_arr = np.empty_like(mcz_arr)

    for i, mcz in enumerate(mcz_arr):
        np_params["mcz"] = mcz * solar_mass
        result = mismatch(
            cmd,
            l_params,
            rp_params,
            np_params,
            lens_Class,
            prec_Class,
            use_optimized_match=use_optimized_match,
        )

        if isinstance(result, tuple):
            epsilon = result[0]
        else:
            epsilon = result

        ep_arr[i] = epsilon

    # find minimum mismatch in ep_arr
    ep_min = np.min(ep_arr)
    ep_min_idx = np.argmin(ep_arr)
    ep_min_mcz = mcz_arr[ep_min_idx]

    # find maximum mismatch in ep_arr
    ep_max = np.max(ep_arr)
    ep_max_idx = np.argmax(ep_arr)
    ep_max_mcz = mcz_arr[ep_max_idx]

    # find mismatch at Y = l_params["mcz"]
    ep_cen_idx = n_pts // 2

    results = {
        "ep_min": ep_min,
        "ep_min_mcz": ep_min_mcz,
        "ep_max": ep_max,
        "ep_max_mcz": ep_max_mcz,
        "ep_0": ep_arr[ep_cen_idx],
        "ep_0_mcz": mcz_arr[ep_cen_idx],
    }

    return results


def optimize_mismatch_mcz_general(
    params1,
    params2,
    lens_Class=LensingGeo,
    prec_Class=Precessing,
    use_optimized_match=False,
) -> dict:
    n_pts = 101
    mcz_src = params2["mcz"] / solar_mass
    mcz_arr = np.linspace(mcz_src - 1, mcz_src + 1, n_pts)

    mismatch_dict = {
        mcz: mismatch_general(
            {**params1, "mcz": mcz * solar_mass},
            params2,
            lens_Class,
            prec_Class,
            use_optimized_match,
        )
        for mcz in mcz_arr
    }

    ep_arr = np.array([mismatch_dict[mcz]["mismatch"] for mcz in mcz_arr])
    idx_arr = np.array([mismatch_dict[mcz]["index"] for mcz in mcz_arr])
    phi_arr = np.array([mismatch_dict[mcz]["phi"] for mcz in mcz_arr])

    ep_min_idx = np.argmin(ep_arr)
    ep_max_idx = np.argmax(ep_arr)
    ep_src_idx = n_pts // 2

    results = {
        "ep_min": np.min(ep_arr),
        "ep_min_mcz": mcz_arr[ep_min_idx],
        "ep_min_idx": idx_arr[ep_min_idx],
        "ep_min_phi": phi_arr[ep_min_idx],
        "ep_max": np.max(ep_arr),
        "ep_max_mcz": mcz_arr[ep_max_idx],
        "ep_max_idx": idx_arr[ep_max_idx],
        "ep_max_phi": phi_arr[ep_max_idx],
        "ep_src": ep_arr[ep_src_idx],
        "ep_src_idx": idx_arr[ep_src_idx],
        "ep_src_phi": phi_arr[ep_src_idx],
    }

    return results


def optimize_mismatch_mcz_general_slice(
    params1,
    params2,
    lens_Class=LensingGeo,
    prec_Class=Precessing,
    use_optimized_match=False,
) -> dict:
    n_pts = 101
    mcz_src = params2["mcz"] / solar_mass
    mcz_arr = np.linspace(mcz_src - 1, mcz_src + 1, n_pts)

    mismatch_dict = {
        mcz: mismatch_general_slice(
            {**params1, "mcz": mcz * solar_mass},
            params2,
            lens_Class,
            prec_Class,
            use_optimized_match,
        )
        for mcz in mcz_arr
    }

    ep_arr = np.array([mismatch_dict[mcz]["mismatch"] for mcz in mcz_arr])
    idx_arr = np.array([mismatch_dict[mcz]["index"] for mcz in mcz_arr])
    phi_arr = np.array([mismatch_dict[mcz]["phi"] for mcz in mcz_arr])

    ep_min_idx = np.argmin(ep_arr)
    ep_max_idx = np.argmax(ep_arr)
    ep_src_idx = n_pts // 2

    results = {
        "ep_min": np.min(ep_arr),
        "ep_min_mcz": mcz_arr[ep_min_idx],
        "ep_min_idx": idx_arr[ep_min_idx],
        "ep_min_phi": phi_arr[ep_min_idx],
        "ep_max": np.max(ep_arr),
        "ep_max_mcz": mcz_arr[ep_max_idx],
        "ep_max_idx": idx_arr[ep_max_idx],
        "ep_max_phi": phi_arr[ep_max_idx],
        "ep_src": ep_arr[ep_src_idx],
        "ep_src_idx": idx_arr[ep_src_idx],
        "ep_src_phi": phi_arr[ep_src_idx],
    }

    return results


def optimize_mismatch_mcz_psd(
    params1,
    params2,
    lens_Class=LensingGeo,
    prec_Class=Precessing,
    use_optimized_match=False,
) -> dict:
    n_pts = 101
    mcz_src = params2["mcz"] / solar_mass
    mcz_arr = np.linspace(mcz_src - 1, mcz_src + 1, n_pts)

    mismatch_dict = {
        mcz: mismatch_general_psd(
            {**params1, "mcz": mcz * solar_mass},
            params2,
            lens_Class,
            prec_Class,
            use_optimized_match,
        )
        for mcz in mcz_arr
    }

    ep_arr = np.array([mismatch_dict[mcz]["mismatch"] for mcz in mcz_arr])
    idx_arr = np.array([mismatch_dict[mcz]["index"] for mcz in mcz_arr])
    phi_arr = np.array([mismatch_dict[mcz]["phi"] for mcz in mcz_arr])

    ep_min_idx = np.argmin(ep_arr)
    ep_max_idx = np.argmax(ep_arr)
    ep_src_idx = n_pts // 2

    results = {
        "ep_min": np.min(ep_arr),
        "ep_min_mcz": mcz_arr[ep_min_idx],
        "ep_min_idx": idx_arr[ep_min_idx],
        "ep_min_phi": phi_arr[ep_min_idx],
        "ep_max": np.max(ep_arr),
        "ep_max_mcz": mcz_arr[ep_max_idx],
        "ep_max_idx": idx_arr[ep_max_idx],
        "ep_max_phi": phi_arr[ep_max_idx],
        "ep_src": ep_arr[ep_src_idx],
        "ep_src_idx": idx_arr[ep_src_idx],
        "ep_src_phi": phi_arr[ep_src_idx],
    }

    return results


def find_optimized_params(t_params, s_params, opt_match=True):
    gammaP_results = optimize_mismatch_gammaP_general(
        t_params, s_params, use_optimized_match=opt_match
    )

    t_params["gamma_P"] = gammaP_results["ep_min_gamma"]
    ep_min_idx = gammaP_results["ep_min_idx"]

    delta_t = get_gw(s_params)["waveform"].delta_t
    t_params["t_c"] = t_params["t_c"] - ep_min_idx * delta_t

    mismatch_results1 = mismatch_general(
        t_params, s_params, use_optimized_match=opt_match
    )
    # testing
    updated_idx = mismatch_results1["index"]  # testing
    t_params["t_c"] = t_params["t_c"] - updated_idx * delta_t  # testing

    mismatch_results2 = mismatch_general(
        t_params, s_params, use_optimized_match=opt_match
    )  # testing
    phi = mismatch_results2["phi"]  # testing
    t_params["phi_c"] = phi

    updated_mismatch_results = mismatch_general(
        t_params, s_params, use_optimized_match=opt_match
    )

    return {
        "updated_t_params": t_params,
        "updated_s_params": s_params,
        "updated_mismatch_results": updated_mismatch_results,
    }
