#############################
# Section 1: Import Modules #
#############################


# if running on Google Colab, uncomment the following lines
# import sys
# !{sys.executable} -m pip install pycbc ligo-common --no-cache-dir

# import py scripts
from modules.Classes_ver2 import *
from modules.default_params_ver2 import *

# import modules
import numpy as np

error_handler = np.seterr(invalid="raise")

import matplotlib

matplotlib.rcParams["figure.dpi"] = 150
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib import colors
import matplotlib.image as mpimg
from scipy.integrate import simps
from scipy.integrate import odeint
import scipy.special as sc
from scipy.optimize import fsolve
import math
import sympy as sp
import pandas as pd
import mpmath as mp
from pycbc.filter import match, optimized_match, make_frequency_series
from pycbc.types import FrequencySeries, TimeSeries
from pycbc import catalog
from ipywidgets import interact, interactive, fixed, interact_manual, SelectionSlider
import os
from datetime import datetime
import time
import pickle
import copy
from typing import Union, Type, Tuple, List, Dict, Any
from fractions import Fraction


######################################
# Section 2: Shortcuts & Convenience #
######################################


def default_plot_fontsizes():
    plt.rcParams["figure.dpi"] = 150
    plt.rcParams["axes.titlesize"] = 24
    plt.rcParams["axes.labelsize"] = 24
    plt.rcParams["xtick.labelsize"] = 18
    plt.rcParams["ytick.labelsize"] = 18
    plt.rcParams["legend.fontsize"] = 20
    plt.rcParams["figure.titlesize"] = 24


def set_to_params(*args):
    """
    Returns a tuple of deep copies of the input arguments.

    Args:
        *args: Any number of arguments to be deep copied.

    Returns:
        A tuple of deep copies of the input arguments.
    """

    args_copy = [copy.deepcopy(arg) for arg in args]
    return tuple(args_copy)


def set_to_location(loc_dict: dict, *args):
    """
    Sets the location of each argument in `args` to the values specified in `loc_dict`.

    Args:
        loc_dict (dict): A dictionary containing the location values to set for each argument.
        *args: One or more dictionaries representing the arguments to modify.

    Returns:
        tuple: A tuple containing the modified versions of each argument in `args`.
    """

    args_copy = [copy.deepcopy(arg) for arg in args]

    for arg_copy in args_copy:
        arg_copy["theta_J"] = loc_dict["theta_J"]
        arg_copy["phi_J"] = loc_dict["phi_J"]
        arg_copy["theta_S"] = loc_dict["theta_S"]
        arg_copy["phi_S"] = loc_dict["phi_S"]

    return tuple(args_copy)


def get_gw(
    params: dict, f_min=20, delta_f=0.25, lens_Class=LensingGeo, prec_Class=Precessing
):
    """
    Calculates the GW for a given set of parameters.

    Args:
        params (dict): A dictionary containing the parameters.
        f_min (float): The minimum frequency.
        delta_f (float): The frequency step size.
        lens_Class (class): The class for lensing parameters.
        prec_Class (class): The class for precessing parameters.

    Returns:
        dict: A dictionary containing the following keys:
        - "strain" (np.ndarray): The complex FrequencySeries strain.
        - "phase" (np.ndarray): The GW phase.
        - "f_array" (np.ndarray): The frequency array.
    """
    if "MLz" in params and "y" in params:  # lensing parameters, use lens_Class
        gw_inst = lens_Class(params)
    else:  # precessing parameters, use prec_Class
        gw_inst = prec_Class(params)

    f_cut = gw_inst.f_cut()
    f_arr = np.arange(f_min, f_cut, delta_f)
    h = gw_inst.strain(f_arr, delta_f=delta_f)
    phase = np.unwrap(np.angle(h))

    return {"strain": h, "phase": phase, "f_array": f_arr}


def get_MLz_from_td(td, y):
    """
    Calculates the lens mass [solar mass] from the given time delay [second] and source position [dimensionless].

    Args:
        td (float or ndarray): The time delay [second].
        y (float or ndarray): The source position of the lens [dimensionless].

    Returns:
        float or ndarray: The calculated lens mass [solar mass].
    """
    divisor = 2 * (
        y * np.sqrt(y**2 + 4)
        + 2 * np.log((np.sqrt(y**2 + 4) + y) / (np.sqrt(y**2 + 4) - y))
    )
    return (td / divisor) / solar_mass


def get_td_from_MLz(MLz, y):
    """
    Calculates the time delay [second] from the given lens mass [solar mass] and source position [dimensionless], based on equation 16b in Saif et al. 2023.

    Args:
        MLz (float or ndarray): The lens mass [solar mass].
        y (float or ndarray): The source position of the lens [dimensionless].

    Returns:
        float or ndarray: The calculated time delay [second].
    """
    td_val = (
        2
        * MLz
        * solar_mass
        * (
            y * np.sqrt(y**2 + 4)
            + 2 * np.log((np.sqrt(y**2 + 4) + y) / (np.sqrt(y**2 + 4) - y))
        )
    )
    return td_val


def get_I_from_y(y):
    """
    Calculates the flux ratio [dimensionless] from the given source position [dimensionless], based on equations 16-17 in Saif et al. 2023.

    Args:
        y (float or ndarray): The source position of the lens [dimensionless].

    Returns:
        float or ndarray: The calculated flux ratio [dimensionless].
    """
    # plus magnification, equation 18 in Takahashi & Nakamura 2003, also 16a in Saif et al. 2023
    mu_plus = 1 / 2 + (y**2 + 2) / (2 * y * np.sqrt(y**2 + 4)) + 0j

    # minus magnification, equation 18 in Takahashi & Nakamura 2003, also 16a in Saif et al. 2023
    mu_minus = 1 / 2 - (y**2 + 2) / (2 * y * np.sqrt(y**2 + 4)) + 0j

    return np.abs(mu_minus) / np.abs(mu_plus)


def get_y_from_I(I):
    """
    Calculates the source position [dimensionless] from the given flux ratio [dimensionless]. Assumes I < 1 for valid calculations (positive y).

    Args:
        I (float or ndarray): The flux ratio [dimensionless]. Must be less than 1.

    Returns:
        float or ndarray: The calculated source position [dimensionless]. For ndarray inputs, returns an ndarray of source positions corresponding to each flux ratio.
    """
    # Validate input
    if np.any(I >= 1):
        raise ValueError("Flux ratio must be less than 1.")

    if isinstance(I, float):
        y_roots = fsolve(lambda y: get_I_from_y(y) - I, 1)[0]
    elif isinstance(I, np.ndarray):
        y_roots = np.zeros_like(I)
        for i, I_val in enumerate(I):
            y_roots[i] = fsolve(lambda y: get_I_from_y(y) - I_val, 1)[0]

    return y_roots


def get_fcut_from_mcz(mcz, eta=0.25):
    """
    Calculates f_cut [Hz] from the given mcz [solar mass] and eta [dimensionless].

    Args:
        mcz (float): The chirp mass [solar mass].
        eta (float): The symmetric mass ratio [dimensionless]. Default is 0.25.

    Returns:
        float: The calculated cutoff frequency at ISCO [Hz].
    """
    return eta ** (3 / 5) / (6 ** (3 / 2) * np.pi * mcz * solar_mass)


def get_mcz_from_fcut(fcut, eta=0.25):
    """
    Calculates mcz [solar mass] from the given f_cut [Hz] and eta [dimensionless].

    Args:
        fcut (float): The cutoff frequency at ISCO [Hz].
        eta (float): The symmetric mass ratio [dimensionless]. Default is 0.25.

    Returns:
        float: The calculated chirp mass [solar mass].
    """
    return eta ** (3 / 5) / (6 ** (3 / 2) * np.pi * fcut) / solar_mass


def number_of_prec_cycles(params: dict, f_min=20) -> float:
    """
    Calculates the number of precession cycles between the minimum frequency and the cutoff frequency.

    Args:
        params (dict): A dictionary containing the precessing parameters.
        f_min (float, optional): The minimum frequency. Default is 20 Hz.

    Returns:
        n_cycles (float): The number of precession cycles between the minimum frequency and the cutoff frequency.
    """

    assert (
        "theta_tilde" in params and "omega_tilde" in params
    ), "params must be precessing parameters"
    inst = Precessing(params)
    f_cut = inst.f_cut()
    phi_LJ_min = inst.phi_LJ(f_min)
    phi_LJ_cut = inst.phi_LJ(f_cut)
    n_cycles = (phi_LJ_cut - phi_LJ_min) / (2 * np.pi)
    return n_cycles


def number_of_lens_cycles(params: dict, f_min=20) -> float:
    """
    Calculates the number of modulation cycles in the lensed waveform between the minimum frequency and the cutoff frequency.

    Args:
        params (dict): A dictionary containing the lensing parameters.
        f_min (float, optional): The minimum frequency. Default is 20 Hz.

    Returns:
        n_cycles (float): The number of modulation cycles in the lensed waveform between the minimum frequency and the cutoff frequency.
    """

    assert "MLz" in params and "y" in params, "params must be lensing parameters"
    inst = LensingGeo(params)
    f_cut = inst.f_cut()
    td = inst.td()
    n_cycles = (f_cut - f_min) * td
    return n_cycles


def get_lens_limits_for_RP_L(
    params: dict,
    lower: Union[str, float] = "min",
    upper: Union[str, float] = "max",
    y=0.25,
    f_min=20,
) -> dict:
    """
    Calculates the lower and upper limits of the lens mass [solar mass] and time delay [second] such that the number of modulation cycles in the lensed waveform is comparable to the number of precession cycles.

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
        td_max = n_prec_cycles / (f_cut - f_min)
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


def pickle_data(data, dir: str, filename: str) -> str:
    """
    Pickles the given data and saves it as a file with the specified filename.

    Args:
        data: The data to be pickled.
        dir (str): The directory to save the pickled data.
        filename (str): The name of the file to save the pickled data.

    Returns:
        str: The filepath of the saved file.
    """
    now = datetime.now()
    filename = filename + "_" + now.strftime("%Y-%m-%d_%H-%M-%S") + ".pkl"
    filepath = os.path.join(dir, filename)
    with open(filepath, "wb") as f:
        pickle.dump(data, f)
        print("Results saved as", filepath)
    return filepath


def timer_decorator(func):
    def wrapper(*args, **kwargs):
        start_time = time.time()
        result = func(*args, **kwargs)
        end_time = time.time()
        total_time = end_time - start_time
        hours, remainder = divmod(total_time, 3600)
        minutes, seconds = divmod(remainder, 60)
        print(
            f"Total time to run the script: {int(hours)}:{int(minutes)}:{round(seconds, 2)} (h:m:s)"
        )
        return result

    return wrapper


###################################
# Section 3: Data & Plot Handling #
###################################


def omit_numerical_errors(arr, n=16, order=1.5) -> np.ndarray:
    """
    Omits numerical errors in an array by replacing values that are greater than a certain order of the median with NaN.

    Args:
        arr (array-like): The input array.
        n (int): The number of neighbors to consider when calculating the median. Default is 16.
        order (float): The order of the median used to determine if a value is an error. Default is 1.5.

    Returns:
        np.ndarray: The modified array with numerical errors omitted.
    """

    assert len(arr) >= n, "Array length must be greater than or equal to n."
    arr_copy = np.array(arr)

    for i in range(len(arr_copy)):
        if i < n // 2:
            neighbors = arr_copy[:n]
        elif i >= len(arr_copy) - n // 2:
            neighbors = arr_copy[-n:]
        else:
            neighbors = arr_copy[i - n // 2 : i + n // 2]

        median = np.nanmedian(neighbors)
        if arr_copy[i] > order * median:
            arr_copy[i] = np.nan

    return arr_copy


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


###########################################
# Section 4: Inclination & Special Coords #
###########################################


def calculate_cosJN_params(params: dict) -> float:
    """
    Calculates the cosine of the angle between the total angular momentum (J) of the BBH system and the line of sight (N).

    Parameters:
    - params (dict): A dictionary containing the following keys:
        - "theta_J" (float): The polar angle of J with respect to the detector frame.
        - "phi_J" (float): The azimuthal angle of J with respect to the detector frame.
        - "theta_S" (float): The polar angle of N with respect to the detector frame.
        - "phi_S" (float): The azimuthal angle of N with respect to the detector frame.
        - etc.

    Returns:
    - float: The cosine of the angle between J and N.
    """
    return np.sin(params["theta_J"]) * np.sin(params["theta_S"]) * np.cos(
        params["phi_J"] - params["phi_S"]
    ) + np.cos(params["theta_J"]) * np.cos(params["theta_S"])


def calculate_cosJN(phi_S, theta_S, phi_J, theta_J):
    """
    Calculates the cosine of the angle between the total angular momentum (J) of the BBH system and the line of sight (N).

    Parameters:
    - "phi_S" (float): The azimuthal angle of N with respect to the detector frame.
    - "theta_S" (float): The polar angle of N with respect to the detector frame.
    - "phi_J" (float): The azimuthal angle of J with respect to the detector frame.
    - "theta_J" (float): The polar angle of J with respect to the detector frame.
    - etc.

    Returns:
    - float: The cosine of the angle between J and N.
    """
    print("order of arguments: phi_S, theta_S, phi_J, theta_J")
    return np.sin(theta_J) * np.sin(theta_S) * np.cos(phi_J - phi_S) + np.cos(
        theta_J
    ) * np.cos(theta_S)


def find_FaceOn_coords(fix, fixed_phi, fixed_theta):
    n_pts = 150
    phi_arr = np.linspace(0, 2 * np.pi, n_pts)
    theta_arr = np.linspace(0, np.pi, n_pts)
    X, Y = np.meshgrid(phi_arr, theta_arr)

    if fix == "S":
        Z = calculate_cosJN(fixed_phi, fixed_theta, X, Y)
    else:  # fix == 'J'
        Z = calculate_cosJN(X, Y, fixed_phi, fixed_theta)

    # condition where |Z| = 1 within error
    cond = np.isclose(np.abs(Z), 1, rtol=0, atol=1e-3)
    return X[cond], Y[cond]


def find_EdgeOn_coords(fix, fixed_phi, fixed_theta):
    n_pts = 150
    phi_arr = np.linspace(0, 2 * np.pi, n_pts)
    theta_arr = np.linspace(0, np.pi, n_pts)
    X, Y = np.meshgrid(phi_arr, theta_arr)

    if fix == "S":
        Z = calculate_cosJN(fixed_phi, fixed_theta, X, Y)
    else:  # fix == 'J'
        Z = calculate_cosJN(X, Y, fixed_phi, fixed_theta)

    # condition where |Z| = 0 within error
    cond = np.isclose(np.abs(Z), 0, rtol=0, atol=1e-2)
    return X[cond], Y[cond]


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

    n_pts = 150
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

    n_pts = 150
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


##################
# Section 5: SNR #
##################


def Sn(f_arr, f_min=20, delta_f=0.25, frequencySeries=True):
    """aLIGO noise curve from arXiv:0903.0338"""

    Sn_val = np.zeros_like(f_arr)
    for i in range(len(f_arr)):
        if f_arr[i] < f_min:
            Sn_val[i] = np.inf
        else:
            S0 = 1e-49
            f0 = 215
            Sn_temp = (
                np.power(f_arr[i] / f0, -4.14)
                - 5 * np.power(f_arr[i] / f0, -2)
                + 111
                * (
                    (1 - np.power(f_arr[i] / f0, 2) + 0.5 * np.power(f_arr[i] / f0, 4))
                    / (1 + 0.5 * np.power(f_arr[i] / f0, 2))
                )
            )
            Sn_val[i] = Sn_temp * S0

    if frequencySeries:
        return FrequencySeries(Sn_val, delta_f=delta_f)
    return Sn_val


def SNR(
    params: dict,
    f_min=20,
    delta_f=0.25,
    psd=None,
    lens_Class=LensingGeo,
    prec_Class=Precessing,
):
    """
    Calculates the Signal-to-Noise Ratio (SNR) for a given dictionary of parameters.

    Parameters
    ----------
    params : dict
        A dictionary containing the parameters.
    f_min : float, optional
        The lower frequency cutoff. Defaults to 20 Hz.
    delta_f : float, optional
        The frequency step size. Defaults to 0.25 Hz.
    psd : FrequencySeries, optional
        The power spectral density of the detector noise. If not provided, it will be calculated based on the aLIGO noise curve from arXiv:0903.0338, as a function of the source waveform's frequency range. Defaults to None.
    lens_Class : class, optional
        The class to use for lensing parameters. Defaults to LensingGeo.
    prec_Class : class, optional
        The class to use for precessing parameters. Defaults to Precessing.

    Returns
    -------
    float
        The calculated SNR value.
    """
    if "MLz" in params and "y" in params:  # lensing parameters, use lens_Class
        gw_inst = lens_Class(params)
    else:  # precessing parameters, use prec_Class
        gw_inst = prec_Class(params)

    f_cut = gw_inst.f_cut()
    f_arr = np.arange(f_min, f_cut, delta_f)
    if psd is None:
        psd = Sn(f_arr)
    h = gw_inst.strain(f_arr, delta_f=delta_f)

    # calculate SNR
    integrand = np.abs(h) ** 2 / psd
    integrated_inner_product = simps(integrand, f_arr)
    snr = np.sqrt(4 * np.real(integrated_inner_product))

    return snr


#######################
# Section 6: Mismatch #
#######################


def mismatch(
    t_params: dict,  # template parameters
    s_params: dict,  # source parameters
    f_min=20,
    delta_f=0.25,
    psd=None,
    lens_Class=LensingGeo,
    prec_Class=Precessing,
    use_opt_match=True,
) -> dict:
    """
    Calculates the mismatch between two waveforms using the given parameters.

    Parameters:
    -----------
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

    Returns:
    --------
    dict
        A dictionary containing the following keys:
        - "mismatch" (float): The mismatch between the two waveforms.
        - "index" (int): The number of samples to shift the source waveform to match with the template.
        - "phi" (float): The phase to rotate the complex source waveform to match with the template.
    """

    t_gw = get_gw(t_params, f_min, delta_f, lens_Class, prec_Class)["strain"]
    s_gw = get_gw(s_params, f_min, delta_f, lens_Class, prec_Class)["strain"]
    t_gw.resize(len(s_gw))

    if psd is None:
        f_arr = get_gw(s_params, f_min, delta_f, lens_Class, prec_Class)["f_array"]
        psd = Sn(f_arr)

    match_func = optimized_match if use_opt_match else match
    match_val, index, phi = match_func(t_gw, s_gw, psd, return_phase=True)  # type: ignore

    mismatch = 1 - match_val

    return {"mismatch": mismatch, "index": index, "phi": phi}


################################################
# Section 7: Optimize Mismatch Over Parameters #
################################################


def optimize_mismatch_gammaP(
    t_params: dict,  # template parameters
    s_params: dict,  # source parameters
    f_min=20,
    delta_f=0.25,
    psd=None,
    lens_Class=LensingGeo,
    prec_Class=Precessing,
    use_opt_match=True,
) -> dict:
    """
    Optimizes the mismatch between the precessing template and the signal by varying the initial precessing phase gamma_P of the template.

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

    Returns
    -------
    dict
        A dictionary containing the following keys:
        - "ep_min" (float): The minimum mismatch value.
        - "ep_min_gammaP" (float): The gamma_P value corresponding to the minimum mismatch.
        - "ep_min_idx" (int): The number of samples to shift to get the minimum mismatch at ep_min_gammaP.
        - "ep_min_phi" (float): The phase to rotate the complex waveform to get the minimum mismatch at ep_min_gammaP.
        - "ep_max" (float): The maximum mismatch value.
        - "ep_max_gammaP" (float): The gamma_P value corresponding to the maximum mismatch.
        - "ep_max_idx" (int): The number of samples to shift to get the maximum mismatch at ep_max_gammaP.
        - "ep_max_phi" (float): The phase to rotate the complex waveform to get the maximum mismatch at ep_max_gammaP.
        - "ep_0" (float): The mismatch value at gamma_P = 0.
        - "ep_0_idx" (int): The number of samples to shift to get the mismatch at gamma_P = 0.
        - "ep_0_phi" (float): The phase to rotate the complex waveform to get the mismatch at gamma_P = 0.
    """

    t_params_copy, s_params_copy = set_to_params(t_params, s_params)

    # condition that t_params must be precessing parameters and already contain gamma_P
    if "gamma_P" not in t_params_copy:
        raise ValueError("t_params must be precessing parameters")

    gamma_arr = np.linspace(0, 2 * np.pi, 100)

    mismatch_dict = {
        gamma_P: mismatch(
            {**t_params_copy, "gamma_P": gamma_P},
            s_params_copy,
            f_min,
            delta_f,
            psd,
            lens_Class,
            prec_Class,
            use_opt_match,
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
        "ep_min_gammaP": gamma_arr[ep_min_idx],
        "ep_min_idx": idx_arr[ep_min_idx],
        "ep_min_phi": phi_arr[ep_min_idx],
        "ep_max": np.max(ep_arr),
        "ep_max_gammaP": gamma_arr[ep_max_idx],
        "ep_max_idx": idx_arr[ep_max_idx],
        "ep_max_phi": phi_arr[ep_max_idx],
        "ep_0": ep_arr[0],
        "ep_0_idx": idx_arr[0],
        "ep_0_phi": phi_arr[0],
    }

    return results


def optimize_mismatch_mcz(
    t_params: dict,  # template parameters
    s_params: dict,  # source parameters
    f_min=20,
    delta_f=0.25,
    psd=None,
    lens_Class=LensingGeo,
    prec_Class=Precessing,
    use_opt_match=True,
) -> dict:
    """
    Optimizes the mismatch between the unlensed template and the lensed signal by varying the chirp mass of the template.

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

    Returns
    -------
    dict
        A dictionary containing the following keys:
        - "ep_min" (float): The minimum mismatch value.
        - "ep_min_mcz" (float): The chirp mass value corresponding to the minimum mismatch.
        - "ep_min_idx" (int): The number of samples to shift to get the minimum mismatch at ep_min_mcz.
        - "ep_min_phi" (float): The phase to rotate the complex waveform to get the minimum mismatch at ep_min_mcz.
        - "ep_max" (float): The maximum mismatch value.
        - "ep_max_mcz" (float): The chirp mass value corresponding to the maximum mismatch.
        - "ep_max_idx" (int): The number of samples to shift to get the maximum mismatch at ep_max_mcz.
        - "ep_max_phi" (float): The phase to rotate the complex waveform to get the maximum mismatch at ep_max_mcz.
        - "ep_src" (float): The mismatch value at the source chirp mass.
        - "ep_src_idx" (int): The number of samples to shift to get the mismatch at the source chirp mass.
        - "ep_src_phi" (float): The phase to rotate the complex waveform to get the mismatch at the source chirp mass.
    """

    t_params_copy, s_params_copy = set_to_params(t_params, s_params)

    n_pts = 101
    mcz_src = s_params_copy["mcz"] / solar_mass
    mcz_arr = np.linspace(mcz_src - 1, mcz_src + 1, n_pts)

    mismatch_dict = {
        mcz: mismatch(
            {**t_params_copy, "mcz": mcz * solar_mass},
            s_params_copy,
            f_min,
            delta_f,
            psd,
            lens_Class,
            prec_Class,
            use_opt_match,
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


def find_optimized_coalescence_params(
    t_params: dict,  # template parameters
    s_params: dict,  # source parameters
    f_min=20,
    delta_f=0.25,
    psd=None,
    lens_Class=LensingGeo,
    prec_Class=Precessing,
    use_opt_match=True,
    get_updated_mismatch_results=False,
) -> dict:
    """
    Finds the optimized time and phase of coalescence in the template parameters for the template waveform to match with the source waveform.

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
        f"idx = {idx:.6g}, phi = {phi:.6g}, both should be ~0 if get_updated_mismatch_results is True"
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
