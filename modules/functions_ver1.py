#############################
# Section 1: Import Modules #
#############################


# if running on Google Colab, uncomment the following lines
# import sys
# !{sys.executable} -m pip install pycbc ligo-common --no-cache-dir

# import py scripts
from modules.Classes_ver1 import *
from modules.default_params_ver1 import *

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
import math
import sympy as sp
import pandas as pd
import mpmath as mp
from pycbc.filter import match, optimized_match, make_frequency_series
from pycbc.types import FrequencySeries, TimeSeries
from pycbc import catalog
import ipywidgets
from ipywidgets import interact, interactive, fixed, interact_manual
from datetime import datetime
import pickle
import copy
from typing import Union, Type, Tuple, List, Dict


########################
# Section 2: Shortcuts #
########################


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
    Calculate the GW for a given set of parameters.

    Args:
        params (dict): A dictionary containing the parameters.
        f_min (float): The minimum frequency.
        delta_f (float): The frequency step size.
        lens_Class (class): The class for lensing parameters.
        prec_Class (class): The class for precessing parameters.

    Returns:
        dict: A dictionary containing the following keys:
        - "waveform" (np.ndarray): The complex FrequencySeries.
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

    return {"waveform": h, "phase": phase, "f_array": f_arr}


def get_MLz_from_time_delay(time_delay, y):
    """
    Calculate the lens mass [solar mass] from the given time delay [second] and source position y [dimensionless].

    Args:
        time_delay (float): The time delay [second].
        y (float): The source position of the lens [dimensionless].

    Returns:
        float: The calculated lens mass [solar mass].
    """
    divisor = 2 * (
        y * np.sqrt(y**2 + 4)
        + 2 * np.log((np.sqrt(y**2 + 4) + y) / (np.sqrt(y**2 + 4) - y))
    )
    return (time_delay / divisor) / solar_mass


def get_fcut_from_mcz(mcz, eta):
    """
    Calculate f_cut [Hz] from the given mcz [solar mass] and eta [dimensionless].

    Args:
        mcz (float): The chirp mass [solar mass].
        eta (float): The symmetric mass ratio [dimensionless].

    Returns:
        float: The calculated cutoff frequency at ISCO [Hz].
    """
    return eta ** (3 / 5) / (6 ** (3 / 2) * np.pi * mcz * solar_mass)


def get_mcz_from_fcut(fcut, eta):
    """
    Calculate mcz [solar mass] from the given f_cut [Hz] and eta [dimensionless].

    Args:
        fcut (float): The cutoff frequency at ISCO [Hz].
        eta (float): The symmetric mass ratio [dimensionless].

    Returns:
        float: The calculated chirp mass [solar mass].
    """
    return eta ** (3 / 5) / (6 ** (3 / 2) * np.pi * fcut) / solar_mass


###########################################
# Section 3: Inclination & Special Coords #
###########################################


def cos_i_JN_params(params: dict) -> float:
    """
    Calculate the cosine of the angle between the total angular momentum (J) of the BBH system and the line of sight (N).

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


def cos_i_JN(phi_S, theta_S, phi_J, theta_J):
    """
    Calculate the cosine of the angle between the total angular momentum (J) of the BBH system and the line of sight (N).

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
        Z = cos_i_JN(fixed_phi, fixed_theta, X, Y)
    else:  # fix == 'J'
        Z = cos_i_JN(X, Y, fixed_phi, fixed_theta)

    # condition where |Z| = 1 within error
    cond = np.isclose(np.abs(Z), 1, rtol=0, atol=1e-3)
    return X[cond], Y[cond]


def find_EdgeOn_coords(fix, fixed_phi, fixed_theta):
    n_pts = 150
    phi_arr = np.linspace(0, 2 * np.pi, n_pts)
    theta_arr = np.linspace(0, np.pi, n_pts)
    X, Y = np.meshgrid(phi_arr, theta_arr)

    if fix == "S":
        Z = cos_i_JN(fixed_phi, fixed_theta, X, Y)
    else:  # fix == 'J'
        Z = cos_i_JN(X, Y, fixed_phi, fixed_theta)

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
        Z = cos_i_JN(fixed_phi, fixed_theta, X, Y)
    else:  # fix == 'J'
        Z = cos_i_JN(X, Y, fixed_phi, fixed_theta)

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


def cos_i_JN_contour(fix, fixed_phi, fixed_theta):
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
        Z = cos_i_JN(fixed_phi, fixed_theta, X, Y)
    else:  # fix == 'J'
        Z = cos_i_JN(X, Y, fixed_phi, fixed_theta)

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
# Section 4: SNR #
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
    params: dict, f_min=20, delta_f=0.25, lens_Class=LensingGeo, prec_Class=Precessing
):
    """
    Calculate the Signal-to-Noise Ratio (SNR) for a given dictionary of parameters.

    Args:
        params (dict): A dictionary containing the parameters.
        f_min (float, optional): The minimum frequency to consider. Defaults to 20 Hz.
        delta_f (float, optional): The frequency step size. Defaults to 0.25 Hz.
        lens_Class (class, optional): The class to use for lensing parameters. Defaults to LensingGeo.
        prec_Class (class, optional): The class to use for precessing parameters. Defaults to Precessing.

    Returns:
        float: The calculated SNR value.
    """
    if "MLz" in params and "y" in params:  # lensing parameters, use lens_Class
        gw_inst = lens_Class(params)
    else:  # precessing parameters, use prec_Class
        gw_inst = prec_Class(params)

    f_cut = gw_inst.f_cut()
    f_arr = np.arange(f_min, f_cut, delta_f)
    psd_n = Sn(f_arr)
    h = gw_inst.strain(f_arr, delta_f=delta_f)

    # calculate SNR
    integrand = np.abs(h) ** 2 / psd_n
    integrated_inner_product = simps(integrand, f_arr)
    snr = np.sqrt(4 * np.real(integrated_inner_product))

    return snr


#######################
# Section 5: Mismatch #
#######################


def mismatch_epsilon(
    cmd, l_params, rp_params, np_params, lens_Class=LensingGeo, prec_Class=Precessing
):
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

    Returns:
        float: The mismatch between the two waveforms.
    """

    mismatch = None

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

        mismatch = 1 - match(h_L, h_RP, psd_n)[0]

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

        mismatch = 1 - match(h_L, h_NP, psd_n)[0]

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

        mismatch = 1 - match(h_RP, h_NP, psd_n)[0]

    return mismatch


def optimize_mismatch_gammaP(
    cmd, l_params, rp_params, np_params, lens_Class=LensingGeo, prec_Class=Precessing
) -> dict:
    """
    Optimize the mismatch between the signal and template waveform by varying the polarization angle gamma_P.

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
        - "g_min": gamma_P value corresponding to minimum mismatch
        - "ep_max": maximum mismatch value
        - "g_max": gamma_P value corresponding to maximum mismatch
        - "ep_0": mismatch value at gamma_P = 0
    """

    gamma_array = np.linspace(0, 2 * np.pi, 100)
    mismatch_array = np.empty_like(gamma_array)

    for i, gamma_P in enumerate(gamma_array):
        rp_params["gamma_P"] = gamma_P
        mismatch = mismatch_epsilon(
            cmd, l_params, rp_params, np_params, lens_Class, prec_Class
        )
        mismatch_array[i] = mismatch

    results = {
        "ep_min": np.min(mismatch_array),
        "g_min": gamma_array[np.argmin(mismatch_array)],
        "ep_max": np.max(mismatch_array),
        "g_max": gamma_array[np.argmax(mismatch_array)],
        "ep_0": mismatch_array[0],
    }

    return results
