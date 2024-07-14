#############################
# Section 1: Import Modules #
#############################

# if running on Google Colab, uncomment the following lines
# import sys
# !{sys.executable} -m pip install pycbc ligo-common --no-cache-dir

# import py scripts
from modules.Classes_ver0 import *
from modules.default_params_ver0 import *

# import modules
import numpy as np

error_handler = np.seterr(invalid="raise")
import matplotlib

matplotlib.rcParams["figure.dpi"] = 150
import matplotlib.pyplot as plt
from matplotlib import colors
import matplotlib.image as mpimg
from scipy.integrate import simps
from scipy.integrate import odeint
import scipy.special as sc
import math
import sympy as sp
import pandas as pd
import mpmath as mp
from pycbc.filter import match
from pycbc.types import FrequencySeries
from ipywidgets import interact, interactive, fixed, interact_manual
from datetime import datetime
import pickle
import copy

#################################
# Section 2: Shortcut Functions #
#################################


def set_to_params(*args):
    args_copy = [copy.deepcopy(arg) for arg in args]
    return tuple(args_copy)


def set_to_location(loc_dict: dict, *args):
    args_copy = [copy.deepcopy(arg) for arg in args]

    for arg_copy in args_copy:
        arg_copy["theta_J"] = loc_dict["theta_J"]
        arg_copy["phi_J"] = loc_dict["phi_J"]
        arg_copy["theta_S"] = loc_dict["theta_S"]
        arg_copy["phi_S"] = loc_dict["phi_S"]

    return tuple(args_copy)


#################################
# Section 3: Mismatch Functions #
#################################


def Sn(f, delta_f=0.25, frequencySeries=True):
    """ALIGO noise curve from arXiv:0903.0338"""
    Sn_val = np.zeros_like(f)
    fs = 20
    for i in range(len(f)):
        if f[i] < fs:
            Sn_val[i] = np.inf
        else:
            S0 = 1e-49
            f0 = 215
            Sn_temp = (
                np.power(f[i] / f0, -4.14)
                - 5 * np.power(f[i] / f0, -2)
                + 111
                * (
                    (1 - np.power(f[i] / f0, 2) + 0.5 * np.power(f[i] / f0, 4))
                    / (1 + 0.5 * np.power(f[i] / f0, 2))
                )
            )
            Sn_val[i] = Sn_temp * S0

    if frequencySeries:
        return FrequencySeries(Sn_val, delta_f=delta_f)
    return Sn_val


def mismatch_epsilon(
    l_params, rp_params, np_params, cmd, lens_Class=LensingGeo, prec_Class=Precessing
):
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


def mismatch_epsilon_min_max(
    l_params, rp_params, np_params, cmd, lens_Class=LensingGeo, prec_Class=Precessing
):
    gamma_array = np.linspace(0, 2 * np.pi, 100)
    mismatch_array = np.empty_like(gamma_array)

    for i, gamma_P in enumerate(gamma_array):
        rp_params["gamma_P"] = gamma_P
        mismatch = mismatch_epsilon(
            l_params, rp_params, np_params, cmd, lens_Class, prec_Class
        )
        mismatch_array[i] = mismatch

    ind_min = np.argmin(mismatch_array)
    Epsilon_min = mismatch_array[ind_min]
    gamma_P_min = gamma_array[ind_min]

    ind_max = np.argmax(mismatch_array)
    Epsilon_max = mismatch_array[ind_max]
    gamma_P_max = gamma_array[ind_max]

    return Epsilon_min, gamma_P_min, Epsilon_max, gamma_P_max, mismatch_array[0]


#################################
# Section 4: Plotting Functions #
#################################


def cos_i_JN(phi_J, theta_J, phi_S, theta_S):
    """cosine of the angle between the total angular momentum and the line of sight"""
    return np.sin(theta_J) * np.sin(theta_S) * np.cos(phi_J - phi_S) + np.cos(
        theta_J
    ) * np.cos(theta_S)


def find_FaceOn_coords_J(phi_S, theta_S):
    # find the face-on J coordinates where abs(cos_i_JN) = 1
    n_pts = 150
    phi_J_arr = np.linspace(0, 2 * np.pi, n_pts)
    theta_J_arr = np.linspace(0, np.pi, n_pts)
    X, Y = np.meshgrid(phi_J_arr, theta_J_arr)
    Z = cos_i_JN(X, Y, phi_S, theta_S)
    # condition where Z = 1 within error
    cond = np.isclose(np.abs(Z), 1, rtol=0, atol=1e-3)
    # get phi_J, theta_J where condition is True
    return X[cond], Y[cond]


def find_EdgeOn_coords_J(phi_S, theta_S):
    # find the edge-on J coordinates where cos_i_JN = 0
    n_pts = 150
    phi_J_arr = np.linspace(0, 2 * np.pi, n_pts)
    theta_J_arr = np.linspace(0, np.pi, n_pts)
    X, Y = np.meshgrid(phi_J_arr, theta_J_arr)
    Z = cos_i_JN(X, Y, phi_S, theta_S)
    # condition where Z = 0 within error
    cond = np.isclose(np.abs(Z), 0, rtol=0, atol=1e-2)
    # get phi_J, theta_J where condition is True
    return X[cond], Y[cond]


def plot_special_coords(phi_S, theta_S):
    phi_J_FaceOn, theta_J_FaceOn = find_FaceOn_coords_J(phi_S, theta_S)
    plt.scatter(phi_J_FaceOn, np.cos(theta_J_FaceOn), s=1, c="white", label="face-on")
    phi_J_EdgeOn, theta_J_EdgeOn = find_EdgeOn_coords_J(phi_S, theta_S)
    plt.scatter(phi_J_EdgeOn, np.cos(theta_J_EdgeOn), s=1, c="black", label="edge-on")
    plt.legend(bbox_to_anchor=(1.2, 1.2), loc="upper left", borderaxespad=0.0)


def inclination_contour(phi_S, theta_S):
    # contour plot of inclination angle between J and N as a function of theta_J and phi_J
    n_pts = 100
    phi_J_mesh = np.linspace(0, 2 * np.pi, n_pts)
    theta_J_mesh = np.linspace(0, np.pi, n_pts)
    X, Y = np.meshgrid(phi_J_mesh, theta_J_mesh)
    Z = cos_i_JN(X, Y, phi_S, theta_S)
    plt.contourf(X, np.cos(Y), Z, levels=40, cmap="viridis")
    plt.colorbar(label=r"$\cos \iota_{JN}$")
    plt.ylabel(r"$\cos \theta_J$")
    plt.xlabel(r"$\phi_J$")
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
    plt.title(
        r"inclination contour at $\phi_S$ = {:.3g}, $\theta_S$ = {:.3g}".format(
            phi_S, theta_S
        )
    )
