#############################
# Section 1: Import Modules #
#############################


# if running on Google Colab, uncomment the following lines
# import sys
# !{sys.executable} -m pip install pycbc ligo-common --no-cache-dir

# import py scripts
from modules.Lensing import *
from modules.Precessing import *

# import modules
import numpy as np

error_handler = np.seterr(invalid="raise")

import matplotlib

matplotlib.rcParams["figure.dpi"] = 150
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from scipy.integrate import simps
from scipy.integrate import odeint
import scipy.special as sc
from scipy.optimize import fsolve
from pycbc.filter import match, optimized_match, make_frequency_series
from pycbc.types import FrequencySeries, TimeSeries
from pycbc import catalog
import ipywidgets
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


def get_gw_vec(f_min=20, delta_f=0.25, **kwargs):
    if "MLz" in kwargs and "y" in kwargs:  # lensing parameters
        f_cut_arr = np.array([L_f_cut(**kwargs)])
        f_arr_arr = np.array([np.arange(f_min, f_cut, delta_f) for f_cut in f_cut_arr])
        strain_arr = L_strain(f_arr_arr, **kwargs)
    elif "omega_tilde" in kwargs and "theta_tilde" in kwargs:  # precessing parameters
        f_cut_arr = np.array([P_f_cut(**kwargs)])
        f_arr_arr = np.array([np.arange(f_min, f_cut, delta_f) for f_cut in f_cut_arr])
        strain_arr = P_strain(f_arr_arr, **kwargs)
    return {"f_array": f_arr_arr, "strain": strain_arr}
