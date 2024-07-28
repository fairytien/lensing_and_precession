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

######################################
# Section 2: Shortcuts & Convenience #
######################################


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
