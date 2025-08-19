#############################
# Section 1: Import Modules #
#############################


import numpy as np

# Be robust to environments where np.seterr may be unavailable or altered
try:
    error_handler = np.seterr(invalid="raise")
except Exception:  # pragma: no cover - environment-specific safeguard
    error_handler = None

# Define converters/constants
SOLMASS2SEC = 4.92624076 * 1e-6  # solar mass -> seconds
GIGAPC2SEC = 1.02927125 * 1e17  # gigaparsec -> seconds
YEAR2SEC = 31557600  # year -> seconds


#################################
# Section 2: Lensing Parameters #
#################################


lens_params_0 = {
    "theta_S": np.pi / 4,
    "phi_S": 0.0,
    "theta_J": np.pi / 2,  # J == L (no precession)
    "phi_J": np.pi / 2,  # J == L (no precession)
    "mcz": 10 * SOLMASS2SEC,  # chirp mass, solar mass = sec
    "dist": 1.5 * GIGAPC2SEC,  # luminosity distance, gigaparsec = sec
    "eta": 0.25,  # symmetric mass ratio, dimensionless
    "t_c": 0.0,  # coalescence time, sec
    "phi_c": 0.0,  # coalescence phase, dimensionless
    "y": 0.25,  # source position, dimensionless
    "MLz": 1e3 * SOLMASS2SEC,  # lens mass, solar mass = sec
}


lens_params_1 = {
    "theta_S": np.pi / 4,
    "phi_S": 0,
    "theta_J": 8 * np.pi / 9,  # J == L (no precession)
    "phi_J": np.pi / 4,  # J == L (no precession)
    "mcz": 20 * SOLMASS2SEC,
    "dist": 1.5 * GIGAPC2SEC,
    "eta": 0.25,
    "t_c": 0.0,
    "phi_c": 0.0,
    "y": 0.25,
    "MLz": 2e3 * SOLMASS2SEC,
}


####################################
# Section 3: Precessing Parameters #
####################################


RP_params_0 = {
    "theta_S": np.pi / 4,
    "phi_S": 0.0,
    "theta_J": np.pi / 2,
    "phi_J": np.pi / 2,
    "mcz": 10 * SOLMASS2SEC,
    "dist": 1.5 * GIGAPC2SEC,
    "eta": 0.25,
    "t_c": 0.0,
    "phi_c": 0.0,
    "theta_tilde": 4.0,  # precession amplitude, dimensionless
    "omega_tilde": 2.0,  # precession frequency, dimensionless
    "gamma_P": 0.0,  # initial precessing phase, dimensionless
}


NP_params_0 = {
    "theta_S": np.pi / 4,
    "phi_S": 0.0,
    "theta_J": np.pi / 2,
    "phi_J": np.pi / 2,
    "mcz": 10 * SOLMASS2SEC,
    "dist": 1.5 * GIGAPC2SEC,
    "eta": 0.25,
    "t_c": 0.0,
    "phi_c": 0.0,
    "theta_tilde": 0.0,
    "omega_tilde": 0.0,
    "gamma_P": 0.0,
}


RP_params_1 = {
    "theta_S": np.pi / 4,
    "phi_S": 0,
    "theta_J": 8 * np.pi / 9,
    "phi_J": np.pi / 4,
    "mcz": 20 * SOLMASS2SEC,
    "dist": 1.5 * GIGAPC2SEC,
    "eta": 0.25,
    "t_c": 0.0,
    "phi_c": 0.0,
    "theta_tilde": 4.0,
    "omega_tilde": 2.0,
    "gamma_P": 0.0,
}


NP_params_1 = {
    "theta_S": np.pi / 4,
    "phi_S": 0,
    "theta_J": 8 * np.pi / 9,
    "phi_J": np.pi / 4,
    "mcz": 20 * SOLMASS2SEC,
    "dist": 1.5 * GIGAPC2SEC,
    "eta": 0.25,
    "t_c": 0.0,
    "phi_c": 0.0,
    "theta_tilde": 0.0,
    "omega_tilde": 0.0,
    "gamma_P": 0.0,
}


##################################################################
## Precessing Parameters for Different Distribution Percentiles ##
##################################################################

# (omega_tilde, theta_tilde) pairs in order of distribution percentiles 1%, 50%, and 95% for equal-mass, maximally spinning BBHs

omega_theta_tilde_pairs = {
    0.05: {"omega_tilde": 1, "theta_tilde": 1},
    0.50: {"omega_tilde": 2, "theta_tilde": 4},
    0.95: {"omega_tilde": 3, "theta_tilde": 8},
}


############################
# Section 4: Sky Locations #
############################


loc_params = {}

loc_params["Saif"] = {
    "faceon": {
        "theta_S": np.pi / 6,
        "phi_S": np.pi / 4,
        "theta_J": np.pi / 6,
        "phi_J": np.pi / 4,
    },
    "edgeon": {
        "theta_S": np.pi / 6,
        "phi_S": np.pi / 3,
        "theta_J": 2 * np.pi / 3,
        "phi_J": np.pi / 3,
    },
    "random": {
        "theta_S": 0,
        "phi_S": np.pi / 4,
        "theta_J": np.pi / 3,
        "phi_J": np.pi / 4,
    },
}

loc_params["Taman"] = {
    "faceon": {
        "theta_S": np.pi / 4,
        "phi_S": 0,
        "theta_J": np.pi / 4,
        "phi_J": 0,
    },
    "edgeon": {
        "theta_S": np.pi / 4,
        "phi_S": 0,
        "theta_J": np.pi / 2,
        "phi_J": np.pi / 2,
    },
    "random": {
        "theta_S": np.pi / 4,
        "phi_S": 0,
        "theta_J": 8 * np.pi / 9,
        "phi_J": np.pi / 4,
    },
}

loc_params["Tien"] = {
    "faceon": {
        "theta_S": np.pi / 8,
        "phi_S": np.pi / 3,
        "theta_J": np.pi / 8,
        "phi_J": np.pi / 3,
    },
    "edgeon": {
        "theta_S": np.pi / 6,
        "phi_S": np.pi / 3,
        "theta_J": 2 * np.pi / 3,
        "phi_J": np.pi / 3,
    },
    "random": {
        "theta_S": np.pi / 6,
        "phi_S": np.pi / 4,
        "theta_J": np.pi / 3,
        "phi_J": np.pi / 4,
    },
}
