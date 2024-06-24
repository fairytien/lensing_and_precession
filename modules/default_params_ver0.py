#############################
# Section 1: Import Modules #
#############################

import numpy as np

error_handler = np.seterr(invalid="raise")

#################################
# Section 2: Lensing Parameters #
#################################

solar_mass = 4.92624076 * 1e-6  # [solar_mass] = sec
giga_parsec = 1.02927125 * 1e17  # [giga_parsec] = sec
year = 31557600  # [year] = sec

lensing_params_0 = {
    "theta_S": np.pi / 4,
    "phi_S": 0.0,
    "theta_L": np.pi / 2,  # J == L (no precession)
    "phi_L": np.pi / 2,  # J == L (no precession)
    "mcz": 10 * solar_mass,
    "dist": 1.5 * giga_parsec,
    "eta": 0.25,  # symmetric mass ratio, dimensionless
    "tc": 0.0,
    "phi_c": 0.0,
    "y": 0.25,
    "MLz": 1e3 * solar_mass,
}

####################################
# Section 3: Precessing Parameters #
####################################

solar_mass = 4.92624076 * 1e-6  # [solar_mass] = sec
giga_parsec = 1.02927125 * 1e17  # [giga_parsec] = sec
year = 31557600  # [year] = sec

RP_params_0 = {
    "theta_S": np.pi / 4,
    "phi_S": 0.0,
    "theta_J": np.pi / 2,
    "phi_J": np.pi / 2,
    "mcz": 10 * solar_mass,
    "dist": 1.5 * giga_parsec,
    "eta": 0.25,
    "tc": 0.0,
    "phi_c": 0.0,
    "theta_tilde": 4.0,
    "omega_tilde": 2.0,
    "gamma_P": 0.0,
}

NP_params_0 = {
    "theta_S": np.pi / 4,
    "phi_S": 0.0,
    "theta_J": np.pi / 2,
    "phi_J": np.pi / 2,
    "mcz": 10 * solar_mass,
    "dist": 1.5 * giga_parsec,
    "eta": 0.25,
    "tc": 0.0,
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
    "05%": {"omega_tilde": 1, "theta_tilde": 1},
    "50%": {"omega_tilde": 2, "theta_tilde": 4},
    "95%": {"omega_tilde": 3, "theta_tilde": 8},
}

############################
# Section 4: Sky Locations #
############################

# (theta, phi) in order of face-on, edge-on, and random

sky_locs_J_S = [
    (np.pi / 6, np.pi / 4),
    (2 * np.pi / 3, np.pi / 3),
    (np.pi / 3, np.pi / 4),
]  # from Saif's paper
sky_locs_S_S = [
    (np.pi / 6, np.pi / 4),
    (np.pi / 6, np.pi / 3),
    (0, np.pi / 4),
]  # from Saif's paper
sky_locs_J_E = [
    (np.pi / 4, 0),
    (np.pi / 2, np.pi / 2),
    (8 * np.pi / 9, np.pi / 4),
]  # from Evangelos's paper
sky_locs_S_E = [
    (np.pi / 4, 0),
    (np.pi / 4, 0),
    (np.pi / 4, 0),
]  # from Taman's (for edge-on)

# ----------------------------------
# rewrite sky_locs_J_S and sky_locs_S_S in dictionary form

loc_params = {}

loc_params["sys1"] = {
    "faceon": {
        "phi_S": np.pi / 4,
        "theta_S": np.pi / 6,
        "phi_J": np.pi / 4,
        "theta_J": np.pi / 6,
    },
    "edgeon": {
        "phi_S": np.pi / 3,
        "theta_S": np.pi / 6,
        "phi_J": np.pi / 3,
        "theta_J": 2 * np.pi / 3,
    },
    "random": {
        "phi_S": np.pi / 4,
        "theta_S": 000000000,
        "phi_J": np.pi / 4,
        "theta_J": np.pi / 3,
    },
}

loc_params["sys2"] = {
    "faceon": {
        "phi_S": 0,
        "theta_S": np.pi / 4,
        "phi_J": 000000000,
        "theta_J": np.pi / 4,
    },
    "edgeon": {
        "phi_S": 0,
        "theta_S": np.pi / 4,
        "phi_J": np.pi / 2,
        "theta_J": np.pi / 2,
    },
    "random": {
        "phi_S": 0,
        "theta_S": np.pi / 4,
        "phi_J": np.pi / 4,
        "theta_J": 8 * np.pi / 9,
    },
}
