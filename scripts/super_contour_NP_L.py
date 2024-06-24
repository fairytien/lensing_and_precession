import sys

sys.path.insert(
    0, "/Users/fairytien/Documents/TEXAS_Bridge_2324/code/lensing_and_precession/"
)
from modules.Classes_ver2 import *
from modules.default_params_ver1 import *
from modules.functions_ver2 import *
from modules.contours_ver2 import *


@timer_decorator
def main():
    print("Number of cores: ", cpu_count())

    mcz = 30
    lens_params_1["mcz"] = RP_params_1["mcz"] = NP_params_1["mcz"] = mcz * solar_mass
    lens_params, RP_params, NP_params = set_to_params(
        lens_params_1, RP_params_1, NP_params_1
    )

    # Find td limits and assign parameters
    RP_params["omega_tilde"] = 3
    limits = get_lens_limits_for_RP_L(RP_params, lower=0.5)
    td_min, td_max = limits["td_min"], limits["td_max"]
    td_arr = np.linspace(td_min, td_max, 40)
    I_arr = np.linspace(0.1, 0.5, 40)

    print("Finished assigning parameters")

    results = get_super_contour(
        NP_params, lens_params, td_arr, I_arr, what_template="NP"
    )

    filepath = pickle_data(results, "data", "super_contour_NP_L_mcz" + str(mcz))


if __name__ == "__main__":
    main()
