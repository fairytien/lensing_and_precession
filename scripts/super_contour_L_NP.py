import sys, os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from modules.contours_ver2 import *


@timer_decorator
def main():
    print("Number of cores: ", cpu_count())

    # Assign parameters
    lens_params, RP_params, NP_params = set_to_location(
        loc_params["Taman"]["random"], lens_params_1, RP_params_1, NP_params_1
    )
    mcz = 20
    lens_params["mcz"] = RP_params["mcz"] = NP_params["mcz"] = mcz * solar_mass
    # RP_params["omega_tilde"] = 3
    # limits = get_lens_limits_for_RP_L(RP_params, lower=0.5)
    # td_min, td_max = limits["td_min"], limits["td_max"]
    td_arr = np.linspace(0.02, 0.06, 40)  # To be in geometric optics regime
    I_arr = np.linspace(0.1, 0.9, 40)

    print("Finished assigning parameters")

    results = create_super_contour(
        NP_params, lens_params, td_arr, I_arr, what_template="NP"
    )

    filepath = pickle_data(results, "data", "super_contour_NP_L_mcz" + str(mcz))


if __name__ == "__main__":
    main()
