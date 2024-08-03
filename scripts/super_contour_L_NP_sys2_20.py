import sys, os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from modules.contours_ver2 import *


@timer_decorator
def main():
    print("Number of cores: ", cpu_count())

    # Assign parameters
    lens_params, RP_params, NP_params = set_to_location(
        loc_params["Taman"]["edgeon"], lens_params_1, RP_params_1, NP_params_1
    )
    mcz = 20
    lens_params["mcz"] = RP_params["mcz"] = NP_params["mcz"] = mcz * solar_mass
    td_arr = np.linspace(0.02, 0.07, 40)  # To be in geometric optics regime
    I_arr = np.linspace(0.1, 0.9, 40)
    print("Finished assigning parameters")

    results = create_super_contour(
        NP_params, lens_params, td_arr, I_arr, what_template="NP"
    )

    filepath = pickle_data(results, "data", "sys2_super_contour_L_NP_mcz" + str(mcz))


if __name__ == "__main__":
    main()
