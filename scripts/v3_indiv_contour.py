import sys, os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from modules.contours_ver3 import *


@timer_decorator
def main():
    print("Number of cores: ", cpu_count())

    # Assign parameters
    lens_params = set_to_location(loc_params["Taman"]["random"], lens_params_1)[0]
    mcz = 20
    lens_params["mcz"] = mcz * solar_mass
    I = 0.5
    td = 0.03
    y = get_y_from_I(I)
    MLz = get_MLz_from_td(td, y)
    lens_params["y"] = y
    lens_params["MLz"] = MLz * solar_mass

    # Load the RP template bank
    filepath = "data/sys3_template_bank_mcz" + str(mcz)
    with open(filepath, "rb") as f:
        RP_template_bank = pickle.load(f)

    print("Finished assigning parameters")

    results = create_mismatch_contour(RP_template_bank, lens_params)

    filepath = pickle_data(results, "data", "v3_sys3_indiv_contour_mcz" + str(mcz))


if __name__ == "__main__":
    main()
