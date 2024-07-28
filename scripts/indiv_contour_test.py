import sys, os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from modules.contours_ver2 import *
from modules.default_params_ver1 import *


@timer_decorator
def main():
    print("Number of cores: ", cpu_count())

    # Assign parameters
    lens_params, RP_params = set_to_params(lens_params_1, RP_params_1)
    mcz = 40
    lens_params["mcz"] = RP_params["mcz"] = mcz * solar_mass
    I = 0.607
    td = 0.0198
    y = get_y_from_I(I)
    MLz = get_MLz_from_td(td, y)
    lens_params["y"] = y
    lens_params["MLz"] = MLz * solar_mass
    print("Finished assigning parameters")

    results = create_mismatch_contour_parallel(RP_params, lens_params)

    filepath = pickle_data(results, "data", "null_indiv_contour_mcz" + str(mcz))


if __name__ == "__main__":
    main()
