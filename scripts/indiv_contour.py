import sys

sys.path.insert(
    0, "/Users/fairytien/Documents/TEXAS_Bridge_2324/code/lensing_and_precession/"
)
from modules.Classes_ver2 import *
from modules.default_params_ver2 import *
from modules.functions_ver2 import *
from modules.contours_ver2 import *


@timer_decorator
def main():
    print("Number of cores: ", cpu_count())

    lens_params, RP_params = set_to_location(
        loc_params["Taman"]["random"], lens_params_1, RP_params_1
    )
    mcz = 40
    lens_params["mcz"] = RP_params["mcz"] = mcz * solar_mass
    lens_params["MLz"] = 2e3 * solar_mass

    results = mismatch_contour_parallel(RP_params, lens_params)

    filepath = pickle_data(results, "data", "indiv_contour_mcz" + str(mcz))


if __name__ == "__main__":
    main()
