from modules.Classes_ver2 import *
from modules.default_params_ver1 import *
from modules.functions_ver2 import *
from modules.contours_ver2 import *


@timer_decorator
def main():
    print("Number of cores: ", cpu_count())

    mcz = 40
    lens_params_1["mcz"] = RP_params_1["mcz"] = mcz * solar_mass
    lens_params, RP_params = set_to_params(lens_params_1, RP_params_1)

    results = mismatch_contour_parallel(RP_params, lens_params)

    filepath = pickle_data(results, "data", "TACC_indiv_contour_mcz" + str(mcz))


if __name__ == "__main__":
    main()
