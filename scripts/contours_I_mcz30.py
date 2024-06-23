import os
from modules.Classes_ver2 import *
from modules.default_params_ver1 import *
from modules.functions_ver2 import *
from modules.contours_ver1 import *


@timer_decorator
def main():
    print("Number of cores: ", cpu_count())

    # Get the array index from the environment variable
    # idx = int(os.getenv('SLURM_ARRAY_TASK_ID', 0))

    mcz = 30
    lens_params_1["mcz"] = RP_params_1["mcz"] = mcz * solar_mass
    lens_params, RP_params = set_to_params(lens_params_1, RP_params_1)

    y_arr_long = np.linspace(0.5, 3, 200)[::-1]
    # y_arr = np.array_split(y_arr_long, 20)[idx]
    y_arr = y_arr_long

    print("Finished assigning parameters")

    results = create_mismatch_contours_I(RP_params, lens_params, 0.02, y_arr)

    filepath = pickle_data(results, "data", "TACC_contours_mcz30_I")


if __name__ == "__main__":
    main()
