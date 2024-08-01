import sys, os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from modules.contours_ver3 import *


@timer_decorator
def main():
    print("Number of cores: ", cpu_count())

    # Get the array index from the environment variable
    idx = int(os.getenv("SLURM_ARRAY_TASK_ID", 0))

    # Assign parameters
    lens_params = set_to_location(loc_params["Taman"]["random"], lens_params_1)[0]
    mcz = 40
    lens_params["mcz"] = mcz * solar_mass

    td_arr_long = np.linspace(0.02, 0.07, 40)  # To be in geometric optics regime
    td_arr = np.array_split(td_arr_long, 1)[idx]
    I_arr = np.linspace(0.1, 0.9, 40)

    # Load the RP template bank
    filepath = "data/sys3_template_bank_mcz" + str(mcz)
    with open(filepath, "rb") as f:
        RP_template_bank = pickle.load(f)

    print("Finished assigning parameters")

    results = create_super_contour(RP_template_bank, lens_params, td_arr, I_arr)

    filepath = pickle_data(
        results, "data", "v3_sys3_super_contour_mcz" + str(mcz) + "_" + str(idx)
    )


if __name__ == "__main__":
    main()
