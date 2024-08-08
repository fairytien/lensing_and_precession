import sys, os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from modules.contours_ver3 import *


@timer_decorator
def main():
    print("Number of cores: ", cpu_count())

    # Assign parameters
    lens_params = set_to_location(loc_params["Taman"]["edgeon"], lens_params_1)[0]
    mcz = 40
    lens_params["mcz"] = mcz * solar_mass
    td_arr = np.linspace(0.02, 0.07, 5)  # To be in geometric optics regime
    I_arr = np.linspace(0.1, 0.9, 5)
    print("Finished assigning parameters")

    # Load the RP template grid
    filepath = "data/sys2_template_grid_mcz" + str(mcz) + ".npz"
    RP_template_grid = np.load(filepath, allow_pickle=True)["template_grid"]
    print("Finished loading RP template grid")

    results = create_super_contour(RP_template_grid, lens_params, td_arr, I_arr)

    filepath = pickle_data(results, "data", "v3_sys2_super_contour_mcz" + str(mcz))


if __name__ == "__main__":
    main()
