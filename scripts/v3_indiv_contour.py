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
    I = 0.5
    td = 0.03
    y = get_y_from_I(I)
    MLz = get_MLz_from_td(td, y)
    lens_params["y"] = y
    lens_params["MLz"] = MLz * solar_mass
    print("Finished assigning parameters")

    # Load the RP template grid
    filepath = "data/sys2_template_grid_mcz" + str(mcz) + ".npz"
    RP_template_grid = np.load(filepath, allow_pickle=True)["template_grid"]
    print("Finished loading RP template grid")

    results = create_mismatch_contour(RP_template_grid, lens_params)

    filepath = pickle_data(results, "data", "v3_sys2_indiv_contour_mcz" + str(mcz))


if __name__ == "__main__":
    main()
