import sys, os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from modules.contours_ver3 import *


@timer_decorator
def main():
    print("Number of cores: ", cpu_count())

    # Assign parameters
    lens_params = set_to_location(loc_params["Taman"]["edgeon"], lens_params_1)[0]
    # Get the mcz value from the environment variable
    mcz = int(os.environ.get("MCZ_VALUE", 40))  # Default to 40 if not set
    lens_params["mcz"] = mcz * solar_mass
    I = 0.5
    td = 0.03
    y = get_y_from_I(I)
    MLz = get_MLz_from_td(td, y)
    lens_params["y"] = y
    lens_params["MLz"] = MLz * solar_mass
    print("Finished assigning parameters")

    # Load the RP template grid from the environment variable
    template_grid_path = os.environ.get("TEMPLATE_GRID_PATH")
    if template_grid_path:
        template_grid = np.load(template_grid_path, allow_pickle=True)["template_grid"]
        print("Finished loading RP template grid")
    else:
        print("TEMPLATE_GRID_PATH environment variable is not set.")
        return

    results = create_mismatch_contour(template_grid, lens_params)

    # Save results to $HOME directory
    home_dir = os.environ.get("HOME", ".")
    filepath = pickle_data(
        results,
        os.path.join(home_dir, "lensing_and_precession/data"),
        "v3_sys2_indiv_contour_mcz" + str(mcz),
    )


if __name__ == "__main__":
    main()
