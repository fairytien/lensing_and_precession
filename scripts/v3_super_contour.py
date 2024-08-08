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
    td_arr = np.linspace(0.02, 0.06, 2)  # To be in geometric optics regime
    I_arr = np.linspace(0.1, 0.9, 2)
    print("Finished assigning parameters")

    # Load the RP template grid from the environment variable
    template_grid_path = os.environ.get("TEMPLATE_GRID_PATH")
    if template_grid_path:
        template_grid = np.load(template_grid_path, allow_pickle=True)["template_grid"]
        print("Finished loading RP template grid")
    else:
        print("TEMPLATE_GRID_PATH environment variable is not set.")
        return

    f_cut = get_fcut_from_mcz(lens_params["mcz"])
    f_arr = np.arange(20, f_cut, 0.25)
    psd = Sn(f_arr)
    results = create_super_contour(template_grid, lens_params, td_arr, I_arr, psd=psd)

    # Save results to $HOME directory
    home_dir = os.environ.get("HOME", ".")
    filepath = pickle_data(
        results,
        os.path.join(home_dir, "lensing_and_precession/data"),
        "v3_sys2_super_contour_mcz" + str(mcz),
    )


if __name__ == "__main__":
    main()
