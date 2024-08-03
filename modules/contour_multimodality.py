#############################
# Section 1: Import Modules #
#############################


# import py scripts
from modules.contours_ver2 import *

# import libraries
from scipy.optimize import minimize
from scipy.interpolate import RegularGridInterpolator


######################################
# Section 2: Finding Multiple Minima #
######################################


def is_near(
    point1: Union[tuple, list, np.ndarray],
    point2: Union[tuple, list, np.ndarray],
    threshold=0.5,
):
    """Checks if point1 is near point2 within a given threshold.

    Args:
        point1 (Union[tuple, list, np.ndarray]): The first point.
        point2 (Union[tuple, list, np.ndarray]): The second point.
        threshold (float, optional): The threshold value for determining nearness. Defaults to 0.5.

    Returns:
        bool: True if the points are near each other, False otherwise.
    """
    distance = np.linalg.norm(np.array(point1) - np.array(point2))
    return distance < threshold


def filter_near_duplicates(results, threshold=0.5):
    """
    Filters out near-duplicate results based on a given threshold.

    Args:
        results (list): A list of tuples, each containing an array of coordinates and a corresponding value.
        threshold (float, optional): The threshold value for determining near-duplicates. Defaults to 0.5.

    Returns:
        list: A filtered list of tuples containing non-duplicate coordinates and values.
    """
    filtered = []
    for coord, z in results:
        # Convert coordinates to a hashable type
        coord_tuple = tuple(coord)
        if not any(
            is_near(coord_tuple, tuple(existing_coord), threshold)
            for existing_coord, _ in filtered
        ):
            filtered.append((coord, z))
    return filtered


def find_local_minima(
    Z: np.ndarray,
    x=np.linspace(0, 4, 41),
    y=np.linspace(0, 15, 151),
    print_results=True,
) -> list:
    """Finds local minima in a 2D dataset.

    Parameters
    ----------
    Z : np.ndarray
        The 2D dataset to analyze.
    x : np.ndarray, optional
        The x-coordinates of the dataset. Defaults to np.linspace(0, 4, 41).
    y : np.ndarray, optional
        The y-coordinates of the dataset. Defaults to np.linspace(0, 15, 151).

    Returns
    -------
    list
        A list of tuples, each containing the coordinates of a local minimum and the corresponding value.
    """
    # Interpolate the dataset
    Z = Z.T  # Transpose the matrix to match the x and y dimensions
    interpolator = RegularGridInterpolator((x, y), Z)

    # Define the objective function using the interpolator
    def objective_function(xy: Union[tuple, np.ndarray]) -> float:
        """Objective function to minimize.

        Args:
            xy (Union[tuple, np.ndarray]): The coordinates to evaluate.

        Returns:
            float: The value of the objective function at the given coordinates.
        """
        # Check if the point is within the bounds
        if xy[0] < x[0] or xy[0] > x[-1] or xy[1] < y[0] or xy[1] > y[-1]:
            return np.inf  # Return a high value to penalize out-of-bounds points
        else:
            # RegularGridInterpolator expects a tuple or an array of coordinates
            return interpolator(xy)

    # Define multiple starting points
    starting_points = [
        (x[0], y[0]),  # Bottom-left corner
        (x[-1], y[0]),  # Bottom-right corner
        (x[0], y[-1]),  # Top-left corner
        (x[-1], y[-1]),  # Top-right corner
        (x[len(x) // 2], y[len(y) // 2]),  # Center
        (x[0], y[len(y) // 2]),  # Left-center
        (x[-1], y[len(y) // 2]),  # Right-center
        (x[len(x) // 2], y[0]),  # Bottom-center
        (x[len(x) // 2], y[-1]),  # Top-center
    ]

    results = []

    for point in starting_points:
        result = minimize(objective_function, point, method="Nelder-Mead")
        results.append((result.x, result.fun))

    # Round the coordinates to a given precision
    rounded_results = [(np.round(coord, 1), z) for coord, z in results]
    filtered_results = filter_near_duplicates(rounded_results, threshold=0.5)

    # Discard coords with z values greater than the mean of Z
    Z_mean = np.mean(Z)
    filtered_results = [(coord, z) for coord, z in filtered_results if z < Z_mean]

    # Print or process the results
    if print_results:
        for coord, z in filtered_results:
            print(f"Local minimum at {coord}: {z}")

    return filtered_results


def find_local_minima_gradient_descent(
    Z: np.ndarray,
    x=np.linspace(0, 4, 41),
    y=np.linspace(0, 15, 151),
    print_results=True,
) -> list:
    # Interpolate the dataset
    Z = Z.T  # Transpose the matrix to match the x and y dimensions
    interpolator = RegularGridInterpolator((x, y), Z)

    # Define the objective function using the interpolator
    def objective_function(xy: Union[tuple, np.ndarray]) -> float:
        # Check if the point is within the bounds
        if xy[0] < x[0] or xy[0] > x[-1] or xy[1] < y[0] or xy[1] > y[-1]:
            return np.inf  # Return a high value to penalize out-of-bounds points
        else:
            # RegularGridInterpolator expects a tuple or an array of coordinates
            return interpolator(xy)

    # Define multiple starting points
    starting_points = [
        (x[0], y[0]),  # Bottom-left corner
        (x[-1], y[0]),  # Bottom-right corner
        (x[0], y[-1]),  # Top-left corner
        (x[-1], y[-1]),  # Top-right corner
        (x[len(x) // 2], y[len(y) // 2]),  # Center
        (x[0], y[len(y) // 2]),  # Left-center
        (x[-1], y[len(y) // 2]),  # Right-center
        (x[len(x) // 2], y[0]),  # Bottom-center
        (x[len(x) // 2], y[-1]),  # Top-center
    ]

    def approximate_gradient(f, xy, bounds, h=1e-5):
        grad = np.zeros_like(xy)
        for i in range(len(xy)):
            x_plus_h = np.array(xy)
            x_minus_h = np.array(xy)
            x_plus_h[i] = min(
                x_plus_h[i] + h, bounds[i][1]
            )  # Ensure x_plus_h is within upper bound
            x_minus_h[i] = max(
                x_minus_h[i] - h, bounds[i][0]
            )  # Ensure x_minus_h is within lower bound
            f_plus_h = f(x_plus_h)
            f_minus_h = f(x_minus_h)
            grad[i] = (f_plus_h - f_minus_h) / (
                2 * h if x_plus_h[i] != x_minus_h[i] else h
            )
        return grad

    def gradient_descent(
        f, grad_approx, start, bounds, learning_rate=0.1, max_iter=100, tol=1e-6
    ):
        point = np.array(start)
        for _ in range(max_iter):
            gradient = grad_approx(f, point, bounds)
            next_point = point - learning_rate * gradient
            # Ensure next_point stays within bounds
            next_point = np.clip(
                next_point, [b[0] for b in bounds], [b[1] for b in bounds]
            )
            if np.linalg.norm(f(next_point) - f(point)) < tol:
                break
            point = next_point
        return point, f(point)

    # Define bounds for each dimension, e.g., [(min_x, max_x), (min_y, max_y)]
    bounds = [(x[0], x[-1]), (y[0], y[-1])]

    # Modify the calls to gradient_descent to include bounds
    results = []

    for point in starting_points:
        result_point, result_fun = gradient_descent(
            objective_function, approximate_gradient, point, bounds
        )
        results.append((result_point, result_fun))

    # Round and filter results as before
    rounded_results = [(np.round(coord, 1), z) for coord, z in results]
    filtered_results = filter_near_duplicates(rounded_results, threshold=0.5)

    if print_results:
        for coord, z in filtered_results:
            print(f"Local minimum at {coord}: {z}")

    return filtered_results


#######################################
# Section 3: Tracking Multiple Minima #
#######################################


def track_minima(
    results: dict,
    Z: np.ndarray,
    td: float,
    I: float,
    step: int,
    z_thres: float,
    dist_thres=1.0,
) -> dict:
    """Tracks multiple minima in 2D datasets over successive steps.

    Args:
        results (dict): A dictionary to store the tracked minima.
        Z (np.ndarray): The 2D dataset to analyze.
        td (float): The time delay corresponding to the dataset.
        I (float): The flux ratio corresponding to the dataset.
        step (int): The current step in the tracking process.
        z_thres (float): The threshold value for determining minima.
        dist_thres (float, optional): The threshold distance for matching minima. Defaults to 0.6.

    Returns:
        dict: The updated dictionary containing the tracked minima.
    """

    for coord, z in find_local_minima(Z, print_results=False):
        best_match_key = None
        best_match_distance = float("inf")

        for key in results.keys():
            distance = np.linalg.norm(coord - results[key]["coord_arr"][-1])
            steps_since_last = step - results[key]["step_arr"][-1]
            if distance < best_match_distance and z < z_thres and steps_since_last < 5:
                best_match_key = key
                best_match_distance = distance

        if best_match_key is not None and best_match_distance < dist_thres:
            results[best_match_key]["coord_arr"] = np.append(
                results[best_match_key]["coord_arr"], [coord], axis=0
            )
            results[best_match_key]["td_arr"] = np.append(
                results[best_match_key]["td_arr"], td
            )
            results[best_match_key]["I_arr"] = np.append(
                results[best_match_key]["I_arr"], I
            )
            results[best_match_key]["step_arr"] = np.append(
                results[best_match_key]["step_arr"], step
            )
            results[best_match_key]["ep_arr"] = np.append(
                results[best_match_key]["ep_arr"], z
            )
        else:
            results[len(results)] = {
                "coord_arr": np.array([coord]),
                "td_arr": np.array([td]),
                "I_arr": np.array([I]),
                "step_arr": np.array([step]),
                "ep_arr": np.array([z]),
            }

    return results
