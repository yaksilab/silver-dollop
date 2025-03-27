import numpy as np
import matplotlib.pyplot as plt
from scipy.interpolate import splrep, splev
from scipy.optimize import brentq
from astroglial_analysis.my_types import PIdentifier, Intersection, Label, IDRegion
from astroglial_analysis.utils import get_formated_region_coords, rotate_region
from astroglial_analysis.pca import get_pcs


def compute_cumulative_length(points):
    """
    Computes the cumulative arc length for a set of points.

    Parameters:
    - points (ndarray): An array of shape (N, 2) representing the (x, y) coordinates.

    Returns:
    - cumulative_length (ndarray): Cumulative arc length for each point.
    - total_length (float): Total arc length of the curve.
    """
    # Calculate differences between consecutive points
    deltas = np.diff(points, axis=0)
    segment_lengths = np.sqrt((deltas[:, 0]) ** 2 + (deltas[:, 1]) ** 2)
    cumulative_length = np.insert(np.cumsum(segment_lengths), 0, 0)
    total_length = cumulative_length[-1]

    return cumulative_length, total_length


def parametrize_curve(points):
    """
    Parametrizes a 2D curve based on its arc length.

    Parameters:
    - points (ndarray): An array of shape (N, 2) representing the (x, y) coordinates.

    Returns:
    - t_values (ndarray): Parameter values ranging from 0 to total arc length.
    - tck (tuple): Tuple containing spline representations for x(t) and y(t).
    - x_fit (ndarray): Fitted x-coordinates.
    - y_fit (ndarray): Fitted y-coordinates.
    - total_length (float): Total arc length of the curve.
    """
    cumulative_length, total_length = compute_cumulative_length(points)
    u = cumulative_length / total_length  # Normalize to [0, 1]

    # Fit parametric splines for x(t) and y(t)
    spline_x = splrep(u, points[:, 0], s=0, k=1)
    spline_y = splrep(u, points[:, 1], s=0, k=1)

    u_fit = np.linspace(0, 1, int(total_length))
    x_fit = splev(u_fit, spline_x)
    y_fit = splev(u_fit, spline_y)

    t_values = u_fit * total_length

    return t_values, (spline_x, spline_y), x_fit, y_fit, total_length


def find_intersections(tck, t_interval, a, b, tol=1e-2) -> list[Intersection]:
    """
    Finds intersection points between the parameterized curve and the line y = ax + b.

    Parameters:
    - tck (tuple): Tuple containing spline representations for x(t) and y(t).
    - t_interval (ndarray): Array of t values corresponding to the curve. starts at 0 and ends at total_length
    - a (float): Slope of the line.
    - b (float): y-intercept of the line.
    - tol (float): Tolerance for root finding.

    Returns:
    - intersections (list): List of `Intersection` (t, x, y) where the curve intersects the line.
    """
    spline_x, spline_y = tck

    # Defining the difference function f(t) = y(t) - (a * x(t) + b)
    total_length = t_interval[-1]

    if np.isinf(a):
        # For vertical line x = b, find where x(t) - b = 0
        def f(t):
            u = t / total_length
            x = splev(u, spline_x)
            return x - b
    else:
        # For regular line y = ax + b, find where y(t) - (ax(t) + b) = 0
        def f(t):
            u = t / total_length
            x = splev(u, spline_x)
            y = splev(u, spline_y)
            return y - (a * x + b)

    f_samples = f(t_interval)

    # find where the sign of f(t) changes
    sign_changes = np.where(np.diff(np.sign(f_samples)))[0]

    intersections = []
    for idx in sign_changes:
        t1 = t_interval[idx]
        t2 = t_interval[idx + 1]
        try:
            # root in the interval [t1, t2]
            root_t = brentq(f, t1, t2, xtol=tol)
            # Evaluate x and y at root_t
            u = root_t / total_length
            x = splev(u, spline_x)
            y = splev(u, spline_y)
            intersections.append((root_t, x, y))
        except ValueError:
            print(f"No root found in interval [{t1}, {t2}]")
            pass

    return intersections


def get_all_intersections(
    tck, t_samples, p_identifier: PIdentifier, upper: bool, tol=1e-2
) -> list[tuple[Label, Intersection, float]]:
    """
    Finds all intersection points between the parameterized curve defined
    by `tck` and the line y = ax + b for in an interval defined by `total_length`.

    Parameters:
    - tck (tuple): Tuple containing spline representations for x(t) and y(t).
    - total_length (float): Total euclidian length of the curve
    - p_identifier (L,Lindentifier): Identifier for the line that The PC of a process make being `a`, `b`
    and `cm` where c is center of mass of the process. It also contains the label for the process.
    - n_points (int): Number of points to sample the line.
    - tol (float): Tolerance for root finding.

    Returns:
    - intersections_list (list): List of tuples (label, Intersection, distance) where the curve intersects the line.
    """

    intersections_list = []

    for label, (a, b, cm) in p_identifier:
        intersections = find_intersections(tck, t_samples, a, b, tol)

        if not intersections:
            print(f"No intersection found for process {label}")
            continue

        cm_np = np.array(cm)

        if len(intersections) > 1:
            inter_points = np.array([[inter[1], inter[2]] for inter in intersections])

            distances = np.linalg.norm(cm_np - inter_points, axis=1)

            closest_idx = np.argmin(distances)
            closest_inter = intersections[closest_idx]
        else:
            closest_inter = intersections[0]

        diff_vec = cm_np - np.array([closest_inter[1], closest_inter[2]])
        distance = np.linalg.norm(diff_vec)

        change_sign = diff_vec[1]
        if change_sign < 0 and upper:
            continue
        elif change_sign > 0 and not upper:
            continue

        intersections_list.append((label, closest_inter, distance))
    return intersections_list


def uniform_align_processes(
    intersection_list: list[tuple[Label, Intersection, float]], masks, upper: bool
) -> tuple[list[IDRegion], np.ndarray]:
    """
    Aligns the processes based on the intersection points.

    Parameters:
    - intersection_list (list): List of tuples (label, Intersection, distance) where the curve intersects the line.
    - masks (np.ndarray): A labeled mask array.
    - upper (bool): A boolean flag indicating whether to consider the upper part of the region.

    Returns:
    - aligned_processes (list): List of aligned processes.
    - corresponding_matrix (np.ndarray): Corresponding matrix containing the original and aligned coordinates with the label.
    """

    aligned_processes = []
    corresponding_matrix = []
    for label, (t, x, y), d in intersection_list:
        coords = get_formated_region_coords(np.where(masks == label))
        pc, _, _ = get_pcs(coords)
        rotated_coords = rotate_region(pc, coords, upper)
        mean = np.mean(rotated_coords, axis=0)
        translation = np.array([t, d]) - mean
        aligned_coords = rotated_coords + translation

        min_y = np.min(aligned_coords[:, 1])
        if min_y < 0:
            aligned_coords[:, 1] -= min_y

        aligned_processes.append((label, aligned_coords))
        if not upper:
            aligned_coords = np.flip(aligned_coords, axis=0)

        for orig, rot in zip(coords, aligned_coords):
            row = np.array([label, orig[0], orig[1], rot[0], rot[1]])
            corresponding_matrix.append(row)

    corresponding_matrix = np.array(corresponding_matrix, dtype=int)

    return aligned_processes, corresponding_matrix


def plot_parametrized_curve(points, x_fit, y_fit, color, label):
    plt.scatter(points[:, 0], points[:, 1], s=20, color=color, alpha=0.8)
    plt.plot(x_fit, y_fit, color, label=label)
    plt.legend()
