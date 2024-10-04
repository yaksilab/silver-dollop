import numpy as np
import matplotlib.pyplot as plt
from .utils import get_formated_region_coords, rotate_region
from .pca import get_pcs
from sklearn.metrics import r2_score
from sklearn.metrics import mean_squared_error

Region = np.ndarray[int]


def custom_insertion_sort_line(line, x_threshold, y_threshold):
    sorted_line = [line[0]]  # Start with the first element
    for i in range(1, len(line)):
        current_item = line[i]
        current_coord = current_item[0]
        inserted = False
        for j in range(len(sorted_line)):
            prev_item = sorted_line[j]
            prev_coord = prev_item[0]
            x_diff = abs(current_coord[0] - prev_coord[0])
            y_diff = abs(current_coord[1] - prev_coord[1])
            if x_diff <= x_threshold and y_diff <= y_threshold:
                # Insert before prev_item
                sorted_line.insert(j, current_item)
                inserted = True
                break
        if not inserted:
            # Append at the end
            sorted_line.append(current_item)
    return sorted_line


def get_cellbody_center(region: Region, upper: bool, body_size: int = 150):
    pc, eigenvalue, covar = get_pcs(region)
    rotated_region = rotate_region(pc, region, upper)

    if upper:
        sorted_indices = np.argsort(rotated_region[:, 1])
        body = rotated_region[sorted_indices[:body_size]]
    else:
        sorted_indices = np.argsort(rotated_region[:, 1])[::-1]
        body = rotated_region[sorted_indices[:body_size]]
    # body = rotate_region(pc, -covar, body)
    return np.mean(body, axis=0), body


def get_line(region_labels, mask_array, upper: bool):
    """
    Determines and returns a sorted line of region labels based on their x-axis values.
    Args:
        region_labels (list): A list of region labels to be processed.
        mask_array (numpy.ndarray): a labeled mask array.
        upper (bool): A boolean flag indicating whether to consider the upper part of the region.
    Returns:
        tuple: A tuple containing:
            - line (list): A list of tuples where each tuple is ((x,y), region_label).
            - body (list): A list of body coordinates for each region.
    """
    line = []
    body = []
    for region_label in region_labels:
        region = np.where(mask_array == region_label)
        region = get_formated_region_coords(region)
        body_center, bod = get_cellbody_center(region, upper)
        line.append(
            (body_center, region_label)
        )  # Append x-axis value instead of entire body
        body.append(bod)

    line.sort(key=lambda x: x[0][0])  # Sort the line based on x-axis value

    return line, body


def remove_outliers(line, coefficients, threshold=2):
    y_pred = np.polyval(coefficients, line[:, 0])

    residuals = line[:, 1] - y_pred

    std_dev = np.std(residuals)

    outliers = np.abs(residuals) > (threshold * std_dev)

    return line[~outliers], line[outliers]


def best_fit_polynomial(line, r_threshold=0.90):
    # Set the range of degrees to try
    degrees = range(1, 10)
    r_scores = []
    mses = []
    best_degree = 0
    best_r_score = -np.inf
    best_mse = np.inf
    best_coefficients = None

    # Loop through each degree
    for degree in degrees:
        # Perform polynomial regression
        coefficients = np.polyfit(line[:, 0], line[:, 1], degree)

        # Predict the y values
        y_pred = np.polyval(coefficients, line[:, 0])

        # Calculate R-squared score and MSE
        r_score = r2_score(line[:, 1], y_pred)
        mse = mean_squared_error(line[:, 1], y_pred)

        # Store the scores
        r_scores.append(r_score)
        mses.append(mse)

        # Check if this is the best degree
        if r_score > best_r_score and mse < best_mse and r_score < r_threshold:
            best_r_score = r_score
            best_mse = mse
            best_degree = degree
            best_coefficients = coefficients
    return best_degree, best_r_score, best_mse, best_coefficients


def align_regions(cleaned_line_label: list, masks, upper: bool):
    distance_shift = 0
    aligned_regions = []
    for i in range(len(cleaned_line_label) - 1):
        # plt.scatter(body[i][:, 0], body[i][:, 1], s=1, color="red")

        region1 = np.where(masks == cleaned_line_label[i][1])
        region1 = get_formated_region_coords(region1)
        pc, eigenvalue, covar = get_pcs(region1)
        # plt.scatter(region1[:, 0], region1[:, 1], s=1)

        region1 = rotate_region(pc, region1, upper)
        # plt.scatter(region1[:, 0], region1[:, 1], s=1)
        if upper:
            min_y1 = np.min(region1[:, 1])
            region1[:, 1] -= int(min_y1)
        else:
            max_y1 = np.max(region1[:, 1])
            region1[:, 1] -= int(max_y1)

        region1[:, 0] += distance_shift
        aligned_regions.append(region1)
        # plt.scatter(region1[:, 0], region1[:, 1], s=1)

        distance = np.sqrt(
            (cleaned_line_label[i + 1][0][0] - cleaned_line_label[i][0][0]) ** 2
            + (cleaned_line_label[i + 1][0][1] - cleaned_line_label[i][0][1]) ** 2
        )

        x_distance = cleaned_line_label[i + 1][0][0] - cleaned_line_label[i][0][0]

        distance_shift += distance - x_distance

        # print(f"Distance {i}: ", distance, x_distance)
    return aligned_regions
