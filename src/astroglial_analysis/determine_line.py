import numpy as np
import matplotlib.pyplot as plt
from .utils import get_formated_region_coords, rotate_region
from .pca import get_pcs
from scipy.interpolate import splprep, splev
from scipy.integrate import quad
from collections import defaultdict

Region = np.ndarray[int]


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


def get_line(
    region_labels, mask_array, upper: bool, delta_x: float = 20
) -> tuple[list, list]:
    """
    Determines and returns a sorted line of region labels based on their x-axis values.
    Args:
        region_labels (list): A list of region labels to be processed.
        mask_array (numpy.ndarray): a labeled mask array.
        upper (bool): A boolean flag indicating whether to consider the upper part of the region.
        delta_x (float): The threshold for considering regions as close on the x-axis.
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

    # initial sort
    line.sort(key=lambda x: x[0][0])  # Sort the line based on x-axis value

    # Group together regions that are close to each other on the x-axis
    # TODO: Maybe better to group based on total distance rather then just x-axis
    sorted_line = []
    current_group = []
    group_start_x = None

    for item in line:
        x, y = item[0]
        if not current_group:
            current_group.append(item)
            group_start_x = x
        elif abs(x - group_start_x) <= delta_x:
            current_group.append(item)
        else:
            # Sort the current group by y-axis before adding to sorted_line
            current_group.sort(
                key=lambda item: item[0][1], reverse=upper
            )  # Descending y
            sorted_line.extend(current_group)
            # Start a new group
            current_group = [item]
            group_start_x = x

    if current_group:
        current_group.sort(key=lambda item: item[0][1], reverse=upper)  # Ascending y
        sorted_line.extend(current_group)

    # if sorted_line:
    #     min_x = sorted_line[0][0][0]
    #     sort_translted = [((x - min_x, y), label) for ((x, y), label) in sorted_line]

    return sorted_line, body


def remove_outliers(line, coefficients, threshold=2):
    y_pred = np.polyval(coefficients, line[:, 0])

    residuals = line[:, 1] - y_pred

    std_dev = np.std(residuals)

    outliers = np.abs(residuals) > (threshold * std_dev)

    return line[~outliers], line[outliers]


def align_regions(cleaned_line_label: list, masks, upper: bool):
    distance_shift = 0
    aligned_regions = []

    distance_shift -= cleaned_line_label[0][0][0]
    for i in range(len(cleaned_line_label) - 1):

        region1 = np.where(masks == cleaned_line_label[i][1])
        region1 = get_formated_region_coords(region1)
        pc, _, _ = get_pcs(region1)

        region1 = rotate_region(pc, region1, upper)
        if upper:
            min_y1 = np.min(region1[:, 1])
            region1[:, 1] -= int(min_y1)
        else:
            max_y1 = np.max(region1[:, 1])
            region1[:, 1] -= int(max_y1)
            region1[:, 1] = -region1[:, 1]

        region1[:, 0] += distance_shift
        aligned_regions.append(region1)

        distance = np.sqrt(
            (cleaned_line_label[i + 1][0][0] - cleaned_line_label[i][0][0]) ** 2
            + (cleaned_line_label[i + 1][0][1] - cleaned_line_label[i][0][1]) ** 2
        )

        x_distance = cleaned_line_label[i + 1][0][0] - cleaned_line_label[i][0][0]

        distance_shift += distance - x_distance

    last_region = np.where(masks == cleaned_line_label[-1][1])
    last_region = get_formated_region_coords(last_region)
    pc, eigenvalue, covar = get_pcs(last_region)
    last_region = rotate_region(pc, last_region, upper)
    if upper:
        min_y1 = np.min(last_region[:, 1])
        last_region[:, 1] -= int(min_y1)
    else:
        max_y1 = np.max(last_region[:, 1])
        last_region[:, 1] -= int(max_y1)
        last_region[:, 1] = -last_region[:, 1]
    last_region[:, 0] += distance_shift
    aligned_regions.append(last_region)

    return aligned_regions
