import numpy as np
import matplotlib.pyplot as plt
from .utils import get_formated_region_coords, rotate_region
from .pca import get_pcs
from .my_types import Region, ParamCurveLine, IDRegion


def get_cellbody_center(region: Region, upper: bool, body_size: int = 150):
    pc, _, _ = get_pcs(region)
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
) -> tuple[ParamCurveLine, list]:
    """
    Determines and returns a sorted line of region labels based on their x-axis values.
    Args:
        region_labels (list): A list of region labels to be processed.
        mask_array (numpy.ndarray): a labeled mask array.
        upper (bool): A boolean flag indicating whether to consider the upper part of the region.
        delta_x (float): The threshold for considering regions as close on the x-axis.
    Returns:
        tuple: A tuple containing:
            - param_curve_line (ParamCurveLine)
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


def uniform_align_comp_cell(
    param_curve: ParamCurveLine, masks, upper: bool
) -> list[IDRegion]:
    distance_shift = 0
    aligned_regions = []
    corresponding_matrix = []

    distance_shift -= param_curve[0][0][0]
    for i in range(len(param_curve) - 1):
        label = param_curve[i][1]
        org_coords = get_formated_region_coords(np.where(masks == label))
        pc, _, _ = get_pcs(org_coords)

        aligned_coords = rotate_region(pc, org_coords, upper)
        if upper:
            min_y1 = np.min(aligned_coords[:, 1])
            aligned_coords[:, 1] -= int(min_y1)
        else:
            max_y1 = np.max(aligned_coords[:, 1])
            aligned_coords[:, 1] -= int(max_y1)
            aligned_coords[:, 1] = -aligned_coords[:, 1]

        aligned_coords[:, 0] += distance_shift
        aligned_regions.append((label, aligned_coords))
        for orig, rot in zip(org_coords, aligned_coords):
            row = np.array([label, orig[0], orig[1], rot[0], rot[1]])
            corresponding_matrix.append(row)

        distance = np.sqrt(
            (param_curve[i + 1][0][0] - param_curve[i][0][0]) ** 2
            + (param_curve[i + 1][0][1] - param_curve[i][0][1]) ** 2
        )

        x_distance = param_curve[i + 1][0][0] - param_curve[i][0][0]

        distance_shift += distance - x_distance

    last_label = param_curve[-1][1]
    last_region_coords = np.where(masks == last_label)
    last_region_coords = get_formated_region_coords(last_region_coords)
    pc, _, _ = get_pcs(last_region_coords)
    aligned_coords = rotate_region(pc, aligned_coords, upper)
    if upper:
        min_y1 = np.min(aligned_coords[:, 1])
        aligned_coords[:, 1] -= int(min_y1)
    else:
        max_y1 = np.max(aligned_coords[:, 1])
        aligned_coords[:, 1] -= int(max_y1)
        aligned_coords[:, 1] = -aligned_coords[:, 1]
    aligned_coords[:, 0] += distance_shift

    for orig, rot in zip(last_region_coords, aligned_coords):
        row = np.array([last_label, orig[0], orig[1], rot[0], rot[1]])
        corresponding_matrix.append(row)

    corresponding_matrix = np.array(corresponding_matrix, dtype=int)

    aligned_regions.append((last_label, aligned_coords))

    return aligned_regions, corresponding_matrix
