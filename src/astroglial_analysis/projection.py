import numpy as np
import matplotlib.pyplot as plt
from utils import get_formated_region_coords
from pca import get_pcs, get_variance_direction


def sort_regions_by_centroid_and_xy_position(region_labels, mask_array):
    """
    Sorts regions by centroid and xy position from left to right and top to bottom.
    if two regions have same cen

    Args:
    region_labels (list): List of region labels
    mask_array (np.ndarray): Mask array

    Returns:
    list: List of sorted regions

    """

    regions = []
    for label in region_labels:
        region = np.where(mask_array == label)
        region = get_formated_region_coords(region)
        centroid = np.mean(region, axis=0)
        regions.append((label, centroid))

    sorted_region_labels = sorted(regions, key=lambda item: (item[1][0], item[1][1]))

    return sorted_region_labels


def calculate_sorted_distance_list(region_labels, mask_array):

    sorted_regions = sort_regions_by_centroid_and_xy_position(region_labels, mask_array)

    distances = []

    for i in range(len(sorted_regions) - 1):
        distance = np.linalg.norm(sorted_regions[i][1] - sorted_regions[i + 1][1])
        distances.append(distance)

    return sorted_regions, distances


def project_masks_on_line(region_labels, mask_array):
    """
    moves every region on x-axis where centroid of the region has (x,0) coordinates
    and the next region in the sorted list is at some defined distance from the previous region

    Args:
    region_labels (list): List of region labels
    mask_array (np.ndarray): Mask array
    line_direction (np.ndarray): Direction of the line

    Returns:
    list: List of projected regions

    """

    sorted_regions, distances = calculate_sorted_distance_list(
        region_labels, mask_array
    )

    print(len(sorted_regions))
    distance_shift = 0
    for i in range(len(sorted_regions) - 1):
        region = get_formated_region_coords(
            np.where(mask_array == sorted_regions[i][0])
        )

        centroid = sorted_regions[i][1]
        min_y = np.max(region[:, 1])
        distance_shift += distances[i]
        region[:, 0] += int(distance_shift)
        region[:, 1] -= int(min_y)

        plt.scatter(region[:, 0], region[:, 1], s=1)


def upper_lower(region_labels, mask_array):
    upper = []
    lower = []
    print(mask_array.shape[0] // 2)
    for label in region_labels:
        region = np.where(mask_array == label)
        region = get_formated_region_coords(region)
        centroid = np.mean(region, axis=0)

        if centroid[1] < mask_array.shape[0] // 2:
            upper.append(label)
        else:
            lower.append(label)

    return upper, lower


# p0 = r"tests\data\combined_mean_image_seg.npy"

# masks = np.load(p0, allow_pickle=True).item()["masks"]

# classifications, body, processes, body_and_processes = classify_masks(masks)

# processes_plus_body_and_processes = np.concatenate((processes, body_and_processes))

# upper, lower = upper_lower(body_and_processes, masks)


# project_masks_on_line(upper, masks)
# plt.gca().invert_yaxis()
# plt.show()
