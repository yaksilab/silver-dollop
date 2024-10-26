import numpy as np
from .utils import get_formated_region_coords
from .sub_segmentation import subsegment_region


def create_subsegmented_mask(
    masks: np.ndarray, labels: list[int], segment_length: int
) -> tuple[np.ndarray, dict[int, dict[int, np.ndarray]]]:
    new_masks = np.zeros_like(masks)
    largest_label = np.max(masks)
    print(f"Largest label: {largest_label}")
    current_label = largest_label + 1
    coords = {}
    relations = {}
    for region_label in labels:
        region = np.where(masks == region_label)
        region_coords = get_formated_region_coords(region)
        subsegments = subsegment_region(region_coords, segment_length)
        sub_seg = {}

        for subsegment in subsegments:
            sub_seg[current_label] = subsegment

            original_coord = get_formated_region_coords(subsegment, revert=True)
            new_masks[original_coord[0], original_coord[1]] = current_label
            relations[region_label].append(current_label)
            current_label += 1

        coords[region_label] = sub_seg
    return new_masks, coords, relations


def create_cp_mask(data_matrix: np.ndarray, masks: np.ndarray) -> np.ndarray:
    new_masks = np.zeros_like(masks)
    cols = data_matrix[:, 3].astype(int)
    rows = data_matrix[:, 4].astype(int)
    labels = data_matrix[:, 1]
    new_masks[rows, cols] = labels
    return new_masks
