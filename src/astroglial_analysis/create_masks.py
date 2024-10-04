import numpy as np
from utils import get_formated_region_coords
from sub_segmentation import subsegment_region
from classifier2 import classify_masks

# Load the masks
p0 = r"tests\data\combined_mean_image1_seg.npy"
masks = np.load(p0, allow_pickle=True).item()["masks"]
mask_file = np.load(p0, allow_pickle=True)
new_maskfile = mask_file.copy()
# Classify the masks
classifications, body, processes, body_and_processes = classify_masks(masks)
upper, lower = body_and_processes["upper"], body_and_processes["lower"]


def create_subsegmented_mask(
    masks: np.ndarray, labels: list[int], segment_length: int
) -> tuple[np.ndarray, dict[int, dict[int, np.ndarray]]]:
    new_masks = np.zeros_like(masks)
    largest_label = np.max(masks)
    print(f"Largest label: {largest_label}")
    current_label = largest_label + 1
    coords = {}
    for region_label in labels:
        region = np.where(masks == region_label)
        region_coords = get_formated_region_coords(region)
        subsegments = subsegment_region(region_coords, segment_length)
        sub_seg = {}

        for subsegment in subsegments:
            sub_seg[current_label] = subsegment

            original_coord = get_formated_region_coords(subsegment, revert=True)
            new_masks[original_coord[0], original_coord[1]] = current_label
            current_label += 1
        coords[region_label] = sub_seg
    return new_masks, coords


# # Create the new mask array
# segment_length = 10
# new_masks, coords = create_subsegmented_mask(masks, upper, lower, segment_length)
# new_maskfile.item()["masks"] = new_masks

# print(coords.keys())
# # print(coords[upper[0]].keys())
# # print(coords[upper[0]][list(coords[upper[0]].keys())[2]].shape)
# print(coords[1][269])

# # # Save the new mask array to a .npy file
# # output_path = r"tests\data\subsegmented_masks.npy"
# # np.save(output_path, new_maskfile)
# # print(f"New mask array saved to {output_path}")
