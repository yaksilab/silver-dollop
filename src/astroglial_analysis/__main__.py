from .classifier2 import classify_masks
from .create_masks import create_subsegmented_mask
from astroglial_segmentation import create_suite2p_masks_extract_traces
import numpy as np
import os
import argparse


def main(working_directory):
    # Determine the files to be used for classification
    combined_mean_image_seg_path = os.path.join(
        working_directory, "combined_mean_image_seg.npy"
    )
    mask_file = np.load(combined_mean_image_seg_path, allow_pickle=True)
    new_maskfile = mask_file.copy()
    masks = mask_file.item()["masks"]

    # Classify the masks
    classifications, body, processes, body_and_processes = classify_masks(masks)
    upper, lower = body_and_processes["upper"], body_and_processes["lower"]
    labels = upper + lower + processes

    # Create the new mask array
    segment_length = 10
    new_masks, coords = create_subsegmented_mask(masks, labels, segment_length)
    new_maskfile.item()["masks"] = new_masks

    # Save the new mask array to a .npy file
    output_path = os.path.join(working_directory, "subsegmented_masks_seg.npy")
    np.save(output_path, new_maskfile)
    print(f"New mask array saved to {output_path}")

    # Determine the files to be used for Suite2p mask creation
    subsegmented_masks_seg_path = os.path.join(
        working_directory, "subsegmented_masks_seg.npy"
    )
    create_suite2p_masks_extract_traces(working_directory, subsegmented_masks_seg_path)


if __name__ == "__main__":

    parser = argparse.ArgumentParser(
        description="This pipline takes a directory containing cellpose masks and creates subsegmented masks and then extracts traces from the subsegmented masks using Suite2p"
    )
    parser.add_argument(
        "working_directory",
        type=str,
        help="The directory containing the cellpose '*_seg.npy' mask files to be processed",
    )
    args = parser.parse_args()

    main(args.working_directory)
