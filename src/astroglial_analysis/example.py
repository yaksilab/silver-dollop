import numpy as np
import matplotlib.pyplot as plt
from scipy.ndimage import label
from skimage.measure import regionprops
import random


p0 = r"C:\YaksiData\astrolglialAnalysis\tests\data\combined_mean_image_seg.npy"
p1 = r"C:\YaksiData\astrolglialAnalysis\tests\data\CP2_s1_039189_mean_image_seg.npy"
p2 = r"C:\YaksiData\astrolglialAnalysis\tests\data\CP2_s3_039234_mean_image_seg.npy"

masks = np.load(p0, allow_pickle=True).item()["masks"]
print(masks.shape)
print(len(np.unique(masks)))


def transform_and_plot_masks(mask_array, num_masks=5, x_shift_const=50):
    # Extract unique labels (excluding background which is usually 0)
    unique_labels = np.unique(mask_array)
    unique_labels = unique_labels[unique_labels != 0]

    # Randomly select num_masks labels to plot
    selected_labels = random.sample(list(unique_labels), num_masks)

    # Calculate the middle line of the array
    middle_line = mask_array.shape[0] // 2

    # Initialize a list to store transformed coordinates
    transformed_coords = []

    for label_val in selected_labels:
        # Extract coordinates for the current mask
        coords = np.column_stack(np.where(mask_array == label_val))

        # Identify the centroid of the cell body
        centroid = np.mean(coords, axis=0)
        body_coords = coords[
            np.linalg.norm(coords - centroid, axis=1) < 20
        ]  # Assuming cell body within 20 pixels

        if body_coords.size == 0:
            continue

        # Determine if the mask should be above or below the x-axis
        if centroid[0] < middle_line:
            # Cell is above the middle line, shift it above the x-axis
            y_shift = coords[:, 0].min()  # Shift by the minimum y value (foot)
            relative_y_shift = centroid[0] - y_shift
        else:
            # Cell is below the middle line, shift it below the x-axis
            y_shift = coords[:, 0].max()  # Shift by the maximum y value (foot)
            relative_y_shift = centroid[0] - y_shift
        # Transform coordinates
        transformed = coords.astype(
            float
        ).copy()  # Convert to float to avoid casting issues
        transformed[:, 0] -= y_shift

        # Adjust y coordinates to maintain relative position and attach to x-axis
        if centroid[0] < middle_line:
            transformed[:, 0] += relative_y_shift  # Maintain relative y-position
        else:
            transformed[:, 0] += relative_y_shift  # Maintain relative y-position

        transformed[:, 1] += x_shift_const

        transformed_coords.append(transformed)

    # Plot the transformed masks
    plt.figure(figsize=(12, 8))
    for coords in transformed_coords:
        plt.scatter(coords[:, 1], coords[:, 0], s=1)

    plt.axhline(0, color="red", linestyle="--")
    plt.gca().invert_yaxis()
    plt.title("Transformed Masks of Astroglial Cells")
    plt.xlabel("X-axis")
    plt.ylabel("Y-axis")
    plt.show()


# Example usage:
# Assuming 'mask_array' is your 2D numpy array containing segmented masks
mask_array = masks  # Load your mask array
transform_and_plot_masks(mask_array, num_masks=272, x_shift_const=50)
