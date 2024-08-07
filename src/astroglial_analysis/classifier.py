import numpy as np
from scipy.ndimage import label
from skimage.measure import regionprops, label as skimage_label
import matplotlib.pyplot as plt


p0 = r"C:\YaksiData\astrolglialAnalysis\tests\data\combined_mean_image_seg.npy"
p1 = r"C:\YaksiData\astrolglialAnalysis\tests\data\CP2_s1_039189_mean_image_seg.npy"
p2 = r"C:\YaksiData\astrolglialAnalysis\tests\data\CP2_s3_039234_mean_image_seg.npy"

masks = np.load(p0, allow_pickle=True).item()["masks"]


unique_labels = np.unique(masks)

print(unique_labels)
print(len(unique_labels))


# Function to calculate elongation and center of mass
def calculate_elongation_center_of_mass(mask):
    labeled_mask = masks
    regions = regionprops(labeled_mask)

    features = []
    for region in regions:

        elongation = region.eccentricity

        center = region.centroid

        minr, minc, maxr, maxc = region.bbox
        geometric_center = ((minr + maxr) / 2, (minc + maxc) / 2)

        max_dist = np.linalg.norm([maxr - minr, maxc - minc]) / 2
        center_shift = (
            np.linalg.norm(np.array(center) - np.array(geometric_center)) / max_dist
        )

        features.append((elongation, center_shift, labeled_mask == region.label))

    return features


# Function to classify masks
def classify_masks(masks):
    classifications = []
    features = calculate_elongation_center_of_mass(masks)
    label = 1
    body = []
    processes = []
    body_and_processes = []

    for elongation, center_shift, mask in features:
        # Define thresholds for classification
        elongation_threshold = 0.85
        shift_threshold = 0.1

        if elongation < elongation_threshold:
            classification = 1
            body.append(label)
            label += 1
        elif center_shift < shift_threshold:
            classification = 2
            processes.append(label)
            label += 1
        else:
            classification = 3
            body_and_processes.append(label)
            label += 1

        classifications.append((classification, mask))

    return classifications, body, processes, body_and_processes


# Apply the classifier
classifications = classify_masks(masks)

# Create an empty RGB image to visualize the masks with different colors
color_map = {
    1: [1, 0, 0],  # Red
    2: [0, 0, 1],  # Blue
    3: [0, 1, 0],  # Green
}

colored_image = np.zeros((*masks.shape, 3))

# for classification, mask in classifications:
#     color = color_map[classification]
#     for i in range(3):
#         colored_image[:, :, i] += mask * color[i]

# # Ensure the colors do not exceed the range [0, 1]
# colored_image = np.clip(colored_image, 0, 1)

# # Plot the colored masks
# plt.figure(figsize=(10, 10))
# plt.imshow(colored_image)
# plt.title("Cell Mask Classification")
# plt.axis("off")

# # Add legend
# import matplotlib.patches as mpatches

# red_patch = mpatches.Patch(color="red", label="Cell Body Only")
# blue_patch = mpatches.Patch(color="blue", label="Orphan Process")
# green_patch = mpatches.Patch(color="green", label="Cell Body with Process")
# plt.legend(handles=[red_patch, blue_patch, green_patch], loc="lower right")

# plt.show()
