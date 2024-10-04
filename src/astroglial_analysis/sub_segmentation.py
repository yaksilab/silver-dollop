import numpy as np
import matplotlib.pyplot as plt
from pca import get_pcs, get_variance_direction
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA


def subsegment_region(region_coords, segment_length):
    # Get the denormalized principal component
    pc, eigenvalue, covar = get_pcs(region_coords)

    # Project the original region coordinates onto the principal component
    projections = np.dot(region_coords, pc)

    # Determine the number of segments
    min_proj, max_proj = projections.min(), projections.max()
    num_segments = int((max_proj - min_proj) / segment_length)

    # Create subsegments
    subsegments = []
    for i in range(num_segments):
        start = min_proj + i * segment_length
        end = start + segment_length
        mask = (projections >= start) & (projections < end)
        subsegment = region_coords[mask]
        subsegments.append(subsegment)

    return subsegments


def visualize_subsegments(subsegments):
    for subsegment in subsegments:
        plt.scatter(subsegment[:, 0], subsegment[:, 1], s=2)


def subsegment_region_y_axis(region_coords, segment_length):
    # Sort the region coordinates by their y-values
    sorted_coords = region_coords[np.argsort(region_coords[:, 1])]

    # Determine the number of segments
    min_y, max_y = sorted_coords[:, 1].min(), sorted_coords[:, 1].max()
    num_segments = int((max_y - min_y) / segment_length)

    # Create subsegments
    subsegments = []
    for i in range(num_segments):
        start_y = min_y + i * segment_length
        end_y = start_y + segment_length
        mask = (sorted_coords[:, 1] >= start_y) & (sorted_coords[:, 1] < end_y)
        subsegment = sorted_coords[mask]
        subsegments.append(subsegment)

    return subsegments


# p0 = r"tests\data\combined_mean_image_seg.npy"
# masks = np.load(p0, allow_pickle=True).item()["masks"]
# labels = np.unique(masks)
# labels = labels[labels != 0]


# # Example usage
# p0 = r"tests\data\combined_mean_image_seg.npy"
# masks = np.load(p0, allow_pickle=True).item()["masks"]

# classifications, body, processes, body_and_processes = classify_masks(masks)
# upper_lower = upper_lower(body_and_processes, masks)

# region = np.where(masks == upper_lower[0][0])
# region = get_formated_region_coords(region)


# segment_length = 10
# subsegments = subsegment_region(region, segment_length)
# visualize_subsegments(subsegments)
# plt.gca().invert_yaxis()
# plt.show()
