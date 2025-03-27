import numpy as np
import matplotlib.pyplot as plt
from .pca import get_pcs


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


def sub_segment(data_matrix, subsegment_length):
    """
    Adds sub_segment_label and subsegment_number to the data matrix.

    Parameters:
    - data (np.ndarray): Original data matrix with shape (N, 5).
    - subsegment_length (int): Length of each subsegment based on y_rotated.

    Returns:
    - new_data (np.ndarray): Updated data matrix with shape (N, 7).
    """

    cell_labels = data_matrix[:, 0]
    y_rotated = data_matrix[:, 4]

  
    subsegment_number = (y_rotated // subsegment_length) + 1

 
    unique_pairs = np.unique(np.column_stack((cell_labels, subsegment_number)), axis=0)

    # Assign unique sub_segment_labels starting from max(cell_label) + 1
    max_cell_label = cell_labels.max()
    new_labels_start = max_cell_label + 1
    sub_segment_label_map = {
        (pair[0], pair[1]): new_labels_start + idx
        for idx, pair in enumerate(unique_pairs)
    }

    # Map each row to its sub_segment_label
    sub_segment_label = np.array(
        [
            sub_segment_label_map[(cl, sn)]
            for cl, sn in zip(cell_labels, subsegment_number)
        ]
    )

    new_data = np.column_stack(
        (
            cell_labels,
            sub_segment_label,
            subsegment_number,
            data_matrix[
                :, 1:6
            ],  # x_original, y_original, x_rotated, y_rotated, class label
        )
    )

    new_data = new_data.astype(int)

    return new_data
