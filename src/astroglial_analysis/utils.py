import numpy as np
import os
from astroglial_analysis.my_types import Features, Labels


def rotate_region(pc, region_coords, up: bool):
    x, y = pc[0], pc[1]

    # Determine the rotation angle
    if up:
        angle_to_y = -np.arctan(x / y)
        # Find the lowest point (lowest y-coordinate)
        rotation_point = region_coords[np.argmin(region_coords[:, 1])]
    else:
        angle_to_y = -np.arctan(x / y)
        # Find the highest point (highest y-coordinate)
        rotation_point = region_coords[np.argmax(region_coords[:, 1])]

    rotation_angle = angle_to_y

    # Subtract the rotation point so that the rotation happens around this point
    region_coords = region_coords - rotation_point

    # Create the rotation matrix
    rotation_matrix = np.array(
        [
            [np.cos(rotation_angle), -np.sin(rotation_angle)],
            [np.sin(rotation_angle), np.cos(rotation_angle)],
        ]
    )

    # Rotate the region
    rotated_coords = np.dot(region_coords, rotation_matrix)

    # Add back the rotation point to return the coordinates to their original position
    return rotated_coords + rotation_point


def get_formated_region_coords(region_coords, revert=False):
    if revert:
        region_coords = np.flip(region_coords, axis=1)
        region_coords = np.transpose(region_coords)
    else:
        region_coords = np.transpose(region_coords)
        region_coords = np.flip(region_coords, axis=1)
    return region_coords


def extract_features(masks, classifications):
    """
    Extracts features from masks and classifications, including:
    - Area
    - Centroid (x, y)
    - Bounding box width and height
    - Eccentricity (shape measure)
    - Directions of the first and second principal components (PC1 and PC2)
    - Magnitude of the center-of-mass shift from the geometric center
    - Spatial context features (relative position, neighbor counts, etc.)

    Args:
        masks (np.ndarray): A 2D array where each pixel is labeled with a cell ID.
        classifications (list of tuples): Each tuple (class_label, cell_label)
            class_label: The type of the cell.
            cell_label: The ID of the cell in the mask.

    Returns:
        features (np.ndarray): A 2D array of shape (n_cells, n_features).
        targets (np.ndarray): A 1D array of shape (n_cells,) representing class labels.
        class_labels (np.ndarray): A 1D array of shape (n_cells,) representing cell IDs.
    """
    # First pass to collect all centroids and region coords for context features
    all_centroids = {}
    all_region_coords = {}
    image_height, image_width = masks.shape

    for _, cell_label in classifications:
        mask = np.where(masks == cell_label)
        region_coords = get_formated_region_coords(mask)

        if len(region_coords) == 0:
            continue

        centroid = np.mean(region_coords, axis=0)  # [mean_x, mean_y]
        all_centroids[cell_label] = centroid
        all_region_coords[cell_label] = region_coords

    # Calculate global structure information
    if all_centroids:
        all_centers = np.array(list(all_centroids.values()))
        global_mean_y = np.mean(all_centers[:, 1])
    else:
        global_mean_y = image_height / 2

    # Second pass to extract features
    features = []
    targets = []
    class_labels = []

    for class_label, cell_label in classifications:
        if cell_label not in all_centroids:
            continue

        region_coords = all_region_coords[cell_label]
        centroid = all_centroids[cell_label]
        area = len(region_coords)

        # Compute bounding box
        min_xy = np.min(region_coords, axis=0)
        max_xy = np.max(region_coords, axis=0)
        width = max_xy[0] - min_xy[0] + 1
        height = max_xy[1] - min_xy[1] + 1
        min_y = min_xy[1]
        max_y = max_xy[1]

        geometric_center = np.array(
            [(min_xy[0] + max_xy[0]) / 2.0, (min_xy[1] + max_xy[1]) / 2.0]
        )

        # Compute the magnitude of the shift between centroid and geometric center
        center_of_mass_shift_vector = centroid - geometric_center
        center_of_mass_shift_magnitude = np.linalg.norm(center_of_mass_shift_vector)

        # Compute covariance matrix and eigen decomposition for PCA
        if area > 1:
            cov = np.cov(region_coords, rowvar=False)
            eigenvals, eigenvecs = np.linalg.eig(cov)

            # Sort eigenvalues and eigenvectors in descending order of eigenvalues
            idx = np.argsort(eigenvals)[::-1]
            eigenvals = eigenvals[idx]
            eigenvecs = eigenvecs[:, idx]

            # Principal component directions (unit vectors)
            pc1_dir = eigenvecs[:, 0] / np.linalg.norm(eigenvecs[:, 0])
            pc2_dir = eigenvecs[:, 1] / np.linalg.norm(eigenvecs[:, 1])

            if eigenvals[0] <= 0:
                eccentricity = 0.0
            else:
                major_axis_length = 4.0 * np.sqrt(eigenvals[0])
                minor_axis_length = 4.0 * np.sqrt(eigenvals[-1])
                eccentricity = np.sqrt(
                    1 - (minor_axis_length**2 / major_axis_length**2)
                )
        else:
            # If the cell has only one pixel, skip PCA computations
            eccentricity = 0.0
            pc1_dir = np.array([1.0, 0.0])
            pc2_dir = np.array([0.0, 1.0])

        # New features - position relative to global structure
        relative_y_position = (centroid[1] - global_mean_y) / image_height

        # Count neighbors above and below within a radius
        radius = max(50, 2 * np.sqrt(area))  # Adaptive radius based on cell size
        neighbors_above = 0
        neighbors_below = 0

        for other_label, other_centroid in all_centroids.items():
            if other_label == cell_label:
                continue

            distance = np.linalg.norm(centroid - other_centroid)
            if distance < radius:
                if other_centroid[1] < centroid[1]:  # y-coordinate, above in image
                    neighbors_above += 1
                else:  # below in image
                    neighbors_below += 1

        # Direction vector relative to closest cells in same cluster
        closest_distances = []
        for other_label, other_centroid in all_centroids.items():
            if other_label == cell_label:
                continue
            distance = np.linalg.norm(centroid - other_centroid)
            closest_distances.append((distance, other_centroid))

        closest_distances.sort(key=lambda x: x[0])
        direction_to_neighbors_x = 0.0
        direction_to_neighbors_y = 0.0

        if len(closest_distances) >= 3:
            # Average direction to 3 closest neighbors
            for i in range(3):
                if i < len(closest_distances):
                    _, neighbor = closest_distances[i]
                    direction = neighbor - centroid
                    if np.linalg.norm(direction) > 0:
                        direction = direction / np.linalg.norm(direction)
                        direction_to_neighbors_x += direction[0]
                        direction_to_neighbors_y += direction[1]

            norm = np.sqrt(direction_to_neighbors_x**2 + direction_to_neighbors_y**2)
            if norm > 0:
                direction_to_neighbors_x /= norm
                direction_to_neighbors_y /= norm

        # Calculate vertical position quartile
        vertical_quartile = np.floor(
            4
            * (centroid[1] - np.min(all_centers[:, 1]))
            / (np.max(all_centers[:, 1]) - np.min(all_centers[:, 1]) + 1e-5)
        )

        # Collect features for this cell - original + new spatial context features
        cell_features = [
            area,
            centroid[0],
            centroid[1],
            width,
            height,
            eccentricity,
            pc1_dir[0],
            pc1_dir[1],
            pc2_dir[0],
            pc2_dir[1],
            center_of_mass_shift_magnitude,
            min_y,
            max_y,
            relative_y_position,
            neighbors_above,
            neighbors_below,
            neighbors_above - neighbors_below,  # Net vertical neighbor balance
            direction_to_neighbors_x,
            direction_to_neighbors_y,
            vertical_quartile,
        ]

        features.append(cell_features)
        targets.append(class_label)
        class_labels.append(cell_label)

    features = np.array(features)
    targets = np.array(targets)
    class_labels = np.array(class_labels)
    return features, targets, class_labels


## Is used for preparing the data for training the neural network
def prepare_cell_data(data_path, dataset_numbers, columns=None, verbose=True):
    train_features_list = []
    train_targets_list = []
    script_dir = os.path.dirname(__file__)

    for i in dataset_numbers:

        classifications_file = os.path.join(
            script_dir,
            data_path,
            f"classifications{i}.npy",
        )
        masks_file = os.path.join(
            script_dir,
            data_path,
            f"combined_mean_image{i}_seg.npy",
        )
        classifications_file = os.path.normpath(classifications_file)
        masks_file = os.path.normpath(masks_file)

        masks = np.load(masks_file, allow_pickle=True).item()["masks"]
        classifications = np.load(classifications_file, allow_pickle=True).item()[
            "classifications"
        ]

        features, targets, _ = extract_features(masks, classifications)

        if columns is not None:
            features = features[:, columns]

        train_features_list.append(features)
        train_targets_list.append(targets)

    train_features = np.concatenate(train_features_list)
    train_targets = np.concatenate(train_targets_list)

    adjusted_train_targets = train_targets - 1

    num_classes = 5
    targets_one_hot = np.zeros((adjusted_train_targets.size, num_classes))
    targets_one_hot[np.arange(adjusted_train_targets.size), adjusted_train_targets] = 1

    if verbose:
        unique, counts = np.unique(train_targets, return_counts=True)
        total_samples = adjusted_train_targets.size
        print(f"Total samples: {total_samples}")
        for cls, count in zip(unique, counts):
            print(f"Class {int(cls)}: {count} samples")
        if columns is not None:
            print(f"Selected feature columns: {columns}")

    return train_features, adjusted_train_targets, targets_one_hot


def prepare_data_for_prediction(masks, columns=None, verbose=True):
    """
    Uses the masks file to prepare data so it can be feed to the neural network for prediction
    Args:
        masks (np.ndarray): The segmentation masks with labeled regions
        columns (list): list of feature columns to select from the data
        verbose (bool): if True, prints the number of samples in each class

    Returns:
        features (np.ndarray): features of the data
        cell_labels (np.ndarray): labels of the cells
    """

    features_list = []
    cell_labels = np.unique(masks)[1:]  # Skip background (0)

    # First pass to collect all centroids for spatial context features
    all_centroids = {}
    all_region_coords = {}
    image_height, image_width = masks.shape

    for label in cell_labels:
        mask = np.where(masks == label)
        region_coords = get_formated_region_coords(mask)

        if len(region_coords) == 0:
            continue

        centroid = np.mean(region_coords, axis=0)  # [mean_x, mean_y]
        all_centroids[label] = centroid
        all_region_coords[label] = region_coords

    # Calculate global structure information
    if all_centroids:
        all_centers = np.array(list(all_centroids.values()))
        global_mean_y = np.mean(all_centers[:, 1])
        # Optional: Find main ridge line through K-means or polynomial fitting
        # main_ridge = fit_ridge_line(all_centers)
    else:
        global_mean_y = image_height / 2

    # Second pass to extract features with spatial context
    for label in cell_labels:
        if label not in all_centroids:
            continue

        region_coords = all_region_coords[label]
        centroid = all_centroids[label]
        area = len(region_coords)

        # Compute bounding box
        min_xy = np.min(region_coords, axis=0)
        max_xy = np.max(region_coords, axis=0)
        width = max_xy[0] - min_xy[0] + 1
        height = max_xy[1] - min_xy[1] + 1
        min_y = min_xy[1]
        max_y = max_xy[1]

        geometric_center = np.array(
            [(min_xy[0] + max_xy[0]) / 2.0, (min_xy[1] + max_xy[1]) / 2.0]
        )

        # Compute the magnitude of the shift between centroid and geometric center
        center_of_mass_shift_vector = centroid - geometric_center
        center_of_mass_shift_magnitude = np.linalg.norm(center_of_mass_shift_vector)

        # Compute covariance matrix and eigen decomposition for PCA
        if area > 1:
            cov = np.cov(region_coords, rowvar=False)
            eigenvals, eigenvecs = np.linalg.eig(cov)

            # Sort eigenvalues and eigenvectors in descending order of eigenvalues
            idx = np.argsort(eigenvals)[::-1]
            eigenvals = eigenvals[idx]
            eigenvecs = eigenvecs[:, idx]

            # Principal component directions (unit vectors)
            pc1_dir = eigenvecs[:, 0] / np.linalg.norm(eigenvecs[:, 0])
            pc2_dir = eigenvecs[:, 1] / np.linalg.norm(eigenvecs[:, 1])

            if eigenvals[0] <= 0:
                eccentricity = 0.0
            else:
                major_axis_length = 4.0 * np.sqrt(eigenvals[0])
                minor_axis_length = 4.0 * np.sqrt(eigenvals[-1])
                eccentricity = np.sqrt(
                    1 - (minor_axis_length**2 / major_axis_length**2)
                )
        else:
            # If the cell has only one pixel, skip PCA computations
            eccentricity = 0.0
            pc1_dir = np.array([1.0, 0.0])
            pc2_dir = np.array([0.0, 1.0])

        # New features - position relative to global structure
        relative_y_position = (centroid[1] - global_mean_y) / image_height

        # Count neighbors above and below within a radius
        radius = max(50, 2 * np.sqrt(area))  # Adaptive radius based on cell size
        neighbors_above = 0
        neighbors_below = 0

        for other_label, other_centroid in all_centroids.items():
            if other_label == label:
                continue

            distance = np.linalg.norm(centroid - other_centroid)
            if distance < radius:
                if other_centroid[1] < centroid[1]:  # y-coordinate, above in image
                    neighbors_above += 1
                else:  # below in image
                    neighbors_below += 1

        # Direction vector relative to closest cells in same cluster
        closest_distances = []
        for other_label, other_centroid in all_centroids.items():
            if other_label == label:
                continue
            distance = np.linalg.norm(centroid - other_centroid)
            closest_distances.append((distance, other_centroid))

        closest_distances.sort(key=lambda x: x[0])
        direction_to_neighbors_x = 0.0
        direction_to_neighbors_y = 0.0

        if len(closest_distances) >= 3:
            # Average direction to 3 closest neighbors
            for i in range(3):
                if i < len(closest_distances):
                    _, neighbor = closest_distances[i]
                    direction = neighbor - centroid
                    if np.linalg.norm(direction) > 0:
                        direction = direction / np.linalg.norm(direction)
                        direction_to_neighbors_x += direction[0]
                        direction_to_neighbors_y += direction[1]

            norm = np.sqrt(direction_to_neighbors_x**2 + direction_to_neighbors_y**2)
            if norm > 0:
                direction_to_neighbors_x /= norm
                direction_to_neighbors_y /= norm

        # Collect features for this cell - original + new spatial context features
        cell_features = [
            area,
            centroid[0],
            centroid[1],
            width,
            height,
            eccentricity,
            pc1_dir[0],
            pc1_dir[1],
            pc2_dir[0],
            pc2_dir[1],
            center_of_mass_shift_magnitude,
            min_y,
            max_y,
            # New features
            relative_y_position,
            neighbors_above,
            neighbors_below,
            neighbors_above - neighbors_below,  # Net vertical neighbor balance
            direction_to_neighbors_x,
            direction_to_neighbors_y,
            # Optional: vertical position quartile within global structure
            np.floor(
                4
                * (centroid[1] - np.min(all_centers[:, 1]))
                / (np.max(all_centers[:, 1]) - np.min(all_centers[:, 1]) + 1e-5)
            ),
        ]

        features_list.append(cell_features)

    features = np.array(features_list)
    if columns is not None:
        features = features[:, columns]

    if verbose:
        total_samples = len(features_list)
        print(f"Total samples: {total_samples}")
        if columns is not None:
            print(f"Selected feature columns: {columns}")

    return features, cell_labels
