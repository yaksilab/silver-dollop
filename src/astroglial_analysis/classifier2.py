import numpy as np
import matplotlib.pyplot as plt
from skimage.measure import regionprops, label
import cv2
from astroglial_analysis.determine_line import get_cellbody_center
from astroglial_analysis.my_types import Region
from astroglial_analysis.pca import get_pcs
import warnings


def get_ab(coords):
    pc, _, _ = get_pcs(coords)
    mean = np.mean(coords, axis=0)
    if np.isclose(pc[0], 0):
        a = float("inf")
        b = mean[0]  # x = b equation for vertical line
    else:
        a = pc[1] / pc[0]
        b = mean[1] - a * mean[0]

    return a, b, mean


def label_region(region: Region) -> float:
    region1 = region.copy()
    miny = np.min(region[:, 0])
    minx = np.min(region[:, 1])
    maxy = np.max(region[:, 0])
    maxx = np.max(region[:, 1])
    region1[:, 0] -= miny
    region1[:, 1] -= minx
    image = np.zeros((int(maxy - miny + 1), int(maxx - minx + 1)))
    image[region1[:, 0].astype(int), region1[:, 1].astype(int)] = 1

    labeled = label(image)
    region_props = regionprops(labeled)
    region1 = region_props[0]

    return region_props[0]


def calculate_elongation_center_of_mass(masks):
    labeled_mask = masks
    regions = regionprops(labeled_mask)

    features = []
    for region in regions:
        elongation = region.eccentricity
        center = region.centroid
        coords = region.coords
        if len(coords) < 3:
            continue

        minr, minc, maxr, maxc = region.bbox
        geometric_center = ((minr + maxr) / 2, (minc + maxc) / 2)  # uses bounding box
        rect = cv2.minAreaRect(region.coords)

        min_area_rec_center = rect[
            0
        ]  # center of the minimum area rectangle better normal bounding box.
        max_dist = np.linalg.norm([maxr - minr, maxc - minc]) / 2

        # relative shift
        center_shift = np.linalg.norm(np.array(center) - np.array(min_area_rec_center))

        # Calculate the direction of the shift
        shift_direction = -np.array(min_area_rec_center) + np.array(center)
        norm = np.linalg.norm(shift_direction)
        shift_direction_normalized = (
            shift_direction / norm if norm > 0 else np.array([0, 0])
        )

        # Calculate covariance matrix
        covariance_matrix = np.cov(coords, rowvar=False)
        covariance = covariance_matrix[0][1]
        features.append(
            (
                elongation,
                center_shift,
                labeled_mask == region.label,
                shift_direction_normalized,
                region,
                covariance,
            )
        )

    return features


def classify_masks_nn(masks, model_path, columns=None):
    """
    DEPRECATED: This function replaces classify_masks() with a neural network-based classifier.
    Args:
        masks (np.ndarray): Segmentation masks with labeled regions.
        model_path (str): Path to the pretrained neural network (.pkl).
        columns (list or None): Feature columns to use (optional).
    Returns:
        classifications (list of tuples): (predicted_class, region_label) for each region.
        body (dict): {"upper": [...], "lower": [...]} for class 5
        processes (dict): {"upper": [...], "lower": [...]} for classes 3, 4
        complete_cell (dict): {"upper": [...], "lower": [...]} for classes 1, 2
    """
    warnings.warn(
        "classify_masks_nn() replaces classify_masks() and uses a neural network. This function is experimental.",
        DeprecationWarning,
    )
    from neural_network.utils import load_neural_network
    from astroglial_analysis.utils import prepare_data_for_prediction
    import numpy as np

    features, cell_labels = prepare_data_for_prediction(masks, columns)
    classifier = load_neural_network(model_path)
    predicted_classes = classifier.predict_classes(features)
    predicted_classes = predicted_classes + 1
    classifications = [
        (int(pred), int(label)) for pred, label in zip(predicted_classes, cell_labels)
    ]

    body = {"total": []}
    processes = {"upper": [], "lower": []}
    complete_cell = {"upper": [], "lower": []}
    for pred, label in classifications:
        if pred == 1:
            complete_cell["upper"].append(label)
        elif pred == 2:
            complete_cell["lower"].append(label)
        elif pred == 3:
            processes["upper"].append(label)
        elif pred == 4:
            processes["lower"].append(label)
        elif pred == 5:
            body["total"].append(label)
    return classifications, body, processes, complete_cell


def classify_masks(masks, body_size=200):
    classifications = []
    features = calculate_elongation_center_of_mass(masks)
    # mask_shape = masks.shape  # (y,x)

    body = []
    body_means = []  # New list to store body mean coordinates
    processes = []
    complete_cell = {"upper": [], "lower": []}
    means_upper = []
    means_lower = []
    process_means = []
    process_labels = []

    # Define thresholds for classification
    elongation_threshold = 0.85
    elong_thresh = 0.98
    shift_threshold = 2

    for elongation, center_shift, mask, shift_direction, region, cov in features:
        classification = None
        coords = region.coords  # This in (y,x) format
        coords = np.flip(coords, axis=1)  # Now in (x,y) format

        if elongation < elongation_threshold:
            classification = 1
            body.append(region.label)
            body_means.append(np.mean(coords, axis=0))  # Record mean for body
        elif center_shift < shift_threshold or region.area < 1.5 * body_size:
            classification = 2
            processes.append(region.label)
            process_means.append(np.mean(coords, axis=0))
            process_labels.append(region.label)
        else:
            if (cov >= 0 and shift_direction[0] <= 0) or (
                cov < 0 and shift_direction[0] < 0
            ):
                _, bod = get_cellbody_center(coords, True, body_size)
                elong = label_region(bod)
                if elong.eccentricity > elong_thresh:
                    processes.append(region.label)
                    classification = 2
                    process_means.append(np.mean(coords, axis=0))
                    process_labels.append(region.label)
                else:
                    classification = 3
                    complete_cell["upper"].append(region.label)
                    means_upper.append(np.mean(coords, axis=0))
            elif (cov < 0 and shift_direction[0] > 0) or (
                cov > 0 and shift_direction[0] > 0
            ):
                center, bod = get_cellbody_center(coords, False, body_size)
                elong = label_region(bod)
                if elong.eccentricity > elong_thresh:
                    processes.append(region.label)
                    classification = 2
                    process_means.append(np.mean(coords, axis=0))
                    process_labels.append(region.label)
                else:
                    classification = 3
                    complete_cell["lower"].append(region.label)
                    means_lower.append(np.mean(coords, axis=0))
            else:
                print(
                    "cannot classify region: ",
                    region.label,
                    "with cov: ",
                    cov,
                    "and center_shift: ",
                    shift_direction,
                )

        classifications.append((classification, region.label))

    # Reclassification Step for complete_cell and processes
    if means_upper or means_lower:
        mean_upper = np.mean(means_upper, axis=0)
        mean_lower = np.mean(means_lower, axis=0)
        total_mean = (mean_upper + mean_lower) / 2

        reclassified_upper = []
        reclassified_lower = []

        # Reclassify lower cells
        for label, mean in zip(complete_cell["lower"], means_lower):
            if mean[1] > total_mean[1] * 1.2:  # Comparing y-coordinate
                reclassified_upper.append(label)
            else:
                reclassified_lower.append(label)

        # Reclassify upper cells
        for label, mean in zip(complete_cell["upper"], means_upper):
            if mean[1] < total_mean[1] * 0.8:  # Comparing y-coordinate
                reclassified_lower.append(label)
            else:
                reclassified_upper.append(label)

        complete_cell["upper"] = reclassified_upper
        complete_cell["lower"] = reclassified_lower

    # Reclassify Processes into Upper and Lower
    processes_upper = []
    processes_lower = []
    if process_means:
        if means_upper and means_lower:
            mean_upper = np.mean(means_upper, axis=0)
            mean_lower = np.mean(means_lower, axis=0)
            total_mean = (mean_upper + mean_lower) / 2
            total_mean_y = total_mean[1]
        else:
            total_mean_y = np.mean([mean[1] for mean in process_means])
        for label, mean in zip(process_labels, process_means):
            if mean[1] > total_mean_y * 1.1:
                processes_upper.append(label)
            elif mean[1] < total_mean_y * 0.9:
                processes_lower.append(label)
            else:
                if mean[1] >= total_mean_y:
                    processes_upper.append(label)
                else:
                    processes_lower.append(label)

    processes = {"upper": processes_upper, "lower": processes_lower}
    complete_cell["processes_upper"] = processes_upper
    complete_cell["processes_lower"] = processes_lower

    # New: Reclassify Body into Upper and Lower Groups
    body_upper = []
    body_lower = []
    if body_means:
        total_body_mean = (np.mean(body_means, axis=0) + total_mean) / 2
        for label, mean in zip(body, body_means):
            if mean[1] > total_body_mean[1]:
                body_upper.append(label)
            else:
                body_lower.append(label)
        # Replace body list with a dictionary classification
        body = {"upper": body_upper, "lower": body_lower}

        final_classifications = []
    for label in complete_cell["upper"]:
        final_classifications.append((1, label))
    for label in complete_cell["lower"]:
        final_classifications.append((2, label))
    for label in processes["upper"]:
        final_classifications.append((3, label))
    for label in processes["lower"]:
        final_classifications.append((4, label))
    # Combine both upper and lower body parts with the same label 5
    for label in body["upper"] + body["lower"]:
        final_classifications.append((5, label))

    return final_classifications, body, processes, complete_cell


# DEPRECATED: Mark old function as deprecated but keep for backward compatibility

import warnings

old_classify_masks = classify_masks


def classify_masks(*args, **kwargs):
    warnings.warn(
        "classify_masks() is deprecated. Use classify_masks_nn() instead.",
        DeprecationWarning,
    )
    return old_classify_masks(*args, **kwargs)


def visualize_classifications(masks, classifications):

    for classification, label in classifications:
        if classification == 1:
            # complete cell upper
            color = "red"
        elif classification == 2:
            # complete cell lower
            color = "blue"
        elif classification == 3:
            # processes upper
            color = "green"
        elif classification == 4:
            # processes lower
            color = "yellow"
        elif classification == 5:
            # body upper
            color = "magenta"
        elif classification == 6:
            # body lower
            color = "cyan"
        else:
            color = "red"

        region = np.where(masks == label)
        region = np.array(region).T
        plt.scatter(region[:, 1], region[:, 0], s=1, c=color)

        # Calculate the center of the region
        center_y, center_x = np.mean(region, axis=0)

        # Add the label text at the center with black color
        plt.text(
            center_x,
            center_y,
            str(label),
            color="black",
            fontsize=6,
            ha="center",
            va="center",
        )
