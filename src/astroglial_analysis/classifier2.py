import numpy as np
import matplotlib.pyplot as plt
from skimage.measure import regionprops, label
import cv2
from .determine_line import get_cellbody_center
from .my_types import Region
from astroglial_analysis.pca import get_pcs


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
        coords = region.coords
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


def classify_masks(masks, body_size=200):
    classifications = []
    features = calculate_elongation_center_of_mass(masks)
    # mask_shape = masks.shape  # (y,x)

    body = []
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
        coords = np.flip(coords, axis=1)  # This is now in (x,y) format

        if elongation < elongation_threshold:
            classification = 1
            body.append(region.label)
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
        # Calculate the overall total mean y-coordinate
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

    return classifications, body, processes, complete_cell


def visualize_classifications(masks, classifications):

    for classification, label in classifications:
        if classification == 1:
            color = "red"
        elif classification == 2:
            color = "blue"
        else:
            color = "green"

        region = np.where(masks == label)
        region = np.array(region).T
        plt.scatter(region[:, 1], region[:, 0], s=1, c=color)
