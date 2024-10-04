import numpy as np
import matplotlib.pyplot as plt
from skimage.measure import regionprops, label
import cv2
from .determine_line import get_cellbody_center
from .my_types import Region


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
        geometric_center = ((minr + maxr) / 2, (minc + maxc) / 2)
        rect = cv2.minAreaRect(region.coords)

        min_area_rec_center = rect[0]
        # print("center", center, "cv2_center", cv2_center)

        max_dist = np.linalg.norm([maxr - minr, maxc - minc]) / 2
        # relative shift
        center_shift = np.linalg.norm(np.array(center) - np.array(min_area_rec_center))
        # plt.scatter(region.coords[:, 1], region.coords[:, 0], s=5)
        # plt.scatter(center[1], center[0], c="red", s=15)
        # plt.scatter(geometric_center[1], geometric_center[0], c="blue", s=15)
        # plt.scatter(min_area_rec_center[1], min_area_rec_center[0], c="green", s=15)

        # Calculate the direction of the shift
        shift_direction = -np.array(min_area_rec_center) + np.array(center)
        shift_direction_normalized = shift_direction / np.linalg.norm(shift_direction)

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

    body = []
    processes = []
    body_and_processes = {"upper": [], "lower": []}

    for elongation, center_shift, mask, shift_direction, region, cov in features:
        # Define thresholds for classification
        elongation_threshold = 0.85
        elong_thresh = 0.98
        shift_threshold = 2
        classification = None
        coords = region.coords  # This in (y,x) format
        coords = np.flip(coords, axis=1)  # This is now in (x,y) format

        if elongation < elongation_threshold:
            classification = 1
            body.append(region.label)
        elif center_shift < shift_threshold or region.area < 1.5 * body_size:
            classification = 2
            processes.append(region.label)

        else:

            if (cov >= 0 and shift_direction[0] <= 0) or (
                cov < 0 and shift_direction[0] < 0
            ):
                center, bod = get_cellbody_center(coords, True, body_size)

                elong = label_region(bod)

                if elong.eccentricity > elong_thresh:
                    processes.append(region.label)
                    classification = 2
                else:
                    classification = 3
                    body_and_processes["upper"].append(region.label)

            elif (cov < 0 and shift_direction[0] > 0) or (
                cov > 0 and shift_direction[0] > 0
            ):

                center, bod = get_cellbody_center(coords, False, body_size)
                elong = label_region(bod)

                if elong.eccentricity > elong_thresh:
                    processes.append(region.label)
                    classification = 2
                else:
                    classification = 3
                    body_and_processes["lower"].append(region.label)
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

    return classifications, body, processes, body_and_processes


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
