import matplotlib.pyplot as plt
import numpy as np
from pca import get_variance_direction
from determine_line import get_formated_region_coords


def plot_pcs(pc, eigenvalue, covar, start):

    eigenvalue = np.sqrt(eigenvalue)
    x, y = get_variance_direction(pc, covar)
    x, y = pc

    # Plot the PCA
    plt.quiver(
        start[0],
        start[1],
        x * eigenvalue,
        y * eigenvalue,
        scale=1,
        scale_units="xy",
        angles="xy",
    )


def plot_scatter(region_coords):
    plt.scatter(region_coords[:, 0], region_coords[:, 1])


def plot_upper_lower(labels: list, mask_array: np.ndarray, upper: bool):
    """
    Plots the upper or lower part of the region.
    Args:
        labels (list): A list of region labels to be processed.
        mask_array (numpy.ndarray): a labeled mask array.
        upper (bool): A boolean flag indicating whether to consider the upper part of the region.
    """
    for region_label in labels:
        region = np.where(mask_array == region_label)
        region = get_formated_region_coords(region)
        plt.scatter(region[:, 0], region[:, 1], s=1)
