import matplotlib.pyplot as plt
import numpy as np
from .pca import get_variance_direction


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
