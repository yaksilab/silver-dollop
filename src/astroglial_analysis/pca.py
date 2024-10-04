import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA


def get_pcs(region_coords):
    scalar = StandardScaler()
    region_coords_normalized = scalar.fit_transform(region_coords)
    pca = PCA(n_components=1)
    pca.fit(region_coords_normalized)
    pc = pca.components_[0]
    eigenvalue = pca.explained_variance_[0]
    covariance = pca.get_covariance()[0][1]

    # Denormalize the principal component
    pc_denormalized = pc * scalar.scale_
    pc_denormalized /= np.linalg.norm(pc_denormalized)  # Ensure it's a unit vector

    return pc_denormalized, eigenvalue, covariance


def get_variance_direction(pc, covar):

    x = pc[0]
    y = pc[1]
    if covar < 0 and x < 0:
        x = -x
        y = -y
    return x, y
