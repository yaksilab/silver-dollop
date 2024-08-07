import numpy as np
from skimage.measure import regionprops, label as skimage_label
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
import scipy.ndimage
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
from pca import get_pcs
from plot import plot_pcs
from plot import plot_scatter
from classifier import classify_masks
from utils import rotate_region


p0 = r"C:\YaksiData\astrolglialAnalysis\tests\data\combined_mean_image_seg.npy"
p1 = r"C:\YaksiData\astrolglialAnalysis\tests\data\CP2_s1_039189_mean_image_seg.npy"
p2 = r"C:\YaksiData\astrolglialAnalysis\tests\data\CP2_s3_039234_mean_image_seg.npy"

masks = np.load(p0, allow_pickle=True).item()["masks"]
masks = masks.copy()
labels = np.unique(masks)
labels = labels[labels != 0]

classifications, body, processes, body_and_processes = classify_masks(masks)
processes_plus_body_and_processes = np.concatenate((processes, body_and_processes))

# remove body from masks
for label in body:
    masks[masks == label] = 0


# plt.imshow(masks)
# for label in processes_plus_body_and_processes:
#     region = np.where(masks == label)
#     region = np.transpose(region)
#     region = np.flip(region, axis=1)

#     pc, eigenvalue, covar = get_pcs(region)
#     center = np.mean(region, axis=0)
#     plot_pcs(pc, eigenvalue, covar, center)
# plt.show()

# testing region rotation
# selected_region = np.where(masks == 235)
# original_region = selected_region
# selected_region = np.transpose(selected_region)
# selected_region = np.flip(selected_region, axis=1)

# pc, eigenvalue, covar = get_pcs(selected_region)

# rotate_region = rotate_region(pc, covar, selected_region)
# plt.scatter(selected_region[:, 0], selected_region[:, 1])
# plt.scatter(rotate_region[:, 0], rotate_region[:, 1])
# plt.show()
