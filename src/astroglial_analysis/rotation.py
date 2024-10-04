import numpy as np
import matplotlib.pyplot as plt
from .utils import get_formated_region_coords
from .pca import get_pcs, get_variance_direction
from .projection import upper_lower


# p0 = r"tests\data\combined_mean_image1_seg.npy"
# p1 = r"tests\data\CP2_s1_039189_mean_image_seg.npy"
# p2 = r"tests\data\CP2_s3_039234_mean_image_seg.npy"

# masks = np.load(p0, allow_pickle=True).item()["masks"]
# print(masks.shape)
# masks = masks.copy()
# labels = np.unique(masks)
# labels = labels[labels != 0]

# # classifications, body, processes, body_and_processes = classify_masks(masks)
# processes_plus_body_and_processes = np.concatenate((processes, body_and_processes))
# upper, processes_plus_body_and_processes = upper_lower(
#     processes_plus_body_and_processes, masks
# )


def rotate_region(angle, region, rotation_point):
    rotation_matrix = np.array(
        [
            [np.cos(angle), -np.sin(angle)],
            [np.sin(angle), np.cos(angle)],
        ]
    )

    region = region - rotation_point
    rotated_region = np.dot(region, rotation_matrix)
    return rotated_region + rotation_point


def get_x_intersect(pc, mean, y_center) -> int:
    return (y_center - mean[1] + (pc[1] / pc[0]) * mean[0]) / (pc[1] / pc[0])


# ## 60 degrees in randians
# r_angle_min = np.pi / 6

# increase_constant = 0.00065

# rotation_point = [390, 255]


# plt.imshow(masks)

# for i, label in enumerate(upper):
#     region = np.where(masks == label)
#     region = get_formated_region_coords(region)
#     mean = np.mean(region, axis=0)

#     pc, eigenvalue, covar = get_pcs(region)
#     pc = get_variance_direction(pc, eigenvalue)
#     x_intersect = get_x_intersect(pc, mean, rotation_point[1])
#     x_distance = np.abs(x_intersect - rotation_point[0])

#     # plt.scatter(region[:, 0], region[:, 1])
#     # plot_pcs(pc, 0.1 * eigenvalue, covar, mean)

#     if x_intersect < 390:
#         # t_angle = -np.arctan(pc[0] / pc[1])
#         r_angle = r_angle_min + x_distance * increase_constant
#         # region = rotate_region(t_angle, region, mean)
#         # region += mean
#         region = rotate_region(r_angle, region, rotation_point)
#         plt.scatter(region[:, 0], region[:, 1], s=1)
#         # rotation_angle -= decrease_angle
#     else:
#         r_angle = -np.arctan(pc[0] / pc[1])
#         region = rotate_region(r_angle, region, mean)
#         plt.scatter(region[:, 0], region[:, 1], s=1)

# for i, label in enumerate(processes_plus_body_and_processes):
#     region = np.where(masks == label)
#     region = get_formated_region_coords(region)
#     mean = np.mean(region, axis=0)

#     pc, eigenvalue, covar = get_pcs(region)
#     pc = get_variance_direction(pc, eigenvalue)
#     x_intersect = get_x_intersect(pc, mean, rotation_point[1])
#     x_distance = np.abs(x_intersect - rotation_point[0])

#     # plt.scatter(region[:, 0], region[:, 1])
#     # plot_pcs(pc, 0.1 * eigenvalue, covar, mean)

#     if x_intersect < 390:
#         # t_angle = -np.arctan(pc[0] / pc[1])
#         r_angle = r_angle_min + x_distance * increase_constant

#         region = rotate_region(-r_angle, region, rotation_point)
#         plt.scatter(region[:, 0], region[:, 1], s=1)
#     else:
#         r_angle = -np.arctan(pc[0] / pc[1])
#         region = rotate_region(r_angle, region, mean)
#         plt.scatter(region[:, 0], region[:, 1], s=1)
# # plt.gca().invert_yaxis()
# plt.show()
