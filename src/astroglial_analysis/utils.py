import numpy as np


def rotate_region(pc, cova, region_coords):
    x, y = pc[0], pc[1]
    angle_to_x = np.arctan2(y, x)
    if cova >= 0:
        rotation_angle = np.pi / 2 - angle_to_x  # Counterclockwise rotation
    else:
        rotation_angle = -(np.pi / 2 - angle_to_x)  # Clockwise rotation

    rotation_matrix = np.array(
        [
            [np.cos(rotation_angle), -np.sin(rotation_angle)],
            [np.sin(rotation_angle), np.cos(rotation_angle)],
        ]
    )

    rotated_coords = np.dot(region_coords, rotation_matrix)

    return rotated_coords
