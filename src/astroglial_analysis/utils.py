import numpy as np


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
