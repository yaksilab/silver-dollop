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


def rotate_region1(pc, region_coords, up: bool):
    """
    Rotates the given region coordinates around a specific point and returns both
    the rotated coordinates and a matrix containing original and rotated coordinates.

    Parameters:
    - pc: A tuple or list containing the x and y components (pc[0], pc[1]).
    - region_coords: A NumPy array of shape (N, 2) containing the coordinates to rotate.
    - up: A boolean flag determining the rotation direction.

    Returns:
    - result_matrix: A NumPy array of shape (N, 4) in the form [x_original, y_original, x_rotated, y_rotated].
    """

    # Extract x and y components from pc
    x, y = pc[0], pc[1]

    # Determine the rotation angle based on the 'up' flag
    if up:
        angle_to_y = -np.arctan(x / y)
        # Find the lowest point (minimum y-coordinate) for rotation
        rotation_point = region_coords[np.argmin(region_coords[:, 1])]
    else:
        angle_to_y = -np.arctan(x / y)
        # Find the highest point (maximum y-coordinate) for rotation
        rotation_point = region_coords[np.argmax(region_coords[:, 1])]

    rotation_angle = angle_to_y

    # Store the original coordinates before transformation
    original_coords = region_coords.copy()

    # Translate coordinates so that rotation occurs around the rotation_point
    translated_coords = region_coords - rotation_point

    # Create the 2D rotation matrix
    rotation_matrix = np.array(
        [
            [np.cos(rotation_angle), -np.sin(rotation_angle)],
            [np.sin(rotation_angle), np.cos(rotation_angle)],
        ]
    )

    # Apply the rotation matrix to the translated coordinates
    rotated_translated_coords = np.dot(translated_coords, rotation_matrix)

    # Translate the coordinates back to the original position
    rotated_coords = rotated_translated_coords + rotation_point

    # Create the (N, 4) matrix: [x_original, y_original, x_rotated, y_rotated]
    result_matrix = np.hstack((original_coords, rotated_coords))

    return result_matrix


def get_formated_region_coords(region_coords, revert=False):
    if revert:
        region_coords = np.flip(region_coords, axis=1)
        region_coords = np.transpose(region_coords)
    else:
        region_coords = np.transpose(region_coords)
        region_coords = np.flip(region_coords, axis=1)
    return region_coords
