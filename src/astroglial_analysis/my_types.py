import numpy as np
from enum import IntEnum


Region = np.ndarray[tuple[int, 2]]
"""
Region: A numpy array of shape (n, 2) where n is the number of points in the region.
"""


Label = int
Labels = list[int]


Cm = tuple[float, float]
"""
(x,y)
"""


LIdentifier = tuple[float, float, Cm]
"""
Line identifier: (a,b,cm)
"""

PIdentifier = list[tuple[Label, LIdentifier]]
"""
Process identifier: (Label, LIdentifier)
"""

TIntersection = int
"""
Intersection point on the curve parameterized by t
"""

Intersection = tuple[TIntersection, float, float]
"""
Intersection point: (t, x, y)
"""

ParamCurveLine = list[tuple[Cm, Label]]
"""
list of tuples: (Cm, Label)
where `Cm` is the center of mass of the cell body of a complete cell.
"""

IDRegion = tuple[Label, Region]
"""
Aligned region: (Label, Region)
"""


class Features(IntEnum):
    """
    Enumerates feature columns returned by extract_features function

    Usage:
        selected_features = [Features.AREA, Features.CENTROID_X, Features.CENTROID_Y]
        features = extract_features(masks, selected_features)
    """

    AREA = 0
    CENTROID_X = 1
    CENTROID_Y = 2
    WIDTH = 3
    HEIGHT = 4
    ECCENTRICITY = 5
    PC1_DIR_X = 6
    PC1_DIR_Y = 7
    PC2_DIR_X = 8
    PC2_DIR_Y = 9
    CENTER_OF_MASS_SHIFT = 10
    MIN_Y = 11
    MAX_Y = 12

    RELATIVE_Y_POSITION = 13
    NEIGHBORS_ABOVE = 14
    NEIGHBORS_BELOW = 15
    NEIGHBOR_BALANCE = 16
    DIR_TO_NEIGHBORS_X = 17
    DIR_TO_NEIGHBORS_Y = 18
    VERTICAL_QUARTILE = 19

