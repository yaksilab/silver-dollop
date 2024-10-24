import numpy as np

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
