import numpy as np


Region = np.ndarray[int]

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