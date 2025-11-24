import numpy
import pandas as pd
from typing import Literal


def distance(a, b):
    return ((abs(a - b) / ((a + b) / 2)) * 100)

<<<<<<< Updated upstream
=======
def volume(Dp1, Dg1, Dp2, Dg2, Dp3, Dg3, F1, F2, F3):

    # Dp = Pinion Diameter
    # Dg = Gear Diameter
    # F = Face Width

    if Dp2 >= Dg1:
        bottomOvershoot = 0.5 * (Dp1 - Dg1)
    else:
        bottomOvershoot = 0

    if Dg2 - Dp1 + 0.5 * (Dp2 - Dg1) >= 0:
        topOvershoot = Dg2 - Dp1 + 0.5 * (Dp2 - Dg1)
    else:
        topOvershoot = 0

    allowance = 0.2                 # in inches
    minimumGearFaceSpacing = 0.7   # in inches

    width = F1 + F2 + 2 * allowance + minimumGearFaceSpacing            # in inches
    height = Dp1 + Dg1 + bottomOvershoot + topOvershoot + allowance * 2 # in inches
    depth = allowance * 2 + max(Dp1, Dg1, Dp2, Dg2)                     # in inches

    volume = width * height * depth  # in cubic inches
    return volume
>>>>>>> Stashed changes
