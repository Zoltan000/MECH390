import numpy
import pandas as pd
from typing import Literal


def distance(a, b):
    return (((b - a) / ((b + a) / 2)) * 100)

def volume(Dp1, Dg1, Dp2, Dg2, Dp3, Dg3, F1, F2, F3):

    # Dp = Pinion Diameter
    # Dg = Gear Diameter
    # F = Face Width
    allowance = 0.2                # in inches
    minimumGearFaceSpacing = 0.7   # in inches

    if Dp2 >= Dg1:
        bottomOverShoot1 = 0.5 * (Dp2 - Dg1)
    else:
        bottomOverShoot1 = 0

    if Dg2 - Dp1 + 0.5 * (Dp2 - Dg1) >= 0:
        topOverShoot1 = Dg2 - Dp1 + 0.5 * (Dp2 - Dg1)
    else:
        topOverShoot1 = 0
    
    if Dp3 == None or Dg3 == None or F3 == None:                                # if two stage, calculate and return the volume for just the two stages
        width = F1 + F2 + 2 * allowance + minimumGearFaceSpacing                # in inches
        height = Dp1 + Dg1 + bottomOverShoot1 + topOverShoot1 + allowance * 2   # in inches
        depth = allowance * 2 + max(Dp1, Dg1, Dp2, Dg2)                         # in inches
        volume = width * height * depth                                         # in cubic inches
        
        return volume                                                           # in cubic inches

    if Dp3 >= Dg2:
        topOverShoot2 = 0.5 * (Dp3 - Dg2)
    else:
        topOverShoot2 = 0

    if Dg3 - Dp2 + 0.5 * (Dp3 - Dg2) >= 0:
        bottomOverShoot2 = Dg3 - Dp2 + 0.5 * (Dp3 - Dg2)
    else:
        bottomOverShoot2 = 0

    width = F1 + F2 + F3 + allowance * 2 + minimumGearFaceSpacing * 2                                           # in inches
    height = Dp1 + Dg1 + bottomOverShoot1 + topOverShoot1 + topOverShoot2 + bottomOverShoot2 + allowance * 2    # in inches
    depth = allowance * 2 + max(Dp1, Dg1, Dp2, Dg2, Dp3, Dg3)                                                   # in inches
    volume = width * height * depth                                                                             # in cubic inches
    
    return volume