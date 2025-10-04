import numpy
import pandas as pd
import functions as fn
from typing import Literal
import constants as c
import lookupTables as lt



def bending_stress(wp, n1, Pnd, Np1, Np2, Helix):
    # where:
        # wp = input speed (RPM)
        # n1 = stage 1 input ratio
        # Pnd = diametral pitch (teeth/inch)
        # Np1 = pinion teeth number stage 1
        # Np2 = pinion teeth number stage 2
        # Helix = helix angle (degrees)
    
    Nc1=wp * c.L * 60
    Yn1=1.3558*Nc1^-0.0178
    st1_ = st1 * c.SF * Kr / (1000*Yn1)
    
    return st1_  # in psi

def contact_stress(F, P, b, d, C, I):
    """
    Calculate contact stress using the AGMA formula.
    
    Parameters:
    F : Applied load (N)
    P : Diametral pitch (teeth/inch)
    b : Face width (inches)
    d : Pitch diameter (inches)
    C : Elastic coefficient (dimensionless)
    I : Geometry factor (dimensionless)
    
    Returns:
    Contact stress (psi)
    """
    # Convert units where necessary
    F_lbf = F * 0.224809  # Convert N to lbf
    b_in = b              # Face width in inches
    d_in = d              # Pitch diameter in inches
    P_in = P              # Diametral pitch in teeth/inch
    
    # AGMA contact stress formula
    sigma_c = C * ((F_lbf * P_in) / (b_in * d_in * I))**0.5
    
    return sigma_c  # in psi
