import math
import numpy
import pandas as pd
import functions as fn
from typing import Literal
import constants as c
import lookupTables as lt
from scipy.interpolate import RegularGridInterpolator, interp2d


def bending_stress(wp,  n1, Pnd, Np1, Helix):
    # Input validation
    if not (1200 <= wp <= 3600):        #wp = input speed in RPM
        raise ValueError("wp (input speed) should be between 1200 and 3600 RPM.")
    if not (1 <= n1 <= 10):             #n1 = stage 1 input ratio
        raise ValueError("n1 (stage 1 input ratio) should be between 1 and 10.")
    if Pnd not in [4, 5, 6, 8, 10]:     #Pnd = Normal diametral pitch (teeth/inch)
        raise ValueError("Pnd (Normal diametral pitch) must be one of: 4, 5, 6, 8, 10.")
    if not (10 <= Np1 <= 100):          #Np1 = pinion teeth number stage 1
        raise ValueError("Np1 (pinion teeth number stage 1) should be between 10 and 100.")
#    if not (10 <= Np2 <= 100):          #Np2 = pinion teeth number stage 2
#        raise ValueError("Np2 (pinion teeth number stage 2) should be between 10 and 100.")
    if Helix not in [15, 20, 25]:       #Helix = helix angle (degrees)
        raise ValueError("Helix (helix angle) should be 15, 20, or 25 degrees.")
      
    ''' Important Values '''
    P= wp / 240                                               #input power in HP
    Pd= Pnd * numpy.cos(numpy.radians(Helix))                 #Diametral pitch in teeth/inch               
    wf= wp / 12 + 100                                         #Ouput speed in RPM
    n= wp / wf                                                #overall ratio
    n2= n / n1                                                #stage 2 ratio
 

    ''' Intermediary Calculations 1 '''
    Ng1= n1 * Np1                                             #gear teeth number stage 1
    Dp1= Np1 / Pd                                             #pitch diameter stage 1 in inches
    vt1= numpy.pi * Dp1 * wp / 12                             #pitch line velocity in ft/s

    Kv=(c.C/(c.C+numpy.sqrt(vt1)))**(-c.B)                    #Dynamic factor

    Wt1= 33000 * P / vt1                                      #Tangential load in lbf

    Px= math.pi / (Pd * numpy.tan(numpy.radians(Helix)))      #Axial pitch in inches
    F= 0.25 * (math.ceil(2 * Px / 0.25))                      #Face width in inches, rounded up to nearest 0.25 inch

    # J_base 2D interpolation
    J_teeth= [20, 30, 60, 150, 500]
    J_angles= [15, 20, 25]
    J_table= [
    [0.46, 0.46, 0.44],
    [0.50, 0.49, 0.47],
    [0.541, 0.53, 0.51],
    [0.58, 0.56, 0.54],
    [0.595, 0.58, 0.55]
]
    interp_func_J = RegularGridInterpolator((J_teeth, J_angles), J_table)

    def J_base_interp(Np1, Helix):
        if Np1 < 20:
            return 0.4
        return float(interp_func_J([[Np1, Helix]]))

    J_base= J_base_interp(Np1, Helix)


    # K 2D interpolation
    K_teeth  = [20, 30, 50, 75, 150, 500]
    K_angles = [15, 20, 25]
    K_table = [
        [0.93,  0.935, 0.94],
        [0.955, 0.96,  0.961],
        [0.982, 0.983, 0.985],
        [1.00,  1.00,  1.00],
        [1.02,  1.02,  1.015],
        [1.035, 1.032, 1.03],
]
    interp_func_K = RegularGridInterpolator((K_teeth, K_angles), K_table)

    def k_interp(Ng1, Helix):
        if Ng1 < 20 or Ng1 > 500:
            return 1.04
        return float(interp_func_K([[Ng1, Helix]]))

    K= k_interp(Ng1, Helix)
    
    J=J_base * K

    # Km calculation
    Cma= 0.127 + 0.0158  * F - 1.093 * (10 ** -4) * F * F
    Cpf= F / (10 * Dp1) - 0.0375 + 0.0125 * F if F > 1 else F / (10 * Dp1) - 0.025
    Km= 1 + Cma + Cpf
    
    st1= c.Ko * c.Ks * Km * c.Kb * Wt1 * Kv * Pd / (F * J)    #Bending stress in psi
    Nc1=wp * c.L * 60                                         #number of cycles for stage 1
    Yn1=1.3558*Nc1**-0.0178                                   #Bending cycle factor for stage 1

    st1_ = st1 * c.SF * c.Kr / (1000*Yn1)                     #Bending stress for stage 1 in ksi
    sat = 36.8403
    print(fn.distance(st1_, sat),'%')
    return st1_                                               # in ksi



'''
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
'''