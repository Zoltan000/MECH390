import math
import numpy
import constants as c
import functions as f
from scipy.interpolate import RegularGridInterpolator, interp2d
import functools as ft

@ft.cache
def important_values(wp,  n1, n2):
    '''
    # Input validation
    if not (1200 <= wp <= 3600):        #wp = input speed in RPM
        raise ValueError("wp (input speed) should be between 1200 and 3600 RPM.")
    if not (1 <= n1 <= 10):             #n1 = stage 1 input ratio
        raise ValueError("n1 (stage 1 input ratio) should be between 1 and 10.")
    if Pnd not in [4, 5, 6, 8, 10]:     #Pnd = Normal diametral pitch (teeth/inch)
        raise ValueError("Pnd (Normal diametral pitch) must be one of: 4, 5, 6, 8, 10.")
    if not (10 <= Np1 <= 100):          #Np1 = pinion teeth number stage 1
        raise ValueError("Np1 (pinion teeth number stage 1) should be between 10 and 100.")
    if Helix not in [15, 20, 25]:       #Helix = helix angle (degrees)
        raise ValueError("Helix (helix angle) should be 15, 20, or 25 degrees.")
    '''
        
    ''' Important Values '''
    nominal = False
    P= wp / 240 if not nominal else 10                        #input power in HP, if nominal is true then P=10HP
    wf= wp / 12 + 100 if not nominal else 250                 #Ouput speed in RPM, if nominal is true then wf=250RPM
    n= wp / wf                                                #overall ratio
    
    if n2 == None:
        nX= n / n1                                            #where nX is n2 for 2-stage
    else:
        nX = n / (n1 * n2)                                    #where nX is n3 for 3-stage

    '''
    print(f"\n=== important_values ===")
    print(f"Inputs: wp={wp}, n1={n1}, Pnd={Pnd}, Np1={Np1}, Helix={Helix}")
    print(f"P (power) = {P} HP")
    print(f"Pd (diametral pitch) = {Pd} teeth/inch")
    print(f"wf (output speed) = {wf} RPM")
    print(f"n (overall ratio) = {n}")
    print(f"n2 (stage 2 ratio) = {n2}")
    '''

    return P, wf, nX



@ft.cache
def intermediary_calculations(w,  nX, PdX, NpX, HelixX, P):

    ''' Intermediary Calculations '''    
    NgX= nX * NpX                                              #Number of Teeth of Stage X gear
    Dgx= NgX / PdX                                             #Diameter of Stage X gear in inches   
    DpX= NpX / PdX                                             #Diameter of Stage X pinion in inches
    vtX= numpy.pi * DpX * w / 12                               #pitch line velocity in ft/s

    Kv=(c.C/(c.C+numpy.sqrt(vtX)))**(-c.B)                     #Dynamic factor

    WtX= 33000 * P / vtX                                       #Tangential load in lbf

    Px= math.pi / (PdX * numpy.tan(numpy.radians(HelixX)))     #Axial pitch in inches
    F= 0.25 * (math.ceil(2 * Px / 0.25))                       #Face width in inches, rounded up to nearest 0.25 inch

    # Km calculation
    Cma= 0.127 + 0.0158  * F - 1.093 * (10 ** -4) * F * F
    Cpf= F / (10 * DpX) - 0.0375 + 0.0125 * F if F > 1 else F / (10 * DpX) - 0.025
    Km= 1 + Cma + Cpf

    NcX=w * c.L * 60                                           #number of cycles for stage 1
    '''
    print(f"\n=== intermediary_calculations ===")
    print(f"Ng1 (gear teeth stage 1) = {NgX}")
    print(f"Dg1 (gear pitch diameter stage 1) = {Dgx} inches")
    print(f"Dp1 (pitch diameter stage 1) = {DpX} inches")
    print(f"vt1 (pitch line velocity) = {vtX} ft/s")
    print(f"Kv (dynamic factor) = {Kv}")
    print(f"Wt1 (tangential load) = {WtX} lbf")
    print(f"Px (axial pitch) = {Px} inches")
    print(f"F (face width) = {F} inches")
    print(f"Cma = {Cma}")
    print(f"Cpf = {Cpf}")
    print(f"Km (load distribution factor) = {Km}")
    print(f"Nc1 (number of cycles stage 1) = {NcX}")
    '''
    return NgX, DpX, Kv, WtX, F, Km, NcX, Dgx




def bending_stress( Pd, Np1, Helix, Ng1, Kv, Wt1, F, Km, Nc1):

    ''' Gepmetry factor J calculation '''
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
    
    J=J_base * K                                              #Geometry factor

    ''' Bending Stress Calculation '''
    Yn1=1.3558*Nc1**-0.0178                                   #Bending cycle factor for stage 1
    
    st1= c.Ko * c.Ks * Km * c.Kb * Wt1 * Kv * Pd / (F * J)    #Bending stress in ksi
    st1_ = st1 * c.SF * c.Kr / (1000*Yn1)                     #Bending stress for stage 1 in ksi
    '''
    print(f"\n=== bending_stress ===")
    print(f"J_base = {J_base}")
    print(f"K = {K}")
    print(f"J (geometry factor) = {J}")
    print(f"Yn1 (bending cycle factor) = {Yn1}")
    print(f"st1 (raw bending stress) = {st1} ksi")
    print(f"st1_ (final bending stress) = {st1_} ksi")
    print(f"Constants: Ko={c.Ko}, Ks={c.Ks}, Kb={c.Kb}, SF={c.SF}, Kr={c.Kr}")
    '''
    return st1_                                               # in ksi



def contact_stress(n1, Pd, Np1, Helix, Dp1, Kv, Wt1, F, Km, Nc1, Dgx):
    
    ''' Pitting Resistance Factor I Calculation '''
    rp= Dp1 / 2                                             #Pitch radius in inches
    rg= Dgx / 2
    
    Tpressure =numpy.arctan((numpy.tan(numpy.radians(c.Pa))/numpy.cos(numpy.radians(Helix))))  #Transverse pressure angle in radians
    rbp= rp * numpy.cos(Tpressure) 
    rbg= rg * numpy.cos(Tpressure)

    Pnd= Pd / numpy.cos(numpy.radians(Helix))                     #Normal diametral pitch
    Z= math.sqrt((rp + 1/Pnd)**2 -rbp**2) + math.sqrt((rg + 1/Pnd)**2 - rbg**2) - (rp + rg) * numpy.sin(Tpressure)

    cP= math.pi * 2 * rp / Np1
    pN= cP * numpy.cos(numpy.radians(c.Pa))
    mN= pN / (0.95 * Z)
    I= (numpy.cos(Tpressure) * numpy.sin(Tpressure) * n1) / (2 * mN * (n1 + 1))        #Pitting resistance factor
    
    ''' Contact Stress Calculation '''
    Zn1= 1.4488 * Nc1**-0.023                                                          #Contact cycle factor for stage 1
    
    sc1= c.Cp * numpy.sqrt((Wt1 * c.Ko * c.Ks * Km * Kv)/(F * Dp1 * I))                #Contact stress in ksi
    sc1_= sc1 * c.SF * c.Kr / (1000 * Zn1)                                             #Contact stress for stage 1 in ksi

    '''
    print(f"\n=== contact_stress ===")
    print(f"Dg1 (gear pitch diameter stage 1) = {Dg1} inches")
    print(f"rp (pitch radius pinion) = {rp} inches")
    print(f"rg (pitch radius gear) = {rg} inches")
    print(f"Tpressure (transverse pressure angle) = {numpy.degrees(Tpressure)} degrees")
    print(f"rbp (base radius pinion) = {rbp} inches")
    print(f"rbg (base radius gear) = {rbg} inches")
    print(f"Z = {Z}")
    print(f"cP (circular pitch) = {cP}")
    print(f"pN (normal pitch) = {pN}")
    print(f"mN = {mN}")
    print(f"I (pitting resistance factor) = {I}")
    print(f"Zn1 (contact cycle factor) = {Zn1}")
    print(f"sc1 (raw contact stress) = {sc1} ksi")
    print(f"sc1_ (final contact stress) = {sc1_} ksi")
    print(f"Constants: Cp={c.Cp}, Ko={c.Ko}, Ks={c.Ks}, SF={c.SF}, Kr={c.Kr}")
    '''
    return sc1_  # in ksi

def results(wp,  n1, n2, Pd1, Np1, Helix1, Pd2, Np2, Helix2, Pd3, Np3, Helix3):
    n3 = None                                                                                        # gotta initiallize n3 so it stops complaining about it
    if n2 is None:
        skipStage3 = True                                                                           
        P, wf, n2 = important_values(wp,  n1, n2)                                                    # if we only have 2 stages, use the calculated n2
    else:
        skipStage3 = False
        P, wf, n3 = important_values(wp,  n1, n2)                                                    # if we have 3 stages, use the imported n2 and the calculated n3

    ''' Stage 1'''
    Ng1, Dp1, Kv1, Wt1, F1, Km1, Nc1, Dg1 = intermediary_calculations(wp,  n1, Pd1, Np1, Helix1, P)
    Stage1BendingStress = bending_stress(Pd1, Np1, Helix1, Ng1, Kv1, Wt1, F1, Km1, Nc1)
    Stage1ContactStress = contact_stress(n1, Pd1, Np1, Helix1, Dp1, Kv1, Wt1, F1, Km1, Nc1, Dg1)

    ''' Stage 2'''
    wi = wp / n1
    Ng2, Dp2, Kv2, Wt2, F2, Km2, Nc2, Dg2 = intermediary_calculations(wi,  n2, Pd2, Np2, Helix2, P)
    Stage2BendingStress = bending_stress(Pd2, Np2, Helix2, Ng2, Kv2, Wt2, F2, Km2, Nc2)
    Stage2ContactStress = contact_stress(n2, Pd2, Np2, Helix2, Dp2, Kv2, Wt2, F2, Km2, Nc2, Dg2)


    if skipStage3 == True:
        Volume = f.volume(Dp1, Dg1, Dp2, Dg2, None, None, F1, F2, None)
        return wf, P, Volume, Stage1BendingStress, Stage1ContactStress, Stage2BendingStress, Stage2ContactStress, None, None# if we only have 2 stages, skip stage 3
    
    ''' Stage 3'''
    wi2 = wi / n2
    Ng3, Dp3, Kv3, Wt3, F3, Km3, Nc3, Dg3 = intermediary_calculations(wi2,  n3, Pd3, Np3, Helix3, P)
    Stage3BendingStress = bending_stress(Pd3, Np3, Helix3, Ng3, Kv3, Wt3, F3, Km3, Nc3)
    Stage3ContactStress = contact_stress(n3, Pd3, Np3, Helix3, Dp3, Kv3, Wt3, F3, Km3, Nc3, Dg3)

    Volume = f.volume(Dp1, Dg1, Dp2, Dg2, Dp3, Dg3, F1, F2, F3)

    return wf, P, Volume, Stage1BendingStress, Stage1ContactStress, Stage2BendingStress, Stage2ContactStress, Stage3BendingStress, Stage3ContactStress