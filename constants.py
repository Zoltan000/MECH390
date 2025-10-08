import numpy as np
import pandas as pd
import lookupTables as lt

'''Constants that don't change'''
Ks=1                                 #Size Factor (Assuming Pd always > 5)
Kb=1                                 #Rim Thickness Factor (Assuming solid gears)
Cp=2300                              #Elastic Coefficient for steel on steel [sqrt(psi)]
Pa=20                                #Pressure Angle (degrees)

'''Constants that do change'''
SF=1                                 #Service Factor (pick a number between 1 and 1.5)
L=lt.Design_Life("Electric motors, industrial blowers, general industrial machines")       #design life in [insert unit]
Ko=lt.ko_menu("Light", "Light")      #Overload Factor (pick a combination from the table)
Kr=lt.Kr_menu("0.999")               #Reliability Factor (pick a number from the table)

#Kv Constants
B= 0.25*(lt.B_menu("Av7")-5)**0.667  #who knows what these are
C=50+56*(1-B)                        #Antony never labeled them in the excel
