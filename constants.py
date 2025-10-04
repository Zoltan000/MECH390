import numpy as np
import pandas as pd
import lookupTables as lt

# Constants that don't change
L=lt.Design_Life("Domestic")                #design life in [insert unit]


# Constants that do change
SF=1                    #Service Factor (pick a number between 1 and 1.5)
ko=lt.ko_menu("Light", "Light")
Kr=lt.Kr_menu(0.99)
