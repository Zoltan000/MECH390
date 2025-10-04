import numpy
import pandas as pd
from typing import Literal


ko_Weight = Literal["Uniform", "Light", "Medium", "Heavy"]
def ko_menu(output: ko_Weight, input: ko_Weight):
    match f"{output}/{input}":
        case "Uniform/Uniform":
            return 1
        case "Uniform/Light":
            return 1.25
        case "Uniform/Medium":
            return 1.5
        case "Uniform/Heavy":
            return 1.75
        case "Light/Uniform":
            return 1.2
        case "Light/Light":
            return 1.4
        case "Light/Medium":
            return 1.75
        case "Light/Heavy":
            return 2.25
        case "Medium/Uniform":
            return 1.3
        case "Medium/Light":
            return 1.7
        case "Medium/Medium":
            return 2
        case "Medium/Heavy":
            return 2.75
        case "Heavy/Uniform":
            return 1.4
        case "Heavy/Light":
            return 2
        case "Heavy/Medium":
            return 2.5
        case "Heavy/Heavy":
            return 3.25
        case _:
            raise ValueError("Invalid input combination for K0 factor.")
        

Kr_weight = Literal["0.9", "0.99", "0.999", "0.9999"]        
def Kr_menu(quality: Kr_weight):
    match quality:
        case "0.9":
            return 0.85
        case "0.99":
            return 1
        case "0.999":
            return 1.25
        case "0.9999":
            return 1.5
        case _:
            raise ValueError("wrong value bucko")



DL = Literal[
    "Domestic",
    "Aircraft Engines",
    "Automotive",
    "Agricultural Equipment",
    "Elevators, industrial fans, multipurpose gearing",
    "Electric motors, industrial blowers, general industrial machines",
    "Pumps and compressors",
    "Critical equipment in continuous 24-h operation"
]

def Design_Life(use: DL):
    match use:
        case "Domestic":
            return 1500
        case "Aircraft Engines":
            return 3000
        case "Automotive":
            return 3500
        case "Agricultural Equipment":
            return 5000
        case "Elevators, industrial fans, multipurpose gearing":
            return 12000
        case "Electric motors, industrial blowers, general industrial machines":
            return 25000
        case "Pumps and compressors":
            return 50000
        case "Critical equipment in continuous 24-h operation":
            return 150000
        case _:
            raise ValueError("Invalid input for Design Life.")

Avs = Literal["Av6", "Av7", "Av8", "Av9", "Av10", "Av11", "Av12"]
def B_menu(AVs: Avs):
    match AVs:
        case "Av6":
            return 6
        case "Av7":
            return 7
        case "Av8":
            return 8
        case "Av9":
            return 9
        case "Av10":
            return 10
        case "Av11":
            return 11
        case "Av12":
            return 12
        case _:
            raise ValueError("Invalid AV value.")