import numpy
import pandas as pd
from typing import Literal


def xlookup(lookup_value, csv_path):
    """
    Mimics Excel's XLOOKUP: finds lookup_value in the first column and returns the corresponding value from the second column.
    """
    df = pd.read_csv(csv_path)
    lookup_column = df.columns[0]
    return_column = df.columns[1]
    match = df[df[lookup_column] == lookup_value]
    if not match.empty:
        return match.iloc[0][return_column]
    else:
        return None  # or raise an error if preferred
    




def distance(sigma_b, sigma_c):
    """
    assumin b and c are percentages (0-100)
    calculates the successfullness of the output
    """
    
    return (abs(100-sigma_b) + abs(100-sigma_c)) /2