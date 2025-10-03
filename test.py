import numpy
import pandas as pd

def xlookup(lookup_value, lookup_column, return_column, csv_path):
    """
    Mimics Excel's XLOOKUP: finds lookup_value in lookup_column and returns the corresponding value from return_column.
    """
    df = pd.read_csv(csv_path)
    match = df[df[lookup_column] == lookup_value]
    if not match.empty:
        return match.iloc[0][return_column]
    else:
        return None  # or raise an error if preferred
    

