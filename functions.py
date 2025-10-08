import numpy
import pandas as pd
from typing import Literal


def distance(a, b):
    return ((abs(a - b) / ((a + b) / 2)) * 100)

