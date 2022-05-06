
import numpy as np


def normalized(y, threshold=0):
    y -= y.mean()
    stddev = y.std()
    if stddev > threshold:
        y /= stddev
    return y
