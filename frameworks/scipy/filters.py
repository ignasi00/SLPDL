
from functools import lru_cache
import numpy as np


def apply_preemphasis(y, preemCoef=0.97):
    y[1:] = y[1:] - preemCoef * y[:-1]
    y[0] *= (1 - preemCoef)
    return y
