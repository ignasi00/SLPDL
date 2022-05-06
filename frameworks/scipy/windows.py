# Maybe a subpackage with windows should be done and manage the get_windows with cache on this level
## Maybe windows, preemphasis, etc can be applied at the pytorch datasaet layer => ¿defined here in numpy or there in torch?.

from functools import lru_cache
import numpy as np


@lru_cache(maxsize=10)
def get_windows(n, type_='hamming'):
    if type_ == 'hamming' : return hamming_window(n)
    #elif type_ == '' : return f(n)
    else : raise Exception('Windows not contemplated in get_windows (¿yet?)')

# Hamming window
def hamming_window(n):
    coefs = np.arange(n)
    window = 0.54 - 0.46 * np.cos(2 * np.pi * coefs / (n - 1))
    return window
