# Maybe a subpackage with windows should be done and manage the get_windows with cache on this level
## Maybe windows, preemphasis, etc can be applied at the pytorch datasaet layer => ¿defined here in numpy or there in torch?.

from functools import lru_cache
import numpy as np


@lru_cache(maxsize=10)
def get_windows(n, type_='hamming'):
    if type_ == 'hamming' : return hamming_window(n)
    elif type_ == 'blackman' : return blackman_window(n)
    else : raise Exception('Windows not contemplated in get_windows (¿yet?)')

# Hamming window
def hamming_window(n):
    coefs = np.arange(n)
    window = 0.54 - 0.46 * np.cos(2 * np.pi * coefs / (n - 1))
    return window

# Blackman window
def blackman_window(n, alpha=0.16):
    coefs = np.arange(n)
    window = (1 - alpha) / 2 - 0.5 * np.cos(2 * np.pi * coefs / (n - 1)) + alpha / 2 * np.cos(4 * np.pi * coefs / (n - 1))
    return window
