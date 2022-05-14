
from numba import jit
import numpy as np
import scipy.linalg
import scipy.spatial


@jit
def dtw(x, y, metric='sqeuclidean'):
  """
  Computes Dynamic Time Warping (DTW) of two sequences.
  :param array x: N1*M array
  :param array y: N2*M array
  :param func dist: distance used as cost measure
  """
  r, c = len(x), len(y)

  D = np.zeros((r + 1, c + 1))
  D[0, 1:] = np.inf
  D[1:, 0] = np.inf

  # Initialize the matrix with dist(x[i], y[j])
  D[1:, 1:] = scipy.spatial.distance.cdist(x, y, metric)

  for i in range(r):
    for j in range(c):
      min_prev = min(D[i, j], D[i+1, j], D[i, j+1])
      # D[i+1, j+1] = dist(x[i], y[j]) + min_prev
      D[i+1, j+1] += min_prev


    if len(x) == 1:
        path = zeros(len(y)), range(len(y))
    elif len(y) == 1:
        path = range(len(x)), zeros(len(x))
    else:
        path = _traceback(D)
    return D[-1, -1], path

def _traceback(D):
    i, j = np.array(D.shape) - 2

    path = [(i, j)]
    
    while (i > 0) or (j > 0):
        tb = np.argmin((D[i, j], D[i, j + 1], D[i + 1, j]))

        if tb == 0:
            i -= 1
            j -= 1
        elif tb == 1:
            i -= 1
        else:  # (tb == 2):
            j -= 1
        
        path.insert(0, (i, j))

    return np.array(path)
