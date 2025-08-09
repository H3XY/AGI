# helper utilities for data shaping, serialization, etc.
import numpy as np

def ensure_vector(x, dim=2):
    x = np.asarray(x)
    if x.size == dim:
        return x
    v = np.zeros(dim)
    v[:min(dim, x.size)] = x.flatten()[:dim]
    return v
