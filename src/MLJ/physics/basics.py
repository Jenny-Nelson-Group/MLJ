import numpy as np
from scipy.special import genlaguerre
import MLJ.physics.constants as const
from functools import lru_cache

@lru_cache(maxsize=128)
def laguerre_2d(N_vib_initial, N_vib_final, huang_rhys):
    """Core physics calculation: generates the 2D Franck-Condon factor base."""
    laguerre_base = np.zeros((N_vib_initial + 1, N_vib_final + 1))
    for i in range(N_vib_initial + 1):
        j = np.arange(i, N_vib_final + 1)
        k = j - i
        poly = [genlaguerre(i, kk)(huang_rhys) for kk in k]
        laguerre_base[i, j] = poly

    laguerre_base.setflags(write=False)
    return laguerre_base

def gaussian(x, mean, sigma):
    """Returns non-normalised Gaussian or 1 if sigma is 0."""
    if sigma == 0:
        return np.array([1])
    else:
        gaussian_weight =  np.exp(-0.5 * ((x - mean) / sigma)**2)
        return gaussian_weight

def gaussian_norm(x, mean, sigma):
    if sigma == 0:
        return np.array([1.0])
    return (1.0 / (sigma * np.sqrt(2 * np.pi))) * np.exp(-0.5 * ((x - mean) / sigma)**2)

def dirac_delta(num: int, pos=0):
    """"""
    weights = np.zeros(num)
    delta_index = round((pos+1)/2 * (num-1))
    weights[delta_index] = 1
    return weights

def boltzmann(energy, temperature):
    """Calculate the Boltzmann factor with NumPy broadcasting."""
    energy = np.asarray(energy, dtype=float)
    temperature = np.asarray(temperature, dtype=float)
    return np.exp(
        -energy / (temperature * const.BOLTZMANN_CONSTANT_EV)
    )

def integral(y, x=None, axis=-1):
    y = np.asarray(y)

    if y.shape[axis] == 1:
        return np.squeeze(y, axis=axis)

    return np.trapezoid(y=y, x=x, axis=axis)