import numpy as np
import MLJ.physics.constants as const

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


def boltzmann(energy, temperature):
    return np.exp(
        -energy[:, None] / (temperature[None, :] * const.BOLTZMANN_CONSTANT_EV)
    )

def integral(y, x):
    if y.shape[0] == 1:
        return y[0]
    
    return np.trapezoid(y, x=x, axis=0)