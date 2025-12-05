import numpy as np
import MLJ.physics.constants as const

def gaussian(x, mean, sigma):
    return np.exp(-0.5 * ((x - mean) / sigma)**2)


def boltzmann(energy, temperature):
    return np.exp(
        -energy[:, None] / (temperature[None, :] * const.BOLTZMANN_CONSTANT_EV)
    )
