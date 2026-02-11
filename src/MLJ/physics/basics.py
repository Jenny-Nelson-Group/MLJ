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
        gaussian_weight = np.exp(-0.5 * ((x - mean) / sigma) ** 2)
        return gaussian_weight


def gaussian_norm(x, mean, sigma):
    if sigma == 0:
        return np.array([1.0])

    normalisation = 1.0 / (sigma * np.sqrt(2 * np.pi))
    return normalisation * np.exp(-0.5 * ((x - mean) / sigma) ** 2)


def dirac_delta(num: int, pos=0):
    """"""
    weights = np.zeros(num)
    delta_index = round((pos + 1) / 2 * (num - 1))
    weights[delta_index] = 1
    return weights


def current_to_electrons(current):
    """
    Returns number of electrons flowing in a given current.

    Parameters
    ----------
    current : float
        Current [A]

    Returns
    -------
    float
        Number of electrons per second [1/s]
    """
    return current / const.UNIT_CHARGE


def boltzmann(energy, temperature):
    """
    Calculate the Boltzmann factor with automatic 1D-to-2D grid broadcasting.

    If both inputs are 1D arrays, they are automatically broadcast into a
    2D grid of shape (E, T). Otherwise, standard NumPy broadcasting rules apply.

    Parameters
    ----------
    energy : float or array_like
        Energy values [eV]. If 1D of length N, and temperature is 1D
        of length M, result is broadcast to (N, M).
    temperature : float or array_like
        Temperature values [K].

    Returns
    -------
    ndarray or float
        The Boltzmann factor exp(-E / kT). Shape follows standard
        broadcasting, except for the 1D-1D case which returns
        (n_energies, n_temps) for convenience.
    """
    energy = np.asarray(energy, dtype=float)
    temperature = np.asarray(temperature, dtype=float)

    # Automatic broadcasting ONLY if both are raw 1D arrays
    if energy.ndim == 1 and temperature.ndim == 1:
        energy = energy[:, np.newaxis]
        temperature = temperature[np.newaxis, :]

    return np.exp(-energy / (temperature * const.BOLTZMANN_CONSTANT_EV))


def integral(y, x=None, axis=-1):
    y = np.asarray(y)

    if y.shape[axis] == 1:
        return np.squeeze(y, axis=axis)

    return np.trapezoid(y=y, x=x, axis=axis)
