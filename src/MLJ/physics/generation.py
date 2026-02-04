import numpy as np
from typing import Callable
from MLJ.physics.basics import integral, gaussian
from MLJ.physics.config import config

def laser_profile_gaussian(photon_energies: np.ndarray, laser_mean_energy: float=None, laser_broadening: float=None ) -> np.ndarray:
    """
    Calculate the Gaussian laser intensity profile.

    Parameters
    ----------
    photon_energies : np.ndarray
        Array of photon energies at which to evaluate the intensity, in eV.
        Shape: (N,).
    laser_mean_energy : float, optional
        Center energy (peak) of the laser in eV. If None, uses config.laser_mean_energy.
    laser_broadening : float, optional
        The characteristic width (standard deviation) of the Gaussian profile in eV.
        If None, uses config.laser_broadening.

    Returns
    -------
    np.ndarray
        The relative intensity profile [dimensionless].
        Shape: (N,), matching the length of photon_energies.
        Peak intensity is centered at laser_mean_energy.
    """
    laser_mean_energy = config.laser_mean_energy if laser_mean_energy is None else laser_mean_energy
    laser_broadening = config.laser_broadening if laser_broadening is None else laser_broadening

    return gaussian(x=photon_energies, mean=laser_mean_energy, sigma=laser_broadening)

def excited_state_generation(
    k_abs: np.ndarray,
    photon_energies: np.ndarray,
    laser_intenstiy_profile_func: Callable[[np.ndarray], np.ndarray] = laser_profile_gaussian
) -> np.ndarray:
    """
    Calculate the total excited state generation rate by integrating the product
    of the absorption rate and the laser intensity profile over the energy spectrum.

    Parameters
    ----------
    k_abs : np.ndarray
        Energy-dependent absorption rate.
        Shape: 2D array (N_photon_energies, N_temperatures).
    photon_energies : np.ndarray
        Energy grid in eV.
        Shape: (N,).
    laser_intenstiy_profile_func : Callable
        A function that accepts photon_energies (N,) and returns a
        dimensionless relative intensity profile (N,).
        Defaults to `laser_profile_gaussian`.

    Returns
    -------
    np.ndarray
        Total excited state generation rate [states/second] due to input spectrum.
        This represents the spectral overlap integral between the absorption
        rate/cross-section and the laser source.
        Shape: (N_temperatures, )
    """
    # Generate the intensity profile using the passed function
    laser_intensity = laser_intenstiy_profile_func(photon_energies)
    integrand = k_abs.reshape(k_abs.shape[0], -1) * laser_intensity[:,None]
    return integral(integrand, photon_energies, axis=0)
