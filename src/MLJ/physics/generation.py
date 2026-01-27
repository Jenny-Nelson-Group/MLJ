import numpy as np
from typing import Callable
from MLJ.physics.basics import integral
from MLJ.physics.config import config

def laser_intensity_gaussian(photon_energies: np.ndarray, laser_power: float=None, laser_broadening: float=None ) -> np.ndarray:
    """
    Calculate the Gaussian laser intensity profile.

    Parameters
    ----------
    photon_energies : np.ndarray
        Array of photon energies at which to evaluate the intensity, in eV. 
        Shape: (N,).
    laser_power : float, optional
        Center energy (peak) of the laser in eV. If None, uses config.laser_power.
    laser_broadening : float, optional
        The characteristic width (standard deviation) of the Gaussian profile in eV.
        If None, uses config.laser_broadening.

    Returns
    -------
    np.ndarray
        The relative intensity profile [dimensionless]. 
        Shape: (N,), matching the length of photon_energies. 
        Peak intensity is centered at laser_power.
    """
    laser_power = config.laser_power if laser_power is None else laser_power 
    laser_broadening = config.laser_broadening if laser_broadening is None else laser_broadening 

    return np.exp(-((photon_energies - laser_power) / laser_broadening)**2)


def excited_state_generation(
    k_abs: np.ndarray, 
    photon_energies: np.ndarray, 
    laser_intenstiy_profile_func: Callable[[np.ndarray], np.ndarray] = laser_intensity_gaussian
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
        Defaults to `laser_intensity_gaussian`.

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
    
    return integral(k_abs * laser_intensity, photon_energies)
