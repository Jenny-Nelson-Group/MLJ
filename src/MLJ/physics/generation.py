import numpy as np
from typing import Callable
from MLJ.physics.basics import integral, gaussian
from MLJ.physics.config import config


def laser_profile_gaussian(
    photon_energies: np.ndarray,
    laser_mean_energy: float = None,
    laser_broadening: float = None,
) -> np.ndarray:
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
    laser_mean_energy = (
        config.laser_mean_energy if laser_mean_energy is None else laser_mean_energy
    )
    laser_broadening = (
        config.laser_broadening if laser_broadening is None else laser_broadening
    )

    return gaussian(x=photon_energies, mean=laser_mean_energy, sigma=laser_broadening)


def excited_state_generation(
    k_abs: np.ndarray,
    photon_energies: np.ndarray,
    laser_intenstiy_profile_func: Callable[
        [np.ndarray], np.ndarray
    ] = laser_profile_gaussian,
) -> np.ndarray:
    """
    Calculate the total excited state generation rate by integrating the product
    of the absorption rate and the laser intensity profile over the energy spectrum.

    Parameters
    ----------
    k_abs : np.ndarray
        Energy-dependent absorption rate.
        Shape: (n_states, n_photon_energies, n_temperatures)
    photon_energies : np.ndarray
        Energy grid in eV.
        Shape: (n_photon_energies,).
    laser_intenstiy_profile_func : Callable
        A function that accepts photon_energies (n_photon_energies,) and returns a
        dimensionless relative intensity profile (n_photon_energies,).
        Defaults to `laser_profile_gaussian`.

    Returns
    -------
    np.ndarray
        Total excited state generation rate [states/second] due to input spectrum.
        This represents the spectral overlap integral between the absorption
        rate/cross-section and the laser source.
        Shape: (n_states, n_temperatures)
    """
    # Validate k_abs shape against photon_energies
    k_abs = np.asarray(k_abs)
    if k_abs.ndim != 3 or k_abs.shape[1] != len(photon_energies):
        raise ValueError(
            "k_abs must have shape (n_states, n_photon_energies, n_temperatures) and "
            "match photon_energies: got "
            f"ndim={k_abs.ndim}, shape={k_abs.shape}, len(photon_energies)={len(photon_energies)}."
        )

    # laser_intensity shape: (n_photon_energies,)
    laser_intensity = laser_intenstiy_profile_func(photon_energies)

    # Define integrand for the entire spectrum for every state and every temperature
    integrand = k_abs * laser_intensity[None, :, None]

    # Integrate over photon_energies (axis 1), return (n_states, n_temperatures)
    return integral(integrand, photon_energies, axis=1)
