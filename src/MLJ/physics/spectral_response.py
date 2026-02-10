import numpy as np
from typing import Sequence, Optional

from MLJ.physics.config import config
from MLJ.physics.constants import REDUCED_PLANCK_CONSTANT_JS, SPEED_OF_LIGHT


def emission(
    populations: Sequence[np.ndarray],
    recombination_rates: Sequence[np.ndarray],
) -> np.ndarray:
    """
    Calculate total emission flux from state populations and recombination rates.

    Parameters
    ----------
    populations : Sequence[np.ndarray]
        population for each state. Can be a sequence of floats or 1D arrays.
        shape = (n_states, n_temperatures)
    recombination_rates : Sequence[np.ndarray]
        Recombination rates for each state. Must match population shapes.
        shape = (n_states, n_photon_energies, n_temperatures)

    Returns
    -------
    np.ndarray
        Total emission flux summed over all states. Shape: (n_photon_energies, n_temps,).
    """
    # 1. Check input shapes (n_states, n_conditions)
    populations = np.asarray(populations)
    recombination_rates = np.asarray(recombination_rates)

    # Check 1: Dimensions (2D and 3D)
    # Check 2: Match (M, N) with (M, _, N) using [::2]
    if (
        populations.ndim != 2
        or recombination_rates.ndim != 3
        or populations.shape != recombination_rates.shape[::2]
    ):
        raise ValueError(
            f"Input mismatch. Populations {populations.shape} must match "
            f"outer dimensions of Rates {recombination_rates.shape}."
        )

    # 2. Flux calculation: sum over states (axis 0)
    return np.sum(populations[:, None, :] * recombination_rates, axis=0)


def absorption(
    photon_energies: np.ndarray,
    spectral_absorption_rates: np.ndarray,
    weights: Optional[Sequence[float]] = None,
    refractive_index: Optional[float] = None,
    photon_density: Optional[float] = None,
) -> np.ndarray:
    """
    Calculate spectral absoprtion from absorption rates.

    Parameters
    ----------
    photon_energies : np.ndarray
        1D array of photon energies
        Shape: (n_photon_energies,)
    spectral_absorption_rates : np.ndarray
        The transition rate density per unit energy for each state/condition.
        Shape: (n_states, n_photon_energies, n_temps)
    weights : Sequence[float], optional
        Statistical weights for each electronic state.
        Defaults to uniform weighting (1/n_states).
        Shape: (n_states,)
    refractive_index : float, optional
        Real part of the complex refractive index. If None, retrieves from `config`.
    photon_density : float, optional
        Number of photons or sites per unit volume. If None, retrieves from `config`.
        Typical units: [m^-3].

    Returns
    -------
    total_absorption : np.ndarray
        The weighted macroscopic absorption coefficient alpha.
        Shape: (n_photon_energies, n_temps)
        Units: [m^-1] (if input units are SI).
    """
    # Check that the dimensionality is correct
    spectral_absorption_rates = np.asarray(spectral_absorption_rates)
    if spectral_absorption_rates.ndim != 3:
        raise ValueError(
            f"spectral_absorption_rate must have ndim=3,"
            f"but it has ndim={spectral_absorption_rates.ndim}"
        )

    n_states, _, _ = spectral_absorption_rates.shape
    refractive_index = refractive_index if refractive_index is not None else config.refractive_index
    photon_density = photon_density if photon_density is not None else config.photon_density

    # Pre-calculated constants
    volume = 1e-30  # 1 Angstrom^3 in m^3
    numerical_pre_factor = (REDUCED_PLANCK_CONSTANT_JS**3 * SPEED_OF_LIGHT**2 * np.pi**2) / 4

    weights = np.asarray(weights if weights is not None else [1 / n_states] * n_states)
    # Reshape weights to match dimensionality of n_states
    weights = weights.reshape(-1, 1, 1)

    # Reshape energy to (1, n_photon_energies, 1)
    energy_scaling = (1.0 / photon_energies**2).reshape(1, -1, 1)

    # Calculate state-specific alpha: (n_states, n_photon_energies, n_temps)
    alpha_states = (
        spectral_absorption_rates
        * refractive_index
        * energy_scaling
        * (1.0 / (photon_density * volume))
        * numerical_pre_factor
    )

    # Sum over states axis (axis 0) resulting shape: (n_photon_energies, n_temps)
    total_absorption = np.sum(alpha_states * weights, axis=0)

    return total_absorption


# TODO: Implement absorptance
