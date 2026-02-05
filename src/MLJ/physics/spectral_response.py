import numpy as np
from typing import Sequence, Optional

from MLJ.physics.config import config
from MLJ.physics.constants import REDUCED_PLANCK_CONSTANT_JS, SPEED_OF_LIGHT


def emission(
    populations: Sequence[np.ndarray | float],
    recombination_rates: Sequence[np.ndarray | float],
) -> np.ndarray:
    """
    Calculate total emission flux from state populations and recombination rates.

    Parameters
    ----------
    populations : Sequence[np.ndarray | float]
        population for each state. Can be a sequence of floats or 1D arrays.
    recombination_rates : Sequence[np.ndarray | float]
        Recombination rates for each state. Must match population shapes.

    Returns
    -------
    np.ndarray
        Total emission flux summed over all states. Shape: (n_conditions,).
    """
    # 1. Standardize both to (n_states, n_conditions)
    populations = np.array([np.atleast_1d(p) for p in populations])
    recombination_rates = np.array([np.atleast_1d(r) for r in recombination_rates])

    if populations.shape != recombination_rates.shape[::2]:
        raise ValueError(
            f"Shape mismatch: populations {populations.shape} vs rates {recombination_rates.shape}"
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
        Shape: (n_photons,)
    spectral_absorption_rates : np.ndarray
        The transition rate density per unit energy for each state/condition.
        Shape: (n_photons, n_conditions, n_states)
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
        Shape: (n_photons, n_conditions)
        Units: [m^-1] (if input units are SI).
    """
    n_energies, n_cond, n_states = spectral_absorption_rates.shape
    refractive_index = (
        refractive_index if refractive_index is not None else config.refractive_index
    )
    photon_density = (
        photon_density if photon_density is not None else config.photon_density
    )

    # Pre-calculated constants
    volume = 1e-30  # 1 Angstrom^3 in m^3
    numerical_pre_factor = (
        REDUCED_PLANCK_CONSTANT_JS**3 * SPEED_OF_LIGHT**2 * np.pi**2
    ) / 4

    if weights is None:
        w = np.full(n_states, 1.0 / n_states)
    else:
        w = np.asarray(weights)

    # Reshape energy to (n_energies, 1, 1)
    energy_scaling = (1.0 / photon_energies**2).reshape(-1, 1, 1)

    # Calculate state-specific alpha: (n_photons, n_conditions, n_states)
    alpha_states = (
        spectral_absorption_rates
        * refractive_index
        * energy_scaling
        * (1.0 / (photon_density * volume))
        * numerical_pre_factor
    )

    # 4. Weighted Sum over states (axis 2)
    # Resulting shape: (n_photons, n_conditions)
    total_absorption = np.sum(alpha_states * w, axis=2)

    return total_absorption


# TODO: Implement absorptance a
