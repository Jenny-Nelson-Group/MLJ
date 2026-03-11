import numpy as np
from typing import Sequence, Optional

from MLJ.physics.config import config
from MLJ.physics.constants import (
    REDUCED_PLANCK_CONSTANT_JS,
    SPEED_OF_LIGHT,
    BOLTZMANN_CONSTANT_J,
    UNIT_CHARGE,
)


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
    temperatures: np.ndarray,
    transitions: Optional[Sequence[float]],
    weights: Optional[Sequence[float]] = None,
    refractive_index: Optional[float] = None,
    device_thickness: Optional[float] = None,
) -> np.ndarray:
    """
    Calculate absorption coefficient alpha(hw) using a piecewise
    model combining transition rates and the square-root law.

    The model follows:
    - Below threshold (Eg + 2kbT): Weighted sum of rates that depend on states in system.
    - Above threshold: Square-root law for direct semiconductors.

    Args:
        photon_energies: 1D array of photon energies.
            Units: [eV]. Shape: (n_E,)
        spectral_absorption_rates: Transition rates
            for each state and temperature.
            Units: [1/s]. Shape: (n_states, n_E, n_T)
        temperatures: 1D array of temperatures.
            Units: [K]. Shape: (n_T,)
        transitions: Sequence of transition objects.
            Units: n.A. . Shape: (n_transitions,)
        weights: Statistical distribution weights for each transition state.
            Defaults to uniform distribution (1/n_states).
            Units: [Dimensionless]. Shape: (n_states,)
        refractive_index: Refractive Index of the system
            Units: [Dimensionless]. Shape: scalar float
        device_thickness: The active layer thickness (d) of the semiconductor.
            Units: [m]. Shape: scalar float

    Returns:
        alpha: The resulting absorption coefficient.
            Units: [1/m]. Shape: (n_E, n_T)
    """

    spectral_absorption_rates = np.asarray(spectral_absorption_rates)
    if spectral_absorption_rates.ndim != 3:
        raise ValueError(f"Expected 3D rates, got {spectral_absorption_rates.ndim}D")

    photon_energies_j = photon_energies * UNIT_CHARGE  # [J]
    n_states, n_E, n_T = spectral_absorption_rates.shape

    # Define external parameters that are independent of the transitions
    n = refractive_index if refractive_index is not None else config.refractive_index
    device_thickness = device_thickness if device_thickness is not None else config.device_thickness
    volume_of_molecular_site = config.volume_of_molecular_site

    # These factors are needed to convert k_abs into the final form for alpha
    prefactor = (
        REDUCED_PLANCK_CONSTANT_JS**3 * SPEED_OF_LIGHT**2 * np.pi**2
    ) / 2  # TODO! what is correct 1/2 or 1/4?
    # Energy scaling: Shape (1, n_E, 1)
    energy_scaling = (1.0 / photon_energies_j**2).reshape(1, -1, 1)

    # State Weights: Shape (n_states, 1, 1) for broadcasting
    if weights is None:
        weights = np.full(n_states, 1.0 / n_states)
    weights = np.asarray(weights).reshape(-1, 1, 1)

    # Calculate Low-Energy Absorption (e.g., alpha_CT + alpha_EX)
    alpha_states = (
        spectral_absorption_rates
        * n
        * energy_scaling
        * (1.0 / volume_of_molecular_site)
        * prefactor
    )
    # Sum over states -> Result shape: (n_E, n_T)
    alpha_low = np.sum(alpha_states * weights, axis=0)

    # Calculate High-Energy Square-Root Law
    photon_energies_j = photon_energies_j.reshape(-1, 1)  # (n_E, 1)
    temperatures = temperatures.reshape(1, -1)  # (1, n_T)

    # Determine threshold when Square root law holds
    energy_transitions = np.array(
        [transition.state_high_energy.energy for transition in transitions]
    )
    highest_energy_transition = transitions[np.argmax(energy_transitions)]
    optical_bandgap = (
        highest_energy_transition.state_high_energy.energy
        - highest_energy_transition.state_low_energy.energy
        + highest_energy_transition.lambda_outer
    )

    optical_bandgap_j = optical_bandgap * UNIT_CHARGE
    threshold = optical_bandgap_j + (2.0 * BOLTZMANN_CONSTANT_J * temperatures)
    high_energy_mask = photon_energies_j >= threshold  # (n_E, n_T)

    alpha_0 = 2.0 / device_thickness

    alpha = alpha_low.copy()
    E_grid, T_grid = np.broadcast_arrays(photon_energies_j, temperatures)

    # Square root behaviour of direct semicondcutor: alpha_0 * sqrt((hbar_omega - Eg) / (kB * T))
    alpha[high_energy_mask] = alpha_0 * np.sqrt(
        (E_grid[high_energy_mask] - optical_bandgap_j)
        / (BOLTZMANN_CONSTANT_J * T_grid[high_energy_mask])
    )
    return alpha


def absorptance(
    alpha: np.ndarray,
    device_thickness: Optional[float] = None,
) -> np.ndarray:
    """
    Calculate the spectral absorptance A(E) using the Beer-Lambert law.

    Args:
        alpha: Absorption coefficient [m^-1].
               Shape: (n_photon_energies, n_temps)
        device_thickness: Thickness of the device (d) [m].

    Returns:
        absorptance: Dimensionless fraction of absorbed photons [0, 1].
                     Shape: (n_photon_energies, n_temps)
    """
    d = device_thickness if device_thickness is not None else config.device_thickness

    # Absorptance shape: (n_E, n_T)
    absorptance = 1 - np.exp(-2 * d * alpha)

    return absorptance
