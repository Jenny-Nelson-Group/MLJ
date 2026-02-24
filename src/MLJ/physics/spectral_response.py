import numpy as np
from typing import Sequence, Optional

from MLJ.physics.config import config
from MLJ.physics.constants import REDUCED_PLANCK_CONSTANT_JS, SPEED_OF_LIGHT, BOLTZMANN_CONSTANT_J


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
    optical_bandgap: float,
    spectral_absorption_rates: np.ndarray,
    temperatures_K: np.ndarray,
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
        photon_energies: 1D array of photon energies [J]. Shape: (n_E,)
        optical_bandgap: The energy gap Eg [J].
        spectral_absorption_rates: Transition rate density per unit energy.
            Shape: (n_states, n_E, n_T)
        temperatures_K: 1D array of temperatures [K]. Shape: (n_T,)
        weights: Statistical weights for each state. Defaults to 1/n_states.
        refractive_index: Real part of refractive index (n).
        device_thickness: Thickness of device (d) [m].

    Returns:
        alpha: Piecewise absorption coefficien. Shape: (n_E, n_T)
    """
    # 1. Dimensions & Fallbacks
    if spectral_absorption_rates.ndim != 3:
        raise ValueError(f"Expected 3D rates, got {spectral_absorption_rates.ndim}D")

    n_states, n_E, n_T = spectral_absorption_rates.shape

    # Use provided values or fall back to global config
    n = refractive_index if refractive_index is not None else config.refractive_index
    device_thickness = device_thickness if device_thickness is not None else config.device_thickness

    # These factors are needed to convert k_abs into the final form for alpha
    V_E = 1e-30  # 1 Angstrom^3 in m^3
    kB = BOLTZMANN_CONSTANT_J
    prefactor = (REDUCED_PLANCK_CONSTANT_JS**3 * SPEED_OF_LIGHT**2 * np.pi**2) / 2
    energy_scaling = (1.0 / photon_energies**2).reshape(
        1, -1, 1
    )  # Energy scaling: Shape (1, n_E, 1)

    # 3. Process Weights: Shape (n_states, 1, 1) for broadcasting
    if weights is None:
        weights = np.full(n_states, 1.0 / n_states)
    weights = np.asarray(weights).reshape(-1, 1, 1)

    # 4. Calculate Low-Energy Absorption (alpha_CT + alpha_ex equivalent)
    alpha_states = spectral_absorption_rates * n * energy_scaling * (1.0 / V_E) * prefactor
    # Sum over states -> Result shape: (n_E, n_T)
    alpha_low = np.sum(alpha_states * weights, axis=0)

    # 5. Calculate High-Energy Square-Root Law
    photon_energies = photon_energies.reshape(-1, 1)  # (n_E, 1)
    temperatures = temperatures_K.reshape(1, -1)  # (1, n_T)

    alpha_0 = 2.0 / device_thickness
    # Formula: alpha_0 * sqrt((hbar_omega - Eg) / (kB * T))
    # Set to zero via np.max to avoid imaginary frequencies
    sqrt_term = np.sqrt(np.maximum(photon_energies - optical_bandgap, 0) / (kB * temperatures))
    alpha_high = alpha_0 * sqrt_term

    # 6. Apply Piecewise Filter
    threshold = optical_bandgap + (2.0 * kB * temperatures)  # Shape (1, n_T)

    # E_hw < threshold compares (n_E, 1) with (1, n_T) to create (n_E, n_T) mask
    alpha = np.where(photon_energies < threshold, alpha_low, alpha_high)

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
