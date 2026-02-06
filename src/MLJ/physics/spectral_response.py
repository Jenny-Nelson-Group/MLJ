import numpy as np
from typing import Sequence


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


# Absorption spectrum: to be implemented
