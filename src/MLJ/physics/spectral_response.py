import numpy as np
from typing import Sequence


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


# Absorption spectrum: to be implemented
