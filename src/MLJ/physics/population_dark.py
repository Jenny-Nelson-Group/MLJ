import numpy as np
from typing import Sequence
from MLJ.physics.config import config
from MLJ.physics.basics import boltzmann
import MLJ.physics.constants as const


def dark_population(state, temperatures: np.ndarray = None) -> np.ndarray:
    """
    Calculate the thermal (dark) population of a specific state.

    Uses the Boltzmann distribution to weight the scalar Density of States (DoS)
    based on the system temperature.

    Parameters
    ----------
    state : Object
        An object representing the electronic state.
    temperatures : np.ndarray, optional
        Array of temperatures in Kelvin. Shape: (M,).
        If None, defaults to config.temperatures_K.

    Returns
    -------
    np.ndarray
        The Boltzmann-weighted population [dimensionless/normalized density].
        Shape: (M,), where each element corresponds to a temperature in the
        input 'temperatures' array.
    """
    temperatures = config.temperatures_K if temperatures is None else temperatures

    # Calculate boltzmann weighted DoS and Square to obtain population
    boltzmann_factor = boltzmann(state.energy, temperatures)
    return (state.density_of_states * boltzmann_factor)**2

def states_dark_population(
    states: Sequence,
    temperatures: np.ndarray | None = None,
    weights: Sequence[float] | None = None,
) -> np.ndarray:
    """
    Calculate the dark population for one or more states with optional weighting.

    Parameters
    ----------
    states : Sequence[Object]
        A sequence of state objects.
    temperatures : np.ndarray, optional
        Array of temperatures in Kelvin. Shape: (M,).
        If None, defaults to config.temperatures_K.
    weights : Sequence[float], optional
        Scalar weights for each state in `states`. If None, defaults to 1/N
        weighting. Must be the same length as `states`.

    Returns
    -------
    np.ndarray
        Array of shape (states.length, temperatures.length).
        Each row contains the state's population [dimensionless] across all temperatures.
    """
    temperatures = config.temperatures_K if temperatures is None else temperatures

    # Check that at least one state is given
    n_states = len(states)
    if n_states == 0:
        raise ValueError(f"At least one state must be given.")

    # assign even weights, if no weights are given, and check their shape
    if weights is None:
        weights_arr = np.full(n_states, 1.0 / n_states, dtype=float)
    else:
        weights_arr = np.asarray(weights, dtype=float)

    if weights_arr.shape != (n_states,):
        raise ValueError(f"Mismatch: {n_states} states but {weights_arr.size} weights.")

    # Calculate populations of each state
    populations = np.stack([dark_population(s, temperatures) for s in states], axis=0)

    # Return weighted populations (via broadcasting)
    return populations * weights_arr[:, None]