import numpy as np
from typing import Sequence
from MLJ.physics.config import config
from MLJ.physics.basics import boltzmann
from MLJ.physics.state import State


def states_dark_population(
    states: Sequence[State] | State,
    temperatures: np.ndarray | None = None,
    weights: Sequence[float] | None = None,
    voltage: float = 0.0,
) -> np.ndarray:
    """
    Calculate the dark population for one or more states with optional weighting and applied bias.

    Parameters
    ----------
    states : Sequence[State]
        Sequence of State objects. Shape = (n_states,)
    temperatures : np.ndarray, optional
        Array of temperatures in Kelvin. Shape = (n_temperatures,)
        If None, defaults to config.temperatures_K.
    weights : Sequence[float], optional
        Scalar weights per state. Defaults to 1/N.
    voltage : float
        Applied bias to push population out of equilibrium.

    Returns
    -------
    np.ndarray
        Array of shape (n_states, n_temperatures).
        Each row contains the state's population [dimensionless] across all temperatures.
    """
    # 1. Handle input values
    states = np.asarray(states)
    n_states = len(states)

    temperatures = config.temperatures_K if temperatures is None else temperatures
    energies, dos = np.array(
        [(s.energy, s.density_of_states) for s in states]
    ).T  # (n_states)
    weights = np.asarray(weights if weights is not None else [1 / n_states] * n_states)

    if len(weights) != n_states:
        raise ValueError(f"Mismatch: {n_states} states but {len(weights)} weights.")

    # calculate and return weighted dark populations
    populations = dos[:, None] * boltzmann(
        energies - voltage, temperatures
    )  # (n_states,n_temps)
    return populations * weights[:, None]
