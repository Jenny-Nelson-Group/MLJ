import numpy as np
from typing import List, Union, Tuple
from MLJ.physics.config import config
import MLJ.physics.constants as const

def dark_population(state) -> np.ndarray:
    """
    Calculate the thermal (dark) population of a specific state.

    Uses the Boltzmann distribution to weight the Density of States (DoS) 
    based on the system temperature.

    Parameters
    ----------
    state : Object
        An object containing .density_of_states (np.ndarray) and 
        .energy (float or np.ndarray) in eV.

    Returns
    -------
    np.ndarray
        The Boltzmann-weighted population.
    """
    temperatures = config.temperatures_K
    kB = const.BOLTZMANN_CONSTANT_EV
    
    # Calculate boltzmann weighted DoS
    boltzmann_weighted_dos = state.density_of_states * np.exp(-state.energy / (kB * temperatures))
    
    #Square to obtain population
    return boltzmann_weighted_dos**2

def states_dark_population(
    states: List, 
    weights: List[float] = None
) -> Union[np.ndarray, List[np.ndarray]]:
    """
    Calculate the dark population for one or more states with optional weighting.
    """
    # Fix: Default weights to 1/N for each state
    if weights is None:
        n_states = len(states)
        weights = [1.0 / n_states] * n_states
    
    if len(states) != len(weights):
        raise ValueError(f"Mismatch: {len(states)} states but {len(weights)} weights.")

    # Calculate populations
    results = [w * dark_population(s) for s, w in zip(states, weights)]

    # Return single array if one state, else return the list
    return results[0] if len(results) == 1 else results