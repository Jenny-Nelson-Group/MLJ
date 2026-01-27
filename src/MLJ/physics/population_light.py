import numpy as np
from typing import Sequence
from dataclasses import dataclass


@dataclass
class StateBundle:
    """Represents a single physical state/level."""

    dark_population: np.ndarray | float
    generation_rate: np.ndarray | float
    recombination_rate: np.ndarray | float


class TransitionMatrix:
    def __init__(self, n_states: int, n_conditions: int):
        self.n_states = n_states
        self.n_conditions = n_conditions
        self.k_transition = np.zeros((n_states, n_states, n_conditions))
        self.k_recombination = np.zeros((n_states, n_conditions))

    def set_recombination(self, recombination_rates: Sequence[np.ndarray | float]):
        """Standardizes input sequence [krec1, krec2, ...] to (n_states, n_conditions)."""
        self.k_recombination = np.array(
            [np.broadcast_to(r, (self.n_conditions,)) for r in recombination_rates]
        )

    def set_transitions(self, rates_map: dict):
        """Pass a map {(from_idx, to_idx): rate_array (with length conditions)}."""
        for (i, j), rate in rates_map.items():
            if i != j:
                self.k_transition[i, j, :] = rate

    @property
    def full_system_matrix(self) -> np.ndarray:
        """
        Builds the Batch-First (M, L, L) matrix for the solver.
        A_ij = -k_{j->i}
        A_ii = k_recomb_i + sum_j(k_{i->j})
        """
        # 1. Total rate leaving each state i: shape (L, M)
        sum_out = self.k_transition.sum(axis=1)

        # 2. Build the (n_states, n_states, n_conditions) matrix
        # Swap -> A_ij = -K_ji
        transition_matrix = -self.k_transition.transpose(1, 0, 2)
        diag_idx = np.arange(self.n_states)
        transition_matrix[diag_idx, diag_idx, :] = self.k_recombination + sum_out

        # 3. Final transpose to Batch-First (n_conditions, n_states, n_states) for np.linalg.solve
        return np.moveaxis(transition_matrix, -1, 0)


def states_light_population(
    states: Sequence[StateBundle], transition_matrix: TransitionMatrix
) -> np.ndarray:
    """
    Solve steady-state rate equations for a collection of StateBundles.
    """
    # --- 1. Determine n_states and n_conditions
    n_states = len(states)
    if n_states == 0:
        raise ValueError("At least one StateBundle must be provided.")

    n_conditions = np.atleast_1d(states[0].dark_population).size

    # --- 2. Create parameter arrays from input bundles
    population_dark = np.array(
        [np.broadcast_to(s.dark_population, (n_conditions,)) for s in states]
    )
    generation_rate = np.array(
        [np.broadcast_to(s.generation_rate, (n_conditions,)) for s in states]
    )
    recombination_rate = transition_matrix.k_recombination

    # --- 3. Build Source Vector b (n_conditions, n_states) and get system matrix
    source_terms = (generation_rate + recombination_rate * population_dark).T
    system_matrix = transition_matrix.full_system_matrix

    return np.linalg.solve(system_matrix, source_terms).T.squeeze()
