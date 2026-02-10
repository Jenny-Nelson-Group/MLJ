import numpy as np
from typing import Sequence, Dict, Tuple
from dataclasses import dataclass, field


@dataclass(frozen=True)
class TransitionMatrix:
    """
    An immutable representation of a state transition system for steady-state solvers.

    This class standardizes recombination rates and transition tensors into a
    batch-first system matrix suitable for `np.linalg.solve`. It automatically
    infers dimensions from the provided input sequences.

    Attributes:
        rates_to_ground (Sequence[np.ndarray | float]): Recombination rates to ground for each state.
            If arrays are provided, their length defines 'n_temperatures'.
        transfers (Dict[Tuple[int, int], np.ndarray | float]): Mapping of
            (from_state, to_state) indices to transition rates.
        k_recombination (np.ndarray): Standardized recombination rates with
            shape (n_states, n_temperatures).
        full_system_matrix (np.ndarray): The assembled batch-first matrix
            for the solver with shape (n_temperatures, n_states, n_states).

    Notes:
        The system matrix $A$ is constructed such that for each condition $m$:
        - Off-diagonal: $A_{i,j} = -k_{j \to i}$
        - Diagonal: $A_{i,i} = k_{rec, i} + \sum_{j \neq i} k_{i \to j}$
    """

    rates_to_ground: Sequence[np.ndarray | float]
    transfers: Dict[Tuple[int, int], np.ndarray | float] | None = None
    k_recombination: np.ndarray = field(init=False)
    full_system_matrix: np.ndarray = field(init=False)

    def __post_init__(self):
        # 1. Standardize recombination (k_recombination)
        # Check that all elements in rates and transfers have the same length
        transfers = self.transfers or {}
        all_rates = list(self.rates_to_ground) + list(transfers.values())
        if len({len(x) if isinstance(x, np.ndarray) else 1 for x in all_rates}) > 1:
            raise ValueError("Provided recombination rates have inconsistent lengths.")

        k_rec_ground = np.array([np.atleast_1d(r) for r in self.rates_to_ground])
        n_states, n_cond = k_rec_ground.shape

        # 2. Build transition tensor (n_states, n_states, n_temperatures)
        k_trans = np.zeros((n_states, n_states, n_cond))
        for (i, j), rate in transfers.items():
            # Only map transitions between excited states (i > 0 and j > 0)
            if i > 0 and j > 0:
                if i != j:
                    k_trans[i - 1, j - 1, :] = rate

        # 3. Assemble system matrix A
        sum_out = k_trans.sum(axis=1)
        sys_mat = -k_trans.transpose(1, 0, 2)
        diag_idx = np.arange(n_states)
        sys_mat[diag_idx, diag_idx, :] = k_rec_ground + sum_out

        # 4. Finalize Batch-First Matrix: (n_temperatures, n_states, n_states)
        full_system_matrix = np.moveaxis(sys_mat, -1, 0)

        # Safety: Lock the internal arrays
        k_rec_ground.flags.writeable = False
        full_system_matrix.flags.writeable = False

        # Apply to frozen instance
        object.__setattr__(self, "k_recombination", k_rec_ground)
        object.__setattr__(self, "full_system_matrix", full_system_matrix)

    @property
    def shape(self) -> Tuple[int, int]:
        """Returns the shape as Tuple (n_states, n_temperatures)."""
        n_cond, n_states, _ = self.full_system_matrix.shape
        return (n_states, n_cond)


def solve_population(
    transition_matrix: TransitionMatrix,
    dark_population: Sequence[np.ndarray],
    generation_rate: Sequence[np.ndarray] | None = None,
) -> np.ndarray:
    """
    Solve steady-state rate equations for a multi-state system across all conditions.

    Parameters
    ----------
    transition_matrix : TransitionMatrix
        Assembled system matrix of shape (n_temperatures, n_states, n_states).
    dark_population : Sequence[np.ndarray]
        Thermal equilibrium populations for each state.
        Shape = (n_states, n_temperatures)
    generation_rate : Sequence[np.ndarray], optional
        External generation rates for each state. If None, defaults to zero.
        Shape = (n_states, n_temperatures)

    Returns
    -------
    np.ndarray
        Steady-state populations with shape (n_states, n_temperatures).

    Raises
    ------
    ValueError
        If input lengths or condition counts do not match the transition matrix.
    """
    # ensure inputs are arrays
    dark_population = np.asarray(dark_population)
    if generation_rate is None:
        generation_rate = np.zeros_like(dark_population)
    else:
        generation_rate = np.asarray(generation_rate)

    # --- 1. Determine input shape consistency
    if dark_population.ndim != 2:  # (n_states, n_temps)
        raise ValueError(
            f"dark_population must be 2D (n_states, n_temps). "
            f"Got {dark_population.ndim}D with shape {dark_population.shape}."
        )

    if len({dark_population.shape, generation_rate.shape, transition_matrix.shape}) > 1:
        raise ValueError(
            f"Input dimensions must match (n_states, n_temps)."
            f"Got {dark_population.shape}, {generation_rate.shape}, and {transition_matrix.shape}."
        )

    # --- 2. Prepare terms:
    recombination_rate = transition_matrix.k_recombination  # (n_states, n_temps)
    source_terms = (generation_rate + recombination_rate * dark_population).T[
        ..., np.newaxis
    ]  # (n_temps, n_states)
    system_matrix = transition_matrix.full_system_matrix  # (n_temps, n_states)

    # --- 3. Solve Ax = b,  x = result (populations)
    # system_matrix A: (n_temps, n_states, n_states)
    # source_terms b:  (n_temps, n_states, 1)
    # solution x:      (n_temps, n_states, 1)
    solution = np.linalg.solve(system_matrix, source_terms)  # (n_temps, n_states, 1)
    return solution[..., 0].T  # (n_states, n_temps)
