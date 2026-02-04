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
        transitions (Dict[Tuple[int, int], np.ndarray | float]): Mapping of
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
    transitions: Dict[Tuple[int, int], np.ndarray | float] = field(default_factory=dict)
    k_recombination: np.ndarray = field(init=False)
    full_system_matrix: np.ndarray = field(init=False)

    def __post_init__(self):
        # 1. Standardize recombination (k_recombination)
        # Check that all elements in rates and transitions have the same length
        all_rates = list(self.rates_to_ground) + list(self.transitions.values())
        if len({len(x) if isinstance(x, np.ndarray) else 1 for x in all_rates}) > 1:
            raise ValueError("Provided recombination rates have inconsistent lengths.")

        k_rec_ground = np.array([np.atleast_1d(r) for r in self.rates_to_ground])
        n_states, n_cond = k_rec_ground.shape

        # 2. Build transition tensor (n_states, n_states, n_temperatures)
        k_trans = np.zeros((n_states, n_states, n_cond))
        for (i, j), rate in self.transitions.items():
            if i != j:
                k_trans[i, j, :] = rate

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


def states_light_population(
    transition_matrix: TransitionMatrix,
    dark_population: Sequence[np.ndarray | float],
    generation_rate: Sequence[np.ndarray | float] | None = None,
) -> np.ndarray:
    """
    Solve steady-state rate equations for a multi-state system across all conditions.

    Parameters
    ----------
    transition_matrix : TransitionMatrix
        Assembled system matrix of shape (n_temperatures, n_states, n_states).
    dark_population : Sequence[np.ndarray | float]
        Thermal equilibrium populations for each state. Length must be n_states.
    generation_rate : Sequence[np.ndarray | float], optional
        External generation rates for each state. If None, defaults to zero.

    Returns
    -------
    np.ndarray
        Steady-state populations with shape (n_states, n_temperatures).

    Raises
    ------
    ValueError
        If input lengths or condition counts do not match the transition matrix.
    """
    # --- 1. Determine input shape consistency
    pop_dark = np.array([np.atleast_1d(p) for p in dark_population])

    if generation_rate is None:
        gen_rate = np.zeros_like(pop_dark)
    else:
        gen_rate = np.array([np.atleast_1d(g) for g in generation_rate])

    if pop_dark.shape != gen_rate.shape:
        raise ValueError("Shape of dark population and generation rates don't match.")

    if transition_matrix.shape != pop_dark.shape:
        raise ValueError("Transition Matrix dimensions and population don't match.")

    # --- 2. Create parameter arrays from input bundles
    recombination_rate = transition_matrix.k_recombination

    # --- 3. Build Source Vsector b (n_temperatures, n_states) and get system matrix
    source_terms = (gen_rate + recombination_rate * pop_dark).T[..., np.newaxis]
    system_matrix = transition_matrix.full_system_matrix
    result = np.linalg.solve(system_matrix, source_terms)
    return result.T.squeeze()
