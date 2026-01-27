import numpy as np
from typing import Sequence


def states_light_population(
    dark_populations: Sequence,
    generation_rates: Sequence,
    recombination_rates: Sequence,
    transition_rates: Sequence | None = None,
) -> np.ndarray:
    n_states = len(dark_populations)

    if n_states == 0:
        raise ValueError("At least one state must be given.")

    if len(generation_rates) != n_states or len(recombination_rates) != n_states:
        raise ValueError(
            "dark_populations, generation_rates, recombination_rates must have the same length."
        )

    if n_states == 1:
        population = rate_equation_one_state(
            dark_populations[0],
            generation_rates[0],
            recombination_rates[0],
        )
        return np.stack([population], axis=0)

    if n_states == 2:
        if transition_rates is None or len(transition_rates) != 2:
            raise ValueError(
                "For two states, transition_rates must be a sequence of length 2."
            )

        return rate_equation_two_states(
            dark_populations[0],
            dark_populations[1],
            generation_rates[0],
            generation_rates[1],
            recombination_rates[0],
            recombination_rates[1],
            transition_rates[0],
            transition_rates[1],
        )

    raise NotImplementedError("Only 1- and 2-state systems are implemented.")


def rate_equation_one_state(population_dark, generation, k_total_recombination):
    """Solve the rate equation for one state."""
    population_light = population_dark + generation / k_total_recombination
    return population_light


def rate_equation_two_states(
    state_1_population_dark,
    state_2_population_dark,
    state_1_generation,
    state_2_generation,
    state_1_k_total_recombination,
    state_2_k_total_recombination,
    k_state_1_to_state_2,
    k_state_2_to_state_1,
):
    population_light_state_1 = (
        state_1_generation
        + state_2_generation
        + state_2_k_total_recombination * state_2_population_dark
        + (state_2_k_total_recombination / k_state_2_to_state_1)
        + (
            state_2_k_total_recombination
            * state_1_k_total_recombination
            / k_state_2_to_state_1
        )
        * state_1_population_dark
    ) / (
        state_1_k_total_recombination
        + state_2_k_total_recombination * k_state_1_to_state_2 / k_state_2_to_state_1
        + state_2_k_total_recombination
        * state_1_k_total_recombination
        / k_state_2_to_state_1
    )

    population_light_state_2 = (
        state_2_generation
        + state_1_generation
        + state_1_k_total_recombination * state_1_population_dark
        + (state_1_k_total_recombination / k_state_1_to_state_2)
        + (
            state_1_k_total_recombination
            * state_2_k_total_recombination
            / k_state_1_to_state_2
        )
        * state_2_population_dark
    ) / (
        state_2_k_total_recombination
        + state_1_k_total_recombination * k_state_2_to_state_1 / k_state_1_to_state_2
        + state_1_k_total_recombination
        * state_2_k_total_recombination
        / k_state_1_to_state_2
    )

    return np.array([population_light_state_1, population_light_state_2])
