# src/MLJ/transition.py
#####################################################################################
# MLJ Package
#
# Classes containing core parameters of quantum states.
# Author: Jolanda S Müller, Tim Rein, Imperial College London
# Copyright (c) 2025, Imperial College London, BSD 3-Clause License
# Date: November 2025
#####################################################################################

from MLJ.physics.state import State
import MLJ.physics.coupling as cpl
from enum import Enum
import numpy as np
from functools import cached_property
from MLJ.helpers.caching import ReactiveModule


class ProcessType(Enum):
    """Enum class to distinguish the different transition types."""

    ABSORPTION = "absorption"
    RECOMBINATION = "recombination"


class Transition(ReactiveModule):
    GROUND_STATE = State(energy=0.0, name="Ground State")

    """Represents a transition between two quantum states."""

    def __init__(
        self,
        state_high_energy: State = None,
        state_low_energy: State = None,
        oscillator_strength: float = 1,
        static_dipole_moment: float = 3 * 3.33e-30 / 1.6e-19,
        lambda_inner: float = 0.02,
        lambda_outer: float = 0.02,
        k_transfer: np.ndarray | None = None,
    ) -> None:
        if state_low_energy is None and state_high_energy is None:
            raise ValueError("At least one State must be given.")

        state_low_energy = state_low_energy or Transition.GROUND_STATE
        state_high_energy = state_high_energy or Transition.GROUND_STATE

        self.determine_high_low_energy_state(state_low_energy, state_high_energy)

        self.oscillator_strength: float = oscillator_strength
        self.static_dipole_moment: float = static_dipole_moment

        self.lambda_inner: float = lambda_inner
        self.lambda_outer: float = lambda_outer

        self.electronic_coupling_rad_func = cpl.transition_dipole_moment
        self.electronic_coupling_nrad_func = cpl.mulliken_hush_coupling

        self.k_transfer = np.atleast_1d(k_transfer) if k_transfer is not None else None

        self.start_caching()

    def determine_high_low_energy_state(self, state_low_energy, state_high_energy):
        """Assign which state is the higher energy state and which state is the lower energy state."""
        if state_high_energy.energy > state_low_energy.energy:
            self.state_low_energy: State = state_low_energy
            self.state_high_energy: State = state_high_energy
        else:
            self.state_low_energy: State = state_high_energy
            self.state_high_energy: State = state_low_energy

    @cached_property
    def electronic_coupling_radiative(self):
        """Lazy-calculated radiative coupling."""
        # Using @property here means Rate functions just call `transition.coupling_rad`
        return self.electronic_coupling_rad_func(self)

    @cached_property
    def electronic_coupling_non_radiative(self):
        """Lazy-calculated non-radiative coupling."""
        # By default, cpl.coupling_strength_nrad will call transition.coupling_rad internally
        return self.electronic_coupling_nrad_func(self)

    @cached_property
    def mean_gibbs_energy(self) -> float:
        """Lazily calculate the electronic energy difference."""
        return self.state_high_energy.energy - self.state_low_energy.energy

    @cached_property
    def gibbs_energy_grid(self) -> np.ndarray:
        """Lazily calculate the energy grid for disorder integration."""
        return self.state_high_energy.energy_grid - self.state_low_energy.energy

    @property
    def disorder_weights(self):
        return self.state_high_energy.disorder_weights

    @property
    def disorder_distribution(self):
        """Return the disorder distribution of the high energy state."""
        return self.state_high_energy.disorder_distribution

    @cached_property
    def huang_rhys(self):
        """Calculate the Huang Rhys Factor."""
        if self.state_high_energy.vib_spacing != 0:
            return self.lambda_inner / self.state_high_energy.vib_spacing
        else:
            raise ValueError(
                "state_high_energy.vib_spacing must not be zero when computing Huang-Rhys."
            )

    def __repr__(self) -> str:
        return (
            f"Transition(Low-energy state ='{self.state_low_energy.name}', High-energy state='{self.state_high_energy.name}', "
            f"Energy Difference={self.mean_gibbs_energy:.4f} eV, Huang Rhys Factor={self.huang_rhys:.4f})"
        )

    @property
    def name(self):
        """Indicates the states involved in the transition."""
        return f"{self.state_high_energy.name}<->{self.state_low_energy.name}"

    @property
    def index(self):
        """Returns the tuple of indices (high, low) for the transition."""
        return (self.state_high_energy.index, self.state_low_energy.index)

    @classmethod
    def assign_indices(cls, transitions: list["Transition"]):
        """
        Gathers all unique states from a list of transitions, sorts them by
        energy, and assigns an .index attribute to each state.
        Ground state (lowest energy) is always 0.
        """
        # 1. Collect all unique state objects
        unique_states = set()
        for t in transitions:
            unique_states.add(t.state_high_energy)
            unique_states.add(t.state_low_energy)

        # Add the default GROUND_STATE if no 0-energy state exists
        if not any(s.energy == 0 for s in unique_states):
            unique_states.add(cls.GROUND_STATE)

        # 2. Sort states by their energy attribute
        # Lower energy gets lower index
        sorted_states = sorted(list(unique_states), key=lambda s: s.energy)

        # 3. Assign the index to the state objects
        for idx, state in enumerate(sorted_states):
            state.index = idx

        return sorted_states
