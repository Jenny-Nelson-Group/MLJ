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
from enum import Enum
import numpy as np

class TransitionType(Enum):
    """Enum class to distinguish the different transition types."""
    ABSORPTION = "absorption"
    RECOMBINATION = "recombination"

class Transition:
    """Represents a transition between two quantum states."""
    def __init__(self,
                state_high_energy: State,  # typically the excited or CT state
                state_low_energy: State = State(),   # default to Ground State with Energy 0
                transition_type: TransitionType = TransitionType.RECOMBINATION,
                oscillator_strength: float = 1,
                dipole_moment: float = 1,
                lambda_inner: float = 0.02,
                lambda_outer: float = 0.02) -> None:

        if state_low_energy is None or state_high_energy is None:
            raise ValueError("Two valid states must be given.")

        self.transition_type: TransitionType = transition_type

        self.determine_high_low_energy_state(state_low_energy, state_high_energy)
        self.set_gibbs_energy_grid()

        self.oscillator_strength: float = oscillator_strength
        self.dipole_moment: float = dipole_moment

        self.lambda_inner: float = lambda_inner
        self.lambda_outer: float = lambda_outer

    def determine_high_low_energy_state(self, state_low_energy, state_high_energy):
        """Assign which state is the higher energy state and which state is the lower energy state."""
        if state_high_energy.energy > state_low_energy.energy:
            self.state_low_energy: State = state_low_energy
            self.state_high_energy: State = state_high_energy
        else:
            self.state_low_energy: State = state_high_energy
            self.state_high_energy: State = state_low_energy

    def set_gibbs_energy_grid(self):
        """Calculate the spread of energies based on the disorder."""
        self.mean_gibbs_energy: float = self.state_high_energy.energy - self.state_low_energy.energy
        self.gibbs_energy_grid = self.state_high_energy.energy_grid - self.state_low_energy.energy
    
    @property
    def disorder_weights(self):
        return self.state_high_energy.disorder_weights

    def set_type_absorption(self):
        """Set the transition type to Absorption."""
        self.transition_type = TransitionType.ABSORPTION
        return self

    def set_type_recombination(self):
        """Set the transition type to Recombination."""
        self.transition_type = TransitionType.RECOMBINATION
        return self

    @property
    def disorder_distribution(self):
        """Return the disorder distribution of the high energy state."""
        return self.state_high_energy.disorder_distribution

    @property
    def huang_rhys(self):
        """Calculate the Huang Rhys Factor."""
        if self.state_high_energy.vib_spacing != 0:
            return self.lambda_inner / self.state_high_energy.vib_spacing
        else:
            raise ValueError("state_high_energy.vib_spacing must not be zero when computing Huang-Rhys.")

    def __repr__(self) -> str:
        return (f"Transition(Low-energy state ='{self.state_low_energy.name}', High-energy state='{self.state_high_energy.name}', "
                f"Energy Difference={self.mean_gibbs_energy:.4f} eV, Huang Rhys Factor={self.huang_rhys:.4f}), Type={self.transition_type}")
