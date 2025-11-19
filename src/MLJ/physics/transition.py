# src/MLJ/transition.py
#####################################################################################
# MLJ Package
#
# Classes containing core parameters of quantum states. 
# Author: Jolanda S Müller, Tim Rein, Imperial College London
# Copyright (c) 2025, Imperial College London, BSD 3-Clause License
# Date: November 2025
#####################################################################################

from typing import Callable, Any, Optional
from MLJ.physics.state import State
from enum import Enum

class TransitionType(Enum):
    """Enum class to distinguish the different transition types."""
    ABSORPTION = "absorption"
    RECOMBINATION = "recombination"

class Transition:
    """Represents a transition between two quantum states."""
    def __init__(self,
                state_low_energy: State,   # typically the ground state
                state_high_energy: State,  # typically the excited or CT state
                transition_type: TransitionType = TransitionType.RECOMBINATION,
                oscillator_strength: float = 1,
                dipole_moment: float = 1,
                lambda_inner: float = 0.02,
                lambda_outer: float = 0.02) -> None:
                
        if state_low_energy is None or state_high_energy is None:
            raise ValueError("Two valid states must be given.")
        
        self.transition_type: TransitionType = transition_type
        self.state_low_energy: State = state_low_energy
        self.state_high_energy: State = state_high_energy
        
        # Calculate the energy difference: High and low energy state
        self.energy_difference: float = state_high_energy.energy - state_low_energy.energy
        self.oscillator_strength: float = oscillator_strength
        self.dipole_moment: float = dipole_moment
        
        self.lambda_inner: float = lambda_inner
        self.lambda_outer: float = lambda_outer
    
    @property
    def huang_rhys(self):
        """Calculate the Huang Rhys Factor."""
        if self.state_high_energy.hW != 0:
            return self.lambda_inner / self.state_high_energy.hW
        else:
            raise ValueError("state_high_energy.hW must not be zero when computing Huang-Rhys.")
    

    def __repr__(self) -> str:
        return (f"Transition(Low-energy state ='{self.state_low_energy.name}', High-energy state='{self.state_high_energy.name}', "
                f"Energy Difference={self.energy_difference:.4f} eV, Huang Rhys Factor={self.huang_rhys:.4f})")

