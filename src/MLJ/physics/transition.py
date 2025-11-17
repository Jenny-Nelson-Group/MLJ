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

def default_V_function(f: float) -> float:
    """Placeholder for the coupling strength V between the states. """
    # Example calculation for example mulliken Hush
    return f

class Transition:
    """Represents a transition between two quantum states."""
    def __init__(self,
                 state_a: State,  # typically the ground state
                 state_b: State,  # typically the excited or CT state
                 oscillator_strength: float = 1,
                 dipole_moment: float = 1,
                 lambda_inner: float = 0.02,
                 lambda_outer: float = 0.02) -> None:
        
        if state_a is None:
            raise ValueError("State A cannot be None.")
        if state_b is None:
            raise ValueError("State B cannot be None.")
        
        self.state_a: State = state_a
        self.state_b: State = state_b
        
        # Calculate the energy difference: Energy B - Energy A
        self.energy_difference: float = state_b.energy - state_a.energy
        self.oscillator_strength: float = oscillator_strength
        self.dipole_moment: float = dipole_moment
        
        self.lambda_inner: float = lambda_inner
        self.lambda_outer: float = lambda_outer
    
    @property
    def huang_rhys(self):
        # Huang Rhys
        if self.state_b.hW != 0:
            self.huang_rhys: float = self.lambda_inner / self.state_b.hW
        else:
            self.huang_rhys: float = 0.0
    

    def __repr__(self) -> str:
        return (f"Transition(State A='{self.state_a.name}', State B='{self.state_b.name}', "
                f"Energy Diff={self.energy_difference:.4f} eV, Huang Rhys Factor={self.huang_rhys:.4f})")

