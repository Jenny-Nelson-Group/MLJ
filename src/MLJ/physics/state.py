# src/MLJ/state.py
#####################################################################################
# MLJ Package
#
# Classes containing core parameters of quantum states.
# Author: Jolanda S Müller, Tim Rein, Imperial College London
# Copyright (c) 2025, Imperial College London, BSD 3-Clause License
# Date: November 2025
#####################################################################################

class State:
    """Represents a quantum state with its properties."""
    def __init__(
            self,
            index: int = 0,
            name: str = "Ground State",
            number_of_vibronic_modes: int = 15,
            vib_spacing: float = 0.1500,
            disorder_sigma: float = 0.05,
            disorder_number_of_states: float = 21,
            disorder_integration_cut_off: float = 2.5,
            energy: float = 0.0
        ) -> None:

        self.index: int = index
        self.name: str = name
        self.number_of_vibronic_modes: int = number_of_vibronic_modes
        self.vib_spacing: float = vib_spacing
        self.disorder_sigma: float = disorder_sigma
        self.disorder_number_of_states: float = disorder_number_of_states
        self.disorder_integration_cut_off: float = disorder_integration_cut_off
        self.energy: float = energy

    def __repr__(self) -> str:
        return f"State(Name='{self.name}', Energy={self.energy:.4f} eV)"
