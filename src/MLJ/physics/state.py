# src/MLJ/state.py
#####################################################################################
# MLJ Package
#
# Classes containing core parameters of quantum states.
# Author: Jolanda S Müller, Tim Rein, Imperial College London
# Copyright (c) 2025, Imperial College London, BSD 3-Clause License
# Date: November 2025
#####################################################################################

from MLJ.physics.basics import dirac_delta, gaussian, gaussian_norm
from typing import Callable
import numpy as np

DistributionFunction = Callable[[np.ndarray, float, float], np.ndarray]

class State:
    """
    Quantum state with vibronic structure and energetic disorder.

    Parameters
    ----------
    index : int, optional
        Integer identifier for the state.
    name : str, optional
        Human-readable label for the state.
    energy : float, optional
        Electronic energy of the state in eV.
    number_of_vibronic_modes : int, optional
        Number of vibronic levels included in the model.
    vib_spacing : float, optional
        Spacing between adjacent vibronic levels (eV).
    disorder_sigma : float, optional
        Standard deviation of the energetic disorder distribution (eV).
        If set to 0, disorder is disabled and a discrete delta single energy is assumed.
    disorder_number_of_states : int, optional
        Number of discrete disorder samples used for numerical integration.
        If set to 1, disorder is treated as absent.
    disorder_integration_cut_off : float, optional
        Symmetric cut-off in units of `sigma` for evaluating the disorder distribution.
        The integration bounds become: mean ± cut_off * sigma.
    disorder_distribution : DistributionFunction or None, optional
        Callable with signature ``f(x, mean, sigma)`` returning probability weights.
        If ``None`` and disorder is active, a Gaussian is used by default.
        When disorder is disabled (sigma == 0 or number_of_states == 1),
        this parameter is ignored and a discrete delta single energy is assumed.
    """
    def __init__(
            self,
            index: int = 0,
            name: str = "Ground State",
            energy: float = 0.0,
            number_of_vibronic_modes: int = 15,
            vib_spacing: float = 0.1500,
            disorder_sigma: float = 0.05,
            disorder_number_of_states: int = 21,
            disorder_integration_cut_off: float = 2.5,
            disorder_distribution: 'DistributionFunction | None' = None,
        ) -> None:

        self.index: int = index
        self.name: str = name
        self.energy: float = energy
        self.number_of_vibronic_modes: int = number_of_vibronic_modes
        self.vib_spacing: float = vib_spacing
        self.disorder_integration_cut_off = disorder_integration_cut_off

        if (disorder_number_of_states == 1 or disorder_sigma == 0):
            self.disorder_number_of_states= 1
            self.disorder_sigma = 0
            self.disorder_distribution = single_value
        else:
            self.disorder_sigma = disorder_sigma
            self.disorder_number_of_states = disorder_number_of_states
            self.disorder_distribution = (
                gaussian_distribution if disorder_distribution is None else disorder_distribution
            )

        self.energy_grid: np.ndarray = self.set_disorder_grid()
        self.disorder_weights = self.set_disorder_weights()

    def set_disorder_grid(self) -> np.ndarray:
        cut_off = self.disorder_integration_cut_off * self.disorder_sigma
        energy_grid = np.linspace(self.energy - cut_off, self.energy + cut_off, self.disorder_number_of_states)
        return energy_grid
    
    def set_disorder_weights(self) -> np.ndarray:
        energy_grid = self.energy_grid
        weights = self.disorder_distribution(state=self, x=energy_grid)
        return weights

    def __repr__(self) -> str:
        return f"State(Name='{self.name}', Energy={self.energy:.4f} eV)"

DistributionFunction = Callable[[State, np.ndarray], np.ndarray]

def single_value(state: State, x: np.ndarray) -> np.ndarray:
    return np.ones_like(x, dtype=float)

def gaussian_distribution(state: State, x: np.ndarray) -> np.ndarray:
    """
    Default energetic disorder distribution: Gaussian around `state.energy`
    with width `state.disorder_sigma`.
    """
    return gaussian_norm(x, state.energy, state.disorder_sigma)

def gaussian_distribution_nonnorm(state: State, x: np.ndarray) -> np.ndarray:
    """
    Default energetic disorder distribution: Gaussian around `state.energy`
    with width `state.disorder_sigma`.
    """
    return gaussian(x, state.energy, state.disorder_sigma)