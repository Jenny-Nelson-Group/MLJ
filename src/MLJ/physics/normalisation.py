# src/MLJ/normalisation.py
#####################################################################################
# MLJ Package
#
# Normalisation Module: Calculate Partition Functions for Normalisation
# Author: Jolanda S Müller, Tim Rein,  Imperial College London
# Copyright (c) 2025, Imperial College London, BSD 3-Clause License
# Date: November 2025
#####################################################################################

from MLJ.physics.transition import Transition, ProcessType
from MLJ.physics.basics import boltzmann, integral
from MLJ.physics.config import config
import numpy as np
from typing import Sequence

def partition_function(transition: Transition,
                       temperatures: np.ndarray,
                       process: ProcessType,
                       ) -> Sequence[float]:
    """Return the normalisation factor from the integrated partition function for each temperature.
    For reference formula see: https://www.nature.com/articles/s41467-021-23975-3 eq. 8 and 9

    Parameters
    ----------
    transition : Transition
        Containing states, reorganisation energies, coupling strengths, and disorder parameters.
    temperatures : np.ndarray
        1D array of temperatures [K]
    process : ProcessType
        The direction of the transition (ProcessType.ABSORPTION or ProcessType.RECOMBINATION).

    Returns
    -------
    partition_function : array, shape(N_temperatures)
        FCWD evaluated at each photon energy (averaged over vibronic states).
    """

    if (process == ProcessType.RECOMBINATION):
        boltzmann_electronic_states = boltzmann(
        transition.gibbs_energy_grid[:, None],
        temperatures[None, :]
    )
    else:
        boltzmann_electronic_states = np.ones((1, len(temperatures)))

    integrand = transition.disorder_weights[:, None] * boltzmann_electronic_states

    partition_function  = integral(y=integrand, x=transition.gibbs_energy_grid, axis=0)

    return partition_function