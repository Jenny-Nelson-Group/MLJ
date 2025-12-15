# src/MLJ/normalisation.py
#####################################################################################
# MLJ Package
#
# Normalisation Module: Calculate Partition Functions for Normalisation
# Author: Jolanda S Müller, Tim Rein,  Imperial College London
# Copyright (c) 2025, Imperial College London, BSD 3-Clause License
# Date: November 2025
#####################################################################################

from MLJ.physics.transition import TransitionType, Transition
from MLJ.physics.basics import boltzmann, integral
from MLJ.physics.config import config
import numpy as np
from typing import Sequence

def partition_function(transition: Transition,
                       temperatures: np.ndarray,
                       ) -> Sequence[float]:
    """Return the normalisation factor from the integrated partition function for each temperature."""

    if (transition.transition_type == TransitionType.RECOMBINATION):
        boltzmann_factor = boltzmann(transition.gibbs_energy_grid, temperatures)
    else:
        boltzmann_factor = np.ones(len(temperatures))

    integrand = transition.disorder_weights[:, None] * boltzmann_factor

    partition_function  = integral(y=integrand, x=transition.gibbs_energy_grid, axis=0)

    return partition_function