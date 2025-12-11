# src/MLJ/normalisation.py
#####################################################################################
# MLJ Package
#
# Normalisation Module: Calculate Partition Functions for Normalisation
# Author: Jolanda S Müller, Tim Rein,  Imperial College London
# Copyright (c) 2025, Imperial College London, BSD 3-Clause License
# Date: November 2025
#####################################################################################


from MLJ.physics.transition import TransitionType
from MLJ.physics.transition import Transition
from MLJ.physics.basics import boltzmann, integral
import numpy as np
from typing import Sequence


def partition_function(transition: Transition) -> Sequence[float]:
    """Return the normalisation factor from the integrated partition function for each temperature."""

    temperatures = np.array([300.0, 200.0])
    
    if (transition.transition_type == TransitionType.RECOMBINATION):
        boltzmann_factor = boltzmann(transition.gibbs_energy_grid, temperatures)
    else:
        boltzmann_factor = np.ones(len(temperatures))

    integrand = transition.disorder_weights[:, None] * boltzmann_factor

    partition_function  = integral(y=integrand, x=transition.gibbs_energy_grid)

    return partition_function