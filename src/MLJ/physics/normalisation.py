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

    temperatures = np.array([300.0,200.0])
    gibbs_energies = transition.gibbs_energies

    weights = transition.disorder_distribution(
                x = gibbs_energies,
                mean = transition.mean_gibbs_energy,
                sigma = transition.state_high_energy.disorder_sigma
                )
    
    if (transition.transition_type == TransitionType.RECOMBINATION):
        boltzmann_factor = boltzmann(gibbs_energies, temperatures)
    else:
        boltzmann_factor = np.ones(len(temperatures))

    integrand = weights[:, None] * boltzmann_factor

    partition_function  = integral(y=integrand, x=gibbs_energies)

    return partition_function