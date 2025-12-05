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
from MLJ.physics.basics import boltzmann
import numpy as np
from typing import Sequence


def partition_function(transition: Transition) -> Sequence[float]:
    """Return the normalisation factor from the integrated partition function for each temperature."""

    temperatures = np.array([300.0,200.0])

    disorder_energies = transition.disorder_distribution(
                x = transition.gibbs_energies,
                mean = transition.mean_gibbs_energy,
                sigma = transition.state_high_energy.disorder_sigma
                )
    if (transition.transition_type == TransitionType.RECOMBINATION):
        boltzmann_factor = boltzmann(transition.gibbs_energies, temperatures)
    else:
        boltzmann_factor = 1

    integrand = disorder_energies[:, None] * boltzmann_factor
    partition_function = np.trapezoid(integrand, x=transition.gibbs_energies, axis=0)

    return partition_function