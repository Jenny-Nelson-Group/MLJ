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
import MLJ.physics.constants as const
import numpy as np
from typing import Sequence

def partition_function(transition: Transition) -> Sequence[float]:

    temperature = np.array([300.0,200.0])

    disorder_energies = disorder_function(transition.gibbs_energies, transition.mean_gibbs_energy, transition.state_high_energy.disorder_sigma)
    boltzmann_factor =  boltzmann(transition.gibbs_energies, temperature) if (transition.transition_type == TransitionType.RECOMBINATION) else 1
    partition_function = np.trapezoid(transition.gibbs_energies, disorder_energies * boltzmann_factor)

    return partition_function

def boltzmann(energy, temperature):
    return np.exp(-energy/(temperature*const.BOLTZMANN_CONSTANT_EV))

def disorder_function(energy, mean_energy, sigma):
    return np.exp(-0.5*(energy-mean_energy)^2/sigma^2)
