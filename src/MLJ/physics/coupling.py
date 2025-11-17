# src/MLJ/coupling.py
#####################################################################################
# MLJ Package
#
# Module for calculating radiative and non radiative coupling strengths
# Author: Jolanda S Müller, Tim Rein,  Imperial College London
# Copyright (c) 2025, Imperial College London, BSD 3-Clause License
# Date: November 2025
#####################################################################################

import MLJ.physics.constants as const
import numpy as np

def coupling_strength_rad(transition) -> float:   
    """ Calculates the radiative coupling strength M of the transition."""
    
    f_osc = transition.oscillator_strength
    E_mid = transition.energy_difference
    hW = transition.state_b.hW

    M = np.sqrt((3/2) * ((const.hbar**2) * f_osc)  /  ((E_mid - hW) * const.me))
    
    return M


def coupling_strength_nrad(transition) -> float:   
    """ Calculates the non-radiative coupling strength M of the transition based on Mulliken Hush."""
    
    M = coupling_strength_rad(transition)

    E_mid = transition.energy_difference
    dmu = transition.dipole_moment

    V = (M * E_mid) / np.sqrt(dmu^2 - 4*M^2)

    return V


# other coupling theory can go here... 