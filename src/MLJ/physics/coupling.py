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
    E_mid = transition.mean_gibbs_energy
    hW = transition.state_high_energy.vib_spacing

    transition_dipole_moment = np.sqrt((3/2) 
                                       * (const.REDUCED_PLANCK_CONSTANT_EVS**2) * f_osc
                                       /  ((E_mid - hW) * const.ELECTRON_MASS)
                                       )
    return transition_dipole_moment


def coupling_strength_nrad(transition) -> float:
    """ Calculates the non-radiative coupling strength M of the transition based on Mulliken Hush."""

    rad_coupling = coupling_strength_rad(transition)

    E_mid = transition.mean_gibbs_energy
    dmu = transition.dipole_moment

    non_rad_coupling = (rad_coupling * E_mid) / np.sqrt(dmu**2 + 4*rad_coupling**2)
    return non_rad_coupling


# other coupling theory can go here...