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

def transition_dipole_moment(transition) -> float:
    """ Calculates the radiative coupling strength M of the transition in units of [m]"""

    f_osc = transition.oscillator_strength
    E_mid = transition.mean_gibbs_energy
    hW = transition.state_high_energy.vib_spacing

    transition_dipole_moment = np.sqrt((3/2) * f_osc
                                       * const.REDUCED_PLANCK_CONSTANT_JS  # J * s = kg m2 s-2 * s = kg m2 s-1
                                       * const.REDUCED_PLANCK_CONSTANT_EVS # eV * s
                                       / (E_mid - hW)                      # /eV
                                       / const.ELECTRON_MASS               # /kg
                                       )
    return transition_dipole_moment                                        # m


def mulliken_hush_coupling(transition) -> float:
    """ Calculates the non-radiative coupling strength M of the transition based on Mulliken Hush."""

    rad_coupling = transition.electronic_coupling_radiative

    E_mid = transition.mean_gibbs_energy
    static_dipole_moment = transition.static_dipole_moment

    non_rad_coupling = (rad_coupling * E_mid) / np.sqrt(static_dipole_moment**2 + 4*rad_coupling**2)

    return non_rad_coupling


# other coupling theory can go here...