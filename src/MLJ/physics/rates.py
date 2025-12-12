# src/MLJ/rates.py
#####################################################################################
# MLJ Package
#
# Module to calculate the spectral and integrated transition rates
# Author: Jolanda S Müller, Tim Rein, Imperial College London
# Copyright (c) 2025, Imperial College London, BSD 3-Clause License
# Date: November 2025
#####################################################################################

from MLJ.physics.transition import Transition
from MLJ.physics.normalisation import partition_function
import MLJ.physics.FCWD as fcwd
import MLJ.physics.constants as const
import MLJ.physics.coupling as cpl
import MLJ.physics.basics as integral
import numpy as np

PREFACTOR_ABS_REC = 1/(3*np.pi*const.VACUUM_PERMITTIVITY_SI*const.REDUCED_PLANCK_CONSTANT_EVS**4)

def absorption_spectral(energies, transition):
    """Calculate spectral rates of absorption of a transition based on the coupling function."""
    # e.g. coupling for absoption is radiative coupling "M = sqrt(f_osc ... )

    photon_density = 1

    transition.set_type_absorption()

    fcwd_abs = fcwd.fcwd_abs(energies, transition.set_type_absorption())

    rad_coupling = cpl.coupling_strength_rad(transition)  # plan: I can also add this as a tuneable property of transition, like the disorder distribution in states

    normalisation = partition_function(transition)

    prefactor_2 = (energies/const.SPEED_OF_LIGHT)**3

    integrand = rad_coupling * fcwd_abs * transition.disorder_weights

    k_absorption = 1/normalisation * PREFACTOR_ABS_REC * prefactor_2 * integral(y=integrand, x=transition.gibbs_energy_grid) # array of length(energies)

    return k_absorption


def k_radiative_spectral(energies, transition):
    """Calculate spectral rates of radiative recombination of a transition based on the coupling function."""
    # e.g. coupling for absoption is "M = sqrt(f_osc ... )
    transition.set_type_recombination()

    fcwd_rec = fcwd.fcwd_rec(energies, transition)

    rad_coupling = cpl.coupling_strength_rad(transition)

    normalisation = partition_function(transition)

    prefactor_2 = (energies/const.SPEED_OF_LIGHT)**3

    integrand = rad_coupling * fcwd_rec * transition.disorder_weights

    k_radiative = 1/normalisation * PREFACTOR_ABS_REC * prefactor_2 * integral(y=integrand, x=transition.gibbs_energy_grid) # array of length(energies)

    return k_radiative

def k_radiative_total(energies, transition):
    """Calculate total radiative recombination rate of a transition based on the coupling function."""

    k_radiative_spec = k_radiative_spectral(energies, transition)
    k_radiative_total = integral(y=k_radiative_spec, x=energies, axis=0)

    return k_radiative_total


def k_non_radiative_total(energies, transition):
    """Calculate total non-radiative recombination rate of a transition based on the coupling function."""
    # e.g. coupling for non radiative transition is e.g.  V = function of radiative coupling M (mullken hush approximation)

    transition.set_type_recombination()

    fcwd_rec_0 = fcwd.fcwd_rec(energies=0, transition=transition)

    nonrad_coupling = cpl.coupling_strength_nrad(transition)

    normalisation = partition_function(transition)

    prefactor = 2*np.pi/const.REDUCED_PLANCK_CONSTANT_EVS

    integrand = nonrad_coupling * fcwd_rec_0 * transition.disorder_weights

    k_nonradiative = 1/normalisation * prefactor * integral(y=integrand, x=transition.gibbs_energy_grid, axis=0)

    return k_nonradiative

def k_recombination_total(energies, transition):
    return k_non_radiative_total(energies, transition) + k_radiative_total(energies, transition)
