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
from MLJ.physics.normalisation import Zrec, Zabs
import MLJ.physics.FCWD as fcwd
import MLJ.physics.constants as const
import MLJ.physics.coupling as cpl


def absorption_spectral(energies, transition):
    """Calculate spectral rates of absorption of a transition based on the coupling function."""
    # e.g. coupling for absoption is radiative coupling "M = sqrt(f_osc ... )

    fcwd_abs = fcwd.fcwd_abs(energies, transition.set_type_absorption())

    rad_coupling = cpl.coupling_strength_rad(transition)

    normalisation = 1 #1/Zabs()

    prefactor = 4/3/const.REDUCED_PLANCK_CONSTANT_EVS

    k_absorption = ... # integral normalisation * rad_coupling^2  * fcwd_abs  * boltzmann * disorder      # array of length(energies)

#krE(wavei)=krE(wavei)+4/3/hbarEV*params.results.FCWDEm(istate,wavei)*(power(params.Dmu,2))/const.eps0/power(const.c*hbarEV/E,3)*StateEnergyspacing*exp(-(energy-params.DG0)^2/2/params.sigma^2);

    return k_absorption


def k_radiative_spectral(energies, transition):
    """Calculate spectral rates of radiative recombination of a transition based on the coupling function."""
    # e.g. coupling for absoption is "M = sqrt(f_osc ... )

    fcwd_rec = fcwd.fcwd_rec(energies, transition.set_type_recombination())

    rad_coupling = cpl.coupling_strength_rad(transition)

    normalisation = 1/Zrec()

    k_radiative = ... # integral normalisation *  rad_coupling^2  * fcwd_rec  * boltzmann * disorder      # array of length(energies)

    K_radiative = ... # integral(k_r) -> one value representing total rad recomb.

    return k_radiative

def k_radiative_total(energies, transition):
    """Calculate total radiative recombination rate of a transition based on the coupling function."""

    K_radiative_total = ... #integral  k_radiative_spectral(energies, transition, coupling_function)

    return K_radiative_total


def k_non_radiative_total(energies, transition):
    """Calculate total non-radiative recombination rate of a transition based on the coupling function."""
    # e.g. coupling for non radiative transition is e.g.  V = function of radiative coupling M (mullken hush approximation)

    fcwd_rec = fcwd.fcwd_rec(energies, transition.set_type_recombination())

    nonrad_coupling = cpl.coupling_strength_nrad(transition)

    normalisation = 1/Zrec()

    K_nr = ... # integral nonrad_coupling^2  * fcwd_rec * boltzmann * disorder

    return None
