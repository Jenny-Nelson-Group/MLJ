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
from MLJ.physics.state import State
from MLJ.physics.normalisation import partition_function
from MLJ.physics.config import config
import MLJ.physics.FCWD as fcwd
import MLJ.physics.constants as const
import MLJ.physics.coupling as cpl
from MLJ.physics.basics import integral, boltzmann
import numpy as np


_prefactor_abs_rec = 1/(3*np.pi*const.VACUUM_PERMITTIVITY_EV*const.REDUCED_PLANCK_CONSTANT_EVS**4)
class Rates:
    def __init__(self,
            transition: Transition,
            photon_energies: np.ndarray,
            temperatures: np.ndarray = None,
            photon_density: float = 1,
            ) -> None:

            self.transition = transition
            self.photon_energies = photon_energies
            self.temperatures = config.temperatures_K if temperatures is None else temperatures
            self.photon_density = config.photon_density if photon_density is None else photon_density
            self.partition_function = partition_function

    def calculate_rates(self):
        self.k_radiative_spectral  = k_radiative_spectral(self.photon_energies, self.transition, self.temperatures)
        self.k_radiative_total     = k_radiative_total(self.photon_energies, self.transition, self.temperatures)
        self.k_non_radiative_total = k_non_radiative_total(self.transition, self.temperatures)
        self.k_recombination_total = self.k_radiative_total + self.k_non_radiative_total


def absorption_spectral(photon_energies, transition, temperatures, photon_density):
    """Calculate spectral rates of absorption of a transition based on the coupling function."""
    # e.g. coupling for absoption is radiative coupling "M = sqrt(f_osc ... )

    transition.set_type_absorption()
    fcwd_abs = fcwd.fcwd(photon_energies=photon_energies, transition=transition, temperatures=temperatures)
    rad_coupling = cpl.coupling_strength_rad(transition)  # plan: I can also add this as a tuneable property of transition, like the disorder distribution in states
    normalisation = partition_function(transition=transition, temperatures=temperatures)
    energy_part = (photon_energies/const.SPEED_OF_LIGHT)**3
    integrand = rad_coupling * fcwd_abs * transition.disorder_weights[None, :, None]
    integrated_over_disorder = integral(y=integrand, x=transition.gibbs_energy_grid, axis=1)

    k_absorption =  photon_density * _prefactor_abs_rec * 1/normalisation[None,:] * energy_part[:,None] * integrated_over_disorder
    return k_absorption


def k_radiative_spectral(photon_energies, transition, temperatures):
    """Calculate spectral rates of radiative recombination of a transition based on the coupling function."""
    # e.g. coupling for absoption is "M = sqrt(f_osc ... )
    # [photon_energies, disorder_energy_grid, temperatures]
    transition.set_type_recombination()
    fcwd_rec = fcwd.fcwd(photon_energies=photon_energies, transition=transition, temperatures=temperatures)
    normalisation = partition_function(transition=transition,temperatures=temperatures)
    rad_coupling = cpl.coupling_strength_rad(transition)
    energy_part = (photon_energies/const.SPEED_OF_LIGHT)**3
    boltzmann_factor = boltzmann(transition.gibbs_energy_grid[None, :, None], temperatures[None, None, :])

    integrand = rad_coupling * fcwd_rec * transition.disorder_weights[None, :, None] * boltzmann_factor
    integrated_over_disorder = integral(y=integrand, x=transition.gibbs_energy_grid, axis=1)

    k_radiative = _prefactor_abs_rec * 1/normalisation[None,:] * energy_part[:,None] * integrated_over_disorder

    print("prefactor", _prefactor_abs_rec)
    print("normalisation", 1/normalisation[None,:])
    print("energypart", energy_part[:,None])
    print("disorder integral", integrated_over_disorder)

    return k_radiative

def k_radiative_total(photon_energies, transition, temperatures):
    """Calculate total radiative recombination rate of a transition based on the coupling function."""
    k_radiative_spec = k_radiative_spectral(photon_energies=photon_energies, transition=transition, temperatures=temperatures)
    k_radiative_total = integral(y=k_radiative_spec, x=photon_energies, axis=0)
    return k_radiative_total


def k_non_radiative_total(transition, temperatures):
    """Calculate total non-radiative recombination rate of a transition based on the coupling function."""
    # e.g. coupling for non radiative transition is e.g.  V = function of radiative coupling M (mullken hush approximation)

    transition.set_type_recombination()
    fcwd_rec_0 = fcwd.fcwd(photon_energies=0, transition=transition, temperatures=temperatures)
    nonrad_coupling = cpl.coupling_strength_nrad(transition)
    normalisation = partition_function(transition=transition,temperatures=temperatures)

    prefactor = 2*np.pi/const.REDUCED_PLANCK_CONSTANT_EVS # correct


    integrand = nonrad_coupling**2 * fcwd_rec_0 * transition.disorder_weights[None, :, None]
    integrated_over_disorder = integral(y=integrand, x=transition.gibbs_energy_grid, axis=1)

    k_nonradiative = 1/normalisation[None,:] * prefactor * integrated_over_disorder
    return k_nonradiative.squeeze(axis=0)

def k_recombination_total(photon_energies, transition, temperatures, photon_density):

    recomb_rate_const_non_radiative = k_non_radiative_total(energie=photon_energies,
                                                              transition=transition,
                                                              temperatures=temperatures,
                                                              )

    recomb_rate_const_radiative = k_radiative_total(photon_energies=photon_energies,
                                                    transition=transition,
                                                    temperatures=temperatures,
                                                    photon_density=photon_density,
                                                    )

    return recomb_rate_const_non_radiative + recomb_rate_const_radiative