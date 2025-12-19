# src/MLJ/rates.py
#####################################################################################
# MLJ Package
#
# Module to calculate the spectral and integrated transition rates
# Author: Jolanda S Müller, Tim Rein, Imperial College London
# Copyright (c) 2025, Imperial College London, BSD 3-Clause License
# Date: November 2025
#####################################################################################

from MLJ.physics.transition import Transition, ProcessType
from MLJ.physics.state import State
from MLJ.physics.normalisation import partition_function
from MLJ.physics.config import config
import MLJ.physics.FCWD as fcwd
import MLJ.physics.constants as const
import MLJ.physics.coupling as cpl
from MLJ.physics.basics import integral, boltzmann
from functools import cached_property
import numpy as np

_prefactor_rad = 1/(3*np.pi*const.VACUUM_PERMITTIVITY_EV*const.REDUCED_PLANCK_CONSTANT_EVS**4)
_prefactor_nrad = 2*np.pi/const.REDUCED_PLANCK_CONSTANT_EVS
class Rates:
    """
    Manager for calculating radiative and non-radiative transition rates.
    This class serves as a manager that lazily computes and caches transition.

    Parameters
    ----------
    transition : Transition
        Object containing states, coupling strengths, and disorder parameters.
    photon_energies : np.ndarray
        1D array of photon energies [eV] at which to evaluate spectral rates.
    temperatures : np.ndarray, optional
        1D array of temperatures [K]. Defaults to `config.temperatures_K`.
    photon_density : float, optional
        The incident photon flux or density. Used to scale absorption rates.
        Defaults to `config.photon_density`.
    """
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

    @cached_property
    def rate_absorption_spectral(self):
        """Spectral radiative absorption rate."""
        return self.rate_calculation(process = ProcessType.ABSORPTION, is_non_radiative=False)

    @cached_property
    def rate_radiative_spectral(self):
        """Spectral radiative recombination rate."""
        return self.rate_calculation(process = ProcessType.RECOMBINATION, is_non_radiative=False)

    @cached_property
    def rate_radiative_total(self):
        """Total radiative rate integrated over photon energies."""
        return integral(y=self.rate_radiative_spectral, x=self.photon_energies, axis=0)

    @cached_property
    def rate_non_radiative_total(self):
        """Total non-radiative rate"""
        knrad = self.rate_calculation(process = ProcessType.RECOMBINATION, is_non_radiative=True)
        return knrad.squeeze(axis=0)
        
    @cached_property
    def rate_recombination_total(self):
        """The total recombination rate (Radiative + Non-Radiative)."""
        return self.rate_radiative_total + self.rate_non_radiative_total

    @cached_property
    def boltzmann_factor(self):
        """ Broadcasting the grid and temperatures into a 3D tensor. Shape: [1, N_disorder, N_temperatures] """
        grid = self.transition.gibbs_energy_grid
        return boltzmann(grid[None, :, None], self.temperatures[None, None, :])
    
    @cached_property
    def norm_recombination(self):
        """Partition function result cached for the current temperatures."""
        return partition_function(self.transition, self.temperatures, ProcessType.RECOMBINATION)

    @cached_property
    def norm_absorption(self):
        """Partition function result cached for the current temperatures."""
        return partition_function(self.transition, self.temperatures, ProcessType.ABSORPTION)

    @cached_property
    def energy_prefactor(self):
        """Energy prefactor for radiative transitions."""
        return (self.photon_energies/const.SPEED_OF_LIGHT)**3
    

    def rate_calculation(self, process, is_non_radiative):
        """
        Core physics engine for calculating the rates.
        Radiative (absorption and recombination) and non-radiative rates are 
        handeled together by this function.

        Parameters
        ----------
        process : ProcessType
            The direction of the transition (e.g., ABSORPTION or RECOMBINATION).
        is_non_radiative : bool
            If True, calculates the non-radiatvie rate.

        Returns
        -------
        np.ndarray
            A 2D array of transition rates.
            - Shape: (N_photon_energies, N_temperatures)
            - For non-radiative rates: The energy axis has length 1 (at 0 eV).
            - Units: check... 
        """
        transition = self. transition

        # 1. Setup Process-Specific Parameters
        if is_non_radiative:
            calculation_energies = np.array([0.0])
            energy_term = np.array([1.0])
            prefactor = _prefactor_nrad            
            coupling = transition.coupling_non_radiative
        else:
            calculation_energies = self.photon_energies
            energy_term = self.energy_prefactor
            prefactor = _prefactor_rad
            coupling = transition.coupling_radiative
        
        # 2. Get FCWD
        fcwd_ = fcwd.fcwd(photon_energies=calculation_energies, transition=transition,
                          temperatures=self.temperatures, process=process)

        # 3. Apply Boltzmann to recombination processes and select partition function
        integrand = fcwd_ * transition.disorder_weights[None, :, None]
        match process:
            case ProcessType.RECOMBINATION:
                integrand *= self.boltzmann_factor
                norm = self.norm_recombination
            case ProcessType.ABSORPTION:
                norm = self.norm_absorption
            case _:
                raise ValueError(f"Process {process} not recognized by Rate Engine.")

        # 4. Integrate
        disorder_integral = integral(y = integrand, x = transition.gibbs_energy_grid, axis=1)

        # Get Rate
        rate = prefactor * coupling**2 * energy_term[:,None]  * disorder_integral / norm[None,:]
        if process==ProcessType.ABSORPTION:
            rate *= self.photon_density
        
        return rate 
