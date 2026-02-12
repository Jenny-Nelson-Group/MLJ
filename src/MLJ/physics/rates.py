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
from MLJ.physics.normalisation import partition_function
from MLJ.physics.config import config
import MLJ.physics.FCWD as fcwd
import MLJ.physics.constants as const
from MLJ.physics.basics import integral, boltzmann
from MLJ.helpers.caching import read_only_cached_property, ReactiveModule

import numpy as np

_prefactor_rad = 1 / (
    3 * np.pi * const.VACUUM_PERMITTIVITY_EV * const.REDUCED_PLANCK_CONSTANT_EVS**4
)
_prefactor_nrad = 2 * np.pi / const.REDUCED_PLANCK_CONSTANT_EVS


class Rates(ReactiveModule):
    """
    Manager for calculating radiative and non-radiative transition rates.
    This class serves as a manager that lazily computes and caches transition.

    Parameters
    ----------
    transition : Transition
        Containing states, reorganisation energies, coupling strengths, and disorder parameters.
    photon_energies : np.ndarray
        1D array of photon energies [eV] at which to evaluate spectral rates.
    temperatures : np.ndarray, optional
        1D array of temperatures [K]. Defaults to `config.temperatures_K`.
    """

    def __init__(
        self,
        transition: Transition,
        photon_energies: np.ndarray = None,
        temperatures: np.ndarray = None,
    ) -> None:
        self.transition = transition
        self.temperatures = config.temperatures_K if temperatures is None else temperatures
        self.photon_energies = (
            config.photon_energies if photon_energies is None else photon_energies
        )

        self.start_caching()

    # ----------------------------------------- Property Caching --------------------------------------------#
    @read_only_cached_property
    def rate_absorption_spectral(self):
        """Spectral radiative absorption rate."""
        return self.rate_calculation(process=ProcessType.ABSORPTION, is_non_radiative=False)

    @read_only_cached_property
    def rate_radiative_spectral(self):
        """Spectral radiative recombination rate."""
        return self.rate_calculation(process=ProcessType.RECOMBINATION, is_non_radiative=False)

    @read_only_cached_property
    def rate_radiative_total(self):
        """Total radiative rate integrated over photon energies."""
        return integral(y=self.rate_radiative_spectral, x=self.photon_energies, axis=0)

    @read_only_cached_property
    def rate_non_radiative_total(self):
        """Total non-radiative rate"""
        knrad = self.rate_calculation(process=ProcessType.RECOMBINATION, is_non_radiative=True)
        return knrad.squeeze(axis=0)

    @read_only_cached_property
    def rate_recombination_total(self):
        """The total recombination rate (Radiative + Non-Radiative)."""
        return self.rate_radiative_total + self.rate_non_radiative_total

    @read_only_cached_property
    def boltzmann_electronic_states(self):
        """Broadcasting the grid and temperatures into a 3D tensor (shape: (1, N_disorder_energies, N_temperatures))"""
        grid = self.transition.gibbs_energy_grid
        return boltzmann(grid[None, :, None], self.temperatures[None, None, :])

    @read_only_cached_property
    def norm_recombination(self):
        """Partition function (shape: (N_disorder_energies, N_temperatures)) result cached for the current temperatures."""
        return partition_function(self.transition, self.temperatures, ProcessType.RECOMBINATION)

    @read_only_cached_property
    def norm_absorption(self):
        """Partition function (shape: (N_disorder_energies, N_temperatures)) result cached for the current temperatures."""
        return partition_function(self.transition, self.temperatures, ProcessType.ABSORPTION)

    @read_only_cached_property
    def photon_phase_space(self):
        """Energy prefactor for radiative transitions with (shape(photon_energies))"""
        return (self.photon_energies / const.SPEED_OF_LIGHT) ** 3

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
        transition = self.transition

        # 1. Setup Process-Specific Parameters
        if is_non_radiative:
            calculation_energies = np.array([0.0])
            energy_term = np.array([1.0])
            prefactor = _prefactor_nrad
            coupling = transition.electronic_coupling_non_radiative
        else:
            calculation_energies = self.photon_energies
            energy_term = self.photon_phase_space
            prefactor = _prefactor_rad
            coupling = transition.electronic_coupling_radiative

        # 2. Get FCWD
        fcwd_ = fcwd.fcwd(
            photon_energies=calculation_energies,
            transition=transition,
            temperatures=self.temperatures,
            process=process,
        )

        # 3. Determine the integrand
        integrand = fcwd_ * transition.disorder_weights[None, :, None]

        # apply Boltzmann to recombination processes and select partition function
        match process:
            case ProcessType.RECOMBINATION:
                integrand *= self.boltzmann_electronic_states
                norm = self.norm_recombination
            case ProcessType.ABSORPTION:
                norm = self.norm_absorption
            case _:
                raise ValueError(f"Process {process} not recognized by Rate Engine.")

        # 4. Integrate
        disorder_integral = integral(y=integrand, x=transition.gibbs_energy_grid, axis=1)

        # Get Rate
        rate = (
            prefactor  # shape(1)
            * coupling**2  # shape(1)
            * energy_term[:, None]  # shape(photon_energies,1)
            * disorder_integral  # shape(photon_energies,temperatures)
            / norm[None, :]
        )  # shape(1,temperatures)

        return rate
