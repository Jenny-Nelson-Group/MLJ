# src/MLJ/FCDW.py
#####################################################################################
# MLJ Package
#
# Functions calculating the Franck-Condon Parameters. Model contains two scripts:
# mathematical implementation that works standalone, and a wrapper for integration
# classes from the package.
# Author: Jolanda S Müller, Tim Rein, Imperial College London
# Copyright (c) 2025, Imperial College London, BSD 3-Clause License
# Date: November 2025
#####################################################################################

import numpy as np
from scipy.special import factorial
from typing import Sequence
from MLJ.physics.transition import Transition, ProcessType
import MLJ.physics.constants as const
from MLJ.physics.basics import boltzmann, laguerre_2d


# Convenient wrapper function used by other modules in the package.
def fcwd(
    photon_energies: Sequence[float],
    transition: Transition,
    temperatures: np.ndarray,
    process: ProcessType,
) -> Sequence[float]:
    """
    High-level FCWD (Franck-Condon Weighted Density) interface using Transition objects.
    For reference formula see: https://journals.aps.org/prx/pdf/10.1103/PhysRevX.8.031055 eq. 8

    Parameters
    ----------
    photon_energies : np.ndarray
        1D array of photon energies [eV] at which to evaluate spectral rates.
    transition : Transition
       Containing states, reorganisation energies, coupling strengths, and disorder parameters.
    temperatures : np.ndarray
        1D array of temperatures [K]
    process : ProcessType
        The direction of the transition (ProcessType.ABSORPTION or ProcessType.RECOMBINATION).

    Returns
    -------
    fcwd : array, shape(N_omegas, N_disorder_energies, N_temperatures)
        FCWD evaluated at each photon energy (averaged over vibronic states).
    """

    # assign number of vibrational modes in initial and final state
    match process:
        case ProcessType.ABSORPTION:
            N_vib_initial = transition.state_low_energy.number_of_vibronic_modes
            N_vib_final = transition.state_high_energy.number_of_vibronic_modes
        case ProcessType.RECOMBINATION:
            N_vib_initial = transition.state_high_energy.number_of_vibronic_modes
            N_vib_final = transition.state_low_energy.number_of_vibronic_modes

    # change sign depending on process type
    sign = 1 if process is ProcessType.ABSORPTION else -1
    gibbs_energy_grid = sign * transition.gibbs_energy_grid
    photon_energies = sign * photon_energies

    return compute_fcwd(
        photon_energies=photon_energies,
        gibbs_energy_grid=gibbs_energy_grid,
        temperatures=temperatures,
        outer_reorganisation_energy=transition.lambda_outer,
        huang_rhys=transition.huang_rhys,
        vib_spacing=transition.state_high_energy.vib_spacing,
        n_vib_modes_initial=N_vib_initial,
        n_vib_modes_final=N_vib_final,
    )


# Purely mathematical implementation to be used stand-alone.
def compute_fcwd(
    photon_energies: Sequence[float],
    gibbs_energy_grid: Sequence[float],
    temperatures: np.ndarray,
    outer_reorganisation_energy: float,
    huang_rhys: float,
    vib_spacing: float,
    n_vib_modes_initial: int,
    n_vib_modes_final: int,
) -> Sequence[float]:
    """
    Low level implementation of FCWD (Franck-Condon Weighted Density) given numerical parameters.
    For reference formula see: https://journals.aps.org/prx/pdf/10.1103/PhysRevX.8.031055 eq. 8

    Parameters
    ----------
    photon_energies : np.ndarray
        1D array of photon energies [eV] at which to evaluate spectral rates.
        (Positive for Absorption, Negative for Emission)
    gibbs_energy_grid : np.ndarray
        free energy difference (Array of energies if we consider disorder) (eV)
    temperatures : np.ndarray
        1D array of temperatures [K]
    lambda_outer : float
        outer reorganization energy
    huang_rhys : float
        Huang-Rhys factor (contains the inner reorganisation energy)
    vib_spacing : float
        vibrational quantum (hΩ) (eV)
    n_vib_modes_initial : int
        Number of vibrational modes in the initial state.
    n_vib_modes_final
        Number of vibrational modes in the final state.

    Returns
    -------
    fcwd : array, shape(N_omegas, N_disorder_energies, N_temperatures)
        FCWD evaluated at each photon energy (averaged over vibronic states).
    """

    # --- Load constants and transition parameters ---
    boltzmann_eV = const.BOLTZMANN_CONSTANT_EV
    v_initial = np.arange(n_vib_modes_initial + 1)
    v_final = np.arange(n_vib_modes_final + 1)

    # 1. Define the 'Shapes' of your axes using None
    # Dimension Order: [v_i, v_f, E_phot, E_grid, Temp]
    v_initial_mat = v_initial[:, None, None, None, None]  # (Ni, 1, 1, 1, 1)
    v_final_mat = v_final[None, :, None, None, None]  # (1, Nf, 1, 1, 1)
    photon_energies_mat = photon_energies[None, None, :, None, None]  # (1, 1, Ne, 1, 1)
    gibbs_energy_mat = gibbs_energy_grid[None, None, None, :, None]  # (1, 1, 1, Ng, 1)
    temperature_mat = temperatures[None, None, None, None, :]  # (1, 1, 1, 1, Nt)

    vib_diff = v_final_mat - v_initial_mat

    # 2D Laguerre table (v_i × v_f) and broadcast to right shape
    laguerre_base = laguerre_2d(n_vib_modes_initial, n_vib_modes_final, huang_rhys)
    laguerre_mat = laguerre_base[:, :, None, None, None]

    # Huang Rhys Part
    factor1 = (
        np.exp(-huang_rhys)
        * (huang_rhys ** (vib_diff))
        * factorial(v_initial_mat)
        / factorial(v_final_mat)
        * (laguerre_mat**2)
    )

    # Exponential
    factor2 = np.exp(
        -(
            (
                -photon_energies_mat
                + gibbs_energy_mat
                + outer_reorganisation_energy
                + vib_diff * vib_spacing
            )
            ** 2
        )
        / (4 * outer_reorganisation_energy * boltzmann_eV * temperature_mat)
    )

    # Boltzmann population of the initial state
    boltzmann_initial_vib_states = boltzmann(
        energy=v_initial_mat * vib_spacing, temperature=temperature_mat
    )

    normalisation = 1 / np.sqrt(
        4 * np.pi * outer_reorganisation_energy * boltzmann_eV * temperature_mat
    )
    fcwd_n = normalisation * factor1 * factor2 * boltzmann_initial_vib_states

    # Sum over v_i and v_f, as they are first and second dimension generated by np.meshgrid
    fcwd = np.sum(fcwd_n, axis=(0, 1))

    return fcwd
