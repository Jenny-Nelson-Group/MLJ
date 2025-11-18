# src/MLJ/FCDW.py
#####################################################################################
# MLJ Package
#
# Functions calculating the Franck-Condon Parameters
# Author: Jolanda S Müller, Tim Rein, Imperial College London
# Copyright (c) 2025, Imperial College London, BSD 3-Clause License
# Date: November 2025
#####################################################################################

import numpy as np
from scipy.special import factorial, genlaguerre

import src.MLJ.physics.constants as const


def fcwd(photon_energies, transition, transition_type ='abs'):
    """
    Compute the FCWD (Franck–Condon Weighted Density) using MLJ theory.
    For reference formula see: https://journals.aps.org/prx/pdf/10.1103/PhysRevX.8.031055 eq. 8

    Parameters
    ----------
    photon_energies : array-like
        Photon energies ω in eV.
    transition : Transition
        Transition object containing λ_o, Huang–Rhys factor S, level spacing hW, 
        free energy difference E, and vibronic state dimensions.
    transition_type : str ('abs' or 'rec')
        'abs' => absorption
        'rec' => emission / recombination

    Returns
    -------
    fcwd : array, shape(N_omegas,N_disorder_energies,N_temperatures)
        FCWD evaluated at each photon energy (averaged over vibronic states).
    """

    # --- Load constants and transition parameters ---
    T = np.array([300.0,200.0])
    kB = const.kB_eV                     # Boltzmann constant in eV/K
    l_i = transition.lambda_inner      # inner reorganization energy
    l_o = transition.lambda_outer        # outer reorganization energy
    E = transition.energy_difference     # free energy difference (Array of energies if we consider disorder) (eV)
    # S = transition.huang_rhys            # Huang–Rhys factor (contains the inner reorganisation energy)
    hW = transition.state_b.hW           # vibrational quantum (hΩ) (eV)
    S = l_i/hW

    N_vib_init = transition.state_a.number_of_vibronic_modes
    N_vib_final = transition.state_b.number_of_vibronic_modes

    v_i = np.arange(N_vib_init + 1)
    v_f = np.arange(N_vib_final + 1)

    # Build 5D meshgrid to enable vectorization of the calculations
    v_i_mat, v_f_mat, photon_energies_mat, E_mat, T_mat = np.meshgrid(v_i, v_f, photon_energies, E, T, indexing='ij')

    dv = v_f_mat - v_i_mat
    
    # 2D Laguerre table (v_i × v_f)
    laguerre_base = np.zeros((N_vib_init+1, N_vib_final+1))

    for i in range(N_vib_init + 1):
        for j in range(N_vib_final + 1):
            k = j - i
            if k >= 0:
                laguerre_base[i, j] = genlaguerre(i, k)(S)
            else:
                laguerre_base[i, j] = 0.0

    # Now broadcast to the 5D meshgrid shape
    laguerre_mat = laguerre_base[:, :, None, None, None]

    # Huang Rhys Part
    factor1 = (
        np.exp(-S)
        * (S ** (dv))
        * factorial(v_i_mat) / factorial(v_f_mat)
        * (laguerre_mat ** 2)
    )

    # Exponential
    if transition_type == 'rec':
        factor2 = np.exp(-(photon_energies_mat - E_mat + l_o + dv * hW ) ** 2 / (4 * l_o * kB * T_mat))
    elif transition_type == 'abs':
        factor2 = np.exp(-(-photon_energies_mat + E_mat + l_o + dv * hW ) ** 2 / (4 * l_o * kB * T_mat))
    else:
        raise TypeError("Invalid transition type.")

    # Boltzmann population of the initial state
    factor3 = np.exp(-(v_i_mat * hW) / (kB * T_mat))

    normalisation = 1 / np.sqrt(4 * np.pi * l_o * kB * T_mat)
    fcwd_n = normalisation * factor1 * factor2 * factor3

    # Sum over v_i and v_f
    fcwd = np.sum(fcwd_n, axis=(0, 1))

    return fcwd
