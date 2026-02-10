import numpy as np
from MLJ.physics.population_light import TransitionMatrix, solve_population
from MLJ.physics.basics import boltzmann
import MLJ as mlj


def test_detailed_balance_preserves_boltzmann():
    """
    If transitions satisfy detailed balance, the solver should
    preserve the Boltzmann-distributed dark population.
    """
    # 1. Physical Constants (Arbitrary units for simplicity)
    energy_LE, energy_CT = 1.5, 1.4  # Energies in eV
    temperature = 300

    # 2. Define Rates satisfying Detailed Balance: k_up / k_down = exp(-dE / kbT)
    k_LECT = 1.0
    k_CTLE = k_LECT / boltzmann(energy_LE - energy_CT, temperature)

    rates = [0.1, 0.1]  # Recombination to ground
    transfers = {(0, 1): k_LECT, (1, 0): k_CTLE}
    tm = TransitionMatrix(rates_to_ground=rates, transfers=transfers)

    # 3. Define P_dark following Boltzmann distribution
    # P = DoS * exp(-E/kbT) -- assuming equal DoS and weight 1 here
    p0 = boltzmann(energy_LE, temperature)
    p1 = boltzmann(energy_CT, temperature)
    p_dark = [np.array([p0]), np.array([p1])]

    # 4. Solve at G=0
    result = solve_population(dark_population=p_dark, transition_matrix=tm, generation_rate=None)

    # 5. Assert: The solver should not shift the populations
    expected = np.array([[p0], [p1]])
    np.testing.assert_allclose(
        result,
        expected,
        atol=1e-10,
        err_msg="Detailed balance violated: P_dark was not a stationary state.",
    )


def test_detailed_balance_dark():
    """
    If transitions satisfy detailed balance, the solver should
    preserve the Boltzmann-distributed dark population.
    """
    n_temps = 10
    mlj.config.temperatures_K = np.linspace(50, 350, n_temps)
    mlj.config.photon_energies = np.linspace(0.8, 2.0, 300)

    """Define the states"""
    gs = mlj.State(name="S0", energy=0.0)
    le = mlj.State(name="LE", energy=1.5, disorder_sigma=0)
    ct = mlj.State(name="CT", energy=1.35, disorder_sigma=0)

    """Define the transitions"""
    trans_LE = mlj.Transition(le, gs, lambda_outer=0.05)
    trans_CT = mlj.Transition(ct, gs, oscillator_strength=3)
    trans_LECT = mlj.Transition(le, ct, k_transfer=np.ones(n_temps))

    two_state_system = mlj.StateSystem([trans_LE, trans_CT, trans_LECT], voltage=0)

    inital_dark = two_state_system.populations_dark
    solved_dark = solve_population(
        two_state_system.transition_matrix, two_state_system.populations_dark, generation_rate=None
    )
    # check that the initial dark distirbution follows boltzmann statics
    # 1) When Pop are thermally generated, P = DOS * exp(-E/kT)
    # for E2 > E1  -->  P2 / P1 = exp(-|deltaE|/kT)
    P2_to_P1 = boltzmann(le.energy - ct.energy, mlj.config.temperatures_K)
    np.testing.assert_allclose(
        inital_dark[1] / inital_dark[0],  # P2/P1  or LE/CT
        P2_to_P1,
        atol=0,
        err_msg="Detailed balance violated: P_dark was not a stationary state.",
    )

    # check that the rates k21 and k12 follow the boltzmann statistic
    # 2) to obey detailed balance:
    # P1 * k10 + P1 * k12 - P2 * k21 = Gen + P1_dark * k10
    # in the dark, P1 != P1_dark and Gen != 0, thus we check:
    # P2 / P1 = k12 / k21
    np.testing.assert_allclose(
        two_state_system.transfers[(1, 2)] / two_state_system.transfers[(2, 1)],
        P2_to_P1,
        atol=0,
        err_msg="Detailed balance violated: P_dark was not a stationary state.",
    )

    # 3) Lastly, if we run the solver on a system without generation,
    # the initial dark population should be recovered
    np.testing.assert_allclose(
        inital_dark,
        solved_dark,
        atol=0,
        err_msg="Detailed balance violated: P_dark was not a stationary state.",
    )
