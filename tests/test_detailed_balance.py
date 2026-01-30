import numpy as np
from MLJ.physics.population_light import TransitionMatrix, states_light_population
from MLJ.physics.basics import boltzmann


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
    transitions = {(0, 1): k_LECT, (1, 0): k_CTLE}
    tm = TransitionMatrix(rates=rates, transitions=transitions)

    # 3. Define P_dark following Boltzmann distribution
    # P = DoS * exp(-E/kbT) -- assuming equal DoS and weight 1 here
    p0 = boltzmann(energy_LE, temperature)
    p1 = boltzmann(energy_CT, temperature)
    p_dark = [np.array([p0]), np.array([p1])]

    # 4. Solve at G=0
    result = states_light_population(
        dark_population=p_dark, transition_matrix=tm, generation_rate=None
    )

    # 5. Assert: The solver should not shift the populations
    expected = np.array([p0, p1])
    print(expected)
    print(result)
    np.testing.assert_allclose(
        result,
        expected,
        atol=1e-10,
        err_msg="Detailed balance violated: P_dark was not a stationary state.",
    )
