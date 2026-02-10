import pytest
import numpy as np
from MLJ.physics.population_light import TransitionMatrix, solve_population


def test_transition_matrix_float_rates():
    """Test that scalars are cast into the correct shape."""
    rates = [0.1, 0.2, 0.3]  # Scalar rates
    tm = TransitionMatrix(rates_to_ground=rates)

    # n_states should be 2, n_temperatures should be 1
    assert tm.k_recombination.shape == (3, 1)
    assert tm.full_system_matrix.shape == (1, 3, 3)
    np.testing.assert_array_equal(tm.k_recombination[0], [0.1])


def test_transition_matrix_array_rates():
    """Test that arrays are cast into the correct shape."""
    rates = [
        np.array([0.5, 0.6, 0.7]),
        np.array([0.2, 0.3, 0.4]),
    ]  # Scalar and 1D array
    tm = TransitionMatrix(rates_to_ground=rates)

    # n_states should be 2, n_temperatures should be 3
    assert tm.k_recombination.shape == (2, 3)
    assert tm.full_system_matrix.shape == (3, 2, 2)
    np.testing.assert_array_equal(tm.k_recombination[0], [0.5, 0.6, 0.7])


def test_transition_matrix_inconsistent_k():
    """Test that inconsistent recombination rates throw an error."""
    rates = [np.array([0.5, 0.7]), np.array([0.2, 0.3, 0.4])]  # Scalar and 1D array
    # Assert that the initialization fails with ValueError
    with pytest.raises(ValueError):
        TransitionMatrix(rates_to_ground=rates)


def test_transition_matrix_with_dict_1temp():
    """Test that transitions are correctly placed in the full system matrix."""
    rates = [0.1, 0.1]
    # Transition from state 1 to state 2 at rate 0.5
    transfers = {(1, 2): 0.5}

    tm = TransitionMatrix(rates_to_ground=rates, transfers=transfers)
    expected = np.array([[[0.6, 0.0], [-0.5, 0.1]]])

    np.testing.assert_allclose(tm.full_system_matrix, expected)


def test_transition_matrix_with_dict_conditions():
    """Test that a scalar transition works with array recombination rates."""
    rates = [np.array([0.1, 0.2]), np.array([0.3, 0.4])]
    transfers = {(1, 2): np.array([0.5, 0.6])}

    tm = TransitionMatrix(rates_to_ground=rates, transfers=transfers)
    expected_matrix = np.array(
        [
            [[0.6, 0.0], [-0.5, 0.3]],  # Condition 1
            [[0.8, 0.0], [-0.6, 0.4]],  # Condition 2
        ]
    )

    assert tm.full_system_matrix.shape == (2, 2, 2)
    # Verify the matrix is correct
    np.testing.assert_allclose(tm.full_system_matrix, expected_matrix)


def test_immutability():
    """Ensure the matrix and rates cannot be modified after creation."""
    tm = TransitionMatrix(rates_to_ground=[0.1, 0.2])

    with pytest.raises(AttributeError):
        tm.k_recombination = np.zeros((2, 1))

    # Check that underlying numpy arrays are read-only
    with pytest.raises(ValueError):
        tm.full_system_matrix[0, 0, 0] = 99


def test_solver_accuracy_floats():
    """Test a simple 2-state steady-state solution."""
    # State 0: Gen=1, Rec=0.1. State 1: Gen=0, Rec=0.1.
    # Transition 0 -> 1 at rate 0.5
    rates = np.array([0.1, 0.1])
    transfers = {(1, 2): 0.5}
    tm = TransitionMatrix(rates_to_ground=rates, transfers=transfers)

    # inputs are (n_states, n_temps)
    result = solve_population(
        dark_population=np.array([[0.0], [0.0]]),
        generation_rate=np.array([[1.0], [0.0]]),
        transition_matrix=tm,
    )

    # Hand-calculated steady state:
    # State 0: 1.0 = (0.1 + 0.5) * n0  => n0 = 1.0 / 0.6 = 1.666
    # State 1: 0.5 * n0 = 0.1 * n1     => n1 = 5 * n0 = 8.333
    expected = np.array([[1.0 / 0.6], [5.0 / 0.6]])  # output should be (n_states, n_temps)
    np.testing.assert_allclose(result, expected)


def test_solver_accuracy_arrays():
    """Test a simple 2-state steady-state solution with multiple temperatures."""
    rates = [np.array([0.2, 0.1]), np.array([0.2, 0.1])]
    trans = {(1, 2): np.array([0.3, 0.5])}
    tm = TransitionMatrix(rates_to_ground=rates, transfers=trans)

    result = solve_population(
        dark_population=[np.array([0.1, 0.0]), np.array([0.2, 0.0])],
        generation_rate=[np.array([1.2, 1.0]), np.array([0.1, 0.0])],
        transition_matrix=tm,
    )

    expected = np.array([1.0 / 0.6, 5.0 / 0.6])
    np.testing.assert_allclose(result[:, 1], expected)


def test_no_generation_defaults_to_zero():
    """Solver should work with None generation_rate, defaulting to zeros."""
    # 2 states, 2 conditions
    rates = [np.array([0.5, 0.5]), np.array([0.5, 0.5])]
    tm = TransitionMatrix(rates_to_ground=rates)

    # Simple case: pop=1.0, gen=None (0.0), rec=0.5
    # Steady state: 0 = gen - rec * (n - pop) => n = pop (if gen=0)
    pop = [np.array([1.0, 2.0]), np.array([1.0, 2.0])]

    result = solve_population(dark_population=pop, transition_matrix=tm)

    # With zero generation and no transitions, light population should equal dark population
    expected = np.array([[1.0, 2.0], [1.0, 2.0]])
    np.testing.assert_allclose(result, expected)


def test_steady_state_in_dark():
    """In the dark (generation=0), the steady-state population should return the dark (thermal) population."""
    # 1. Setup a system with 2 states and some transitions
    # Rates and transitions shouldn't matter for the result,
    # as long as the system is stable.
    rates = [0.1, 0.2]
    transfers = {(0, 1): 0, (1, 0): 0}
    tm = TransitionMatrix(rates_to_ground=rates, transfers=transfers)

    # 2. Define an arbitrary dark populations (2 states, 1 temp)
    p_dark = [[1.0], [0.5]]

    # 3. Solve with generation_rate = 0 (or None)
    # The solver uses: A * n = G + k_rec * n_dark
    result = solve_population(dark_population=p_dark, transition_matrix=tm, generation_rate=None)

    # 4. Assert result == p_dark
    expected = np.array(p_dark)
    np.testing.assert_allclose(result, expected, err_msg="Solver failed to recover P_dark at G=0")


def test_mismatched_generation_population():
    """Solver should fail if the dimensions of population and generation mismatch."""
    tm = TransitionMatrix(rates_to_ground=[0.1, 0.2])

    with pytest.raises(ValueError):
        solve_population(dark_population=[0.0, 1.0], generation_rate=[0.0], transition_matrix=tm)


def test_mismatched_matrix_population():
    """Solver should fail on TransitionMatrix and population/rate mismatch."""
    tm = TransitionMatrix(rates_to_ground=[0.1])

    with pytest.raises(ValueError):
        solve_population(
            dark_population=[0.0, 0.2], generation_rate=[1.0, 0.0], transition_matrix=tm
        )


def test_mismatched_conditions():
    """Solver should fail on TransitionMatrix and population/rate mismatch."""
    tm = TransitionMatrix(rates_to_ground=[np.array([0.0, 1.0]), np.array([0.0, 1.0])])

    with pytest.raises(ValueError):
        solve_population(
            dark_population=[np.array([0.0]), np.array([0.0, 1.0])],
            generation_rate=[np.array([0.0, 1.0]), np.array([0.0, 1.0])],
            transition_matrix=tm,
        )
