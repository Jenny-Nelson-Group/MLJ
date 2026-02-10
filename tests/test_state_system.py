import pytest
import numpy as np
from MLJ.physics.basics import boltzmann
from MLJ.physics.state import State
from MLJ.physics.transition import Transition
from MLJ.physics.simulate_system import StateSystem


@pytest.fixture
def basic_setup():
    """Sets up a simple LE and CT system."""
    s_le = State(energy=2.0, name="LE")
    s_ct = State(energy=1.5, name="CT")

    # Transitions to ground
    t_le_g = Transition(state_high_energy=s_le)  # state_low_energy defaults to GROUND
    t_ct_g = Transition(state_high_energy=s_ct)

    # Inter-state transfer
    t_le_ct = Transition(state_high_energy=s_le, state_low_energy=s_ct)

    return [t_le_g, t_ct_g, t_le_ct], s_le, s_ct


def test_initialization_and_indexing(basic_setup):
    transitions, s_le, s_ct = basic_setup
    system = StateSystem(transitions=transitions)

    # Check that ground was popped and we have 2 excited states
    assert len(system.sorted_states) == 2
    # Energies should be sorted: CT (1.5) then LE (2.0)
    assert system.sorted_states[0].energy == 1.5
    assert system.sorted_states[1].energy == 2.0

    # Verify mapping: rates[0] should belong to CT (index 1), rates[1] to LE (index 2)
    assert system.rates[0].transition.state_high_energy.name == "CT"
    assert system.rates[1].transition.state_high_energy.name == "LE"


def test_transfer_dict_keys(basic_setup):
    transitions, s_le, s_ct = basic_setup
    system = StateSystem(transitions=transitions)

    # CT is index 1, LE is index 2. Ground is 0.
    # The transfer dict should contain (1,2) and (2,1) but NOT (1,0)
    keys = system.transfers.keys()
    assert (1, 2) in keys
    assert (2, 1) in keys
    assert (1, 0) not in keys  # (1,0) goes into the 'rates' array/matrix diagonal


def test_population_conservation(basic_setup):
    transitions, _, _ = basic_setup
    system = StateSystem(transitions=transitions)

    # Get populations under light (shape: n_cond, n_states_states)
    pop_light = system.populations_light

    # Sum of excited state populations must be <= 1.0
    # (The remainder is ground state population)
    total_excited_pop = np.sum(pop_light, axis=1)
    assert np.all(total_excited_pop <= 1.0)
    assert np.all(total_excited_pop >= 0.0)


def test_reactivity_cache_invalidation(basic_setup):
    transitions, _, _ = basic_setup
    system = StateSystem(transitions=transitions)

    # Trigger first calculation
    first_matrix = system.transition_matrix

    # Change temperature - should invalidate cache
    new_temps = np.array([300.0, 310.0])
    system.temperatures = new_temps

    # Accessing matrix should now trigger a recalculation with new shape
    second_matrix = system.transition_matrix
    assert second_matrix.full_system_matrix.shape[0] == 2
    assert second_matrix is not first_matrix


def test_empty_transfer_logic(basic_setup):
    # Setup where k_transfer is None
    _, s_le, s_ct = basic_setup
    t_le_ct = Transition(state_high_energy=s_le, state_low_energy=s_ct)
    t_le_ct.k_transfer = None  # Explicitly None

    system = StateSystem(transitions=[t_le_ct, Transition(s_le), Transition(s_ct)])

    # Should not crash and should return zeros
    k_down = system.transfers[(s_ct.index, s_le.index)]
    assert np.all(k_down == 0)


def test_transfer_with_rate_and_boltzmann(basic_setup):
    """
    Verifies that when k_transfer is provided, the uphill rate
    correctly obeys detailed balance.
    """
    s_le = State(energy=2.0, name="LE")
    s_ct = State(energy=1.5, name="CT")

    # Transitions to ground
    test_rate = [10e9]
    t_le_g = Transition(s_le)
    t_ct_g = Transition(s_ct)
    t_le_ct = Transition(s_le, s_ct, k_transfer=test_rate)

    transitions = [t_le_g, t_ct_g, t_le_ct]

    # Initialize system at a specific temperature
    system = StateSystem(transitions=transitions, temperatures=np.array([300.0]))

    # Get the indices assigned by the system
    idx_ct = s_ct.index
    idx_le = s_le.index

    # Retrieve from the transfer_dict
    k_down = system.transfers[(idx_le, idx_ct)]  # high -> low
    k_up = system.transfers[(idx_ct, idx_le)]  # low -> high

    # 1. Check downhill rate matches input
    assert k_down == test_rate

    # 2. Check uphill rate matches detailed balance: k_up = k_down * exp(-dE / kT)
    dE = s_le.energy - s_ct.energy
    expected_k_up = test_rate * boltzmann(dE, np.array([300.0]))

    # Use approx for floating point comparison
    assert k_up == pytest.approx(expected_k_up)
    assert k_up < k_down  # Uphill must be slower than downhill


def test_scalar_k_transfer_handling():
    """Tests that passing a float k_transfer doesn't crash the matrix validation."""
    s_le = State(energy=2.0)
    s_ct = State(energy=1.5)

    # User passes a float, but temperatures is an array
    t_le_ct = Transition(s_le, s_ct, k_transfer=1e6)
    system = StateSystem(transitions=[t_le_ct, Transition(s_le), Transition(s_ct)])

    # This access triggers TransitionMatrix.__post_init__ validation
    try:
        _ = system.transition_matrix
    except TypeError as e:
        pytest.fail(f"TransitionMatrix failed with scalar k_transfer: {e}")


@pytest.mark.parametrize("n_temps", [1, 3])
@pytest.mark.parametrize("n_energies", [1, 50])
def test_system_property_shapes(n_temps, n_energies):
    """
    Verification test to ensure all cached properties return
    consistent shapes relative to inputs.
    """
    # 1. Setup states and transitions
    s1 = State(energy=1.8, name="S1")
    s2 = State(energy=1.6, name="S2")

    k_transfer = np.ones(n_temps)
    # Simple 3-state system (Ground, S1, S2)
    transitions = [Transition(s1), Transition(s2), Transition(s1, s2, k_transfer=k_transfer)]

    energies = np.linspace(1.0, 3.0, n_energies)
    temps = np.linspace(100, 300, n_temps)

    system = StateSystem(transitions=transitions, photon_energies=energies, temperatures=temps)

    n_states = 2  # S1 and S2

    # 2. Check each cached property for correct dimensions

    # Rates: Should be a 1D array of length n_states containing Rates objects
    assert system.rates.shape == (n_states,)

    # Transfers: Should be a dict (logic-based, but must have content)
    assert len(system.transfers) > 0

    # Transition Matrix: n_cond x n_states x n_states
    mat = system.transition_matrix.full_system_matrix
    assert mat.shape == (n_temps, n_states, n_states)
    assert system.transition_matrix.shape == (n_states, n_temps)

    # Generation: n_states x n_energies x n_temps
    assert system.generation.shape == (n_states, n_temps)

    # Populations: n_states x n_temps
    assert system.populations_light.shape == (n_states, n_temps)
    assert system.populations_dark.shape == (n_states, n_temps)

    # Emission PL: n_energies x n_temps (summed or spectral)
    # Adjust based on your emission() function's return shape
    assert system.emission_photoluminescence.shape == (n_energies, n_temps)

    # Absorption: n_energies x n_temps
    assert system.absorbance.shape == (n_energies, n_temps)


def test_strict_shape_validation():
    """
    Verifies that StateSystem raises ValueError when k_transfer
    does not exactly match the temperature array shape.
    """
    s_le = State(energy=2.0)
    s_ct = State(energy=1.5)

    # 3 temperatures defined
    temps = np.array([100, 200, 300])

    # Case A: k_transfer is a scalar (will have shape (1,) via atleast_1d)
    # This should FAIL because (1,) != (3,)
    t_scalar = Transition(s_le, s_ct, k_transfer=1e6)
    with pytest.raises(ValueError, match="Inconsistent shapes"):
        sys_scalar = StateSystem(transitions=[t_scalar], temperatures=temps)
        _ = sys_scalar.transfers

    # Case B: k_transfer is correct shape (3,)
    # This should PASS
    t_correct = Transition(s_le, s_ct, k_transfer=np.ones(3) * 1e6)
    sys_correct = StateSystem(transitions=[t_correct], temperatures=temps)
    assert sys_correct.transfers[(s_ct.index, s_le.index)].shape == (3,)

    # Case C: k_transfer is None
    # This should PASS because it's forced to zeros_like(temps)
    t_none = Transition(s_le, s_ct, k_transfer=None)
    sys_none = StateSystem(transitions=[t_none], temperatures=temps)
    assert sys_none.transfers[(s_ct.index, s_le.index)].shape == (3,)
