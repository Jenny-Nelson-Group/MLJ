# tests/test_transition.py
import pytest
import numpy as np
from MLJ.physics.transition import Transition
from MLJ.physics.transition import ProcessType
from MLJ.physics.state import State

def test_import_transition():
    assert hasattr(Transition, "__doc__")
    assert hasattr(ProcessType, "__doc__")

def test_transition_energy():
    ground_state = State()
    excited_state = State("LE",energy=1.5)

    transition1 = Transition(state_low_energy=ground_state, state_high_energy=excited_state)
    transition2 = Transition(excited_state, ground_state)
    transition3 = Transition(excited_state)
    transition4 = Transition(ground_state, excited_state)

    energy_difference = abs(excited_state.energy - ground_state.energy)
    assert energy_difference == transition1.mean_gibbs_energy
    assert energy_difference == transition2.mean_gibbs_energy
    assert energy_difference == transition3.mean_gibbs_energy
    assert energy_difference == transition4.mean_gibbs_energy

def test_huang_rhys():
    ground_state = State()
    excited_state = State("LE",energy=1.5)

    transition = Transition(ground_state, excited_state)
    huang_rhys = transition.lambda_inner / excited_state.vib_spacing
    assert huang_rhys == transition.huang_rhys

def test_huang_rhys_cache_invalidation():
    ground_state = State()
    excited_state = State("LE", energy=1.5)
    transition = Transition(ground_state, excited_state, lambda_inner=0.15)

    huang_rhys_1 = transition.huang_rhys
    transition.lambda_inner = 0.07
    huang_rhys_2 = transition.huang_rhys
    assert huang_rhys_1 != huang_rhys_2

def test_repr():
    from MLJ.physics.state import State
    from MLJ.physics.transition import Transition

    ground_state = State()
    excited_state = State("LE",energy=1.5)

    transition = Transition(ground_state, excited_state)
    assert isinstance(repr(transition), str)


def test_edge_case_no_disorder():
    """Test that without disorder the gibbs_energy_grid contain only one value equal to the mean."""
    from MLJ.physics.state import State
    from MLJ.physics.transition import Transition

    ground_state = State()
    excited_state = State("LE",energy=1.5,disorder_sigma=0)
    transition = Transition(ground_state, excited_state)

    assert len(transition.gibbs_energy_grid) == 1
    assert transition.gibbs_energy_grid[0] == transition.mean_gibbs_energy
    assert transition.disorder_weights == np.array([1])

def test_assign_indices_automatic_sorting():
    """
    Test that assign_indices correctly identifies unique states and
    assigns indices 0, 1, 2 based on increasing energy.
    """
    # 1. Create states with out-of-order energies
    s_ct = State(energy=1.5, name="CT")
    s_le = State(energy=2.1, name="LE")

    # 2. Create transitions (using varied order)
    t_le_0 = Transition(state_high_energy=s_le)
    t_ct_0 = Transition(state_high_energy=s_ct)
    t_le_ct = Transition(state_high_energy=s_le, state_low_energy=s_ct)

    transitions = [t_le_0, t_ct_0, t_le_ct]

    # 3. Run the class method
    sorted_states = Transition.assign_indices(transitions)

    # 4. Assertions
    # Check that indices were assigned correctly based on energy
    assert s_ct.index == 1
    assert s_le.index == 2

    # Check the return list order
    assert sorted_states[0].name == "Ground State"
    assert sorted_states[2].name == "LE"

    # Check that the transition index property works correctly
    assert t_le_ct.index == (2, 1)
    assert t_ct_0.index == (1, 0)
    assert t_le_0.index == (2, 0)

def test_assign_indices_unique_states():
    """
    Test that if the same state is used in multiple transitions,
    it is only indexed once.
    """
    s0 = State(energy=0.0)
    s1 = State(energy=1.0)
    s2 = State(energy=2.0)

    # Both transitions share s1
    t1 = Transition(s1, s0)
    t2 = Transition(s2, s1)

    all_states = Transition.assign_indices([t1, t2])

    # Only 3 unique states should be indexed
    assert len(all_states) == 3
    assert s1.index == 1