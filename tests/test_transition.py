# tests/test_transition.py

import numpy as np
from MLJ.physics.transition import Transition
from MLJ.physics.transition import ProcessType
from MLJ.physics.state import State

def test_import_transition():
    assert hasattr(Transition, "__doc__")
    assert hasattr(ProcessType, "__doc__")

def test_transition_energy():
    from MLJ.physics.state import State
    from MLJ.physics.transition import Transition

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
    from MLJ.physics.state import State
    from MLJ.physics.transition import Transition

    ground_state = State()
    excited_state = State("LE",energy=1.5)

    transition = Transition(ground_state, excited_state)
    huang_rhys = transition.lambda_inner / excited_state.vib_spacing
    assert huang_rhys == transition.huang_rhys

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
