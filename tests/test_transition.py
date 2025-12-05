# tests/test_transition.py

def test_import_transition():
    from MLJ.physics.transition import Transition
    from MLJ.physics.transition import TransitionType
    assert hasattr(Transition, "__doc__")
    assert hasattr(TransitionType, "__doc__")

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

def test_transition_type():
    from MLJ.physics.state import State
    from MLJ.physics.transition import Transition
    from MLJ.physics.transition import TransitionType

    excited_state = State("LE",energy=1.5)
    transition = Transition(excited_state)

    assert transition.transition_type == TransitionType.RECOMBINATION

def test_change_transition_type():
    from MLJ.physics.state import State
    from MLJ.physics.transition import Transition
    from MLJ.physics.transition import TransitionType

    excited_state = State("LE",energy=1.5)
    transition = Transition(excited_state)

    assert transition.transition_type == TransitionType.RECOMBINATION
    assert transition.set_type_absorption().transition_type == TransitionType.ABSORPTION
    assert transition.set_type_recombination().transition_type == TransitionType.RECOMBINATION


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
