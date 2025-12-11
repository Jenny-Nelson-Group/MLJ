# tests/test_normalisation.py
import numpy as np
from MLJ.physics.state import gaussian_distribution

def test_import_normalisation():
    from MLJ.physics.normalisation import partition_function
    assert hasattr(partition_function, "__doc__")


def test_output_format_type():
    from MLJ.physics.state import State
    from MLJ.physics.transition import Transition
    from MLJ.physics.transition import TransitionType
    from MLJ.physics.normalisation import partition_function

    transition = Transition(State("LE",energy=1.5))

    Znorm = partition_function(transition=transition)

    assert len(Znorm) > 0
    assert isinstance(Znorm[0], float)


def test_edge_case_zero_gibbs_no_disorder():
    from MLJ.physics.state import State
    from MLJ.physics.transition import Transition
    from MLJ.physics.transition import TransitionType
    from MLJ.physics.normalisation import partition_function

    ground_state = State(energy=0.0, disorder_number_of_states=1, disorder_sigma=0)
    transition = Transition(state_low_energy=ground_state, state_high_energy=ground_state, transition_type = TransitionType.ABSORPTION)
    Znorm_abs = partition_function(transition=transition)

    assert transition.gibbs_energy_grid == np.array([0.])
    assert Znorm_abs[0] == 1

def test_case_absorption():
    from MLJ.physics.state import State
    from MLJ.physics.transition import Transition
    from MLJ.physics.transition import TransitionType
    from MLJ.physics.normalisation import partition_function

    ground_state = State(energy=0.0, disorder_number_of_states=21, disorder_sigma=0.1, disorder_distribution=gaussian_distribution, disorder_integration_cut_off=4)
    transition = Transition(state_low_energy=ground_state, state_high_energy=ground_state, transition_type = TransitionType.ABSORPTION)
    Znorm_abs = partition_function(transition=transition)

    assert np.isclose(Znorm_abs[0], 1, rtol=1e-3)
