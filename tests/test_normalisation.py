# tests/test_normalisation.py
import numpy as np
from MLJ.physics.state import gaussian_distribution
from MLJ.physics.normalisation import partition_function
from MLJ.physics.state import State
from MLJ.physics.transition import Transition
from MLJ.physics.transition import ProcessType


def test_import_normalisation():
    assert hasattr(partition_function, "__doc__")


def test_output_format_type():
    transition = Transition(State("LE", energy=1.5))
    Znorm_abs = partition_function(
        transition=transition,
        temperatures=np.array([200, 300]),
        process=ProcessType.ABSORPTION,
    )
    Znorm_rec = partition_function(
        transition=transition,
        temperatures=np.array([200, 300]),
        process=ProcessType.RECOMBINATION,
    )

    assert len(Znorm_abs) > 0
    assert len(Znorm_rec) > 0
    assert isinstance(Znorm_abs[0], float)
    assert isinstance(Znorm_rec[0], float)


def test_edge_case_zero_gibbs_no_disorder():
    ground_state = State(energy=0.0, disorder_number_of_states=1, disorder_sigma=0)
    transition = Transition(state_low_energy=ground_state, state_high_energy=ground_state)
    Znorm_abs = partition_function(
        transition=transition,
        temperatures=np.array([200, 300]),
        process=ProcessType.ABSORPTION,
    )

    assert transition.gibbs_energy_grid == np.array([0.0])
    assert Znorm_abs[0] == 1


def test_case_absorption():
    """
    In the case of a normalised Gaussian with large integration range,
    the partition function should be close to 1.
    """
    ground_state = State(
        energy=0.0,
        disorder_number_of_states=21,
        disorder_sigma=0.1,
        disorder_scaling_cut_off=False,
        disorder_distribution=gaussian_distribution,
        disorder_integration_cut_off=1,
    )

    transition = Transition(state_low_energy=ground_state, state_high_energy=ground_state)
    Znorm_abs = partition_function(
        transition=transition,
        temperatures=np.array([200, 300]),
        process=ProcessType.ABSORPTION,
    )

    assert np.isclose(Znorm_abs[0], 1, rtol=1e-3)
