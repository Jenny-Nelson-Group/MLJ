# tests/test_basics.py
import numpy as np
import MLJ.physics.basics as bs

def test_true():
    """Set to False if you want to check that failed tests trigger something."""
    assert True

def test_dirac_delta():
    number_of_states = 5
    weights = bs.dirac_delta(num=number_of_states)
    assert len(weights) == number_of_states
    assert sum(weights) == 1

def test_dirac_delta_odd_even():
    weights_4 = bs.dirac_delta(num=4)
    weights_5 = bs.dirac_delta(num=5)
    weights_6 = bs.dirac_delta(num=6)
    weights_7 = bs.dirac_delta(num=7)
    assert all(weights_4 == np.array([ 0., 0., 1., 0.]))
    assert all(weights_5 == np.array([0., 0., 1., 0., 0.]))
    assert all(weights_6 == np.array([0., 0., 1., 0., 0., 0.]))
    assert all(weights_7 == np.array([0., 0., 0., 1., 0., 0., 0.]))

def test_dirac_pos():
    number_of_states = 7
    weights_0 = bs.dirac_delta(num=number_of_states, pos=0)
    weights_1 = bs.dirac_delta(num=number_of_states, pos=-1)
    weights_2 = bs.dirac_delta(num=number_of_states, pos=1)
    weights_3 = bs.dirac_delta(num=number_of_states, pos=0.3)
    weights_4 = bs.dirac_delta(num=number_of_states, pos=0.4)
    weights_5 = bs.dirac_delta(num=number_of_states, pos=0.51)
    assert all(weights_0 == np.array([0., 0., 0., 1., 0., 0., 0.]))
    assert all(weights_1 == np.array([1., 0., 0., 0., 0., 0., 0.]))
    assert all(weights_2 == np.array([0., 0., 0., 0., 0., 0., 1.]))
    assert all(weights_3 == np.array([0., 0., 0., 0., 1., 0., 0.]))
    assert all(weights_4 == np.array([0., 0., 0., 0., 1., 0., 0.]))
    assert all(weights_5 == np.array([0., 0., 0., 0., 0., 1., 0.]))

def test_dirac_edge_case_one():
    number_of_states = 1
    weights = bs.dirac_delta(num=number_of_states)
    assert weights == np.array([1.])

def test_dirac_edge_case_one_pos():
    number_of_states = 1
    weights_1 = bs.dirac_delta(num=number_of_states, pos=-1)
    weights_2 = bs.dirac_delta(num=number_of_states, pos=1)
    assert weights_1 == np.array([1.])
    assert weights_2 == np.array([1.])