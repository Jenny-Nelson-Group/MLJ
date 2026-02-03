# tests/test_basics.py
import pytest
import numpy as np
from MLJ.physics.basics import boltzmann, dirac_delta
from numpy.testing import assert_allclose


def test_true():
    """Set to False if you want to check that failed tests trigger something."""
    assert True


def test_boltzmann_scalar_scalar():
    # Both floats: should return a scalar (or 0-d array)
    res = boltzmann(1.0, 300.0)
    assert np.isscalar(res) or res.shape == ()


def test_boltzmann_scalar_1d():
    # One scalar, one 1D: should return 1D array
    temps = np.array([300.0, 400.0])
    res = boltzmann(1.0, temps)
    assert res.shape == (2,)


def test_boltzmann_1d_scalar():
    # One 1D, one scalar: should return 1D array
    energies = np.array([1.0, 1.2])
    res = boltzmann(energies, 300.0)
    assert res.shape == (2,)


def test_boltzmann_1d_1d_grid():
    # Both 1D: triggers the 'Safety Net' to create an (N, M) grid
    energies = np.array([1.0, 1.2, 1.4])  # N=3
    temps = np.array([300.0, 400.0])  # M=2
    res = boltzmann(energies, temps)

    assert res.shape == (3, 2)
    # Verify a specific value: res[row, col] -> boltz(E[row], T[col])
    expected_val = boltzmann(1.0, 300.0)
    assert_allclose(res[0, 0], expected_val)


def test_boltzmann_1d_2d_explicit():
    # One is already 2D: bypasses safety net, uses standard broadcasting
    # (3, 1) and (1, 2) -> (3, 2)
    energies = np.array([1.0, 1.2, 1.4])[:, None]
    temps = np.array([300.0, 400.0])[None, :]
    res = boltzmann(energies, temps)

    assert res.shape == (3, 2)


def test_boltzmann_2d_2d_elementwise():
    # Both 2D: bypasses safety net, performs element-wise math
    # Useful for spatial maps or pre-aligned data
    energies = np.ones((2, 2))
    temps = np.full((2, 2), 300.0)
    res = boltzmann(energies, temps)

    assert res.shape == (2, 2)


def test_boltzmann_3d_tensor():
    # Test larger tensors (e.g., State x X-coord x Y-coord)
    energies = np.ones((5, 10, 10))
    temp = 300.0
    res = boltzmann(energies, temp)

    assert res.shape == (5, 10, 10)


def test_boltzmann_invalid_mixed_shapes():
    # Energies is 1D (size 3), Temps is 2D (2x2)
    # These cannot be broadcast together by standard NumPy rules
    energies = np.array([1.0, 1.2, 1.4])
    temps = np.ones((2, 2)) * 300.0

    with pytest.raises(ValueError):
        boltzmann(energies, temps)


def test_dirac_delta():
    number_of_states = 5
    weights = dirac_delta(num=number_of_states)
    assert len(weights) == number_of_states
    assert sum(weights) == 1


def test_dirac_delta_odd_even():
    weights_4 = dirac_delta(num=4)
    weights_5 = dirac_delta(num=5)
    weights_6 = dirac_delta(num=6)
    weights_7 = dirac_delta(num=7)
    assert all(weights_4 == np.array([0.0, 0.0, 1.0, 0.0]))
    assert all(weights_5 == np.array([0.0, 0.0, 1.0, 0.0, 0.0]))
    assert all(weights_6 == np.array([0.0, 0.0, 1.0, 0.0, 0.0, 0.0]))
    assert all(weights_7 == np.array([0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0]))


def test_dirac_pos():
    number_of_states = 7
    weights_0 = dirac_delta(num=number_of_states, pos=0)
    weights_1 = dirac_delta(num=number_of_states, pos=-1)
    weights_2 = dirac_delta(num=number_of_states, pos=1)
    weights_3 = dirac_delta(num=number_of_states, pos=0.3)
    weights_4 = dirac_delta(num=number_of_states, pos=0.4)
    weights_5 = dirac_delta(num=number_of_states, pos=0.51)
    assert all(weights_0 == np.array([0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0]))
    assert all(weights_1 == np.array([1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]))
    assert all(weights_2 == np.array([0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0]))
    assert all(weights_3 == np.array([0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0]))
    assert all(weights_4 == np.array([0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0]))
    assert all(weights_5 == np.array([0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0]))


def test_dirac_edge_case_one():
    number_of_states = 1
    weights = dirac_delta(num=number_of_states)
    assert weights == np.array([1.0])


def test_dirac_edge_case_one_pos():
    number_of_states = 1
    weights_1 = dirac_delta(num=number_of_states, pos=-1)
    weights_2 = dirac_delta(num=number_of_states, pos=1)
    assert weights_1 == np.array([1.0])
    assert weights_2 == np.array([1.0])
