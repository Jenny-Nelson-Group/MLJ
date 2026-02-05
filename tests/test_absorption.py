import pytest
import numpy as np

from src.MLJ.physics.spectral_response import absorption


# --- Mocking the dependencies ---
# Since your function depends on a global 'config' and specific constants,
# we mock them here for the test environment.
class MockConfig:
    refractive_index = 1.5
    photon_density = 1e24


config = MockConfig()

# --- The Tests ---


def test_absorption_output_shape():
    """Verify the output shape is (n_photons, n_conditions)."""
    n_energies = 50
    n_cond = 5
    n_states = 2

    energies = np.linspace(1.0, 3.0, n_energies)
    rates = np.ones((n_energies, n_cond, n_states))

    result = absorption(energies, rates)

    # Expected shape: (n_photons, n_conditions)
    assert result.shape == (n_energies, n_cond)


def test_absorption_uniform_weighting():
    """Ensure that providing no weights defaults to 1/N weighting."""
    n_energies, n_cond, n_states = 10, 2, 4
    energies = np.ones(n_energies)
    rates = np.ones((n_energies, n_cond, n_states))

    # Total absorption with 4 states and uniform weight (0.25 each)
    # should be exactly the same as 1 state with weight 1.0 (if rates are identical)
    result_4_states = absorption(energies, rates)

    single_rate = np.ones((n_energies, n_cond, 1))
    result_1_state = absorption(energies, single_rate, weights=[1.0])

    np.testing.assert_allclose(result_4_states, result_1_state)


def test_absorption_custom_weights():
    """Verify that custom weights are applied correctly to the states."""
    n_energies, n_cond, n_states = 10, 1, 2
    energies = np.ones(n_energies)

    # State 0 has rate 1.0, State 1 has rate 0.0
    rates = np.zeros((n_energies, n_cond, n_states))
    rates[:, :, 0] = 1.0

    # If we weight state 0 at 100%, we should get full value
    res_full = absorption(energies, rates, weights=[1.0, 0.0])
    # If we weight state 0 at 50%, we should get half value
    res_half = absorption(energies, rates, weights=[0.5, 0.5])

    np.testing.assert_allclose(res_half, res_full * 0.5)


def test_absorption_mismatched_dimensions():
    """Test that mismatched energy and rate dimensions raise an error."""
    energies = np.linspace(1, 10, 50)  # length 50
    rates = np.ones((40, 5, 2))  # length 40

    # This should raise a broadcasting error because 50 != 40
    with pytest.raises(ValueError):
        absorption(energies, rates)
