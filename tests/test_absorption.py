import pytest
import numpy as np

from MLJ.physics.spectral_response import absorption


# --- The Tests ---
def test_absorption_output_shape():
    """Verify the output shape is (n_photon_energies, n_conditions)."""
    n_states = 2
    n_photon_energies = 50
    n_conditions = 5

    photon_energies = np.linspace(1.0, 3.0, n_photon_energies)
    rates = np.ones((n_states, n_photon_energies, n_conditions))

    result = absorption(photon_energies, rates)

    # Expected shape: (n_photon_energies, n_conditions)
    assert result.shape == (n_photon_energies, n_conditions)


def test_absorption_uniform_weighting():
    """Ensure that providing no weights defaults to 1/N weighting."""
    n_states, n_photon_energies, n_conditions = 2, 10, 4
    photon_energies = np.ones(n_photon_energies)
    rates = np.ones((n_states, n_photon_energies, n_conditions))

    # Total absorption with 4 states and uniform weight (0.25 each)
    # should be exactly the same as 1 state with weight 1.0 (if rates are identical)
    result_4_states = absorption(photon_energies, rates)

    single_rate = np.ones((1, n_photon_energies, n_conditions))
    result_1_state = absorption(photon_energies, single_rate, weights=[1.0])

    np.testing.assert_allclose(result_4_states, result_1_state)


def test_absorption_custom_weights():
    """Verify that custom weights are applied correctly to the states."""
    n_states, n_photon_energies, n_conditions = 2, 10, 1
    photon_energies = np.ones(n_photon_energies)

    # State 0 has rate 1.0, State 1 has rate 0.0
    rates = np.zeros((n_states, n_photon_energies, n_conditions))
    rates[0, :, :] = 1.0

    # If we weight state 0 at 100%, we should get full value
    res_full = absorption(photon_energies, rates, weights=[1.0, 0.0])
    # If we weight state 0 at 50%, we should get half value
    res_half = absorption(photon_energies, rates, weights=[0.5, 0.5])

    np.testing.assert_allclose(res_half, res_full * 0.5)


def test_absorption_mismatched_dimensions():
    """Test that mismatched energy and rate dimensions raise an error."""
    photon_energies = np.linspace(1, 10, 50)  # length 50
    rates = np.ones((2, 40, 5))  # length 40

    # This should raise a broadcasting error because 50 != 40
    with pytest.raises(ValueError):
        absorption(photon_energies, rates)
