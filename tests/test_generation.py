import pytest
import numpy as np
from MLJ.physics.config import config
from MLJ.physics.generation import laser_profile_gaussian, excited_state_generation


@pytest.fixture
def energy_grid():
    """Standard energy grid for testing (1.0 to 3.0 eV)."""
    return np.linspace(1.0, 3.0, 500)


def test_laser_intensity_peak(energy_grid):
    """Verify that the laser intensity peaks at config.laser_mean_energy."""
    config.laser_mean_energy = 2.0
    config.laser_broadening = 0.1

    intensity = laser_profile_gaussian(energy_grid)

    # The maximum intensity should be at the index closest to 2.0 eV
    peak_idx = np.argmax(intensity)
    assert np.isclose(energy_grid[peak_idx], 2.0, atol=0.01)


def test_custom_laser_profile_is_called():
    """Verify the function uses the provided laser profile callable."""
    n_energies = 10
    photon_energies = np.linspace(1, 5, n_energies)
    k_abs = np.ones((1, n_energies, 1))

    # Test custom laser profiles (delta peak, all zeros)
    res_ones = excited_state_generation(
        k_abs, photon_energies, lambda x: np.ones_like(x)
    )
    res_zeros = excited_state_generation(
        k_abs, photon_energies, lambda x: np.zeros_like(x)
    )
    assert np.all(res_zeros == 0)
    assert np.all(res_ones > 0)


def test_linearity_scaling():
    """Check that doubling k_abs doubles the output (Linearity)."""
    energies = np.linspace(1, 10, 5)
    k_abs = np.ones((1, 5, 1))

    val1 = excited_state_generation(k_abs, energies)
    val2 = excited_state_generation(k_abs * 2, energies)

    assert np.isclose(val2, 2 * val1).all()


def test_excited_state_generation_logic(energy_grid):
    """Test the integration of k_abs and laser intensity."""
    config.laser_mean_energy = 2.0
    config.laser_broadening = 0.2

    n_photon_energies = len(energy_grid)
    n_states = 1
    n_temps = 1

    # Case 1: k_abs is zeros -> no excited states
    k_abs_zero = np.zeros((n_states, n_photon_energies, n_temps))
    gen_zero = excited_state_generation(k_abs_zero, energy_grid)
    assert gen_zero == 0.0

    # Case 2: Constant k_abs -> population should be positive
    k_abs_const = np.ones((n_states, n_photon_energies, n_temps))
    gen_const = excited_state_generation(k_abs_const, energy_grid)
    assert gen_const > 0


@pytest.mark.parametrize(
    "n_states, n_energies, n_temps",
    [
        (1, 1, 1),  # 1 state, 1 energy, 1 temp
        (1, 3, 1),  # 1 state, 3 energies, 1 temp
        (1, 1, 3),  # 1 state, 1 energy, 3 temps
        (1, 3, 3),  # 3 energies, 3 temps
        (2, 1, 1),  # Multiple states and 1 energy, 1 temps
        (2, 5, 3),  # Multiple states and temps
    ],
)
def test_excited_state_generation_valid_shapes(n_states, n_energies, n_temps):
    """Verify that valid 3D k_abs arrays return the expected (n_states, n_temps) shape."""
    energy = np.ones(n_energies)
    k_abs = np.ones((n_states, n_energies, n_temps))

    gen = excited_state_generation(k_abs, energy)

    assert gen.shape == (n_states, n_temps)


@pytest.mark.parametrize(
    "k_shape, energy_shape",
    [
        ((3,), (3,)),  # k_abs is 1D
        ((1, 3), (3,)),  # k_abs is 2D
        ((1, 3, 3), (5,)),  # Energy mismatch (3 vs 5)
        ((2, 1, 3), (3,)),  # Energy mismatch (1 vs 3)
    ],
)
def test_excited_state_generation_invalid_inputs(k_shape, energy_shape):
    """Verify that dimensionality or length mismatches trigger a ValueError."""
    k_abs = np.ones(k_shape)
    energy = np.ones(energy_shape)

    with pytest.raises(ValueError):
        excited_state_generation(k_abs, energy)
