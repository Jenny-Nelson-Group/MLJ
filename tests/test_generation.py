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

def test_excited_state_generation_logic(energy_grid):
    """Test the integration of k_abs and laser intensity."""
    config.laser_mean_energy = 2.0
    config.laser_broadening = 0.2

    # Case 1: k_abs is zeros -> no excited states
    k_abs_zero = np.zeros_like(energy_grid)
    gen_zero = excited_state_generation(k_abs_zero, energy_grid)
    assert gen_zero == 0.0

    # Case 2: Constant k_abs -> population should be positive
    k_abs_const = np.ones_like(energy_grid)
    gen_const = excited_state_generation(k_abs_const, energy_grid)
    assert gen_const > 0

def test_1d():
    """Check that 1d arrays are integrated to a scalar."""
    energy = np.array([1.3, 1.4, 15])
    k_abs = np.array([1.0, 2.0, 3.0])

    gen = excited_state_generation(k_abs, energy)
    print(gen)
    assert gen.shape == (1,)

def test_2d():
    """Check 2 temperatures leads to 2d output."""
    energy = np.array([1.3, 1.4, 15])
    k_abs = np.array([[1.0, 2.0],[1.0, 2.0],[1.0, 2.0]])

    gen = excited_state_generation(k_abs, energy)
    print(gen)
    assert gen.shape == (2,)


def test_math_mismatch_error():
    """Ensure behavior when arrays don't match."""
    energy_short = np.array([1.0, 2.0])
    k_abs_long = np.array([1.0, 2.0, 3.0])

    with pytest.raises(ValueError):
        # This should fail during multiplication or inside the integral function
        excited_state_generation(k_abs_long, energy_short)
