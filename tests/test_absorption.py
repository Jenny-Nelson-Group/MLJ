import pytest
import numpy as np

from MLJ.physics.spectral_response import absorption
from MLJ.physics.constants import BOLTZMANN_CONSTANT_J, UNIT_CHARGE


def test_absorption_output_shape():
    """Verify the output shape is (n_E, n_T)."""
    n_states, n_E, n_T = 2, 50, 5

    # Defined with eV
    photon_energies = np.linspace(1.0, 3.0, n_E)
    optical_bandgap = 1.25

    rates = np.ones((n_states, n_E, n_T))
    temperatures = np.linspace(50, 350, n_T)

    result = absorption(photon_energies, optical_bandgap, rates, temperatures)

    assert result.shape == (n_E, n_T)


def test_absorption_piecewise_logic():
    """Verify that values above the threshold use the square-root law."""
    n_E, n_T = 100, 1

    # Range from 0.5 eV to 3.0 eV
    photon_energies = np.linspace(0.5, 5.0, n_E)
    optical_bandgap = 1.5

    temperatures = np.array([300.0])
    rates = np.zeros((1, n_E, n_T))

    result = absorption(photon_energies, optical_bandgap, rates, temperatures, device_thickness=1.0)

    # Threshold = Eg + 2kbT but now in Joules
    threshold = optical_bandgap * UNIT_CHARGE + (2.0 * BOLTZMANN_CONSTANT_J * temperatures[0])

    # Find first index where energy exceeds threshold
    high_energy_indices = np.where(photon_energies * UNIT_CHARGE > threshold)[0]
    high_energy_idx = high_energy_indices[0]

    # Above threshold, alpha should be positive (sqrt law)
    assert result[high_energy_idx, 0] > 0
    # Check that it increases with energy
    assert result[-1, 0] > result[high_energy_idx, 0]


def test_absorption_mismatched_dimensions():
    """Test that mismatched energy and rate dimensions raise an error."""
    # Define in eV
    photon_energies = np.linspace(1.0, 3.0, 50)
    rates = np.ones((2, 40, 5))  # Mismatch: 40 vs 50
    temperatures = np.linspace(50, 350, 5)
    optical_bandgap = 1.5

    with pytest.raises(ValueError):
        absorption(photon_energies, optical_bandgap, rates, temperatures)


def test_absorption_wrong_rates_ndim():
    """Test that rates not being 3D raises the specific ValueError."""
    photon_energies = np.linspace(1.0, 3.0, 50)
    rates = np.ones((10, 2))  # 2D instead of 3D
    temperatures = np.linspace(50, 350, 5)
    optical_bandgap = 1.0

    with pytest.raises(ValueError, match="Expected 3D rates"):
        absorption(photon_energies, optical_bandgap, rates, temperatures)
