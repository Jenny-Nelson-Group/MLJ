import pytest
import numpy as np

from MLJ.physics.spectral_response import absorption
from MLJ.physics.constants import BOLTZMANN_CONSTANT_J


def test_absorption_output_shape():
    """Verify the output shape is (n_E, n_T)."""
    n_states, n_E, n_T = 2, 50, 5

    # Energies in Joules (approx 1-3 eV)
    photon_energies = np.linspace(1.6e-19, 4.8e-19, n_E)
    rates = np.ones((n_states, n_E, n_T))
    temperatures = np.linspace(50, 400, n_T)
    optical_bandgap = 2.0e-19  # Bandgap in Joules

    # Added missing mandatory args: optical_bandgap and temperatures_K
    result = absorption(photon_energies, optical_bandgap, rates, temperatures)

    assert result.shape == (n_E, n_T)


def test_absorption_piecewise_logic():
    """Verify that values above the threshold use the square-root law."""
    n_E, n_T = 100, 1
    # Wide range of energies to cross the threshold
    photon_energies = np.linspace(0.5e-19, 10.0e-19, n_E)
    temperatures = np.array([300.0])
    optical_bandgap = 2.0e-19
    rates = np.zeros((1, n_E, n_T))  # Rates 0 to see square-root law clearly

    result = absorption(photon_energies, optical_bandgap, rates, temperatures, device_thickness=1.0)

    # Threshold = Eg + 2kbT
    threshold = optical_bandgap + (2.0 * BOLTZMANN_CONSTANT_J * temperatures[0])

    # For E > threshold, the result should be alpha_0 * sqrt(...)
    # Since alpha_0 = 2/thickness and thickness=1, alpha_0 = 2.0
    high_energy_idx = np.where(photon_energies > threshold)[0][0]
    assert result[high_energy_idx, 0] > 0
    # Check square root scaling roughly
    assert result[-1, 0] > result[high_energy_idx, 0]


def test_absorption_mismatched_dimensions():
    """Test that mismatched energy and rate dimensions raise a Broadcasting error."""
    # Length 50 vs Rate length 40
    photon_energies = np.linspace(1.6e-19, 4.8e-19, 50)
    rates = np.ones((2, 40, 5))
    temperatures = np.ones(5)
    optical_bandgap = 2.0e-19

    # This will fail during the energy_scaling multiplication or piecewise mask
    with pytest.raises(ValueError):
        absorption(photon_energies, optical_bandgap, rates, temperatures)


def test_absorption_wrong_rates_ndim():
    """Test that rates not being 3D raises the specific ValueError in the function."""
    photon_energies = np.ones(10)
    rates = np.ones((10, 2))  # 2D instead of 3D
    temperatures = np.ones(2)
    optical_bandgap = 1.0

    with pytest.raises(ValueError, match="Expected 3D rates"):
        absorption(photon_energies, optical_bandgap, rates, temperatures)
