import pytest
import numpy as np
import MLJ as mlj
from MLJ.physics.spectral_response import absorption
from MLJ.physics.constants import BOLTZMANN_CONSTANT_J, UNIT_CHARGE


def test_absorption_output_shape():
    """Verify the output shape is (n_E, n_T)."""
    n_states, n_E, n_T = 2, 50, 5
    photon_energies = np.linspace(1.0, 3.0, n_E)
    temperatures = np.linspace(50, 350, n_T)
    rates = np.ones((n_states, n_E, n_T))

    # Setup real MLJ objects
    gs = mlj.State(name="S0", energy=0.0)
    le = mlj.State(name="LE", energy=1.5)
    trans_LE = mlj.Transition(le, gs, lambda_outer=0.05)
    transitions = [trans_LE]

    result = absorption(photon_energies, rates, temperatures, transitions=transitions)

    assert result.shape == (n_E, n_T)


def test_absorption_piecewise_logic():
    """Verify that values above the threshold use the square-root law."""
    n_E, n_T = 100, 1
    photon_energies = np.linspace(0.5, 5.0, n_E)
    temperatures = np.array([300.0])
    rates = np.zeros((1, n_E, n_T))

    # Define transition for 1.5 eV bandgap
    gs = mlj.State(name="S0", energy=0.0)
    le = mlj.State(name="LE", energy=1.5)
    lambda_outer = 0.05
    trans_LE = mlj.Transition(le, gs, lambda_outer=lambda_outer)
    transitions = [trans_LE]

    # Expected bandgap used in code: (1.5 - 0.0) + 0.05 = 1.55 eV
    optical_bandgap = 1.55

    result = absorption(
        photon_energies, rates, temperatures, transitions=transitions, device_thickness=1.0
    )

    threshold = optical_bandgap * UNIT_CHARGE + (2.0 * BOLTZMANN_CONSTANT_J * temperatures[0])
    high_energy_indices = np.where(photon_energies * UNIT_CHARGE > threshold)[0]

    # Ensure there is a high energy region to test
    assert len(high_energy_indices) > 0
    high_energy_idx = high_energy_indices[0]

    # Above threshold, alpha should be positive (from square-root law)
    assert result[high_energy_idx, 0] > 0
    # Values should increase as energy increases in the square-root regime
    assert result[-1, 0] > result[high_energy_idx, 0]


def test_absorption_mismatched_dimensions():
    """Test that mismatched energy and rate dimensions raise a broadcasting error."""
    photon_energies = np.linspace(1.0, 3.0, 50)
    rates = np.ones((2, 40, 5))  # Mismatch: 40 vs 50
    temperatures = np.linspace(50, 350, 5)

    gs = mlj.State(name="S0", energy=0.0)
    le = mlj.State(name="LE", energy=1.5)
    transitions = [mlj.Transition(le, gs)]

    # This will raise a ValueError when applying energy_scaling (50,1) to rates (2,40,5)
    with pytest.raises(ValueError):
        absorption(photon_energies, rates, temperatures, transitions=transitions)


def test_absorption_wrong_rates_ndim():
    """Test that rates not being 3D raises the specific ValueError."""
    photon_energies = np.linspace(1.0, 3.0, 50)
    rates = np.ones((10, 2))  # 2D instead of 3D
    temperatures = np.linspace(50, 350, 5)

    gs = mlj.State(name="S0", energy=0.0)
    le = mlj.State(name="LE", energy=1.5)
    transitions = [mlj.Transition(le, gs)]

    with pytest.raises(ValueError, match="Expected 3D rates"):
        absorption(photon_energies, rates, temperatures, transitions=transitions)
