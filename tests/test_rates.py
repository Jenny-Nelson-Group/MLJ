from MLJ.physics.state import State
from MLJ.physics.transition import Transition
from MLJ.physics.rates import Rates
import pytest
import numpy as np


# --- FIXTURES (Parametrised for ordered and disordered) ---
@pytest.fixture(params=["ordered", "disordered"])
def sample_transition(request):
    """
    Sets up a minimal physical system to test the Rates manager.
    Provides a Transition object for both ordered and disordered cases.
    pytest will run every test that uses this fixture TWICE.
    """
    # 1. Setup simple states and transition
    if request.param == "ordered":
        n_states = 1
    if request.param == "disordered":
        n_states = 21

    gs = State(name="GS", vib_spacing=0.15, disorder_number_of_states=n_states)
    le = State(name="LE", energy=1.5, vib_spacing=0.15, disorder_number_of_states=n_states)

    return Transition(
        state_low_energy=gs,
        state_high_energy=le,
        lambda_inner=0.1,
        lambda_outer=0.1,
    )


def test_import_and_documentation():
    """Test that Rates class was successfully imported and has a docstring."""
    assert hasattr(Rates, "__doc__")


def test_rates_output_shape(sample_transition):
    """Test the output shape of the different rate results."""
    resolution_photon_energies = 100
    resolution_temperatures = 10
    energies = np.linspace(1.5, 2.5, resolution_photon_energies)
    temperatures = np.linspace(100, 200, resolution_temperatures)
    rates = Rates(
        transition=sample_transition,
        photon_energies=energies,
        temperatures=temperatures,
    )

    disorder_number_of_states = rates.transition.state_high_energy.disorder_number_of_states

    # check shape of intermediate cached values
    assert rates.boltzmann_electronic_states.shape == (
        1,
        disorder_number_of_states,
        resolution_temperatures,
    )
    assert rates.norm_recombination.shape == (resolution_temperatures,)
    assert rates.norm_absorption.shape == (resolution_temperatures,)
    assert rates.photon_phase_space.shape == (resolution_photon_energies,)

    # check shape of final rates
    assert rates.rate_radiative_spectral.shape == (
        resolution_photon_energies,
        resolution_temperatures,
    )
    assert rates.rate_absorption_spectral.shape == (
        resolution_photon_energies,
        resolution_temperatures,
    )
    assert rates.rate_radiative_total.shape == (resolution_temperatures,)
    assert rates.rate_non_radiative_total.shape == (resolution_temperatures,)
    assert rates.rate_recombination_total.shape == (resolution_temperatures,)


def test_low_and_no_sigma():
    """Test that results of sigma=0 and very low sigma agree."""
    temperatures = np.array([50, 100, 150, 200, 250, 300, 350])
    photon_energies = np.linspace(0.5, 2.5, 200)

    exciton_low_sigma = State(
        name="Local Exciton",
        energy=1.35,
        disorder_sigma=0.00001,
        disorder_number_of_states=21,
    )

    exciton_no_sigma = State(
        name="Local Exciton",
        energy=1.35,
        disorder_sigma=0,
        disorder_number_of_states=1,
    )

    transition_low_sigma = Transition(state_high_energy=exciton_low_sigma)
    transition_no_sigma = Transition(state_high_energy=exciton_no_sigma)

    rates_low_sigma = Rates(
        transition=transition_low_sigma,
        photon_energies=photon_energies,
        temperatures=temperatures,
    )
    rates_no_sigma = Rates(
        transition=transition_no_sigma,
        photon_energies=photon_energies,
        temperatures=temperatures,
    )

    np.testing.assert_allclose(
        rates_low_sigma.rate_radiative_total,
        rates_no_sigma.rate_radiative_total,
        rtol=1e-5,
    )
    np.testing.assert_allclose(
        rates_low_sigma.rate_non_radiative_total,
        rates_no_sigma.rate_non_radiative_total,
        rtol=1e-5,
    )


def test_cache_invalidation():
    """Test that the values are correctly recalculated if the properties of Rates change."""
    temperatures_1 = np.array([100, 200, 300])
    temperatures_2 = np.array([150, 250, 350])
    photon_energies_1 = np.linspace(1.0, 3.0, 100)
    photon_energies_2 = np.linspace(0.5, 2.5, 200)

    transition_1 = Transition(
        State(
            name="Local Exciton",
            energy=1.50,
            disorder_sigma=0.01,
            disorder_number_of_states=21,
        )
    )

    transition_2 = Transition(
        State(
            name="Local Exciton",
            energy=1.40,
            disorder_sigma=0.02,
            disorder_number_of_states=21,
        )
    )

    rates = Rates(
        transition=transition_1,
        photon_energies=photon_energies_1,
        temperatures=temperatures_1,
    )

    # Swapping the transition should trigger cache invalidation
    rate_1 = rates.rate_recombination_total
    rates.transition = transition_2
    rate_2 = rates.rate_recombination_total
    assert not np.array_equal(rate_1, rate_2)

    # changing the old transition after saw should not trigger cache invalidation
    transition_1.lambda_inner = 999.0
    rate_2_check = rates.rate_recombination_total
    assert np.array_equal(rate_2_check, rate_2)

    # changing top level attributes should trigger chache invalidation
    rates.temperatures = temperatures_2
    rate_3 = rates.rate_recombination_total
    rates.photon_energies = photon_energies_2
    rate_4 = rates.rate_recombination_total
    assert not np.array_equal(rate_2, rate_3)
    assert not np.array_equal(rate_3, rate_4)

    # changing nested attributes should trigger cache invalidation
    rates.transition.lambda_inner = 0.783
    rate_5 = rates.rate_recombination_total
    rates.transition.lambda_outer = 0.567
    rate_6 = rates.rate_recombination_total
    rates.transition.state_high_energy.energy = 1.3
    rate_7 = rates.rate_recombination_total
    assert not np.array_equal(rate_4, rate_5)
    assert not np.array_equal(rate_5, rate_6)
    assert not np.array_equal(rate_6, rate_7)

    # swapping the state object inside transition should trigger c.i.
    new_state = State(energy=2.0)
    rates.transition.state_high_energy = new_state
    rate_8 = rates.rate_recombination_total
    assert not np.array_equal(rate_7, rate_8)

    # Check that None assignment does not raise AttributeError or TypeError
    rates.photon_density = None
    rates.photon_density = 1.0


def test_observable_cleanup():
    t1 = Transition(State(energy=1.0))
    t2 = Transition(State(energy=2.0))
    rates = Rates(transition=t1, photon_energies=np.array([1, 2]))

    # Switch to t2
    rates.transition = t2
    initial_rate = rates.rate_recombination_total
    assert "rate_recombination_total" in rates.__dict__

    # Modify t1 (the discarded object)
    t1.lambda_inner = 0.5

    # The cache should NOT have cleared; it should still be identical
    # If the cleanup failed, t1 would have cleared the rates cache
    # even though it's no longer the active transition.
    assert "rate_recombination_total" in rates.__dict__
    assert np.array_equal(rates.rate_recombination_total, initial_rate)
