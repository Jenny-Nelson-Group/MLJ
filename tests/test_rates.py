from MLJ.physics.state import State, gaussian_distribution_nonnorm
from MLJ.physics.transition import Transition, ProcessType
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

    gs = State(name='GS', vib_spacing=0.15, disorder_number_of_states=n_states)
    le = State(name='LE', energy=1.5, vib_spacing=0.15, disorder_number_of_states=n_states)

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
    rates = Rates(transition=sample_transition, photon_energies=energies, temperatures=temperatures)

    disorder_number_of_states = rates.transition.state_high_energy.disorder_number_of_states

    # check shape of intermediate cached values
    assert rates.boltzmann_electronic_states.shape == (1, disorder_number_of_states, resolution_temperatures)
    assert rates.norm_recombination.shape == (resolution_temperatures,)
    assert rates.norm_absorption.shape == (resolution_temperatures,)
    assert rates.photon_phase_space.shape == (resolution_photon_energies,)

    # check shape of final rates
    assert rates.rate_radiative_spectral.shape == (resolution_photon_energies, resolution_temperatures)
    assert rates.rate_absorption_spectral.shape == (resolution_photon_energies, resolution_temperatures)
    assert rates.rate_radiative_total.shape == (resolution_temperatures,)
    assert rates.rate_non_radiative_total.shape == (resolution_temperatures,)
    assert rates.rate_recombination_total.shape == (resolution_temperatures,)

def test_rates_example_values(sample_transition):
    """Test a handful of example values, to make sure we get consistent results."""
    # We can replace this with something more sophisticated in the future,
    # especially since example values might change when updating theory.

    temperatures=np.array([50,100,150,200,250,300,350])
    photon_energies = np.linspace(0.5, 2.5, 200)

    GS = State(number_of_vibronic_modes=15)
    LE = State(name="Local Exciton",
            index=1,
            energy=1.35,
            number_of_vibronic_modes=5,
            vib_spacing=0.15,
            disorder_sigma=0.001,
            disorder_number_of_states=21,
            disorder_integration_cut_off=5,
            disorder_distribution=gaussian_distribution_nonnorm,
            )

    transition =  Transition(state_low_energy=GS,
                            state_high_energy=LE,
                            lambda_inner=0.1,
                            lambda_outer=0.1,
                            oscillator_strength=2.56,
                            static_dipole_moment=3*3.33e-30/1.6e-19
                            )

    rates = Rates(transition=transition, photon_energies=photon_energies,
                  temperatures=temperatures, photon_density=1)

    # Define your expected values as arrays
    expected_rad = np.array([1.45558199e+08, 1.45876661e+08, 1.46166909e+08,
                            1.46460002e+08, 1.46793113e+08, 1.47214535e+08, 1.47763387e+08])

    expected_nrad = np.array([6.96412539e+09, 1.01356447e+10, 1.06990542e+10,
                            1.09853505e+10, 1.16983874e+10, 1.30201902e+10, 1.49750185e+10])

    # Perform the assertions
    np.testing.assert_allclose(rates.rate_radiative_total, expected_rad, rtol=1e-7)
    np.testing.assert_allclose(rates.rate_non_radiative_total, expected_nrad, rtol=1e-7)


def test_low_and_no_sigma():
    """Test that results of sigma=0 and very low sigma agree."""
    temperatures=np.array([50,100,150,200,250,300,350])
    photon_energies = np.linspace(0.5, 2.5, 200)

    exciton_low_sigma = State(name="Local Exciton",
            energy=1.35,
            disorder_sigma=0.00001,
            disorder_number_of_states=21,
            )

    exciton_no_sigma = State(name="Local Exciton",
            energy=1.35,
            disorder_sigma=0,
            disorder_number_of_states=1,
            )

    transition_low_sigma =  Transition(state_high_energy=exciton_low_sigma)
    transition_no_sigma =  Transition(state_high_energy=exciton_no_sigma)

    rates_low_sigma = Rates(transition=transition_low_sigma,
                            photon_energies=photon_energies, temperatures=temperatures)
    rates_no_sigma = Rates(transition=transition_no_sigma,
                            photon_energies=photon_energies, temperatures=temperatures)

    np.testing.assert_allclose(rates_low_sigma.rate_radiative_total, rates_no_sigma.rate_radiative_total, rtol=1e-5)
    np.testing.assert_allclose(rates_low_sigma.rate_non_radiative_total, rates_no_sigma.rate_non_radiative_total, rtol=1e-5)
