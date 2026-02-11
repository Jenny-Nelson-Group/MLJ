from MLJ import State, Transition, Rates
from MLJ.physics.state import gaussian_distribution_nonnorm
import numpy as np


def test_rates_example_values():
    """Test a handful of example values, to make sure we get consistent results."""
    # We can replace this with something more sophisticated in the future,
    # especially since example values might change when updating theory.

    temperatures = np.array([50, 100, 150, 200, 250, 300, 350])
    photon_energies = np.linspace(0.5, 2.5, 200)

    GS = State(number_of_vibronic_modes=15)
    LE = State(
        name="Local Exciton",
        index=1,
        energy=1.35,
        number_of_vibronic_modes=5,
        vib_spacing=0.15,
        disorder_sigma=0.001,
        disorder_number_of_states=21,
        disorder_integration_cut_off=5,
        disorder_scaling_cut_off=True,
        disorder_distribution=gaussian_distribution_nonnorm,
    )

    transition = Transition(
        state_low_energy=GS,
        state_high_energy=LE,
        lambda_inner=0.1,
        lambda_outer=0.1,
        oscillator_strength=2.56,
        static_dipole_moment=3 * 3.33e-30 / 1.6e-19,
    )

    rates = Rates(
        transition=transition,
        photon_energies=photon_energies,
        temperatures=temperatures,
        photon_density=1,
    )

    # Define your expected values as arrays
    expected_rad = np.array(
        [
            1.45558199e08,
            1.45876661e08,
            1.46166909e08,
            1.46460002e08,
            1.46793113e08,
            1.47214535e08,
            1.47763387e08,
        ]
    )

    expected_nrad = np.array(
        [
            6.96412539e09,
            1.01356447e10,
            1.06990542e10,
            1.09853505e10,
            1.16983874e10,
            1.30201902e10,
            1.49750185e10,
        ]
    )

    # Perform the assertions
    np.testing.assert_allclose(rates.rate_radiative_total, expected_rad, rtol=1e-7)
    np.testing.assert_allclose(rates.rate_non_radiative_total, expected_nrad, rtol=1e-7)
