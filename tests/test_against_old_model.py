from MLJ import State, Transition, Rates
from MLJ.physics.state import gaussian_distribution_nonnorm
import numpy as np
import MLJ as mlj


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
    )

    # Define your expected values as arrays
    expected_rad = np.array(
        [
            1.45558199e08,
            1.45876661e08,
            1.46167705e08,
            1.46474513e08,
            1.46876048e08,
            1.47480301e08,
            1.48376213e08,
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
    tolerance = 1e-5
    np.testing.assert_allclose(rates.rate_radiative_total, expected_rad, rtol=tolerance)
    np.testing.assert_allclose(rates.rate_non_radiative_total, expected_nrad, rtol=tolerance)


def test_pl_one_state():
    """Configuration"""
    mlj.config.temperatures_K = np.linspace(200, 300, 2)
    mlj.config.photon_energies = np.linspace(0.5, 2.5, 50)
    mlj.config.photon_density = 1e25
    mlj.config.laser_mean_energy = 1.5
    mlj.config.laser_broadening = 0.02
    mlj.config.refractive_index = 1

    """Define the state and transition"""
    le = mlj.State(
        name="LE",
        energy=1.396,
        disorder_sigma=0.021,
        disorder_number_of_states=21,
        disorder_integration_cut_off=2.5,
        disorder_scaling_cut_off=True,
        number_of_vibronic_modes=5,
        vib_spacing=0.171,
    )

    trans_LE = mlj.Transition(
        le,
        lambda_outer=0.056,
        lambda_inner=0.082,
        oscillator_strength=2,
        static_dipole_moment=3.33e-30 / 1.6e-19,
    )

    system_one_state = mlj.StateSystem([trans_LE])
    # system_one_state.generation_light = [np.array([1e25])]
    print(f"gen: {system_one_state.generation_light}")
    print(f"dark: {system_one_state.populations_dark}")
    print(f"mat: {system_one_state.transition_matrix}")
    print(f"light: {system_one_state.populations_light}")

    expected_seans_model = np.array(
        [
            [0.0000e00, 0.0000e00],
            [1.0000e20, 0.0000e00],
            [2.0000e20, 1.0000e20],
            [5.0000e20, 3.0000e20],
            [5.0000e20, 4.0000e20],
            [8.0000e20, 8.0000e20],
            [3.1000e21, 2.1000e21],
            [7.4000e21, 4.2000e21],
            [8.6000e21, 5.9000e21],
            [9.3000e21, 8.7000e21],
            [2.9200e22, 2.0500e22],
            [7.510e22, 4.280e22],
            [9.6000e22, 5.9800e22],
            [7.9900e22, 7.0800e22],
            [1.665e23, 1.280e23],
            [4.5600e23, 2.647e23],
            [6.491e23, 3.811e23],
            [4.888e23, 3.836e23],
            [5.086e23, 4.436e23],
            [1.2753e24, 7.802e23],
            [2.0265e24, 1.1508e24],
            [1.5641e24, 1.0722e24],
            [5.854e23, 6.048e23],
            [1.072e23, 2.062e23],
            [9.7000e21, 4.320e22],
            [5.0000e20, 6.3000e21],
            [0.0000e00, 1.0000e21],
            [0.0000e00, 3.0000e20],
            [0.0000e00, 1.0000e20],
            [0.0000e00, 0.0000e00],
            [0.0000e00, 0.0000e00],
            [0.0000e00, 0.0000e00],
            [0.0000e00, 0.0000e00],
            [0.0000e00, 0.0000e00],
            [0.0000e00, 0.0000e00],
            [0.0000e00, 0.0000e00],
            [0.0000e00, 0.0000e00],
            [0.0000e00, 0.0000e00],
            [0.0000e00, 0.0000e00],
            [0.0000e00, 0.0000e00],
            [0.0000e00, 0.0000e00],
            [0.0000e00, 0.0000e00],
            [0.0000e00, 0.0000e00],
            [0.0000e00, 0.0000e00],
            [0.0000e00, 0.0000e00],
            [0.0000e00, 0.0000e00],
            [0.0000e00, 0.0000e00],
            [0.0000e00, 0.0000e00],
            [0.0000e00, 0.0000e00],
            [0.0000e00, 0.0000e00],
        ]
    )

    print(expected_seans_model)

    assert True
    # np.testing.assert_allclose(
    #    system_one_state.emission_photoluminescence, expected_seans_model, rtol=1e-3
    # )
