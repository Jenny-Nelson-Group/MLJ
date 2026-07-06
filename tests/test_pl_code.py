from MLJ import State, Transition, Rates
from MLJ.physics.state import gaussian_distribution_nonnorm
import numpy as np

# To successfully pass this test one needs the erreonous! settings from the PL-Temp-Fit code
# 1. Set all Laguerre polynomials to 1
# 2. Use these constants:

# #Constants how they are used in the pl_fit
# UNIT_CHARGE                 = 1.6e-19               # Elementary Charge (C)
# SPEED_OF_LIGHT              = 3e8                   # Speed of Light (m/s)

# PLANCK_CONSTANT_JS          = 6.62e-34              # Planck constant (J*s)
# REDUCED_PLANCK_CONSTANT_JS  = 1.05361e-34           # Reduced Planck constant (J*s);  h/(2π)
# REDUCED_PLANCK_CONSTANT_EVS = 6.58506e-16           # Reduced Planck constant (eV*s); h/q/(2π)

# BOLTZMANN_CONSTANT_J        = 1.380649e-23          # Boltzmann constant in J/K
# BOLTZMANN_CONSTANT_EV       = 8.6173303e-5          # Boltzmann constant in eV/K

# ELECTRON_MASS               = 9.1e-31               # Electron mass in kg

# VACUUM_PERMITTIVITY_SI      = 8.85e-12              # Vacuum Permittivity epsilon_0 (F/m)
# #VACUUM_PERMITTIVITY_EV      = 55.26349406           # Vacuum Permittivity epsilon_0 (e2 eV-1 um-2)
# VACUUM_PERMITTIVITY_EV      = VACUUM_PERMITTIVITY_SI * 6.242e18
# # VACUUM_PERMITTIVITY_EV      = VACUUM_PERMITTIVITY_SI / UNIT_CHARGE  # Vacuum Permittivity epsilon_0 (eV-1 C m-2)

# 3. Check that you consider the same amount of vibrtional states, i.e.
# v_initial = np.arange(n_vib_modes_initial + 1)
# v_final = np.arange(n_vib_modes_final + 1)
# should only run up to without the +1
# 4. Double check the electronic coupling element.


def test_rates_example_values():
    """Test radiative and non-radiative rates obtained through the pl_temp_fit approach."""

    temperatures = np.array([50, 100, 200, 300])
    photon_energies = np.arange(0, 5, 0.01)

    GS = State(number_of_vibronic_modes=15)
    LE = State(
        name="Local Exciton",
        index=1,
        energy=1.4,
        number_of_vibronic_modes=2,
        vib_spacing=0.15,
        disorder_sigma=0.01,
        disorder_number_of_states=20,
        disorder_integration_cut_off=0.1,  # This is corresponds to 0.1 eV cutoff without scaling
        disorder_scaling_cut_off=False,
        disorder_distribution=gaussian_distribution_nonnorm,
    )

    transition = Transition(
        state_low_energy=GS,
        state_high_energy=LE,
        lambda_inner=0.1,
        lambda_outer=0.1,
        oscillator_strength=0.5,
        static_dipole_moment=3 * 3.33e-30 / 1.6e-19,
    )

    V_nr = transition.electronic_coupling_non_radiative  # PL Temp Fit Reference: 0.6926602668799182
    V_r = transition.electronic_coupling_radiative  # PL Temp Fit Reference: 2.1388141999123014e-10

    print("V_nr:", V_nr)
    print("V_r:", V_r)

    rates = Rates(
        transition=transition,
        photon_energies=photon_energies,
        temperatures=temperatures,
    )

    # The original code has a reverse temperature ordering
    # non_radiative_rates_expected = [6.38652756e+09, 5.13000047e+09, 3.04963337e+09, 1.49762696e+09][::-1]
    # radiative_rates_expected =  [31034962.01222066, 30595884.36215721, 30045644.03087955, 29140230.07632121][::-1]

    # With correct vib state numbering
    non_radiative_rates_expected = [6.38652885e09, 5.13000047e09, 3.04963337e09, 1.49762696e09][
        ::-1
    ]
    radiative_rates_expected = [
        31037404.94082008,
        30595891.69687299,
        30045644.03087975,
        29140230.07632121,
    ][::-1]

    # Perform the assertions
    print("Radiative rates MLJ:", rates.rate_radiative_total)
    print("Radiative rates PL:", radiative_rates_expected)
    print(
        "Relative Difference Radiative:",
        abs((rates.rate_radiative_total - radiative_rates_expected)) / radiative_rates_expected,
    )
    print("Non Radiative rates MLJ:", rates.rate_non_radiative_total)
    print("Non Radiative rates PL:", non_radiative_rates_expected)
    print(
        "Relative Difference Non-Radiative:",
        abs((rates.rate_non_radiative_total - non_radiative_rates_expected))
        / non_radiative_rates_expected,
    )
    np.testing.assert_allclose(rates.rate_radiative_total, radiative_rates_expected, rtol=1e-4)
    np.testing.assert_allclose(
        rates.rate_non_radiative_total, non_radiative_rates_expected, rtol=1e-4
    )
