# tests/test_fcwd.py
import numpy as np
import pytest
from scipy.signal import find_peaks
from scipy.integrate import trapezoid

from MLJ.physics.FCWD import fcwd
from MLJ.physics.state import State
from MLJ.physics.transition import Transition, TransitionType
from MLJ.physics.config import config 

# --- FIXTURES (Parametrised for ordered and disordered) ---
@pytest.fixture(params=["ordered", "disordered"])
def sample_transition(request):
    """
    Provides a Transition object for both ordered and disordered cases.
    pytest will run every test that uses this fixture TWICE.
    """
    vibrational_spacing = 0.15 
    gs_le_energy_gap = 1.5       
    
    if request.param == "ordered":
        n_states = 1
    if request.param == "disordered":
        # Note: Using the same number for both states to ensure a 1:1 energy gap grid
        n_states = 10 
        
    gs = State(name='GS', vib_spacing=vibrational_spacing, disorder_number_of_states=n_states)
    le = State(name='LE', energy=gs_le_energy_gap, vib_spacing=vibrational_spacing, disorder_number_of_states=n_states)
    
    return Transition(
        state_low_energy=gs,
        state_high_energy=le,
        transition_type=TransitionType.ABSORPTION,
        lambda_inner=0.15,
        lambda_outer=0.05
    )

# --- TESTS ---
# 1. Test output shape 
# 2. Test temperature dependent broadening
# 3. Test functionality when considering no vibrational spacing 
# 4. Test that recombination peak is always less or equal in energy compared to emission
# 5. Test that peak intensity ratio of the first two transition equals Huang-Rhys Factor

def test_fcwd_output_shape(sample_transition):
    """Verifies that the output array has the correct (Energy, Disorder, Temp) dimensions."""
    n_photon_energies = 50
    n_temperatures = 2
    
    photon_energies = np.linspace(1.0, 2.0, n_photon_energies)
    temps = np.array([100.0, 300.0])
    
    fcwd_spectrum = fcwd(photon_energies, sample_transition, temperatures=temps)
    
    # Expected shape: (50, number_of_states, 2)
    number_of_states = sample_transition.state_high_energy.disorder_number_of_states
    assert fcwd_spectrum.shape == (n_photon_energies, number_of_states, n_temperatures)


def test_temperature_broadening(sample_transition):
    """Verifies that high temperature lowers the peak intensity (area stays same, width increases)."""
    photon_energies = np.linspace(1.0, 2.0, 100)
    
    # Compare 50K and 500K; 0 and 0 index since no disorder and only single temperature is considered
    fcwd_spectrum_cold = fcwd(photon_energies, sample_transition, temperatures=np.array([50.0]))[:, 0, 0]
    fcwd_spectrum_hot = fcwd(photon_energies, sample_transition, temperatures=np.array([500.0]))[:, 0, 0]

    assert np.max(fcwd_spectrum_cold) > np.max(fcwd_spectrum_hot), "Cold peak should be sharper and higher than hot peak"

def test_zero_vibrational_levels(sample_transition, request):
    """
    Checks FCWD logic, with a peak position check only for the ordered case.
    """
    # 1. Force ground vibrational levels
    sample_transition.state_low_energy.number_of_vibronic_modes = 0
    sample_transition.state_high_energy.number_of_vibronic_modes = 0
    
    # 2. Define energies
    energy_value_00_peak = 1.5 + sample_transition.lambda_outer
    photon_energies = np.linspace(energy_value_00_peak - 0.5, energy_value_00_peak + 0.5, 100)

    # 3. Calculate FCWD
    fcwd_spectrum = fcwd(photon_energies, sample_transition)
    y_values = fcwd_spectrum[:, 0, 0]

    # 4. Standard Assertions
    assert not np.isnan(y_values).any()
    assert (fcwd_spectrum >= 0).all()

    # 5. Access the parameter directly from the 'request' object
    fixture_param = request.node.callspec.params.get("sample_transition")

    if fixture_param == "ordered":
        peak_idx = np.argmax(y_values)
        assert np.isclose(photon_energies[peak_idx], energy_value_00_peak, atol=0.05)
    else:
        # Logic for disordered state (if needed)
        print("Skipping centered-peak check for disordered state.")

def test_abs_vs_rec_peak_shift(sample_transition):
    """Verifies that Recombination occurs at lower energy than Absorption."""
    photon_energies = np.linspace(0.5, 2.5, 100)
    
    # Case 1: Absorption
    sample_transition.transition_type = TransitionType.ABSORPTION
    fcwd_spectrum_abs = fcwd(photon_energies, sample_transition)[:, 0, 0]
    peak_abs = photon_energies[np.argmax(fcwd_spectrum_abs)]
    
    # Case 2: Recombination
    sample_transition.transition_type = TransitionType.RECOMBINATION
    fcwd_spectrum_rec = fcwd(photon_energies, sample_transition)[:, 0, 0]
    peak_rec = photon_energies[np.argmax(fcwd_spectrum_rec)]
    
    # Emission (Recombination) peak should be less or equal than the recombination peak 
    assert peak_rec <= peak_abs, f"Rec peak ({peak_rec}) should be < Abs peak ({peak_abs})"

#Iterate over different lamda_inner with fixed vibrational spacing
# And verify that t of the huang rhys factor
@pytest.mark.parametrize("lambda_inner", [0.1, 0.5])
def test_huang_rhys_factor_ratio(lambda_inner):
    """
    Checks if the ratio of I(0-1) / I(0-0) in the calculated spectrum 
    matches the input Huang-Rhys factor (S).
    """
    # 1. Set the vibrational spacing to be the same for both states (No Dushinsky Effect)
    vibrational_spacing = 0.15

    # 2. Setup States and Transition
    config.temperatures_K = np.array([40.0]) #Set low-temperature for more pronounced peaks
    gs_le_energy_gap = 1.5  # LE is 1.5 eV above ground
    ground_state = State(name='GS', vib_spacing=vibrational_spacing) 
    excited_state = State(name="LE", energy=gs_le_energy_gap,vib_spacing=vibrational_spacing) 

    #Set inner reorganisation energy for the transition that uniquely determines the Huang-Rhys
    huang_rhys_target = lambda_inner/vibrational_spacing

    absorption_transition = Transition(
        state_low_energy=ground_state, 
        state_high_energy=excited_state, 
        transition_type=TransitionType.ABSORPTION,
        lambda_inner=lambda_inner
    )

    # Sanity Check: Verify the Transition calculates the correct S
    assert np.isclose(absorption_transition.huang_rhys, huang_rhys_target), \
        f"Internal calculation mismatch: Expected S={huang_rhys_target}, Got {absorption_transition.huang_rhys}"
    
    # 3. Calculate FCWD Values (Spectrum)
    resolution = 500  
    
    # Range: Start slightly below gs_le_energy_gap (1.5 eV) to gs_le_energy_gap + 5*vibrational_spacing
    photon_energies = np.linspace(gs_le_energy_gap - 0.5, gs_le_energy_gap + 5 * vibrational_spacing, resolution)
    fcwd_spectrum_abs = fcwd(photon_energies, absorption_transition)
    y_abs = fcwd_spectrum_abs[:, 0, 0] 

    # 4. Extract Intensities
    intensity_value_00_peak, intensity_value_01_peak = get_i00_i01_intensities(y_abs)

    # Handle peak finding failure
    if intensity_value_00_peak is None or intensity_value_01_peak is None:
        pytest.fail(f"Could not resolve 0-0 and 0-1 peaks for S={huang_rhys_target}. Check resolution or range.")
        
    # 5. Calculate the ratio
    huang_rhys_calculated = intensity_value_01_peak / intensity_value_00_peak

    # 6. Assertion
    # Use a reasonable tolerance for numerical and peak finding errors.
    assert np.isclose(huang_rhys_calculated, huang_rhys_target, rtol=0.05), \
        f"FCWD ratio I(0-1)/I(0-0) ({huang_rhys_calculated:.3f}) does not match input S factor ({huang_rhys_target:.3f})."

def test_fcwd_integral_normalization(sample_transition):
    """
    Verifies that the integral of the FCWD over the energy range is 1.0.
    This checks if the Gaussian broadening and Boltzmann weights are normalized.
    """
    # Use a wide energy range to capture the full wings of the spectrum
    # Range should cover E_gap +/- several units of lambda and vib_spacing
    photon_energies = np.linspace(0.1, 3.0, 1000) 
    
    # Test Absorption
    sample_transition.transition_type = TransitionType.ABSORPTION
    y_abs = fcwd(photon_energies, sample_transition)[:, 0, 0]
    area_abs = trapezoid(y_abs, photon_energies)
    
    # Test Recombination
    sample_transition.transition_type = TransitionType.RECOMBINATION
    y_rec = fcwd(photon_energies, sample_transition)[:, 0, 0]
    area_rec = trapezoid(y_rec, photon_energies)

    # Assertions: Allow a small tolerance for numerical integration and truncated tails
    assert np.isclose(area_abs, 1.0, atol=1e-2), f"Abs area was {area_abs}, expected 1.0"
    assert np.isclose(area_rec, 1.0, atol=1e-2), f"Rec area was {area_rec}, expected 1.0"

# --- HELPER FUNCTION ---
def get_i00_i01_intensities(spectrum_intensity):
    """
    Finds the 0-0 and 0-1 peak intensities in a spectrum.
    """
    peaks, properties = find_peaks(
        spectrum_intensity, 
        height=0.1 * np.max(spectrum_intensity),
        distance=5
    )
    
    if len(peaks) < 2:
        return None, None
    
    # Sort the peaks by energy/index
    peak_indices = sorted(peaks) 
    I_00 = spectrum_intensity[peak_indices[0]] # Intensity of the first peak (0-0)
    I_01 = spectrum_intensity[peak_indices[1]] # Intensity of the second peak (0-1)
    
    return I_00, I_01
