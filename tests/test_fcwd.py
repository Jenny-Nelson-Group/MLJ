# tests/test_fcwd.py
import numpy as np
import pytest
from scipy.signal import find_peaks

from MLJ.physics.FCWD import fcwd
from MLJ.physics.state import State
from MLJ.physics.transition import Transition, TransitionType
from MLJ.physics.config import config 

# --- FIXTURES ---
# Set-Up an exemplarly state for each test
@pytest.fixture
def sample_transition():
    """Provides a standard Transition object for testing."""
    vib_spacing = 0.15 # [eV]
    E_gap = 1.5       # [eV] energy gap between GS and LE
    
    # Initialize states with no disorder
    gs = State(name='GS', vib_spacing=vib_spacing,disorder_number_of_states=1)
    le = State(name='LE', energy=E_gap, vib_spacing=vib_spacing,disorder_number_of_states=1)
    
    # Initialize Transition
    trans = Transition(
        state_low_energy=gs,
        state_high_energy=le,
        transition_type=TransitionType.ABSORPTION,
        lambda_inner=0.15, # Results in S = 1.0
        lambda_outer=0.05
    )
    return trans

# --- TESTS ---
# 1. Test output shape 
# 2. Test temperature dependent broadening
# 3. Test functionality when considering no vibrational spacing 
# 4. Test that recombination peak is always less or equal in energy compared to emission
# 5. Test that peak intensity ratio of the first two transition equals Huang-Rhys Factor

def test_fcwd_output_shape(sample_transition):
    """Verifies that the output array has the correct (Energy, Disorder, Temp) dimensions."""
    n_energies = 50
    n_temps = 2
    
    photon_energies = np.linspace(1.0, 2.0, n_energies)
    temps = np.array([100.0, 300.0])
    
    res = fcwd(photon_energies, sample_transition, temperatures=temps)
    
    # Expected shape: (50, 1, 2); 1 because no disorder is used
    assert res.shape == (n_energies, 1, n_temps)

def test_temperature_broadening(sample_transition):
    """Verifies that high temperature lowers the peak intensity (area stays same, width increases)."""
    energies = np.linspace(1.0, 2.0, 100)
    
    # Compare 50K and 500K; 0 and 0 index since no disorder and only single temperature is considered
    res_cold = fcwd(energies, sample_transition, temperatures=np.array([50.0]))[:, 0, 0]
    res_hot = fcwd(energies, sample_transition, temperatures=np.array([500.0]))[:, 0, 0]
    
    assert np.max(res_cold) > np.max(res_hot), "Cold peak should be sharper and higher than hot peak"

def test_zero_vibrational_levels(sample_transition):
    """
    Simpler test: Checks if the code works correctly when 
    number_of_vibronic_modes is set to 0 (0-0 transition only).
    """
    # 1. Force both states to only have the ground vibrational level (v=0)
    sample_transition.state_low_energy.number_of_vibronic_modes = 0
    sample_transition.state_high_energy.number_of_vibronic_modes = 0
    
    # 2. Define energies around the expected 0-0 position (E_gap + lambda_outer)
    E_00 = 1.5 + sample_transition.lambda_outer
    photon_energies = np.linspace(E_00 - 0.5, E_00 + 0.5, 100)

    # 3. Calculate FCWD
    # The output should still be a 3D array: (200, 1, 1)
    res = fcwd(photon_energies, sample_transition)
    y_values = res[:, 0, 0]

    # 4. Assertions
    assert not np.isnan(y_values).any(), "FCWD returned NaN for zero vib-levels" #Function returns value
    assert res.all() >= 0, "FCWD values must be non-negative" 
    peak_idx = np.argmax(y_values)
    assert np.isclose(photon_energies[peak_idx], E_00, atol=0.05) #Peak should be centered at 0-0 position

def test_abs_vs_rec_peak_shift(sample_transition):
    """Verifies that Recombination occurs at lower energy than Absorption."""
    energies = np.linspace(0.5, 2.5, 100)
    
    # Case 1: Absorption
    sample_transition.transition_type = TransitionType.ABSORPTION
    res_abs = fcwd(energies, sample_transition)[:, 0, 0]
    peak_abs = energies[np.argmax(res_abs)]
    
    # Case 2: Recombination
    sample_transition.transition_type = TransitionType.RECOMBINATION
    res_rec = fcwd(energies, sample_transition)[:, 0, 0]
    peak_rec = energies[np.argmax(res_rec)]
    
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
    vib_spacing = 0.15

    # 2. Setup States and Transition
    config.temperatures_K = np.array([40.0]) #Set low-temperature for more pronounced peaks
    E_gap = 1.5  # LE is 1.5 eV above ground
    ground_state = State(name='GS', vib_spacing=vib_spacing) 
    excited_state = State(name="LE", energy=E_gap,vib_spacing=vib_spacing) 

    #Set inner reorganisation energy for the transition that uniquely determines the Huang-Rhys
    S_target = lambda_inner/vib_spacing

    abs_trans = Transition(
        state_low_energy=ground_state, 
        state_high_energy=excited_state, 
        transition_type=TransitionType.ABSORPTION,
        lambda_inner=lambda_inner
    )

    # Sanity Check: Verify the Transition calculates the correct S
    assert np.isclose(abs_trans.huang_rhys, S_target), \
        f"Internal calculation mismatch: Expected S={S_target}, Got {abs_trans.huang_rhys}"
    
    # 3. Calculate FCWD Values (Spectrum)
    res = 500  
    
    # Range: Start slightly below E_gap (1.5 eV) to E_gap + 3*w_ev
    photon_energies = np.linspace(E_gap - 0.5, E_gap + 5 * vib_spacing, res)

    # Use no disorder and only single temperature -> 2x(index 0) for the most accurate I(0-1)/I(0-0) ≈ S ratio
    fcwd_values_abs = fcwd(photon_energies, abs_trans)
    y_abs = fcwd_values_abs[:, 0, 0] 

    # 4. Extract Intensities
    I_00, I_01 = get_i00_i01_intensities(y_abs)

    # Handle peak finding failure
    if I_00 is None or I_01 is None:
        pytest.fail(f"Could not resolve 0-0 and 0-1 peaks for S={S_target}. Check resolution or range.")
        
    # 5. Calculate the ratio
    S_calculated = I_01 / I_00

    # 6. Assertion
    # Use a reasonable tolerance for numerical and peak finding errors.
    assert np.isclose(S_calculated, S_target, rtol=0.05), \
        f"FCWD ratio I(0-1)/I(0-0) ({S_calculated:.3f}) does not match input S factor ({S_target:.3f})."

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