import pytest
import numpy as np
from MLJ.physics.state import State
from MLJ.physics.config import config
import MLJ.physics.constants as const
from MLJ.physics.population_dark import dark_population, states_dark_population

def create_test_state(name="LE", energy=1.8):
    """
    Helper to generate states using the real MLJ.physics.state.State class.
    """
    return State(
        name=name,
        energy=energy,
        number_of_vibronic_modes=12,
        vib_spacing=0.2,
        disorder_sigma=0.1,
        disorder_number_of_states=5,
        disorder_integration_cut_off=2.5,
    )

def test_single_le_state_population():
    """Test Case 1: Single LE State."""
    # Ensure a fixed temperature for reproducibility
    config.temperatures_K = 300.0
    le_state = create_test_state(name="LE", energy=1.5)
    
    # Run
    pop = states_dark_population([le_state])
    
    # Assertions
    assert isinstance(pop, float), "Should return a single float for one state"
    assert pop.size > 0, "Population array should not be empty"
    assert np.all(pop >= 0), "Population cannot be negative"

def test_le_ct_two_state_weighting():
    """Test Case 2: LE and CT states with."""
    config.temperatures_K = 300.0
    
    # LE at 1.5eV, CT at 1.8eV (CT is higher in energy, so lower population)
    le_state = create_test_state(name="LE", energy=1.8)
    ct_state = create_test_state(name="CT", energy=1.5)
    
    ratio_ct_exciton = 2.0
    # Following your logic: LE has weight 1.0, CT has weight 1/ratio_ct_exciton
    custom_weights = [1.0, 1.0/ratio_ct_exciton]
    
    # Run
    results = states_dark_population([le_state, ct_state], weights=custom_weights)
    
    # Assertions
    assert isinstance(results, list), "Should return a list for multiple states"
    assert len(results) == 2
    
    pop_le = results[0]
    pop_ct = results[1]
    
    # Verify the weight was applied to CT
    # We compare a 'pure' calculation to the weighted one
    pure_ct_pop = dark_population(ct_state)
    assert np.allclose(pop_ct, (1.0/ratio_ct_exciton) * pure_ct_pop), "CT weight 1/rcte not applied correctly"
    
    # Verify the LE weight remained 1.0
    pure_le_pop = dark_population(le_state)
    assert np.allclose(pop_le, pure_le_pop), "LE weight should be 1.0"

def test_default_1_over_n_weighting():
    """Verify that if no weights are passed, 1/N is used."""
    states = [create_test_state("S1", 1.4), create_test_state("CT", 1.0)]
    
    results = states_dark_population(states)
    
    # Each result should be half of the standard dark_population
    expected_val = 0.5 * dark_population(states[0])
    assert np.allclose(results[0], expected_val)