# tests/test_state.py

def test_import_state():
    from MLJ.physics.state import State
    assert hasattr(State, "__doc__")


def test_default_state():
    from MLJ.physics.state import State

    default_state = State()
    assert default_state.energy == 0.0
    assert default_state.name == "Ground State"
    assert default_state.vib_spacing == 0.15

def test_custom_state():
    from MLJ.physics.state import State

    custom_state = State(name="LE",
                        energy=1.5,
                        number_of_vibronic_modes=12,
                        vib_spacing=0.2,
                        disorder_sigma=0.1,
                        disorder_number_of_states=5,
                        disorder_integration_cut_off=2.5,
                        )

    assert custom_state.energy == 1.5
    assert custom_state.name == "LE"
    assert custom_state.number_of_vibronic_modes == 12
    assert custom_state.vib_spacing == 0.2
    assert custom_state.disorder_sigma == 0.1
    assert custom_state.disorder_integration_cut_off == 2.5
    assert custom_state.disorder_number_of_states == 5

def test_edge_case_sigma_zero():
    """Test the edge case of no disorder via sigma=0."""
    from MLJ.physics.state import State

    no_disorder_state = State(name="LE",
                              energy=1.5,
                              number_of_vibronic_modes=12,
                              vib_spacing=0.2,
                              disorder_sigma=0,
                              disorder_number_of_states=5,
                              disorder_integration_cut_off=2.5,
                              )
    
    assert no_disorder_state.disorder_sigma == 0
    assert no_disorder_state.disorder_number_of_states == 1

def test_edge_case_one_state():
    """Test the edge case of no disorder via number of states = 1."""
    from MLJ.physics.state import State

    no_disorder_state = State(name="LE",
                              energy=1.5,
                              number_of_vibronic_modes=12,
                              vib_spacing=0.2,
                              disorder_sigma=0.1,
                              disorder_number_of_states=1,
                              disorder_integration_cut_off=2.5,
                              )
    
    assert no_disorder_state.disorder_sigma == 0
    assert no_disorder_state.disorder_number_of_states == 1