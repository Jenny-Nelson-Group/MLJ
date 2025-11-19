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

    default_state = State(name="LE", energy=1.5, number_of_vibronic_modes=12,vib_spacing=0.2,sigma=0.1)
    assert default_state.energy == 1.5
    assert default_state.name == "LE"
    assert default_state.number_of_vibronic_modes == 12
    assert default_state.vib_spacing == 0.2
    assert default_state.sigma == 0.1