import MLJ as mlj
import numpy as np
from MLJ.physics.state import gaussian_distribution

mlj.config.temperatures_K = np.linspace(100, 300, 1)

for energy in [1.2, 1.3, 1.4, 1.5, 1.6]:
    # define your all your states like this:
    ground_state = mlj.State()
    state = mlj.State(
        name="name of your state",  # this is not obligatory (it's only for yourself)
        energy=energy,  # energy of your state (with respect to 0 ground)
        number_of_vibronic_modes=5,  # number of vibronic modes considered
        vib_spacing=0.15,  # spacing between vibronic states
        disorder_sigma=0.002,  # set to 0 for no disorder
        disorder_number_of_states=21,  # set to 1 for no disorder
        disorder_distribution=gaussian_distribution,  # irrelevant if no disorder
    )

    # define each relevant transitions between states
    transition = mlj.Transition(
        state_low_energy=ground_state,  # lower energy state
        state_high_energy=state,  # higher energy state
        lambda_inner=0.1,  # inner reorganisation energy
        lambda_outer=0.1,  # outer reorganisation energy
        oscillator_strength=2.56,  # oscillator strength
        static_dipole_moment=3 * 3.33e-30 / 1.6e-19,  # static dipole moment
    )

    system = mlj.StateSystem(transition)

    mlj.plot_PL(system)  # label=f"S1 = {energy} eV", normalise="each")

mlj.show()
