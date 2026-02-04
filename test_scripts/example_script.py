import MLJ as mlj
from MLJ.physics.state import gaussian_distribution
import numpy as np
import matplotlib.pyplot as plt

# define your all your states like this:
ground_state = mlj.State()
state = mlj.State(
    name="name of your state",  # this is not obligatory (it's only for yourself)
    energy=1.35,  # energy of your state (with respect to 0 ground)
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


photon_energies = np.linspace(0.8, 2.5, 200)  # spectral region of interest

# calculate your rates
rates = mlj.Rates(
    transition=transition,
    photon_energies=photon_energies,
    temperatures=np.array([200, 300]),
)

print(f"Non-radiative rate = {rates.rate_non_radiative_total}")
print(f"Radiative rate = {rates.rate_radiative_total}")
print(f"Total Recombination = {rates.rate_recombination_total}")

plt.plot(photon_energies, rates.rate_radiative_spectral)
plt.show()
