from MLJ.physics.state import State
from MLJ.physics.transition import Transition
from MLJ.physics.spectral_response import emission
from MLJ.physics.rates import Rates
from MLJ.physics.population_dark import states_dark_population
from MLJ.physics.population_light import states_light_population, TransitionMatrix
from MLJ.physics.generation import excited_state_generation
import numpy as np


import matplotlib.pyplot as plt

# settings
state_LE = State(energy=1.5)
transition_LE = Transition(state_LE)
photon_energies = np.linspace(0.8,2.5,500)
temperatures = np.array([100,200,300])

rates_LE = Rates(transition_LE, photon_energies,temperatures)

transition_matrix = TransitionMatrix([rates_LE.rate_recombination_total])
gen_LE = excited_state_generation(rates_LE.rate_absorption_spectral, photon_energies)
gens = [gen_LE]
pops_dark = states_dark_population(state_LE, temperatures)
pop_light = states_light_population(transition_matrix, pops_dark, gens)


pl_spectra = emission(
    populations = [pop_light],
    recombination_rates = [rates_LE.rate_radiative_spectral]
)

plt.plot(photon_energies, pl_spectra)
plt.show()