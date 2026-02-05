from MLJ.physics.transition import Transition
from MLJ.physics.spectral_response import emission
from MLJ.physics.rates import Rates
from MLJ.physics.config import config
from MLJ.physics.population_dark import states_dark_population
from MLJ.physics.population_light import states_light_population, TransitionMatrix
from MLJ.physics.generation import excited_state_generation
from MLJ.helpers.caching import read_only_cached_property, ReactiveModule
import numpy as np


class StateSystem(ReactiveModule):
    def __init__(
        self,
        transition: Transition,
        photon_energies: np.ndarray = None,
        temperatures: np.ndarray = None,
        photon_density: float = None,
    ) -> None:
        self.transition = transition
        self.photon_energies = (
            config.photon_energies if photon_energies is None else photon_energies
        )
        self.temperatures = (
            config.temperatures_K if temperatures is None else temperatures
        )
        self.photon_density = (
            config.photon_density if photon_density is None else photon_density
        )
        self.start_caching()

    @read_only_cached_property
    def rates(self):
        return Rates(self.transition, self.photon_energies, self.temperatures)

    @read_only_cached_property
    def transition_matrix(self):
        return TransitionMatrix([self.rates.rate_recombination_total])

    @read_only_cached_property
    def generation(self):
        return [
            excited_state_generation(
                self.rates.rate_absorption_spectral, self.photon_energies
            )
        ]

    @read_only_cached_property
    def populations_dark(self):
        return states_dark_population(
            self.transition.state_high_energy, self.temperatures
        )

    @read_only_cached_property
    def population_light(self):
        return states_light_population(
            self.transition_matrix, self.populations_dark, self.generation
        )

    @read_only_cached_property
    def emission_photoluminescence(self):
        return emission(
            populations=[self.population_light],
            recombination_rates=[self.rates.rate_radiative_spectral],
        )

    @read_only_cached_property
    def absorption(self):
        # ToDo: placeholder for actual absorption spectrum
        return self.rates.rate_absorption_spectral
