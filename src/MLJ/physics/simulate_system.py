from MLJ.physics.transition import Transition
from MLJ.physics.spectral_response import emission
from MLJ.physics.rates import Rates
from MLJ.physics.config import config
from MLJ.physics.population_dark import states_dark_population
from MLJ.physics.population_light import states_light_population, TransitionMatrix
from MLJ.physics.generation import excited_state_generation
from MLJ.helpers.caching import read_only_cached_property, ReactiveModule
from typing import Sequence
import numpy as np


class StateSystem(ReactiveModule):
    def __init__(
        self,
        transitions: Sequence[Transition] | Transition,
        photon_energies: np.ndarray = None,
        temperatures: np.ndarray = None,
        photon_density: float = None,
    ) -> None:
        self.transitions = np.atleast_1d(transitions)
        self.photon_energies = (
            config.photon_energies if photon_energies is None else photon_energies
        )
        self.temperatures = (
            config.temperatures_K if temperatures is None else temperatures
        )
        self.photon_density = (
            config.photon_density if photon_density is None else photon_density
        )
        self.n_transitions = len(self.transitions)
        self.start_caching()

    @read_only_cached_property
    def rates(self):
        """Returns a NumPy array of Rates objects for each transition."""
        return np.array(
            [
                Rates(trans, self.photon_energies, self.temperatures)
                for trans in self.transitions
            ],
            dtype=object,
        )

    @read_only_cached_property
    def transition_matrix(self):
        total_recombination_rates = [r.rate_recombination_total for r in self.rates]
        return TransitionMatrix(total_recombination_rates)

    @read_only_cached_property
    def generation(self):
        # (n_states, n_photon_energies, n_temps)
        absorption_rates = [r.rate_absorption_spectral for r in self.rates]
        return excited_state_generation(absorption_rates, self.photon_energies)

    @read_only_cached_property
    def populations_dark(self):
        """Returns a NumPy array of Rates objects for each transition."""
        states = [t.state_high_energy for t in self.transitions]
        return states_dark_population(states, self.temperatures)

    @read_only_cached_property
    def populations_light(self):
        return states_light_population(
            self.transition_matrix, self.populations_dark, self.generation
        )

    @read_only_cached_property
    def emission_photoluminescence(self):
        k_rad_spectral = [r.rate_radiative_spectral for r in self.rates]
        return emission(
            populations=self.populations_light,
            recombination_rates=k_rad_spectral,
        )

    @read_only_cached_property
    def absorption(self):
        # ToDo: placeholder for actual absorption spectrum
        k_abs_spectral = np.sum(
            [r.rate_absorption_spectral for r in self.rates], axis=0
        )
        return k_abs_spectral
