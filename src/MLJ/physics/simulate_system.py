from MLJ.physics.transition import Transition
from MLJ.physics.spectral_response import emission
from MLJ.physics.rates import Rates
from MLJ.physics.basics import boltzmann
from MLJ.physics.config import config
from MLJ.physics.population_dark import states_dark_population
from MLJ.physics.population_light import states_light_population, TransitionMatrix
from MLJ.physics.generation import excited_state_generation
from MLJ.helpers.caching import read_only_cached_property, ReactiveModule
from typing import Sequence, Tuple, Dict
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
            photon_energies if photon_energies is not None else config.photon_energies
        )
        self.temperatures = (
            temperatures if temperatures is not None else config.temperatures_K
        )
        self.photon_density = photon_density or config.photon_density
        self.n_transitions = len(self.transitions)
        self._system_data
        self.start_caching()

    @read_only_cached_property
    def sorted_states(self):
        return Transition.assign_indices(self.transitions)[1:]

    @read_only_cached_property
    def _system_data(self) -> Tuple[np.ndarray, Dict]:
        """Internal factory cache to ensure build_rate_system runs exactly once."""
        n_states = len(self.sorted_states)
        rates = np.full(n_states, None, dtype=object)
        transfer_dict = {}
        for trans in self.transitions:
            high_idx, low_idx = trans.index
            if low_idx == 0:
                # transitions involving the ground state
                rates_idx = Rates(trans, self.photon_energies, self.temperatures)
                rates[high_idx - 1] = rates_idx
            else:
                if trans.k_transfer is None:
                    k_down = np.zeros_like(self.temperatures)
                else:
                    k_down = np.atleast_1d(trans.k_transfer)

                if k_down.shape != self.temperatures.shape:
                    raise ValueError(
                        f"Inconsistent shapes: in transition({trans.state_high_energy.name}"
                        f"->{trans.state_low_energy.name}), k_transfer has shape {k_down.shape}"
                        f", but temperatures have shape {self.temperatures.shape}."
                    )
                k_up = k_down * boltzmann(trans.mean_gibbs_energy, self.temperatures)
                transfer_dict[(low_idx, high_idx)] = k_down # downhill: high -> low
                transfer_dict[(high_idx, low_idx)] = k_up   # uphill: low -> high

        return rates, transfer_dict

    @property
    def rates(self):
        """Accesses the sorted rates (first part of the cached system data)."""
        return self._system_data[0]

    @property
    def transfers(self):
        """Accesses the transfer dict (second part of the cached system data)."""
        return self._system_data[1]

    @read_only_cached_property
    def transition_matrix(self):
        k_ground_total = [r.rate_recombination_total for r in self.rates]
        return TransitionMatrix(k_ground_total, self.transfers)

    @read_only_cached_property
    def generation(self):
        # (n_states, n_photon_energies, n_temps)
        absorption_rates = [r.rate_absorption_spectral for r in self.rates]
        return excited_state_generation(absorption_rates, self.photon_energies)

    @read_only_cached_property
    def populations_dark(self):
        """Returns a NumPy array of Rates objects for each transition."""
        return states_dark_population(self.sorted_states, self.temperatures)

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
