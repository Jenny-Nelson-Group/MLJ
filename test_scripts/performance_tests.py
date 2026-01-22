import time
from MLJ.physics.FCWD import fcwd
from MLJ.physics.normalisation import partition_function
from MLJ.physics.state import State
from MLJ.physics.rates import Rates
from MLJ.physics.transition import Transition, ProcessType
import MLJ.physics.constants as const
import numpy as np
import matplotlib.pyplot as plt
from MLJ.physics.state import gaussian_distribution, gaussian_distribution_nonnorm


temperatures = temperatures=np.linspace(100,300,100)
photon_energies = np.linspace(0.5, 2.5, 100)

GS = State(number_of_vibronic_modes=15)
LE = State(name="Local Exciton",
           index=1,
           energy=1.35,
           number_of_vibronic_modes=5,
           vib_spacing=0.15,
           disorder_sigma=0.001,
           disorder_number_of_states=21,
           disorder_integration_cut_off=5,
           disorder_distribution=gaussian_distribution_nonnorm,
           )

transition =  Transition(state_low_energy=GS,
                         state_high_energy=LE,
                         lambda_inner=0.1,
                         lambda_outer=0.1,
                         oscillator_strength=2.56,
                         static_dipole_moment=3*3.33e-30/1.6e-19
                         )

rates = Rates(transition, photon_energies, temperatures)

# Time the first (calculation) call
start = time.perf_counter()
_ = rates.rate_recombination_total
end = time.perf_counter()
print(f"First call (calculation): {end - start:.4f}s")

# Time the second (cached) call
start = time.perf_counter()
_ = rates.rate_recombination_total
end = time.perf_counter()
print(f"Second call (cache hit): {end - start:.8f}s")