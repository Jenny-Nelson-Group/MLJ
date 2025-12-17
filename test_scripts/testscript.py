from MLJ.physics.FCWD import fcwd
from MLJ.physics.normalisation import partition_function
from MLJ.physics.state import State
from MLJ.physics.rates import Rates
from MLJ.physics.transition import Transition, TransitionType
import MLJ.physics.constants as const
import numpy as np
import matplotlib.pyplot as plt
from MLJ.physics.state import gaussian_distribution, gaussian_distribution_nonnorm

temperatures = temperatures=np.array([100,150,200,250,300,400])
res=200
photon_energies = np.linspace(0.5, 2.5, res)


GS = State(number_of_vibronic_modes=15)
LE = State(name="Local Exciton",
           index=1,
           energy=1.35,
           number_of_vibronic_modes=5,
           vib_spacing=0.15,
           disorder_sigma=0.1,
           disorder_number_of_states=5,
           disorder_integration_cut_off=5,
           disorder_distribution=gaussian_distribution_nonnorm,
           )

transition =  Transition(state_low_energy=GS,
                         state_high_energy=LE,
                         lambda_inner=0.05,
                         lambda_outer=0.075,
                         dipole_moment=1
                         )
print(transition)

Zrec = partition_function(transition, temperatures=temperatures)

#print(Zrec)


fcwd_values_abs = fcwd(photon_energies, transition.set_type_absorption(), temperatures=temperatures)
fcwd_values_rec = fcwd(photon_energies, transition.set_type_recombination(), temperatures=temperatures)

x= photon_energies
y_abs = fcwd_values_abs[:,0,0]
y_rec = fcwd_values_rec[:,0,0]


print("-----------------------------------------------------")

rates = Rates(photon_energies=photon_energies, transition=transition, temperatures=temperatures, photon_density=1)
rates.calculate_rates()

print("rad", rates.k_radiative_total)
print("nonrad", rates.k_non_radiative_total)
plt.plot(photon_energies, rates.k_radiative_spectral)
plt.show()
