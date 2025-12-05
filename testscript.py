from MLJ.physics.FCWD import fcwd
from MLJ.physics.normalisation import partition_function
from MLJ.physics.state import State
from MLJ.physics.transition import Transition, TransitionType
import numpy as np
import matplotlib.pyplot as plt

GS = State(number_of_vibronic_modes=15)
LE = State(name="Local Exciton", index=1, energy=1.5, number_of_vibronic_modes=5)

transition =  Transition(state_low_energy=GS, state_high_energy=LE, lambda_inner=0.8, lambda_outer=0.01)
print(transition)

Zrec = partition_function(transition)

print(Zrec)

res=200
energies = np.linspace(0.8, 1.8, res)
fcwd_values_abs = fcwd(energies, transition.set_type_absorption())
fcwd_values_rec = fcwd(energies, transition.set_type_recombination())

x= energies
y_abs = fcwd_values_abs[:,0,0]
y_rec = fcwd_values_rec[:,0,0]

plt.plot(x, y_abs)
plt.plot(x, y_rec)
plt.show()