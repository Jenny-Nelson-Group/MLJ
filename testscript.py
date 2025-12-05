from MLJ.physics.FCWD import fcwd
from MLJ.physics.state import State
from MLJ.physics.transition import Transition, TransitionType
import numpy as np
import matplotlib.pyplot as plt

GS = State(number_of_vibronic_modes=15)
LE = State(name="Local Exciton", index=1, energy=1.5, number_of_vibronic_modes=5)

abs =  Transition(state_low_energy=GS, state_high_energy=LE, transition_type=TransitionType.ABSORPTION, lambda_inner=0.8, lambda_outer=0.01)
rec =  Transition(state_low_energy=GS, state_high_energy=LE, transition_type=TransitionType.RECOMBINATION, lambda_inner=0.8, lambda_outer=0.01)

print(abs)
print(rec)

res=200
energies = np.linspace(0.8, 1.8, res)
fcwd_values_abs = fcwd(energies, abs)
fcwd_values_rec = fcwd(energies, rec)

x= energies
y_abs=fcwd_values_abs
y_abs = y_abs.reshape(res, 2)

y_rec=fcwd_values_rec
y_rec = y_rec.reshape(res, 2)

plt.plot(x, y_abs[:, 0])
#plt.plot(x, y_abs[:, 1])
plt.plot(x, y_rec[:, 0])
#plt.plot(x, y_rec[:, 1])
plt.show()