from MLJ.physics.FCWD import fcwd
from MLJ.physics.state import State
from MLJ.physics.transition import Transition, TransitionType
import numpy as np
import matplotlib.pyplot as plt

GS = State()
LE = State(name="Local Exciton", index=1, energy=1.5)

exc =  Transition(state_low_energy=GS, state_high_energy=LE, transition_type=TransitionType.ABSORPTION)

print(exc)

res=200
energies = np.linspace(0.8, 1.8, res)
fcwd_values = fcwd(energies, exc)

x= energies
y=fcwd_values
y2 = y.reshape(res, 2)
plt.plot(x, y2[:, 0])
plt.plot(x, y2[:, 1])
plt.show()