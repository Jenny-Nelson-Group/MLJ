import MLJ as mlj
import matplotlib.pyplot as plt
import numpy as np
from MLJ.physics.config import config

config.temperatures_K = np.array([100, 300])
gs = mlj.State(name="S0", energy=0.0)
le = mlj.State(name="LE", energy=1.5)
ct = mlj.State(name="CT", energy=1.3)
trans_LE = mlj.Transition(le, gs)
trans_CT = mlj.Transition(ct, gs)
trans_LECT = mlj.Transition(le, ct, k_transfer=[10e5, 10e10])
system = mlj.StateSystem([trans_LE, trans_CT, trans_LECT])

print(system.transfers)
print(trans_CT.name)
print(trans_LE.name)
print(trans_LECT.name)

plt.plot(system.photon_energies, system.emission_photoluminescence)
plt.plot(system.photon_energies, system.absorption)
plt.show()
