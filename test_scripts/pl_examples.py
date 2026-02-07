import MLJ as mlj
import matplotlib.pyplot as plt
import numpy as np
from MLJ.physics.config import config

config.temperatures_K = np.array([100,300])
le = mlj.State(energy=1.5)
ct = mlj.State(energy=1.3)
trans_LE = mlj.Transition(le)
trans_CT = mlj.Transition(ct)
trans_LECT = mlj.Transition(le,ct, k_transfer=[10e5,10e10])
system = mlj.StateSystem([trans_LE, trans_CT, trans_LECT])

print(system.transfers)

plt.plot(system.photon_energies, system.emission_photoluminescence)
plt.plot(system.photon_energies, system.absorption)
plt.show()
