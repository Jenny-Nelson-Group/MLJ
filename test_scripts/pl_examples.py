import MLJ as mlj
from MLJ.helpers.plotting import plot_PL
from MLJ.physics.config import config
import matplotlib.pyplot as plt
import numpy as np

n_temps = 10
config.temperatures_K = np.linspace(50, 350, n_temps)
config.photon_energies = np.linspace(0.8, 2.0, 300)
k_LECT = np.ones(n_temps)
gs = mlj.State(name="S0", energy=0.0)
le = mlj.State(name="LE", energy=1.5, disorder_sigma=0.02)
ct = mlj.State(name="CT", energy=1.35, disorder_sigma=0.02)
trans_LE = mlj.Transition(le, gs, lambda_outer=0.05)
trans_CT = mlj.Transition(ct, gs, oscillator_strength=3)
trans_LECT = mlj.Transition(le, ct, k_transfer=k_LECT)
system_1 = mlj.StateSystem([trans_LE])
system_2 = mlj.StateSystem([trans_LE, trans_CT, trans_LECT])

print(system_2.transfers)
print(trans_CT.name)
print(trans_LE.name)
print(trans_LECT.name)

# plot_PL(system_1)
plot_PL(system_2)
plt.show()
