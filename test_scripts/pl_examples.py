import MLJ as mlj
from MLJ.helpers.plotting import plot_PL, plot_EL
from MLJ.physics.config import config
import matplotlib.pyplot as plt
import numpy as np

n_temps = 10
config.temperatures_K = np.linspace(50, 350, n_temps)
config.photon_energies = np.linspace(0.8, 2.0, 300)
config.photon_density = 1e10
k_LECT = np.ones(n_temps)
gs = mlj.State(name="S0", energy=0.0)
le = mlj.State(name="LE", energy=1.5, disorder_sigma=0.02)
ct = mlj.State(name="CT", energy=1.35, disorder_sigma=0.02)
trans_LE = mlj.Transition(le, gs, oscillator_strength=2.5, lambda_inner=0.05, lambda_outer=0.05)
trans_CT = mlj.Transition(ct, gs, oscillator_strength=1e-4, lambda_inner=0.05, lambda_outer=0.05)
trans_LECT = mlj.Transition(le, ct, k_transfer=k_LECT)
system_1 = mlj.StateSystem([trans_LE], injection_current=10)
system_2 = mlj.StateSystem([trans_LE, trans_CT, trans_LECT])

print(system_2.transfers)
print(trans_CT.name)
print(trans_LE.name)
print(trans_LECT.name)

rates_CT = system_2.rates[0]
rates_LE = system_2.rates[1]

print(f"Non-radiative rate = {rates_LE.rate_non_radiative_total}")
print(f"Radiative rate = {rates_LE.rate_radiative_total}")
print(f"Total Recombination = {rates_LE.rate_recombination_total}")


print(system_1.populations_light)
print(system_1.populations_dark)
print(system_1.populations_injection)

# plot_PL(system_1)
plot_EL(system_2, normalise="all")
plot_PL(system_2, normalise="all")
plt.show()
