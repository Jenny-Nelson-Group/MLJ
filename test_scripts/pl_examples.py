import MLJ as mlj
import matplotlib.pyplot as plt

transition_1 = mlj.Transition(mlj.State(energy=1.5))
transition_2 = mlj.Transition(mlj.State(energy=1.3))
system = mlj.StateSystem([transition_1, transition_2])


plt.plot(system.photon_energies, system.emission_photoluminescence)
plt.plot(system.photon_energies, system.absorption)
plt.show()
