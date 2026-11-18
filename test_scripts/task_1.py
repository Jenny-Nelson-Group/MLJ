import MLJ as mlj
import numpy as np

mlj.config.temperatures_K = np.linspace(100, 300, 1)
for energy in [1.2, 1.3, 1.4, 1.5, 1.6]:
    excited_state = mlj.State(name="S1", energy=energy)
    system = mlj.StateSystem(mlj.Transition(excited_state))
    mlj.plot_PL(system)  # label=f"S1 = {energy} eV", normalise="each")


mlj.show()
