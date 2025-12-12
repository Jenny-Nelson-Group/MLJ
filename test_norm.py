from MLJ.physics.state import State
from MLJ.physics.transition import Transition
from MLJ.physics.transition import TransitionType
from MLJ.physics.basics import gaussian_norm, gaussian
from MLJ.physics.normalisation import partition_function
import matplotlib.pyplot as plt
import numpy as np
from MLJ.physics.state import gaussian_distribution


model_implementation = "Znorm: trapz, normgauss, cut_off=scales"
distr = gaussian_distribution

for i in [1,2,3]:
    situation = i
    Znorm = []
    match situation:
        case 1:
            test_range=np.linspace(1e-10,1,99) # sigma
        case 2:
            test_range=np.linspace(2,100,99).astype(int) # number
        case 3:
            test_range=np.linspace(0.1,5,99) # cut off

    sigma_fix = 0.2
    nr_fix = 21
    cut_off_fix = 2.5

    for x in test_range:
        ground_state = State(energy=0.0, disorder_number_of_states=3, disorder_sigma=0.1)

        match situation:
            case 1:
                excited_state = State(energy=1.5, disorder_number_of_states=nr_fix, disorder_sigma=x, disorder_integration_cut_off=cut_off_fix, disorder_distribution=distr)
            case 2:
                excited_state = State(energy=1.5, disorder_number_of_states=x, disorder_sigma=sigma_fix, disorder_integration_cut_off=cut_off_fix, disorder_distribution=distr)
            case 3:
                excited_state = State(energy=1.5, disorder_number_of_states=nr_fix, disorder_sigma=sigma_fix, disorder_integration_cut_off=x, disorder_distribution=distr)

        transition = Transition(state_low_energy=ground_state, state_high_energy=excited_state, transition_type = TransitionType.RECOMBINATION)
        Znorm_abs = partition_function(transition=transition)
        Znorm.append(Znorm_abs)


    plt.plot(test_range, Znorm)
    plt.ylabel(model_implementation)

    match situation:
        case 1:
            plt.xlabel("sigma"); plt.title(f"sigma = changes, nr of states = {nr_fix}, cut off = {cut_off_fix}, ")
            plt.ylim(0,5)
        case 2:
            plt.xlabel("number of states"); plt.title(f"sigma = {sigma_fix}, nr of states = change, cut off = {cut_off_fix}")
        case 3:
            plt.xlabel("cut off"); plt.title(f"sigma = {sigma_fix}, nr of states = {nr_fix}, cut off = changes")

    plt.show()
