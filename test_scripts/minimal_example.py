import MLJ as mlj

excited_state = mlj.State(name="S1", energy=1.6)
transition = mlj.Transition(excited_state)
one_state_system = mlj.StateSystem(transition)

# plot PL
mlj.plot_PL(one_state_system)
mlj.show()
