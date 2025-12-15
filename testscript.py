from MLJ.physics.FCWD import fcwd
from MLJ.physics.normalisation import partition_function
from MLJ.physics.state import State
from MLJ.physics.rates import Rates
from MLJ.physics.transition import Transition, TransitionType
import MLJ.physics.constants as const
import numpy as np
import matplotlib.pyplot as plt
from MLJ.physics.basics import integral


temperatures = temperatures=np.array([100,300])
res=200
photon_energies = np.linspace(0.8, 1.8, res)


GS = State(number_of_vibronic_modes=15)
LE = State(name="Local Exciton", index=1, energy=1.5, number_of_vibronic_modes=5)

transition =  Transition(state_low_energy=GS, state_high_energy=LE, lambda_inner=0.8, lambda_outer=0.01)
print(transition)

Zrec = partition_function(transition, temperatures=temperatures)

#print(Zrec)


fcwd_values_abs = fcwd(photon_energies, transition.set_type_absorption(), temperatures=temperatures)
fcwd_values_rec = fcwd(photon_energies, transition.set_type_recombination(), temperatures=temperatures)

x= photon_energies
y_abs = fcwd_values_abs[:,0,0]
y_rec = fcwd_values_rec[:,0,0]

#plt.plot(x, y_abs)
#plt.plot(x, y_rec)
#plt.show()
import MLJ.physics.coupling as cpl

_prefactor_abs_rec = 1/(3*np.pi*const.VACUUM_PERMITTIVITY_SI*const.REDUCED_PLANCK_CONSTANT_EVS**4)
transition.set_type_recombination()
fcwd_rec = fcwd(photon_energies=photon_energies, transition=transition, temperatures=temperatures)
normalisation = partition_function(transition=transition,temperatures=temperatures)
rad_coupling = cpl.coupling_strength_rad(transition)
prefactor_2 = (photon_energies/const.SPEED_OF_LIGHT)**3
print(rad_coupling)

weights3d=transition.disorder_weights.reshape(1,21,1)
print(f"fcwd: {fcwd_rec.shape}")
print(f"weights: {weights3d.reshape(1,21,1).shape}")

weights_3d       = np.broadcast_to(weights3d, fcwd_rec.shape)


print(weights_3d.shape)
#print(len(integrand))

integrand = rad_coupling * fcwd_rec * weights_3d

print(integrand.shape)

integr =  integral(y=integrand, x=transition.gibbs_energy_grid,axis=1)
print("intg:",integr.shape)
prefactor_2_3d = np.broadcast_to(prefactor_2.reshape(200,1), (len(photon_energies),len(temperatures)))
print("pref:", prefactor_2_3d.shape)
norm3d = normalisation.reshape(1,2)
normalisation_3d = np.broadcast_to(norm3d,  (len(photon_energies),len(temperatures)))
print("norm", normalisation_3d.shape)

k_radiative = 1/normalisation_3d * _prefactor_abs_rec * prefactor_2_3d  # array of length(photon_energies)
print(k_radiative.shape)

print("-----------------------------------------------------")

rates = Rates(photon_energies=photon_energies, transition=transition, temperatures=temperatures, photon_density=1)
rates.calculate_rates()

print("rad", rates.k_radiative_total)
print("nonrad", rates.k_non_radiative_total)
plt.plot(photon_energies, rates.k_radiative_spectral)
plt.show()
