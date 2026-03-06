# MLJ
Marcus-Levich-Jortner Formalism to calculate rate constants in organic semiconductors.

## Background
To model the physics within organic semiconductors it is important to understand at which rate the system transitions between different states.
Within pertubation theory the transition rate $k_{i \to f}$  between an intial state i and final state f is commonly calculated using Fermi's Golden Rule:

$$k_{i \to f} = \frac{2\pi}{\hbar} |\langle f | \hat{H}' | i \rangle|^2 \rho(E_f)$$

Where:
* **$k_{i \to f}$**: Transition probability per unit time.
* **$\langle f | \hat{H}' | i \rangle$**: The matrix element of the perturbation $\hat{H}'$ between final and initial states.
* **$\rho(E_f)$**: The density of final states.

If we now apply th Born-Oppenheimer approximation we can factorise the electronic and nuclear overlap such that the rate expression becomes:
$$k_{i \to f} = \frac{2\pi}{\hbar} \frac{1}{Z_{norm}} \int_{E_{lower}}^{E_{upper}} \left| \langle \psi_{el,f} | \hat{H}' | \psi_{el,i} \rangle \right|^2 \cdot FCWD(E) g(E) dE$$

Where the **Franck-Condon Weighted Density of States** is defined as:
$$FCWD(E) = \sum_{m,n} P_m^{(i)} \left| \langle \psi_{vib,n}^{(f)} | \psi_{vib,m}^{(i)} \rangle \right|^2 \delta(E - \Delta E_{mn})$$

#### Parameter Definitions:
* **$\langle \psi_{el,f} | \hat{H}' | \psi_{el,i} \rangle$**: The electronic coupling matrix element.
* **$P_m^{(i)}$**: The Boltzmann population of the initial vibrational state $m$.
* **$\langle \psi_{vib,n}^{(f)} | \psi_{vib,m}^{(i)} \rangle$**: The overlap integral between vibrational wavefunctions (Franck-Condon factors).
* **$\delta(E - \Delta E_{mn})$**: The Dirac delta function ensuring energy conservation during the transition.

## License
This project is licensed under the BSD 3-Clause License - see the [LICENSE](LICENSE) file for details.

Work in progress
