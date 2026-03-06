# MLJ
Marcus-Levich-Jortner formalism to calculate rate constants in organic semiconductors.
A Python implementation of the semi-classical MLJ theory for modeling charge and energy transfer in molecular systems.

---

## Key Physical Assumptions
The implementation of the Marcus-Levich-Jortner (MLJ) formalism in this library is based on the following physical framework:

* **Non-Adiabatic Perturbative Regime**: Electronic coupling ($J$) is assumed to be weak compared to the reorganisation energy ($J \ll \lambda$).
* **Harmonic Approximation**: Potential energy surfaces of initial and final states are treated as parabolas with identical curvatures (displaced harmonic oscillators).
* **Condon Approximation**: The electronic transition dipole moment (or coupling matrix element) is assumed to be independent of nuclear geometry.
* **Effective High-Frequency Mode**: All "inner-sphere" vibrations (e.g., carbon-carbon stretching) are summarized into a single effective mode ($\Omega$) with a combined Huang-Rhys factor ($S$).
* **Classical Low-Frequency Bath**: Intermolecular and solvent modes are treated classically as a single reorganization energy term ($\lambda_l$).
* **Detailed Balance**: The resulting rate equations maintain thermodynamic consistency and obey detailed balance.

---

## Theoretical Background

To model the physics within organic semiconductors, it is essential to determine the rates at which a system transitions between different electronic states.

### 1. General Transition Rate
Within perturbation theory, the transition rate $k_{i \to f}$ between an initial state $|i\rangle$ and final state $|f\rangle$ is commonly calculated using Fermi's Golden Rule:

$$k_{i \to f} = \frac{2\pi}{\hbar} |\langle f | \hat{H}' | i \rangle|^2 \rho(E_f)$$

Where:
* **$k_{i \to f}$**: Transition probability per unit time.
* **$\langle f | \hat{H}' | i \rangle$**: The matrix element of the perturbation $\hat{H}'$ between states.
* **$\rho(E_f)$**: The density of final states.

### 2. Born-Oppenheimer Approximation
By applying the Born-Oppenheimer approximation, we factorize the electronic and nuclear overlap. For a system with a distribution of electronic states $g(E)$, the rate expression becomes:

$$k_{i \to f} = \frac{2\pi}{\hbar} \frac{1}{Z_{norm}} \int_{E_{lower}}^{E_{upper}} \left| \langle \psi_{el,f} | \hat{H}' | \psi_{el,i} \rangle \right|^2 \cdot FCWD(E) g(E) dE$$

The Franck-Condon Weighted Density of States (FCWD) describes the nuclear part of the transition:

$$FCWD(E) = \sum_{m,n} P_m^{(i)} \left| \langle \psi_{vib,n}^{(f)} | \psi_{vib,m}^{(i)} \rangle \right|^2 \delta(E - \Delta E_{mn})$$

**Parameter Definitions:**
* **$P_m^{(i)}$**: Boltzmann population of the initial vibrational state $m$.
* **$\langle \psi_{vib,n}^{(f)} | \psi_{vib,m}^{(i)} \rangle$**: Overlap integral between vibrational wavefunctions (Franck-Condon factors).
* **$g(E)$**: Function describing the distribution of electronic states, with normalization $Z_{norm}$.



### 3. Marcus-Levich-Jortner (MLJ) Expression
Under the assumption of a single high-frequency quantized mode ($\Omega$) and a classical low-frequency bath ($\lambda_l$), we utilize the following analytical expression for the FCWD:

$$FCWD(E) = \frac{1}{\sqrt{4\pi\lambda_l k_B T}} \sum_{w=0}^{\infty} \sum_{t=0}^{\infty} \frac{e^{-S} S^{w-t} t!}{w!} \left[ L_t^{w-t}(S) \right]^2 \exp\left( -\frac{(E + \lambda_l + (w-t)\hbar\Omega)^2}{4\lambda_l k_B T} \right) \exp\left( -\frac{t\hbar\Omega}{k_B T} \right)$$

**Key Parameters:**
* **$\lambda_l$ / $\lambda_v$**: Low-frequency (outer) and high-frequency (inner) reorganization energies.
* **$S$**: Huang-Rhys factor ($S = \lambda_v / \hbar\Omega$), quantifying electron-vibration coupling strength.
* **$L_t^{w-t}(S)$**: Associated Laguerre polynomials arising from the vibrational overlaps.
* **$w, t$**: Summation indices for the initial and final vibrational quanta.
* **Exponentials**: Describe the energy distribution of accessible transitions and the thermal population of the vibrational states.

The general expressions provided above can be tailored to specific physical processes by modifying, for example, the interaction Hamiltonian or the thermal population of the energetic states. These adaptations allow for the precise determination of absorption, radiative emission, and non-radiative decay rates.

For a comprehensive derivation and application of these modified frameworks to organic semiconductors, refer to the following literature:
* **[J. Yan et al. (2021)](https://www.nature.com/articles/s41467-021-23975-3)**: *Nature Communications*
* **[M. Azzouzi et al. (2018)](https://link.aps.org/pdf/10.1103/PhysRevX.8.031055)**: *Physical Review X*

---

## Usage Example
Work in progress

---

## Installation Instruction
Work in progress

---
## License
This project is licensed under the BSD 3-Clause License - see the [LICENSE](LICENSE) file for details.
