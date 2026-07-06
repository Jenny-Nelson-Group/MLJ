# MLJ
Marcus-Levich-Jortner formalism to calculate rate constants in organic semiconductors.
A Python implementation of the semi-classical MLJ theory for modeling charge and energy transfer in molecular systems.

---

## Contents

- [Features](#features)
- [Installation](#installation)
- [Quick Start](#quick-start)
- [Key Physical Assumptions](#key-physical-assumptions)
- [Theoretical Background](#theoretical-background)
- [License](#license)

---

## Features

- Compute **radiative** and **non-radiative recombination** rates as well as resulting **absorption** and **emission** spectra within a single MLJ framework.
- Franck–Condon Weighted Density (FCWD) with a single effective high-frequency mode and a classical low-frequency bath.
- Support for **static energetic disorder** via a configurable distribution over electronic states.
- Supports **temperature-dependent** calculations.
- Clean object-oriented set-up (`State`, `Transition`, `StateSystem`) which allows for standalone use with built-in plotting helpers (`plot_Absorbance`, `plot_PL`).
- Can be used in further downstream tasks like fitting experimental data or transport modelling.


---

## Installation

> **Requires Python ≥ 3.11.**

```bash
git clone <https://github.com/Jenny-Nelson-Group/MLJ.git>
cd MLJ
pip install -e .
```

Installing in editable mode (`-e`) is recommended during development so changes to
the source take effect immediately.

To include the development dependencies (tests, linting, pre-commit hooks):

```bash
pip install -e ".[dev]"      # or: pip install -r requirements_dev.txt
```

---

## Quick Start

Simple demonstration of the capabilities of the library. Ready to run examples can be found in the examples folder.
```python
import MLJ as mlj
import numpy as np
import matplotlib.pyplot as plt

# 1. Global configuration: temperature and photon-energy grids
n_temps = 10
mlj.config.temperatures_K = np.linspace(50, 350, n_temps)
mlj.config.photon_energies = np.linspace(0.8, 2.0, 300)      # eV

# 2. Define electronic states (energies in eV)
gs = mlj.State(name="S0", energy=0.0)
le = mlj.State(name="LE", energy=1.5, disorder_sigma=0.02)
ct = mlj.State(name="CT", energy=1.35, disorder_sigma=0.02)

# 3. Define transitions between states
trans_LE   = mlj.Transition(le, gs, lambda_outer=0.05)
trans_CT   = mlj.Transition(ct, gs, oscillator_strength=3)
trans_LECT = mlj.Transition(le, ct, k_transfer=np.ones(n_temps))

# 4. Assemble a system and plot its optical response
system = mlj.StateSystem([trans_LE])

mlj.plot_Absorbance(system, normalise=True)
mlj.plot_PL(system, normalise=True)
plt.show()
```

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
Within perturbation theory, the rate between an initial state $|i\rangle$ and final
state $|f\rangle$ follows Fermi's Golden Rule:

$$k_{i \to f} = \frac{2\pi}{\hbar}\,\bigl|\langle f | \hat{H}' | i \rangle\bigr|^2\,\rho(E_f)$$

- $k_{i \to f}$ — transition probability per unit time
- $\langle f | \hat{H}' | i \rangle$ — matrix element of the perturbation $\hat{H}'$
- $\rho(E_f)$ — density of final states

### 2. Born-Oppenheimer Approximation
Applying the Born–Oppenheimer approximation factorises the electronic and nuclear
overlaps. For a distribution of electronic states $g(E)$ with normalisation
$Z_\mathrm{norm}$:

$$k_{i \to f} = \frac{2\pi}{\hbar}\,\frac{1}{Z_\mathrm{norm}} \int_{E_\mathrm{lower}}^{E_\mathrm{upper}} \bigl|\langle \psi_{el,f} | \hat{H}' | \psi_{el,i} \rangle\bigr|^2 \, \mathrm{FCWD}(E)\, g(E)\, \mathrm{d}E$$

The Franck–Condon Weighted Density of states describes the nuclear part:

$$\mathrm{FCWD}(E) = \sum_{m,n} P_m^{(i)} \bigl|\langle \psi_{vib,n}^{(f)} | \psi_{vib,m}^{(i)} \rangle\bigr|^2\, \delta\!\left(E - \Delta E_{mn}\right)$$

- $P_m^{(i)}$ — Boltzmann population of initial vibrational state $m$
- $\langle \psi_{vib,n}^{(f)} | \psi_{vib,m}^{(i)} \rangle$ — vibrational overlap (Franck–Condon factor)
- $g(E)$ — distribution of electronic states, normalised by $Z_\mathrm{norm}$

### 3. Marcus-Levich-Jortner (MLJ) Expression
Under the assumption of a single high-frequency quantized mode ($\Omega$) and a classical low-frequency bath ($\lambda_l$), we utilize the following analytical expression for the FCWD:

$$FCWD(E) = \frac{1}{\sqrt{4\pi\lambda_l k_B T}} \sum_{w=0}^{\infty} \sum_{t=0}^{\infty} \frac{e^{-S} S^{w-t} t!}{w!} \left[ L_t^{w-t}(S) \right]^2 \exp\left( -\frac{(E + \lambda_l + (w-t)\hbar\Omega)^2}{4\lambda_l k_B T} \right) \exp\left( -\frac{t\hbar\Omega}{k_B T} \right)$$

**Parameters**

- $\lambda_l\,/\,\lambda_v$ — low-frequency (outer) and high-frequency (inner) reorganisation energies
- $S = \lambda_v / \hbar\Omega$ — Huang–Rhys factor (electron–vibration coupling strength)
- $L_t^{\,w-t}(S)$ — associated Laguerre polynomials from the vibrational overlaps
- $w,\,t$ — final and initial vibrational quantum numbers
- The two exponentials describe, respectively, the energy distribution of accessible transitions and the thermal population of the initial vibrational states

The general expressions provided above can be tailored to specific physical processes by modifying, for example, the interaction Hamiltonian or the thermal population of the energetic states. These adaptations allow for the precise determination of absorption, radiative emission, and non-radiative decay rates.

For a comprehensive derivation and application of these modified frameworks to organic semiconductors, refer to the following literature:
* **[J. Yan et al. (2021)](https://www.nature.com/articles/s41467-021-23975-3)**: *Nature Communications*
* **[M. Azzouzi et al. (2018)](https://link.aps.org/pdf/10.1103/PhysRevX.8.031055)**: *Physical Review X*

---
## License
This project is licensed under the BSD 3-Clause License - see the [LICENSE](LICENSE) file for details.
