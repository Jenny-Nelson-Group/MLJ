import seaborn as sns
import pandas as pd
import matplotlib.pyplot as plt


def _base_plotter(energies, values, temperatures, ylabel, ax=None, normalise=False, **kwargs):
    """Internal function to handle data processing for plotting."""
    ax = ax or plt.gca()

    # Normalization Logic
    if normalise:
        values = values / values.max()
        ylabel = f"normalised {ylabel}"

    # Data Wrangling
    df = pd.DataFrame(
        values,
        index=energies,
        columns=temperatures,
    ).melt(ignore_index=False, var_name="Temperature (K)", value_name=ylabel)

    # Plotting
    sns.lineplot(
        data=df,
        x=df.index,
        y=ylabel,
        hue="Temperature (K)",
        palette="coolwarm",
        linewidth=2,
        ax=ax,
        **kwargs,
    )

    # Formatting
    ax.set_xlabel("Photon Energy (eV)", fontweight="bold")
    ax.set_ylabel(f"{ylabel} (arb. u.)", fontweight="bold")
    ax.grid(True, linestyle="--", alpha=0.6)
    return ax


# --- Clean Public Functions ---


def plot_PL(system, ax=None, normalise=False, **kwargs):
    return _base_plotter(
        system.photon_energies,
        system.emission_photoluminescence,
        system.temperatures,
        "PL Intensity",
        ax,
        normalise=normalise,
        **kwargs,
    )


def plot_EL(system, ax=None, normalise=False, **kwargs):
    return _base_plotter(
        system.photon_energies,
        system.emission_electroluminescence,
        system.temperatures,
        "EL Intensity",
        ax,
        normalise=normalise,
        **kwargs,
    )


def plot_Absorbance(system, ax=None, normalise=False, **kwargs):
    return _base_plotter(
        system.photon_energies,
        system.absorbance,
        system.temperatures,
        "Absorbance",
        ax,
        normalise=normalise,
        **kwargs,
    )
