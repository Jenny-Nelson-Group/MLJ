import seaborn as sns
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib
import numpy as np

_AUTO_COLOUR_GID = "mlj_auto_colour"


def _recolour_lines(ax, cmap):
    """Spread all auto-coloured MLJ lines on `ax` evenly over the colour map."""
    lines = [line for line in ax.get_lines() if line.get_gid() == _AUTO_COLOUR_GID]
    cmap = matplotlib.colormaps.get_cmap(cmap)
    # stop at 0.9 so the last curve isn't too pale (e.g. yellow end of viridis)
    for line, colour in zip(lines, cmap(np.linspace(0, 0.9, len(lines)))):
        line.set_color(colour)


def _base_plotter(
    energies,
    values,
    temperatures,
    ylabel,
    ax=None,
    normalise=False,
    label=None,
    cmap=None,
    **kwargs,
):
    """Internal function to handle data processing for plotting."""
    ax = ax or plt.gca()

    # Normalization Logic
    if normalise == "each":
        # Normalise each temperature curve (column) to its own maximum
        values = values / values.max(axis=0)
        ylabel = f"individually normalised {ylabel}"
    elif normalise == "all":
        # Global normalisation to the absolute maximum
        values = values / values.max()
        ylabel = f"globally normalised {ylabel}"

    # Single temperature: plot one line with the next colour in the cycle,
    # so repeated calls (e.g. in a loop) give distinct, labelled curves
    if len(temperatures) == 1:
        auto_colour = "color" not in kwargs and "c" not in kwargs
        ax.plot(
            energies,
            values[:, 0],
            linewidth=2,
            label=label or f"{temperatures[0]:g} K",
            gid=_AUTO_COLOUR_GID if auto_colour else None,
            **kwargs,
        )
        # remember the colour map on the axis, so later calls without cmap= keep it
        if cmap is not None:
            ax._mlj_cmap = cmap
        _recolour_lines(ax, getattr(ax, "_mlj_cmap", "viridis"))
        ax.legend()

    else:
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
            palette=cmap or "coolwarm",
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


def plot_PL(system, ax=None, normalise=False, label=None, **kwargs):
    return _base_plotter(
        system.photon_energies,
        system.emission_photoluminescence,
        system.temperatures,
        "PL Intensity",
        ax,
        normalise=normalise,
        label=label,
        **kwargs,
    )


def plot_EL(system, ax=None, normalise=False, label=None, **kwargs):
    return _base_plotter(
        system.photon_energies,
        system.emission_electroluminescence,
        system.temperatures,
        "EL Intensity",
        ax,
        normalise=normalise,
        label=label,
        **kwargs,
    )


def plot_Absorbance(system, ax=None, normalise=False, label=None, **kwargs):
    return _base_plotter(
        system.photon_energies,
        system.absorbance,
        system.temperatures,
        "Absorbance",
        ax,
        normalise=normalise,
        label=label,
        **kwargs,
    )
