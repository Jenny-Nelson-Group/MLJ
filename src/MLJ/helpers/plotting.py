import seaborn as sns
import pandas as pd
import matplotlib.pyplot as plt
from MLJ.physics.simulate_system import StateSystem


def plot_PL(system: StateSystem):
    df = pd.DataFrame(
        system.emission_photoluminescence,
        index=system.photon_energies,
        columns=system.temperatures,
    ).melt(ignore_index=False, var_name="Temperature (K)", value_name="PL Intensity")

    # Visual Fluff
    sns.set_style("ticks")  # Clean, professional look
    plt.figure(figsize=(9, 6))

    plot = sns.lineplot(
        data=df,
        x=df.index,
        y="PL Intensity",
        hue="Temperature (K)",
        palette="coolwarm",
        linewidth=2,
    )

    # Aesthetic touches
    plt.title(
        f"Photoluminescence Spectrum: {len(system.sorted_states)} State System",
        fontsize=14,
        pad=15,
    )
    plt.xlabel("Photon Energy (eV)", fontweight="bold")
    plt.ylabel("Intensity (arb. u.)", fontweight="bold")
    plt.grid(True, linestyle="--", alpha=0.6)

    # Move legend outside so it doesn't block the peaks
    plt.legend(bbox_to_anchor=(1.05, 1), loc="upper left", title="Temperature")
    plt.tight_layout()

    return plot
