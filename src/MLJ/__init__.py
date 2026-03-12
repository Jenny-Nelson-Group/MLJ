# src/MLJ/__init__.py
#####################################################################################
# MLJ Package
#
# API: Entry point when running from within a python script,
# or called by cli.py when package is run from command line.
# Author: Jolanda S Müller, Tim Rein, Imperial College London
# Copyright (c) 2025, Imperial College London, BSD 3-Clause License
# Date: November 2025
#####################################################################################

__version__ = "0.0.1"

# import and expose
from MLJ.physics.state import State
from MLJ.physics.transition import Transition
from MLJ.physics.rates import Rates
from MLJ.physics.simulate_system import StateSystem
from MLJ.helpers.plotting import plot_PL, plot_EL, plot_Absorbance
from MLJ.physics.config import config

__all__ = [
    "State",
    "Transition",
    "Rates",
    "StateSystem",
    "plot_PL",
    "plot_EL",
    "plot_Absorbance",
    "config",
]


def run(*, example=False, **kwargs):
    """Runs the program with given arguments."""

    if example:
        print("MLJ executed with example keyword.")
        print(kwargs)
    else:
        print("MLJ executed.")
        print(kwargs)
