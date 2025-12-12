# src/MLJ/config.py
#####################################################################################
# MLJ Package
#
# Default Parameters Module: Module for all default parametes used throughout the package
# Author: Jolanda S Müller, Tim Rein,  Imperial College London
# Copyright (c) 2025, Imperial College London, BSD 3-Clause License
# Date: November 2025
#####################################################################################

import numpy as np
from dataclasses import dataclass, field

@dataclass
class _Config:
    """
    Dataclass storing config paramters used by the whole package.
    These can be changed at runtime and changes will remain valid for the rest
    of the session. After restarting, values will default back to the ones
    that are defined in this file.
    """
    temperatures_K: np.ndarray = field(default_factory=lambda: np.array([300.0]))


config = _Config() # public singleton (making sure that only one instance of the config is used across the package)

__all__ = ["config"]