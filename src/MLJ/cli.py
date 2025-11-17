# src/MLJ/cli.py
#####################################################################################
# MLJ Package
#
# Command Line Interface: Entry point parsing command line keywords
# Author: Jolanda S Müller, Tim Rein, Imperial College London
# Copyright (c) 2025, Imperial College London, BSD 3-Clause License
# Date: November 2025
#####################################################################################

from __future__ import annotations
from MLJ import __version__
import argparse
from pathlib import Path
from . import run  # this is from __init__

def build_parser():
    p = argparse.ArgumentParser(
        prog="MLJ",
        description="Marcus-Levich-Jortner Formalism to calculate rate constants.",
        argument_default=argparse.SUPPRESS,
    )
    p.add_argument("--example", action="store_true", help="Example Keyword has been activated.")
    p.add_argument("--version",  action="version", version=f"%(prog)s v{__version__}")
    return p

def parse(argv=None):
    p = build_parser()
    args = p.parse_args(argv)
    user_config = vars(args)
    return user_config

def main(argv=None):
    """Read and parse command line input keywords, and run accordingly."""

    user_config = parse(argv)

    example = True if user_config.get("example") else False
    user_config.pop("example", None)

    # run in appropriate mode
    run(example=example, **user_config)