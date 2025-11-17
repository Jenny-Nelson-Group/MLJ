# src/MLJ/__main__.py
#####################################################################################
# MLJ Package
#
# Package main, called when executing "python3 MLJ.py",
# forwards call directly to console line input interpreter in cli.py
# Authors: Jolanda S Müller, Tim Rein, Imperial College London
# Copyright (c) 2025, Imperial College London, BSD 3-Clause License
# Date: November 2025
#####################################################################################

from .cli import main
if __name__ == "__main__":
    main()