#!/bin/bash
# setup.sh


# 1. Create venv if it doesn't exist
if [ ! -d "venv" ]; then
    echo "Creating virtual environment..."
    python3 -m venv venv
fi

# 2. Activate the environment
source venv/bin/activate

# 3. Upgrade pip and install requirements
echo "Installing dependencies..."
pip install --upgrade pip
pip install -r requirements.txt
pip install -r requirements_dev.txt

# 4. Install pre-commit hooks
pre-commit install
pre-commit install

echo "-----------------------------------------------"
echo "Setup complete. Virtual environment is ready."
echo "Note: Ensure you use 'source venv/bin/activate'"
echo "before running tests or starting work."
echo "-----------------------------------------------"
