#!/bin/bash
# Set up the local Python environment and launch the MarketPulse dashboard.
# Stop immediately if environment setup, installation, or startup fails.
set -e

# Resolve paths relative to this script, regardless of the caller's directory.
cd "$(dirname "$0")"

# Create the project's virtual environment on first run, then activate it.
if [ ! -d ".venv" ]; then
    python3 -m venv .venv
fi
source .venv/bin/activate

# Install the required packages into the active virtual environment.
python -m pip install --upgrade pip
pip install -r requirements.txt

# Start the local web app; Streamlit displays its URL in the terminal.
streamlit run app.py
