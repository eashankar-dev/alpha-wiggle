#!/usr/bin/env bash
# Create / update the Python virtual environment and install requirements.
# Usage: ./scripts/setup_env.sh

set -euo pipefail

# if workspace path contains spaces we need to quote it when activating
py="$(command -v python3)"
if [ -z "$py" ]; then
    echo "python3 not found in PATH" >&2
    exit 1
fi

# create venv if it doesn't exist
if [ ! -d ".venv" ]; then
    echo "creating virtual environment in .venv"
    "$py" -m venv .venv
else
    echo "virtual environment .venv already exists"
fi

# install dependencies using the venv executables directly
# this avoids problems with activation when the path contains spaces
.venv/bin/python -m pip install --upgrade pip
.venv/bin/python -m pip install -r requirements.txt

echo "environment ready; you can run scripts via '.venv/bin/python' or activate manually"