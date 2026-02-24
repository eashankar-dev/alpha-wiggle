#!/usr/bin/env bash
# Simple wrapper to execute the core pipeline steps using the project venv.
# Usage: ./scripts/run_pipeline.sh [extra args passed to export_factor_inputs.py]

set -euo pipefail

# ensure the environment exists
if [ ! -d ".venv" ]; then
    echo "virtualenv .venv not found; run ./scripts/setup_env.sh first" >&2
    exit 1
fi

# use the venv Python explicitly to avoid any PATH/activation issues
PY=".venv/bin/python"

# run the two main stages as package modules
# 1. export inputs
"$PY" -m price_momentum.export_factor_inputs "$@"
# 2. generate visuals (uses same defaults)
"$PY" -m price_momentum.generate_visuals

echo "pipeline finished; outputs in output/ and output_adj/ visual subfolders"