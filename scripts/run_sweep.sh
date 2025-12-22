#!/usr/bin/env bash
set -euo pipefail

# Run a sweep defined in configs/sweep/*.yaml
# Usage:
#   ./scripts/run_sweep.sh configs/sweep/mu_sweep.yaml

SWEEP="${1:-configs/sweep/mu_sweep.yaml}"
PYTHONPATH=src python -m vecsim.cli sweep --config "$SWEEP"
