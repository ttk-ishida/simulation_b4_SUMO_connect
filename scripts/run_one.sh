#!/usr/bin/env bash
set -euo pipefail

# Run a single experiment.
# Usage:
#   ./scripts/run_one.sh configs/base.yaml configs/scenario/x_intersection_2lane.yaml configs/scheduler/pred_aware.yaml
#
# Note:
#  - This assumes you run from repo root.
#  - If you haven't installed the package, use: PYTHONPATH=src python -m vecsim.cli ...
BASE="${1:-configs/base.yaml}"
SCEN="${2:-configs/scenario/x_intersection_2lane.yaml}"
SCHED="${3:-configs/scheduler/fifo.yaml}"

PYTHONPATH=src python -m vecsim.cli run --config "$BASE" --scenario "$SCEN" --scheduler "$SCHED"
