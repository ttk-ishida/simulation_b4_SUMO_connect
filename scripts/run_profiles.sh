#!/usr/bin/env bash
set -euo pipefail

# Run selected scheduler(s) against selected traffic profiles in order.
# Usage:
#   ./scripts/run_profiles.sh
#
# Customize by commenting out lines below.

PYTHON="${PYTHON:-/Users/tatsukiishida/simulation_parallel_b4/.venv/bin/python}"
BASE_CONFIG="${BASE_CONFIG:-configs/base.yaml}"
RUNNER="${RUNNER:-vecsim.cli}"

SCHEDULERS_LIST=$(cat <<'EOF'
configs/scheduler/fifo.yaml
#configs/scheduler/edf.yaml
#configs/scheduler/v_edf.yaml
#configs/scheduler/pred_aware.yaml
EOF
)

PROFILES_LIST=$(cat <<'EOF'
configs/scenario/alpha_change_exp3_v2/alpha_mid_high_to_low.yaml
configs/scenario/alpha_change_exp3_v2/alpha_mid_low_to_high.yaml
configs/scenario/alpha_change_exp3_v2/alpha_alternate_300.yaml
configs/scenario/alpha_change_exp3_v2/alpha_random_300.yaml
EOF
)

run_one() {
  local scen="$1"
  local sched="$2"
  echo "[RUN] scenario=$scen | scheduler=$sched"
  PYTHONPATH=src "$PYTHON" -m "$RUNNER" run --config "$BASE_CONFIG" --scenario "$scen" --scheduler "$sched"
}

while IFS= read -r sched; do
  [[ -z "$sched" || "$sched" =~ ^# ]] && continue
  while IFS= read -r scen; do
    [[ -z "$scen" || "$scen" =~ ^# ]] && continue
    run_one "$scen" "$sched"
  done <<< "$PROFILES_LIST"
done <<< "$SCHEDULERS_LIST"
