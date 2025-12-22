#!/usr/bin/env bash
set -euo pipefail

# Build SUMO network files for the included scenario.
# Usage:
#   ./scripts/make_scenario.sh

(cd scenarios/x_intersection_2lane && ./make_net.sh)