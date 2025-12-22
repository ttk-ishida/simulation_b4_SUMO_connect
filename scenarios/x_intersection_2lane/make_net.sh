#!/usr/bin/env bash
set -euo pipefail

# Generate SUMO network file net.net.xml from nodes/edges/types.
# Requirements: SUMO (netconvert) installed and available in PATH.
# Usage:
#   cd scenarios/x_intersection_2lane
#   ./make_net.sh

netconvert   --node-files nodes.nod.xml   --edge-files edges.edg.xml   --type-files types.typ.xml   --output-file net.net.xml   --no-internal-links false   --junctions.join true   --roundabouts.guess true   --geometry.remove true   --verbose false

echo "Generated net.net.xml"
