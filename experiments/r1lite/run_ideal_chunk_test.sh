#!/usr/bin/env bash
# Run the R1 Lite test that accumulates residuals from an ideal commanded state.
set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd -- "${SCRIPT_DIR}/../.." && pwd)"
ROS_SETUP="${R1LITE_ROS_SETUP:-/opt/ros/humble/setup.bash}"
GALAXEA_SETUP="${R1LITE_GALAXEA_SETUP:-$HOME/galaxea/install/setup.bash}"

# ROS Humble's cv_bridge on the R1 Lite is built against the system NumPy 1.x.
# Ignore incompatible user-site packages such as the robot's NumPy 2.x install.
export PYTHONNOUSERSITE=1

set +u  # ROS setup scripts reference variables that may not be set yet.
for setup_file in "$ROS_SETUP" "$GALAXEA_SETUP"; do
  if [[ ! -f "$setup_file" ]]; then
    echo "ROS setup file not found: $setup_file" >&2
    echo "Override it with R1LITE_ROS_SETUP or R1LITE_GALAXEA_SETUP." >&2
    exit 1
  fi
  # shellcheck disable=SC1090
  source "$setup_file"
done
set -u

cd "$REPO_ROOT"
exec python3 experiments/r1lite/inference_r1lite_fastwam_ideal_chunk.py \
  --rollouts-dir /home/r1lite/Documents/FastWAM/rollouts/r1lite \
  --observations-dir /home/r1lite/Documents/FastWAM/observations \
  --policy-config fastwam_ideal_chunk \
  "$@"
