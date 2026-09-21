#!/usr/bin/env bash
# Launch static HDF5 action replay on the R1 Lite ROS 2 PC.
set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd -- "${SCRIPT_DIR}/../.." && pwd)"
ROS_SETUP="${R1LITE_ROS_SETUP:-/opt/ros/humble/setup.bash}"
GALAXEA_SETUP="${R1LITE_GALAXEA_SETUP:-$HOME/galaxea/install/setup.bash}"

# ROS Humble's cv_bridge on the robot uses system NumPy 1.x.
export PYTHONNOUSERSITE=1

set +u  # ROS setup files can reference unset variables.
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

if ! python3 -c 'import h5py' >/dev/null 2>&1; then
  echo "python3 cannot import h5py; install it in the robot's ROS Python environment." >&2
  exit 1
fi

cd "$REPO_ROOT"
exec python3 experiments/r1lite/static_inference_r1lite_fastwam.py \
  --rollouts-dir /home/r1lite/Documents/FastWAM/rollouts/r1lite "$@"
