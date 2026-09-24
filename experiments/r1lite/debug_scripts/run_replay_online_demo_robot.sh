#!/usr/bin/env bash
# Robot-side HDF5 replay client. Requires the workstation FastWAM server.
set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd -- "${SCRIPT_DIR}/../.." && pwd)"
ROS_SETUP="${R1LITE_ROS_SETUP:-/opt/ros/humble/setup.bash}"
GALAXEA_SETUP="${R1LITE_GALAXEA_SETUP:-$HOME/galaxea/install/setup.bash}"
export PYTHONNOUSERSITE=1

set +u
source "$ROS_SETUP"
source "$GALAXEA_SETUP"
set -u

cd "$REPO_ROOT"
exec python3 experiments/r1lite/replay_online_demo_robot.py "$@"
