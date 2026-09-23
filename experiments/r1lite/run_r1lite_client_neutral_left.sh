#!/usr/bin/env bash
# Robot-side A/B client using neutral left-arm proprio for model conditioning.
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
exec python3 experiments/r1lite/inference_r1lite_fastwam_neutral_left.py \
  --rollouts-dir /home/r1lite/Documents/FastWAM/rollouts/r1lite \
  --observations-dir /home/r1lite/Documents/FastWAM/observations \
  "$@"
