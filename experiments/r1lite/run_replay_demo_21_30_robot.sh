#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd -- "${SCRIPT_DIR}/../.." && pwd)"
export PYTHONNOUSERSITE=1
export ROS_DOMAIN_ID="${ROS_DOMAIN_ID:-61}"
export FASTRTPS_DEFAULT_PROFILES_FILE="${FASTRTPS_DEFAULT_PROFILES_FILE:-/opt/galaxea/find_server/super_client_configuration_file.xml}"

set +u
source "${R1LITE_ROS_SETUP:-/opt/ros/humble/setup.bash}"
source "${R1LITE_GALAXEA_SETUP:-$HOME/galaxea/install/setup.bash}"
set -u

cd "$REPO_ROOT"
exec python3 experiments/r1lite/replay_demo_range_robot.py \
  --hdf5 /home/r1lite/Documents/stack_bowl_20260822_000619.hdf5 \
  --offline-results /home/r1lite/Documents/FastWAM/demo_21_30_offline_results \
  --demo-start 21 --demo-end 30 --num-chunks 5 \
  "$@"
