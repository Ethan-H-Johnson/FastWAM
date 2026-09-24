#!/usr/bin/env bash
set -euo pipefail

TMUX_PANE="mobiman:0.4"
GAIN_CONFIG="/home/r1lite/galaxea/install/mobiman/share/mobiman/config/joint_controller_kp_kd.toml"
SETUP_FILE="/home/r1lite/galaxea/install/setup.bash"
DISCOVERY_PROFILE="/opt/galaxea/find_server/super_client_configuration_file.xml"

if [[ ! -f "$GAIN_CONFIG" ]]; then
    echo "Missing original gain configuration: $GAIN_CONFIG" >&2
    exit 1
fi

if ! tmux has-session -t mobiman 2>/dev/null; then
    echo "The robot's mobiman tmux session is not running." >&2
    exit 1
fi

echo "Stopping the current R1 Lite joint tracker in $TMUX_PANE..."
tmux send-keys -t "$TMUX_PANE" C-c

for _ in {1..20}; do
    if ! pgrep -f '/r1_lite_jointTracker_demo_node([[:space:]]|$)' >/dev/null; then
        break
    fi
    sleep 0.25
done

if pgrep -f '/r1_lite_jointTracker_demo_node([[:space:]]|$)' >/dev/null; then
    echo "The old joint tracker did not stop; refusing to start a second instance." >&2
    exit 1
fi

launch_command="source /opt/ros/humble/setup.bash && source $SETUP_FILE && export ROS_DOMAIN_ID=61 && export FASTRTPS_DEFAULT_PROFILES_FILE=$DISCOVERY_PROFILE && ros2 launch mobiman r1_lite_jointTrackerdemo_fast_launch.py config_file:=$GAIN_CONFIG"

echo "Starting the joint tracker with the original Galaxea gains..."
tmux send-keys -t "$TMUX_PANE" "$launch_command" C-m
sleep 3

if ! pgrep -f '/r1_lite_jointTracker_demo_node([[:space:]]|$)' >/dev/null; then
    echo "Joint tracker did not start. Inspect it with: tmux capture-pane -pt $TMUX_PANE -S -80" >&2
    exit 1
fi

echo "Joint tracker restarted with: $GAIN_CONFIG"
echo "Expected right-arm Kp: [140, 200, 120, 80, 80, 80]"
echo "Expected right-arm Kd: [10, 30, 5, 10, 10, 10]"
