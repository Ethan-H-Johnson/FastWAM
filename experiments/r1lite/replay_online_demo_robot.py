#!/usr/bin/env python3
"""Run FastWAM on recorded HDF5 observations, then execute on the R1 Lite.

The workstation server still loads the model. This robot-side client sends
recorded images and proprio to /infer, then applies each returned action to
fresh ROS joint feedback, exactly as the normal client does. Recorded states
are used only for model input and a live-pose safety check, never as commands.
"""
from __future__ import annotations

import argparse
from collections import deque
from datetime import datetime
import json
from pathlib import Path
import time

import cv2
import h5py
import numpy as np
import rclpy

from inference_r1lite_fastwam import (
    EnterHotkey,
    ObservationBundle,
    RobotIO,
    TrajectoryRecorder,
    post_json,
    request_actions,
    reset_to_initial_position,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--hdf5", type=Path, required=True, help="HDF5 file on the robot")
    parser.add_argument("--demo", default="demo_0")
    parser.add_argument("--server", required=True, help="Workstation URL, e.g. http://10.42.0.90:8000")
    parser.add_argument("--prompt", required=True)
    parser.add_argument("--execute", action="store_true", help="Publish ROS targets; otherwise dry-run")
    parser.add_argument("--start-index", type=int, default=0)
    parser.add_argument("--num-chunks", type=int, default=1)
    parser.add_argument("--replan-steps", type=int, help="Default: execute the full returned action chunk")
    parser.add_argument("--control-hz", type=float, default=20.0)
    parser.add_argument("--request-timeout", type=float, default=120.0)
    parser.add_argument("--max-arm-pose-error", type=float, default=0.15,
                        help="Stop if any live arm joint differs from recorded state by more than this many radians")
    parser.add_argument("--max-gripper-pose-error", type=float, default=15.0)
    parser.add_argument("--gripper-min", type=float, default=0.0)
    parser.add_argument("--gripper-max", type=float, default=100.0)
    parser.add_argument("--output-dir", type=Path,
                        default=Path(__file__).resolve().parents[2] / "replay_online_results")
    return parser.parse_args()


def read_image(dataset: h5py.Dataset, index: int) -> np.ndarray:
    jpeg = np.asarray(dataset[index], dtype=np.uint8)
    bgr = cv2.imdecode(jpeg, cv2.IMREAD_COLOR)
    if bgr is None:
        raise ValueError(f"Cannot decode HDF5 image at frame {index}")
    return cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)


def dataset_observation(demo: h5py.Group, index: int) -> ObservationBundle:
    obs = demo["obs"]
    images = {
        "head": read_image(obs["agentview_rgb_jpeg"], index),
        "left_wrist": read_image(obs["wrist_left_rgb_jpeg"], index),
        "right_wrist": read_image(obs["wrist_right_rgb_jpeg"], index),
    }
    return ObservationBundle(
        timestamp_s=float(index),
        state=np.asarray(demo["joint_states"][index], dtype=np.float32),
        images=images,
        sample_timestamps_s={},
    )


def check_live_pose(live: np.ndarray, recorded: np.ndarray, args: argparse.Namespace) -> None:
    arm_error = float(np.max(np.abs(live[:12] - recorded[:12])))
    gripper_error = float(np.max(np.abs(live[12:14] - recorded[12:14])))
    print(f"Live/recorded pose difference: arms {arm_error:.3f} rad, grippers {gripper_error:.2f}")
    if arm_error > args.max_arm_pose_error or gripper_error > args.max_gripper_pose_error:
        raise RuntimeError("Live pose differs from the dataset; stopping before the next robot command")


def recorded_action(demo: h5py.Group, index: int) -> np.ndarray:
    action = np.asarray(demo["actions"][index], dtype=np.float32).copy()
    action[12:14] = demo["absolute_actions"][index, 12:14]
    return action


def main() -> None:
    args = parse_args()
    if args.start_index < 0 or args.num_chunks < 1 or args.control_hz <= 0:
        raise ValueError("Invalid start index, number of chunks, or control rate")
    if args.replan_steps is not None and args.replan_steps < 1:
        raise ValueError("--replan-steps must be positive")
    server = args.server.rstrip("/")
    args.output_dir.mkdir(parents=True, exist_ok=True)
    run_name = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
    trace_path = args.output_dir / f"{run_name}_{args.demo}.jsonl"

    with h5py.File(args.hdf5, "r") as file:
        demo = file[f"data/{args.demo}"]
        if args.start_index >= len(demo["joint_states"]):
            raise ValueError("Start index exceeds the demonstration length")

        rclpy.init()
        recorder = TrajectoryRecorder()
        robot = RobotIO("raw", max(args.control_hz, 20.0), 3.0, recorder)
        hotkey = None
        published_any = False
        try:
            robot.spin_for(1.0)
            observation = dataset_observation(demo, args.start_index)
            check_live_pose(robot.local_state(), observation.state, args)
            print("EXECUTE ENABLED" if args.execute else "DRY RUN (add --execute to move the robot)")
            input("Press Enter to start replay; press Enter again at any time to stop: ")
            hotkey = EnterHotkey(on_stop=robot.hold_current_feedback if args.execute else None)
            hotkey.running.set()
            run_id = post_json(server + "/runs/start", {
                "prompt": args.prompt, "num_demos": 1,
                "execute": args.execute, "control_hz": args.control_hz,
            }, args.request_timeout)["run_id"]
            index = args.start_index
            with trace_path.open("w", encoding="utf-8") as trace:
                for chunk_id in range(1, args.num_chunks + 1):
                    if not hotkey.running.is_set() or index >= len(demo["joint_states"]):
                        break
                    robot.spin_for(0.10)
                    observation = dataset_observation(demo, index)
                    check_live_pose(robot.local_state(), observation.state, args)
                    payload = robot.make_payload(args.prompt, str(run_id), "demo_1", chunk_id, observation)
                    actions = request_actions(server + "/infer", payload, args.request_timeout)
                    n_execute = len(actions) if args.replan_steps is None else min(args.replan_steps, len(actions))
                    action_queue = deque(enumerate(actions[:n_execute], start=1))
                    print(f"Chunk {chunk_id}: dataset frame {index}, executing {len(action_queue)} actions")
                    robot.spin_for(0.05)

                    while action_queue and hotkey.running.is_set():
                        step, action = action_queue.popleft()
                        frame = index + step - 1
                        if frame >= len(demo["joint_states"]):
                            break
                        tick_started = time.monotonic()
                        measured_before = robot.local_state().copy()
                        recorded_state = np.asarray(demo["joint_states"][frame], dtype=np.float32)
                        check_live_pose(measured_before, recorded_state, args)
                        target = robot.prepare_targets(
                            action, args.gripper_min, args.gripper_max, current_state=measured_before
                        )
                        published = False
                        if args.execute:
                            published = robot.publish_targets_if_enabled(target, hotkey.running)
                            if not published:
                                break
                            published_any = True
                            robot.start_hold()
                        remaining = 1.0 / args.control_hz - (time.monotonic() - tick_started)
                        if remaining > 0:
                            robot.spin_for(remaining)
                        measured_after = robot.local_state().copy()
                        truth = recorded_action(demo, frame)
                        record = {
                            "chunk": chunk_id, "dataset_frame": frame, "action_step": step,
                            "model_action": action.tolist(), "ground_truth_action": truth.tolist(),
                            "measured_before": measured_before.tolist(), "command_target": target.tolist(),
                            "measured_after": measured_after.tolist(), "published": published,
                        }
                        trace.write(json.dumps(record) + "\n")
                        trace.flush()
                    index += n_execute
            print(f"Replay trace: {trace_path}")
            if published_any:
                robot.hold_current_feedback()
                if hotkey.running.is_set():
                    print("Replay finished. Holding current pose; press Enter to stop and reset.")
                    while hotkey.running.is_set():
                        robot.spin_for(0.05)
                reset_to_initial_position(robot, execute=True)
        except (Exception, KeyboardInterrupt):
            if published_any:
                robot.hold_current_feedback()
                print("Replay stopped on error; holding current feedback until Enter. Reset skipped.")
                while hotkey is not None and hotkey.running.is_set():
                    robot.spin_for(0.05)
            raise
        finally:
            if hotkey is not None:
                hotkey.close()
            recorder.close()
            robot.close()
            robot.destroy_node()
            rclpy.shutdown()


if __name__ == "__main__":
    main()
