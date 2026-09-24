#!/usr/bin/env python3
"""Replay one HDF5 demo with the same relative execution used by FastWAM."""

from __future__ import annotations

import argparse
from datetime import datetime
import json
from pathlib import Path
import time

import h5py
import numpy as np
import rclpy

from inference_r1lite_fastwam import EnterHotkey, RobotIO, TrajectoryRecorder


ACTION_DIM = 14
PROJECT_ROOT = Path(__file__).resolve().parents[2]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--hdf5", type=Path, required=True)
    parser.add_argument("--demo", default="demo_0", help="Demo name or number, for example demo_12 or 12")
    parser.add_argument("--control-hz", type=float, default=20.0)
    parser.add_argument("--max-steps", type=int, default=None)
    parser.add_argument("--feedback-timeout-s", type=float, default=10.0)
    parser.add_argument("--initial-move-s", type=float, default=5.0)
    parser.add_argument("--initial-arm-tolerance", type=float, default=0.12)
    parser.add_argument("--initial-gripper-tolerance", type=float, default=15.0)
    parser.add_argument("--gripper-min", type=float, default=0.0)
    parser.add_argument("--gripper-max", type=float, default=100.0)
    parser.add_argument("--execute", action="store_true")
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=PROJECT_ROOT / "relative_hdf5_replay_results",
    )
    return parser.parse_args()


def normalize_demo_name(value: str) -> str:
    return value if value.startswith("demo_") else f"demo_{value}"


def load_demo(path: Path, demo_name: str) -> tuple[np.ndarray, np.ndarray, str]:
    with h5py.File(path, "r") as file:
        key = f"data/{demo_name}"
        if key not in file:
            available = sorted(file.get("data", {}).keys())
            raise KeyError(f"{demo_name!r} is not in {path}. Available demos: {available}")
        demo = file[key]
        if "actions" not in demo or "joint_states" not in demo:
            raise KeyError(f"{key} must contain actions and joint_states")
        actions = np.asarray(demo["actions"], dtype=np.float32)
        initial_state = np.asarray(demo["joint_states"][0], dtype=np.float32)
        task = str(demo.attrs.get("task_description", demo.attrs.get("subtask_key", "unknown")))

    if actions.ndim != 2 or actions.shape[1] < ACTION_DIM:
        raise ValueError(f"Expected actions shaped [T, >=14], got {actions.shape}")
    if initial_state.shape != (ACTION_DIM,):
        raise ValueError(f"Expected a 14-D initial joint state, got {initial_state.shape}")
    if not np.isfinite(actions[:, :ACTION_DIM]).all() or not np.isfinite(initial_state).all():
        raise ValueError("The selected demo contains non-finite state or action values")
    return actions[:, :ACTION_DIM], initial_state, task


def wait_for_start(robot: RobotIO, hotkey: EnterHotkey, demo_name: str, execute: bool) -> None:
    mode = "MOVE AND REPLAY" if execute else "DRY RUN"
    print(f"{mode}: press Enter to start {demo_name}; press Enter again to stop immediately.")
    while rclpy.ok() and not hotkey.running.is_set():
        robot.spin_for(0.05)


def move_to_demo_start(
    robot: RobotIO,
    hotkey: EnterHotkey,
    target: np.ndarray,
    args: argparse.Namespace,
) -> bool:
    start = robot.local_state().copy()
    print("Recorded initial state:", np.array2string(target, precision=4, suppress_small=True))
    print(
        f"Moving to the recorded initial pose over {args.initial_move_s:.1f}s; "
        f"maximum arm travel={np.max(np.abs(target[:12] - start[:12])):.4f} rad"
    )
    if not args.execute:
        print("DRY RUN: no initial-position or replay targets will be published.")
        return False

    steps = max(1, int(np.ceil(args.initial_move_s * args.control_hz)))
    period = 1.0 / args.control_hz
    for step in range(1, steps + 1):
        if not hotkey.running.is_set():
            return False
        tick_started = time.monotonic()
        command = start + (step / steps) * (target - start)
        if not robot.publish_targets_if_enabled(command, hotkey.running):
            return False
        robot.start_hold()
        remaining = period - (time.monotonic() - tick_started)
        if remaining > 0:
            robot.spin_for(remaining)

    robot.spin_for(0.3)
    measured = robot.local_state().copy()
    arm_error = float(np.max(np.abs(measured[:12] - target[:12])))
    gripper_error = float(np.max(np.abs(measured[12:14] - target[12:14])))
    print(f"Initial-pose error: arms={arm_error:.5f} rad, grippers={gripper_error:.3f}")
    if arm_error > args.initial_arm_tolerance or gripper_error > args.initial_gripper_tolerance:
        raise RuntimeError(
            "Robot did not reach the selected demo's initial pose: "
            f"limits are {args.initial_arm_tolerance:.3f} rad and "
            f"{args.initial_gripper_tolerance:.3f} gripper units"
        )
    return hotkey.running.is_set()


def main() -> None:
    args = parse_args()
    if args.control_hz <= 0 or args.initial_move_s <= 0 or args.feedback_timeout_s <= 0:
        raise ValueError("--control-hz, --initial-move-s, and --feedback-timeout-s must be positive")
    if args.max_steps is not None and args.max_steps <= 0:
        raise ValueError("--max-steps must be positive")
    if args.gripper_min > args.gripper_max:
        raise ValueError("--gripper-min cannot exceed --gripper-max")

    args.hdf5 = args.hdf5.expanduser().resolve()
    demo_name = normalize_demo_name(args.demo)
    actions, initial_state, task = load_demo(args.hdf5, demo_name)
    if args.max_steps is not None:
        actions = actions[: args.max_steps]

    run_dir = args.output_dir.expanduser().resolve() / datetime.now().strftime("run_%Y%m%d_%H%M%S_%f")
    run_dir.mkdir(parents=True, exist_ok=False)
    metadata = {
        "hdf5": str(args.hdf5),
        "demo": demo_name,
        "task": task,
        "steps": len(actions),
        "control_hz": args.control_hz,
        "initial_move_s": args.initial_move_s,
        "execute": args.execute,
        "execution": "arms=live_feedback+stored_residual; grippers=stored_absolute_action",
    }
    (run_dir / "run.json").write_text(json.dumps(metadata, indent=2) + "\n", encoding="utf-8")

    rclpy.init()
    recorder = TrajectoryRecorder()
    robot = RobotIO("raw", max(args.control_hz, 20.0), 3.0, recorder)
    hotkey = EnterHotkey(on_stop=robot.hold_current_feedback if args.execute else None)
    records: list[dict[str, object]] = []
    try:
        print(f"Waiting up to {args.feedback_timeout_s:.1f}s for arm and gripper feedback...")
        feedback_deadline = time.monotonic() + args.feedback_timeout_s
        while rclpy.ok() and time.monotonic() < feedback_deadline:
            try:
                robot.local_state()
                break
            except RuntimeError:
                robot.spin_for(0.1)
        else:
            missing = [
                name
                for name, value in (
                    ("left arm", robot.arm["left"]),
                    ("right arm", robot.arm["right"]),
                    ("left gripper", robot.gripper["left"]),
                    ("right gripper", robot.gripper["right"]),
                )
                if value is None
            ]
            raise RuntimeError(
                "Timed out waiting for ROS feedback: " + ", ".join(missing)
            )
        print("Robot feedback is ready.")
        wait_for_start(robot, hotkey, demo_name, args.execute)
        if not move_to_demo_start(robot, hotkey, initial_state, args):
            return

        print(
            f"Replaying {len(actions)} stored actions at {args.control_hz:.2f} Hz using "
            "FastWAM live-relative arm execution."
        )
        period = 1.0 / args.control_hz
        for index, action in enumerate(actions):
            if not hotkey.running.is_set():
                break
            tick_started = time.monotonic()
            measured_before = robot.local_state().copy()
            target = robot.prepare_targets(
                action,
                args.gripper_min,
                args.gripper_max,
                current_state=measured_before,
            )
            if not robot.publish_targets_if_enabled(target, hotkey.running):
                break
            robot.start_hold()
            remaining = period - (time.monotonic() - tick_started)
            if remaining > 0:
                robot.spin_for(remaining)
            measured_after = robot.local_state().copy()
            records.append(
                {
                    "step": index + 1,
                    "stored_action": action.tolist(),
                    "measured_before": measured_before.tolist(),
                    "commanded_target": target.tolist(),
                    "measured_after": measured_after.tolist(),
                    "tracking_error": (measured_after - target).tolist(),
                    "tick_duration_s": time.monotonic() - tick_started,
                }
            )
            if (index + 1) % 32 == 0 or index + 1 == len(actions):
                print(f"Replayed {index + 1}/{len(actions)} actions")

        (run_dir / "command_feedback.json").write_text(
            json.dumps({"steps": records}, indent=2) + "\n",
            encoding="utf-8",
        )
        print(f"Saved replay log: {run_dir / 'command_feedback.json'}")
        if hotkey.running.is_set():
            print("Replay finished. Holding the final target; press Enter to stop.")
            while hotkey.running.is_set():
                robot.spin_for(0.05)
    except KeyboardInterrupt:
        print("Emergency exit: holding current feedback.")
        if args.execute:
            robot.hold_current_feedback()
        raise
    finally:
        if records and not (run_dir / "command_feedback.json").exists():
            (run_dir / "command_feedback.json").write_text(
                json.dumps({"steps": records}, indent=2) + "\n",
                encoding="utf-8",
            )
        hotkey.close()
        robot.close()
        recorder.close()
        robot.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == "__main__":
    main()
