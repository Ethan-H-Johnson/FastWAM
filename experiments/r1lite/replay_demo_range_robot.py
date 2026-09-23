#!/usr/bin/env python3
"""Execute precomputed offline FastWAM trajectories on the R1 Lite.

No model or inference server is used. Each absolute arm target is reconstructed
from the recorded state that defined its training label:

    target_arm[t] = recorded_state_arm[t] + predicted_delta_arm[t]

Gripper outputs are already absolute. This isolates hardware tracking from
model inference and recorded/live proprio divergence.
"""
from __future__ import annotations

import argparse
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
    JOINT_LIMITS_RAD,
    RobotIO,
    TrajectoryRecorder,
    reset_to_initial_position,
)

PROJECT_ROOT = Path(__file__).resolve().parents[2]
PLOT_DIMENSIONS = [(6 + i, f"Right J{i + 1}") for i in range(6)] + [(13, "Right gripper")]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--hdf5", type=Path, required=True)
    parser.add_argument("--offline-results", type=Path, required=True)
    parser.add_argument("--seed", type=int, choices=(0, 1), default=0)
    parser.add_argument("--demo-start", type=int, default=21)
    parser.add_argument("--demo-end", type=int, default=30)
    parser.add_argument("--num-chunks", type=int, default=5)
    parser.add_argument("--all-chunks", action="store_true",
                        help="Execute every saved step instead of limiting to --num-chunks")
    parser.add_argument("--action-horizon", type=int, default=32)
    parser.add_argument("--control-hz", type=float, default=20.0)
    parser.add_argument("--chunk-pause-s", type=float, default=1.2,
                        help="Hold time simulating one online inference call")
    parser.add_argument("--initial-move-s", type=float, default=5.0)
    parser.add_argument("--initial-arm-tolerance", type=float, default=0.12)
    parser.add_argument("--initial-gripper-tolerance", type=float, default=15.0)
    parser.add_argument("--gripper-min", type=float, default=0.0)
    parser.add_argument("--gripper-max", type=float, default=100.0)
    parser.add_argument("--execute", action="store_true")
    parser.add_argument("--output-dir", type=Path,
                        default=PROJECT_ROOT / "demo_21_30_robot_results")
    return parser.parse_args()


def wait_for_start(robot: RobotIO, hotkey: EnterHotkey, demo_name: str, seed: int) -> None:
    print(f"Press Enter to position and START {demo_name} with seed {seed}. Enter again STOPS it.")
    while rclpy.ok() and not hotkey.running.is_set():
        robot.spin_for(0.05)


def hold_for(robot: RobotIO, seconds: float, running) -> bool:
    deadline = time.monotonic() + seconds
    while running.is_set() and time.monotonic() < deadline:
        robot.spin_for(min(0.05, max(0.0, deadline - time.monotonic())))
    return running.is_set()


def move_to_initial_pose(robot: RobotIO, target: np.ndarray, hotkey: EnterHotkey,
                         args: argparse.Namespace) -> bool:
    start = robot.local_state().copy()
    arm_travel = float(np.max(np.abs(target[:12] - start[:12])))
    print(f"Moving to recorded initial pose over {args.initial_move_s:.1f}s; max arm travel={arm_travel:.3f} rad")
    if not args.execute:
        print("DRY RUN: initial positioning targets are not published")
        return hotkey.running.is_set()

    steps = max(1, int(np.ceil(args.initial_move_s * args.control_hz)))
    for step in range(1, steps + 1):
        if not hotkey.running.is_set():
            return False
        tick = time.monotonic()
        command = start + (step / steps) * (target - start)
        if not robot.publish_targets_if_enabled(command, hotkey.running):
            return False
        robot.start_hold()
        remaining = 1.0 / args.control_hz - (time.monotonic() - tick)
        if remaining > 0:
            robot.spin_for(remaining)

    robot.spin_for(0.2)
    measured = robot.local_state()
    arm_error = float(np.max(np.abs(measured[:12] - target[:12])))
    gripper_error = float(np.max(np.abs(measured[12:14] - target[12:14])))
    print(f"Initial-pose error: arms={arm_error:.4f} rad, grippers={gripper_error:.2f}")
    if arm_error > args.initial_arm_tolerance or gripper_error > args.initial_gripper_tolerance:
        raise RuntimeError("Robot did not reach the recorded initial pose within tolerance")
    return hotkey.running.is_set()


def build_targets(recorded_states: np.ndarray, predicted_actions: np.ndarray,
                  left_hold: np.ndarray, gripper_min: float, gripper_max: float) -> np.ndarray:
    if recorded_states.shape != predicted_actions.shape or recorded_states.shape[1] != 14:
        raise ValueError("Recorded states and predicted actions must both have shape [T, 14]")
    targets = recorded_states.astype(np.float32, copy=True)
    targets[:, :12] += predicted_actions[:, :12]
    targets[:, :6] = np.clip(targets[:, :6], JOINT_LIMITS_RAD[:, 0], JOINT_LIMITS_RAD[:, 1])
    targets[:, 6:12] = np.clip(targets[:, 6:12], JOINT_LIMITS_RAD[:, 0], JOINT_LIMITS_RAD[:, 1])
    targets[:, 12:14] = np.clip(predicted_actions[:, 12:14], gripper_min, gripper_max)
    targets[:, :6] = left_hold[:6]
    targets[:, 12] = left_hold[12]
    return targets


def write_tracking_chart(path: Path, records: list[dict[str, object]], title: str) -> None:
    target = np.asarray([record["target"] for record in records], dtype=np.float32)
    measured = np.asarray([record["measured_after"] for record in records], dtype=np.float32)
    chunks = np.asarray([record["chunk"] for record in records], dtype=np.int32)
    canvas = np.full((1030, 1400, 3), 255, dtype=np.uint8)
    font = cv2.FONT_HERSHEY_SIMPLEX
    cv2.putText(canvas, title, (40, 30), font, 0.75, (0, 0, 0), 2)
    cv2.putText(canvas, "target: blue line | measured: orange line | vertical: next chunk", (40, 58), font, 0.55, (0, 0, 0), 1)
    boundaries = [i for i in range(1, len(chunks)) if chunks[i] != chunks[i - 1]]
    for panel, (dim, name) in enumerate(PLOT_DIMENSIONS):
        row, col = divmod(panel, 2)
        x0, y0 = 90 + col * 700, 100 + row * 218
        x1, y1 = x0 + 570, y0 + 138
        scale = 180.0 / np.pi if dim < 12 else 1.0
        target_values = target[:, dim] * scale
        measured_values = measured[:, dim] * scale
        values = np.concatenate((target_values, measured_values))
        margin = max(float(np.ptp(values)) * 0.06, 0.001)
        low, high = float(values.min()) - margin, float(values.max()) + margin
        mae = float(np.abs(measured_values - target_values).mean())
        unit = "deg" if dim < 12 else "gripper units"
        cv2.putText(canvas, f"{name} | MAE={mae:.5f} {unit}", (x0, y0 - 10), font, 0.46, (0, 0, 0), 1)
        cv2.rectangle(canvas, (x0, y0), (x1, y1), (120, 120, 120), 1)

        def point(index: int, value: float) -> tuple[int, int]:
            x = x0 + int(index * (x1 - x0) / max(len(records) - 1, 1))
            y = y1 - int((value - low) * (y1 - y0) / max(high - low, 1e-9))
            return x, y

        target_points = np.asarray([point(i, float(v)) for i, v in enumerate(target_values)], np.int32)
        measured_points = np.asarray([point(i, float(v)) for i, v in enumerate(measured_values)], np.int32)
        cv2.polylines(canvas, [target_points], False, (200, 80, 20), 2, cv2.LINE_AA)
        cv2.polylines(canvas, [measured_points], False, (0, 130, 230), 2, cv2.LINE_AA)
        for boundary in boundaries:
            x, _ = point(boundary, low)
            cv2.line(canvas, (x, y0), (x, y1), (180, 180, 180), 1, cv2.LINE_AA)
    cv2.imwrite(str(path), canvas)


def save_demo(output: Path, source_demo: str, seed: int,
              records: list[dict[str, object]]) -> dict[str, object] | None:
    output.mkdir(parents=True, exist_ok=True)
    (output / "command_feedback.json").write_text(json.dumps({"steps": records}, indent=2) + "\n")
    if not records:
        return None
    write_tracking_chart(output / "model_target_vs_measured.png", records,
                         f"{source_demo} | offline seed {seed}")
    target = np.asarray([record["target"] for record in records], dtype=np.float32)
    measured = np.asarray([record["measured_after"] for record in records], dtype=np.float32)
    error = measured - target
    summary = {
        "source_demo": source_demo,
        "offline_seed": seed,
        "steps": len(records),
        "chunks": sorted({int(record["chunk"]) for record in records}),
        "per_dimension_tracking_mae": {
            name: float(np.abs(error[:, dim]).mean()) for dim, name in PLOT_DIMENSIONS
        },
        "metric_units": {"right_arm_joints": "radians", "right_gripper": "native gripper units"},
        "chart": str(output / "model_target_vs_measured.png"),
    }
    (output / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    return summary


def main() -> None:
    args = parse_args()
    if args.demo_end < args.demo_start or args.num_chunks < 1 or args.action_horizon < 1:
        raise ValueError("Invalid demo range, chunk count, or action horizon")
    if args.control_hz <= 0 or args.chunk_pause_s < 0 or args.initial_move_s <= 0:
        raise ValueError("Invalid timing argument")
    args.hdf5 = args.hdf5.expanduser().resolve()
    args.offline_results = args.offline_results.expanduser().resolve()
    run_name = datetime.now().strftime(f"run_%Y%m%d_%H%M%S_%f_seed_{args.seed}")
    args.output_dir = args.output_dir.expanduser().resolve() / run_name
    args.output_dir.mkdir(parents=True, exist_ok=False)

    metadata = {
        "mode": "precomputed_offline_trajectory",
        "hdf5": str(args.hdf5),
        "offline_results": str(args.offline_results),
        "offline_seed": args.seed,
        "source_demos": [f"demo_{n}" for n in range(args.demo_start, args.demo_end + 1)],
        "num_chunks": args.num_chunks,
        "all_chunks": args.all_chunks,
        "action_horizon": args.action_horizon,
        "control_hz": args.control_hz,
        "chunk_pause_s": args.chunk_pause_s,
        "execute": args.execute,
        "target_definition": "recorded_state[t] + predicted_delta[t]; grippers absolute",
    }
    (args.output_dir / "run.json").write_text(json.dumps(metadata, indent=2) + "\n")

    rclpy.init()
    recorder = TrajectoryRecorder()
    robot = RobotIO("raw", max(args.control_hz, 20.0), 3.0, recorder)
    hotkey = EnterHotkey(on_stop=robot.hold_current_feedback if args.execute else None)
    demo_summaries: list[dict[str, object]] = []
    tracking_sum = np.zeros(14, dtype=np.float64)
    tracking_count = 0
    try:
        robot.spin_for(1.0)
        print("EXECUTE ENABLED" if args.execute else "DRY RUN: no robot commands will be published")
        with h5py.File(args.hdf5, "r") as file:
            for demo_number in range(args.demo_start, args.demo_end + 1):
                demo_name = f"demo_{demo_number}"
                demo = file[f"data/{demo_name}"]
                arrays_path = args.offline_results / demo_name / "actions.npz"
                with np.load(arrays_path) as arrays:
                    predicted = np.asarray(arrays[f"predicted_action_seed_{args.seed}"], dtype=np.float32)
                available = min(len(predicted), len(demo["joint_states"]))
                count = available if args.all_chunks else min(args.num_chunks * args.action_horizon, available)
                recorded_states = np.asarray(demo["joint_states"][:count], dtype=np.float32)
                predicted = predicted[:count]
                initial_pose = np.asarray(demo["joint_states"][0], dtype=np.float32)

                wait_for_start(robot, hotkey, demo_name, args.seed)
                live_start = robot.local_state().copy()
                initial_pose[:6] = live_start[:6]
                initial_pose[12] = live_start[12]
                targets = build_targets(
                    recorded_states, predicted, live_start, args.gripper_min, args.gripper_max
                )
                records: list[dict[str, object]] = []
                published_any = False
                if not move_to_initial_pose(robot, initial_pose, hotkey, args):
                    save_demo(args.output_dir / demo_name, demo_name, args.seed, records)
                    if args.execute:
                        reset_to_initial_position(robot, execute=True)
                    continue
                published_any = args.execute

                for chunk_start in range(0, count, args.action_horizon):
                    if not hotkey.running.is_set():
                        break
                    chunk = chunk_start // args.action_horizon + 1
                    chunk_end = min(chunk_start + args.action_horizon, count)
                    print(f"{demo_name} chunk {chunk}: holding {args.chunk_pause_s:.2f}s for simulated inference")
                    if not hold_for(robot, args.chunk_pause_s, hotkey.running):
                        break
                    print(f"{demo_name} chunk {chunk}: executing steps {chunk_start}-{chunk_end - 1}")

                    for index in range(chunk_start, chunk_end):
                        if not hotkey.running.is_set():
                            break
                        tick = time.monotonic()
                        before = robot.local_state().copy()
                        target = targets[index]
                        published = False
                        if args.execute:
                            published = robot.publish_targets_if_enabled(target, hotkey.running)
                            if not published:
                                break
                            robot.start_hold()
                            published_any = True
                        remaining = 1.0 / args.control_hz - (time.monotonic() - tick)
                        if remaining > 0:
                            robot.spin_for(remaining)
                        after = robot.local_state().copy()
                        records.append({
                            "source_demo": demo_name,
                            "offline_seed": args.seed,
                            "chunk": chunk,
                            "dataset_frame": index,
                            "action_step": index - chunk_start + 1,
                            "model_action": predicted[index].tolist(),
                            "recorded_state": recorded_states[index].tolist(),
                            "measured_before": before.tolist(),
                            "target": target.tolist(),
                            "measured_after": after.tolist(),
                            "tracking_error": (after - target).tolist(),
                            "published": published,
                        })

                if hotkey.running.is_set():
                    print(f"{demo_name} configured chunks finished. Holding; press Enter to STOP and reset.")
                    while hotkey.running.is_set():
                        robot.spin_for(0.05)

                summary = save_demo(args.output_dir / demo_name, demo_name, args.seed, records)
                if summary is not None:
                    demo_summaries.append(summary)
                    target = np.asarray([r["target"] for r in records], dtype=np.float32)
                    measured = np.asarray([r["measured_after"] for r in records], dtype=np.float32)
                    tracking_sum += np.abs(measured - target).sum(axis=0)
                    tracking_count += len(records)
                print(f"Saved {demo_name} under {args.output_dir / demo_name}")
                if args.execute and published_any:
                    reset_to_initial_position(robot, execute=True)

        run_summary = {
            **metadata,
            "completed_demos": demo_summaries,
            "aggregate_per_dimension_tracking_mae": {
                name: float(tracking_sum[dim] / tracking_count)
                for dim, name in PLOT_DIMENSIONS
            } if tracking_count else {},
        }
        (args.output_dir / "summary.json").write_text(json.dumps(run_summary, indent=2) + "\n")
        print(f"Completed trajectory replay. Results: {args.output_dir}")
    except KeyboardInterrupt:
        print("Emergency exit: holding current feedback; automatic reset skipped.")
        if args.execute:
            robot.hold_current_feedback()
        raise
    except Exception:
        print("Replay failed: holding current feedback; automatic reset skipped.")
        if args.execute:
            robot.hold_current_feedback()
        raise
    finally:
        hotkey.close()
        recorder.close()
        robot.close()
        robot.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == "__main__":
    main()
