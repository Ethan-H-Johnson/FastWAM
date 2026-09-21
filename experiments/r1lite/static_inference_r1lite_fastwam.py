#!/usr/bin/env python3
"""Replay one HDF5 R1 Lite demonstration through the robot-side FastWAM controls.

The source ``actions`` contain 14 deltas. The existing robot client expects
12 arm deltas followed by two absolute gripper targets, so this client takes
those gripper targets from ``absolute_actions`` in the same demonstration.
Replay groups consecutive actions into 32-step server-sized chunks by default.
Execution is opt-in with --execute; Enter starts replay and Enter stops it.
"""
from __future__ import annotations

import argparse
import os
from pathlib import Path
import time
import traceback

import h5py
import numpy as np


DEFAULT_DATASET = Path(__file__).resolve().parents[2] / "stack_bowl_new_20260805_040218.hdf5"
ACTION_DIM = 14
DEFAULT_CHUNK_SIZE = 32  # Server default: training num_frames (33) minus one.
JOINTS_PER_ARM = 6
MAX_JOINT_DELTA_RAD = 0.25
MIN_EE_Z = {
    "left": 0.2126,
    "right": 0.2312,
}
JOINT_LIMITS_RAD = np.asarray(
    [
        (-1.20, 0.70),
        (-0.40, 2.80),
        (-2.35, 0.40),
        (-0.80, 1.20),
        (-0.70, 0.70),
        (-1.10, 1.30),
    ],
    dtype=np.float32,
)


def end_effector_z(joints: np.ndarray) -> float:
    """Compute gripper-link Z from the installed R1 Lite URDF chain."""
    q2, q3, q4, q5 = joints[1:5]
    q23 = q2 + q3
    q234 = q23 + q4
    return float(
        0.25836
        + 0.3 * np.sin(q2)
        - 0.1747 * np.sin(q23)
        + 0.075485 * np.cos(q23)
        - np.sin(q234) * (0.08 + 0.104153 * np.cos(q5))
    )


def prepare_replay_targets(
    action: np.ndarray,
    current_state: np.ndarray,
    gripper_min: float,
    gripper_max: float,
) -> np.ndarray | None:
    """Apply PI0.5's joint-delta and end-effector-Z safety checks."""
    action = np.asarray(action[:ACTION_DIM], dtype=np.float32)
    joint_delta = np.clip(action[:12], -MAX_JOINT_DELTA_RAD, MAX_JOINT_DELTA_RAD)

    current = np.asarray(current_state, dtype=np.float32)
    targets = current.copy()
    targets[:12] += joint_delta
    targets[12:14] = np.clip(action[12:14], gripper_min, gripper_max)

    for index, side in enumerate(("left", "right")):
        start = index * JOINTS_PER_ARM
        stop = start + JOINTS_PER_ARM
        targets[start:stop] = np.clip(
            targets[start:stop],
            JOINT_LIMITS_RAD[:, 0],
            JOINT_LIMITS_RAD[:, 1],
        )
        target_z = end_effector_z(targets[start:stop])
        current_z = end_effector_z(current[start:stop])
        if target_z < MIN_EE_Z[side]:
            print(
                f"Z BOUNDARY: {side} target {target_z:.3f} m is below "
                f"{MIN_EE_Z[side]:.3f} m (current {current_z:.3f} m). "
                "Holding and stopping replay."
            )
            return None

    return targets


def list_demos(dataset: Path) -> None:
    with h5py.File(dataset, "r") as source:
        if "data" not in source:
            raise ValueError(f"No /data group in {dataset}")
        names = sorted(source["data"], key=lambda name: int(name.removeprefix("demo_")))
        for name in names:
            group = source[f"data/{name}"]
            print(
                f"{name}: {len(group['actions'])} steps, "
                f"{float(group.attrs['control_hz']):g} Hz, "
                f"success={bool(group.attrs.get('success', False))}"
            )


def load_demo(
    dataset: Path, demo: str, start_step: int, max_steps: int | None
) -> tuple[np.ndarray, np.ndarray, float, int]:
    """Load the selected episode as arm deltas plus absolute gripper targets."""
    with h5py.File(dataset, "r") as source:
        key = f"data/{demo}"
        if key not in source:
            raise ValueError(f"{key} not found in {dataset}; use --list-demos to inspect available episodes")
        group = source[key]
        for field in ("actions", "absolute_actions", "joint_states"):
            if field not in group:
                raise ValueError(f"{key}/{field} is missing")
        shape = group["actions"].shape
        if len(shape) != 2 or shape[1] != ACTION_DIM:
            raise ValueError(f"Expected {key}/actions to have shape (steps, {ACTION_DIM}), got {shape}")
        if any(group[field].shape != shape for field in ("absolute_actions", "joint_states")):
            raise ValueError(f"Action and state shapes differ in {key}")
        if start_step >= shape[0]:
            raise ValueError(f"Start step {start_step} is beyond the {shape[0]} steps in {demo}")
        stop_step = shape[0] if max_steps is None else min(shape[0], start_step + max_steps)
        deltas = np.asarray(group["actions"][start_step:stop_step], dtype=np.float32)
        absolute = np.asarray(group["absolute_actions"][start_step:stop_step], dtype=np.float32)
        states = np.asarray(group["joint_states"][start_step:stop_step], dtype=np.float32)
        if not (np.isfinite(deltas).all() and np.isfinite(absolute).all() and np.isfinite(states).all()):
            raise ValueError(f"Selected steps in {demo} contain nonfinite values")
        if not np.allclose(deltas, absolute - states, rtol=0, atol=1e-4):
            raise ValueError(f"{demo}/actions do not match absolute_actions - joint_states")
        control_hz = float(group.attrs["control_hz"])
        if not np.isfinite(control_hz) or control_hz <= 0:
            raise ValueError(f"Invalid control_hz for {demo}: {control_hz}")
        actions = np.concatenate((deltas[:, :12], absolute[:, 12:14]), axis=1)
        return actions, states[0].copy(), control_hz, stop_step


def action_chunks(
    actions: np.ndarray, start_step: int, chunk_size: int, replan_steps: int | None
):
    """Yield server-sized windows and the prefix executed before the next window."""
    cursor = 0
    chunk_id = 0
    while cursor < len(actions):
        chunk_id += 1
        chunk = actions[cursor:cursor + chunk_size]
        n_execute = len(chunk) if replan_steps is None else min(replan_steps, len(chunk))
        yield chunk_id, start_step + cursor, chunk, n_execute
        cursor += n_execute


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", type=Path, default=DEFAULT_DATASET, help="Source HDF5 dataset")
    parser.add_argument("--list-demos", action="store_true", help="List episodes and exit without ROS")
    parser.add_argument("--demo", help="Episode to replay, for example demo_0")
    parser.add_argument("--start-step", type=int, default=0, help="Zero-based first dataset step")
    parser.add_argument("--max-steps", type=int, help="Limit the number of replayed steps")
    parser.add_argument(
        "--chunk-size", type=int, default=DEFAULT_CHUNK_SIZE,
        help="Actions per dataset window (default: server horizon of 32)",
    )
    parser.add_argument(
        "--replan-steps", type=int,
        help="Actions to execute from each window before selecting the next one (default: full window)",
    )
    parser.add_argument("--control-hz", type=float, help="Override the episode's recorded control rate")
    parser.add_argument("--execute", action="store_true", help="Publish robot targets; default is dry-run")
    parser.add_argument("--gripper-min", type=float, default=0.0)
    parser.add_argument("--gripper-max", type=float, default=100.0)
    parser.add_argument("--max-feedback-age-ms", type=float, default=250.0)
    parser.add_argument("--max-start-arm-error-rad", type=float, default=0.15)
    parser.add_argument("--max-start-gripper-error", type=float, default=10.0)
    parser.add_argument(
        "--rollouts-dir",
        type=Path,
        default=Path(__file__).resolve().parents[2] / "rollouts" / "r1lite",
    )
    return parser.parse_args()


def feedback_is_fresh(robot: object, max_age_s: float) -> bool:
    """Check that all four arm and gripper streams have recent ROS stamps."""
    now = robot.get_clock().now().nanoseconds / 1e9
    for name in ("left_arm", "right_arm", "left_gripper", "right_gripper"):
        samples = robot._samples[name]
        if not samples or not 0 <= now - samples[-1][0] <= max_age_s:
            return False
    return True


def main() -> None:
    args = parse_args()
    if args.list_demos:
        list_demos(args.dataset)
        return
    if not args.demo or not args.demo.startswith("demo_") or not args.demo[5:].isdigit():
        raise ValueError("Specify --demo demo_N (or use --list-demos)")
    if (
        args.start_step < 0
        or (args.max_steps is not None and args.max_steps <= 0)
        or args.chunk_size <= 0
        or (args.replan_steps is not None and args.replan_steps <= 0)
        or (args.control_hz is not None and (not np.isfinite(args.control_hz) or args.control_hz <= 0))
        or not np.isfinite(
            [args.gripper_min, args.gripper_max, args.max_feedback_age_ms,
             args.max_start_arm_error_rad, args.max_start_gripper_error]
        ).all()
        or args.gripper_min > args.gripper_max
        or args.max_feedback_age_ms <= 0
        or args.max_start_arm_error_rad < 0
        or args.max_start_gripper_error < 0
    ):
        raise ValueError("Invalid replay step, control rate, gripper range, or feedback limit")

    actions, recorded_start, recorded_hz, stop_step = load_demo(
        args.dataset, args.demo, args.start_step, args.max_steps
    )
    control_hz = args.control_hz or recorded_hz
    print(f"Loaded {len(actions)} steps from {args.demo} [{args.start_step}:{stop_step}] at {control_hz:g} Hz.")

    # Import ROS components only after the dataset has passed validation, so
    # --list-demos and source checks also work on machines without ROS.
    import rclpy
    from inference_r1lite_fastwam import (
        ActionStepLogger,
        ClientObservationRecorder,
        EnterHotkey,
        RobotIO,
        RolloutSession,
        TrajectoryRecorder,
        reset_to_initial_position,
        wall_time,
    )

    session = RolloutSession.create(
        args.rollouts_dir,
        "static_fastwam_replay",
        {
            "source_dataset": str(args.dataset.resolve()),
            "source_demo": args.demo,
            "start_step": args.start_step,
            "stop_step": stop_step,
            "recorded_control_hz": recorded_hz,
            "control_hz": control_hz,
            "chunk_size": args.chunk_size,
            "replan_steps": args.replan_steps,
            "execute": args.execute,
            "max_start_arm_error_rad": args.max_start_arm_error_rad,
            "max_start_gripper_error": args.max_start_gripper_error,
            "created_at": wall_time(),
        },
    )
    rclpy.init()
    video_recorder = TrajectoryRecorder()
    robot = RobotIO("raw", max(control_hz, 20.0), 3.0, video_recorder)
    step_logger = ActionStepLogger(session)
    event_logger = ClientObservationRecorder(session)
    hotkey = EnterHotkey(on_stop=robot.hold_current_feedback if args.execute else None)
    run_demo = "demo_1"
    started = False
    completed = 0
    chunks_loaded = 0
    failed = False
    z_boundary_reached = False
    print("EXECUTE ENABLED" if args.execute else "DRY RUN: add --execute to move the robot")
    print("Press Enter to start replay; press Enter again to stop and reset.")
    try:
        while rclpy.ok() and not hotkey.running.is_set():
            robot.spin_for(0.05)
        while rclpy.ok() and hotkey.running.is_set() and not feedback_is_fresh(
            robot, args.max_feedback_age_ms / 1000.0
        ):
            robot.spin_for(0.05)
        if not rclpy.ok():
            raise RuntimeError("ROS shut down before replay could begin")
        if not hotkey.running.is_set():
            raise RuntimeError("Replay stopped before fresh feedback arrived")
        if args.execute:
            measured_start = robot.local_state()
            arm_error = float(np.max(np.abs(measured_start[:12] - recorded_start[:12])))
            gripper_error = float(np.max(np.abs(measured_start[12:14] - recorded_start[12:14])))
            if arm_error > args.max_start_arm_error_rad or gripper_error > args.max_start_gripper_error:
                raise RuntimeError(
                    f"Robot start pose differs from {args.demo} step {args.start_step}: "
                    f"arm {arm_error:.3f} rad, gripper {gripper_error:.3f} units; "
                    "move to the recorded start pose or adjust the start-error limits"
                )
        video_recorder.start_demo(session.demo_dir(run_demo))
        event_logger.append_event(run_demo, "replay_started", {"source_demo": args.demo, "start_step": args.start_step})
        started = True
        period = 1.0 / control_hz
        for chunk_id, chunk_start, chunk, n_execute in action_chunks(
            actions, args.start_step, args.chunk_size, args.replan_steps
        ):
            if not hotkey.running.is_set():
                break
            chunks_loaded += 1
            event_logger.append_event(
                run_demo, "chunk_selected",
                {"chunk_id": chunk_id, "dataset_start_step": chunk_start,
                 "actions_available": len(chunk), "actions_to_execute": n_execute},
            )
            print(f"Chunk {chunk_id}: dataset steps {chunk_start}:{chunk_start + len(chunk)}; executing {n_execute}.")
            for action_step, action in enumerate(chunk[:n_execute], start=1):
                if not hotkey.running.is_set():
                    break
                if not feedback_is_fresh(robot, args.max_feedback_age_ms / 1000.0):
                    raise RuntimeError("Arm or gripper feedback is missing or stale; stopping replay")
                tick_started = time.monotonic()
                measured_before = robot.local_state().copy()
                target = prepare_replay_targets(
                    action, measured_before, args.gripper_min, args.gripper_max
                )
                if target is None:
                    z_boundary_reached = True
                    event_logger.append_event(
                        run_demo,
                        "z_boundary",
                        {"chunk_id": chunk_id, "action_step": action_step},
                    )
                    break
                published = False
                if args.execute:
                    if not robot.publish_targets_if_enabled(target, hotkey.running):
                        break
                    robot.start_hold()
                    published = True
                else:
                    source_step = chunk_start + action_step - 1
                    print(f"[dry-run] dataset step {source_step}: {np.array2string(target, precision=4)}")
                remaining = period - (time.monotonic() - tick_started)
                if remaining > 0:
                    robot.spin_for(remaining)
                measured_after = robot.local_state().copy()
                step_logger.submit(
                    {
                        "run_id": "static_replay",
                        "demo_id": run_demo,
                        "chunk_id": chunk_id,
                        "action_step": action_step,
                        "client_time": wall_time(),
                        "model_action": action.tolist(),
                        "measured_before": measured_before.tolist(),
                        "commanded_target": target.tolist(),
                        "measured_after": measured_after.tolist(),
                        "published": published,
                        "tick_duration_s": time.monotonic() - tick_started,
                    }
                )
                completed += 1
            if z_boundary_reached or not hotkey.running.is_set():
                break
        if args.execute:
            robot.hold_current_feedback()
        video_recorder.stop_demo()
        event_logger.append_event(
            run_demo, "replay_stopped", {"steps_executed": completed, "chunks_loaded": chunks_loaded}
        )
        print(f"Replay stopped after {completed} steps; running reset.")
        reset_to_initial_position(robot, args.execute)
        event_logger.append_event(run_demo, "reset_finished")
    except KeyboardInterrupt:
        print("\nEmergency exit requested; automatic reset is skipped.")
        if started:
            event_logger.append_event(run_demo, "emergency_exit", {"steps_executed": completed})
    except Exception:
        failed = True
        traceback.print_exc()
        if args.execute:
            robot.hold_current_feedback()
        if started:
            event_logger.append_event(run_demo, "replay_failed", {"steps_executed": completed})
    finally:
        hotkey.close()
        video_recorder.close()
        step_logger.close()
        robot.close()
        robot.destroy_node()
        rclpy.shutdown()
    # EnterHotkey's daemon can remain blocked inside input() at interpreter
    # shutdown, so exit only after all recorders and ROS resources are closed.
    os._exit(1 if failed else 0)


if __name__ == "__main__":
    main()
