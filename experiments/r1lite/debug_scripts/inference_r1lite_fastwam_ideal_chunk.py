#!/usr/bin/env python3
"""Test ideal-state accumulation for a FastWAM policy on the R1 Lite.

The workstation returns [d_left_6, d_right_6, left_gripper, right_gripper].
At the first action in each chunk, this client adds the arm residual to the
synchronized feedback state included in that chunk's model observation. For the
rest of the chunk, it adds each residual to the preceding commanded arm
position, treating that command as the state the robot would have reached in an
ideal world. Gripper outputs remain absolute. This is an experimental execution
mode; accumulated arm commands are still clipped to the configured per-action
delta and the robot joint limits.
"""
from __future__ import annotations

import argparse
import base64
from collections import deque
from dataclasses import dataclass
from datetime import datetime
import json
import os
from pathlib import Path
import queue
import re
import shutil
import signal
import subprocess
import threading
import time
import traceback
from typing import Callable
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen

import cv2
from cv_bridge import CvBridge
import numpy as np
import rclpy
from rclpy.node import Node
from rclpy.qos import HistoryPolicy, QoSProfile, ReliabilityPolicy
from sensor_msgs.msg import CompressedImage, Image, JointState

ARM_FEEDBACK_TOPICS = {"left": "/hdas/feedback_arm_left", "right": "/hdas/feedback_arm_right"}
GRIPPER_FEEDBACK_TOPICS = {"left": "/hdas/feedback_gripper_left", "right": "/hdas/feedback_gripper_right"}
ARM_TARGET_TOPICS = {
    "left": "/motion_target/target_joint_state_arm_left",
    "right": "/motion_target/target_joint_state_arm_right",
}
GRIPPER_TARGET_TOPICS = {
    "left": "/motion_target/target_position_gripper_left",
    "right": "/motion_target/target_position_gripper_right",
}
HEAD_CAMERA_TOPIC = "/hdas/camera_head/left_raw/image_raw_color/compressed"
RAW_WRIST_TOPICS = {
    "left_wrist": "/hdas/camera_wrist_left/color/image_raw",
    "right_wrist": "/hdas/camera_wrist_right/color/image_raw",
}
COMPRESSED_WRIST_TOPICS = {
    "left_wrist": "/hdas/camera_wrist_left/color/image_raw/compressed",
    "right_wrist": "/hdas/camera_wrist_right/color/image_raw/compressed",
}
JOINTS_PER_ARM, ACTION_DIM = 6, 14
RESET_SCRIPT = Path(__file__).resolve().parent / "reset_to_initial_position.sh"
JOINT_LIMITS_RAD = np.asarray(
    [(-1.20, 0.70), (-0.40, 2.80), (-2.35, 0.40), (-0.80, 1.20), (-0.70, 0.70), (-1.10, 1.30)],
    dtype=np.float32,
)
VIDEO_FPS = 30.0


def _safe_path_component(value: str) -> str:
    cleaned = re.sub(r"[^A-Za-z0-9._-]+", "_", value).strip("._")
    if not cleaned:
        raise ValueError("policy config must contain at least one letter or number")
    return cleaned


@dataclass(frozen=True)
class RolloutSession:
    root: Path

    @classmethod
    def create(cls, rollouts_dir: Path, policy_config: str, metadata: dict[str, object]) -> "RolloutSession":
        now = datetime.now().astimezone()
        root = (
            rollouts_dir.expanduser()
            / _safe_path_component(policy_config)
            / now.strftime("%Y%m%d")
            / f"run_{now.strftime('%H%M%S_%f')}"
        )
        root.mkdir(parents=True, exist_ok=False)
        (root / "run.json").write_text(json.dumps(metadata, indent=2, sort_keys=True) + "\n", encoding="utf-8")
        return cls(root=root)

    def demo_dir(self, demo_id: str) -> Path:
        path = self.root / demo_id
        path.mkdir(parents=True, exist_ok=True)
        return path


@dataclass(frozen=True)
class ObservationBundle:
    """One temporally aligned robot observation selected from ROS message stamps."""

    timestamp_s: float
    state: np.ndarray
    images: dict[str, np.ndarray]
    sample_timestamps_s: dict[str, float]


class RobotIO(Node):
    def __init__(self, wrist_transport: str, hold_hz: float, observation_buffer_s: float, video_recorder: "TrajectoryRecorder") -> None:
        super().__init__("r1lite_fastwam_ideal_chunk_inference")
        self.bridge = CvBridge()
        self.arm: dict[str, np.ndarray | None] = {"left": None, "right": None}
        self.arm_names: dict[str, list[str] | None] = {"left": None, "right": None}
        self.gripper: dict[str, float | None] = {"left": None, "right": None}
        self.images: dict[str, np.ndarray | None] = {"head": None, "left_wrist": None, "right_wrist": None}
        self._observation_buffer_s = observation_buffer_s
        self._last_observation_stamp = float('-inf')
        self._invalid_stamp_warned: set[str] = set()
        self.video_recorder = video_recorder
        self._samples: dict[str, deque[tuple[float, np.ndarray]]] = {
            "head": deque(),
            "left_wrist": deque(),
            "right_wrist": deque(),
            "left_arm": deque(),
            "right_arm": deque(),
            "left_gripper": deque(),
            "right_gripper": deque(),
        }
        self._command_lock = threading.Lock()
        self._last_targets: np.ndarray | None = None
        self._hold_enabled = threading.Event()
        self._hold_closed = threading.Event()
        self._hold_hz = hold_hz

        qos = QoSProfile(reliability=ReliabilityPolicy.BEST_EFFORT, history=HistoryPolicy.KEEP_LAST, depth=1)
        for side, topic in ARM_FEEDBACK_TOPICS.items():
            self.create_subscription(JointState, topic, self._arm_callback(side), qos)
        for side, topic in GRIPPER_FEEDBACK_TOPICS.items():
            self.create_subscription(JointState, topic, self._gripper_callback(side), qos)
        self.create_subscription(CompressedImage, HEAD_CAMERA_TOPIC, self._image_callback("head"), qos)
        wrist_topics = RAW_WRIST_TOPICS if wrist_transport == "raw" else COMPRESSED_WRIST_TOPICS
        wrist_type = Image if wrist_transport == "raw" else CompressedImage
        for name, topic in wrist_topics.items():
            self.create_subscription(wrist_type, topic, self._image_callback(name), qos)
        self.arm_publishers = {side: self.create_publisher(JointState, topic, 10) for side, topic in ARM_TARGET_TOPICS.items()}
        self.gripper_publishers = {side: self.create_publisher(JointState, topic, 10) for side, topic in GRIPPER_TARGET_TOPICS.items()}
        self._hold_thread = threading.Thread(target=self._hold_loop, daemon=True)
        self._hold_thread.start()

    @staticmethod
    def _message_timestamp_s(message: Image | CompressedImage | JointState) -> float:
        """Preserve publisher stamps; receipt times cannot establish capture alignment."""
        stamp = message.header.stamp
        timestamp_s = float(stamp.sec) + float(stamp.nanosec) * 1e-9
        return timestamp_s

    def _append_sample(self, name: str, timestamp_s: float, value: np.ndarray) -> None:
        if timestamp_s <= 0.0:
            if name not in self._invalid_stamp_warned:
                self.get_logger().warning(f"{name}: missing ROS timestamp; excluded from synchronized observations")
                self._invalid_stamp_warned.add(name)
            return
        samples = self._samples[name]
        if samples and timestamp_s <= samples[-1][0]:
            return
        samples.append((timestamp_s, value.copy()))
        cutoff = timestamp_s - self._observation_buffer_s
        while samples and samples[0][0] < cutoff:
            samples.popleft()

    def _arm_callback(self, side: str) -> Callable[[JointState], None]:
        def callback(message: JointState) -> None:
            if len(message.position) >= JOINTS_PER_ARM:
                self.arm[side] = np.asarray(message.position[:JOINTS_PER_ARM], dtype=np.float32)
                self.arm_names[side] = list(message.name[:JOINTS_PER_ARM])
                self._append_sample(f"{side}_arm", self._message_timestamp_s(message), self.arm[side])
        return callback

    def _gripper_callback(self, side: str) -> Callable[[JointState], None]:
        def callback(message: JointState) -> None:
            if message.position:
                self.gripper[side] = float(message.position[0])
                self._append_sample(
                    f"{side}_gripper",
                    self._message_timestamp_s(message),
                    np.asarray([self.gripper[side]], dtype=np.float32),
                )
        return callback

    def _image_callback(self, name: str) -> Callable[[Image | CompressedImage], None]:
        def callback(message: Image | CompressedImage) -> None:
            try:
                if isinstance(message, CompressedImage):
                    # Decode the compressed payload directly. This avoids the
                    # cv_bridge compressed path, which can reject valid JPEG
                    # format strings published by the head camera.
                    encoded = np.frombuffer(bytes(message.data), dtype=np.uint8)
                    bgr = cv2.imdecode(encoded, cv2.IMREAD_COLOR)
                    if bgr is None:
                        raise ValueError("OpenCV could not decode compressed image bytes")
                else:
                    bgr = self.bridge.imgmsg_to_cv2(message, desired_encoding="bgr8")
            except Exception as exc:  # noqa: BLE001
                self.get_logger().warning(f"Could not decode {name} image: {exc}")
                return

            self.images[name] = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
            timestamp_s = self._message_timestamp_s(message)
            self._append_sample(name, timestamp_s, self.images[name])
            if name == "head" and timestamp_s > 0.0 and self.video_recorder.active.is_set():
                try:
                    frame = self.trajectory_frame()
                    if frame is not None:
                        self.video_recorder.submit(timestamp_s, frame)
                except Exception as exc:  # noqa: BLE001
                    self.get_logger().warning(f"Could not build trajectory frame: {exc}")
        return callback

    def spin_for(self, seconds: float) -> None:
        deadline = time.monotonic() + seconds
        while rclpy.ok() and time.monotonic() < deadline:
            rclpy.spin_once(self, timeout_sec=min(0.01, deadline - time.monotonic()))

    def local_state(self) -> np.ndarray:
        if not all(x is not None for x in self.arm.values()) or not all(x is not None for x in self.gripper.values()):
            raise RuntimeError("Robot arm/gripper feedback is incomplete")
        return np.concatenate((self.arm["left"], self.arm["right"], np.asarray([self.gripper["left"], self.gripper["right"]], dtype=np.float32)))

    def synchronized_observation(self, max_skew_s: float, max_age_s: float) -> ObservationBundle | None:
        """Return the newest observation whose full timestamp span is ``max_skew_s``.

        A head-camera timestamp is used as the candidate capture time. For every
        other stream, the nearest buffered ROS-stamped sample is selected. The
        earliest and latest selected samples must be no more than ``max_skew_s``
        apart. This makes the state and all three images refer to one bounded
        time interval, rather than independently reading whatever callback
        arrived most recently.
        """
        if max_skew_s <= 0.0 or any(not samples for samples in self._samples.values()):
            return None

        now = self.get_clock().now().nanoseconds / 1e9
        for target_time, head in reversed(self._samples["head"]):
            if target_time <= self._last_observation_stamp:
                continue
            selected: dict[str, tuple[float, np.ndarray]] = {"head": (target_time, head)}
            for name, samples in self._samples.items():
                if name == "head":
                    continue
                nearest = min(samples, key=lambda item: abs(item[0] - target_time))
                selected[name] = nearest
            selected_times = [sample[0] for sample in selected.values()]
            if max(selected_times) - min(selected_times) > max_skew_s:
                continue
            if min(selected_times) < now - max_age_s or max(selected_times) > now:
                continue

            state = np.concatenate(
                (
                    selected["left_arm"][1],
                    selected["right_arm"][1],
                    selected["left_gripper"][1],
                    selected["right_gripper"][1],
                )
            ).astype(np.float32, copy=False)
            return ObservationBundle(
                timestamp_s=target_time,
                state=state.copy(),
                images={name: selected[name][1].copy() for name in ("head", "left_wrist", "right_wrist")},
                sample_timestamps_s={name: value[0] for name, value in selected.items()},
            )
        return None

    @staticmethod
    def _encode_jpeg(image: np.ndarray) -> str:
        success, encoded = cv2.imencode(".jpg", cv2.cvtColor(image, cv2.COLOR_RGB2BGR), (cv2.IMWRITE_JPEG_QUALITY, 90))
        if not success:
            raise RuntimeError("Failed to encode camera image")
        return base64.b64encode(encoded.tobytes()).decode("ascii")

    def trajectory_frame(self) -> np.ndarray | None:
        """Human-readable head-over-wrists frame for the rollout video."""
        head, left, right = self.images["head"], self.images["left_wrist"], self.images["right_wrist"]
        if head is None or left is None or right is None:
            return None
        if head.shape[0] < 640 or head.shape[1] < 640 or left.shape[:2] != (360, 640) or right.shape[:2] != (360, 640):
            return None
        h, w = head.shape[:2]
        head = head[(h - 640) // 2 : (h + 640) // 2, (w - 640) // 2 : (w + 640) // 2]
        wrists = np.concatenate(
            (
                cv2.resize(left, (320, 180), interpolation=cv2.INTER_AREA),
                cv2.resize(right, (320, 180), interpolation=cv2.INTER_AREA),
            ),
            axis=1,
        )
        return np.concatenate((head, wrists), axis=0).copy()

    # Payload for logging
    def make_payload(
        self,
        prompt: str,
        run_id: str,
        demo_id: str,
        chunk_id: int,
        observation: ObservationBundle,
    ) -> dict[str, object]:
        return {
            "prompt": prompt,
            "state": observation.state.tolist(),
            "head_image": self._encode_jpeg(observation.images["head"]),
            "left_wrist_image": self._encode_jpeg(observation.images["left_wrist"]),
            "right_wrist_image": self._encode_jpeg(observation.images["right_wrist"]),
            "run_id": run_id,
            "demo_id": demo_id,
            "chunk_id": chunk_id,
        }

    def prepare_targets(
        self,
        action: np.ndarray,
        max_delta: float,
        gripper_min: float,
        gripper_max: float,
        current_state: np.ndarray | None = None,
    ) -> np.ndarray:
        action = np.asarray(action, dtype=np.float32)
        if action.shape != (ACTION_DIM,) or not np.isfinite(action).all():
            raise ValueError(f"Expected one finite {ACTION_DIM}-D action, got {action.shape}")
        target = self.local_state() if current_state is None else np.asarray(current_state, dtype=np.float32).copy()
        target[:12] += np.clip(action[:12], -max_delta, max_delta)
        target[:6] = np.clip(target[:6], JOINT_LIMITS_RAD[:, 0], JOINT_LIMITS_RAD[:, 1])
        target[6:12] = np.clip(target[6:12], JOINT_LIMITS_RAD[:, 0], JOINT_LIMITS_RAD[:, 1])
        target[12:14] = np.clip(action[12:14], gripper_min, gripper_max)
        return target

    def _publish_targets_locked(self, target: np.ndarray) -> None:
        stamp = self.get_clock().now().to_msg()
        for arm_index, side in enumerate(("left", "right")):
            start = arm_index * JOINTS_PER_ARM
            arm = JointState()
            arm.header.stamp = stamp
            arm.name = self.arm_names[side] or []
            arm.position = target[start:start + JOINTS_PER_ARM].tolist()
            self.arm_publishers[side].publish(arm)
            gripper = JointState()
            gripper.header.stamp = stamp
            gripper.position = [float(target[12 + arm_index])]
            self.gripper_publishers[side].publish(gripper)

    def publish_targets(self, target: np.ndarray) -> None:
        target = np.asarray(target, dtype=np.float32).copy()
        with self._command_lock:
            self._last_targets = target
            self._publish_targets_locked(target)

    def publish_targets_if_enabled(self, target: np.ndarray, enabled: threading.Event) -> bool:
        target = np.asarray(target, dtype=np.float32).copy()
        with self._command_lock:
            if not enabled.is_set():
                return False
            self._last_targets = target
            self._publish_targets_locked(target)
        return True

    def start_hold(self) -> bool:
        with self._command_lock:
            if self._last_targets is None:
                try:
                    self._last_targets = self.local_state().copy()
                except RuntimeError:
                    return False
            self._publish_targets_locked(self._last_targets)
        self._hold_enabled.set()
        return True

    def hold_current_feedback(self) -> bool:
        with self._command_lock:
            try:
                self._last_targets = self.local_state().copy()
            except RuntimeError:
                return False
            self._publish_targets_locked(self._last_targets)
        self._hold_enabled.set()
        return True

    def stop_hold(self) -> None:
        self._hold_enabled.clear()
        with self._command_lock:
            pass

    def held_targets(self) -> np.ndarray | None:
        with self._command_lock:
            return None if self._last_targets is None else self._last_targets.copy()

    def _hold_loop(self) -> None:
        period = 1.0 / self._hold_hz
        while not self._hold_closed.is_set():
            if not self._hold_enabled.wait(timeout=period):
                continue
            tick_started = time.monotonic()
            with self._command_lock:
                if self._last_targets is not None:
                    self._publish_targets_locked(self._last_targets)
            remaining = period - (time.monotonic() - tick_started)
            if remaining > 0:
                self._hold_closed.wait(remaining)

    def close(self) -> None:
        self._hold_closed.set()
        self._hold_enabled.set()
        self._hold_thread.join()


class EnterHotkey:
    """Toggle inference from a background stdin reader."""

    def __init__(self, on_stop: Callable[[], object] | None = None) -> None:
        self.on_stop = on_stop
        self.running = threading.Event()
        self.closed = threading.Event()
        self.worker = threading.Thread(target=self._run, daemon=True)
        self.worker.start()

    def _run(self) -> None:
        while not self.closed.is_set():
            try:
                input()
            except EOFError:
                return
            if self.closed.is_set():
                return
            if self.running.is_set():
                self.running.clear()
                if self.on_stop is not None:
                    self.on_stop()
                print("\nSTOP requested. Holding latest feedback, then resetting.")
            else:
                self.running.set()
                print("\nSTART requested.")

    def close(self) -> None:
        self.closed.set()


def reset_to_initial_position(robot: RobotIO, execute: bool) -> None:
    """Run the shared arm, gripper, and torso reset script."""
    if not RESET_SCRIPT.is_file():
        raise FileNotFoundError(f"Reset script not found: {RESET_SCRIPT}")

    command = ["bash", str(RESET_SCRIPT)]
    print(f"Running reset script: {RESET_SCRIPT}")
    if not execute:
        subprocess.run(command, check=True)
        print("Reset finished. Press Enter to start inference again.")
        return

    handoff_pose = robot.held_targets()
    if handoff_pose is None:
        raise RuntimeError("No held feedback pose is available for reset handoff")
    command.extend(
        [
            "--execute",
            "--wait-for-takeover",
            "--hold-final",
            "--start-arm-gripper",
            *(f"{value:.9g}" for value in handoff_pose),
        ]
    )
    process = subprocess.Popen(
        command,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        bufsize=1,
    )
    inference_has_control = True

    def wait_for_marker(marker: str) -> None:
        assert process.stdout is not None
        for line in process.stdout:
            print(line, end="")
            if line.strip() == marker:
                return
        return_code = process.wait()
        raise RuntimeError(f"Reset process exited with code {return_code} before reporting {marker}")

    try:
        wait_for_marker("RESET_READY")
        process.send_signal(signal.SIGUSR1)
        wait_for_marker("RESET_TOOK_OVER")
        robot.stop_hold()
        inference_has_control = False

        wait_for_marker("RESET_HOLDING_FINAL")
        robot.spin_for(0.2)
        robot.publish_targets(robot.local_state())
        robot.start_hold()
        inference_has_control = True
        process.send_signal(signal.SIGINT)
        process.wait(timeout=5.0)
    finally:
        if process.poll() is None:
            process.send_signal(signal.SIGINT)
            try:
                process.wait(timeout=5.0)
            except subprocess.TimeoutExpired:
                process.kill()
                process.wait()
        if not inference_has_control:
            robot.spin_for(0.05)
            robot.publish_targets(robot.local_state())
            robot.start_hold()
        if process.returncode not in (0, -signal.SIGINT):
            raise subprocess.CalledProcessError(process.returncode, command)

    print("Reset finished. Press Enter to start inference again.")


def wall_time() -> str:
    return datetime.now().astimezone().isoformat(timespec="milliseconds")


def post_json(endpoint: str, payload: dict[str, object], timeout_s: float) -> dict[str, object]:
    request = Request(
        endpoint,
        data=json.dumps(payload).encode("utf-8"),
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    try:
        with urlopen(request, timeout=timeout_s) as response:  # noqa: S310 - explicit CLI server
            return json.loads(response.read().decode("utf-8"))
    except HTTPError as exc:
        raise RuntimeError(f"Policy server HTTP {exc.code}: {exc.read().decode(errors='replace')}") from exc
    except URLError as exc:
        raise RuntimeError(f"Could not reach policy server: {exc.reason}") from exc


def ask_num_demos() -> int:
    while True:
        try:
            value = int(input("How many demos do you want to run? ").strip())
        except ValueError:
            print("Enter a positive whole number.")
            continue
        if value > 0:
            return value
        print("Enter a positive whole number.")


class ActionStepLogger:
    """Persist execution traces on the robot without blocking control."""

    def __init__(self, session: RolloutSession) -> None:
        self.session = session
        self.pending: queue.Queue[dict[str, object] | None] = queue.Queue()
        self.records: dict[tuple[str, str], list[dict[str, object]]] = {}
        self.feedback_records: dict[tuple[str, str], list[dict[str, object]]] = {}
        self.records_lock = threading.Lock()
        self.worker = threading.Thread(target=self._run, daemon=True)
        self.worker.start()

    def submit(self, record: dict[str, object]) -> None:
        self.pending.put_nowait(record)

    def _run(self) -> None:
        while True:
            record = self.pending.get()
            try:
                if record is None:
                    return
                demo_dir = self.session.demo_dir(str(record["demo_id"]))
                measured_before = np.asarray(record["measured_before"], dtype=np.float32)
                target = np.asarray(record.pop("commanded_target"), dtype=np.float32)
                after = np.asarray(record.pop("measured_after"), dtype=np.float32)
                action = np.asarray(record.pop("model_action"), dtype=np.float32)
                ideal_base = np.asarray(record.pop("ideal_base_state"), dtype=np.float32)
                summary = {
                    "step": int(record["action_step"]),
                    "chunk": int(record["chunk_id"]),
                    "target": np.round(target, 3).tolist(),
                    "measured": np.round(after, 3).tolist(),
                    "error": np.round(target - after, 3).tolist(),
                    "delta": np.round(action[:12], 3).tolist(),
                    "ideal_base": np.round(ideal_base, 3).tolist(),
                }
                feedback = {
                    "chunk_id": int(record["chunk_id"]),
                    "action_step": int(record["action_step"]),
                    "client_time": str(record["client_time"]),
                    "execution_mode": "ideal_chunk_accumulation",
                    "model_action": action.tolist(),
                    "ideal_base_position": ideal_base.tolist(),
                    "commanded_position": target.tolist(),
                    "measured_before": measured_before.tolist(),
                    "measured_after": after.tolist(),
                    "delta_error": (after - target).tolist(),
                    "published": bool(record["published"]),
                    "tick_duration_s": float(record["tick_duration_s"]),
                }
                key = (str(record["run_id"]), str(record["demo_id"]))
                with self.records_lock:
                    self.records.setdefault(key, []).append(summary)
                    self.feedback_records.setdefault(key, []).append(feedback)
                    summaries = list(self.records[key])
                    feedback_steps = list(self.feedback_records[key])
                (demo_dir / "action_summary.json").write_text(
                    json.dumps({"steps": summaries}, separators=(",", ":")) + "\n", encoding="utf-8"
                )
                (demo_dir / "command_feedback.json").write_text(
                    json.dumps({"steps": feedback_steps}, indent=2) + "\n",
                    encoding="utf-8",
                )
            except Exception as exc:  # noqa: BLE001
                print(f"ACTION_LOG_FAILED: {exc}")
            finally:
                self.pending.task_done()

    def close(self) -> None:
        self.pending.join()
        with self.records_lock:
            grouped = {key: list(value) for key, value in self.records.items()}
        for (run_id, demo_id), summaries in grouped.items():
            self._write_graph(demo_id, summaries)
        self.pending.put(None)
        self.worker.join()

    def _write_graph(self, demo_id: str, summaries: list[dict[str, object]]) -> None:
        if not summaries:
            return
        targets = np.asarray([s["target"] for s in summaries], dtype=np.float32)
        measured = np.asarray([s["measured"] for s in summaries], dtype=np.float32)
        canvas = np.full((1640, 1400, 3), 255, dtype=np.uint8)
        font = cv2.FONT_HERSHEY_SIMPLEX
        cv2.putText(canvas, "Target: blue line | measured: orange dots | x: executed action step", (40, 32), font, 0.7, (0, 0, 0), 2)
        names = [f"Left joint {i + 1} (rad)" for i in range(6)] + [f"Right joint {i + 1} (rad)" for i in range(6)] + ["Left gripper (native units)", "Right gripper (native units)"]
        for dim, name in enumerate(names):
            # Separate dimensions keep gripper units from hiding arm errors.
            row, col = divmod(dim, 2)
            x0, y0 = 85 + col * 700, 90 + row * 220
            x1, y1 = x0 + 570, y0 + 140
            both = np.concatenate((targets[:, dim], measured[:, dim]))
            margin = max(float(np.ptp(both)) * 0.05, 0.001)
            lo, hi = float(both.min()) - margin, float(both.max()) + margin
            cv2.putText(canvas, name, (x0, y0 - 15), font, 0.55, (0, 0, 0), 1)
            cv2.rectangle(canvas, (x0, y0), (x1, y1), (100, 100, 100), 1)
            cv2.putText(canvas, f"{hi:.3f}", (x0 - 80, y0 + 8), font, 0.4, (0, 0, 0), 1)
            cv2.putText(canvas, f"{lo:.3f}", (x0 - 80, y1), font, 0.4, (0, 0, 0), 1)
            for values, color, dots in ((targets, (200, 80, 20), False), (measured, (0, 130, 230), True)):
                points = np.asarray([
                    (x0 + int(i * (x1 - x0) / max(len(summaries) - 1, 1)),
                     y1 - int((float(value) - lo) * (y1 - y0) / (hi - lo)))
                    for i, value in enumerate(values[:, dim])
                ], dtype=np.int32)
                if not dots:
                    cv2.polylines(canvas, [points], False, color, 2)
                if dots or len(points) == 1:
                    for point in points:
                        cv2.circle(canvas, tuple(point), 2, color, -1)
            cv2.putText(canvas, "1", (x0, y1 + 20), font, 0.4, (0, 0, 0), 1)
            cv2.putText(canvas, str(len(summaries)), (x1 - 30, y1 + 20), font, 0.4, (0, 0, 0), 1)
        out = self.session.demo_dir(demo_id) / "action_summary.png"
        out.parent.mkdir(parents=True, exist_ok=True)
        cv2.imwrite(str(out), canvas)


class ObservationImageLogger:
    """Save the exact JPEG camera payloads sent to the server in the background."""

    IMAGE_FIELDS = {
        "head_image": "head.jpg",
        "left_wrist_image": "left_wrist.jpg",
        "right_wrist_image": "right_wrist.jpg",
    }

    def __init__(self, observations_dir: Path) -> None:
        self.observations_dir = observations_dir.expanduser()
        self.pending: queue.Queue[tuple[str, str, int, dict[str, object]] | None] = queue.Queue()
        self.worker = threading.Thread(target=self._run, daemon=True)
        self.worker.start()

    def submit(self, run_id: str, demo_id: str, chunk_id: int, payload: dict[str, object]) -> None:
        self.pending.put_nowait((run_id, demo_id, chunk_id, payload.copy()))

    def _run(self) -> None:
        while True:
            item = self.pending.get()
            try:
                if item is None:
                    return
                run_id, demo_id, chunk_id, payload = item
                out = self.observations_dir / run_id / demo_id / "images" / f"chunk_{chunk_id:04d}"
                out.mkdir(parents=True, exist_ok=True)
                for field, filename in self.IMAGE_FIELDS.items():
                    (out / filename).write_bytes(base64.b64decode(str(payload[field]), validate=True))
            except Exception as exc:  # noqa: BLE001
                print(f"OBSERVATION_IMAGE_LOG_FAILED: {exc}")
            finally:
                self.pending.task_done()

    def close(self) -> None:
        self.pending.join()
        self.pending.put(None)
        self.worker.join()


class TrajectoryRecorder:
    """Encode the rollout mosaic as H.264/avc1 without blocking ROS callbacks."""

    def __init__(self) -> None:
        self.pending: queue.Queue[tuple[str, object] | None] = queue.Queue(maxsize=180)
        self.active = threading.Event()
        self.dropped_frames = 0
        self._current_demo_dir: Path | None = None
        self.worker = threading.Thread(target=self._run, daemon=True)
        self.worker.start()

    def start_demo(self, demo_dir: Path) -> None:
        self.dropped_frames = 0
        self.active.set()
        self.pending.put(("start", demo_dir))

    def submit(self, timestamp_s: float, frame: np.ndarray) -> None:
        if not self.active.is_set():
            return
        try:
            self.pending.put_nowait(("frame", (timestamp_s, frame)))
        except queue.Full:
            self.dropped_frames += 1

    def stop_demo(self) -> None:
        if self.active.is_set():
            self.active.clear()
            self.pending.put(("stop", None))

    def _run(self) -> None:
        process: subprocess.Popen[bytes] | None = None
        last_frame: np.ndarray | None = None
        next_frame_time: float | None = None
        encoding_disabled = False

        def close_process() -> None:
            nonlocal process, last_frame, next_frame_time
            if process is not None:
                try:
                    if last_frame is not None:
                        process.stdin.write(cv2.cvtColor(last_frame, cv2.COLOR_RGB2BGR).tobytes())  # type: ignore[union-attr]
                    process.stdin.close()  # type: ignore[union-attr]
                    process.wait(timeout=20)
                except Exception as exc:  # noqa: BLE001
                    print(f"VIDEO_ENCODE_FAILED: {exc}")
                    if process.poll() is None:
                        process.kill()
            process, last_frame, next_frame_time = None, None, None

        while True:
            item = self.pending.get()
            try:
                if item is None:
                    close_process()
                    return
                operation, value = item
                if operation == "start":
                    close_process()
                    self._current_demo_dir = value  # type: ignore[assignment]
                    encoding_disabled = False
                    continue
                if operation == "stop":
                    close_process()
                    continue
                timestamp_s, frame = value  # type: ignore[misc]
                if encoding_disabled:
                    continue
                if process is None:
                    ffmpeg = shutil.which("ffmpeg")
                    if ffmpeg is None:
                        print("VIDEO_ENCODE_FAILED: ffmpeg is not installed; trajectory.mp4 was not written")
                        encoding_disabled = True
                        continue
                    height, width = frame.shape[:2]
                    assert self._current_demo_dir is not None
                    output = self._current_demo_dir / "trajectory.mp4"
                    process = subprocess.Popen(
                        [ffmpeg, "-y", "-loglevel", "error", "-f", "rawvideo", "-pix_fmt", "bgr24", "-s", f"{width}x{height}", "-r", str(VIDEO_FPS), "-i", "-", "-an", "-c:v", "libx264", "-tag:v", "avc1", "-pix_fmt", "yuv420p", str(output)],
                        stdin=subprocess.PIPE,
                    )
                    assert process.stdin is not None
                    process.stdin.write(cv2.cvtColor(frame, cv2.COLOR_RGB2BGR).tobytes())
                    last_frame = frame
                    next_frame_time = float(timestamp_s) + 1.0 / VIDEO_FPS
                    continue
                assert process is not None and process.stdin is not None and next_frame_time is not None
                while last_frame is not None and next_frame_time <= timestamp_s:
                    process.stdin.write(cv2.cvtColor(last_frame, cv2.COLOR_RGB2BGR).tobytes())
                    next_frame_time += 1.0 / VIDEO_FPS
                last_frame = frame
            except Exception as exc:  # noqa: BLE001
                print(f"VIDEO_ENCODE_FAILED: {exc}")
            finally:
                self.pending.task_done()

    def close(self) -> None:
        self.stop_demo()
        self.pending.join()
        self.pending.put(None)
        self.worker.join()


class ClientObservationRecorder:
    """Synchronous low-rate event recorder rooted on the robot filesystem."""

    def __init__(self, session: RolloutSession) -> None:
        self.session = session

    def append_event(self, demo_id: str, event: str, details: dict[str, object] | None = None) -> None:
        demo_dir = self.session.demo_dir(demo_id)
        record = {"event": event, "client_time": wall_time(), "details": details or {}}
        with (demo_dir / "events.jsonl").open("a", encoding="utf-8") as stream:
            stream.write(json.dumps(record, separators=(",", ":")) + "\n")


def request_actions(endpoint: str, payload: dict[str, object], timeout_s: float) -> np.ndarray:
    result = post_json(endpoint, payload, timeout_s)
    actions = np.asarray(result.get("actions"), dtype=np.float32)
    if actions.ndim != 2 or actions.shape[1] != ACTION_DIM or not np.isfinite(actions).all():
        raise RuntimeError(f"Policy returned invalid actions: {actions.shape}")
    return actions


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--server", required=True, help="e.g. http://192.168.1.10:8000")
    parser.add_argument("--prompt", required=True)
    parser.add_argument("--execute", action="store_true", help="Publish targets; default is dry-run")
    parser.add_argument(
        "--control-hz",
        type=float,
        default=20.0,
        help="Action execution rate (default: the 20 Hz rate used by the processed R1 Lite training data)",
    )
    parser.add_argument(
        "--replan-steps",
        type=int,
        default=None,
        help="Actions to execute per query; default executes the complete horizon returned by the trained server",
    )
    parser.add_argument("--request-timeout", type=float, default=30.0)
    parser.add_argument(
        "--rollouts-dir",
        type=Path,
        default=Path(__file__).resolve().parents[2] / "rollouts" / "r1lite",
        help="Robot-local root for dated FastWAM rollout sessions",
    )
    parser.add_argument(
        "--observations-dir",
        type=Path,
        default=Path(__file__).resolve().parents[2] / "observations",
        help="Robot-local root for exact per-query camera payloads",
    )
    parser.add_argument(
        "--policy-config",
        default="fastwam",
        help="Directory name identifying the served FastWAM policy/config",
    )
    parser.add_argument("--max-joint-delta", type=float, default=0.25)
    parser.add_argument("--gripper-min", type=float, default=0.0)
    parser.add_argument("--gripper-max", type=float, default=100.0)
    parser.add_argument("--wrist-transport", choices=("raw", "compressed"), default="raw")
    parser.add_argument(
        "--max-observation-skew-ms",
        type=float,
        default=50.0,
        help="Maximum earliest-to-latest ROS-stamp span within one model observation (default: 50 ms)",
    )
    parser.add_argument(
        "--max-observation-age-ms", type=float, default=250.0,
        help="Oldest permitted sample age at selection, using the ROS clock (default: 250 ms)",
    )
    parser.add_argument(
        "--observation-buffer-s",
        type=float,
        default=3.0,
        help="Timestamped ROS history retained while finding a synchronized observation (default: 3 s)",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if (
        args.control_hz <= 0
        or (args.replan_steps is not None and args.replan_steps <= 0)
        or args.max_joint_delta <= 0
        or args.gripper_min > args.gripper_max
        or args.max_observation_skew_ms <= 0
        or args.observation_buffer_s <= 0
        or args.max_observation_age_ms <= 0
    ):
        raise ValueError("Invalid control, replan, delta, or gripper limits")
    server = args.server.rstrip("/")
    endpoint, period = server + "/infer", 1.0 / args.control_hz
    num_demos = ask_num_demos()
    session = RolloutSession.create(
        args.rollouts_dir,
        args.policy_config,
        {
            "policy_config": args.policy_config,
            "prompt": args.prompt,
            "num_demos": num_demos,
            "execute": args.execute,
            "control_hz": args.control_hz,
            "execution_mode": "ideal_chunk_accumulation",
            "max_observation_skew_ms": args.max_observation_skew_ms,
            "max_observation_age_ms": args.max_observation_age_ms,
            "created_at": wall_time(),
            "host": os.uname().nodename,
        },
    )
    run_response = post_json(
        server + "/runs/start",
        {
            "prompt": args.prompt,
            "num_demos": num_demos,
            "execute": args.execute,
            "control_hz": args.control_hz,
        },
        args.request_timeout,
    )
    run_id = str(run_response["run_id"])
    observation_recorder = ClientObservationRecorder(session)
    action_queue: deque[tuple[int, int, np.ndarray]] = deque()
    chunk_start_states: dict[int, np.ndarray] = {}
    rclpy.init()
    video_recorder = TrajectoryRecorder()
    robot = RobotIO(args.wrist_transport, max(args.control_hz, 20.0), args.observation_buffer_s, video_recorder)
    step_logger = ActionStepLogger(session)
    image_logger = ObservationImageLogger(args.observations_dir)
    print("EXECUTE ENABLED" if args.execute else "DRY RUN: add --execute to move the robot")
    print(
        "IDEAL CHUNK ACCUMULATION: first target uses chunk-start feedback; "
        "later targets add residuals to the preceding commanded position."
    )
    replan_description = args.replan_steps if args.replan_steps is not None else "full trained horizon"
    print(
        f"FastWAM server: {endpoint}; replan steps: {replan_description}; "
        f"max observation skew: {args.max_observation_skew_ms:.1f} ms"
    )
    print(f"Recording {num_demos} demos under {session.root}.")
    print("Press Enter to start demo_1. Press Enter again to stop and reset.")

    hotkey = EnterHotkey(on_stop=robot.hold_current_feedback if args.execute else None)
    inference_started = False
    completed_demos = 0
    chunk_id = 0
    ideal_chunk_id: int | None = None
    ideal_commanded_state: np.ndarray | None = None

    def current_demo_id() -> str:
        return f"demo_{completed_demos + 1}"

    def log_event(event: str, details: dict[str, object] | None = None) -> None:
        observation_recorder.append_event(current_demo_id(), event, details)

    try:
        while rclpy.ok():
            if not hotkey.running.is_set():
                action_queue.clear()
                chunk_start_states.clear()
                ideal_chunk_id = None
                ideal_commanded_state = None
                if inference_started:
                    video_recorder.stop_demo()
                    log_event("demo_stopped", {"chunks_queried": chunk_id})
                    reset_ok = False
                    for attempt in (1, 2):
                        try:
                            reset_to_initial_position(robot, args.execute)
                            reset_ok = True
                            break
                        except Exception as exc:  # noqa: BLE001
                            print(f"Reset attempt {attempt}/2 failed: {exc}")
                            if args.execute:
                                robot.hold_current_feedback()
                            if attempt == 1:
                                robot.spin_for(1.0)
                    if not reset_ok:
                        print(
                            "Automatic reset failed twice; holding last feedback instead of "
                            "resetting. Press Enter to try starting inference again."
                        )
                    log_event("reset_finished" if reset_ok else "reset_failed")
                    inference_started = False
                    completed_demos += 1
                    if completed_demos >= num_demos:
                        print(f"Completed requested {num_demos} demos in {run_id}.")
                        break
                    print(f"Press Enter to start {current_demo_id()}.")
                robot.spin_for(0.05)
                continue

            if not inference_started:
                inference_started = True
                chunk_id = 0
                video_recorder.start_demo(session.demo_dir(current_demo_id()))
                log_event("demo_started")
                print(f"{current_demo_id()} inference started.")

            if not action_queue:
                robot.spin_for(0.10)
                observation = robot.synchronized_observation(
                    args.max_observation_skew_ms / 1000.0, args.max_observation_age_ms / 1000.0
                )
                if observation is None:
                    print(
                        "Waiting for a temporally synchronized state + three-camera observation "
                        f"within {args.max_observation_skew_ms:.1f} ms, "
                        f"no older than {args.max_observation_age_ms:.1f} ms..."
                    )
                    continue
                chunk_id += 1
                started = time.monotonic()
                payload = robot.make_payload(args.prompt, run_id, current_demo_id(), chunk_id, observation)
                image_logger.submit(run_id, current_demo_id(), chunk_id, payload)
                robot._last_observation_stamp = observation.timestamp_s
                log_event("observation_selected", {
                    "chunk_id": chunk_id, "timestamps_s": observation.sample_timestamps_s,
                    "span_ms": 1000.0 * (max(observation.sample_timestamps_s.values()) - min(observation.sample_timestamps_s.values())),
                })
                actions = request_actions(
                    endpoint,
                    payload,
                    args.request_timeout,
                )
                n_execute = len(actions) if args.replan_steps is None else min(args.replan_steps, len(actions))
                if not hotkey.running.is_set():
                    print("Action chunk returned after STOP; discarding it.")
                    continue
                action_queue.extend(
                    (chunk_id, action_step, action.copy())
                    for action_step, action in enumerate(actions[:n_execute], start=1)
                )
                chunk_start_states[chunk_id] = observation.state.copy()
                print(f"Received {len(actions)} actions in {time.monotonic() - started:.3f}s; executing {n_execute}.")

                # Refresh feedback accumulated during the blocking request.
                robot.spin_for(0.05)

            if not hotkey.running.is_set():
                action_queue.clear()
                chunk_start_states.clear()
                ideal_chunk_id = None
                ideal_commanded_state = None
                continue

            tick_started = time.monotonic()
            queued_chunk_id, action_step, action = action_queue.popleft()
            measured_before = robot.local_state().copy()
            if action_step == 1:
                if queued_chunk_id not in chunk_start_states:
                    raise RuntimeError(f"Missing observation state for chunk {queued_chunk_id}")
                ideal_chunk_id = queued_chunk_id
                ideal_commanded_state = chunk_start_states.pop(queued_chunk_id).copy()
                log_event(
                    "ideal_chunk_base_initialized",
                    {
                        "chunk_id": queued_chunk_id,
                        "source": "model_observation",
                        "state": ideal_commanded_state.tolist(),
                        "measured_before_execution": measured_before.tolist(),
                    },
                )
            elif ideal_chunk_id != queued_chunk_id or ideal_commanded_state is None:
                raise RuntimeError(f"Missing ideal commanded state for chunk {queued_chunk_id}, step {action_step}")
            ideal_base_state = ideal_commanded_state.copy()
            target = robot.prepare_targets(
                action,
                args.max_joint_delta,
                args.gripper_min,
                args.gripper_max,
                current_state=ideal_base_state,
            )
            ideal_commanded_state = target.copy()
            published = False
            if args.execute:
                if not robot.publish_targets_if_enabled(target, hotkey.running):
                    action_queue.clear()
                    continue
                robot.start_hold()
                published = True
            else:
                print("[dry-run] target=", np.array2string(target, precision=4))
            remaining = period - (time.monotonic() - tick_started)
            if remaining > 0:
                robot.spin_for(remaining)
            measured_after = robot.local_state().copy()
            step_logger.submit(
                {
                    "run_id": run_id,
                    "demo_id": current_demo_id(),
                    "chunk_id": queued_chunk_id,
                    "action_step": action_step,
                    "client_time": wall_time(),
                    "model_action": action.tolist(),
                    "ideal_base_state": ideal_base_state.tolist(),
                    "measured_before": measured_before.tolist(),
                    "commanded_target": target.tolist(),
                    "measured_after": measured_after.tolist(),
                    "published": published,
                    "tick_duration_s": time.monotonic() - tick_started,
                }
            )
    except KeyboardInterrupt:
        print("\nEmergency exit requested; automatic reset is skipped.")
        if inference_started:
            try:
                log_event("emergency_exit", {"chunks_queried": chunk_id})
            except Exception as exc:  # noqa: BLE001
                print(f"EVENT_LOG_FAILED: {exc}")
        failed = False
    except Exception:  # noqa: BLE001
        traceback.print_exc()
        print("\nUnhandled error in the control loop; exiting.")
        failed = True
    else:
        failed = False
    finally:
        hotkey.close()
        video_recorder.close()
        step_logger.close()
        image_logger.close()
        robot.close()
        robot.destroy_node()
        rclpy.shutdown()

    # EnterHotkey's daemon may still be blocked in input() while holding
    # stdin's internal lock, so exit directly after all cleanup is complete.
    os._exit(1 if failed else 0)


if __name__ == "__main__":
    main()
