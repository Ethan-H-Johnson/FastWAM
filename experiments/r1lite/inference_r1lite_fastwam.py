#!/usr/bin/env python3
"""Run a workstation-hosted FastWAM policy from a Galaxea R1 Lite ROS 2 PC.

The client supports absolute or relative arm outputs, publishes absolute
gripper targets, and holds the last target while the next chunk arrives.
The selected mode must match the action contract reported by the server.
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
import h5py
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


class CommandPublishLogger:
    """Write every ROS target publication without blocking the control thread."""

    def __init__(self, session: RolloutSession) -> None:
        self.path = session.root / "command_publications.jsonl"
        self.pending: queue.Queue[dict[str, object] | None] = queue.Queue(maxsize=10000)
        self.dropped = 0
        self.worker = threading.Thread(target=self._run, name="command-publish-logger", daemon=True)
        self.worker.start()

    def submit(
        self,
        source: str,
        target: np.ndarray,
        demo_id: str | None,
        chunk_id: int | None,
        action_step: int | None,
    ) -> None:
        record = {
            "monotonic_ns": time.monotonic_ns(),
            "wall_time": wall_time(),
            "source": source,
            "demo_id": demo_id,
            "chunk_id": chunk_id,
            "action_step": action_step,
            "target": np.asarray(target, dtype=np.float32).tolist(),
        }
        try:
            self.pending.put_nowait(record)
        except queue.Full:
            self.dropped += 1

    def _run(self) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        with self.path.open("a", encoding="utf-8", buffering=1) as stream:
            while True:
                record = self.pending.get()
                try:
                    if record is None:
                        return
                    stream.write(json.dumps(record, separators=(",", ":")) + "\n")
                finally:
                    self.pending.task_done()

    def close(self) -> None:
        self.pending.join()
        if self.dropped:
            print(f"COMMAND_PUBLISH_LOG_DROPPED: {self.dropped}")
        self.pending.put(None)
        self.worker.join()


class RobotIO(Node):
    def __init__(
        self,
        wrist_transport: str,
        hold_hz: float,
        observation_buffer_s: float,
        video_recorder: "TrajectoryRecorder",
        publish_logger: CommandPublishLogger | None = None,
    ) -> None:
        super().__init__("r1lite_fastwam_inference")
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
        self._publish_logger = publish_logger
        self._active_demo_id: str | None = None
        self._last_target_chunk_id: int | None = None
        self._last_target_action_step: int | None = None
        self._arm_feedback_sequence = {"left": 0, "right": 0}

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
                self._arm_feedback_sequence[side] += 1
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
        gripper_min: float,
        gripper_max: float,
        current_state: np.ndarray | None = None,
    ) -> np.ndarray:
        action = np.asarray(action, dtype=np.float32)
        if action.shape != (ACTION_DIM,) or not np.isfinite(action).all():
            raise ValueError(f"Expected one finite {ACTION_DIM}-D action, got {action.shape}")
        target = self.local_state() if current_state is None else np.asarray(current_state, dtype=np.float32).copy()
        target[:12] += action[:12]
        target[:6] = np.clip(target[:6], JOINT_LIMITS_RAD[:, 0], JOINT_LIMITS_RAD[:, 1])
        target[6:12] = np.clip(target[6:12], JOINT_LIMITS_RAD[:, 0], JOINT_LIMITS_RAD[:, 1])
        target[12:14] = np.clip(action[12:14], gripper_min, gripper_max)
        return target

    def prepare_absolute_targets(
        self,
        target: np.ndarray,
        gripper_min: float,
        gripper_max: float,
    ) -> np.ndarray:
        """Clip an already-absolute 14-D target to safety limits, with no add."""
        target = np.asarray(target, dtype=np.float32).copy()
        if target.shape != (ACTION_DIM,) or not np.isfinite(target).all():
            raise ValueError(f"Expected one finite {ACTION_DIM}-D target, got {target.shape}")
        target[:6] = np.clip(target[:6], JOINT_LIMITS_RAD[:, 0], JOINT_LIMITS_RAD[:, 1])
        target[6:12] = np.clip(target[6:12], JOINT_LIMITS_RAD[:, 0], JOINT_LIMITS_RAD[:, 1])
        target[12:14] = np.clip(target[12:14], gripper_min, gripper_max)
        return target

    def set_active_demo(self, demo_id: str) -> None:
        with self._command_lock:
            self._active_demo_id = demo_id

    def wait_for_right_arm_convergence(
        self,
        target: np.ndarray,
        enabled: threading.Event,
        minimum_wait_s: float,
        timeout_s: float,
        tolerance_rad: float,
        consecutive_samples: int,
    ) -> tuple[bool, float, float]:
        """Wait for fresh right-arm feedback to converge on one published target."""
        started = time.monotonic()
        deadline = started + max(minimum_wait_s, timeout_s)
        last_sequence = self._arm_feedback_sequence["right"]
        consecutive = 0
        max_error = float("inf")

        while rclpy.ok() and enabled.is_set():
            now = time.monotonic()
            if now >= deadline:
                break
            self.spin_for(min(0.01, deadline - now))
            if time.monotonic() - started < minimum_wait_s:
                continue
            sequence = self._arm_feedback_sequence["right"]
            if sequence == last_sequence:
                continue
            last_sequence = sequence
            measured = self.local_state()[6:12]
            max_error = float(np.max(np.abs(measured - target[6:12])))
            consecutive = consecutive + 1 if max_error <= tolerance_rad else 0
            if consecutive >= consecutive_samples:
                return True, time.monotonic() - started, max_error

        if np.isinf(max_error):
            max_error = float(np.max(np.abs(self.local_state()[6:12] - target[6:12])))
        return False, time.monotonic() - started, max_error

    def _publish_targets_locked(self, target: np.ndarray, source: str) -> None:
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
        if self._publish_logger is not None:
            self._publish_logger.submit(
                source,
                target,
                self._active_demo_id,
                self._last_target_chunk_id,
                self._last_target_action_step,
            )

    def publish_targets(self, target: np.ndarray) -> None:
        target = np.asarray(target, dtype=np.float32).copy()
        with self._command_lock:
            self._last_targets = target
            self._last_target_chunk_id = None
            self._last_target_action_step = None
            self._publish_targets_locked(target, "direct")

    def publish_targets_if_enabled(
        self,
        target: np.ndarray,
        enabled: threading.Event,
        chunk_id: int | None = None,
        action_step: int | None = None,
    ) -> bool:
        target = np.asarray(target, dtype=np.float32).copy()
        with self._command_lock:
            if not enabled.is_set():
                return False
            self._last_targets = target
            self._last_target_chunk_id = chunk_id
            self._last_target_action_step = action_step
            self._publish_targets_locked(target, "action_step")
        return True

    def start_hold(self) -> bool:
        with self._command_lock:
            if self._last_targets is None:
                try:
                    self._last_targets = self.local_state().copy()
                except RuntimeError:
                    return False
            self._publish_targets_locked(self._last_targets, "hold_start")
        self._hold_enabled.set()
        return True

    def hold_current_feedback(self) -> bool:
        with self._command_lock:
            try:
                self._last_targets = self.local_state().copy()
            except RuntimeError:
                return False
            self._last_target_chunk_id = None
            self._last_target_action_step = None
            self._publish_targets_locked(self._last_targets, "hold_current_feedback")
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
                    self._publish_targets_locked(self._last_targets, "hold_thread")
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
                summary = {
                    "step": int(record["action_step"]),
                    "chunk": int(record["chunk_id"]),
                    "target": np.round(target, 3).tolist(),
                    "measured": np.round(after, 3).tolist(),
                    "error": np.round(target - after, 3).tolist(),
                    "model_arm_output": np.round(action[:12], 3).tolist(),
                    "action_mode": str(record["action_mode"]),
                }
                feedback = {
                    "chunk_id": int(record["chunk_id"]),
                    "action_step": int(record["action_step"]),
                    "client_time": str(record["client_time"]),
                    "model_action": action.tolist(),
                    "action_mode": str(record["action_mode"]),
                    "commanded_position": target.tolist(),
                    "measured_before": measured_before.tolist(),
                    "measured_after": after.tolist(),
                    "delta_error": (after - target).tolist(),
                    "published": bool(record["published"]),
                    "tick_duration_s": float(record["tick_duration_s"]),
                    "convergence_enabled": bool(record.get("convergence_enabled", False)),
                    "settled": record.get("settled"),
                    "settle_elapsed_s": float(record.get("settle_elapsed_s", 0.0)),
                    "right_arm_max_error_rad": (
                        None
                        if record.get("right_arm_max_error_rad") is None
                        else float(record["right_arm_max_error_rad"])
                    ),
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


def request_actions(
    endpoint: str,
    payload: dict[str, object],
    timeout_s: float,
    expected_action_mode: str,
) -> np.ndarray:
    result = post_json(endpoint, payload, timeout_s)
    server_action_mode = str(result.get("action_semantics", "relative"))
    if server_action_mode != expected_action_mode:
        raise RuntimeError(
            "Client/server action-mode mismatch: "
            f"client={expected_action_mode!r}, server={server_action_mode!r}"
        )
    actions = np.asarray(result.get("actions"), dtype=np.float32)
    if actions.ndim != 2 or actions.shape[1] != ACTION_DIM or not np.isfinite(actions).all():
        raise RuntimeError(f"Policy returned invalid actions: {actions.shape}")
    return actions


def request_actions_while_spinning(
    robot: RobotIO,
    endpoint: str,
    payload: dict[str, object],
    timeout_s: float,
    expected_action_mode: str,
) -> np.ndarray:
    """Run the blocking HTTP request without stopping ROS feedback callbacks.

    Target holding remains owned by ``RobotIO._hold_loop``. The main thread
    keeps spinning this node until the inference worker finishes, so camera and
    joint feedback continue to update throughout the server request.
    """
    result_queue: queue.Queue[tuple[np.ndarray | None, BaseException | None]] = queue.Queue(maxsize=1)

    def infer() -> None:
        try:
            result_queue.put(
                (request_actions(endpoint, payload, timeout_s, expected_action_mode), None)
            )
        except BaseException as exc:  # Propagate the original request failure on the main thread.
            result_queue.put((None, exc))

    worker = threading.Thread(target=infer, name="fastwam-inference-request", daemon=True)
    worker.start()
    while worker.is_alive():
        robot.spin_for(0.01)
        if not rclpy.ok():
            time.sleep(0.01)
    worker.join()

    actions, error = result_queue.get_nowait()
    if error is not None:
        raise error
    assert actions is not None
    return actions


def load_replay_actions(path: Path, demo_name: str) -> tuple[np.ndarray, np.ndarray, np.ndarray, str]:
    """Load one HDF5 demo's actions in two forms:

    - relative: arm deltas from ``actions``, absolute gripper targets from
      ``absolute_actions`` (the model's own convention; see
      serve_fastwam_r1lite.py's action_semantics).
    - absolute: the recorded ``absolute_actions`` targets exactly as
      commanded during collection, for all 14 dims.
    """
    demo_name = demo_name if demo_name.startswith("demo_") else f"demo_{demo_name}"
    with h5py.File(path, "r") as file:
        key = f"data/{demo_name}"
        if key not in file:
            available = sorted(file.get("data", {}).keys())
            raise KeyError(f"{demo_name!r} is not in {path}. Available demos: {available}")
        demo = file[key]
        if "actions" not in demo or "absolute_actions" not in demo or "joint_states" not in demo:
            raise KeyError(f"{key} must contain actions, absolute_actions, and joint_states")
        actions = np.asarray(demo["actions"], dtype=np.float32)
        absolute_actions = np.asarray(demo["absolute_actions"], dtype=np.float32)
        initial_state = np.asarray(demo["joint_states"][0], dtype=np.float32)
        task = str(demo.attrs.get("task_description", demo.attrs.get("subtask_key", "unknown")))

    if actions.shape != absolute_actions.shape or actions.ndim != 2 or actions.shape[1] < ACTION_DIM:
        raise ValueError(f"Expected actions/absolute_actions shaped [T, >={ACTION_DIM}], got {actions.shape}")
    if initial_state.shape != (ACTION_DIM,):
        raise ValueError(f"Expected a {ACTION_DIM}-D initial joint state, got {initial_state.shape}")

    absolute_targets = absolute_actions[:, :ACTION_DIM].copy()
    relative_actions = actions[:, :ACTION_DIM].copy()
    relative_actions[:, 12:14] = absolute_targets[:, 12:14]
    if not np.isfinite(relative_actions).all() or not np.isfinite(absolute_targets).all() or not np.isfinite(initial_state).all():
        raise ValueError("The selected demo contains non-finite state or action values")
    return relative_actions, absolute_targets, initial_state, task


def move_to_replay_start(
    robot: RobotIO,
    hotkey: EnterHotkey,
    target: np.ndarray,
    args: argparse.Namespace,
) -> bool:
    start = robot.local_state().copy()
    print("Replay initial state:", np.array2string(target, precision=4, suppress_small=True))
    print(
        f"Moving to it over {args.replay_initial_move_s:.1f}s; "
        f"maximum arm travel={np.max(np.abs(target[:12] - start[:12])):.4f} rad"
    )
    if not args.execute:
        print("DRY RUN: no initial-position targets published.")
        return True

    steps = max(1, int(np.ceil(args.replay_initial_move_s * args.control_hz)))
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
    print(f"Replay initial-pose error: arms={arm_error:.5f} rad, grippers={gripper_error:.3f}")
    if arm_error > args.replay_initial_arm_tolerance or gripper_error > args.replay_initial_gripper_tolerance:
        raise RuntimeError(
            "Robot did not reach the replay demo's initial pose: limits are "
            f"{args.replay_initial_arm_tolerance:.3f} rad and "
            f"{args.replay_initial_gripper_tolerance:.3f} gripper units"
        )
    return hotkey.running.is_set()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--server", default=None, help="e.g. http://192.168.1.10:8000 (required unless --replay-hdf5 is set)"
    )
    parser.add_argument("--prompt", required=True)
    parser.add_argument(
        "--replay-hdf5",
        type=Path,
        default=None,
        help="Replay one HDF5 demo's stored actions instead of querying a FastWAM server",
    )
    parser.add_argument(
        "--replay-demo", default="demo_0", help="Demo to replay when --replay-hdf5 is set, e.g. demo_12 or 12"
    )
    parser.add_argument(
        "--replay-mode",
        choices=("relative", "absolute"),
        default="relative",
        help=(
            "relative (default): apply the model's own convention (arm deltas onto live "
            "feedback, absolute gripper) each step. absolute: republish the recorded "
            "absolute_actions target exactly, ignoring live feedback for target computation."
        ),
    )
    parser.add_argument("--replay-initial-move-s", type=float, default=5.0)
    parser.add_argument("--replay-initial-arm-tolerance", type=float, default=0.12)
    parser.add_argument("--replay-initial-gripper-tolerance", type=float, default=15.0)
    parser.add_argument("--execute", action="store_true", help="Publish targets; default is dry-run")
    parser.add_argument(
        "--action-mode",
        choices=("relative", "absolute"),
        default="absolute",
        help=(
            "How to interpret live model arm outputs. relative adds the first 12 values "
            "to live feedback; absolute publishes all 14 values as absolute targets."
        ),
    )
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
    parser.add_argument(
        "--wait-for-convergence",
        action="store_true",
        help="Wait after each action until the right arm reaches its target or the settle timeout expires",
    )
    parser.add_argument(
        "--settle-tolerance-deg",
        type=float,
        default=2.0,
        help="Maximum right-arm joint error considered settled (default: 2 degrees)",
    )
    parser.add_argument(
        "--settle-timeout-s",
        type=float,
        default=0.25,
        help="Maximum total time allowed for each action step in convergence mode (default: 0.25 s)",
    )
    parser.add_argument(
        "--settle-consecutive-samples",
        type=int,
        default=2,
        help="Fresh in-tolerance feedback samples required before advancing (default: 2)",
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
        or args.gripper_min > args.gripper_max
        or args.max_observation_skew_ms <= 0
        or args.observation_buffer_s <= 0
        or args.max_observation_age_ms <= 0
        or args.settle_tolerance_deg <= 0
        or args.settle_timeout_s <= 0
        or args.settle_consecutive_samples <= 0
    ):
        raise ValueError("Invalid control, replan, delta, or gripper limits")
    if args.replay_hdf5 is None and args.server is None:
        raise ValueError("--server is required unless --replay-hdf5 is set")
    server = args.server.rstrip("/") if args.server else None
    endpoint = server + "/infer" if server else None
    period = 1.0 / args.control_hz
    replay_actions = replay_initial_state = replay_task = None
    if args.replay_hdf5 is not None:
        relative_actions, absolute_targets, replay_initial_state, replay_task = load_replay_actions(
            args.replay_hdf5, args.replay_demo
        )
        replay_actions = relative_actions if args.replay_mode == "relative" else absolute_targets
        print(
            f"Replay source: {args.replay_hdf5} [{args.replay_demo}] ({replay_task}); "
            f"{len(replay_actions)} actions; mode={args.replay_mode}"
        )
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
            "wait_for_convergence": args.wait_for_convergence,
            "settle_tolerance_deg": args.settle_tolerance_deg,
            "settle_timeout_s": args.settle_timeout_s,
            "settle_consecutive_samples": args.settle_consecutive_samples,
            "max_observation_skew_ms": args.max_observation_skew_ms,
            "max_observation_age_ms": args.max_observation_age_ms,
            "created_at": wall_time(),
            "host": os.uname().nodename,
            "replay_hdf5": str(args.replay_hdf5) if args.replay_hdf5 else None,
            "replay_demo": args.replay_demo if args.replay_hdf5 else None,
        },
    )
    if args.replay_hdf5 is not None:
        run_id = f"replay_{_safe_path_component(args.replay_demo)}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    else:
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
    publish_logger = CommandPublishLogger(session)
    action_queue: deque[tuple[int, int, np.ndarray]] = deque()
    rclpy.init()
    video_recorder = TrajectoryRecorder()
    robot = RobotIO(
        args.wrist_transport,
        max(args.control_hz, 20.0),
        args.observation_buffer_s,
        video_recorder,
        publish_logger,
    )
    step_logger = ActionStepLogger(session)
    image_logger = ObservationImageLogger(args.observations_dir)
    print("EXECUTE ENABLED" if args.execute else "DRY RUN: add --execute to move the robot")
    replan_description = args.replan_steps if args.replan_steps is not None else "full trained horizon"
    if args.replay_hdf5 is not None:
        print(f"REPLAY MODE: {args.replay_hdf5} [{args.replay_demo}]; replan steps: {replan_description}")
    else:
        print(
            f"FastWAM server: {endpoint}; replan steps: {replan_description}; "
            f"max observation skew: {args.max_observation_skew_ms:.1f} ms"
        )
    if args.wait_for_convergence:
        print(
            "CONVERGENCE MODE: right-arm error <= "
            f"{args.settle_tolerance_deg:.2f} deg for {args.settle_consecutive_samples} fresh samples; "
            f"per-step timeout {args.settle_timeout_s:.3f}s."
        )
    print(f"Recording {num_demos} demos under {session.root}.")
    print("Press Enter to start demo_1. Press Enter again to stop and reset.")

    hotkey = EnterHotkey(on_stop=robot.hold_current_feedback if args.execute else None)
    inference_started = False
    completed_demos = 0
    chunk_id = 0
    replay_index = 0

    def current_demo_id() -> str:
        return f"demo_{completed_demos + 1}"

    def log_event(event: str, details: dict[str, object] | None = None) -> None:
        observation_recorder.append_event(current_demo_id(), event, details)

    try:
        while rclpy.ok():
            if not hotkey.running.is_set():
                action_queue.clear()
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
                robot.set_active_demo(current_demo_id())
                video_recorder.start_demo(session.demo_dir(current_demo_id()))
                log_event("demo_started")
                print(f"{current_demo_id()} inference started.")
                if args.replay_hdf5 is not None:
                    replay_index = 0
                    if not move_to_replay_start(robot, hotkey, replay_initial_state, args):
                        continue

            if not action_queue and args.replay_hdf5 is not None:
                n_execute = len(replay_actions) - replay_index
                if args.replan_steps is not None:
                    n_execute = min(n_execute, args.replan_steps)
                if n_execute <= 0:
                    print("Replay demo exhausted; stopping.")
                    if args.execute:
                        robot.hold_current_feedback()
                    hotkey.running.clear()
                    continue
                chunk_id += 1
                chunk = replay_actions[replay_index : replay_index + n_execute]
                replay_index += n_execute
                action_queue.extend(
                    (chunk_id, action_step, action.copy())
                    for action_step, action in enumerate(chunk, start=1)
                )
                print(f"Replaying actions {replay_index - n_execute + 1}-{replay_index}/{len(replay_actions)}")
                robot.spin_for(0.05)

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
                if args.execute and robot.held_targets() is None:
                    if not robot.hold_current_feedback():
                        raise RuntimeError("Could not establish the initial hold target before inference")
                actions = request_actions_while_spinning(
                    robot,
                    endpoint,
                    payload,
                    args.request_timeout,
                    args.action_mode,
                )
                n_execute = len(actions) if args.replan_steps is None else min(args.replan_steps, len(actions))
                if not hotkey.running.is_set():
                    print("Action chunk returned after STOP; discarding it.")
                    continue
                action_queue.extend(
                    (chunk_id, action_step, action.copy())
                    for action_step, action in enumerate(actions[:n_execute], start=1)
                )
                print(f"Received {len(actions)} actions in {time.monotonic() - started:.3f}s; executing {n_execute}.")

                # Allow any callbacks queued at request completion to run.
                robot.spin_for(0.05)

            if not hotkey.running.is_set():
                action_queue.clear()
                continue

            tick_started = time.monotonic()
            queued_chunk_id, action_step, action = action_queue.popleft()
            measured_before = robot.local_state().copy()
            if (
                (args.replay_hdf5 is not None and args.replay_mode == "absolute")
                or (args.replay_hdf5 is None and args.action_mode == "absolute")
            ):
                target = robot.prepare_absolute_targets(action, args.gripper_min, args.gripper_max)
            else:
                target = robot.prepare_targets(
                    action,
                    args.gripper_min,
                    args.gripper_max,
                    current_state=measured_before,
                )
            published = False
            if args.execute:
                if not robot.publish_targets_if_enabled(
                    target,
                    hotkey.running,
                    chunk_id=queued_chunk_id,
                    action_step=action_step,
                ):
                    action_queue.clear()
                    continue
                robot.start_hold()
                published = True
            else:
                print("[dry-run] target=", np.array2string(target, precision=4))
            settled: bool | None = None
            settle_elapsed_s = 0.0
            right_arm_max_error_rad: float | None = None
            if args.execute and args.wait_for_convergence:
                settled, settle_elapsed_s, right_arm_max_error_rad = robot.wait_for_right_arm_convergence(
                    target,
                    hotkey.running,
                    minimum_wait_s=period,
                    timeout_s=args.settle_timeout_s,
                    tolerance_rad=np.deg2rad(args.settle_tolerance_deg),
                    consecutive_samples=args.settle_consecutive_samples,
                )
                if not hotkey.running.is_set():
                    action_queue.clear()
                    continue
            else:
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
                    "action_mode": args.action_mode,
                    "measured_before": measured_before.tolist(),
                    "commanded_target": target.tolist(),
                    "measured_after": measured_after.tolist(),
                    "published": published,
                    "tick_duration_s": time.monotonic() - tick_started,
                    "convergence_enabled": args.wait_for_convergence,
                    "settled": settled,
                    "settle_elapsed_s": settle_elapsed_s,
                    "right_arm_max_error_rad": right_arm_max_error_rad,
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
        robot.close()
        publish_logger.close()
        video_recorder.close()
        step_logger.close()
        image_logger.close()
        robot.destroy_node()
        rclpy.shutdown()

    # EnterHotkey's daemon may still be blocked in input() while holding
    # stdin's internal lock, so exit directly after all cleanup is complete.
    os._exit(1 if failed else 0)


if __name__ == "__main__":
    main()
