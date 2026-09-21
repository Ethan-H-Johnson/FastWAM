#!/usr/bin/env python3
"""Run a workstation-hosted FastWAM policy from a Galaxea R1 Lite ROS 2 PC.

The workstation returns [d_left_6, d_right_6, left_gripper, right_gripper].
This client integrates only arm deltas over fresh feedback, publishes absolute
gripper targets, and holds the last target while the next action chunk arrives.
"""
from __future__ import annotations

import argparse
import base64
from collections import deque
from datetime import datetime
import json
import os
from pathlib import Path
import queue
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


class RobotIO(Node):
    def __init__(self, wrist_transport: str, hold_hz: float) -> None:
        super().__init__("r1lite_fastwam_inference")
        self.bridge = CvBridge()
        self.arm: dict[str, np.ndarray | None] = {"left": None, "right": None}
        self.arm_names: dict[str, list[str] | None] = {"left": None, "right": None}
        self.gripper: dict[str, float | None] = {"left": None, "right": None}
        self.images: dict[str, np.ndarray | None] = {"head": None, "left_wrist": None, "right_wrist": None}
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

    def _arm_callback(self, side: str) -> Callable[[JointState], None]:
        def callback(message: JointState) -> None:
            if len(message.position) >= JOINTS_PER_ARM:
                self.arm[side] = np.asarray(message.position[:JOINTS_PER_ARM], dtype=np.float32)
                self.arm_names[side] = list(message.name[:JOINTS_PER_ARM])
        return callback

    def _gripper_callback(self, side: str) -> Callable[[JointState], None]:
        def callback(message: JointState) -> None:
            if message.position:
                self.gripper[side] = float(message.position[0])
        return callback

    def _image_callback(self, name: str) -> Callable[[Image | CompressedImage], None]:
        def callback(message: Image | CompressedImage) -> None:
            try:
                if isinstance(message, CompressedImage):
                    bgr = self.bridge.compressed_imgmsg_to_cv2(message, desired_encoding="bgr8")
                else:
                    bgr = self.bridge.imgmsg_to_cv2(message, desired_encoding="bgr8")
                self.images[name] = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
            except Exception as exc:  # noqa: BLE001
                self.get_logger().warning("Could not decode %s image: %s", name, exc)
        return callback

    def spin_for(self, seconds: float) -> None:
        deadline = time.monotonic() + seconds
        while rclpy.ok() and time.monotonic() < deadline:
            rclpy.spin_once(self, timeout_sec=min(0.01, deadline - time.monotonic()))

    def ready(self) -> bool:
        return all(x is not None for x in self.arm.values()) and all(x is not None for x in self.gripper.values()) and all(x is not None for x in self.images.values())

    def local_state(self) -> np.ndarray:
        if not all(x is not None for x in self.arm.values()) or not all(x is not None for x in self.gripper.values()):
            raise RuntimeError("Robot arm/gripper feedback is incomplete")
        return np.concatenate((self.arm["left"], self.arm["right"], np.asarray([self.gripper["left"], self.gripper["right"]], dtype=np.float32)))

    @staticmethod
    def _encode_jpeg(image: np.ndarray) -> str:
        success, encoded = cv2.imencode(".jpg", cv2.cvtColor(image, cv2.COLOR_RGB2BGR), (cv2.IMWRITE_JPEG_QUALITY, 90))
        if not success:
            raise RuntimeError("Failed to encode camera image")
        return base64.b64encode(encoded.tobytes()).decode("ascii")

    def make_payload(
        self,
        prompt: str,
        run_id: str,
        demo_id: str,
        chunk_id: int,
    ) -> dict[str, object]:
        if not self.ready():
            raise RuntimeError("Robot observation is incomplete")
        return {
            "prompt": prompt,
            "state": self.local_state().tolist(),
            "head_image": self._encode_jpeg(self.images["head"].copy()),
            "left_wrist_image": self._encode_jpeg(self.images["left_wrist"].copy()),
            "right_wrist_image": self._encode_jpeg(self.images["right_wrist"].copy()),
            "run_id": run_id,
            "demo_id": demo_id,
            "chunk_id": chunk_id,
        }

    def model_input_image(self) -> np.ndarray:
        """Build the visual mosaic used by the server, for robot-local verification."""
        head = self.images["head"]
        left = self.images["left_wrist"]
        right = self.images["right_wrist"]
        if head is None or left is None or right is None:
            raise RuntimeError("Robot observation is incomplete")
        h, w = head.shape[:2]
        if w < 640 or h < 640 or left.shape[:2] != (360, 640) or right.shape[:2] != (360, 640):
            raise RuntimeError("Unexpected R1 Lite camera dimensions")
        x0, y0 = (w - 640) // 2, (h - 640) // 2
        head = head[y0 : y0 + 640, x0 : x0 + 640]
        top = cv2.resize(head, (320, 256), interpolation=cv2.INTER_LINEAR)
        bottom = np.concatenate(
            (cv2.resize(left, (160, 128), interpolation=cv2.INTER_LINEAR),
             cv2.resize(right, (160, 128), interpolation=cv2.INTER_LINEAR)), axis=1
        )
        return np.concatenate((top, bottom), axis=0)

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

    def __init__(self, observations_dir: Path) -> None:
        self.observations_dir = observations_dir
        self.pending: queue.Queue[dict[str, object] | None] = queue.Queue()
        self.records: dict[tuple[str, str], list[dict[str, object]]] = {}
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
                run_dir = self.observations_dir / str(record["run_id"])
                demo_dir = run_dir / str(record["demo_id"])
                demo_dir.mkdir(parents=True, exist_ok=True)
                before = np.asarray(record.pop("measured_before"), dtype=np.float32)
                target = np.asarray(record.pop("commanded_target"), dtype=np.float32)
                after = np.asarray(record.pop("measured_after"), dtype=np.float32)
                action = np.asarray(record.pop("model_action"), dtype=np.float32)
                summary = {
                    "step": int(record["action_step"]),
                    "chunk": int(record["chunk_id"]),
                    "target": np.round(target, 3).tolist(),
                    "measured": np.round(after, 3).tolist(),
                    "error": np.round(target - after, 3).tolist(),
                    "delta": np.round(action[:12], 3).tolist(),
                }
                key = (str(record["run_id"]), str(record["demo_id"]))
                with self.records_lock:
                    self.records.setdefault(key, []).append(summary)
                    summaries = list(self.records[key])
                (demo_dir / "action_summary.json").write_text(
                    json.dumps({"steps": summaries}, separators=(",", ":")) + "\n", encoding="utf-8"
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
            self._write_graph(run_id, demo_id, summaries)
        self.pending.put(None)
        self.worker.join()

    def _write_graph(self, run_id: str, demo_id: str, summaries: list[dict[str, object]]) -> None:
        if not summaries:
            return
        width, height = 1400, 900
        canvas = np.full((height, width, 3), 255, dtype=np.uint8)
        colors = [(int((i * 47) % 220), int((i * 83) % 220), int((i * 131) % 220)) for i in range(14)]
        panels = ((40, 60, 1360, 400, "Target (current + model delta) vs measured"), (40, 500, 1360, 840, "Model arm deltas"))
        for x0, y0, x1, y1, title in panels:
            cv2.rectangle(canvas, (x0, y0), (x1, y1), (30, 30, 30), 2)
            cv2.putText(canvas, title, (x0 + 10, y0 - 12), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 0), 2)
        n = len(summaries)
        def draw_series(values: np.ndarray, y0: int, y1: int) -> None:
            lo, hi = float(np.nanmin(values)), float(np.nanmax(values))
            if hi - lo < 1e-6:
                lo, hi = lo - 1.0, hi + 1.0
            for dim in range(values.shape[1]):
                pts = []
                for i, value in enumerate(values[:, dim]):
                    x = 40 + int(i * 1320 / max(n - 1, 1))
                    y = y1 - int((float(value) - lo) * (y1 - y0) / (hi - lo))
                    pts.append((x, max(y0, min(y1, y))))
                cv2.polylines(canvas, [np.asarray(pts, dtype=np.int32)], False, colors[dim], 1)
        targets = np.asarray([s["target"] for s in summaries], dtype=np.float32)
        measured = np.asarray([s["measured"] for s in summaries], dtype=np.float32)
        deltas = np.zeros_like(targets); deltas[:, :12] = np.asarray([s["delta"] for s in summaries], dtype=np.float32)
        draw_series(targets, 60, 400); draw_series(measured, 60, 400); draw_series(deltas, 500, 840)
        cv2.putText(canvas, "Target=solid color, measured=thin overlay; dimensions 0-13", (50, 875), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 0, 0), 1)
        out = self.observations_dir / run_id / demo_id / "action_summary.png"
        out.parent.mkdir(parents=True, exist_ok=True)
        cv2.imwrite(str(out), canvas)


class ImageSnapshotLogger:
    """Save camera snapshots asynchronously so JPEG encoding never delays control."""

    def __init__(self, observations_dir: Path) -> None:
        self.observations_dir = observations_dir
        self.pending: queue.Queue[tuple[str, str, int, dict[str, np.ndarray]] | None] = queue.Queue()
        self.worker = threading.Thread(target=self._run, daemon=True)
        self.worker.start()

    def submit(self, run_id: str, demo_id: str, chunk_id: int, images: dict[str, np.ndarray]) -> None:
        self.pending.put_nowait((run_id, demo_id, chunk_id, {k: v.copy() for k, v in images.items()}))

    def _run(self) -> None:
        while True:
            item = self.pending.get()
            try:
                if item is None:
                    return
                run_id, demo_id, chunk_id, images = item
                out = self.observations_dir / run_id / demo_id / "images" / f"chunk_{chunk_id:04d}"
                out.mkdir(parents=True, exist_ok=True)
                for name, image in images.items():
                    cv2.imwrite(str(out / f"{name}.jpg"), cv2.cvtColor(image, cv2.COLOR_RGB2BGR), [cv2.IMWRITE_JPEG_QUALITY, 90])
            except Exception as exc:  # noqa: BLE001
                print(f"IMAGE_LOG_FAILED: {exc}")
            finally:
                self.pending.task_done()

    def close(self) -> None:
        self.pending.join()
        self.pending.put(None)
        self.worker.join()


class ClientObservationRecorder:
    """Synchronous low-rate event recorder rooted on the robot filesystem."""

    def __init__(self, root: Path, run_id: str, metadata: dict[str, object]) -> None:
        self.root = root
        self.run_dir = root / run_id
        self.run_dir.mkdir(parents=True, exist_ok=True)
        (self.run_dir / "run.json").write_text(
            json.dumps(metadata, indent=2, sort_keys=True) + "\n", encoding="utf-8"
        )

    def append_event(self, run_id: str, demo_id: str, event: str, details: dict[str, object] | None = None) -> None:
        demo_dir = self.root / run_id / demo_id
        demo_dir.mkdir(parents=True, exist_ok=True)
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
        default=15.0,
        help="Action execution rate; 15 Hz matches the physical R1Lite demonstration capture rate",
    )
    parser.add_argument(
        "--replan-steps",
        type=int,
        default=None,
        help="Actions to execute per query; default executes the complete horizon returned by the trained server",
    )
    parser.add_argument("--request-timeout", type=float, default=30.0)
    parser.add_argument(
        "--observations-dir",
        type=Path,
        default=Path(__file__).resolve().parents[2] / "observations",
        help="Robot-local root for run/demo events and action traces",
    )
    parser.add_argument("--max-joint-delta", type=float, default=0.25)
    parser.add_argument("--gripper-min", type=float, default=0.0)
    parser.add_argument("--gripper-max", type=float, default=100.0)
    parser.add_argument("--wrist-transport", choices=("raw", "compressed"), default="raw")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if (
        args.control_hz <= 0
        or (args.replan_steps is not None and args.replan_steps <= 0)
        or args.max_joint_delta <= 0
        or args.gripper_min > args.gripper_max
    ):
        raise ValueError("Invalid control, replan, delta, or gripper limits")
    server = args.server.rstrip("/")
    endpoint, period = server + "/infer", 1.0 / args.control_hz
    num_demos = ask_num_demos()
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
    observation_recorder = ClientObservationRecorder(
        args.observations_dir,
        run_id,
        {
            "run_id": run_id,
            "prompt": args.prompt,
            "num_demos": num_demos,
            "execute": args.execute,
            "control_hz": args.control_hz,
            "created_at": wall_time(),
            "host": os.uname().nodename,
        },
    )
    action_queue: deque[tuple[int, int, np.ndarray]] = deque()
    rclpy.init()
    robot = RobotIO(args.wrist_transport, max(args.control_hz, 20.0))
    step_logger = ActionStepLogger(args.observations_dir)
    image_logger = ImageSnapshotLogger(args.observations_dir)
    print("EXECUTE ENABLED" if args.execute else "DRY RUN: add --execute to move the robot")
    replan_description = args.replan_steps if args.replan_steps is not None else "full trained horizon"
    print(f"FastWAM server: {endpoint}; replan steps: {replan_description}")
    print(f"Recording {num_demos} demos under {args.observations_dir / run_id}.")
    print("Press Enter to start demo_1. Press Enter again to stop and reset.")

    hotkey = EnterHotkey(on_stop=robot.hold_current_feedback if args.execute else None)
    inference_started = False
    completed_demos = 0
    chunk_id = 0

    def current_demo_id() -> str:
        return f"demo_{completed_demos + 1}"

    def log_event(event: str, details: dict[str, object] | None = None) -> None:
        observation_recorder.append_event(run_id, current_demo_id(), event, details)

    try:
        while rclpy.ok():
            if not hotkey.running.is_set():
                action_queue.clear()
                if inference_started:
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
                log_event("demo_started")
                print(f"{current_demo_id()} inference started.")

            if not action_queue:
                robot.spin_for(0.10)
                if not robot.ready():
                    print("Waiting for robot feedback and all three cameras...")
                    continue
                chunk_id += 1
                started = time.monotonic()
                payload = robot.make_payload(args.prompt, run_id, current_demo_id(), chunk_id)
                image_logger.submit(
                    run_id,
                    current_demo_id(),
                    chunk_id,
                    {"model_input": robot.model_input_image()},
                )
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
                print(f"Received {len(actions)} actions in {time.monotonic() - started:.3f}s; executing {n_execute}.")

                # Refresh feedback accumulated during the blocking request.
                robot.spin_for(0.05)

            if not hotkey.running.is_set():
                action_queue.clear()
                continue

            tick_started = time.monotonic()
            queued_chunk_id, action_step, action = action_queue.popleft()
            measured_before = robot.local_state().copy()
            target = robot.prepare_targets(
                action,
                args.max_joint_delta,
                args.gripper_min,
                args.gripper_max,
                current_state=measured_before,
            )
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
