"""Test input and reset handoff logic without importing ROS on the host."""
from __future__ import annotations

import ast
import json
import os
import queue
import signal
import subprocess
import tempfile
import threading
import time
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest import mock

import h5py
import numpy as np

SOURCE = Path(__file__).with_name("inference_r1lite_fastwam.py")


def isolated_definition(name: str, environment: dict[str, object]) -> object:
    tree = ast.parse(SOURCE.read_text(encoding="utf-8"))
    definition = next(node for node in tree.body if isinstance(node, (ast.ClassDef, ast.FunctionDef)) and node.name == name)
    future = ast.ImportFrom(module="__future__", names=[ast.alias(name="annotations")], level=0)
    module = ast.fix_missing_locations(ast.Module(body=[future, definition], type_ignores=[]))
    exec(compile(module, str(SOURCE), "exec"), environment)  # noqa: S102 - isolated repo definition
    return environment[name]


class OperatorInputTest(unittest.TestCase):
    def test_rapid_enter_stop_and_score_use_one_reader(self) -> None:
        lines: queue.Queue[str] = queue.Queue()
        stopped = threading.Event()
        operator_type = isolated_definition("OperatorInput", {
            "threading": threading, "queue": queue,
            "input": lambda: lines.get(timeout=3),
        })
        operator = operator_type(on_stop=stopped.set)
        try:
            lines.put("")
            self.assertTrue(operator.running.wait(2))
            lines.put("")
            self.assertTrue(self._wait_until(lambda: not operator.running.is_set()))
            self.assertFalse(stopped.is_set())

            lines.put("")
            self.assertTrue(operator.running.wait(2))
            self.assertTrue(operator.mark_recording(lambda: None))
            lines.put("")
            self.assertTrue(operator.stop_handled.wait(2))
            self.assertTrue(stopped.is_set())
            self.assertFalse(operator.running.is_set())

            lines.put("100")  # typed during reset: ignored, not taken as the score
            self.assertTrue(self._wait_until(lambda: lines.empty()))
            operator.request_score()
            lines.put("0")
            lines.put("100")
            self.assertEqual(operator.wait_for_score(), 100)
            lines.put("")
            self.assertTrue(self._wait_until(lambda: lines.empty()))
            self.assertFalse(operator.running.is_set())
            operator.enable_start()
            lines.put("")
            self.assertTrue(operator.running.wait(2))
        finally:
            operator.close()
            lines.put("")
            operator.worker.join(timeout=2)

    @staticmethod
    def _wait_until(predicate) -> bool:
        deadline = time.monotonic() + 2
        while time.monotonic() < deadline:
            if predicate():
                return True
            time.sleep(0.01)
        return False


class ResetHandoffTest(unittest.TestCase):
    class Robot:
        def __init__(self) -> None:
            self.commands: list[np.ndarray] = []
            self.feedback = np.full(14, 7.0, dtype=np.float32)

        def held_targets(self) -> np.ndarray:
            return np.full(14, 5.0, dtype=np.float32)

        def local_state(self) -> np.ndarray:
            return self.feedback

        def publish_targets(self, target: np.ndarray) -> None:
            self.commands.append(np.asarray(target).copy())

        def spin_for(self, _seconds: float) -> None:
            pass

        def start_hold(self) -> None:
            pass

        def stop_hold(self) -> None:
            pass

    class Process:
        def __init__(self, complete: bool) -> None:
            markers = ["RESET_READY\n", "RESET_TOOK_OVER\n"]
            if complete:
                markers.append("RESET_HOLDING_FINAL\n")
            self.stdout = iter(markers)
            self.returncode: int | None = None

        def send_signal(self, sig: int) -> None:
            if sig == signal.SIGINT:
                self.returncode = -sig

        def wait(self, timeout: float | None = None) -> int:
            if self.returncode is None:
                self.returncode = 1
            return self.returncode

        def poll(self) -> int | None:
            return self.returncode

        def kill(self) -> None:
            self.returncode = -9

    def _reset_function(self):
        return isolated_definition("reset_to_initial_position", {
            "RESET_SCRIPT": SOURCE.with_name("reset_to_initial_position.sh"),
            "ACTION_DIM": 14,
            "Path": Path, "os": os, "json": json, "np": np,
            "subprocess": subprocess, "signal": SimpleNamespace(SIGUSR1=10, SIGINT=signal.SIGINT),
        })

    def test_successful_handoff_commands_reset_target(self) -> None:
        robot = self.Robot()
        reset = self._reset_function()
        with mock.patch.object(subprocess, "Popen", return_value=self.Process(complete=True)):
            reset(robot, True)
        target = json.loads(SOURCE.with_name("initial_robot_position.json").read_text())["position"][:14]
        np.testing.assert_allclose(robot.commands[0], target, atol=1e-6)
        self.assertFalse(np.array_equal(robot.commands[0], robot.feedback))

    def test_failed_reset_reacquires_feedback(self) -> None:
        robot = self.Robot()
        reset = self._reset_function()
        with (
            mock.patch.object(subprocess, "Popen", return_value=self.Process(complete=False)),
            self.assertRaises(subprocess.CalledProcessError),
        ):
            reset(robot, True)
        np.testing.assert_array_equal(robot.commands[-1], robot.feedback)


class ReplaySchemaTest(unittest.TestCase):
    def test_live_rollout_is_not_replayed_as_collection_data(self) -> None:
        loader = isolated_definition("load_replay_actions", {
            "h5py": h5py, "np": np, "ACTION_DIM": 14,
        })
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "run.hdf5"
            with h5py.File(path, "w") as file:
                file.attrs["schema_kind"] = "fastwam_live_rollout_v1"
            with self.assertRaisesRegex(ValueError, "cannot be replayed"):
                loader(path, "demo_1")


if __name__ == "__main__":
    unittest.main()
