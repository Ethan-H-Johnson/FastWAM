#!/usr/bin/env python3
"""One-shot live test that freezes the observed pose during FastWAM inference.

The script instruments ``inference_r1lite_fastwam.py`` without maintaining a
second copy of its ROS/control implementation. It runs one demo, executes three
16-action chunks, then stops and lets the normal reset path take over. The
default remains a dry run; ``--execute`` enables robot motion.
"""

from __future__ import annotations

import json
from pathlib import Path
import sys
import time

import numpy as np

import inference_r1lite_fastwam as live


SETTLE_S = 0.25
MAX_OBSERVATION_HOLD_ERROR_RAD = 0.05
MAX_JOINT_DELTA_RAD = 0.25
ACTION_LIMIT = 16
CHUNK_LIMIT = 3


def _install_diagnostic() -> None:
    if not hasattr(live, "request_actions_while_spinning"):
        raise RuntimeError(
            "This diagnostic requires the live client version that keeps ROS callbacks "
            "spinning during the policy request."
        )

    execute = "--execute" in sys.argv[1:]
    original_sync = live.RobotIO.synchronized_observation
    original_request = live.request_actions_while_spinning
    original_publish = live.RobotIO.publish_targets_if_enabled
    original_prepare = live.RobotIO.prepare_targets
    original_parse_args = live.parse_args

    def parse_args():
        args = original_parse_args()
        # The robot's current experimental client removed the CLI option and
        # prepare_targets parameter but still validates args.max_joint_delta.
        # Supply it for compatibility; prepare_targets below performs the
        # actual clipping used by this diagnostic.
        if not hasattr(args, "max_joint_delta"):
            args.max_joint_delta = MAX_JOINT_DELTA_RAD
        return args

    def synchronized_observation(self, max_skew_s: float, max_age_s: float):
        if execute:
            if not self.hold_current_feedback():
                return None
            self.spin_for(SETTLE_S)

        observation = original_sync(self, max_skew_s, max_age_s)
        if observation is None:
            return None

        measured = self.local_state().copy()
        hold_error = float(np.max(np.abs(measured[:12] - observation.state[:12])))
        if execute:
            if hold_error > MAX_OBSERVATION_HOLD_ERROR_RAD:
                raise RuntimeError(
                    "Refusing to hold a stale observation pose: max arm difference "
                    f"{hold_error:.5f} rad exceeds {MAX_OBSERVATION_HOLD_ERROR_RAD:.5f} rad"
                )
            self.publish_targets(observation.state)
            self.start_hold()

        self._hold_test_observation = observation
        self._hold_test_measured_at_request = measured
        print(f"HOLD_TEST request state difference: max arm={hold_error:.5f} rad")
        return observation

    def request_actions_while_spinning(robot, endpoint, payload, timeout_s):
        started = time.monotonic()
        actions = original_request(robot, endpoint, payload, timeout_s)
        measured_after = robot.local_state().copy()
        observation = robot._hold_test_observation
        drift = measured_after - observation.state
        record = {
            "chunk_id": payload.get("chunk_id"),
            "duration_s": time.monotonic() - started,
            "observation_timestamp_s": float(observation.timestamp_s),
            "observation_state": observation.state.tolist(),
            "measured_at_request": robot._hold_test_measured_at_request.tolist(),
            "measured_after_inference": measured_after.tolist(),
            "state_drift": drift.tolist(),
            "max_arm_drift_rad": float(np.max(np.abs(drift[:12]))),
        }
        print(
            "HOLD_TEST inference drift: "
            f"max arm={record['max_arm_drift_rad']:.5f} rad; "
            f"per-joint={np.array2string(drift[:12], precision=5)}"
        )
        publish_logger = getattr(robot, "_publish_logger", None)
        if publish_logger is not None:
            path = Path(publish_logger.path).parent / "hold_observation_test.jsonl"
            with path.open("a", encoding="utf-8") as stream:
                stream.write(json.dumps(record, separators=(",", ":")) + "\n")
        return actions

    def prepare_targets(self, action, *args, **kwargs):
        safe_action = np.asarray(action, dtype=np.float32).copy()
        safe_action[:12] = np.clip(
            safe_action[:12], -MAX_JOINT_DELTA_RAD, MAX_JOINT_DELTA_RAD
        )
        return original_prepare(self, safe_action, *args, **kwargs)

    def publish_targets_if_enabled(
        self, target, enabled, chunk_id=None, action_step=None, **kwargs
    ):
        published = original_publish(
            self,
            target,
            enabled,
            chunk_id=chunk_id,
            action_step=action_step,
            **kwargs,
        )
        if (
            published
            and chunk_id is not None
            and chunk_id >= CHUNK_LIMIT
            and action_step is not None
            and action_step >= ACTION_LIMIT
        ):
            enabled.clear()
            print(
                f"HOLD_TEST completed chunk {chunk_id}/{CHUNK_LIMIT}; stopping and resetting."
            )
        return published

    live.RobotIO.synchronized_observation = synchronized_observation
    live.RobotIO.prepare_targets = prepare_targets
    live.RobotIO.publish_targets_if_enabled = publish_targets_if_enabled
    live.request_actions_while_spinning = request_actions_while_spinning
    live.parse_args = parse_args
    live.ask_num_demos = lambda: 1


if __name__ == "__main__":
    if "--replan-steps" not in sys.argv[1:]:
        sys.argv[1:1] = ["--replan-steps", str(ACTION_LIMIT)]
    _install_diagnostic()
    live.main()
