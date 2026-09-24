#!/usr/bin/env python3
"""R1 Lite FastWAM client with neutral left-arm model conditioning.

This test variant runs the normal inference client, but replaces only the six
left-arm proprio values sent to the server with their right-arm-only training
means. Live right-arm and gripper feedback are unchanged. Robot execution is
also unchanged so this is a single-variable A/B test of model conditioning.
"""
from __future__ import annotations

import numpy as np

import inference_r1lite_fastwam as client

# runs/right_arm_stack_bowl/dataset_stats.json:
# state.default.global_mean[0:6]
NEUTRAL_LEFT_ARM = np.asarray(
    [
        0.0000316732185,
        -0.00136027753,
        -0.00314065046,
        -0.00299898861,
        -0.0000740331598,
        -0.00302747544,
    ],
    dtype=np.float32,
)


_original_make_payload = client.RobotIO.make_payload


def make_payload_with_neutral_left(
    self: client.RobotIO,
    prompt: str,
    run_id: str,
    demo_id: str,
    chunk_id: int,
    observation: client.ObservationBundle,
) -> dict[str, object]:
    model_state = observation.state.copy()
    model_state[:6] = NEUTRAL_LEFT_ARM
    model_observation = client.ObservationBundle(
        timestamp_s=observation.timestamp_s,
        state=model_state,
        images=observation.images,
        sample_timestamps_s=observation.sample_timestamps_s,
    )
    return _original_make_payload(self, prompt, run_id, demo_id, chunk_id, model_observation)


def main() -> None:
    client.RobotIO.make_payload = make_payload_with_neutral_left
    print("NEUTRAL-LEFT TEST: model input uses training-mean left-arm proprio.")
    print("Live right-arm/gripper proprio and normal action execution are unchanged.")
    client.main()


if __name__ == "__main__":
    main()
