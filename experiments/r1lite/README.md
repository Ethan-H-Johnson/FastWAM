# R1 Lite FastWAM inference

`serve_fastwam_r1lite.py` runs on the GPU workstation. It does all FastWAM-specific work: camera composition, `[-1, 1]` image normalization, proprioception normalization with the run's `dataset_stats.json`, action inference, and action de-normalization.

`inference_r1lite_fastwam.py` runs on the R1 Lite computer. It uses the existing ROS 2 topics:

- feedback: `/hdas/feedback_arm_left`, `/hdas/feedback_arm_right`, `/hdas/feedback_gripper_left`, `/hdas/feedback_gripper_right`;
- commands: `/motion_target/target_joint_state_arm_left`, `/motion_target/target_joint_state_arm_right`, `/motion_target/target_position_gripper_left`, `/motion_target/target_position_gripper_right`;
- head camera: `/hdas/camera_head/left_raw/image_raw_color/compressed`.

Raw wrist image topics are the default because they match the existing r1lite inference client. Use `--wrist-transport compressed` if the robot stack only exposes the `/compressed` variants.

## Contract

```text
state:  [left_arm_6, right_arm_6, left_gripper, right_gripper]
action: [delta_left_arm_6, delta_right_arm_6, left_gripper_target, right_gripper_target]
```

The server returns de-normalized actions. The client adds only the first 12 dimensions to fresh arm feedback; it publishes the final two dimensions as absolute gripper targets.

The associated R1 Lite training config must use 14-D data with arm actions already represented as deltas and absolute gripper actions. Include:

```yaml
delta_action_dim_mask:
  default: [true, true, true, true, true, true,
            true, true, true, true, true, true,
            false, false]
```

The mask does not create deltas. Convert only arm actions to deltas in the training-data conversion path.

## FastWAM camera preprocessing

The server first matches R1 Lite collection framing, then follows FastWAM's RobotWin layout:

```text
live head        -> center-crop 640x640 -> bilinear 320x256
live left wrist  -> native 640x360 -> bilinear 160x128
live right wrist -> native 640x360 -> bilinear 160x128

head above [left wrist | right wrist] -> RGB 320x384 -> CHW float [-1, 1]
```

The collection-size crops happen on the workstation, before the FastWAM mosaic, so live observations match the demonstrations.

## One-time model preparation

Run this in the FastWAM environment before training or inference:

```bash
cd /home/spate308/Documents/FastWAM
mkdir -p checkpoints
export DIFFSYNTH_MODEL_BASE_PATH="$(pwd)/checkpoints"

python scripts/preprocess_action_dit_backbone.py \
  --model-config configs/model/fastwam.yaml \
  --output checkpoints/ActionDiT_linear_interp_Wan22_alphascale_1024hdim.pt \
  --device cuda \
  --dtype bfloat16
```

The workstation launcher sets `DIFFSYNTH_MODEL_BASE_PATH` to this repository's `checkpoints/` directory automatically when it is not already set.

## One-time prompt embedding

The server uses cached text context so the T5 encoder does not need to remain loaded during inference. Generate the embedding for the exact task prompt once:

```bash
cd /home/spate308/Documents/FastWAM
export DIFFSYNTH_MODEL_BASE_PATH="$(pwd)/checkpoints"

python scripts/precompute_text_embeds.py \
  --config-name right_arm_stack_bowl_config \
  +override_instruction="stack the bowl" \
  +overwrite=true
```

This writes the cache under `data/text_embeds_cache/r1_lite_tasks`, as configured by `right_arm_stack_bowl_config.yaml`. The prompt passed to the robot must match the precomputed task string. If the GPU runs out of memory while generating the embedding, run this one-time command with `CUDA_VISIBLE_DEVICES=""` to use CPU.

## Run

Install FastAPI and Uvicorn in the FastWAM environment if necessary:

```bash
python -m pip install 'fastapi>=0.110' 'uvicorn[standard]>=0.27'
```

On the workstation:

```bash
bash experiments/r1lite/run_fastwam_server.sh \
  --checkpoint /path/to/r1lite_fastwam.pt \
  --dataset-stats /path/to/r1lite_dataset_stats.json \
  --config configs/right_arm_stack_bowl_config.yaml \
  --text-embedding-cache-dir data/text_embeds_cache/r1_lite_tasks \
  --host 0.0.0.0 --port 8000
```

`--config` must be the FastWAM Hydra config associated with the R1 Lite checkpoint. Its processor dimensions and normalization statistics must match the training run.

On the robot, source ROS 2 and start in dry-run mode:

```bash
bash experiments/r1lite/run_r1lite_client.sh \
  --server http://WORKSTATION_IP:8000 \
  --prompt 'stack the bowl'
```

After verifying state order, gripper units, target limits, and live camera topics:

```bash
bash experiments/r1lite/run_r1lite_client.sh \
  --server http://WORKSTATION_IP:8000 \
  --prompt 'stack the bowl' --execute --control-hz 15
```

The client executes at 15 Hz, matching the demonstrations' physical capture rate, and by default executes the complete action horizon returned by the trained server (32 actions for this checkpoint). Pass `--replan-steps N` only to deliberately use a shorter receding horizon.

The client initially waits without querying the server or moving the robot. Press Enter to start inference. Press Enter again to stop, hold the latest measured pose, and run the shared arm/gripper/torso reset back to `initial_robot_position.json`. After reset, press Enter to begin another rollout. `Ctrl-C` is the emergency exit and intentionally skips automatic reset.

## Observation and execution logs

At startup the robot client asks how many demos to run. The workstation server allocates the next numbered run and stores everything under:

```text
observations/run_N/
  run.json
  demo_M/
    events.jsonl
    action_steps.jsonl
    chunks/chunk_K/
      model_input.png
      model_input.pt
      request.json
      model_actions.json
      normalized_model_output.pt
```

`model_input.pt` contains the exact normalized image, proprioception, cached text context, mask, and inference arguments captured immediately before `infer_action`. `model_input.png` is the post-crop/post-resize three-camera mosaic before `[-1, 1]` normalization. Each action-step record compares the de-normalized model arm deltas and absolute gripper outputs with the measured state used for integration, the clipped absolute command, and ROS feedback measured after that control period. It also includes applied delta and tracking error fields.

Model input/output files and robot step uploads are written through background queues. Observation serialization and logging network calls do not block the inference or 15 Hz control loops. Use `--observations-dir` on the server to override the default `/home/spate308/Documents/FastWAM/observations` root.
