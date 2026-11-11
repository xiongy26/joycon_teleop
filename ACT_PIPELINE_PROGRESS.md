# ACT Pipeline Implementation Progress

Date: 2026-07-20 (sim validation added 2026-09-18)

This document records the current implementation state for the EL-A3 ACT data collection, training, deployment, and safety preflight workflow.

## Simulation Validation (act_sim/, 2026-09-18)

The full pipeline is now validated end-to-end inside MuJoCo before touching the real robot:

- `act_sim/el_a3_pick_scene.xml` — EL-A3 scene (gravity on, table, 4 cm red cube, `cam_high` overhead + `cam_wrist` gripper cameras). Robot subtree copied verbatim from `robot/el_a3_description/urdf/scene_gripper.xml`.
- `act_sim/env.py` — `ELA3ReachEnv`: 20 Hz position control over 1 kHz physics, name-based joint/qpos addressing (XML element order does not matter), cube spawn window x∈[-0.10,0.10], y∈[-0.42,-0.26] (verified IK-reachable, ≤2 mm error, ≤1.4 mm gravity sag), EGL offscreen cameras.
- `act_sim/expert.py` — DLS position IK (name-resolved qpos/dof addresses) + minimum-jerk segments: home → hover above cube → touch cube top, with bounded timing/waypoint variation.
- `act_sim/agent.py` — GPT-Policy-inspired local observe→decide→execute boundary; emits structured phase decisions and records fresh measured state.
- `act_sim/collect.py` — closed-loop varied demonstration collection into the same HDF5 format as real collection (`sim=True` metadata, sim-time timestamps so qvel/duration are correct) plus `agent_trace.jsonl`.
- `act_sim/evaluate.py` — closed-loop rollouts of trained checkpoints (chunked actions, replan every `query_interval`), 12-episode held-out grid, and deployment-aligned joint target clipping. A rollout is successful when the cube is lifted at least 4 cm and remains grasped; tip-cube distance remains a diagnostic metric; writes `eval_results.json`, `rollout_traces.npz`, and side-by-side mp4 replays.
- `scripts/run_act_sim_pipeline.py` — one command: collect → inspect (existing tool) → train (existing script, unmodified) → evaluate → `report.md` + web export.
- `scripts/export_act_sim_web.py`, `scripts/serve_act_sim_viewer.py`, `web/` — browser viewer built on official `mujoco-js` (MuJoCo compiled to WebAssembly) + three.js: interactive simulation, HDF5 demo replay, policy rollout replay, free/cam_high/cam_wrist views, contact visualization. Mesh rendering uses raw `model.mesh_vert` + mjv scene pose (MuJoCo folds mesh_pos/quat into geom pose at compile time — verified against `mjr_uploadMesh`; do not bake mesh frames in JS).
- `act_data/episode_writer.py` — `append()` now accepts an optional explicit `timestamp` (real-robot path unchanged).
- `act_sim/evaluate.py` CLI: `python -m act_sim.evaluate <ckpt> [--output-dir ...]`.

Verified (2026-09-18):
- 3-episode/3-epoch smoke run completes end-to-end (0% success expected at 3 epochs).
- Full run v1 (24 episodes, 80 epochs, global-avg-pool encoder): 2/12 = 17% success — encoder collapsed spatial information.
- Full run v2 (48 episodes, 150 epochs, 4x4 spatial feature map encoder, `runs/act_sim_v2`): **11/12 = 92% success** under the earlier “hit once in the tail” metric, mean final tip-cube distance 3.2 cm. Re-run evaluation with the current persistent-tail criterion before using this number as a deployment gate. Worst case was the far corner (x=+0.09, y=-0.40). query_interval 5 vs 10 made no difference; more data (>=64 episodes) is the next lever for the last corner.
- Browser viewer verified interactively: mesh assembly correct (raw mesh_vert + scene pose), free/cam_high/cam_wrist views, interactive cube moves, demo replay and policy rollout replay all working against real run data.

## Current Status

The first end-to-end ACT-style pipeline has been implemented:

- Camera + joint-state demonstration collection to HDF5.
- Dataset inspection.
- ACT-style training from collected HDF5 episodes.
- Policy checkpoint loading and real-time deployment.
- Deployment preflight safety checks.

The implementation is CLI-first and does not change the existing MotorStudio teaching GUI behavior.

## Added Files

### Data Collection

- `act_data/camera.py`
  - OpenCV multi-camera capture.
  - Camera spec format: `name=source[:WIDTHxHEIGHT]`, for example `cam_high=0:640x480`.
- `act_data/episode_writer.py`
  - Streams ACT/ALOHA-style HDF5 episodes.
  - Writes `observations/qpos`, `observations/qvel`, `observations/images/<camera>`, `timestamps`, and `action`.
- `act_data/robot.py`
  - EL-A3 lifecycle helper for data collection.
  - Connects robot, enables motors, enters zero-torque teaching mode with optional gravity compensation.
- `scripts/collect_act_episode.py`
  - CLI to collect one ACT demonstration episode.
- `scripts/inspect_act_dataset.py`
  - CLI to inspect HDF5 shape, timing, and basic data validity.

### Training

- `act_data/dataset.py`
  - PyTorch map-style HDF5 dataset.
  - Returns one observation timestep and a padded future action chunk.
- `act_data/policy.py`
  - Compact ACT-style policy:
    - shared CNN image encoder,
    - qpos embedding,
    - transformer encoder/decoder,
    - learned action queries,
    - chunked action output.
- `scripts/train_act_policy.py`
  - CLI training loop with train/validation split.
  - Saves `train_config.json`, `policy_best.pt`, and epoch checkpoints.
  - Checkpoints include `model_config` for deployment reconstruction.

### Deployment

- `scripts/run_act_policy.py`
  - Loads trained checkpoint.
  - Reads cameras and current qpos.
  - Runs policy inference.
  - Sends arm targets through `ELA3RealtimeController`.
  - Supports optional gripper target from action dimension 7.
  - Clips each target by max per-step delta and default arm joint limits before submission.
- `scripts/preflight_act_policy.py`
  - Checks deployment readiness without executing policy motion.
  - Verifies checkpoint, cameras, robot feedback, one inference pass, and first-target safety validation.

### Runtime Dependencies

- `requirements-act.txt`
  - `h5py`
  - `opencv-python`
  - `numpy`
  - `torch`

## Modified Files

- `README.md`
  - Added ACT collection, training, deployment, and preflight usage.
- `ela3_motion/backend.py`
  - Added `read_joint_positions(include_gripper=False)` for 6D or 7D feedback.
  - Kept `read_arm_joints()` as the 6D compatibility wrapper.
- `ela3_motion/realtime.py`
  - Added `read_joint_positions(include_gripper=False)` wrapper for deployment code.

## HDF5 Dataset Format

Each episode stores:

```text
observations/qpos              float32 [T, 7]
observations/qvel              float32 [T, 7]
observations/images/<camera>   uint8   [T, H, W, 3] RGB
timestamps                     float64 [T]
action                         float32 [T, 7]
```

Current action definition:

- `action[t] = qpos[t + 1]`
- final frame action duplicates final qpos

This is intended to make the initial ACT chunked-action training path work. If later you want actions to represent commanded targets, end-effector deltas, or discrete gripper commands, update `act_data/episode_writer.py`.

## Main Commands

Install optional ACT dependencies:

```bash
pip install -r requirements-act.txt
```

Collect one episode:

```bash
python scripts/collect_act_episode.py \
  --task-name pick_cube \
  --camera cam_high=0:640x480 \
  --camera cam_wrist=2:640x480 \
  --rate 30 \
  --output-dir data/act
```

Inspect one episode:

```bash
python scripts/inspect_act_dataset.py data/act/episode_pick_cube_000000.hdf5
```

Train policy:

```bash
python scripts/train_act_policy.py \
  --dataset data/act \
  --cameras cam_high,cam_wrist \
  --chunk-size 50 \
  --batch-size 8 \
  --epochs 100 \
  --device cuda \
  --output-dir checkpoints/pick_cube_act
```

Run deployment preflight:

```bash
python scripts/preflight_act_policy.py \
  --checkpoint checkpoints/pick_cube_act/policy_best.pt \
  --camera cam_high=0:640x480 \
  --camera cam_wrist=2:640x480 \
  --rate 10 \
  --max-target-delta 0.08 \
  --max-joint-velocity 0.6
```

Deploy policy:

```bash
python scripts/run_act_policy.py \
  --checkpoint checkpoints/pick_cube_act/policy_best.pt \
  --camera cam_high=0:640x480 \
  --camera cam_wrist=2:640x480 \
  --rate 10 \
  --query-interval 10 \
  --max-target-delta 0.08 \
  --max-joint-velocity 0.6
```

## Safety Behavior

Deployment uses `ELA3RealtimeController.submit_joint_target()`, so the existing realtime path still handles densification, trajectory queue generation, joint limits, max joint step, and max velocity constraints.

Additional deployment-side checks:

- Policy action must have at least 6 dimensions.
- Arm target is first clipped by `--max-target-delta`.
- Arm target is then clipped to default EL-A3 arm joint limits.
- Default execution is blocking per step to avoid async command buildup.
- `--nonblocking` exists but should only be used after careful testing.

Preflight behavior:

- Does not execute policy motion.
- If robot is available, it connects and reads real feedback.
- It submits the first target through dry-run safety validation.
- `--skip-cameras --skip-robot` can be used for checkpoint-only development checks.

## Verification Completed

Passed:

- `python -m py_compile act_data/*.py scripts/collect_act_episode.py scripts/inspect_act_dataset.py`
- `python scripts/collect_act_episode.py --help`
- `python scripts/train_act_policy.py --help`
- `python scripts/run_act_policy.py --help`
- `python scripts/preflight_act_policy.py --help`
- `python -m py_compile ela3_motion/backend.py ela3_motion/realtime.py act_data/*.py scripts/*.py`
- Temporary minimal checkpoint preflight with `--skip-cameras --skip-robot --device cpu`

Not yet run:

- Real HDF5 collection smoke test.
- Real HDF5 training smoke test.
- Camera hardware read test in this environment.
- Real robot preflight.
- Real robot deployment.

Reason: current environment has `torch`, but does not have `h5py` or `cv2` installed.

## Known Limitations

- The local policy is ACT-style, not a full original ACT CVAE reproduction.
- No dataset normalization statistics are computed yet.
- No image augmentation pipeline yet.
- No action temporal aggregation during deployment beyond chunk reuse/query interval.
- No deployment telemetry log file yet.
- No automatic success/failure labeling for episodes.
- No GUI integration yet; all ACT workflow entry points are CLI scripts.
- HDF5 images are stored raw uint8 arrays, not JPEG-compressed.

## Recommended Next Steps

1. Install `requirements-act.txt` and collect 2-3 short test episodes.
2. Run `inspect_act_dataset.py` on each episode.
3. Add a tiny HDF5 training smoke test with `--epochs 1 --batch-size 1 --device cpu`.
4. Add dataset normalization statistics for qpos/action.
5. Add deployment telemetry logging before first serious robot run.
6. Run `preflight_act_policy.py` with real cameras and robot connected.
7. Start real deployment with conservative parameters:

```bash
--rate 5 --query-interval 5 --max-target-delta 0.03 --max-joint-velocity 0.25
```

## Files To Review First Later

- `scripts/collect_act_episode.py`
- `scripts/train_act_policy.py`
- `scripts/preflight_act_policy.py`
- `scripts/run_act_policy.py`
- `act_data/episode_writer.py`
- `act_data/dataset.py`
- `act_data/policy.py`
