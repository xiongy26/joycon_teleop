# Progress

## 2026-07-20

- Inspected repository structure and existing teaching recorder.
- Confirmed current recorder is insufficient for ACT because it lacks image observations and action dataset.
- Started implementing ACT data collection as a separate module and CLI tools.
- Added `act_data` package with OpenCV multi-camera capture, HDF5 ACT writer, and EL-A3 robot reader.
- Added `scripts/collect_act_episode.py` and `scripts/inspect_act_dataset.py`.
- Documented ACT data collection in `README.md`.
- Added `requirements-act.txt`.
- Added `act_data/dataset.py` for chunked ACT HDF5 training samples.
- Added `act_data/policy.py` with a compact CNN + transformer ACT-style policy.
- Added `scripts/train_act_policy.py` for training, validation, config export, and checkpoint saving.
- Documented ACT-style training usage in `README.md`.
- Added checkpoint model metadata so deployment can reconstruct the trained policy.
- Added `scripts/run_act_policy.py` for real-time camera + qpos inference and EL-A3 target submission.
- Added full-joint feedback helper to realtime/backend layers so deployment can condition on 7D qpos and command arm + gripper.
- Documented policy deployment usage in `README.md`.
- Added `scripts/preflight_act_policy.py` for checkpoint/camera/feedback/inference/motion-safety checks before deployment.
- Documented preflight usage in `README.md`.
- Fixed deployment target handling to clip both per-step delta and default joint limits before submitting targets.

## Verification

- `python -m py_compile act_data/*.py scripts/collect_act_episode.py scripts/inspect_act_dataset.py` passed.
- HDF5 write/read smoke test not run because `h5py` is not installed in the current environment.
- `python -m py_compile act_data/*.py scripts/collect_act_episode.py scripts/inspect_act_dataset.py scripts/train_act_policy.py` passed.
- `python scripts/train_act_policy.py --help` passed.
- `python scripts/collect_act_episode.py --help` still passes without PyTorch import side effects.
- Current environment has `torch`, but lacks `h5py` and `cv2`, so no HDF5 training smoke test was run.
- `python -m py_compile ela3_motion/backend.py ela3_motion/realtime.py act_data/*.py scripts/*.py` passed.
- `python scripts/run_act_policy.py --help` passed.
- Instantiated `ACTPolicy` and verified `export_config()` returns deployment metadata.
- `python scripts/preflight_act_policy.py --help` passed.
- Built a temporary minimal checkpoint and ran `preflight_act_policy.py --skip-cameras --skip-robot --device cpu`; all checks passed.
