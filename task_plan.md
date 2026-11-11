# ACT Data Collection Plan

Goal: Add standard camera + joint-state ACT episode data collection and training for EL-A3 demonstrations.

Current Phase: Verification and delivery

## Phases

- complete: Inspect existing teaching and robot feedback code.
- complete: Implement reusable ACT episode recorder and CLI collection script.
- complete: Add dataset inspection tool.
- complete: Document usage and dependencies.
- complete: Run syntax/import verification.
- complete: Implement ACT HDF5 dataset and training script.
- complete: Document training usage.
- complete: Run training script syntax/help verification.
- complete: Implement trained ACT policy deployment script.
- complete: Document deployment usage.
- complete: Run deployment syntax/help verification.
- complete: Implement deployment preflight safety checker.
- complete: Document preflight usage.
- complete: Run preflight syntax/help verification.
- complete: Resolve user edit conflicts in episode writer and MuJoCo environment.
- complete: Add GPT-Policy-style closed-loop simulation teacher and structured decision trace.
- complete: Add bounded human-like trajectory variation and verify grasp success across varied episodes.
- in_progress: Run end-to-end collection/training/evaluation smoke test and sync the daily workspace.

## Key Questions

- Camera source is not specified, so the first implementation supports generic OpenCV camera indices/devices with configurable names.
- Dataset format targets common ACT/ALOHA-style HDF5 groups: observations/qpos, observations/qvel, observations/images/<camera>, action.

## Decisions Made

- Use HDF5 for compatibility with ACT training pipelines.
- Store images as RGB uint8 arrays under observations/images.
- Define action[t] as the next qpos sample and duplicate the final action at the last frame.
- Keep h5py and OpenCV as runtime imports so existing GUI use is unaffected when they are not installed.
- Add `requirements-act.txt` for optional ACT collection dependencies.
- Implement a lightweight ACT-style policy locally: CNN image encoder + qpos embedding + transformer decoder action queries.
- Policy deployment should use `ELA3RealtimeController` so existing joint limit, densification, and velocity safety checks stay in the control path.
- Preflight should not execute real policy motion; it only loads components, reads feedback, runs one inference, and validates the first target through dry-run safety paths.
- Deployment targets should be clipped by both max per-step delta and joint limits before submission.

## Errors Encountered

- Current environment lacks h5py/cv2, so only syntax/help checks can run until ACT dependencies are installed.
