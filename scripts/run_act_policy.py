#!/usr/bin/env python3
"""Run a trained ACT-style policy on the real EL-A3 arm."""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from act_data.camera import MultiCameraCapture, parse_camera_specs
from ela3_motion import ELA3MotionOptions, ELA3RealtimeController
from ela3_motion.realtime import (
    DEFAULT_ARM_JOINT_LIMITS_LOWER,
    DEFAULT_ARM_JOINT_LIMITS_UPPER,
)


def _load_torch():
    try:
        import torch
    except ImportError as exc:
        raise RuntimeError(
            "PyTorch is required for ACT policy deployment. Install with: pip install torch"
        ) from exc
    return torch


def _load_checkpoint_config(checkpoint_path: Path, payload: dict, args) -> dict:
    model_config = dict(payload.get("model_config") or {})
    config = dict(payload.get("config") or {})

    train_config_path = checkpoint_path.parent / "train_config.json"
    if train_config_path.exists():
        try:
            train_config = json.loads(train_config_path.read_text())
            model_config = {**train_config.get("model_config", {}), **model_config}
            config = {**train_config, **config}
        except json.JSONDecodeError:
            pass

    camera_names = args.cameras.split(",") if args.cameras else model_config.get("camera_names")
    if camera_names is None:
        raise RuntimeError(
            "checkpoint does not include camera names; pass --cameras cam_high,cam_wrist"
        )
    model_config["camera_names"] = list(camera_names)
    model_config.setdefault("num_cameras", len(camera_names))
    model_config.setdefault("image_downsample", config.get("image_downsample", 1))

    required = ["qpos_dim", "action_dim", "chunk_size", "hidden_dim", "nheads"]
    missing = [key for key in required if key not in model_config]
    if missing:
        raise RuntimeError(
            f"checkpoint is missing model config keys {missing}. "
            "Retrain with the current train script or pass a checkpoint saved with model_config."
        )
    model_config.setdefault("num_encoder_layers", config.get("encoder_layers", 4))
    model_config.setdefault("num_decoder_layers", config.get("decoder_layers", 4))
    model_config.setdefault("dropout", config.get("dropout", 0.0))
    return model_config


def _build_policy(checkpoint_path: Path, device: str, args):
    torch = _load_torch()
    from act_data.policy import ACTPolicy

    payload = torch.load(checkpoint_path, map_location=device)
    model_config = _load_checkpoint_config(checkpoint_path, payload, args)
    policy_kwargs = {
        key: model_config[key]
        for key in (
            "qpos_dim",
            "action_dim",
            "num_cameras",
            "chunk_size",
            "hidden_dim",
            "nheads",
            "num_encoder_layers",
            "num_decoder_layers",
            "dropout",
        )
    }
    model = ACTPolicy(**policy_kwargs)
    model.load_state_dict(payload["model"])
    model.to(device)
    model.eval()
    return model, model_config


def _prepare_images(frames: dict[str, np.ndarray], camera_names: list[str], downsample: int):
    torch = _load_torch()
    images = []
    for name in camera_names:
        if name not in frames:
            raise RuntimeError(f"missing frame for camera {name!r}")
        image = frames[name]
        if downsample > 1:
            image = image[::downsample, ::downsample]
        image = np.ascontiguousarray(image.transpose(2, 0, 1))
        images.append(image)
    batch = np.stack(images, axis=0).astype(np.float32) / 255.0
    return torch.from_numpy(batch).unsqueeze(0)


def _clip_target_delta(target: np.ndarray, current: np.ndarray, max_delta: float) -> np.ndarray:
    if max_delta <= 0:
        return target
    delta = np.clip(target - current, -max_delta, max_delta)
    return current + delta


def _safe_arm_target(
    target: np.ndarray,
    current: np.ndarray,
    *,
    max_delta: float,
    lower_limits: np.ndarray = DEFAULT_ARM_JOINT_LIMITS_LOWER,
    upper_limits: np.ndarray = DEFAULT_ARM_JOINT_LIMITS_UPPER,
) -> np.ndarray:
    delta_limited = _clip_target_delta(target, current, max_delta)
    return np.clip(delta_limited, lower_limits, upper_limits)


def run_policy(args: argparse.Namespace) -> None:
    torch = _load_torch()
    device = torch.device(args.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is not available. Use --device cpu.")

    checkpoint_path = Path(args.checkpoint)
    model, model_config = _build_policy(checkpoint_path, str(device), args)
    camera_names = list(model_config["camera_names"])
    image_downsample = int(args.image_downsample or model_config.get("image_downsample", 1))
    qpos_dim = int(model_config["qpos_dim"])
    action_dim = int(model_config["action_dim"])
    chunk_size = int(model_config["chunk_size"])

    camera_specs = parse_camera_specs(args.camera)
    spec_names = [camera.name for camera in camera_specs]
    missing_specs = sorted(set(camera_names) - set(spec_names))
    if missing_specs:
        raise RuntimeError(f"missing camera specs for checkpoint cameras: {missing_specs}")

    options = ELA3MotionOptions(
        can_name=args.can,
        backend=args.backend,
        serial_port=args.serial_port,
        setup_can=args.setup_can,
        can_bitrate=args.can_bitrate,
        host_can_id=args.host_can_id,
        dry_run=args.dry_run,
        disable_on_finish=not args.keep_enabled,
        command_rate_hz=args.rate,
        max_joint_step_rad=args.max_joint_step,
        max_joint_velocity_rad_s=args.max_joint_velocity,
    )

    controller = ELA3RealtimeController(options)
    period = 1.0 / float(args.rate)
    current_chunk = None
    chunk_cursor = 0
    steps_since_query = args.query_interval
    step_count = 0
    deadline = None if args.duration <= 0 else time.perf_counter() + args.duration

    print(f"Loading checkpoint: {checkpoint_path}")
    print(f"Checkpoint cameras: {camera_names}")
    print("Connecting robot and cameras...")
    with MultiCameraCapture(camera_specs) as camera_capture:
        controller.connect()
        try:
            input("Policy is ready. Press Enter to start execution...")
            next_tick = time.perf_counter()
            while True:
                now = time.perf_counter()
                if deadline is not None and now >= deadline:
                    break
                if args.max_steps > 0 and step_count >= args.max_steps:
                    break
                if now < next_tick:
                    time.sleep(min(next_tick - now, 0.005))
                    continue

                include_gripper = qpos_dim >= 7
                qpos = controller.read_joint_positions(include_gripper=include_gripper)
                if qpos is None:
                    raise RuntimeError("failed to read robot joint feedback")
                qpos = np.asarray(qpos[:qpos_dim], dtype=np.float32)
                frames = camera_capture.read()

                if (
                    current_chunk is None
                    or chunk_cursor >= chunk_size
                    or steps_since_query >= args.query_interval
                ):
                    image_tensor = _prepare_images(
                        frames,
                        camera_names,
                        image_downsample,
                    ).to(device=device, non_blocking=True)
                    qpos_tensor = torch.from_numpy(qpos).unsqueeze(0).to(device=device)
                    with torch.no_grad():
                        pred = model(image_tensor, qpos_tensor)
                    current_chunk = pred.squeeze(0).detach().cpu().numpy()
                    chunk_cursor = 0
                    steps_since_query = 0

                action = np.asarray(current_chunk[chunk_cursor], dtype=np.float32)
                if action.shape[0] < 6:
                    raise RuntimeError(f"policy action_dim must be at least 6, got {action.shape[0]}")

                current_arm = qpos[:6].astype(float)
                arm_target = action[:6].astype(float)
                arm_target = _safe_arm_target(
                    arm_target,
                    current_arm,
                    max_delta=args.max_target_delta,
                )
                controller.submit_joint_target(
                    arm_target,
                    dt=period,
                    start_q=current_arm,
                    block=not args.nonblocking,
                    start_limit_tolerance_rad=args.start_limit_tolerance,
                )
                if action_dim >= 7 and not args.disable_gripper:
                    controller.submit_gripper_target(float(action[6]))

                step_count += 1
                chunk_cursor += 1
                steps_since_query += 1
                if step_count % max(1, int(args.rate)) == 0:
                    print(f"executed {step_count} policy steps", end="\r", flush=True)
                next_tick += period
        except KeyboardInterrupt:
            print("\nStopping policy execution...")
        finally:
            controller.stop_motion()
            controller.close(disable=not args.keep_enabled)
    print(f"\nPolicy execution finished after {step_count} steps")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", required=True, help="Path to policy_best.pt")
    parser.add_argument(
        "--camera",
        action="append",
        required=True,
        help="Camera spec name=source[:WIDTHxHEIGHT], matching training camera names",
    )
    parser.add_argument("--cameras", default=None, help="Override comma-separated camera names")
    parser.add_argument("--image-downsample", type=int, default=0)
    parser.add_argument("--rate", type=float, default=10.0, help="Execution loop rate in Hz")
    parser.add_argument("--query-interval", type=int, default=10)
    parser.add_argument("--duration", type=float, default=0.0)
    parser.add_argument("--max-steps", type=int, default=0)
    parser.add_argument("--max-target-delta", type=float, default=0.08)
    parser.add_argument("--max-joint-step", type=float, default=0.08)
    parser.add_argument("--max-joint-velocity", type=float, default=0.6)
    parser.add_argument("--start-limit-tolerance", type=float, default=0.02)
    parser.add_argument("--disable-gripper", action="store_true")
    parser.add_argument("--nonblocking", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--keep-enabled", action="store_true")
    parser.add_argument("--can", default="can0")
    parser.add_argument("--backend", default="socketcan", choices=["socketcan", "slcan"])
    parser.add_argument("--serial-port", default=None)
    parser.add_argument("--can-bitrate", type=int, default=1_000_000)
    parser.add_argument("--host-can-id", type=int, default=253)
    parser.add_argument("--setup-can", action="store_true")
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args(argv)

    if args.rate <= 0:
        parser.error("--rate must be positive")
    if args.query_interval <= 0:
        parser.error("--query-interval must be positive")
    if args.image_downsample < 0:
        parser.error("--image-downsample cannot be negative")

    run_policy(args)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
