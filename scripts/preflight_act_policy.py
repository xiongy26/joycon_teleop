#!/usr/bin/env python3
"""Preflight checks before deploying an ACT-style policy on EL-A3."""

from __future__ import annotations

import argparse
import sys
import time
from dataclasses import dataclass
from pathlib import Path

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from act_data.camera import MultiCameraCapture, parse_camera_specs
from ela3_motion import ELA3MotionOptions, ELA3RealtimeController
from scripts.run_act_policy import (
    _build_policy,
    _load_torch,
    _prepare_images,
    _safe_arm_target,
)
from ela3_motion.realtime import (
    DEFAULT_ARM_JOINT_LIMITS_LOWER,
    DEFAULT_ARM_JOINT_LIMITS_UPPER,
)


@dataclass
class CheckResult:
    name: str
    ok: bool
    message: str


def _print_result(result: CheckResult) -> None:
    status = "OK" if result.ok else "FAIL"
    print(f"[{status}] {result.name}: {result.message}")


def _camera_placeholder(camera_names: list[str], height: int, width: int) -> dict[str, np.ndarray]:
    return {
        name: np.zeros((height, width, 3), dtype=np.uint8)
        for name in camera_names
    }


def _summarize_action(action: np.ndarray, current_arm: np.ndarray, max_target_delta: float) -> str:
    raw_delta = action[:6].astype(float) - current_arm
    clipped = _safe_arm_target(
        action[:6].astype(float),
        current_arm,
        max_delta=max_target_delta,
    )
    clipped_delta = clipped - current_arm
    return (
        f"raw_delta_inf={np.linalg.norm(raw_delta, ord=np.inf):.4f} rad, "
        f"clipped_delta_inf={np.linalg.norm(clipped_delta, ord=np.inf):.4f} rad"
    )


def _placeholder_qpos(qpos_dim: int) -> np.ndarray:
    arm = 0.5 * (DEFAULT_ARM_JOINT_LIMITS_LOWER + DEFAULT_ARM_JOINT_LIMITS_UPPER)
    values = np.zeros((qpos_dim,), dtype=np.float32)
    values[:6] = arm.astype(np.float32)
    return values


def run_preflight(args: argparse.Namespace) -> int:
    torch = _load_torch()
    device = torch.device(args.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is not available. Use --device cpu.")

    results: list[CheckResult] = []
    checkpoint_path = Path(args.checkpoint)

    try:
        model, model_config = _build_policy(checkpoint_path, str(device), args)
        camera_names = list(model_config["camera_names"])
        image_downsample = int(args.image_downsample or model_config.get("image_downsample", 1))
        qpos_dim = int(model_config["qpos_dim"])
        action_dim = int(model_config["action_dim"])
        chunk_size = int(model_config["chunk_size"])
        results.append(
            CheckResult(
                "checkpoint",
                True,
                (
                    f"loaded qpos_dim={qpos_dim}, action_dim={action_dim}, "
                    f"chunk_size={chunk_size}, cameras={camera_names}"
                ),
            )
        )
    except Exception as exc:
        results.append(CheckResult("checkpoint", False, str(exc)))
        for result in results:
            _print_result(result)
        return 1

    frames = None
    if args.skip_cameras:
        frames = _camera_placeholder(camera_names, args.placeholder_height, args.placeholder_width)
        results.append(
            CheckResult(
                "cameras",
                True,
                f"skipped; using placeholder {args.placeholder_width}x{args.placeholder_height}",
            )
        )
    else:
        try:
            camera_specs = parse_camera_specs(args.camera)
            spec_names = [camera.name for camera in camera_specs]
            missing = sorted(set(camera_names) - set(spec_names))
            extra = sorted(set(spec_names) - set(camera_names))
            if missing:
                raise RuntimeError(f"missing camera specs for checkpoint cameras: {missing}")
            with MultiCameraCapture(camera_specs) as capture:
                frames = capture.read()
            shapes = {name: tuple(frames[name].shape) for name in camera_names}
            msg = f"read frames {shapes}"
            if extra:
                msg += f"; unused specs={extra}"
            results.append(CheckResult("cameras", True, msg))
        except Exception as exc:
            results.append(CheckResult("cameras", False, str(exc)))

    qpos = None
    controller = None
    if args.skip_robot:
        qpos = _placeholder_qpos(qpos_dim)
        results.append(
            CheckResult(
                "robot_feedback",
                True,
                f"skipped; using valid placeholder arm={np.round(qpos[:6], 4).tolist()}",
            )
        )
    else:
        options = ELA3MotionOptions(
            can_name=args.can,
            backend=args.backend,
            serial_port=args.serial_port,
            setup_can=args.setup_can,
            can_bitrate=args.can_bitrate,
            host_can_id=args.host_can_id,
            dry_run=True,
            disable_on_finish=not args.keep_enabled,
            command_rate_hz=args.rate,
            max_joint_step_rad=args.max_joint_step,
            max_joint_velocity_rad_s=args.max_joint_velocity,
        )
        controller = ELA3RealtimeController(options)
        try:
            controller.connect()
            include_gripper = qpos_dim >= 7
            deadline = time.time() + max(0.1, args.feedback_timeout)
            while time.time() < deadline:
                raw = controller.read_joint_positions(include_gripper=include_gripper)
                if raw is not None:
                    qpos = np.asarray(raw[:qpos_dim], dtype=np.float32)
                    break
                time.sleep(0.05)
            if qpos is None:
                raise RuntimeError("timed out waiting for valid joint feedback")
            if np.any(~np.isfinite(qpos)):
                raise RuntimeError("qpos contains NaN/Inf")
            results.append(
                CheckResult(
                    "robot_feedback",
                    True,
                    f"qpos_shape={qpos.shape}, arm={np.round(qpos[:6], 4).tolist()}",
                )
            )
        except Exception as exc:
            results.append(CheckResult("robot_feedback", False, str(exc)))

    action = None
    if frames is not None and qpos is not None:
        try:
            image_tensor = _prepare_images(frames, camera_names, image_downsample).to(device)
            qpos_tensor = torch.from_numpy(qpos).unsqueeze(0).to(device)
            with torch.no_grad():
                pred = model(image_tensor, qpos_tensor)
            chunk = pred.squeeze(0).detach().cpu().numpy()
            if chunk.shape != (chunk_size, action_dim):
                raise RuntimeError(f"unexpected policy output shape {chunk.shape}")
            if np.any(~np.isfinite(chunk)):
                raise RuntimeError("policy output contains NaN/Inf")
            action = np.asarray(chunk[0], dtype=np.float32)
            results.append(
                CheckResult(
                    "policy_inference",
                    True,
                    f"output_shape={chunk.shape}, first_action={np.round(action, 4).tolist()}",
                )
            )
        except Exception as exc:
            results.append(CheckResult("policy_inference", False, str(exc)))
    else:
        results.append(
            CheckResult(
                "policy_inference",
                False,
                "skipped because camera frames or qpos were unavailable",
            )
        )

    if action is not None and qpos is not None:
        try:
            if action.shape[0] < 6:
                raise RuntimeError(f"policy action_dim must be at least 6, got {action.shape[0]}")
            current_arm = qpos[:6].astype(float)
            arm_target = _safe_arm_target(
                action[:6].astype(float),
                current_arm,
                max_delta=args.max_target_delta,
            )
            if args.skip_robot:
                from ela3_motion.safety import validate_arm_joint_plan

                validate_arm_joint_plan(
                    [current_arm, arm_target],
                    lower_limits=DEFAULT_ARM_JOINT_LIMITS_LOWER,
                    upper_limits=DEFAULT_ARM_JOINT_LIMITS_UPPER,
                    max_joint_step_rad=args.max_joint_step,
                )
                msg = "validated against default limits; " + _summarize_action(
                    action,
                    current_arm,
                    args.max_target_delta,
                )
            else:
                assert controller is not None
                controller.submit_joint_target(
                    arm_target,
                    dt=1.0 / float(args.rate),
                    start_q=current_arm,
                    block=False,
                    start_limit_tolerance_rad=args.start_limit_tolerance,
                )
                msg = "realtime dry-run accepted first target; " + _summarize_action(
                    action,
                    current_arm,
                    args.max_target_delta,
                )
            if action_dim >= 7 and not args.disable_gripper:
                msg += f", gripper_target={float(action[6]):.4f}"
            results.append(CheckResult("motion_safety", True, msg))
        except Exception as exc:
            results.append(CheckResult("motion_safety", False, str(exc)))

    if controller is not None:
        controller.close(disable=not args.keep_enabled)

    for result in results:
        _print_result(result)
    return 0 if all(result.ok for result in results) else 1


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument(
        "--camera",
        action="append",
        default=[],
        help="Camera spec name=source[:WIDTHxHEIGHT], matching training camera names",
    )
    parser.add_argument("--cameras", default=None, help="Override comma-separated camera names")
    parser.add_argument("--image-downsample", type=int, default=0)
    parser.add_argument("--skip-cameras", action="store_true")
    parser.add_argument("--skip-robot", action="store_true")
    parser.add_argument("--placeholder-width", type=int, default=640)
    parser.add_argument("--placeholder-height", type=int, default=480)
    parser.add_argument("--rate", type=float, default=10.0)
    parser.add_argument("--max-target-delta", type=float, default=0.08)
    parser.add_argument("--max-joint-step", type=float, default=0.08)
    parser.add_argument("--max-joint-velocity", type=float, default=0.6)
    parser.add_argument("--start-limit-tolerance", type=float, default=0.02)
    parser.add_argument("--feedback-timeout", type=float, default=3.0)
    parser.add_argument("--disable-gripper", action="store_true")
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
    if args.image_downsample < 0:
        parser.error("--image-downsample cannot be negative")
    if not args.skip_cameras and not args.camera:
        parser.error("--camera is required unless --skip-cameras is set")
    if args.placeholder_width <= 0 or args.placeholder_height <= 0:
        parser.error("placeholder dimensions must be positive")

    return run_preflight(args)


if __name__ == "__main__":
    raise SystemExit(main())
