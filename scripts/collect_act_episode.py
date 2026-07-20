#!/usr/bin/env python3
"""Collect one camera + joint-state ACT demonstration episode."""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from act_data.camera import MultiCameraCapture, parse_camera_specs
from act_data.episode_writer import ACTEpisodeWriter, EpisodeMetadata
from act_data.robot import ELRobotReader, ZeroTorqueConfig
from ela3_motion.options import ELA3MotionOptions


def _next_episode_path(output_dir: Path, task_name: str) -> tuple[int, Path]:
    output_dir.mkdir(parents=True, exist_ok=True)
    prefix = f"episode_{task_name}_"
    existing = sorted(output_dir.glob(f"{prefix}*.hdf5"))
    max_id = -1
    for path in existing:
        stem = path.stem
        try:
            max_id = max(max_id, int(stem.rsplit("_", 1)[1]))
        except (IndexError, ValueError):
            continue
    episode_id = max_id + 1
    return episode_id, output_dir / f"{prefix}{episode_id:06d}.hdf5"


def _wait_for_enter(prompt: str) -> None:
    input(prompt)


def collect(args: argparse.Namespace) -> Path:
    cameras = parse_camera_specs(args.camera)
    episode_id, output_path = _next_episode_path(Path(args.output_dir), args.task_name)
    metadata = EpisodeMetadata(
        task_name=args.task_name,
        episode_id=episode_id,
        sample_rate_hz=args.rate,
        camera_names=[camera.name for camera in cameras],
        qpos_dim=7,
        action_dim=7,
    )

    options = ELA3MotionOptions(
        can_name=args.can,
        backend=args.backend,
        serial_port=args.serial_port,
        setup_can=args.setup_can,
        can_bitrate=args.can_bitrate,
        host_can_id=args.host_can_id,
    )
    zt = ZeroTorqueConfig(
        enabled=not args.no_zero_torque,
        gravity=not args.no_gravity,
        update_rate_hz=args.gravity_rate,
    )

    print(f"Opening cameras: {[camera.name for camera in cameras]}")
    print(f"Output: {output_path}")
    with MultiCameraCapture(cameras) as camera_capture, ELRobotReader(
        options, zero_torque=zt
    ) as robot:
        _wait_for_enter("Robot is in teaching mode. Press Enter to start recording...")
        period = 1.0 / float(args.rate)
        deadline = None if args.duration <= 0 else time.perf_counter() + args.duration
        next_tick = time.perf_counter()

        with ACTEpisodeWriter(output_path, metadata) as writer:
            try:
                while True:
                    now = time.perf_counter()
                    if deadline is not None and now >= deadline:
                        break
                    if now < next_tick:
                        time.sleep(min(next_tick - now, 0.005))
                        continue

                    qpos = robot.read_qpos(include_gripper=True)
                    frames = camera_capture.read()
                    writer.append(qpos=qpos, images=frames)
                    if writer.count % max(1, int(args.rate)) == 0:
                        print(f"recorded {writer.count} frames", end="\r", flush=True)
                    next_tick += period
            except KeyboardInterrupt:
                print("\nStopping recording...")

        print(f"\nSaved {output_path}")
    return output_path


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Collect one ACT HDF5 episode from EL-A3 joint feedback and cameras."
    )
    parser.add_argument("--task-name", required=True, help="Task name, e.g. pick_cube")
    parser.add_argument(
        "--camera",
        action="append",
        required=True,
        help="Camera spec name=source[:WIDTHxHEIGHT], e.g. cam_high=0:640x480",
    )
    parser.add_argument("--output-dir", default="data/act", help="Episode output directory")
    parser.add_argument("--rate", type=float, default=30.0, help="Sampling rate in Hz")
    parser.add_argument(
        "--duration",
        type=float,
        default=0.0,
        help="Fixed recording duration in seconds; 0 records until Ctrl+C",
    )
    parser.add_argument("--can", default="can0", help="SocketCAN interface name")
    parser.add_argument("--backend", default="socketcan", choices=["socketcan", "slcan"])
    parser.add_argument("--serial-port", default=None, help="Serial port for slcan backend")
    parser.add_argument("--can-bitrate", type=int, default=1_000_000)
    parser.add_argument("--host-can-id", type=int, default=253)
    parser.add_argument("--setup-can", action="store_true", help="Try to bring CAN up")
    parser.add_argument("--no-zero-torque", action="store_true")
    parser.add_argument("--no-gravity", action="store_true")
    parser.add_argument("--gravity-rate", type=float, default=100.0)
    args = parser.parse_args(argv)

    if args.rate <= 0:
        parser.error("--rate must be positive")

    collect(args)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
