#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Minimal real robot trajectory module for standalone GUI."""

from __future__ import annotations

import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parent.parent
SDK_ROOT = PROJECT_ROOT / "el_a3_sdk"


@dataclass
class RealRobotOptions:
    """Real robot execution parameters."""
    can_name: str = "can0"
    backend: str = "socketcan"
    serial_port: Optional[str] = None
    serial_baudrate: int = 115200
    can_bitrate: int = 1000000
    host_can_id: int = 0
    kp: float = 0.5
    kd: float = 0.01
    control_rate_hz: float = 200.0
    command_rate_hz: float = 200.0
    max_joint_step_rad: float = 0.05
    max_joint_velocity_rad_s: float = 2.0
    ik_tolerance_m: float = 0.001
    gripper_pp_velocity_rad_s: float = 1.0
    gripper_pp_acceleration_rad_s2: float = 2.0
    lift_height_m: float = 0.1
    dry_run: bool = False
    disable_on_finish: bool = True
    hold_on_finish: bool = True
    mujoco_viewer: bool = True
    setup_can: bool = True
    feedback_timeout_s: float = 2.0
    feedback_log_interval_s: float = 1.0


def _ensure_sdk_on_path() -> None:
    if str(SDK_ROOT) not in sys.path:
        sys.path.insert(0, str(SDK_ROOT))


def _sdk_resource_kwargs() -> dict:
    inertia_path = SDK_ROOT / "resources" / "config" / "inertia_params.yaml"
    legacy_urdf_path = SDK_ROOT / "resources" / "urdf" / "el_a3_legacy.urdf"
    kwargs = {
        "per_joint_kd_min": {4: 0.005, 5: 0.005, 6: 0.005, 7: 0.02},
        "per_joint_kd_max": {4: 0.10, 5: 0.05, 6: 0.05, 7: 0.10},
        "gravity_joint_scale": {4: 2.0},
    }
    if inertia_path.exists():
        kwargs["inertia_config_path"] = str(inertia_path)
    if legacy_urdf_path.exists():
        kwargs["urdf_path"] = str(legacy_urdf_path)
    return kwargs


def _get_can_state(can_name: str) -> str:
    operstate_file = Path(f"/sys/class/net/{can_name}/operstate")
    try:
        raw = operstate_file.read_text().strip().upper()
        if raw == "UNKNOWN":
            flags_file = Path(f"/sys/class/net/{can_name}/flags")
            if flags_file.exists():
                flags = int(flags_file.read_text().strip(), 16)
                return "UP" if flags & 0x1 else "DOWN"
        return raw
    except (FileNotFoundError, OSError):
        return "UNKNOWN"


def _ensure_socketcan_ready(options: RealRobotOptions) -> None:
    if options.backend == "slcan":
        return

    state = _get_can_state(options.can_name)
    if state == "UP":
        return

    if options.setup_can:
        _ensure_sdk_on_path()
        try:
            from MotorStudio.utils.can_utils import setup_can_interface
        except Exception as exc:
            raise RuntimeError(f"Cannot import MotorStudio CAN utils: {exc}") from exc

        ok, msg = setup_can_interface(options.can_name, options.can_bitrate)
        if not ok:
            raise RuntimeError(
                f"CAN interface {options.can_name} auto-setup failed: {msg}\n"
                f"Run manually: sudo ./el_a3_sdk/scripts/setup_can.sh "
                f"{options.can_name} {options.can_bitrate}"
            )
        state = _get_can_state(options.can_name)
        if state == "UP":
            return

    raise RuntimeError(
        f"CAN interface {options.can_name} is not UP (state: {state}).\n"
        f"Run: sudo ./el_a3_sdk/scripts/setup_can.sh "
        f"{options.can_name} {options.can_bitrate}"
    )


def _connect_real_arm(options: RealRobotOptions):
    _ensure_sdk_on_path()
    _ensure_socketcan_ready(options)
    from el_a3_sdk import ELA3Interface

    kwargs = _sdk_resource_kwargs()
    kwargs.update(
        can_name=options.can_name,
        host_can_id=options.host_can_id,
        default_kp=options.kp,
        default_kd=options.kd,
        backend=options.backend,
        serial_port=options.serial_port,
        serial_baudrate=options.serial_baudrate,
        can_bitrate=options.can_bitrate,
        pp_velocity=options.gripper_pp_velocity_rad_s,
        pp_acceleration=options.gripper_pp_acceleration_rad_s2,
    )
    arm = ELA3Interface(**kwargs)
    if not arm.ConnectPort():
        raise RuntimeError(f"Failed to connect EL-A3: {options.can_name}")
    if not arm.EnableArm():
        arm.DisconnectPort()
        raise RuntimeError("EL-A3 motor enable failed")
    arm.SetPositionPD(kp=options.kp, kd=options.kd)
    arm.start_control_loop(rate_hz=options.control_rate_hz)
    return arm


def build_sdk_queue_trajectory(
    joint_plan: list[np.ndarray],
    *,
    sample_period_s: float,
    control_period_s: float,
    max_joint_velocity_rad_s: float,
    start_q: Optional[np.ndarray] = None,
) -> tuple[list, dict]:
    """Build SDK trajectory from joint plan."""
    _ensure_sdk_on_path()
    from core.ik_core import N_ARM

    if not joint_plan:
        return [], {"queue_waypoints": 0, "queue_points": 0, "duration_s": 0.0}

    raw_waypoints = [np.array(q, dtype=float, copy=True) for q in joint_plan]
    raw_times = [idx * sample_period_s for idx in range(len(raw_waypoints))]

    waypoints: list[np.ndarray] = []
    times: list[float] = []
    if start_q is not None:
        seed = np.array(start_q, dtype=float, copy=True)
        if np.all(np.isfinite(seed)) and len(seed) == N_ARM:
            waypoints.append(seed)
            times.append(0.0)

    if not waypoints:
        waypoints.append(raw_waypoints[0])
        times.append(0.0)
    elif float(np.linalg.norm(raw_waypoints[0] - waypoints[-1], ord=np.inf)) > 1e-9:
        first_delta = float(np.linalg.norm(raw_waypoints[0] - waypoints[-1], ord=np.inf))
        min_duration = (
            first_delta / max_joint_velocity_rad_s
            if max_joint_velocity_rad_s > 0
            else sample_period_s
        )
        times.append(max(sample_period_s, min_duration))
        waypoints.append(raw_waypoints[0])

    for idx, q in enumerate(raw_waypoints[1:], start=1):
        is_final = idx == len(raw_waypoints) - 1
        changed = float(np.linalg.norm(q - waypoints[-1], ord=np.inf)) > 1e-9
        if changed or is_final:
            t = raw_times[idx]
            if t <= times[-1]:
                t = times[-1] + sample_period_s
            waypoints.append(q)
            times.append(t)

    if len(waypoints) == 1:
        from el_a3_sdk.trajectory import TrajectoryPoint

        trajectory = [
            TrajectoryPoint(
                time=0.0,
                positions=waypoints[0].tolist(),
                velocities=[0.0] * N_ARM,
                accelerations=[0.0] * N_ARM,
            )
        ]
        return trajectory, {
            "queue_waypoints": 1,
            "queue_points": 1,
            "duration_s": 0.0,
        }

    durations = []
    for i in range(1, len(times)):
        duration = max(times[i] - times[i - 1], sample_period_s)
        max_delta = float(np.linalg.norm(waypoints[i] - waypoints[i - 1], ord=np.inf))
        if max_joint_velocity_rad_s > 0:
            duration = max(duration, max_delta / max_joint_velocity_rad_s)
        durations.append(duration)

    from el_a3_sdk.trajectory import CubicSplinePlanner

    trajectory = CubicSplinePlanner.plan_waypoints(
        [q.tolist() for q in waypoints],
        durations,
        dt=max(control_period_s, 1e-4),
    )
    if max_joint_velocity_rad_s > 0:
        for point in trajectory:
            if point.velocities:
                point.velocities = np.clip(
                    np.asarray(point.velocities, dtype=float),
                    -max_joint_velocity_rad_s,
                    max_joint_velocity_rad_s,
                ).tolist()

    return trajectory, {
        "queue_waypoints": len(waypoints),
        "queue_points": len(trajectory),
        "duration_s": trajectory[-1].time if trajectory else 0.0,
    }
