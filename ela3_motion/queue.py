"""EL-A3 SDK queue conversion for prepared arm joint trajectories."""

from __future__ import annotations

from typing import Optional, Sequence

import numpy as np

from .types import QueueBuildResult


def build_sdk_queue_trajectory(
    joint_plan: Sequence[np.ndarray],
    *,
    sample_period_s: float,
    control_period_s: float,
    max_joint_velocity_rad_s: float,
    start_q: Optional[np.ndarray] = None,
) -> QueueBuildResult:
    from el_a3_sdk.trajectory import CubicSplinePlanner, TrajectoryPoint

    if not joint_plan:
        return QueueBuildResult(
            sdk_points=[],
            queue_waypoints=0,
            queue_points=0,
            duration_s=0.0,
        )

    raw_waypoints = [np.asarray(q, dtype=float).copy() for q in joint_plan]
    raw_times = [idx * sample_period_s for idx in range(len(raw_waypoints))]

    waypoints: list[np.ndarray] = []
    times: list[float] = []
    if start_q is not None:
        seed = np.asarray(start_q, dtype=float).copy()
        if np.all(np.isfinite(seed)) and seed.shape == raw_waypoints[0].shape:
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
        sdk_points = [
            TrajectoryPoint(
                time=0.0,
                positions=waypoints[0].tolist(),
                velocities=[0.0] * len(waypoints[0]),
                accelerations=[0.0] * len(waypoints[0]),
            )
        ]
        return QueueBuildResult(
            sdk_points=sdk_points,
            queue_waypoints=1,
            queue_points=1,
            duration_s=0.0,
        )

    durations = []
    for i in range(1, len(times)):
        duration = max(times[i] - times[i - 1], sample_period_s)
        max_delta = float(np.linalg.norm(waypoints[i] - waypoints[i - 1], ord=np.inf))
        if max_joint_velocity_rad_s > 0:
            duration = max(duration, max_delta / max_joint_velocity_rad_s)
        durations.append(duration)

    sdk_points = CubicSplinePlanner.plan_waypoints(
        [q.tolist() for q in waypoints],
        durations,
        dt=max(control_period_s, 1e-4),
    )
    if max_joint_velocity_rad_s > 0:
        for point in sdk_points:
            if point.velocities:
                point.velocities = np.clip(
                    np.asarray(point.velocities, dtype=float),
                    -max_joint_velocity_rad_s,
                    max_joint_velocity_rad_s,
                ).tolist()

    return QueueBuildResult(
        sdk_points=sdk_points,
        queue_waypoints=len(waypoints),
        queue_points=len(sdk_points),
        duration_s=sdk_points[-1].time if sdk_points else 0.0,
    )
