"""Pure deployment preparation from joint trajectory to SDK queue."""

from __future__ import annotations

from typing import Optional

import numpy as np

from .bridge import maybe_prepend_start_bridge
from .options import ELA3MotionOptions, StartPolicy
from .queue import build_sdk_queue_trajectory
from .safety import densify_arm_joint_plan, validate_arm_joint_plan
from .types import ELA3Trajectory, PreparedTrajectory


def prepare_trajectory_for_deployment(
    trajectory: ELA3Trajectory,
    *,
    options: ELA3MotionOptions,
    start_q: Optional[np.ndarray],
    start_policy: StartPolicy,
    dry_run: bool,
    lower_limits: np.ndarray,
    upper_limits: np.ndarray,
) -> PreparedTrajectory:
    """Prepare a trajectory for EL-A3 SDK submission without touching hardware."""

    prepared_plan, bridge_inserted, diagnostics = maybe_prepend_start_bridge(
        trajectory.arm,
        start_q=start_q,
        dt=trajectory.dt,
        start_policy=start_policy,
        dry_run=dry_run,
        near_tolerance_rad=options.start_near_tolerance_rad,
        max_joint_velocity_rad_s=options.max_joint_velocity_rad_s,
        min_duration_s=options.bridge_min_duration_s,
    )

    dense_plan = densify_arm_joint_plan(
        prepared_plan,
        max_joint_step_rad=options.max_joint_step_rad,
    )
    validate_arm_joint_plan(
        dense_plan,
        lower_limits=lower_limits,
        upper_limits=upper_limits,
        max_joint_step_rad=options.max_joint_step_rad,
    )
    diagnostics["prepared_points"] = len(prepared_plan)
    diagnostics["dense_points"] = len(dense_plan)

    queue = build_sdk_queue_trajectory(
        dense_plan,
        sample_period_s=trajectory.dt,
        control_period_s=1.0 / float(options.control_rate_hz),
        max_joint_velocity_rad_s=options.max_joint_velocity_rad_s,
        start_q=None,
    )

    return PreparedTrajectory(
        trajectory=trajectory,
        prepared_plan=prepared_plan,
        dense_plan=dense_plan,
        queue=queue,
        bridge_inserted=bridge_inserted,
        diagnostics=diagnostics,
    )
