"""Adapters from planner outputs to EL-A3 deployment trajectories."""

from __future__ import annotations

from typing import Optional, Sequence

import numpy as np

from .types import ELA3Trajectory


def trajectory_from_joint_plan(
    joint_plan: Sequence[Sequence[float] | np.ndarray],
    *,
    dt: float,
    gripper_plan: Optional[Sequence[float]] = None,
    metadata: Optional[dict] = None,
) -> ELA3Trajectory:
    return ELA3Trajectory.from_joint_plan(
        joint_plan,
        dt=dt,
        gripper_plan=gripper_plan,
        metadata=metadata,
    )


def trajectory_from_motion_plan(
    plan,
    *,
    gripper_plan: Optional[Sequence[float]] = None,
    metadata: Optional[dict] = None,
) -> ELA3Trajectory:
    plan_metadata = {
        "source": "motion_plan",
        "ik_method": getattr(plan, "ik_method", None),
        "speed_mode": getattr(plan, "speed_mode", None),
        "visual_strokes": getattr(plan, "visual_strokes", None),
    }
    if metadata:
        plan_metadata.update(metadata)
    return ELA3Trajectory.from_joint_plan(
        getattr(plan, "joint_plan"),
        dt=float(getattr(plan, "sample_period_s")),
        gripper_plan=gripper_plan,
        metadata=plan_metadata,
    )
