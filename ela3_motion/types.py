"""Public trajectory and result data types for EL-A3 motion deployment."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Optional, Sequence

import numpy as np


@dataclass
class ELA3Trajectory:
    """Backend-independent 6-axis EL-A3 arm trajectory."""

    arm: list[np.ndarray]
    dt: float
    gripper: Optional[list[float]] = None
    metadata: dict[str, Any] = field(default_factory=dict)

    @classmethod
    def from_joint_plan(
        cls,
        joint_plan: Sequence[Sequence[float] | np.ndarray],
        *,
        dt: float,
        gripper_plan: Optional[Sequence[float]] = None,
        metadata: Optional[dict[str, Any]] = None,
    ) -> "ELA3Trajectory":
        if dt <= 0:
            raise ValueError("trajectory dt must be positive")
        if not joint_plan:
            raise ValueError("joint trajectory is empty")

        arm: list[np.ndarray] = []
        for idx, point in enumerate(joint_plan):
            arr = np.asarray(point, dtype=float)
            if arr.shape != (6,):
                raise ValueError(
                    f"joint point {idx} must have shape (6,), got {arr.shape}"
                )
            if np.any(~np.isfinite(arr)):
                raise ValueError(f"joint point {idx} contains NaN/Inf")
            arm.append(arr.copy())

        gripper = None
        if gripper_plan is not None:
            if len(gripper_plan) != len(arm):
                raise ValueError("gripper_plan length must match joint_plan length")
            gripper_arr = np.asarray(gripper_plan, dtype=float)
            if np.any(~np.isfinite(gripper_arr)):
                raise ValueError("gripper_plan contains NaN/Inf")
            gripper = gripper_arr.tolist()

        return cls(arm=arm, dt=float(dt), gripper=gripper, metadata=metadata or {})


@dataclass
class QueueBuildResult:
    sdk_points: list
    queue_waypoints: int
    queue_points: int
    duration_s: float


@dataclass
class PreparedTrajectory:
    """Trajectory after start bridging, densification, validation, and queueing."""

    trajectory: ELA3Trajectory
    prepared_plan: list[np.ndarray]
    dense_plan: list[np.ndarray]
    queue: QueueBuildResult
    bridge_inserted: bool
    diagnostics: dict[str, Any] = field(default_factory=dict)


@dataclass
class MotionExecutionResult:
    dry_run: bool
    interrupted: bool
    execution_mode: str
    sent_points: int
    duration_s: float
    queue_waypoints: int
    queue_points: int
    start_q: Optional[list[float]]
    first_target_q: Optional[list[float]]
    final_q: Optional[list[float]]
    bridge_inserted: bool
    diagnostics: dict[str, Any] = field(default_factory=dict)
