"""Reusable EL-A3 motion deployment API.

This package sits above `el_a3_sdk`: it accepts joint trajectories from
MuJoCo or any other planner, prepares them for safe EL-A3 execution, and then
optionally submits the resulting SDK queue through the backend.
"""

from .adapter import trajectory_from_joint_plan, trajectory_from_motion_plan
from .bridge import build_start_bridge, maybe_prepend_start_bridge
from .deployment import prepare_trajectory_for_deployment
from .manager import ELA3MotionManager
from .options import ELA3MotionOptions, ExecutionMode, FinishPolicy, StartPolicy
from .queue import build_sdk_queue_trajectory
from .realtime import ELA3RealtimeController, RealtimeCommandResult
from .safety import densify_arm_joint_plan, validate_arm_joint_plan
from .types import (
    ELA3Trajectory,
    MotionExecutionResult,
    PreparedTrajectory,
    QueueBuildResult,
)

__all__ = [
    "ELA3MotionManager",
    "ELA3MotionOptions",
    "ELA3RealtimeController",
    "ELA3Trajectory",
    "ExecutionMode",
    "FinishPolicy",
    "MotionExecutionResult",
    "PreparedTrajectory",
    "QueueBuildResult",
    "RealtimeCommandResult",
    "StartPolicy",
    "build_sdk_queue_trajectory",
    "build_start_bridge",
    "densify_arm_joint_plan",
    "maybe_prepend_start_bridge",
    "prepare_trajectory_for_deployment",
    "trajectory_from_joint_plan",
    "trajectory_from_motion_plan",
    "validate_arm_joint_plan",
]
