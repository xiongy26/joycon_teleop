"""Backward-compatible aggregate exports for EL-A3 trajectory deployment.

New code should prefer the narrower modules:
`types`, `bridge`, `safety`, `queue`, `adapter`, and `deployment`.
"""

from .adapter import trajectory_from_joint_plan, trajectory_from_motion_plan
from .bridge import build_start_bridge, maybe_prepend_start_bridge
from .deployment import prepare_trajectory_for_deployment
from .queue import build_sdk_queue_trajectory
from .safety import densify_arm_joint_plan, validate_arm_joint_plan
from .types import ELA3Trajectory, MotionExecutionResult, PreparedTrajectory, QueueBuildResult

__all__ = [
    "ELA3Trajectory",
    "MotionExecutionResult",
    "PreparedTrajectory",
    "QueueBuildResult",
    "build_sdk_queue_trajectory",
    "build_start_bridge",
    "densify_arm_joint_plan",
    "maybe_prepend_start_bridge",
    "prepare_trajectory_for_deployment",
    "trajectory_from_joint_plan",
    "trajectory_from_motion_plan",
    "validate_arm_joint_plan",
]
