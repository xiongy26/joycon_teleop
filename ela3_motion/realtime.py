from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Protocol, Sequence, Union

import numpy as np

from .backend import ELA3Backend
from .options import ELA3MotionOptions
from .queue import build_sdk_queue_trajectory
from .safety import densify_arm_joint_plan, validate_arm_joint_plan
from .types import MotionExecutionResult

N_ARM = 6
DEFAULT_ARM_JOINT_LIMITS_LOWER = np.radians(
    np.array([-160.0, 0.0, -230.0, -60.0, -90.0, -90.0], dtype=float)
)
DEFAULT_ARM_JOINT_LIMITS_UPPER = np.radians(
    np.array([160.0, 210.0, 0.0, 90.0, 90.0, 90.0], dtype=float)
)

DiagnosticScalar = Union[str, int, float, bool, None]
DiagnosticValue = Union[DiagnosticScalar, List[float]]
Diagnostics = Dict[str, DiagnosticValue]


class RealtimeBackend(Protocol):
    @property
    def is_connected(self) -> bool: ...

    def connect_enable_and_start(self) -> None: ...

    def read_arm_joints(self) -> Optional[np.ndarray]: ...

    def execute_queue(
        self,
        sdk_points: list,
        *,
        block: bool = True,
    ) -> MotionExecutionResult: ...

    def close(self, *, disable: bool = True) -> None: ...

    def stop_motion(self) -> None: ...


@dataclass(frozen=True)
class RealtimeCommandError(Exception):
    message: str

    def __str__(self) -> str:
        return self.message


@dataclass(frozen=True)
class MissingFeedbackError(RealtimeCommandError):
    pass


@dataclass(frozen=True)
class RealtimeCommandResult:
    dry_run: bool
    submitted_points: int
    duration_s: float
    start_q: List[float]
    target_q: List[float]
    queue_points: int
    diagnostics: Diagnostics = field(default_factory=dict)


class ELA3RealtimeController:
    def __init__(
        self,
        options: Optional[ELA3MotionOptions] = None,
        *,
        backend: Optional[RealtimeBackend] = None,
        lower_limits: Optional[np.ndarray] = None,
        upper_limits: Optional[np.ndarray] = None,
    ) -> None:
        self.options = options or ELA3MotionOptions()
        self._backend = backend
        self._lower_limits = self._normalize_limits(
            lower_limits,
            default=DEFAULT_ARM_JOINT_LIMITS_LOWER,
            name="lower_limits",
        )
        self._upper_limits = self._normalize_limits(
            upper_limits,
            default=DEFAULT_ARM_JOINT_LIMITS_UPPER,
            name="upper_limits",
        )

    @property
    def is_connected(self) -> bool:
        return self._backend is not None and self._backend.is_connected

    def connect(self) -> None:
        if self._backend is None:
            self._backend = ELA3Backend(self.options)
        self._backend.connect_enable_and_start()

    def close(self, *, disable: Optional[bool] = None) -> None:
        if self._backend is None:
            return
        should_disable = self.options.disable_on_finish if disable is None else disable
        self._backend.close(disable=should_disable)

    def read_arm_joints(self) -> Optional[np.ndarray]:
        if self._backend is None:
            return None
        return self._backend.read_arm_joints()

    def stop_motion(self) -> None:
        if self._backend is not None:
            self._backend.stop_motion()

    def submit_joint_target(
        self,
        q_target: Sequence[float],
        *,
        dt: float = 0.02,
        start_q: Optional[Sequence[float]] = None,
        block: bool = False,
    ) -> RealtimeCommandResult:
        target = self._normalize_joint_vector(q_target, name="q_target")
        start = self._resolve_start_q(start_q)

        if dt <= 0:
            raise RealtimeCommandError("dt must be positive")

        joint_plan = [start, target]
        dense_plan = densify_arm_joint_plan(
            joint_plan,
            max_joint_step_rad=self.options.max_joint_step_rad,
        )
        validate_arm_joint_plan(
            dense_plan,
            lower_limits=self._lower_limits,
            upper_limits=self._upper_limits,
            max_joint_step_rad=self.options.max_joint_step_rad,
        )
        queue = build_sdk_queue_trajectory(
            dense_plan,
            sample_period_s=float(dt),
            control_period_s=1.0 / float(self.options.control_rate_hz),
            max_joint_velocity_rad_s=self.options.max_joint_velocity_rad_s,
            start_q=None,
        )

        backend_result = None
        if not self.options.dry_run:
            if self._backend is None:
                raise RealtimeCommandError("EL-A3 backend is not connected")
            backend_result = self._backend.execute_queue(queue.sdk_points, block=block)

        diagnostics: Diagnostics = {
            "input_points": len(joint_plan),
            "dense_points": len(dense_plan),
            "queue_waypoints": queue.queue_waypoints,
            "dt": float(dt),
            "block": block,
        }
        if backend_result is not None:
            diagnostics["backend_duration_s"] = backend_result.duration_s
            diagnostics["backend_sent_points"] = backend_result.sent_points

        return RealtimeCommandResult(
            dry_run=self.options.dry_run,
            submitted_points=0 if self.options.dry_run else queue.queue_points,
            duration_s=queue.duration_s,
            start_q=start.tolist(),
            target_q=target.tolist(),
            queue_points=queue.queue_points,
            diagnostics=diagnostics,
        )

    def _resolve_start_q(self, start_q: Optional[Sequence[float]]) -> np.ndarray:
        if start_q is not None:
            return self._normalize_joint_vector(start_q, name="start_q")
        feedback = self.read_arm_joints()
        if feedback is None:
            raise MissingFeedbackError(
                "start_q was not provided and EL-A3 joint feedback is unavailable"
            )
        return self._normalize_joint_vector(feedback, name="backend feedback")

    @staticmethod
    def _normalize_joint_vector(values: Sequence[float], *, name: str) -> np.ndarray:
        arr = np.asarray(values, dtype=float)
        if arr.shape != (N_ARM,):
            raise RealtimeCommandError(
                f"{name} must be a {N_ARM}-axis joint vector, got shape {arr.shape}"
            )
        if np.any(~np.isfinite(arr)):
            raise RealtimeCommandError(f"{name} contains NaN/Inf")
        return arr.copy()

    @staticmethod
    def _normalize_limits(
        values: Optional[np.ndarray],
        *,
        default: np.ndarray,
        name: str,
    ) -> np.ndarray:
        if values is None:
            return default.copy()
        arr = np.asarray(values, dtype=float)
        if arr.shape != (N_ARM,):
            raise RealtimeCommandError(
                f"{name} must be a {N_ARM}-axis limit vector, got shape {arr.shape}"
            )
        if np.any(np.isnan(arr)):
            raise RealtimeCommandError(f"{name} contains NaN")
        return arr.copy()
