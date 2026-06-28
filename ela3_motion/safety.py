"""Pure trajectory safety checks and densification for EL-A3 deployment."""

from __future__ import annotations

from typing import Sequence

import numpy as np


def validate_arm_joint_plan(
    joint_plan: Sequence[np.ndarray],
    *,
    lower_limits: np.ndarray,
    upper_limits: np.ndarray,
    max_joint_step_rad: float,
) -> None:
    if not joint_plan:
        raise ValueError("joint trajectory is empty")

    q = np.vstack([np.asarray(point, dtype=float) for point in joint_plan])
    if q.ndim != 2 or q.shape[1] != 6:
        raise ValueError(f"joint trajectory must have shape N x 6, got {q.shape}")
    if np.any(~np.isfinite(q)):
        raise ValueError("joint trajectory contains NaN/Inf")
    if np.any(q < lower_limits - 1e-6) or np.any(q > upper_limits + 1e-6):
        raise ValueError("joint trajectory exceeds configured joint limits")

    if len(joint_plan) > 1:
        max_step = max(
            float(
                np.linalg.norm(
                    np.asarray(joint_plan[i]) - np.asarray(joint_plan[i - 1]),
                    ord=np.inf,
                )
            )
            for i in range(1, len(joint_plan))
        )
        if max_step > max_joint_step_rad + 1e-9:
            raise ValueError(
                f"joint trajectory step too large: {max_step:.3f} rad "
                f"> {max_joint_step_rad:.3f} rad"
            )


def densify_arm_joint_plan(
    joint_plan: Sequence[np.ndarray],
    *,
    max_joint_step_rad: float,
) -> list[np.ndarray]:
    if max_joint_step_rad <= 0:
        raise ValueError("max_joint_step_rad must be positive")
    if len(joint_plan) <= 1:
        return [np.asarray(q, dtype=float).copy() for q in joint_plan]

    dense_plan: list[np.ndarray] = [np.asarray(joint_plan[0], dtype=float).copy()]
    for q_next_raw in joint_plan[1:]:
        q_prev = dense_plan[-1]
        q_next = np.asarray(q_next_raw, dtype=float).copy()
        max_delta = float(np.linalg.norm(q_next - q_prev, ord=np.inf))
        steps = max(1, int(np.ceil(max_delta / max_joint_step_rad)))
        for step_idx in range(1, steps + 1):
            alpha = step_idx / steps
            dense_plan.append((1.0 - alpha) * q_prev + alpha * q_next)
    return dense_plan
