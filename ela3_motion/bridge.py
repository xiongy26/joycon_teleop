"""Start-state bridging helpers for EL-A3 trajectory deployment."""

from __future__ import annotations

from typing import Any, Optional, Sequence

import numpy as np

from .options import StartPolicy


def build_start_bridge(
    start_q: np.ndarray,
    target_q: np.ndarray,
    *,
    dt: float,
    max_joint_velocity_rad_s: float,
    min_duration_s: float,
) -> list[np.ndarray]:
    start = np.asarray(start_q, dtype=float)
    target = np.asarray(target_q, dtype=float)
    if start.shape != (6,) or target.shape != (6,):
        raise ValueError("start_q and target_q must have shape (6,)")
    if dt <= 0:
        raise ValueError("dt must be positive")

    max_delta = float(np.linalg.norm(target - start, ord=np.inf))
    duration = max(float(min_duration_s), dt)
    if max_joint_velocity_rad_s > 0:
        duration = max(duration, max_delta / max_joint_velocity_rad_s)
    steps = max(1, int(np.ceil(duration / dt)))

    bridge: list[np.ndarray] = []
    for idx in range(steps + 1):
        t = idx / steps
        alpha = 10.0 * t**3 - 15.0 * t**4 + 6.0 * t**5
        bridge.append((1.0 - alpha) * start + alpha * target)
    return bridge


def maybe_prepend_start_bridge(
    joint_plan: Sequence[np.ndarray],
    *,
    start_q: Optional[np.ndarray],
    dt: float,
    start_policy: StartPolicy,
    dry_run: bool,
    near_tolerance_rad: float,
    max_joint_velocity_rad_s: float,
    min_duration_s: float,
) -> tuple[list[np.ndarray], bool, dict[str, Any]]:
    if not joint_plan:
        raise ValueError("joint trajectory is empty")

    plan = [np.asarray(q, dtype=float).copy() for q in joint_plan]
    diagnostics: dict[str, Any] = {}
    policy = StartPolicy(start_policy)

    if dry_run:
        diagnostics["start_policy_skipped"] = policy.value
        diagnostics["start_policy_skip_reason"] = "dry_run_has_no_real_feedback"
        return plan, False, diagnostics

    if policy == StartPolicy.FROM_MUJOCO_STATE:
        raise RuntimeError("FROM_MUJOCO_STATE cannot be used for real execution")
    if start_q is None:
        raise RuntimeError("real execution requires a valid start_q from hardware")

    start = np.asarray(start_q, dtype=float)
    first = plan[0]
    delta = float(np.linalg.norm(first - start, ord=np.inf))
    diagnostics["start_delta_rad"] = delta

    if policy == StartPolicy.REQUIRE_NEAR_START:
        if delta > near_tolerance_rad:
            raise RuntimeError(
                f"real start differs from first target by {delta:.3f} rad, "
                f"above tolerance {near_tolerance_rad:.3f} rad"
            )
        return plan, False, diagnostics

    if policy in {StartPolicy.BRIDGE_FROM_REAL, StartPolicy.SEED_FROM_REAL}:
        if delta <= near_tolerance_rad:
            return plan, False, diagnostics
        bridge = build_start_bridge(
            start,
            first,
            dt=dt,
            max_joint_velocity_rad_s=max_joint_velocity_rad_s,
            min_duration_s=min_duration_s,
        )
        return bridge + plan[1:], True, diagnostics

    raise ValueError(f"unsupported start_policy: {start_policy}")
