"""Diagnostics helpers for EL-A3 motion execution."""

from __future__ import annotations

import time
from typing import Optional

import numpy as np


def format_feedback_diagnostics(arm) -> str:
    lines = ["EL-A3 feedback diagnostics:"]
    try:
        lines.append(f"  CAN FPS: {arm.GetCanFps():.1f}")
    except Exception as exc:
        lines.append(f"  CAN FPS: read failed ({exc})")

    try:
        lines.append(f"  CAN bus: {arm.GetCanBusState()}")
    except Exception as exc:
        lines.append(f"  CAN bus: read failed ({exc})")

    try:
        states = arm.GetMotorStates()
    except Exception as exc:
        states = {}
        lines.append(f"  GetMotorStates failed: {exc}")

    for motor_id in range(1, 8):
        fb = states.get(motor_id)
        if fb is None:
            lines.append(f"  M{motor_id}: no feedback")
            continue
        age = time.time() - fb.timestamp if fb.timestamp else float("inf")
        age_text = f"{age:.2f}s" if np.isfinite(age) else "n/a"
        lines.append(
            f"  M{motor_id}: valid={fb.is_valid} "
            f"pos={fb.position:.4f} vel={fb.velocity:.4f} "
            f"torque={fb.torque:.4f} fault={fb.fault_code} mode={fb.mode_state} "
            f"temp={fb.temperature:.1f}C age={age_text}"
        )

    try:
        joints = arm.GetArmJointMsgs()
        lines.append(
            "  joint_msg: "
            f"timestamp={joints.timestamp:.6f} "
            f"q={[round(v, 4) for v in joints.to_list(include_gripper=False)[:6]]}"
        )
    except Exception as exc:
        lines.append(f"  joint_msg: read failed ({exc})")

    return "\n".join(lines)


def format_realtime_feedback_line(
    arm,
    *,
    target_q: Optional[np.ndarray] = None,
    elapsed_s: Optional[float] = None,
    stale_after_s: float = 0.5,
) -> tuple[str, bool]:
    now = time.time()
    parts = ["[ela3_motion_feedback]"]
    if elapsed_s is not None:
        parts.append(f"t={elapsed_s:.2f}s")

    alert = False
    try:
        parts.append(f"fps={arm.GetCanFps():.1f}")
    except Exception as exc:
        alert = True
        parts.append(f"fps=ERR({exc})")

    try:
        parts.append(f"bus={arm.GetCanBusState()}")
    except Exception as exc:
        alert = True
        parts.append(f"bus=ERR({exc})")

    try:
        status = arm.GetArmStatus()
        enabled_count = sum(bool(v) for v in status.joint_enabled[:6])
        fault_items = [
            f"M{idx + 1}:{fault}"
            for idx, fault in enumerate(status.joint_faults[:7])
            if fault
        ]
        mode_items = [
            f"M{idx + 1}:{mode}"
            for idx, mode in enumerate(status.joint_mode_states[:6])
        ]
        parts.append(f"enabled={enabled_count}/6")
        parts.append(f"arm_status={status.arm_status}")
        parts.append("faults=" + (",".join(fault_items) if fault_items else "none"))
        parts.append("modes=" + ",".join(mode_items))
        if status.has_fault or enabled_count < 6:
            alert = True
    except Exception as exc:
        alert = True
        parts.append(f"status=ERR({exc})")

    try:
        joints = arm.GetArmJointMsgs()
        q = np.array(joints.to_list(include_gripper=False)[:6], dtype=float)
        q_text = ",".join(f"{v:.3f}" for v in q)
        parts.append(f"q=[{q_text}]")
        if target_q is not None and len(target_q) >= 6:
            err = np.asarray(target_q[:6], dtype=float) - q
            parts.append(f"max_err={float(np.linalg.norm(err, ord=np.inf)):.3f}rad")
    except Exception as exc:
        alert = True
        parts.append(f"q=ERR({exc})")

    try:
        states = arm.GetMotorStates()
        motor_parts = []
        for motor_id in range(1, 7):
            fb = states.get(motor_id)
            if fb is None:
                alert = True
                motor_parts.append(f"M{motor_id}:missing")
                continue
            age = now - fb.timestamp if fb.timestamp else float("inf")
            if (not fb.is_valid) or fb.fault_code or age > stale_after_s:
                alert = True
            age_text = f"{age:.2f}" if np.isfinite(age) else "inf"
            motor_parts.append(
                f"M{motor_id}:v={fb.velocity:.2f},tau={fb.torque:.2f},"
                f"T={fb.temperature:.0f},age={age_text}"
            )
        parts.append("motors={" + "; ".join(motor_parts) + "}")
    except Exception as exc:
        alert = True
        parts.append(f"motors=ERR({exc})")

    return " ".join(parts), alert


def collect_backend_diagnostics(arm) -> dict:
    diagnostics = {}
    try:
        diagnostics["can_fps"] = float(arm.GetCanFps())
    except Exception as exc:
        diagnostics["can_fps_error"] = str(exc)
    try:
        diagnostics["can_bus_state"] = str(arm.GetCanBusState())
    except Exception as exc:
        diagnostics["can_bus_state_error"] = str(exc)
    try:
        status = arm.GetArmStatus()
        diagnostics["arm_status"] = str(status.arm_status)
        diagnostics["has_fault"] = bool(status.has_fault)
        diagnostics["joint_enabled"] = [bool(v) for v in status.joint_enabled]
        diagnostics["joint_faults"] = list(status.joint_faults)
    except Exception as exc:
        diagnostics["arm_status_error"] = str(exc)
    return diagnostics
