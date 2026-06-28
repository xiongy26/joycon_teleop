from __future__ import annotations

import time
from pathlib import Path
from typing import Optional

import numpy as np

from .diagnostics import format_feedback_diagnostics, format_realtime_feedback_line
from .options import ELA3MotionOptions


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SDK_ROOT = PROJECT_ROOT / "el_a3_sdk"
N_ARM = 6


def ensure_sdk_on_path() -> None:
    import sys

    if str(SDK_ROOT) not in sys.path:
        sys.path.insert(0, str(SDK_ROOT))


def sdk_resource_kwargs() -> dict:
    inertia_path = SDK_ROOT / "resources" / "config" / "inertia_params.yaml"
    legacy_urdf_path = SDK_ROOT / "resources" / "urdf" / "el_a3_legacy.urdf"
    kwargs = {
        "per_joint_kd_min": {4: 0.005, 5: 0.005, 6: 0.005, 7: 0.02},
        "per_joint_kd_max": {4: 0.10, 5: 0.05, 6: 0.05, 7: 0.10},
        "gravity_joint_scale": {4: 2.0},
    }
    if inertia_path.exists():
        kwargs["inertia_config_path"] = str(inertia_path)
    if legacy_urdf_path.exists():
        kwargs["urdf_path"] = str(legacy_urdf_path)
    return kwargs


def get_can_state(can_name: str) -> str:
    operstate_file = Path(f"/sys/class/net/{can_name}/operstate")
    try:
        raw = operstate_file.read_text().strip().upper()
        if raw == "UNKNOWN":
            flags_file = Path(f"/sys/class/net/{can_name}/flags")
            if flags_file.exists():
                flags = int(flags_file.read_text().strip(), 16)
                return "UP" if flags & 0x1 else "DOWN"
        return raw
    except (FileNotFoundError, OSError):
        return "UNKNOWN"


def ensure_socketcan_ready(options: ELA3MotionOptions) -> None:
    if options.backend == "slcan":
        return

    state = get_can_state(options.can_name)
    if state == "UP":
        return

    if options.setup_can:
        ensure_sdk_on_path()
        try:
            from MotorStudio.utils.can_utils import setup_can_interface
        except Exception as exc:
            raise RuntimeError(f"cannot import MotorStudio CAN setup helper: {exc}") from exc

        ok, msg = setup_can_interface(options.can_name, options.can_bitrate)
        if not ok:
            raise RuntimeError(f"CAN interface {options.can_name} setup failed: {msg}")
        if get_can_state(options.can_name) == "UP":
            return

    raise RuntimeError(
        f"CAN interface {options.can_name} is not UP (state: {state}). "
        "Run el_a3_sdk/scripts/setup_can.sh or enable setup_can."
    )


def build_interface_kwargs(options: ELA3MotionOptions) -> dict:
    kwargs = sdk_resource_kwargs()
    kwargs.update(
        can_name=options.can_name,
        host_can_id=options.host_can_id,
        default_kp=options.kp,
        default_kd=options.kd,
        backend=options.backend,
        serial_port=options.serial_port,
        serial_baudrate=options.serial_baudrate,
        can_bitrate=options.can_bitrate,
        pp_velocity=options.gripper_pp_velocity_rad_s,
        pp_acceleration=options.gripper_pp_acceleration_rad_s2,
    )
    return kwargs


def connect_real_arm(options: ELA3MotionOptions):
    ensure_sdk_on_path()
    ensure_socketcan_ready(options)
    from el_a3_sdk import ELA3Interface

    arm = ELA3Interface(**build_interface_kwargs(options))
    if not arm.ConnectPort():
        raise RuntimeError(f"failed to connect EL-A3 on {options.can_name}")
    if not arm.EnableArm():
        arm.DisconnectPort()
        raise RuntimeError("failed to enable EL-A3 motors")
    arm.SetPositionPD(kp=options.kp, kd=options.kd)
    arm.start_control_loop(rate_hz=options.control_rate_hz)
    return arm


def connect_real_arm_for_feedback(options: ELA3MotionOptions):
    ensure_sdk_on_path()
    ensure_socketcan_ready(options)
    from el_a3_sdk import ELA3Interface

    arm = ELA3Interface(**build_interface_kwargs(options))
    if not arm.ConnectPort():
        raise RuntimeError(f"failed to connect EL-A3 on {options.can_name}")
    time.sleep(0.2)
    return arm


def read_real_joint_seed(arm) -> Optional[np.ndarray]:
    try:
        joints = arm.GetArmJointMsgs()
        if joints is None or joints.timestamp <= 0:
            return None
        values = np.array(joints.to_list(include_gripper=False)[:N_ARM], dtype=float)
    except Exception:
        return None

    if values.shape != (N_ARM,) or np.any(~np.isfinite(values)):
        return None
    return values


def wait_real_joint_seed(arm, timeout_s: float) -> Optional[np.ndarray]:
    deadline = time.time() + max(0.1, timeout_s)

    while time.time() < deadline:
        seed = read_real_joint_seed(arm)
        if seed is not None:
            try:
                states = arm.GetMotorStates()
                arm_ok = all(
                    states.get(motor_id) is not None and states[motor_id].is_valid
                    for motor_id in range(1, N_ARM + 1)
                )
                if arm_ok:
                    return seed
            except Exception:
                pass
        time.sleep(0.05)

    return None


def hold_enabled_until_shutdown(
    is_shutdown,
    *,
    arm=None,
    target_q: Optional[np.ndarray] = None,
    log_interval_s: float = 0.0,
) -> None:
    print("EL-A3 holding enabled; press Ctrl+C when ready to stop.")
    next_log_time = 0.0
    log_start = time.perf_counter()
    while not is_shutdown():
        if arm is not None and log_interval_s > 0:
            now = time.perf_counter()
            if now >= next_log_time:
                line, alert = format_realtime_feedback_line(
                    arm,
                    target_q=target_q,
                    elapsed_s=now - log_start,
                )
                print(line, flush=True)
                if alert:
                    print(format_feedback_diagnostics(arm), flush=True)
                next_log_time = now + log_interval_s
        time.sleep(0.2)
    print("Stop requested; stopping control loop and disconnecting.")
