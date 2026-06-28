"""Configuration and policy types for EL-A3 motion execution."""

from __future__ import annotations

from dataclasses import dataclass, replace
from enum import Enum
from typing import Optional


class ExecutionMode(str, Enum):
    DRY_RUN = "dry_run"
    REAL = "real"


class StartPolicy(str, Enum):
    BRIDGE_FROM_REAL = "bridge_from_real"
    REQUIRE_NEAR_START = "require_near_start"
    SEED_FROM_REAL = "seed_from_real"
    FROM_MUJOCO_STATE = "from_mujoco_state"


class FinishPolicy(str, Enum):
    HOLD = "hold"
    DISABLE = "disable"
    HOME_THEN_DISABLE = "home_then_disable"
    STAY_CONNECTED = "stay_connected"


@dataclass(frozen=True)
class ELA3MotionOptions:
    can_name: str = "can0"
    backend: str = "socketcan"
    serial_port: Optional[str] = None
    serial_baudrate: int = 2_000_000
    can_bitrate: int = 1_000_000
    host_can_id: int = 253

    kp: float = 60.0
    kd: float = 3.5

    command_rate_hz: float = 60.0
    control_rate_hz: float = 200.0
    max_joint_step_rad: float = 0.12
    max_joint_velocity_rad_s: float = 1.2
    ik_tolerance_m: float = 0.01
    lift_height_m: float = 0.03

    feedback_timeout_s: float = 3.0
    feedback_log_interval_s: float = 10.0

    setup_can: bool = False
    dry_run: bool = False
    mujoco_viewer: bool = False
    hold_on_finish: bool = True
    disable_on_finish: bool = True

    start_near_tolerance_rad: float = 0.08
    bridge_min_duration_s: float = 0.5

    gripper_pp_velocity_rad_s: float = 6.0
    gripper_pp_acceleration_rad_s2: float = 15.0

    def with_overrides(self, **overrides) -> "ELA3MotionOptions":
        valid = {field.name for field in self.__dataclass_fields__.values()}
        unknown = sorted(set(overrides) - valid)
        if unknown:
            raise TypeError(f"unknown ELA3MotionOptions fields: {unknown}")
        return replace(self, **overrides)

    @classmethod
    def from_project_config(cls, **overrides) -> "ELA3MotionOptions":
        from core.project_config import get_execution_defaults, get_real_robot_defaults

        defaults = get_real_robot_defaults()
        execution_defaults = get_execution_defaults()
        options = cls(
            can_name=defaults["can"],
            backend=defaults["backend"],
            serial_port=defaults["serial_port"],
            serial_baudrate=defaults["serial_baudrate"],
            can_bitrate=defaults["can_bitrate"],
            host_can_id=defaults["host_can_id"],
            kp=defaults["kp"],
            kd=defaults["kd"],
            command_rate_hz=defaults["command_rate"],
            control_rate_hz=defaults["control_rate"],
            max_joint_step_rad=defaults["max_joint_step"],
            max_joint_velocity_rad_s=defaults["max_joint_velocity"],
            ik_tolerance_m=defaults["ik_tolerance"],
            lift_height_m=execution_defaults["transition_lift_height"],
            feedback_timeout_s=defaults["feedback_timeout"],
            feedback_log_interval_s=defaults["feedback_log_interval"],
            setup_can=defaults["setup_can"],
            dry_run=defaults["dry_run"],
            mujoco_viewer=defaults["with_mujoco"],
            hold_on_finish=defaults["hold_on_finish"],
            disable_on_finish=defaults["disable_on_finish"],
            gripper_pp_velocity_rad_s=defaults["gripper_pp_velocity"],
            gripper_pp_acceleration_rad_s2=defaults["gripper_pp_acceleration"],
        )
        return options.with_overrides(**overrides) if overrides else options
