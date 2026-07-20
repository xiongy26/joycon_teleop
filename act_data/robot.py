"""EL-A3 robot feedback helpers for ACT data collection."""

from __future__ import annotations

import time
from dataclasses import dataclass

import numpy as np

from ela3_motion.hardware import (
    build_interface_kwargs,
    ensure_sdk_on_path,
    ensure_socketcan_ready,
)
from ela3_motion.options import ELA3MotionOptions


@dataclass(frozen=True)
class ZeroTorqueConfig:
    enabled: bool = True
    gravity: bool = True
    kd: tuple[float, ...] = (0.05, 0.05, 0.05, 0.0125, 0.0125, 0.0125, 0.05)
    update_rate_hz: float = 100.0


class ELRobotReader:
    """Owns EL-A3 lifecycle for demonstration collection."""

    def __init__(
        self,
        options: ELA3MotionOptions,
        *,
        zero_torque: ZeroTorqueConfig = ZeroTorqueConfig(),
    ):
        self.options = options
        self.zero_torque = zero_torque
        self.arm = None
        self._connected = False
        self._zero_torque_active = False

    def __enter__(self) -> "ELRobotReader":
        self.open()
        return self

    def __exit__(self, _exc_type, _exc, _tb) -> None:
        self.close()

    def open(self) -> None:
        if self._connected:
            return
        ensure_sdk_on_path()
        ensure_socketcan_ready(self.options)
        from el_a3_sdk import ELA3Interface

        self.arm = ELA3Interface(**build_interface_kwargs(self.options))
        if not self.arm.ConnectPort():
            raise RuntimeError(f"failed to connect EL-A3 on {self.options.can_name}")
        self._connected = True

        if not self.arm.EnableArm():
            self.close()
            raise RuntimeError("failed to enable EL-A3 motors")
        time.sleep(0.3)

        if self.zero_torque.enabled:
            if self.zero_torque.gravity:
                ok = self.arm.ZeroTorqueModeWithGravity(
                    enable=True,
                    kd=list(self.zero_torque.kd),
                    update_rate=self.zero_torque.update_rate_hz,
                )
            else:
                ok = self.arm.ZeroTorqueMode(enable=True, kd=self.zero_torque.kd[0])
            if not ok:
                self.close()
                raise RuntimeError("failed to enter zero-torque teaching mode")
            self._zero_torque_active = True

    def read_qpos(self, *, include_gripper: bool = True) -> np.ndarray:
        if self.arm is None:
            raise RuntimeError("robot reader is not open")
        joints = self.arm.GetArmJointMsgs()
        if joints is None or joints.timestamp <= 0:
            raise RuntimeError("invalid EL-A3 joint feedback")
        qpos = np.asarray(joints.to_list(include_gripper=include_gripper), dtype=np.float32)
        if qpos.shape != ((7 if include_gripper else 6),):
            raise RuntimeError(f"unexpected qpos shape: {qpos.shape}")
        if np.any(~np.isfinite(qpos)):
            raise RuntimeError("joint feedback contains NaN/Inf")
        return qpos

    def close(self) -> None:
        if self.arm is None:
            return
        try:
            if self._zero_torque_active:
                if self.zero_torque.gravity:
                    self.arm.ZeroTorqueModeWithGravity(enable=False)
                else:
                    self.arm.ZeroTorqueMode(enable=False)
                self._zero_torque_active = False
        finally:
            try:
                self.arm.DisableArm()
            finally:
                self.arm.DisconnectPort()
                self._connected = False
                self.arm = None
