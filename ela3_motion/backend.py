"""EL-A3 SDK backend wrapper for motion manager."""

from __future__ import annotations

import time
from typing import Callable, Optional

import numpy as np

from .diagnostics import collect_backend_diagnostics, format_feedback_diagnostics
from .hardware import (
    build_interface_kwargs,
    ensure_sdk_on_path,
    ensure_socketcan_ready,
    hold_enabled_until_shutdown,
)
from .options import ELA3MotionOptions
from .types import MotionExecutionResult


N_ARM = 6


class ELA3Backend:
    """Small lifecycle wrapper around el_a3_sdk.ELA3Interface."""

    def __init__(self, options: ELA3MotionOptions, *, arm=None) -> None:
        self.options = options
        self._arm = arm
        self._connected = False
        self._control_loop_started = False

    def __enter__(self) -> "ELA3Backend":
        return self

    def __exit__(self, _exc_type, _exc, _tb) -> None:
        self.close(disable=self.options.disable_on_finish)

    @property
    def arm(self):
        return self._arm

    @property
    def is_connected(self) -> bool:
        return self._connected

    def connect(self) -> None:
        if self._connected:
            return
        if self._arm is None:
            ensure_sdk_on_path()
            ensure_socketcan_ready(self.options)
            from el_a3_sdk import ELA3Interface

            self._arm = ELA3Interface(**build_interface_kwargs(self.options))

        if not self._arm.ConnectPort():
            raise RuntimeError(f"failed to connect EL-A3 on {self.options.can_name}")
        self._connected = True

    def enable(self) -> None:
        self.connect()
        if not self._arm.EnableArm():
            self.disconnect()
            raise RuntimeError("failed to enable EL-A3 motors")
        self._arm.SetPositionPD(kp=self.options.kp, kd=self.options.kd)

    def start_control_loop(self) -> None:
        self._arm.start_control_loop(rate_hz=self.options.control_rate_hz)
        self._control_loop_started = True

    def connect_enable_and_start(self) -> None:
        self.connect()
        self.enable()
        self.start_control_loop()

    def read_arm_joints(self) -> Optional[np.ndarray]:
        try:
            joints = self._arm.GetArmJointMsgs()
            if joints is None or joints.timestamp <= 0:
                return None
            values = np.array(joints.to_list(include_gripper=False)[:N_ARM], dtype=float)
        except Exception:
            return None
        if len(values) != N_ARM or np.any(~np.isfinite(values)):
            return None
        return values

    def wait_arm_joints(self, timeout_s: float) -> np.ndarray:
        deadline = time.time() + max(0.1, float(timeout_s))
        while time.time() < deadline:
            q = self.read_arm_joints()
            if q is not None:
                return q
            time.sleep(0.05)
        diagnostics = (
            format_feedback_diagnostics(self._arm)
            if self._arm is not None
            else "EL-A3 arm is not connected"
        )
        raise RuntimeError(
            "timed out waiting for valid EL-A3 joint feedback\n" + diagnostics
        )

    def execute_queue(
        self,
        sdk_points: list,
        *,
        block: bool = True,
        feedback_log_interval_s: float = 0.0,
        progress_callback: Optional[Callable[[float], bool | None]] = None,
    ) -> MotionExecutionResult:
        if not sdk_points:
            raise RuntimeError("SDK trajectory queue is empty")
        ok = self._arm._execute_trajectory_async(sdk_points, block=False)
        if not ok:
            raise RuntimeError("failed to submit SDK trajectory queue")

        start_time = time.perf_counter()
        if block:
            next_log = start_time
            while self._arm.is_moving():
                now = time.perf_counter()
                elapsed = now - start_time
                if progress_callback is not None:
                    keep_running = progress_callback(elapsed)
                    if keep_running is False:
                        self.stop_motion()
                        break
                if feedback_log_interval_s > 0 and now >= next_log:
                    print(f"[ela3_motion] t={elapsed:.2f}s moving", flush=True)
                    next_log += feedback_log_interval_s
                time.sleep(max(1.0 / self.options.control_rate_hz, 0.001))
            if progress_callback is not None:
                progress_callback(time.perf_counter() - start_time)

        duration_s = time.perf_counter() - start_time
        final_q = list(sdk_points[-1].positions[:N_ARM])
        return MotionExecutionResult(
            dry_run=False,
            interrupted=False,
            execution_mode="sdk_trajectory_queue",
            sent_points=len(sdk_points),
            duration_s=duration_s,
            queue_waypoints=0,
            queue_points=len(sdk_points),
            start_q=None,
            first_target_q=None,
            final_q=final_q,
            bridge_inserted=False,
            diagnostics=collect_backend_diagnostics(self._arm),
        )

    def hold_until_interrupt(
        self,
        *,
        target_q: Optional[np.ndarray] = None,
        log_interval_s: float = 0.0,
    ) -> None:
        hold_enabled_until_shutdown(
            lambda: False,
            arm=self._arm,
            target_q=target_q,
            log_interval_s=log_interval_s,
        )

    def stop_motion(self) -> None:
        if self._arm is not None:
            self._arm.cancel_motion()

    def stop_control_loop(self) -> None:
        if self._arm is not None and self._control_loop_started:
            self._arm.stop_control_loop()
            self._control_loop_started = False

    def disable(self) -> None:
        if self._arm is not None:
            self._arm.DisableArm()

    def disconnect(self) -> None:
        if self._arm is not None and self._connected:
            self._arm.DisconnectPort()
            self._connected = False

    def close(self, *, disable: bool = True) -> None:
        try:
            self.stop_control_loop()
        finally:
            try:
                if disable:
                    self.disable()
            finally:
                self.disconnect()
