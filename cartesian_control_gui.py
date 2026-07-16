#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Standalone Cartesian control GUI for EL-A3 gripper."""

from __future__ import annotations

import sys
import time
import json
import os
import logging
from datetime import datetime
from pathlib import Path
from typing import Optional

import numpy as np
from scipy.spatial.transform import Rotation

from PySide6.QtCore import Qt, Slot, QTimer
from PySide6.QtWidgets import (
    QApplication,
    QCheckBox,
    QComboBox,
    QDoubleSpinBox,
    QFileDialog,
    QGroupBox,
    QHBoxLayout,
    QLabel,
    QListWidget,
    QMainWindow,
    QMessageBox,
    QPushButton,
    QSlider,
    QTabWidget,
    QVBoxLayout,
    QWidget,
    QGridLayout,
    QStatusBar,
)

import mujoco
import mujoco.viewer
import mink

# Add current directory to path for core imports
sys.path.insert(0, str(Path(__file__).resolve().parent))

from core.ik_core import (
    ARM_JOINT_LIMITS_LOWER,
    ARM_JOINT_LIMITS_UPPER,
    N_ARM,
    arm_qpos_indices,
    arm_dof_indices,
    ee_site_id,
    EE_SITE_NAME,
    gripper_qpos_index,
    gripper_dof_index,
    gripper_joint_id,
    lock,
    model,
    data,
    fix_base_position,
    rotation_matrix_to_quat,
)
from core.gripper_config import GRIPPER_JOINT_MAX, GRIPPER_JOINT_MIN
from ela3_motion import ELA3MotionOptions, ELA3RealtimeController

import threading
import struct

# Joint control constants
# A5: 限位从 URDF (model.jnt_range) 动态生成，避免与 SDK/MJCF 不一致
JOINT_LIMITS_DEG = {
    i + 1: (
        float(np.degrees(ARM_JOINT_LIMITS_LOWER[i])),
        float(np.degrees(ARM_JOINT_LIMITS_UPPER[i])),
    )
    for i in range(N_ARM)
}
JOINT_LIMITS_DEG[7] = (
    float(np.degrees(GRIPPER_JOINT_MIN)),
    float(np.degrees(GRIPPER_JOINT_MAX)),
)
JOINT_COLORS = ["#e74c3c", "#2ecc71", "#3498db", "#9b59b6", "#f39c12", "#1abc9c", "#e67e22"]
SLIDER_RESOLUTION = 1000

# 判定"是否处于零位"的阈值：任意关节偏离零位超过此值即视为非零位
HOME_THRESHOLD_RAD = np.radians(1.0)


class RealRobotManager:
    """Manage EL-A3 real robot connection."""

    def __init__(self):
        self._controller: ELA3RealtimeController | None = None

    def connect(
        self,
        can_name: str = "can0",
        kp: float = 60.0,
        kd: float = 3.5,
        max_vel_rad_s: float = 1.2,
    ):
        options = ELA3MotionOptions(
            can_name=can_name,
            kp=kp,
            kd=kd,
            control_rate_hz=200.0,
            command_rate_hz=50.0,
            max_joint_velocity_rad_s=float(max_vel_rad_s),
        )
        self._controller = ELA3RealtimeController(
            options,
            lower_limits=ARM_JOINT_LIMITS_LOWER,
            upper_limits=ARM_JOINT_LIMITS_UPPER,
        )
        self._controller.connect()

    def set_max_velocity(self, max_vel_rad_s: float) -> bool:
        """动态调整关节最大速度上限（rad/s），无需重连。"""
        if self._controller is None:
            return False
        # ELA3MotionOptions 为 frozen dataclass，用 with_overrides 生成新实例后整体替换。
        # ELA3RealtimeController.submit_joint_target 每次都从 self.options 读取，
        # 因此替换后下一次下发即生效。
        self._controller.options = self._controller.options.with_overrides(
            max_joint_velocity_rad_s=float(max_vel_rad_s),
        )
        return True

    def disconnect(self):
        if self._controller is not None:
            self._controller.close()
            self._controller = None

    def get_joint_feedback(self) -> np.ndarray | None:
        if self._controller is None:
            return None
        return self._controller.read_arm_joints()

    def send_joint_command(
        self,
        q_target: np.ndarray,
        *,
        start_q: np.ndarray | None = None,
    ):
        if self._controller is None:
            return
        self._controller.submit_joint_target(
            q_target,
            dt=0.02,
            start_q=start_q,
            block=False,
        )

    def send_gripper_command(self, gripper_angle: float) -> bool:
        """下发夹爪目标角度（弧度）。"""
        if self._controller is None:
            return False
        return self._controller.submit_gripper_target(gripper_angle)

    def _get_arm(self):
        """获取底层 el_a3_sdk 的 ELA3Interface 实例（供示教用）。"""
        if self._controller is None:
            return None
        backend = getattr(self._controller, "_backend", None)
        if backend is None:
            return None
        return getattr(backend, "arm", None)

    def enter_zero_torque(self, gravity: bool = True,
                          kd_list: list | None = None) -> bool:
        """进入零力矩（示教）模式；gravity=True 时启用重力补偿。"""
        arm = self._get_arm()
        if arm is None:
            return False
        try:
            if gravity:
                kd = kd_list or [0.05, 0.05, 0.05, 0.0125, 0.0125, 0.0125, 0.05]
                return bool(
                    arm.ZeroTorqueModeWithGravity(
                        enable=True, kd=kd, update_rate=100.0,
                    )
                )
            return bool(arm.ZeroTorqueMode(enable=True, kd=0.05))
        except Exception:
            return False

    def exit_zero_torque(self, gravity: bool = True) -> bool:
        """退出零力矩模式，回到位置控制。"""
        arm = self._get_arm()
        if arm is None:
            return False
        try:
            if gravity:
                return bool(arm.ZeroTorqueModeWithGravity(enable=False))
            return bool(arm.ZeroTorqueMode(enable=False))
        except Exception:
            return False

    @property
    def is_connected(self):
        return self._controller is not None and self._controller.is_connected

try:
    import evdev
    EVDEV_AVAILABLE = True
except ImportError:
    EVDEV_AVAILABLE = False


class JoyConReader:
    """Read Joy-Con (L) sticks (evdev) + buttons (hidraw) + IMU (evdev)."""

    BTN_MAP = {
        309: 0, 544: 1, 310: 4, 311: 5, 312: 6, 313: 7,
        314: 8, 317: 11, 545: 13, 546: 14, 547: 15,
    }

    def __init__(self):
        self.axes = [0.0] * 6
        self.buttons = [0] * 16
        self._dpad_x = 0  # D-pad 左右: -1=左, 0=中, 1=右
        self._dpad_y = 0  # D-pad 上下: -1=上, 0=中, 1=下
        self.imu_gyro = [0.0, 0.0, 0.0]
        self.imu_accel = [0.0, 0.0, 0.0]
        self._gyro_bias = [0.0, 0.0, 0.0]
        self._calibrating = False
        self._calib_samples = [[], [], []]
        self.connected = False
        self._evdev_dev = None
        self._imu_dev = None
        self._hidraw_fd = None
        self._threads = []
        self._running = False
        # 摇杆零点校准
        self._stick_center = [0.0, 0.0]  # X/Y 轴中心偏移
        self._stick_calibrated = False
        self._stick_calib_samples = [[], []]
        self._stick_calib_start = 0.0

    def connect(self) -> bool:
        if not EVDEV_AVAILABLE:
            return False
        for path in evdev.list_devices():
            try:
                d = evdev.InputDevice(path)
                if 'Joy-Con' in d.name and 'IMU' not in d.name:
                    self._evdev_dev = d
                    break
            except Exception:
                continue
        if self._evdev_dev is None:
            return False

        for i in range(64):
            try:
                d = evdev.InputDevice(f'/dev/input/event{i}')
                if 'IMU' in d.name and 'Joy-Con' in d.name:
                    self._imu_dev = d
                    break
                d.close()
            except Exception:
                continue

        self._try_open_hidraw()
        self.connected = True
        self._running = True

        t1 = threading.Thread(target=self._stick_loop, daemon=True, name="joycon_stick")
        t1.start()
        self._threads.append(t1)

        if self._imu_dev:
            t2 = threading.Thread(target=self._imu_loop, daemon=True, name="joycon_imu")
            t2.start()
            self._threads.append(t2)

        if self._hidraw_fd is not None:
            t3 = threading.Thread(target=self._hidraw_loop, daemon=True, name="joycon_hidraw")
            t3.start()
            self._threads.append(t3)

        return True

    def _try_open_hidraw(self):
        import glob as _g
        for h in sorted(_g.glob("/dev/hidraw*")):
            try:
                fd = os.open(h, os.O_RDONLY | os.O_NONBLOCK)
                time.sleep(0.05)
                raw = os.read(fd, 64)
                if raw and raw[0] in (0x3F, 0x30, 0x31, 0x21) and len(raw) >= 8:
                    self._hidraw_fd = fd
                    return
                os.close(fd)
            except (OSError, PermissionError):
                continue

    def disconnect(self):
        self._running = False
        for t in self._threads:
            t.join(timeout=1.0)
        self._threads = []
        for dev in [self._evdev_dev, self._imu_dev]:
            if dev:
                try: dev.close()
                except: pass
        if self._hidraw_fd is not None:
            try: os.close(self._hidraw_fd)
            except: pass
        self._evdev_dev = None
        self._imu_dev = None
        self._hidraw_fd = None
        self.connected = False

    def _stick_loop(self):
        self._stick_calib_start = time.time()
        try:
            for event in self._evdev_dev.read_loop():
                if not self._running:
                    break
                if event.type == 3:
                    if event.code == 0:
                        if not self._stick_calibrated:
                            self._stick_calib_samples[0].append(event.value)
                        self.axes[0] = (event.value - self._stick_center[0]) / 32767.0
                    elif event.code == 1:
                        if not self._stick_calibrated:
                            self._stick_calib_samples[1].append(event.value)
                        self.axes[1] = (event.value - self._stick_center[1]) / 32767.0
                    elif event.code == 16: self._dpad_x = event.value  # ABS_HAT0X
                    elif event.code == 17: self._dpad_y = event.value  # ABS_HAT0Y
                    # 校准结束检查
                    if not self._stick_calibrated and time.time() - self._stick_calib_start > 0.5:
                        self._finish_stick_calibration()
                elif event.type == 1:
                    idx = self.BTN_MAP.get(event.code)
                    if idx is not None:
                        self.buttons[idx] = event.value
                    if event.code == 544:   self._dpad_y = -event.value  # BTN_DPAD_UP
                    elif event.code == 545: self._dpad_y = event.value   # BTN_DPAD_DOWN
                    elif event.code == 546: self._dpad_x = -event.value  # BTN_DPAD_LEFT
                    elif event.code == 547: self._dpad_x = event.value   # BTN_DPAD_RIGHT
        except Exception:
            self.connected = False

    def _finish_stick_calibration(self):
        """用中位数计算摇杆零点（抗异常值）"""
        self._stick_calibrated = True
        for i in range(2):
            if self._stick_calib_samples[i]:
                sorted_vals = sorted(self._stick_calib_samples[i])
                self._stick_center[i] = sorted_vals[len(sorted_vals) // 2]
            else:
                self._stick_center[i] = 0.0
        self._stick_calib_samples = [[], []]
        print(f"[JoyCon] 摇杆零点校准完成: X={self._stick_center[0]:.1f} Y={self._stick_center[1]:.1f}")

    def _imu_loop(self):
        GYRO_SCALE = 2000.0 / 32767.0
        ACCEL_SCALE = 8.0 * 9.81 / 32767.0
        GYRO_RAW_CLAMP = 500.0 / GYRO_SCALE   # 钳位到 ±500 deg/s
        try:
            for event in self._imu_dev.read_loop():
                if not self._running:
                    break
                if event.type == 3:
                    code = event.code
                    if code == 0:   self.imu_accel[0] = event.value * ACCEL_SCALE
                    elif code == 1: self.imu_accel[1] = event.value * ACCEL_SCALE
                    elif code == 2: self.imu_accel[2] = event.value * ACCEL_SCALE
                    elif code == 3:
                        if self._calibrating:
                            self._calib_samples[0].append(event.value)
                        raw = event.value - self._gyro_bias[0]
                        self.imu_gyro[0] = max(-GYRO_RAW_CLAMP, min(GYRO_RAW_CLAMP, raw)) * GYRO_SCALE
                    elif code == 4:
                        if self._calibrating:
                            self._calib_samples[1].append(event.value)
                        raw = event.value - self._gyro_bias[1]
                        self.imu_gyro[1] = max(-GYRO_RAW_CLAMP, min(GYRO_RAW_CLAMP, raw)) * GYRO_SCALE
                    elif code == 5:
                        if self._calibrating:
                            self._calib_samples[2].append(event.value)
                        raw = event.value - self._gyro_bias[2]
                        self.imu_gyro[2] = max(-GYRO_RAW_CLAMP, min(GYRO_RAW_CLAMP, raw)) * GYRO_SCALE
        except Exception:
            pass

    def _hidraw_loop(self):
        """从 hidraw 读取 Joy-Con L 按键状态（L/ZL/SL/SR 等）"""
        while self._running and self._hidraw_fd is not None:
            try:
                raw = os.read(self._hidraw_fd, 64)
                if not raw or len(raw) < 5:
                    continue
                report_id = raw[0]
                if report_id not in (0x3F, 0x30, 0x31, 0x21):
                    continue
                # Byte 3: Left Joy-Con buttons (D-pad + SL/SR/L/ZL)
                btn_byte = raw[3]
                self.buttons[0] = (btn_byte >> 0) & 1  # Down
                # 回读完整 D-pad 状态，防止 evdev 丢失松开事件导致 _dpad 卡住
                dpad_down  = (btn_byte >> 0) & 1
                dpad_up    = (btn_byte >> 1) & 1
                dpad_right = (btn_byte >> 2) & 1
                dpad_left  = (btn_byte >> 3) & 1
                self._dpad_x = dpad_right - dpad_left
                self._dpad_y = dpad_down - dpad_up
                self.buttons[5] = (btn_byte >> 4) & 1  # SL
                self.buttons[7] = (btn_byte >> 5) & 1  # SR
                self.buttons[4] = (btn_byte >> 6) & 1  # L
                self.buttons[6] = (btn_byte >> 7) & 1  # ZL
                # Byte 4: Shared buttons (Minus/Plus/L3/R3)
                if len(raw) >= 5:
                    shared = raw[4]
                    self.buttons[8] = (shared >> 0) & 1  # Minus
                    self.buttons[11] = (shared >> 1) & 1  # Plus
                    self.buttons[14] = (shared >> 2) & 1  # L3
                    self.buttons[15] = (shared >> 3) & 1  # R3
            except BlockingIOError:
                time.sleep(0.002)
            except OSError:
                self.connected = False
                break
            except Exception:
                time.sleep(0.002)

    def start_gyro_calibration(self):
        """开始采集陀螺仪零偏样本（控制器需静止）"""
        self._calib_samples = [[], [], []]
        self._calibrating = True

    def finish_gyro_calibration(self) -> bool:
        """结束采集，用中位数计算零偏（抗异常值）"""
        self._calibrating = False
        for i in range(3):
            if len(self._calib_samples[i]) < 10:
                return False
            sorted_vals = sorted(self._calib_samples[i])
            mid = len(sorted_vals) // 2
            self._gyro_bias[i] = sorted_vals[mid]  # 中位数
        self._calib_samples = [[], [], []]
        return True


class SimpleJoystick:
    """Simple joystick for non-Joy-Con gamepads."""

    def __init__(self, device="/dev/input/js0"):
        self.device = device
        self.axes = [0.0] * 8
        self.buttons = [0] * 16
        self.connected = False
        self._fd = None
        self._thread = None
        self._running = False

    def connect(self) -> bool:
        try:
            self._fd = os.open(self.device, os.O_RDONLY | os.O_NONBLOCK)
        except OSError:
            return False
        self.connected = True
        self._running = True
        self._thread = threading.Thread(
            target=self._read_loop, daemon=True, name="joystick_reader")
        self._thread.start()
        return True

    def disconnect(self):
        self._running = False
        if self._thread:
            self._thread.join(timeout=1.0)
        if self._fd is not None:
            try: os.close(self._fd)
            except: pass
            self._fd = None
        self.connected = False

    def _read_loop(self):
        EVENT_FORMAT = "IhBB"
        EVENT_SIZE = struct.calcsize(EVENT_FORMAT)
        while self._running:
            try:
                raw = os.read(self._fd, EVENT_SIZE)
                if len(raw) != EVENT_SIZE:
                    continue
                _ts, value, etype, number = struct.unpack(EVENT_FORMAT, raw)
                etype &= ~0x80
                if etype == 0x02 and number < 8:
                    self.axes[number] = value / 32767.0
                elif etype == 0x01 and number < 16:
                    self.buttons[number] = value
            except BlockingIOError:
                time.sleep(0.002)
            except OSError:
                self.connected = False
                break


class CartesianControlWindow(QMainWindow):
    """Cartesian end-effector control window."""

    def __init__(self):
        super().__init__()
        self.setWindowTitle("EL-A3控制")
        self.setMinimumSize(440, 620)

        self._real_manager = RealRobotManager()
        self._mode = "sim"
        self._real_status_text = "未连接"
        self._step_size = 0.01
        self._rot_step = np.radians(5.0)
        self._dt = 0.05
        self._ik_conv_tol = 0.0005  # IK 收敛容差 0.5mm（原 2mm）
        self._ik_rot_conv_tol = np.radians(0.25)
        self._ik_hold_cost = 10.0
        # 耦合检测容差（自适应：取 step 的 10%，再 clamp）
        self._pos_drift_tol = 0.001   # 占位，实际由 _calc_drift_tolerances 动态计算
        self._rot_drift_tol = np.radians(0.5)

        self._configuration: mink.Configuration | None = None
        self._ee_task: mink.FrameTask | None = None
        self._ik_limits = None
        self._solver = "daqp"

        self._viewer = None
        self._viewer_ctx = None

        # 轨迹可视化
        self._traj_enabled = True
        self._traj_positions: list[np.ndarray] = []
        self._traj_max_points = 500
        self._traj_min_dist = 0.002  # 最小记录间距 (m)

        self._joystick: Optional[JoyConReader | SimpleJoystick] = None
        self._gamepad_active = False
        self._gamepad_speed_levels = [0.05, 0.10, 0.20, 0.40, 0.80]
        self._gamepad_speed_idx = 2
        self._prev_btn = [0] * 16
        self._imu_mode_active = False  # IMU 姿态模式开关

        # 初始化位姿控制日志
        log_dir = Path(__file__).resolve().parent / "logs"
        log_dir.mkdir(exist_ok=True)
        log_file = log_dir / f"pose_control_{datetime.now():%Y%m%d_%H%M%S}.txt"
        self._pose_logger = logging.getLogger("pose_control")
        self._pose_logger.setLevel(logging.DEBUG)
        self._pose_logger.propagate = False
        fh = logging.FileHandler(log_file, encoding="utf-8")
        fh.setFormatter(logging.Formatter("%(asctime)s.%(msecs)03d  %(message)s", datefmt="%H:%M:%S"))
        self._pose_logger.addHandler(fh)
        self._pose_logger.info(f"=== 日志开始: {log_file.name} ===")

        # Joint control panel state
        self._joint_updating = False
        self._joint_sliders = []
        self._joint_spinboxes = []
        self._joint_feedback_labels = []
        self._joint_move_timer = None
        self._joint_move_start_q = None
        self._joint_move_target_q = None
        self._joint_move_target_gripper = 0.0
        self._joint_target_dirty = [False] * 7
        self._joint_move_step = 0
        self._joint_move_total_steps = 0

        # 示教（零力矩拖动 + 轨迹录制 / 回放）状态
        self._teach_active = False
        self._teach_recording = False
        self._teach_use_gravity = True
        self._teach_sample_rate = 10.0
        self._teach_points: list[dict] = []  # [{t, q(list len N_ARM)}]
        self._teach_rec_start = 0.0
        self._teach_last_sample = -1e9
        self._teach_current_traj: list[dict] | None = None
        self._teach_playback_timer = None
        self._teach_playback_start_q: np.ndarray | None = None
        self._teach_playback_prev_q: np.ndarray | None = None
        self._teach_playback_step = 0
        self._teach_playback_total = 0
        self._teach_playback_duration = 0.0
        self._teach_traj_dir = (
            Path(__file__).resolve().parent / "logs" / "teach_trajectories"
        )
        self._teach_traj_dir.mkdir(parents=True, exist_ok=True)
        self._teach_rec_timer = None
        self._teach_status_timer = None

        # 示教点（离散 waypoint）状态
        self._waypoints: list[dict] = []
        self._waypoint_move_timer = None
        self._waypoint_move_start_q: np.ndarray | None = None
        self._waypoint_move_target_q: np.ndarray | None = None
        self._waypoint_move_prev_q: np.ndarray | None = None
        self._waypoint_move_step = 0
        self._waypoint_move_total = 0

        self._init_ui()

        self._home_timer = None
        self._render_timer = QTimer(self)
        self._render_timer.timeout.connect(self._render_tick)

        self._pos_timer = QTimer(self)
        self._pos_timer.timeout.connect(self._refresh_position)
        self._pos_timer.start(200)

        self._gamepad_timer = QTimer(self)
        self._gamepad_timer.timeout.connect(self._gamepad_tick)

    def _init_ui(self):
        central = QWidget()
        self.setCentralWidget(central)
        root_layout = QVBoxLayout(central)
        root_layout.setSpacing(8)
        root_layout.setContentsMargins(12, 12, 12, 12)

        # Mode selection
        mode_group = QGroupBox("模式")
        mode_layout = QHBoxLayout(mode_group)
        self._mode_combo = QComboBox()
        self._mode_combo.addItems(["仿真", "实机"])
        self._mode_combo.currentIndexChanged.connect(self._on_mode_changed)
        mode_layout.addWidget(QLabel("模式:"))
        mode_layout.addWidget(self._mode_combo, 1)
        self._start_btn = QPushButton("启动视图")
        self._start_btn.clicked.connect(self._on_start_viewer)
        mode_layout.addWidget(self._start_btn)
        self._stop_btn = QPushButton("停止")
        self._stop_btn.setEnabled(False)
        self._stop_btn.clicked.connect(self._on_stop)
        mode_layout.addWidget(self._stop_btn)
        root_layout.addWidget(mode_group)

        # Real robot params
        self._real_group = QGroupBox("实机参数")
        real_layout = QGridLayout(self._real_group)
        real_layout.addWidget(QLabel("CAN:"), 0, 0)
        self._can_combo = QComboBox()
        self._can_combo.addItems(["can0", "can1"])
        self._can_combo.setEditable(True)
        real_layout.addWidget(self._can_combo, 0, 1)
        real_layout.addWidget(QLabel("Kp:"), 0, 2)
        self._kp_spin = QDoubleSpinBox()
        self._kp_spin.setRange(0.01, 120.0)
        self._kp_spin.setValue(60.0)
        self._kp_spin.setSingleStep(5.0)
        real_layout.addWidget(self._kp_spin, 0, 3)
        real_layout.addWidget(QLabel("Kd:"), 1, 0)
        self._kd_spin = QDoubleSpinBox()
        self._kd_spin.setRange(0.001, 5.0)
        self._kd_spin.setValue(3.5)
        self._kd_spin.setSingleStep(0.1)
        self._kd_spin.setDecimals(2)
        real_layout.addWidget(self._kd_spin, 1, 1)

        # 关节最大速度上限（deg/s）。默认值与 ELA3MotionOptions 默认 1.2 rad/s 对齐。
        real_layout.addWidget(QLabel("最大速度 (deg/s):"), 2, 0)
        self._vmax_spin = QDoubleSpinBox()
        self._vmax_spin.setRange(5.0, 180.0)
        self._vmax_spin.setValue(float(np.degrees(1.2)))
        self._vmax_spin.setSingleStep(5.0)
        self._vmax_spin.setDecimals(1)
        self._vmax_spin.setToolTip(
            "限制真机各关节的最大角速度；下发后立即生效，用于减慢真机运动"
        )
        self._vmax_spin.valueChanged.connect(self._on_vmax_changed)
        real_layout.addWidget(self._vmax_spin, 2, 1)

        self._real_connect_btn = QPushButton("连接实机")
        self._real_connect_btn.setStyleSheet(
            "QPushButton { background-color: #27ae60; color: white; font-weight: bold; }"
        )
        self._real_connect_btn.clicked.connect(self._toggle_real_robot)
        real_layout.addWidget(self._real_connect_btn, 1, 2)

        self._real_status_label = QLabel(self._real_status_text)
        self._real_status_label.setStyleSheet("color: #888; font-weight: bold;")
        real_layout.addWidget(self._real_status_label, 1, 3)
        self._real_group.setVisible(False)
        root_layout.addWidget(self._real_group)

        # Tab widget
        self._tabs = QTabWidget()
        root_layout.addWidget(self._tabs)

        # Tab 1: Settings
        tab_settings = QWidget()
        tab_settings_layout = QVBoxLayout(tab_settings)
        tab_settings_layout.setContentsMargins(0, 8, 0, 0)

        # Step size
        step_group = QGroupBox("步长")
        step_layout = QVBoxLayout(step_group)

        pos_row = QHBoxLayout()
        pos_row.addWidget(QLabel("位置 (mm):"))
        self._step_spin = QDoubleSpinBox()
        self._step_spin.setRange(0.5, 100.0)
        self._step_spin.setValue(10.0)
        self._step_spin.setSingleStep(1.0)
        self._step_spin.setDecimals(1)
        self._step_spin.valueChanged.connect(self._on_step_changed)
        pos_row.addWidget(self._step_spin)
        for val in [1, 5, 10, 20, 50]:
            btn = QPushButton(f"{val}")
            btn.setFixedWidth(40)
            btn.clicked.connect(lambda _, v=val: self._step_spin.setValue(v))
            pos_row.addWidget(btn)
        step_layout.addLayout(pos_row)

        rot_row = QHBoxLayout()
        rot_row.addWidget(QLabel("旋转 (deg):"))
        self._rot_spin = QDoubleSpinBox()
        self._rot_spin.setRange(0.5, 45.0)
        self._rot_spin.setValue(5.0)
        self._rot_spin.setSingleStep(0.5)
        self._rot_spin.setDecimals(1)
        self._rot_spin.valueChanged.connect(self._on_rot_changed)
        rot_row.addWidget(self._rot_spin)
        for val in [1, 5, 10, 15, 30]:
            btn = QPushButton(f"{val}")
            btn.setFixedWidth(40)
            btn.clicked.connect(lambda _, v=val: self._rot_spin.setValue(v))
            rot_row.addWidget(btn)
        step_layout.addLayout(rot_row)
        tab_settings_layout.addWidget(step_group)

        # Gamepad
        gp_group = QGroupBox("手柄 (Joy-Con / Xbox)")
        gp_layout = QGridLayout(gp_group)

        self._gp_connect_btn = QPushButton("连接手柄")
        self._gp_connect_btn.clicked.connect(self._toggle_gamepad)
        gp_layout.addWidget(self._gp_connect_btn, 0, 0, 1, 2)

        self._gp_status_label = QLabel("未连接")
        self._gp_status_label.setStyleSheet("color: #888;")
        gp_layout.addWidget(self._gp_status_label, 0, 2, 1, 2)

        gp_layout.addWidget(QLabel("速度:"), 1, 0)
        self._gp_speed_btn = QPushButton("中速 (0.20)")
        self._gp_speed_btn.setFixedWidth(100)
        self._gp_speed_btn.clicked.connect(self._cycle_gamepad_speed)
        gp_layout.addWidget(self._gp_speed_btn, 1, 1)

        self._gp_active_check = QCheckBox("启用手柄控制")
        self._gp_active_check.setEnabled(False)
        self._gp_active_check.toggled.connect(self._on_gamepad_toggled)
        gp_layout.addWidget(self._gp_active_check, 1, 2, 1, 2)

        self._imu_check = QCheckBox("IMU 姿态控制")
        self._imu_check.setEnabled(False)
        self._imu_check.setToolTip("使用 Joy-Con 陀螺仪控制横滚/俯仰/偏航\n按 - 键切换开关，使用前请先校准")
        self._imu_check.toggled.connect(self._on_imu_toggled)
        gp_layout.addWidget(self._imu_check, 2, 0)

        self._imu_calib_btn = QPushButton("校准 IMU")
        self._imu_calib_btn.setEnabled(False)
        self._imu_calib_btn.setFixedWidth(70)
        self._imu_calib_btn.clicked.connect(self._calibrate_imu)
        gp_layout.addWidget(self._imu_calib_btn, 2, 1)

        self._imu_status = QLabel("IMU: ---")
        self._imu_status.setStyleSheet("font-size: 11px; color: #888;")
        gp_layout.addWidget(self._imu_status, 2, 2, 1, 2)

        self._traj_check = QCheckBox("显示末端轨迹")
        self._traj_check.setChecked(True)
        self._traj_check.toggled.connect(self._on_traj_toggled)
        gp_layout.addWidget(self._traj_check, 3, 0)

        info = QLabel(
            "摇杆: XY平移  |  SL/SR: 横滚  |  D-pad: 俯仰/偏航\n"
            "L/ZL: Z升降  |  -: 切换IMU  |  A:切换速度  |  B:回零"
        )
        info.setStyleSheet("font-size: 11px; color: #666;")
        gp_layout.addWidget(info, 4, 0, 1, 4)
        tab_settings_layout.addWidget(gp_group)
        tab_settings_layout.addStretch()

        self._tabs.addTab(tab_settings, "设置")

        # Tab 2: Motion control
        tab_motion = QWidget()
        tab_motion_layout = QVBoxLayout(tab_motion)
        tab_motion_layout.setContentsMargins(0, 8, 0, 0)

        # Motion control buttons
        ctrl_group = QGroupBox("笛卡尔运动控制")
        ctrl_layout = QGridLayout(ctrl_group)
        ctrl_layout.setSpacing(6)
        btn_css = "QPushButton {{ min-width: 70px; min-height: 50px; font-size: 16px; font-weight: bold; color: white; background-color: {}; }}"

        ctrl_layout.addWidget(QLabel("X:"), 0, 0)
        self._btn_x_neg = QPushButton("X -")
        self._btn_x_neg.setStyleSheet(btn_css.format("#e74c3c"))
        self._btn_x_neg.clicked.connect(lambda: self._move("x", -1))
        ctrl_layout.addWidget(self._btn_x_neg, 0, 1)
        self._btn_x_pos = QPushButton("X +")
        self._btn_x_pos.setStyleSheet(btn_css.format("#c0392b"))
        self._btn_x_pos.clicked.connect(lambda: self._move("x", 1))
        ctrl_layout.addWidget(self._btn_x_pos, 0, 2)

        ctrl_layout.addWidget(QLabel("Y:"), 1, 0)
        self._btn_y_neg = QPushButton("Y -")
        self._btn_y_neg.setStyleSheet(btn_css.format("#2ecc71"))
        self._btn_y_neg.clicked.connect(lambda: self._move("y", -1))
        ctrl_layout.addWidget(self._btn_y_neg, 1, 1)
        self._btn_y_pos = QPushButton("Y +")
        self._btn_y_pos.setStyleSheet(btn_css.format("#27ae60"))
        self._btn_y_pos.clicked.connect(lambda: self._move("y", 1))
        ctrl_layout.addWidget(self._btn_y_pos, 1, 2)

        ctrl_layout.addWidget(QLabel("Z:"), 2, 0)
        self._btn_z_neg = QPushButton("Z -")
        self._btn_z_neg.setStyleSheet(btn_css.format("#3498db"))
        self._btn_z_neg.clicked.connect(lambda: self._move("z", -1))
        ctrl_layout.addWidget(self._btn_z_neg, 2, 1)
        self._btn_z_pos = QPushButton("Z +")
        self._btn_z_pos.setStyleSheet(btn_css.format("#2980b9"))
        self._btn_z_pos.clicked.connect(lambda: self._move("z", 1))
        ctrl_layout.addWidget(self._btn_z_pos, 2, 2)

        ctrl_layout.addWidget(QLabel("横滚:"), 3, 0)
        self._btn_r_neg = QPushButton("R -")
        self._btn_r_neg.setStyleSheet(btn_css.format("#9b59b6"))
        self._btn_r_neg.clicked.connect(lambda: self._rotate("roll", -1))
        ctrl_layout.addWidget(self._btn_r_neg, 3, 1)
        self._btn_r_pos = QPushButton("R +")
        self._btn_r_pos.setStyleSheet(btn_css.format("#8e44ad"))
        self._btn_r_pos.clicked.connect(lambda: self._rotate("roll", 1))
        ctrl_layout.addWidget(self._btn_r_pos, 3, 2)

        ctrl_layout.addWidget(QLabel("俯仰:"), 4, 0)
        self._btn_p_neg = QPushButton("P -")
        self._btn_p_neg.setStyleSheet(btn_css.format("#f39c12"))
        self._btn_p_neg.clicked.connect(lambda: self._rotate("pitch", -1))
        ctrl_layout.addWidget(self._btn_p_neg, 4, 1)
        self._btn_p_pos = QPushButton("P +")
        self._btn_p_pos.setStyleSheet(btn_css.format("#e67e22"))
        self._btn_p_pos.clicked.connect(lambda: self._rotate("pitch", 1))
        ctrl_layout.addWidget(self._btn_p_pos, 4, 2)

        ctrl_layout.addWidget(QLabel("偏航:"), 5, 0)
        self._btn_yo_neg = QPushButton("Y -")
        self._btn_yo_neg.setStyleSheet(btn_css.format("#1abc9c"))
        self._btn_yo_neg.clicked.connect(lambda: self._rotate("yaw", -1))
        ctrl_layout.addWidget(self._btn_yo_neg, 5, 1)
        self._btn_yo_pos = QPushButton("Y +")
        self._btn_yo_pos.setStyleSheet(btn_css.format("#16a085"))
        self._btn_yo_pos.clicked.connect(lambda: self._rotate("yaw", 1))
        ctrl_layout.addWidget(self._btn_yo_pos, 5, 2)

        self._btn_home = QPushButton("回零 (Home)")
        self._btn_home.setStyleSheet(
            "QPushButton { min-height: 44px; font-size: 15px; font-weight: bold; "
            "color: white; background-color: #d35400; border-radius: 4px; }"
            "QPushButton:hover { background-color: #e67e22; }"
        )
        self._btn_home.clicked.connect(self._move_home)
        ctrl_layout.addWidget(self._btn_home, 6, 0, 1, 3)
        tab_motion_layout.addWidget(ctrl_group)

        # Position display
        pos_group = QGroupBox("当前末端执行器位姿")
        pos_layout = QVBoxLayout(pos_group)
        self._home_warning_label = QLabel("")
        self._home_warning_label.setStyleSheet(
            "color: white; background-color: #c0392b; font-weight: bold; "
            "padding: 4px; border-radius: 3px;"
        )
        self._home_warning_label.setVisible(False)
        pos_layout.addWidget(self._home_warning_label)
        self._pos_label = QLabel("Pos: ---")
        self._pos_label.setStyleSheet("font-family: monospace; font-size: 13px;")
        pos_layout.addWidget(self._pos_label)
        self._rpy_label = QLabel("RPY: ---")
        self._rpy_label.setStyleSheet("font-family: monospace; font-size: 13px;")
        pos_layout.addWidget(self._rpy_label)
        tab_motion_layout.addWidget(pos_group)
        tab_motion_layout.addStretch()

        self._tabs.addTab(tab_motion, "运动控制")

        # Tab 3: Joint control
        tab_joint = QWidget()
        tab_joint_layout = QVBoxLayout(tab_joint)
        tab_joint_layout.setContentsMargins(0, 8, 0, 0)

        joint_group = QGroupBox("关节控制")
        joint_grid = QGridLayout(joint_group)
        joint_grid.setSpacing(4)

        # Headers
        headers = ["关节", "控制滑块", "目标角度", "当前角度"]
        for col, h in enumerate(headers):
            lbl = QLabel(h)
            lbl.setAlignment(Qt.AlignmentFlag.AlignCenter)
            lbl.setStyleSheet("font-weight: bold;")
            joint_grid.addWidget(lbl, 0, col)

        joint_names = ["L1", "L2", "L3", "L4", "L5", "L6", "L7"]

        for i in range(7):
            row = i + 1
            joint_id = i + 1
            lo, hi = JOINT_LIMITS_DEG[joint_id]

            # Name label
            name_lbl = QLabel(joint_names[i])
            name_lbl.setStyleSheet(f"color: {JOINT_COLORS[i]}; font-weight: bold; font-size: 14px;")
            name_lbl.setFixedWidth(30)
            joint_grid.addWidget(name_lbl, row, 0)

            # Slider
            slider = QSlider(Qt.Orientation.Horizontal)
            slider.setRange(0, SLIDER_RESOLUTION)
            mid = int((0.0 - lo) / (hi - lo) * SLIDER_RESOLUTION) if hi != lo else SLIDER_RESOLUTION // 2
            slider.setValue(max(0, min(SLIDER_RESOLUTION, mid)))
            slider.valueChanged.connect(lambda v, idx=i: self._on_joint_slider_changed(idx, v))
            self._joint_sliders.append(slider)
            joint_grid.addWidget(slider, row, 1)

            # SpinBox
            spinbox = QDoubleSpinBox()
            spinbox.setRange(lo, hi)
            spinbox.setDecimals(1)
            spinbox.setSingleStep(1.0)
            spinbox.setSuffix("°")
            spinbox.setFixedWidth(85)
            spinbox.setValue(0.0 if lo <= 0.0 <= hi else lo)
            spinbox.valueChanged.connect(lambda v, idx=i: self._on_joint_spinbox_changed(idx, v))
            self._joint_spinboxes.append(spinbox)
            joint_grid.addWidget(spinbox, row, 2)

            # Feedback label
            fb_label = QLabel("0.0°")
            fb_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
            fb_label.setFixedWidth(70)
            fb_label.setStyleSheet(f"color: {JOINT_COLORS[i]}; font-family: monospace;")
            self._joint_feedback_labels.append(fb_label)
            joint_grid.addWidget(fb_label, row, 3)

        joint_grid.setColumnStretch(0, 0)
        joint_grid.setColumnStretch(1, 5)
        joint_grid.setColumnStretch(2, 0)
        joint_grid.setColumnStretch(3, 0)
        tab_joint_layout.addWidget(joint_group)

        # Buttons
        btn_layout = QHBoxLayout()
        self._joint_zero_btn = QPushButton("回零")
        self._joint_zero_btn.setFixedHeight(30)
        self._joint_zero_btn.clicked.connect(self._joint_go_zero)
        btn_layout.addWidget(self._joint_zero_btn)

        self._joint_sync_btn = QPushButton("同步当前位置")
        self._joint_sync_btn.setFixedHeight(30)
        self._joint_sync_btn.clicked.connect(self._joint_sync_to_sim)
        btn_layout.addWidget(self._joint_sync_btn)

        btn_layout.addWidget(QLabel("运动时长:"))
        self._joint_duration_spin = QDoubleSpinBox()
        self._joint_duration_spin.setRange(0.2, 30.0)
        self._joint_duration_spin.setDecimals(1)
        self._joint_duration_spin.setSingleStep(0.5)
        self._joint_duration_spin.setValue(1.0)
        self._joint_duration_spin.setSuffix(" s")
        self._joint_duration_spin.setFixedWidth(80)
        btn_layout.addWidget(self._joint_duration_spin)

        btn_layout.addStretch()

        self._joint_send_btn = QPushButton("发送指令")
        self._joint_send_btn.setFixedHeight(30)
        self._joint_send_btn.setStyleSheet("QPushButton { background-color: #2980b9; color: white; font-weight: bold; }")
        self._joint_send_btn.clicked.connect(self._joint_send_command)
        btn_layout.addWidget(self._joint_send_btn)

        tab_joint_layout.addLayout(btn_layout)

        # Current joint angles display
        angles_group = QGroupBox("当前关节角度 (rad)")
        angles_layout = QVBoxLayout(angles_group)
        self._joint_angles_label = QLabel("---")
        self._joint_angles_label.setStyleSheet("font-family: monospace; font-size: 12px;")
        angles_layout.addWidget(self._joint_angles_label)
        tab_joint_layout.addWidget(angles_group)

        tab_joint_layout.addStretch()

        self._tabs.addTab(tab_joint, "关节控制")

        # Tab 4: 示教（零力矩拖动 + 轨迹录制 / 回放）
        self._build_teach_tab()

        # Tab 5: 示教点（离散关键点管理）
        self._build_waypoint_tab()

        # Status bar
        self._status_bar = QStatusBar()
        self.setStatusBar(self._status_bar)
        self._status_bar.showMessage("就绪")

    # ---- 示教 Tab ----------------------------------------------------

    def _build_teach_tab(self):
        tab_teach = QWidget()
        layout = QVBoxLayout(tab_teach)
        layout.setContentsMargins(0, 8, 0, 0)

        warn = QLabel(
            "⚠ 示教依赖实机零力矩模式，仅在“实机”模式下已连接时可用；"
            "启用后请托住机械臂，缓慢移动。"
        )
        warn.setWordWrap(True)
        warn.setStyleSheet(
            "color: white; background-color: #c0392b; font-weight: bold; "
            "padding: 6px; border-radius: 3px;"
        )
        layout.addWidget(warn)

        # 零力矩模式
        zt_group = QGroupBox("零力矩拖动")
        zt_layout = QVBoxLayout(zt_group)
        self._teach_gravity_check = QCheckBox("启用重力补偿 (推荐)")
        self._teach_gravity_check.setChecked(True)
        self._teach_gravity_check.setToolTip(
            "使用 Pinocchio RNEA 实时补偿重力矩，让拖动更省力"
        )
        zt_layout.addWidget(self._teach_gravity_check)

        self._teach_toggle_btn = QPushButton("进入示教模式")
        self._teach_toggle_btn.setStyleSheet(
            "QPushButton { min-height: 40px; font-size: 15px; font-weight: bold; "
            "color: white; background-color: #27ae60; border-radius: 4px; }"
            "QPushButton:hover { background-color: #2ecc71; }"
        )
        self._teach_toggle_btn.clicked.connect(self._on_teach_toggle)
        zt_layout.addWidget(self._teach_toggle_btn)

        self._teach_mode_label = QLabel("状态: 未启用")
        self._teach_mode_label.setStyleSheet(
            "font-family: monospace; color: #888;"
        )
        zt_layout.addWidget(self._teach_mode_label)
        layout.addWidget(zt_group)

        # 轨迹录制
        rec_group = QGroupBox("轨迹录制")
        rec_layout = QGridLayout(rec_group)
        rec_layout.addWidget(QLabel("采样率:"), 0, 0)
        self._teach_rate_spin = QDoubleSpinBox()
        self._teach_rate_spin.setRange(1.0, 50.0)
        self._teach_rate_spin.setValue(10.0)
        self._teach_rate_spin.setSuffix(" Hz")
        self._teach_rate_spin.setDecimals(1)
        self._teach_rate_spin.valueChanged.connect(
            lambda v: setattr(self, "_teach_sample_rate", float(v))
        )
        rec_layout.addWidget(self._teach_rate_spin, 0, 1)

        self._teach_rec_btn = QPushButton("开始录制")
        self._teach_rec_btn.setStyleSheet(
            "QPushButton { background-color: #2980b9; color: white; "
            "font-weight: bold; min-height: 30px; }"
        )
        self._teach_rec_btn.clicked.connect(self._on_teach_record_toggle)
        rec_layout.addWidget(self._teach_rec_btn, 0, 2)

        self._teach_rec_status = QLabel("未录制")
        self._teach_rec_status.setStyleSheet("color: #888;")
        rec_layout.addWidget(self._teach_rec_status, 1, 0, 1, 3)
        layout.addWidget(rec_group)

        # 轨迹列表 + 回放
        traj_group = QGroupBox("轨迹管理")
        traj_layout = QVBoxLayout(traj_group)
        self._teach_traj_list = QListWidget()
        self._teach_traj_list.setMinimumHeight(120)
        traj_layout.addWidget(self._teach_traj_list)

        btn_row = QHBoxLayout()
        self._teach_refresh_btn = QPushButton("刷新列表")
        self._teach_refresh_btn.clicked.connect(self._teach_refresh_list)
        btn_row.addWidget(self._teach_refresh_btn)

        self._teach_load_btn = QPushButton("导入 JSON...")
        self._teach_load_btn.clicked.connect(self._teach_load_file)
        btn_row.addWidget(self._teach_load_btn)

        self._teach_delete_btn = QPushButton("删除选中")
        self._teach_delete_btn.clicked.connect(self._teach_delete_selected)
        btn_row.addWidget(self._teach_delete_btn)
        traj_layout.addLayout(btn_row)

        play_row = QHBoxLayout()
        play_row.addWidget(QLabel("回放倍速:"))
        self._teach_play_speed = QDoubleSpinBox()
        self._teach_play_speed.setRange(0.1, 5.0)
        self._teach_play_speed.setValue(1.0)
        self._teach_play_speed.setDecimals(2)
        self._teach_play_speed.setSingleStep(0.1)
        self._teach_play_speed.setSuffix("×")
        play_row.addWidget(self._teach_play_speed)

        self._teach_play_btn = QPushButton("回放选中")
        self._teach_play_btn.setStyleSheet(
            "QPushButton { background-color: #d35400; color: white; "
            "font-weight: bold; min-height: 30px; }"
        )
        self._teach_play_btn.clicked.connect(self._on_teach_play)
        play_row.addWidget(self._teach_play_btn)

        self._teach_stop_btn = QPushButton("停止回放")
        self._teach_stop_btn.clicked.connect(self._on_teach_stop_play)
        play_row.addWidget(self._teach_stop_btn)
        traj_layout.addLayout(play_row)
        layout.addWidget(traj_group)

        layout.addStretch()
        self._tabs.addTab(tab_teach, "示教")
        self._teach_refresh_list()

    # ---- 示教点 Tab -------------------------------------------------

    def _build_waypoint_tab(self):
        tab_wp = QWidget()
        layout = QVBoxLayout(tab_wp)
        layout.setContentsMargins(0, 8, 0, 0)

        info = QLabel(
            "示教点：记录若干关键关节位姿，支持插入/删除/拖拽排序、"
            "JSON 导入导出，以及按顺序移动到指定点。"
        )
        info.setWordWrap(True)
        info.setStyleSheet(
            "color: #2c3e50; background-color: #ecf0f1; padding: 6px; "
            "border-radius: 3px;"
        )
        layout.addWidget(info)

        wp_group = QGroupBox("示教点列表")
        wp_layout = QVBoxLayout(wp_group)

        self._wp_list = QListWidget()
        self._wp_list.setMinimumHeight(220)
        # 支持拖拽调整示教点顺序
        self._wp_list.setDragDropMode(QListWidget.DragDropMode.InternalMove)
        self._wp_list.model().rowsMoved.connect(self._on_wp_rows_moved)
        wp_layout.addWidget(self._wp_list)

        wp_btn_row1 = QHBoxLayout()
        self._wp_capture_btn = QPushButton("记录当前点")
        self._wp_capture_btn.setStyleSheet(
            "QPushButton { background-color: #16a085; color: white; "
            "font-weight: bold; min-height: 28px; }"
        )
        self._wp_capture_btn.clicked.connect(self._on_wp_capture)
        wp_btn_row1.addWidget(self._wp_capture_btn)

        self._wp_insert_before_btn = QPushButton("插入到选中前")
        self._wp_insert_before_btn.setToolTip(
            "把当前关节角作为新示教点插入到选中项之前"
        )
        self._wp_insert_before_btn.clicked.connect(
            lambda: self._on_wp_insert(after=False)
        )
        wp_btn_row1.addWidget(self._wp_insert_before_btn)

        self._wp_insert_after_btn = QPushButton("插入到选中后")
        self._wp_insert_after_btn.setToolTip(
            "把当前关节角作为新示教点插入到选中项之后"
        )
        self._wp_insert_after_btn.clicked.connect(
            lambda: self._on_wp_insert(after=True)
        )
        wp_btn_row1.addWidget(self._wp_insert_after_btn)

        self._wp_delete_btn = QPushButton("删除选中")
        self._wp_delete_btn.clicked.connect(self._on_wp_delete)
        wp_btn_row1.addWidget(self._wp_delete_btn)

        self._wp_clear_btn = QPushButton("清空全部")
        self._wp_clear_btn.clicked.connect(self._on_wp_clear)
        wp_btn_row1.addWidget(self._wp_clear_btn)
        wp_layout.addLayout(wp_btn_row1)

        wp_btn_row2 = QHBoxLayout()
        self._wp_import_btn = QPushButton("导入 JSON...")
        self._wp_import_btn.clicked.connect(self._on_wp_import)
        wp_btn_row2.addWidget(self._wp_import_btn)

        self._wp_export_btn = QPushButton("导出 JSON...")
        self._wp_export_btn.clicked.connect(self._on_wp_export)
        wp_btn_row2.addWidget(self._wp_export_btn)

        wp_btn_row2.addWidget(QLabel("移动时长:"))
        self._wp_duration_spin = QDoubleSpinBox()
        self._wp_duration_spin.setRange(0.2, 30.0)
        self._wp_duration_spin.setValue(2.0)
        self._wp_duration_spin.setSingleStep(0.5)
        self._wp_duration_spin.setDecimals(1)
        self._wp_duration_spin.setSuffix(" s")
        self._wp_duration_spin.setFixedWidth(80)
        wp_btn_row2.addWidget(self._wp_duration_spin)

        self._wp_goto_btn = QPushButton("移动到选中点")
        self._wp_goto_btn.setStyleSheet(
            "QPushButton { background-color: #d35400; color: white; "
            "font-weight: bold; min-height: 28px; }"
        )
        self._wp_goto_btn.clicked.connect(self._on_wp_goto)
        wp_btn_row2.addWidget(self._wp_goto_btn)
        wp_layout.addLayout(wp_btn_row2)

        self._wp_status = QLabel("示教点: 0")
        self._wp_status.setStyleSheet("color: #888; font-family: monospace;")
        wp_layout.addWidget(self._wp_status)
        layout.addWidget(wp_group)

        layout.addStretch()
        self._tabs.addTab(tab_wp, "示教点")

    # ---- 示教：零力矩模式切换 ---------------------------------------

    def _on_teach_toggle(self):
        if self._teach_active:
            self._teach_exit()
        else:
            self._teach_enter()

    def _teach_enter(self):
        if self._mode != "real" or not self._real_manager.is_connected:
            QMessageBox.warning(
                self, "无法进入示教",
                "示教需要实机模式并已连接机械臂。请先切到“实机”并连接。",
            )
            return
        if self._teach_playback_timer is not None:
            QMessageBox.information(self, "示教", "请先停止回放再进入示教。")
            return
        use_gravity = self._teach_gravity_check.isChecked()
        ok = self._real_manager.enter_zero_torque(gravity=use_gravity)
        if not ok:
            QMessageBox.critical(
                self, "示教失败",
                "无法进入零力矩模式（请检查 SDK 支持与机械臂状态）。",
            )
            return
        self._teach_active = True
        self._teach_use_gravity = use_gravity
        self._teach_toggle_btn.setText("退出示教模式")
        self._teach_toggle_btn.setStyleSheet(
            "QPushButton { min-height: 40px; font-size: 15px; font-weight: bold; "
            "color: white; background-color: #c0392b; border-radius: 4px; }"
            "QPushButton:hover { background-color: #e74c3c; }"
        )
        mode_text = "重力补偿" if use_gravity else "基础"
        self._teach_mode_label.setText(f"状态: 零力矩 ({mode_text}) — 可拖动机械臂")
        self._teach_mode_label.setStyleSheet(
            "font-family: monospace; color: #27ae60; font-weight: bold;"
        )
        self._teach_gravity_check.setEnabled(False)
        self._status_bar.showMessage("已进入示教模式，可手动拖动机械臂")

    def _teach_exit(self):
        # 若正在录制，先停下
        if self._teach_recording:
            self._on_teach_record_toggle()
        ok = self._real_manager.exit_zero_torque(gravity=self._teach_use_gravity)
        self._teach_active = False
        self._teach_toggle_btn.setText("进入示教模式")
        self._teach_toggle_btn.setStyleSheet(
            "QPushButton { min-height: 40px; font-size: 15px; font-weight: bold; "
            "color: white; background-color: #27ae60; border-radius: 4px; }"
            "QPushButton:hover { background-color: #2ecc71; }"
        )
        self._teach_mode_label.setText("状态: 未启用")
        self._teach_mode_label.setStyleSheet(
            "font-family: monospace; color: #888;"
        )
        self._teach_gravity_check.setEnabled(True)
        if ok:
            self._status_bar.showMessage("已退出示教模式")
        else:
            self._status_bar.showMessage("退出示教时接口返回失败，请检查机械臂状态")

    # ---- 示教：录制 ------------------------------------------------

    def _on_teach_record_toggle(self):
        if self._teach_recording:
            self._teach_stop_recording()
        else:
            self._teach_start_recording()

    def _teach_start_recording(self):
        if not self._teach_active:
            QMessageBox.warning(
                self, "无法录制", "请先进入示教模式再开始录制。"
            )
            return
        self._teach_points = []
        self._teach_rec_start = time.time()
        self._teach_last_sample = -1e9
        self._teach_recording = True
        self._teach_rec_btn.setText("停止录制")
        self._teach_rec_btn.setStyleSheet(
            "QPushButton { background-color: #c0392b; color: white; "
            "font-weight: bold; min-height: 30px; }"
        )
        self._teach_rec_status.setText("录制中... 采样数: 0")
        self._teach_rec_status.setStyleSheet("color: #e67e22; font-weight: bold;")
        # 采样定时器：以 2× 采样率触发，靠时间判定去重
        interval_ms = max(10, int(1000.0 / (self._teach_sample_rate * 2.0)))
        self._teach_rec_timer = QTimer(self)
        self._teach_rec_timer.timeout.connect(self._teach_sample_tick)
        self._teach_rec_timer.start(interval_ms)

    def _teach_sample_tick(self):
        if not self._teach_recording:
            return
        q = self._real_manager.get_joint_feedback()
        if q is None:
            return
        now = time.time()
        elapsed = now - self._teach_rec_start
        interval = 1.0 / max(0.1, self._teach_sample_rate)
        if elapsed - self._teach_last_sample < interval * 0.9:
            return
        self._teach_last_sample = elapsed
        self._teach_points.append({"t": elapsed, "q": [float(v) for v in q]})
        self._teach_rec_status.setText(
            f"录制中... 采样数: {len(self._teach_points)}  "
            f"时长: {elapsed:.1f}s"
        )

    def _teach_stop_recording(self):
        self._teach_recording = False
        if self._teach_rec_timer is not None:
            self._teach_rec_timer.stop()
            self._teach_rec_timer.deleteLater()
            self._teach_rec_timer = None
        self._teach_rec_btn.setText("开始录制")
        self._teach_rec_btn.setStyleSheet(
            "QPushButton { background-color: #2980b9; color: white; "
            "font-weight: bold; min-height: 30px; }"
        )
        n = len(self._teach_points)
        if n < 2:
            self._teach_rec_status.setText(
                f"录制结束，样本过少 ({n})，未保存"
            )
            self._teach_rec_status.setStyleSheet("color: #e74c3c;")
            self._teach_points = []
            return
        # 保存
        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        fname = self._teach_traj_dir / f"teach_{ts}.json"
        data_out = {
            "name": fname.stem,
            "created_at": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "sample_rate_hz": float(self._teach_sample_rate),
            "num_joints": int(N_ARM),
            "points": self._teach_points,
        }
        try:
            with open(fname, "w", encoding="utf-8") as f:
                json.dump(data_out, f, indent=2, ensure_ascii=False)
            self._teach_rec_status.setText(
                f"已保存 {fname.name}  样本: {n}  时长: "
                f"{self._teach_points[-1]['t']:.1f}s"
            )
            self._teach_rec_status.setStyleSheet(
                "color: #27ae60; font-weight: bold;"
            )
        except Exception as exc:
            self._teach_rec_status.setText(f"保存失败: {exc}")
            self._teach_rec_status.setStyleSheet("color: #e74c3c;")
        self._teach_points = []
        self._teach_refresh_list()

    # ---- 示教：轨迹管理 --------------------------------------------

    def _teach_refresh_list(self):
        self._teach_traj_list.clear()
        try:
            files = sorted(self._teach_traj_dir.glob("*.json"))
        except Exception:
            files = []
        for f in files:
            try:
                with open(f, "r", encoding="utf-8") as fh:
                    d = json.load(fh)
                pts = d.get("points", [])
                dur = pts[-1]["t"] - pts[0]["t"] if len(pts) >= 2 else 0.0
                self._teach_traj_list.addItem(
                    f"{f.name}   ({len(pts)} pts, {dur:.1f}s)"
                )
            except Exception:
                self._teach_traj_list.addItem(f"{f.name}   (损坏)")

    def _teach_selected_path(self) -> Optional[Path]:
        item = self._teach_traj_list.currentItem()
        if item is None:
            return None
        name = item.text().split()[0]
        p = self._teach_traj_dir / name
        return p if p.is_file() else None

    def _teach_load_file(self):
        fname, _ = QFileDialog.getOpenFileName(
            self, "导入轨迹 JSON", str(self._teach_traj_dir), "JSON (*.json)"
        )
        if not fname:
            return
        src = Path(fname)
        if src.parent != self._teach_traj_dir:
            dst = self._teach_traj_dir / src.name
            try:
                dst.write_bytes(src.read_bytes())
            except Exception as exc:
                QMessageBox.warning(self, "导入失败", str(exc))
                return
        self._teach_refresh_list()

    def _teach_delete_selected(self):
        p = self._teach_selected_path()
        if p is None:
            return
        reply = QMessageBox.question(
            self, "删除轨迹",
            f"确定删除 {p.name} 吗？此操作不可撤销。",
        )
        if reply != QMessageBox.StandardButton.Yes:
            return
        try:
            p.unlink()
        except Exception as exc:
            QMessageBox.warning(self, "删除失败", str(exc))
            return
        self._teach_refresh_list()

    # ---- 示教：回放 ------------------------------------------------

    def _on_teach_play(self):
        if self._teach_playback_timer is not None:
            QMessageBox.information(self, "回放", "已在回放中，请先停止。")
            return
        if self._teach_active:
            QMessageBox.warning(
                self, "无法回放", "请先退出示教模式再回放（回放需要位置控制）。"
            )
            return
        if self._mode != "real" or not self._real_manager.is_connected:
            QMessageBox.warning(
                self, "无法回放", "回放需要在“实机”模式下连接机械臂。"
            )
            return
        p = self._teach_selected_path()
        if p is None:
            QMessageBox.information(self, "回放", "请先选中一条轨迹。")
            return
        try:
            with open(p, "r", encoding="utf-8") as fh:
                d = json.load(fh)
            pts = d.get("points", [])
        except Exception as exc:
            QMessageBox.warning(self, "读取失败", str(exc))
            return
        if len(pts) < 2:
            QMessageBox.warning(self, "回放", "轨迹样本过少。")
            return
        self._teach_current_traj = pts
        # 起点：先用当前反馈作为起始点，插到轨迹最前，避免第一段跳变
        cur_q = self._real_manager.get_joint_feedback()
        if cur_q is None:
            QMessageBox.warning(self, "回放", "未能读取当前关节反馈。")
            return
        self._teach_playback_start_q = np.array(cur_q, dtype=float)
        self._teach_playback_prev_q = self._teach_playback_start_q.copy()

        speed = max(0.1, float(self._teach_play_speed.value()))
        total_duration = (pts[-1]["t"] - pts[0]["t"]) / speed
        tick_ms = 20
        self._teach_playback_total = max(1, int(total_duration * 1000 / tick_ms))
        self._teach_playback_step = 0
        self._teach_playback_duration = total_duration
        self._teach_playback_timer = QTimer(self)
        self._teach_playback_timer.timeout.connect(self._teach_playback_tick)
        self._teach_playback_timer.start(tick_ms)
        self._status_bar.showMessage(
            f"开始回放 {p.name}  速度 {speed:.2f}×  预计 {total_duration:.1f}s"
        )

    def _teach_playback_tick(self):
        if self._teach_playback_timer is None or self._teach_current_traj is None:
            return
        self._teach_playback_step += 1
        t_norm = min(
            self._teach_playback_step / self._teach_playback_total, 1.0
        )
        pts = self._teach_current_traj
        t_traj = pts[0]["t"] + t_norm * (pts[-1]["t"] - pts[0]["t"])

        # 二分找区间
        lo, hi = 0, len(pts) - 1
        while lo + 1 < hi:
            mid = (lo + hi) // 2
            if pts[mid]["t"] <= t_traj:
                lo = mid
            else:
                hi = mid
        t0, t1 = pts[lo]["t"], pts[hi]["t"]
        q0 = np.array(pts[lo]["q"], dtype=float)
        q1 = np.array(pts[hi]["q"], dtype=float)
        if t1 <= t0:
            q_interp = q1
        else:
            alpha = (t_traj - t0) / (t1 - t0)
            q_interp = q0 + (q1 - q0) * alpha

        # 首帧混入 start_q 平滑过渡
        blend_ratio = min(1.0, self._teach_playback_step / 25.0)  # ~0.5s
        q_cmd = (
            (1.0 - blend_ratio) * self._teach_playback_start_q
            + blend_ratio * q_interp
        )

        try:
            self._real_manager.send_joint_command(
                q_cmd, start_q=self._teach_playback_prev_q,
            )
        except Exception as exc:
            self._status_bar.showMessage(f"回放下发失败: {exc}")
            self._on_teach_stop_play()
            return
        self._teach_playback_prev_q = q_cmd

        if t_norm >= 1.0:
            self._on_teach_stop_play(finished=True)

    def _on_teach_stop_play(self, finished: bool = False):
        if self._teach_playback_timer is not None:
            self._teach_playback_timer.stop()
            self._teach_playback_timer.deleteLater()
            self._teach_playback_timer = None
        self._teach_current_traj = None
        self._teach_playback_start_q = None
        self._teach_playback_prev_q = None
        if finished:
            self._status_bar.showMessage("回放完成")
        else:
            self._status_bar.showMessage("回放已停止")

    # ---- 示教点：记录 / 删除 / 清空 / 导入 / 导出 / 移动 ----

    def _wp_current_joint_angles(self) -> np.ndarray | None:
        """获取当前关节角度（实机优先反馈，否则用仿真状态）。"""
        if self._mode == "real" and self._real_manager.is_connected:
            q = self._real_manager.get_joint_feedback()
            if q is not None:
                return np.asarray(q, dtype=float)
        if self._configuration is not None:
            return np.array(
                [self._configuration.q[idx] for idx in arm_qpos_indices],
                dtype=float,
            )
        return None

    def _wp_refresh_list(self):
        self._wp_list.clear()
        for i, wp in enumerate(self._waypoints):
            q_deg = [np.degrees(v) for v in wp["q"]]
            q_str = " ".join(f"{d:+6.1f}" for d in q_deg)
            self._wp_list.addItem(f"[{i+1:02d}] {wp['name']}   {q_str}")
        self._wp_status.setText(f"示教点: {len(self._waypoints)}")

    def _on_wp_capture(self):
        q = self._wp_current_joint_angles()
        if q is None:
            QMessageBox.warning(
                self, "记录失败",
                "无法读取当前关节角度（请先启动视图或连接实机）。",
            )
            return
        name = f"P{len(self._waypoints)+1:02d}"
        self._waypoints.append({
            "name": name,
            "q": [float(v) for v in q],
            "captured_at": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        })
        self._wp_refresh_list()
        self._wp_list.setCurrentRow(len(self._waypoints) - 1)
        self._status_bar.showMessage(f"已记录示教点 {name}")

    def _on_wp_insert(self, after: bool = True):
        """在选中位置的前/后插入当前关节角作为新示教点。"""
        q = self._wp_current_joint_angles()
        if q is None:
            QMessageBox.warning(
                self, "插入失败",
                "无法读取当前关节角度（请先启动视图或连接实机）。",
            )
            return

        row = self._wp_list.currentRow()
        if row < 0 or not self._waypoints:
            # 无选中/空列表：直接追加到末尾
            insert_idx = len(self._waypoints)
        else:
            insert_idx = row + 1 if after else row

        # 名称统一为 P{插入后的序号:02d}；后续再重编所有点保持顺序
        new_point = {
            "name": "",  # 稍后统一重命名
            "q": [float(v) for v in q],
            "captured_at": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        }
        self._waypoints.insert(insert_idx, new_point)
        self._wp_renumber()
        self._wp_refresh_list()
        self._wp_list.setCurrentRow(insert_idx)
        self._status_bar.showMessage(
            f"已在第 {insert_idx + 1} 位插入示教点"
        )

    def _wp_renumber(self):
        """按当前列表顺序重新生成默认名称 P01/P02/...（若之前是默认命名）。"""
        for i, wp in enumerate(self._waypoints):
            name = wp.get("name", "") or ""
            # 仅重编默认命名（P?? 形式或空名），保留用户自定义名
            if (not name) or (
                len(name) == 3 and name[0] == "P" and name[1:].isdigit()
            ):
                wp["name"] = f"P{i+1:02d}"

    def _on_wp_rows_moved(self, _parent, start: int, end: int,
                          _dest, dest_row: int):
        """列表拖拽结束后，把内部 _waypoints 顺序同步过来并重编号。"""
        # Qt 的 rowsMoved：start..end 是原区间，dest_row 是目标位置（在移除前的索引空间）
        if start < 0 or end < start or start >= len(self._waypoints):
            return
        moved = self._waypoints[start:end + 1]
        del self._waypoints[start:end + 1]
        # 移除后目标位置需要修正
        insert_at = dest_row if dest_row <= start else dest_row - (end - start + 1)
        insert_at = max(0, min(insert_at, len(self._waypoints)))
        self._waypoints[insert_at:insert_at] = moved
        self._wp_renumber()
        self._wp_refresh_list()
        self._wp_list.setCurrentRow(insert_at)
        self._status_bar.showMessage("已调整示教点顺序")

    def _on_wp_delete(self):
        row = self._wp_list.currentRow()
        if row < 0 or row >= len(self._waypoints):
            return
        removed = self._waypoints.pop(row)
        self._wp_refresh_list()
        self._status_bar.showMessage(f"已删除示教点 {removed['name']}")

    def _on_wp_clear(self):
        if not self._waypoints:
            return
        reply = QMessageBox.question(
            self, "清空示教点",
            f"确定清空全部 {len(self._waypoints)} 个示教点吗？",
        )
        if reply != QMessageBox.StandardButton.Yes:
            return
        self._waypoints.clear()
        self._wp_refresh_list()
        self._status_bar.showMessage("已清空示教点")

    def _on_wp_export(self):
        if not self._waypoints:
            QMessageBox.information(self, "导出", "当前没有示教点可导出。")
            return
        default = self._teach_traj_dir / (
            f"waypoints_{datetime.now():%Y%m%d_%H%M%S}.json"
        )
        fname, _ = QFileDialog.getSaveFileName(
            self, "导出示教点", str(default), "JSON (*.json)"
        )
        if not fname:
            return
        data_out = {
            "type": "waypoints",
            "created_at": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "num_joints": int(N_ARM),
            "points": self._waypoints,
        }
        try:
            with open(fname, "w", encoding="utf-8") as fh:
                json.dump(data_out, fh, indent=2, ensure_ascii=False)
        except Exception as exc:
            QMessageBox.warning(self, "导出失败", str(exc))
            return
        self._status_bar.showMessage(
            f"已导出 {len(self._waypoints)} 个示教点到 {Path(fname).name}"
        )

    def _on_wp_import(self):
        fname, _ = QFileDialog.getOpenFileName(
            self, "导入示教点", str(self._teach_traj_dir), "JSON (*.json)"
        )
        if not fname:
            return
        try:
            with open(fname, "r", encoding="utf-8") as fh:
                d = json.load(fh)
        except Exception as exc:
            QMessageBox.warning(self, "导入失败", str(exc))
            return
        points = d.get("points") or []
        if not isinstance(points, list) or not points:
            QMessageBox.warning(self, "导入失败", "文件中未找到示教点。")
            return
        loaded: list[dict] = []
        for i, item in enumerate(points):
            if not isinstance(item, dict) or "q" not in item:
                continue
            q = item.get("q")
            if not isinstance(q, list) or len(q) != N_ARM:
                continue
            try:
                q_f = [float(v) for v in q]
            except Exception:
                continue
            loaded.append({
                "name": str(item.get("name") or f"P{i+1:02d}"),
                "q": q_f,
                "captured_at": str(item.get("captured_at") or ""),
            })
        if not loaded:
            QMessageBox.warning(
                self, "导入失败",
                f"未解析到有效示教点（要求 q 为长度 {N_ARM} 的浮点数组）。",
            )
            return
        # 若已有点，询问是替换还是追加
        if self._waypoints:
            reply = QMessageBox.question(
                self, "导入示教点",
                f"已有 {len(self._waypoints)} 个示教点。\n"
                f"「是」追加 {len(loaded)} 个；「否」用导入数据替换全部。",
                QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No
                | QMessageBox.StandardButton.Cancel,
                QMessageBox.StandardButton.Yes,
            )
            if reply == QMessageBox.StandardButton.Cancel:
                return
            if reply == QMessageBox.StandardButton.No:
                self._waypoints = []
        self._waypoints.extend(loaded)
        self._wp_refresh_list()
        self._status_bar.showMessage(
            f"已导入 {len(loaded)} 个示教点（当前共 {len(self._waypoints)}）"
        )

    def _on_wp_goto(self):
        if self._waypoint_move_timer is not None:
            QMessageBox.information(self, "移动", "已有移动任务在进行。")
            return
        if self._teach_active:
            QMessageBox.warning(
                self, "无法移动", "请先退出示教模式再移动到示教点。"
            )
            return
        if self._viewer is None or not self._viewer.is_running():
            QMessageBox.warning(self, "无法移动", "请先启动视图。")
            return
        if self._configuration is None:
            return
        row = self._wp_list.currentRow()
        if row < 0 or row >= len(self._waypoints):
            QMessageBox.information(self, "移动", "请先选中一个示教点。")
            return

        target = np.asarray(self._waypoints[row]["q"], dtype=float)
        # 起点：实机反馈优先
        start = self._wp_current_joint_angles()
        if start is None:
            QMessageBox.warning(self, "移动", "无法获取当前关节角度。")
            return
        # 同步仿真起点
        with lock:
            for i, idx in enumerate(arm_qpos_indices):
                data.qpos[idx] = start[i]
                self._configuration.data.qpos[idx] = start[i]
            fix_base_position(model, data)
            fix_base_position(self._configuration.model, self._configuration.data)
            mujoco.mj_forward(model, data)
            mujoco.mj_forward(self._configuration.model, self._configuration.data)

        duration = float(self._wp_duration_spin.value())
        tick_ms = 20
        self._waypoint_move_total = max(1, int(duration * 1000 / tick_ms))
        self._waypoint_move_step = 0
        self._waypoint_move_start_q = start.copy()
        self._waypoint_move_target_q = target.copy()
        self._waypoint_move_prev_q = start.copy()
        self._waypoint_move_timer = QTimer(self)
        self._waypoint_move_timer.timeout.connect(self._wp_move_tick)
        self._waypoint_move_timer.start(tick_ms)
        self._status_bar.showMessage(
            f"移动到 {self._waypoints[row]['name']}  用时 {duration:.1f}s"
        )

    def _wp_move_tick(self):
        if self._waypoint_move_timer is None:
            return
        self._waypoint_move_step += 1
        t = min(self._waypoint_move_step / self._waypoint_move_total, 1.0)
        t_smooth = t * t * (3.0 - 2.0 * t)
        q_interp = self._waypoint_move_start_q + (
            self._waypoint_move_target_q - self._waypoint_move_start_q
        ) * t_smooth

        with lock:
            for i, idx in enumerate(arm_qpos_indices):
                self._configuration.data.qpos[idx] = q_interp[i]
                data.qpos[idx] = q_interp[i]
            fix_base_position(model, data)
            fix_base_position(self._configuration.model, self._configuration.data)
            mujoco.mj_forward(model, data)
            mujoco.mj_forward(self._configuration.model, self._configuration.data)

        if self._mode == "real" and self._real_manager.is_connected:
            try:
                self._real_manager.send_joint_command(
                    q_interp, start_q=self._waypoint_move_prev_q,
                )
            except Exception as exc:
                self._status_bar.showMessage(f"实机指令发送失败: {exc}")
                self._waypoint_move_timer.stop()
                self._waypoint_move_timer.deleteLater()
                self._waypoint_move_timer = None
                return
        self._waypoint_move_prev_q = q_interp.copy()

        if t >= 1.0:
            self._waypoint_move_timer.stop()
            self._waypoint_move_timer.deleteLater()
            self._waypoint_move_timer = None
            self._status_bar.showMessage("已到达示教点")
            self._refresh_position()

    def _init_mink(self):
        self._configuration = mink.Configuration(model)
        self._configuration.model.jnt_range[gripper_joint_id, :] = np.array(
            [GRIPPER_JOINT_MIN, GRIPPER_JOINT_MAX]
        )
        with lock:
            qpos_init = data.qpos.copy()
        qpos_init[gripper_qpos_index] = np.clip(
            qpos_init[gripper_qpos_index], GRIPPER_JOINT_MIN, GRIPPER_JOINT_MAX
        )
        self._configuration.update(qpos_init)

        self._ee_task = mink.FrameTask(
            frame_name=EE_SITE_NAME,
            frame_type="site",
            position_cost=1.0,
            orientation_cost=1.0,
        )
        self._ik_limits = [
            mink.ConfigurationLimit(
                self._configuration.model,
                gain=0.95,
                min_distance_from_limits=1e-4,
            )
        ]

    # ------------------------------------------------------------------
    # IK 耦合验证：求解后检查未命令轴是否漂移
    # ------------------------------------------------------------------

    def _current_arm_q_from_config(self) -> np.ndarray:
        """Return the arm joints from the active mink configuration."""
        if self._configuration is None:
            return np.zeros(N_ARM, dtype=float)
        return np.array(
            [self._configuration.q[idx] for idx in arm_qpos_indices],
            dtype=float,
        )

    def _arm_limit_violation_reason(
        self,
        q_arm: np.ndarray,
        *,
        tol: float = 1e-6,
    ) -> str | None:
        q = np.asarray(q_arm, dtype=float)
        below = np.where(q < ARM_JOINT_LIMITS_LOWER - tol)[0]
        above = np.where(q > ARM_JOINT_LIMITS_UPPER + tol)[0]
        if below.size == 0 and above.size == 0:
            return None

        parts = []
        for idx in below:
            parts.append(
                f"L{idx + 1}={np.degrees(q[idx]):+.2f}° "
                f"< {np.degrees(ARM_JOINT_LIMITS_LOWER[idx]):+.2f}°"
            )
        for idx in above:
            parts.append(
                f"L{idx + 1}={np.degrees(q[idx]):+.2f}° "
                f"> {np.degrees(ARM_JOINT_LIMITS_UPPER[idx]):+.2f}°"
            )
        return "，".join(parts)

    def _calc_drift_tolerances(self, pos_step: float, rot_step: float):
        """根据当前步长计算耦合检测容差。返回 (pos_tol_m, rot_tol_rad)。"""
        pos_tol = np.clip(pos_step * 0.1, 0.0005, 0.003)   # step 的 10%，clamp [0.5mm, 3mm]
        rot_tol = np.clip(rot_step * 0.1, np.radians(0.5), np.radians(3.0))
        return pos_tol, rot_tol

    def _validate_ik_result(
        self,
        target_pos: np.ndarray | None,
        target_rot: np.ndarray | None,
        pos_tol: float,
        rot_tol: float,
        check_axes_pos: tuple[int, ...] | None = None,
        check_axes_rot: tuple[int, ...] | None = None,
        caller: str = "",
    ) -> tuple[bool, str]:
        """验证 IK 结果是否满足耦合约束。

        check_axes_pos: 需要检查的位置轴索引（0=x,1=y,2=z），None=全查
        check_axes_rot: 需要检查的旋转轴索引（0=x,1=y,2=z），None=全查

        返回 (ok, reason)。ok=False 时 reason 说明哪个轴超标。
        每次调用都会写入日志，包含目标值、实际值、各轴误差和判定结果。
        """
        if self._configuration is None:
            return False, "mink 未初始化"

        actual_pos = self._configuration.data.site_xpos[ee_site_id].copy()
        actual_rot = self._configuration.data.site_xmat[ee_site_id].reshape(3, 3).copy()

        log_lines = [f"[耦合验证] {caller}"]

        # --- 位置漂移 ---
        if target_pos is not None:
            pos_err = np.abs(actual_pos - target_pos)
            pos_axes = check_axes_pos if check_axes_pos is not None else (0, 1, 2)
            axis_name = {0: "X", 1: "Y", 2: "Z"}
            log_lines.append(
                f"  位置目标: [{target_pos[0]*1000:.2f}, {target_pos[1]*1000:.2f}, {target_pos[2]*1000:.2f}]mm"
            )
            log_lines.append(
                f"  位置实际: [{actual_pos[0]*1000:.2f}, {actual_pos[1]*1000:.2f}, {actual_pos[2]*1000:.2f}]mm"
            )
            log_lines.append(
                f"  位置误差: [{pos_err[0]*1000:.3f}, {pos_err[1]*1000:.3f}, {pos_err[2]*1000:.3f}]mm  "
                f"容差: {pos_tol*1000:.3f}mm"
            )
            log_lines.append(
                f"  检查轴(位置): {[axis_name[a] for a in pos_axes]}"
            )
            for a in pos_axes:
                if pos_err[a] > pos_tol:
                    reason = (
                        f"位置 {axis_name[a]} 轴漂移 {pos_err[a]*1000:.2f}mm "
                        f"> 容差 {pos_tol*1000:.2f}mm"
                    )
                    log_lines.append(f"  ❌ 拒绝: {reason}")
                    self._pose_logger.info("\n".join(log_lines))
                    return False, reason

        # --- 姿态漂移（逐轴检查 R/P/Y） ---
        if target_rot is not None:
            drift_mat = target_rot.T @ actual_rot
            # 从旋转误差矩阵的反对称部分提取各轴角度误差
            e = (drift_mat - drift_mat.T) / 2.0
            rot_errors = [abs(e[2, 1]), abs(e[0, 2]), abs(e[1, 0])]
            rot_axes = check_axes_rot if check_axes_rot is not None else (0, 1, 2)
            axis_name = {0: "Roll", 1: "Pitch", 2: "Yaw"}
            log_lines.append(
                f"  姿态误差: R={np.degrees(rot_errors[0]):.3f}° "
                f"P={np.degrees(rot_errors[1]):.3f}° "
                f"Y={np.degrees(rot_errors[2]):.3f}°  "
                f"容差: {np.degrees(rot_tol):.3f}°"
            )
            log_lines.append(
                f"  检查轴(姿态): {[axis_name[a] for a in rot_axes]}"
            )
            for a in rot_axes:
                if rot_errors[a] > rot_tol:
                    reason = (
                        f"姿态 {axis_name[a]} 漂移 {np.degrees(rot_errors[a]):.2f}° "
                        f"> 容差 {np.degrees(rot_tol):.2f}°"
                    )
                    log_lines.append(f"  ❌ 拒绝: {reason}")
                    self._pose_logger.info("\n".join(log_lines))
                    return False, reason

        log_lines.append("  ✅ 通过")
        self._pose_logger.info("\n".join(log_lines))
        return True, ""

    def _calc_ik_steps(self, error_norm: float) -> int:
        """根据笛卡尔空间误差大小动态计算 IK 迭代次数。

        规则：至少 5 次，每 2mm 误差增加 1 次迭代。
        小步长不浪费算力，大步长保证收敛。
        """
        return max(5, int(np.ceil(error_norm / 0.002)))

    @staticmethod
    def _rotation_error_norm(target_rot: np.ndarray, actual_rot: np.ndarray) -> float:
        delta = target_rot @ actual_rot.T
        return float(np.linalg.norm(Rotation.from_matrix(delta).as_rotvec()))

    def _restore_ik_snapshot(self, q_snapshot: np.ndarray) -> None:
        if self._configuration is not None:
            self._configuration.update(q_snapshot)

    def _solve_velocity_ik(
        self,
        target_pos: np.ndarray | None,
        target_rot: np.ndarray | None = None,
        n_steps: int = 15,
        caller: str = "",
    ) -> bool:
        if self._configuration is None or self._ee_task is None:
            return False

        tasks = [self._ee_task]
        current_pos = self._configuration.data.site_xpos[ee_site_id].copy()
        current_rot = self._configuration.data.site_xmat[ee_site_id].reshape(3, 3).copy()
        current_quat = Rotation.from_matrix(current_rot).as_quat()
        entry_q = self._configuration.q.copy()

        resolved_target_pos = current_pos.copy()
        if target_pos is not None:
            resolved_target_pos = np.asarray(target_pos, dtype=float).copy()
        resolved_target_rot = current_rot.copy()
        if target_rot is not None:
            resolved_target_rot = np.asarray(target_rot, dtype=float).copy()

        if target_pos is not None and target_rot is not None:
            mode = "pos+rot"
            self._ee_task.set_position_cost(1.0)
            self._ee_task.set_orientation_cost(1.0)
        elif target_rot is not None:
            mode = "rot_only"
            self._ee_task.set_position_cost(self._ik_hold_cost)
            self._ee_task.set_orientation_cost(1.0)
        else:
            mode = "pos_only"
            self._ee_task.set_position_cost(1.0)
            self._ee_task.set_orientation_cost(self._ik_hold_cost)
        T = np.eye(4)
        T[:3, :3] = resolved_target_rot
        T[:3, 3] = resolved_target_pos
        target_se3 = mink.SE3.from_matrix(T)
        self._ee_task.set_target(target_se3)

        start_violation = self._arm_limit_violation_reason(
            self._current_arm_q_from_config()
        )
        if start_violation is not None:
            self._pose_logger.info(
                f"[IK限位] {caller} 起点越界，拒绝求解: {start_violation}"
            )
            return False

        last_pos_err = float("inf")
        last_rot_err = float("inf")
        for i in range(n_steps):
            try:
                vel = mink.solve_ik(
                    self._configuration,
                    tasks,
                    self._dt,
                    self._solver,
                    safety_break=True,
                    limits=self._ik_limits,
                )
            except Exception as exc:
                self._restore_ik_snapshot(entry_q)
                self._pose_logger.info(
                    f"[IK限位] {caller} 求解失败，可能目标超出工作空间或关节限位: {exc}"
                )
                return False

            gripper_vel_idx = gripper_dof_index - min(arm_dof_indices)
            predicted = (
                self._configuration.q[gripper_qpos_index]
                + vel[gripper_vel_idx] * self._dt
            )
            if predicted > GRIPPER_JOINT_MAX:
                max_vel = (GRIPPER_JOINT_MAX - self._configuration.q[gripper_qpos_index]) / self._dt
                vel[gripper_vel_idx] = min(vel[gripper_vel_idx], max_vel)
            elif predicted < GRIPPER_JOINT_MIN:
                min_vel = (GRIPPER_JOINT_MIN - self._configuration.q[gripper_qpos_index]) / self._dt
                vel[gripper_vel_idx] = max(vel[gripper_vel_idx], min_vel)

            self._configuration.integrate_inplace(vel, self._dt)
            self._configuration.data.qpos[gripper_qpos_index] = np.clip(
                self._configuration.data.qpos[gripper_qpos_index],
                GRIPPER_JOINT_MIN, GRIPPER_JOINT_MAX,
            )
            self._configuration.update()

            violation = self._arm_limit_violation_reason(
                self._current_arm_q_from_config()
            )
            if violation is not None:
                self._restore_ik_snapshot(entry_q)
                self._pose_logger.info(
                    f"[IK限位] {caller} 第 {i + 1}/{n_steps} 步越界，已回滚: {violation}"
                )
                return False

            actual_pos = self._configuration.data.site_xpos[ee_site_id]
            actual_rot = self._configuration.data.site_xmat[ee_site_id].reshape(3, 3)
            last_pos_err = float(np.linalg.norm(actual_pos - resolved_target_pos))
            last_rot_err = self._rotation_error_norm(resolved_target_rot, actual_rot)
            if last_pos_err < self._ik_conv_tol and last_rot_err < self._ik_rot_conv_tol:
                self._log_pose(
                    mode, caller, current_pos, current_quat,
                    resolved_target_pos, resolved_target_rot, i + 1, n_steps,
                )
                return True

        self._log_pose(
            mode, caller, current_pos, current_quat,
            resolved_target_pos, resolved_target_rot, n_steps, n_steps,
        )
        if last_pos_err < self._ik_conv_tol and last_rot_err < self._ik_rot_conv_tol:
            return True

        self._restore_ik_snapshot(entry_q)
        self._pose_logger.info(
            f"[IK收敛] {caller} 未达到完整6D目标，已回滚: "
            f"pos_err={last_pos_err * 1000:.3f}mm, "
            f"rot_err={np.degrees(last_rot_err):.3f}°"
        )
        return False

    def _log_pose(
        self,
        mode: str,
        caller: str,
        src_pos: np.ndarray,
        src_quat: np.ndarray,
        target_pos: np.ndarray | None,
        target_rot: np.ndarray | None,
        steps_used: int,
        steps_max: int,
    ):
        """记录一次位姿控制的完整信息到日志"""
        actual_pos = self._configuration.data.site_xpos[ee_site_id].copy()
        actual_rot = self._configuration.data.site_xmat[ee_site_id].reshape(3, 3).copy()
        actual_quat = Rotation.from_matrix(actual_rot).as_quat()
        q = [self._configuration.q[idx] for idx in arm_qpos_indices]

        pos_err = actual_pos - (target_pos if target_pos is not None else src_pos)
        pos_err_mm = pos_err * 1000
        pos_drift_mm = (actual_pos - src_pos) * 1000

        if target_rot is not None:
            tgt_quat = Rotation.from_matrix(target_rot).as_quat()
            # 确保四元数在同一半球（避免符号歧义）
            if np.dot(actual_quat, tgt_quat) < 0:
                tgt_quat = -tgt_quat
            # 旋转误差（角度，度）
            dot = np.clip(np.abs(np.dot(actual_quat, tgt_quat)), 0, 1)
            rot_err_deg = np.degrees(2 * np.arccos(dot))
        else:
            tgt_quat = None
            if np.dot(actual_quat, src_quat) < 0:
                src_quat = -src_quat
            dot = np.clip(np.abs(np.dot(actual_quat, src_quat)), 0, 1)
            rot_err_deg = np.degrees(2 * np.arccos(dot))

        lines = [
            f"[{caller}] mode={mode}  steps={steps_used}/{steps_max}",
            f"  src_pos:   {np.array2string(src_pos, precision=4, suppress_small=True)}  "
            f"src_quat: {np.array2string(src_quat, precision=4, suppress_small=True)}",
        ]
        if target_pos is not None:
            lines.append(
                f"  tgt_pos:   {np.array2string(target_pos, precision=4, suppress_small=True)}  "
                f"pos_err: [{pos_err_mm[0]:+.2f}, {pos_err_mm[1]:+.2f}, {pos_err_mm[2]:+.2f}] mm"
            )
        if tgt_quat is not None:
            lines.append(
                f"  tgt_quat:  {np.array2string(tgt_quat, precision=4, suppress_small=True)}  "
                f"rot_err: {rot_err_deg:.2f} deg"
            )
        lines += [
            f"  actual_pos: {np.array2string(actual_pos, precision=4, suppress_small=True)}  "
            f"pos_drift: [{pos_drift_mm[0]:+.2f}, {pos_drift_mm[1]:+.2f}, {pos_drift_mm[2]:+.2f}] mm",
            f"  actual_quat: {np.array2string(actual_quat, precision=4, suppress_small=True)}  "
            f"rot_err_vs_src: {rot_err_deg:.2f} deg",
            f"  joints(rad): {np.array2string(np.array(q), precision=4, suppress_small=True)}",
        ]
        self._pose_logger.info("\n".join(lines))

    def _sync_config_to_global(self):
        if self._configuration is None:
            return
        with lock:
            for i, idx in enumerate(arm_qpos_indices):
                data.qpos[idx] = self._configuration.q[idx]
            data.qpos[gripper_qpos_index] = np.clip(
                self._configuration.q[gripper_qpos_index],
                GRIPPER_JOINT_MIN, GRIPPER_JOINT_MAX,
            )
            fix_base_position(model, data)
            mujoco.mj_forward(model, data)

    @Slot(int)
    def _on_mode_changed(self, index: int):
        self._mode = "sim" if index == 0 else "real"
        self._refresh_real_connection_ui()

    @Slot(float)
    def _on_step_changed(self, value: float):
        self._step_size = value / 1000.0

    @Slot(float)
    def _on_rot_changed(self, value: float):
        self._rot_step = np.radians(value)

    @Slot()
    def _on_start_viewer(self):
        if self._viewer is not None:
            self._status_bar.showMessage("视图已在运行中")
            return
        if self._mode == "real":
            if not self._real_manager.is_connected:
                self._status_bar.showMessage("请先连接实机")
                QMessageBox.warning(
                    self,
                    "未连接实机",
                    "请先点击“连接实机”，确认连接成功后再启动视图。",
                )
                return
            self._start_real_mode()
        else:
            self._start_sim_mode()

    def _start_sim_mode(self):
        self._status_bar.showMessage("正在启动 MuJoCo 视图 ...")
        self._start_btn.setEnabled(False)
        try:
            self._init_mink()
            cfg_data = self._configuration.data
            self._viewer_ctx = mujoco.viewer.launch_passive(
                model=self._configuration.model, data=cfg_data,
                show_left_ui=False, show_right_ui=False,
            )
            self._viewer = self._viewer_ctx.__enter__()
            mujoco.mjv_defaultFreeCamera(self._configuration.model, self._viewer.cam)
            self._render_timer.start(20)
            self._set_buttons_enabled(True)
            self._stop_btn.setEnabled(True)
            self._status_bar.showMessage("就绪 - 仿真模式 (mink 速度逆运动学)")
        except Exception as e:
            self._start_btn.setEnabled(True)
            self._status_bar.showMessage(f"启动失败: {e}")
            QMessageBox.critical(self, "启动失败", str(e))

    def _start_real_mode(self):
        self._status_bar.showMessage("正在启动实机视图 ...")
        self._start_btn.setEnabled(False)

        seed = self._real_manager.get_joint_feedback()
        if seed is not None:
            with lock:
                for i, idx in enumerate(arm_qpos_indices):
                    data.qpos[idx] = seed[i]
                fix_base_position(model, data)
                mujoco.mj_forward(model, data)

        try:
            self._init_mink()
            cfg_data = self._configuration.data
            self._viewer_ctx = mujoco.viewer.launch_passive(
                model=self._configuration.model, data=cfg_data,
                show_left_ui=False, show_right_ui=False,
            )
            self._viewer = self._viewer_ctx.__enter__()
            mujoco.mjv_defaultFreeCamera(self._configuration.model, self._viewer.cam)
            self._render_timer.start(20)
            self._set_buttons_enabled(True)
            self._stop_btn.setEnabled(True)
            self._status_bar.showMessage("就绪 - 实机模式 (mink 速度逆运动学)")
        except Exception as e:
            self._start_btn.setEnabled(True)
            self._status_bar.showMessage(f"视图启动失败: {e}")
            QMessageBox.critical(self, "视图启动失败", str(e))

    def _toggle_real_robot(self):
        if self._real_manager.is_connected:
            self._disconnect_real_robot()
        else:
            self._connect_real_robot()

    def _connect_real_robot(self):
        can_name = self._can_combo.currentText().strip() or "can0"
        kp = self._kp_spin.value()
        kd = self._kd_spin.value()
        max_vel_rad_s = float(np.radians(self._vmax_spin.value()))
        self._set_real_connection_pending(f"正在连接 {can_name} ...")
        try:
            self._real_manager.connect(
                can_name=can_name,
                kp=kp,
                kd=kd,
                max_vel_rad_s=max_vel_rad_s,
            )
        except Exception as e:
            self._real_manager.disconnect()
            self._set_real_connection_state(False, f"连接失败: {e}")
            self._status_bar.showMessage(f"实机连接失败: {e}")
            QMessageBox.critical(self, "连接失败", str(e))
            return

        self._set_real_connection_state(True, f"已连接: {can_name}")
        self._status_bar.showMessage(f"实机已连接并使能: {can_name}")

        # 检测当前关节是否处于零位；若不在零位则询问用户是否平滑回零
        self._prompt_home_after_connect()

    def _prompt_home_after_connect(self):
        """连接实机后：若关节不在零位，弹窗询问是否先平滑回零。"""
        feedback = self._real_manager.get_joint_feedback()
        if feedback is None:
            self._status_bar.showMessage(
                "实机已连接，但读不到关节反馈；请手动检查后再运动"
            )
            return

        max_dev = float(np.max(np.abs(feedback)))
        if max_dev <= HOME_THRESHOLD_RAD:
            self._status_bar.showMessage("实机已连接，当前处于零位")
            return

        deg_dev = np.degrees(max_dev)
        joint_str = ", ".join(f"{np.degrees(v):+.1f}°" for v in feedback)
        reply = QMessageBox.question(
            self,
            "机器人当前不在零位",
            f"检测到机器人未处于零位（最大偏差 {deg_dev:.1f}°）。\n"
            f"当前关节角度: [{joint_str}]\n\n"
            "是否现在平滑运动到零位？\n\n"
            "· 选择「是」：机器人将从当前位置平滑归零\n"
            "· 选择「否」：保持当前姿态，可稍后点击「回零」按钮",
            QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No,
            QMessageBox.StandardButton.No,
        )
        if reply == QMessageBox.StandardButton.Yes:
            self._home_real_robot_direct(feedback)
        else:
            self._status_bar.showMessage(
                f"已保留当前姿态（偏差 {deg_dev:.1f}°），请注意安全"
            )

    def _home_real_robot_direct(self, start_q: np.ndarray):
        """直接向实机下发一条从当前位姿到零位的平滑回零指令（不依赖 viewer/mink）。"""
        try:
            target = np.zeros(N_ARM)
            self._real_manager.send_joint_command(target, start_q=start_q)
            self._status_bar.showMessage("已下发回零指令，机器人正在平滑归零...")
        except Exception as e:
            self._status_bar.showMessage(f"回零指令下发失败: {e}")
            QMessageBox.warning(self, "回零失败", str(e))

    def _disconnect_real_robot(self):
        if self._mode == "real" and self._viewer is not None:
            self._on_stop()
        self._real_connect_btn.setEnabled(False)
        self._real_status_label.setText("正在断开 ...")
        self._real_status_label.setStyleSheet("color: #e67e22; font-weight: bold;")
        try:
            self._real_manager.disconnect()
        finally:
            self._set_real_connection_state(False, "未连接")
            if self._mode == "real":
                self._status_bar.showMessage("实机已断开")

    def _set_real_connection_pending(self, text: str):
        self._real_connect_btn.setEnabled(False)
        self._real_connect_btn.setText("连接中...")
        self._real_status_label.setText(text)
        self._real_status_label.setStyleSheet("color: #e67e22; font-weight: bold;")
        self._set_real_params_enabled(False)

    def _set_real_connection_state(self, connected: bool, text: str):
        self._real_status_text = text
        self._real_connect_btn.setEnabled(True)
        self._real_connect_btn.setText("断开实机" if connected else "连接实机")
        if connected:
            self._real_connect_btn.setStyleSheet(
                "QPushButton { background-color: #d35400; color: white; font-weight: bold; }"
            )
            self._real_status_label.setStyleSheet("color: #2ecc71; font-weight: bold;")
        else:
            self._real_connect_btn.setStyleSheet(
                "QPushButton { background-color: #27ae60; color: white; font-weight: bold; }"
            )
            self._real_status_label.setStyleSheet("color: #888; font-weight: bold;")
        self._real_status_label.setText(text)
        self._set_real_params_enabled(not connected)
        self._real_group.setVisible(self._mode == "real" or connected)

    def _set_real_params_enabled(self, enabled: bool):
        self._can_combo.setEnabled(enabled)
        self._kp_spin.setEnabled(enabled)
        self._kd_spin.setEnabled(enabled)

    def _refresh_real_connection_ui(self):
        if self._real_manager.is_connected:
            self._set_real_connection_state(True, self._real_status_text)
        else:
            self._set_real_connection_state(False, self._real_status_text)

    def _render_tick(self):
        if self._viewer is None or not self._viewer.is_running():
            self._render_timer.stop()
            self._viewer = None
            self._viewer_ctx = None
            self._configuration = None
            self._ee_task = None
            self._traj_positions.clear()
            self._set_buttons_enabled(False)
            self._start_btn.setEnabled(True)
            self._stop_btn.setEnabled(False)
            if self._mode == "real" and self._real_manager.is_connected:
                self._status_bar.showMessage("视图已关闭，实机仍保持连接")
            else:
                self._status_bar.showMessage("视图已关闭")
            return
        try:
            cfg_model = self._configuration.model
            cfg_data = self._configuration.data
            with lock:
                mujoco.mj_camlight(cfg_model, cfg_data)
            self._update_trajectory()
            self._viewer.sync()
        except Exception:
            pass

    # ---- 轨迹可视化 ----

    def _on_traj_toggled(self, checked: bool):
        self._traj_enabled = checked
        if not checked:
            self._traj_positions.clear()
            if self._viewer is not None:
                try:
                    self._viewer.user_scn.ngeom = 0
                except Exception:
                    pass

    def _on_imu_toggled(self, checked: bool):
        self._imu_mode_active = checked

    def _on_vmax_changed(self, value_deg_s: float):
        """最大速度 SpinBox 变化时调用；若实机已连接则立即下发新的上限。"""
        v_rad_s = float(np.radians(value_deg_s))
        if self._real_manager.is_connected:
            self._real_manager.set_max_velocity(v_rad_s)
            self._status_bar.showMessage(
                f"实机最大速度已更新: {value_deg_s:.1f} deg/s "
                f"({v_rad_s:.2f} rad/s)"
            )
        else:
            self._status_bar.showMessage(
                f"最大速度已设置为 {value_deg_s:.1f} deg/s，连接实机后生效"
            )

    def _update_trajectory(self):
        """记录末端位置并在 user_scn 中绘制轨迹连线。"""
        if not self._traj_enabled or self._viewer is None or self._configuration is None:
            return

        pos = self._configuration.data.site_xpos[ee_site_id].copy()

        # 仅当移动超过最小距离时才记录新点
        if (not self._traj_positions
                or np.linalg.norm(pos - self._traj_positions[-1]) >= self._traj_min_dist):
            self._traj_positions.append(pos)
            if len(self._traj_positions) > self._traj_max_points:
                self._traj_positions.pop(0)

        # 每帧全量重绘 user_scn
        scn = self._viewer.user_scn
        scn.ngeom = 0
        n = len(self._traj_positions)
        if n < 2:
            return

        radius = 0.003
        rgba = np.array([0.0, 1.0, 0.0, 0.8], dtype=np.float32)
        for i in range(n - 1):
            if scn.ngeom >= scn.maxgeom:
                break
            p0 = self._traj_positions[i].astype(np.float64)
            p1 = self._traj_positions[i + 1].astype(np.float64)
            if np.allclose(p0, p1):
                continue
            # 初始化 geom，再用 mjv_connector 设置位置和朝向
            mujoco.mjv_initGeom(
                scn.geoms[scn.ngeom],
                mujoco.mjtGeom.mjGEOM_CAPSULE,
                np.zeros(3),
                np.zeros(3),
                np.zeros(9),
                rgba,
            )
            mujoco.mjv_connector(
                scn.geoms[scn.ngeom],
                mujoco.mjtGeom.mjGEOM_CAPSULE,
                radius,
                p0,
                p1,
            )
            scn.ngeom += 1

    def _toggle_gamepad(self):
        if self._joystick is not None and self._joystick.connected:
            self._disconnect_gamepad()
        else:
            self._connect_gamepad()

    def _connect_gamepad(self):
        joycon = JoyConReader()
        if joycon.connect():
            self._joystick = joycon
            self._gp_connect_btn.setText("断开连接")
            has_imu = joycon._imu_dev is not None
            self._gp_status_label.setText(f"已连接: Joy-Con {'(+IMU)' if has_imu else '(仅摇杆)'}")
            self._gp_status_label.setStyleSheet("color: #2ecc71;")
            self._gp_active_check.setEnabled(True)
            self._imu_check.setEnabled(has_imu)
            self._imu_calib_btn.setEnabled(has_imu)
            if not has_imu:
                self._imu_check.setChecked(False)
                self._imu_status.setText("IMU: 不可用")
            self._status_bar.showMessage("Joy-Con 已连接")
            return
        for dev in ["/dev/input/js0", "/dev/input/js1", "/dev/input/js2"]:
            joy = SimpleJoystick(dev)
            if joy.connect():
                self._joystick = joy
                self._gp_connect_btn.setText("断开连接")
                self._gp_status_label.setText(f"已连接: {dev}")
                self._gp_status_label.setStyleSheet("color: #2ecc71;")
                self._gp_active_check.setEnabled(True)
                self._imu_check.setEnabled(False)
                self._imu_calib_btn.setEnabled(False)
                self._status_bar.showMessage(f"手柄已连接: {dev}")
                return
        QMessageBox.warning(self, "连接失败", "未找到手柄\n请连接手柄 (USB/蓝牙)")

    def _disconnect_gamepad(self):
        self._gamepad_active = False
        self._gp_active_check.setChecked(False)
        self._gp_active_check.setEnabled(False)
        self._imu_check.setEnabled(False)
        self._imu_check.setChecked(False)
        self._imu_mode_active = False
        if self._joystick:
            self._joystick.disconnect()
            self._joystick = None
        self._gp_connect_btn.setText("连接手柄")
        self._gp_status_label.setText("未连接")
        self._gp_status_label.setStyleSheet("color: #888;")
        self._imu_status.setText("IMU: ---")
        self._imu_calib_btn.setEnabled(False)
        self._gamepad_timer.stop()

    def _calibrate_imu(self):
        if self._joystick is None or not hasattr(self._joystick, 'start_gyro_calibration'):
            return
        self._imu_calib_btn.setEnabled(False)
        self._imu_calib_btn.setText("采集中...")
        self._imu_status.setText("IMU: 请保持手柄静止...")
        self._imu_status.setStyleSheet("font-size: 11px; color: #e67e22;")
        self._joystick.start_gyro_calibration()
        # 延迟 1 秒后完成校准
        QTimer.singleShot(1000, self._finish_imu_calibration)

    def _finish_imu_calibration(self):
        if self._joystick is None:
            return
        ok = self._joystick.finish_gyro_calibration()
        self._imu_calib_btn.setText("校准 IMU")
        self._imu_calib_btn.setEnabled(True)
        if ok:
            bias = self._joystick._gyro_bias
            self._imu_status.setText(
                f"IMU: 已校准 (偏移 R={bias[0]:+.0f} P={bias[1]:+.0f} Y={bias[2]:+.0f})"
            )
            self._imu_status.setStyleSheet("font-size: 11px; color: #2ecc71;")
        else:
            self._imu_status.setText("IMU: 校准失败，请重试")
            self._imu_status.setStyleSheet("font-size: 11px; color: #e74c3c;")

    @Slot(bool)
    def _on_gamepad_toggled(self, checked: bool):
        self._gamepad_active = checked
        if checked:
            if self._viewer is None or not self._viewer.is_running():
                self._gp_active_check.setChecked(False)
                self._status_bar.showMessage("请先启动视图")
                return
            if self._configuration is None:
                self._gp_active_check.setChecked(False)
                return
            self._gamepad_timer.start(20)
            self._status_bar.showMessage("手柄控制已启用")
        else:
            self._gamepad_timer.stop()
            self._status_bar.showMessage("手柄控制已禁用")

    def _cycle_gamepad_speed(self):
        self._gamepad_speed_idx = (self._gamepad_speed_idx + 1) % len(self._gamepad_speed_levels)
        v = self._gamepad_speed_levels[self._gamepad_speed_idx]
        names = ["极慢", "慢速", "中速", "快速", "极快"]
        self._gp_speed_btn.setText(f"{names[self._gamepad_speed_idx]} ({v:.2f})")

    def _gamepad_tick(self):
        if (self._joystick is None or not self._joystick.connected
                or self._configuration is None or self._viewer is None):
            return
        if not self._viewer.is_running():
            self._disconnect_gamepad()
            return

        joy = self._joystick
        speed = self._gamepad_speed_levels[self._gamepad_speed_idx]

        def read_axis(idx):
            v = joy.axes[idx] if idx < len(joy.axes) else 0.0
            return v if abs(v) > 0.12 else 0.0

        lx = read_axis(0)
        ly = read_axis(1)

        # DEBUG: 打印原始轴值 (已禁用, 避免干扰终端)
        # if abs(joy.axes[0]) > 0.01 or abs(joy.axes[1]) > 0.01:
        #     print(f"[DEBUG] axes raw: X={joy.axes[0]:.4f} Y={joy.axes[1]:.4f} | filtered: lx={lx:.4f} ly={ly:.4f}")

        vx = -ly * speed
        vy = lx * speed

        # Z 轴：L=下, ZL=上
        z_down = len(joy.buttons) > 4 and joy.buttons[4]   # L
        z_up = len(joy.buttons) > 6 and joy.buttons[6]     # ZL
        vz = (z_up - z_down) * speed

        # 姿态控制：按键模式（默认）
        w_roll = 0.0
        w_pitch = 0.0
        w_yaw = 0.0

        # Roll: SL=右, SR=左
        if len(joy.buttons) > 5 and joy.buttons[5]:   # SL
            w_roll = speed * 2.0
        if len(joy.buttons) > 7 and joy.buttons[7]:   # SR
            w_roll = -speed * 2.0

        # Pitch/Yaw: D-pad (hat 轴)
        dpad_y = getattr(joy, '_dpad_y', 0)
        dpad_x = getattr(joy, '_dpad_x', 0)
        if dpad_y < 0:   w_pitch = speed * 2.0    # D-pad 上 → Pitch 上
        elif dpad_y > 0: w_pitch = -speed * 2.0   # D-pad 下 → Pitch 下
        if dpad_x < 0:   w_yaw = -speed * 2.0     # D-pad 左 → Yaw 左
        elif dpad_x > 0: w_yaw = speed * 2.0      # D-pad 右 → Yaw 右

        # IMU 模式切换（- 键，边沿触发）
        btn_minus = len(joy.buttons) > 8 and joy.buttons[8]
        if btn_minus and not self._prev_btn[8]:
            if self._imu_check.isEnabled():
                self._imu_mode_active = not self._imu_mode_active
                self._imu_check.setChecked(self._imu_mode_active)

        # IMU 覆盖（激活时取代按键姿态控制）
        imu_active = (self._imu_mode_active
                      and hasattr(joy, 'imu_gyro')
                      and any(abs(g) > 0.5 for g in joy.imu_gyro))
        if imu_active:
            gyro = joy.imu_gyro
            imu_gain = 0.05
            w_max = 0.5  # 角速度上限 (rad/s)
            w_roll = np.clip(np.radians(gyro[0]) * imu_gain, -w_max, w_max)
            w_pitch = np.clip(np.radians(gyro[1]) * imu_gain, -w_max, w_max)
            w_yaw = np.clip(np.radians(gyro[2]) * imu_gain, -w_max, w_max)
            self._imu_status.setText(
                f"IMU: R={gyro[0]:+.0f} P={gyro[1]:+.0f} Y={gyro[2]:+.0f} deg/s"
            )
        elif self._imu_mode_active:
            self._imu_status.setText("IMU: 等待陀螺仪数据...")
        else:
            self._imu_status.setText("IMU: 关闭 (按 - 切换)")

        btn_a = len(joy.buttons) > 0 and joy.buttons[0]

        if btn_a and not self._prev_btn[0]:
            self._cycle_gamepad_speed()

        self._prev_btn = list(joy.buttons[:16])

        if (abs(vx) < 1e-4 and abs(vy) < 1e-4 and abs(vz) < 1e-4
                and abs(w_roll) < 1e-4 and abs(w_pitch) < 1e-4 and abs(w_yaw) < 1e-4):
            return

        dt = 0.02
        start_q = None
        if self._mode == "real" and self._real_manager.is_connected:
            start_q = self._real_manager.get_joint_feedback()
            if start_q is None:
                self._status_bar.showMessage("游戏手柄实机反馈不可用，指令未发送")
                return
            with lock:
                for i, idx in enumerate(arm_qpos_indices):
                    data.qpos[idx] = start_q[i]
                    self._configuration.data.qpos[idx] = start_q[i]
                fix_base_position(model, data)
                fix_base_position(self._configuration.model, self._configuration.data)
                mujoco.mj_forward(model, data)
                mujoco.mj_forward(self._configuration.model, self._configuration.data)

        pos = self._get_current_ee_pos()
        cur_quat = self._get_current_quat()  # [x, y, z, w]

        target_pos = pos + np.array([vx, vy, vz]) * dt
        # 用角速度构造增量四元数，避免 RPY 万向锁
        wvec = np.array([w_roll, w_pitch, w_yaw]) * dt
        angle = np.linalg.norm(wvec)
        if angle > 1e-8:
            dq = Rotation.from_rotvec(wvec).as_quat()
            new_quat = Rotation.from_quat(dq) * Rotation.from_quat(cur_quat)
            target_rot = new_quat.as_matrix()
        else:
            target_rot = self._get_current_rot_matrix()

        pos_error = np.linalg.norm(target_pos - pos)
        rot_error = angle if angle > 1e-8 else 0.0
        n_steps = self._calc_ik_steps(max(pos_error, rot_error))
        q_before_ik = self._configuration.q.copy()
        ok = self._solve_velocity_ik(
            target_pos,
            target_rot=target_rot,
            n_steps=n_steps,
            caller="gamepad",
        )
        if not ok:
            self._restore_ik_snapshot(q_before_ik)
            return

        # 耦合检测（手柄模式：全轴检查）
        pos_tol, rot_tol = self._calc_drift_tolerances(
            max(pos_error, 0.001), max(rot_error, 0.01))
        passed, reason = self._validate_ik_result(
            target_pos, target_rot, pos_tol, rot_tol,
            check_axes_pos=(0, 1, 2),
            check_axes_rot=(0, 1, 2),
            caller="gamepad",
        )
        if not passed:
            self._restore_ik_snapshot(q_before_ik)
            return  # 手柄静默拒绝，不刷状态栏

        self._sync_config_to_global()

        if self._mode == "real" and self._real_manager.is_connected:
            q = self._get_current_joint_angles()
            try:
                self._real_manager.send_joint_command(q, start_q=start_q)
            except Exception as e:
                self._status_bar.showMessage(f"游戏手柄实机指令发送失败: {e}")

        self._refresh_position()

    @Slot()
    def _on_stop(self):
        self._gamepad_timer.stop()
        self._render_timer.stop()
        self._traj_positions.clear()
        if self._joystick is not None:
            self._disconnect_gamepad()
        if self._viewer is not None:
            try: self._viewer_ctx.__exit__(None, None, None)
            except Exception: pass
            self._viewer = None
            self._viewer_ctx = None
        self._configuration = None
        self._ee_task = None
        self._set_buttons_enabled(False)
        self._start_btn.setEnabled(True)
        self._stop_btn.setEnabled(False)
        if self._mode == "real" and self._real_manager.is_connected:
            self._status_bar.showMessage("视图已停止，实机仍保持连接")
        else:
            self._status_bar.showMessage("已停止")

    def _set_buttons_enabled(self, enabled: bool):
        for btn in [
            self._btn_x_neg, self._btn_x_pos,
            self._btn_y_neg, self._btn_y_pos,
            self._btn_z_neg, self._btn_z_pos,
            self._btn_r_neg, self._btn_r_pos,
            self._btn_p_neg, self._btn_p_pos,
            self._btn_yo_neg, self._btn_yo_pos,
            self._btn_home,
        ]:
            btn.setEnabled(enabled)

    def _get_current_ee_pos(self) -> np.ndarray:
        if self._configuration is not None:
            return self._configuration.data.site_xpos[ee_site_id].copy()
        with lock:
            return data.site_xpos[ee_site_id].copy()

    def _get_current_rot_matrix(self) -> np.ndarray:
        if self._configuration is not None:
            return self._configuration.data.site_xmat[ee_site_id].reshape(3, 3).copy()
        with lock:
            return data.site_xmat[ee_site_id].reshape(3, 3).copy()

    def _get_current_rpy(self) -> np.ndarray:
        rot = self._get_current_rot_matrix()
        quat = rotation_matrix_to_quat(rot)
        w, x, y, z = quat
        roll = np.arctan2(2*(w*x + y*z), 1 - 2*(x*x + y*y))
        sinp = np.clip(2*(w*y - z*x), -1, 1)
        pitch = np.arcsin(sinp)
        yaw = np.arctan2(2*(w*z + x*y), 1 - 2*(y*y + z*z))
        return np.array([roll, pitch, yaw])

    def _rpy_to_rot_matrix(self, rpy: np.ndarray) -> np.ndarray:
        r, p, y = rpy
        cr, sr = np.cos(r), np.sin(r)
        cp, sp = np.cos(p), np.sin(p)
        cy, sy = np.cos(y), np.sin(y)
        return np.array([
            [cr*cy + sr*sp*sy, -cr*sy + sr*sp*cy,  cp*sr],
            [         cp*sy,            cp*cy,        -sp],
            [-sr*cy + cr*sp*sy,  sr*sy + cr*sp*cy,  cr*cp],
        ])

    def _get_current_quat(self) -> np.ndarray:
        """返回当前末端姿态四元数 [x, y, z, w]（scipy 格式）"""
        rot = self._get_current_rot_matrix()
        return Rotation.from_matrix(rot).as_quat()  # [x, y, z, w]

    @staticmethod
    def _quat_to_rot_matrix(quat: np.ndarray) -> np.ndarray:
        """四元数 [x, y, z, w] → 3x3 旋转矩阵"""
        return Rotation.from_quat(quat).as_matrix()

    @staticmethod
    def _delta_quat(axis: str, angle_rad: float) -> np.ndarray:
        """构造绕单轴旋转 angle_rad 的四元数 [x, y, z, w]"""
        axis_map = {
            "x": [1, 0, 0], "y": [0, 1, 0], "z": [0, 0, 1],
            "roll": [1, 0, 0], "pitch": [0, 1, 0], "yaw": [0, 0, 1],
        }
        return Rotation.from_rotvec(np.array(axis_map[axis]) * angle_rad).as_quat()

    def _get_current_joint_angles(self) -> np.ndarray:
        if self._configuration is not None:
            return np.array(
                [self._configuration.q[idx] for idx in arm_qpos_indices], dtype=float
            )
        with lock:
            return np.array([data.qpos[idx] for idx in arm_qpos_indices], dtype=float)

    def _move(self, axis: str, direction: int):
        if self._viewer is None or not self._viewer.is_running():
            self._status_bar.showMessage("请先启动视图")
            return
        if self._configuration is None:
            self._status_bar.showMessage("mink 未初始化")
            return

        if self._mode == "real" and self._real_manager.is_connected:
            seed = self._real_manager.get_joint_feedback()
            if seed is not None:
                with lock:
                    for i, idx in enumerate(arm_qpos_indices):
                        data.qpos[idx] = seed[i]
                        self._configuration.data.qpos[idx] = seed[i]
                    fix_base_position(model, data)
                    fix_base_position(self._configuration.model, self._configuration.data)
                    mujoco.mj_forward(model, data)
                    mujoco.mj_forward(self._configuration.model, self._configuration.data)

        pos = self._get_current_ee_pos()
        rot = self._get_current_rot_matrix()
        delta = np.zeros(3)
        delta[{"x": 0, "y": 1, "z": 2}[axis]] = direction * self._step_size
        target_pos = pos + delta

        pos_error = np.linalg.norm(target_pos - pos)
        n_steps = self._calc_ik_steps(pos_error)
        q_before_ik = self._configuration.q.copy()
        ok = self._solve_velocity_ik(
            target_pos,
            target_rot=None,
            n_steps=n_steps,
            caller=f"move_{axis}",
        )
        if not ok:
            self._restore_ik_snapshot(q_before_ik)
            self._status_bar.showMessage("逆运动学求解失败：目标可能超出关节限位或工作空间")
            return

        # 耦合检测：验证未命令的轴没有超标漂移
        pos_tol, rot_tol = self._calc_drift_tolerances(self._step_size, self._rot_step)
        uncmd_pos_axes = {"x": (1, 2), "y": (0, 2), "z": (0, 1)}[axis]
        passed, reason = self._validate_ik_result(
            target_pos, rot, pos_tol, rot_tol,
            check_axes_pos=uncmd_pos_axes,  # Y/Z, X/Z, 或 X/Y
            check_axes_rot=(0, 1, 2),        # Roll, Pitch, Yaw 全部检查
            caller=f"move_{axis}",
        )
        if not passed:
            self._restore_ik_snapshot(q_before_ik)
            self._status_bar.showMessage(f"⚠ 安全拒绝: {reason}，指令未执行")
            return

        self._sync_config_to_global()

        if self._mode == "real" and self._real_manager.is_connected:
            q = self._get_current_joint_angles()
            try:
                self._real_manager.send_joint_command(q)
            except Exception as e:
                self._status_bar.showMessage(f"实机指令发送失败: {e}")
                return

        sign = "+" if direction > 0 else "-"
        self._status_bar.showMessage(
            f"已移动 {axis.upper()}{sign}  {self._step_size * 1000:.1f}mm"
        )
        self._refresh_position()

    def _rotate(self, axis: str, direction: int):
        if self._viewer is None or not self._viewer.is_running():
            self._status_bar.showMessage("请先启动视图")
            return
        if self._configuration is None:
            self._status_bar.showMessage("mink 未初始化")
            return

        if self._mode == "real" and self._real_manager.is_connected:
            seed = self._real_manager.get_joint_feedback()
            if seed is not None:
                with lock:
                    for i, idx in enumerate(arm_qpos_indices):
                        data.qpos[idx] = seed[i]
                        self._configuration.data.qpos[idx] = seed[i]
                    fix_base_position(model, data)
                    fix_base_position(self._configuration.model, self._configuration.data)
                    mujoco.mj_forward(model, data)
                    mujoco.mj_forward(self._configuration.model, self._configuration.data)

        pos = self._get_current_ee_pos()
        cur_quat = self._get_current_quat()  # [x, y, z, w]
        cur_rot = self._get_current_rot_matrix()
        delta_angle = direction * self._rot_step
        dq = self._delta_quat(axis, delta_angle)
        new_quat = Rotation.from_quat(dq) * Rotation.from_quat(cur_quat)
        target_rot = new_quat.as_matrix()

        rot_error = np.linalg.norm(Rotation.from_matrix(target_rot @ cur_rot.T).as_rotvec())
        n_steps = self._calc_ik_steps(rot_error)
        q_before_ik = self._configuration.q.copy()
        ok = self._solve_velocity_ik(
            None,
            target_rot=target_rot,
            n_steps=n_steps,
            caller=f"rotate_{axis}",
        )
        if not ok:
            self._restore_ik_snapshot(q_before_ik)
            self._status_bar.showMessage("逆运动学求解失败：目标可能超出关节限位或工作空间")
            return

        # 耦合检测：旋转模式下检查位置漂移 + 未命令的旋转轴
        pos_tol, rot_tol = self._calc_drift_tolerances(self._step_size, self._rot_step)
        uncmd_rot_axes = {"roll": (1, 2), "pitch": (0, 2), "yaw": (0, 1)}[axis]
        passed, reason = self._validate_ik_result(
            pos, target_rot, pos_tol, rot_tol,
            check_axes_pos=None,        # 位置全部检查
            check_axes_rot=uncmd_rot_axes,  # 只检查未命令的旋转轴
            caller=f"rotate_{axis}",
        )
        if not passed:
            self._restore_ik_snapshot(q_before_ik)
            self._status_bar.showMessage(f"⚠ 安全拒绝: {reason}，指令未执行")
            return

        self._sync_config_to_global()

        if self._mode == "real" and self._real_manager.is_connected:
            q = self._get_current_joint_angles()
            try:
                self._real_manager.send_joint_command(q)
            except Exception as e:
                self._status_bar.showMessage(f"实机指令发送失败: {e}")
                return

        axis_label = {"roll": "横滚", "pitch": "俯仰", "yaw": "偏航"}[axis]
        sign = "+" if direction > 0 else "-"
        self._status_bar.showMessage(
            f"已旋转 {axis_label}{sign}  {np.degrees(self._rot_step):.1f} 度"
        )
        self._refresh_position()

    def _move_home(self):
        if self._home_timer is not None:
            return
        if self._viewer is None or not self._viewer.is_running():
            self._status_bar.showMessage("请先启动视图")
            return
        if self._configuration is None:
            return

        # 记录当前关节位置，启动插值动画
        self._home_start_q = np.array(
            [self._configuration.data.qpos[idx] for idx in arm_qpos_indices], dtype=float
        )
        self._home_target_q = np.zeros(N_ARM)
        self._home_step = 0
        self._home_total_steps = 40  # ~0.8s @ 50Hz
        self._status_bar.showMessage("正在回零...")
        self._home_timer = QTimer(self)
        self._home_timer.timeout.connect(self._home_tick)
        self._home_timer.start(20)  # 50Hz

    def _home_tick(self):
        self._home_step += 1
        t = min(self._home_step / self._home_total_steps, 1.0)
        # 平滑插值 (ease-in-out)
        t_smooth = t * t * (3 - 2 * t)
        q_interp = self._home_start_q + (self._home_target_q - self._home_start_q) * t_smooth

        with lock:
            for i, idx in enumerate(arm_qpos_indices):
                self._configuration.data.qpos[idx] = q_interp[i]
            fix_base_position(self._configuration.model, self._configuration.data)
            mujoco.mj_forward(self._configuration.model, self._configuration.data)

        self._sync_config_to_global()

        if t >= 1.0:
            self._home_timer.stop()
            self._home_timer.deleteLater()
            self._home_timer = None

            # 实体机器人
            if self._mode == "real" and self._real_manager.is_connected:
                try:
                    self._real_manager.send_joint_command(self._home_target_q)
                except Exception as e:
                    self._status_bar.showMessage(f"回零失败: {e}")
                    return

            self._status_bar.showMessage("已回零")
            self._refresh_position()

    def _refresh_position(self):
        try:
            pos = self._get_current_ee_pos()
            rpy = self._get_current_rpy()
            self._pos_label.setText(
                f"Pos:  X={pos[0]:+.4f}m  Y={pos[1]:+.4f}m  Z={pos[2]:+.4f}m"
            )
            self._rpy_label.setText(
                f"RPY:  R={np.degrees(rpy[0]):+.1f} deg  "
                f"P={np.degrees(rpy[1]):+.1f} deg  "
                f"Y={np.degrees(rpy[2]):+.1f} deg"
            )
        except Exception:
            self._pos_label.setText("位置: (未初始化)")
            self._rpy_label.setText("RPY: (未初始化)")

        # Update non-home warning label
        self._update_home_warning()

        # Update joint feedback
        self._update_joint_feedback()

    def _update_home_warning(self):
        """检测当前关节角度并实时刷新'非零位'警告标签。"""
        angles = None
        if self._mode == "real" and self._real_manager.is_connected:
            angles = self._real_manager.get_joint_feedback()
        elif self._configuration is not None:
            angles = np.array(
                [self._configuration.q[idx] for idx in arm_qpos_indices],
                dtype=float,
            )

        if angles is None:
            self._home_warning_label.setVisible(False)
            return

        max_dev = float(np.max(np.abs(angles)))
        if max_dev > HOME_THRESHOLD_RAD:
            self._home_warning_label.setText(
                f"⚠ 机器人未处于零位（最大偏差 {np.degrees(max_dev):.1f}°）"
                f" — 建议点击下方「回零」按钮"
            )
            self._home_warning_label.setVisible(True)
        else:
            self._home_warning_label.setVisible(False)

    # ---- Joint control methods ----

    def _joint_slider_to_deg(self, joint_idx, slider_val):
        lo, hi = JOINT_LIMITS_DEG[joint_idx + 1]
        return lo + (hi - lo) * slider_val / SLIDER_RESOLUTION

    def _joint_deg_to_slider(self, joint_idx, deg_val):
        lo, hi = JOINT_LIMITS_DEG[joint_idx + 1]
        if hi == lo:
            return SLIDER_RESOLUTION // 2
        return int((deg_val - lo) / (hi - lo) * SLIDER_RESOLUTION)

    def _on_joint_slider_changed(self, idx, val):
        if self._joint_updating:
            return
        deg = self._joint_slider_to_deg(idx, val)
        self._joint_updating = True
        self._joint_spinboxes[idx].setValue(deg)
        self._joint_updating = False
        self._joint_target_dirty[idx] = True

    def _on_joint_spinbox_changed(self, idx, val):
        if self._joint_updating:
            return
        self._joint_updating = True
        self._joint_sliders[idx].setValue(self._joint_deg_to_slider(idx, val))
        self._joint_updating = False
        self._joint_target_dirty[idx] = True

    def _joint_set_controls_from_angles(
        self,
        arm_angles: np.ndarray,
        gripper_angle: float | None = None,
        *,
        mark_clean: bool = True,
    ):
        """Update joint target controls from a known robot/simulation state."""
        self._joint_updating = True
        for i in range(N_ARM):
            lo, hi = JOINT_LIMITS_DEG[i + 1]
            deg = float(np.clip(np.degrees(arm_angles[i]), lo, hi))
            self._joint_spinboxes[i].setValue(deg)
            self._joint_sliders[i].setValue(self._joint_deg_to_slider(i, deg))
        if gripper_angle is not None:
            lo, hi = JOINT_LIMITS_DEG[7]
            gripper_deg = float(np.clip(np.degrees(gripper_angle), lo, hi))
            self._joint_spinboxes[6].setValue(gripper_deg)
            self._joint_sliders[6].setValue(self._joint_deg_to_slider(6, gripper_deg))
        self._joint_updating = False
        if mark_clean:
            self._joint_target_dirty = [False] * 7

    def _joint_send_command(self):
        if self._joint_move_timer is not None:
            return
        if self._viewer is None or not self._viewer.is_running():
            self._status_bar.showMessage("请先启动视图")
            return
        if self._configuration is None:
            self._status_bar.showMessage("mink 未初始化")
            return

        # A1: 实机模式下，用真机反馈刷新仿真起点，保证 MuJoCo 与真机同起点
        seed = None
        if self._mode == "real" and self._real_manager.is_connected:
            seed = self._real_manager.get_joint_feedback()
            if seed is not None:
                with lock:
                    for i, idx in enumerate(arm_qpos_indices):
                        data.qpos[idx] = seed[i]
                        self._configuration.data.qpos[idx] = seed[i]
                    fix_base_position(model, data)
                    fix_base_position(self._configuration.model, self._configuration.data)
                    mujoco.mj_forward(model, data)
                    mujoco.mj_forward(self._configuration.model, self._configuration.data)

        # Current joint angles
        q_start = np.array(
            [self._configuration.data.qpos[idx] for idx in arm_qpos_indices],
            dtype=float,
        )
        gripper_start = float(self._configuration.data.qpos[gripper_qpos_index])

        # 实机模式下，未编辑的关节保持当前反馈；只让用户改过的轴成为目标。
        # 否则旧 spinbox 值（常见为 0°）会把其它关节一起拉走。
        use_dirty_targets = self._mode == "real" and self._real_manager.is_connected
        q_target = q_start.copy() if use_dirty_targets else np.zeros(N_ARM)
        for i in range(N_ARM):
            if not use_dirty_targets or self._joint_target_dirty[i]:
                q_target[i] = np.radians(self._joint_spinboxes[i].value())
        if use_dirty_targets and not self._joint_target_dirty[6]:
            gripper_target = gripper_start
        else:
            gripper_target = np.clip(
                np.radians(self._joint_spinboxes[6].value()),
                GRIPPER_JOINT_MIN,
                GRIPPER_JOINT_MAX,
            )

        delta_deg = np.degrees(q_target - q_start)
        dirty_axes = [f"L{i + 1}" for i, dirty in enumerate(self._joint_target_dirty[:N_ARM]) if dirty]
        moved_axes = [f"L{i + 1}={delta_deg[i]:+.2f}°" for i in range(N_ARM) if abs(delta_deg[i]) > 0.05]
        log_msg = (
            "[关节控制下发] "
            f"dirty={dirty_axes or ['none']} "
            f"delta={moved_axes or ['none']} "
            f"start={np.array2string(np.degrees(q_start), precision=2, suppress_small=True)}deg "
            f"target={np.array2string(np.degrees(q_target), precision=2, suppress_small=True)}deg"
        )
        self._pose_logger.info(log_msg)
        print(log_msg, flush=True)

        # If already at target, skip
        if np.allclose(q_start, q_target, atol=1e-4) and \
           abs(gripper_start - gripper_target) < 1e-4:
            self._status_bar.showMessage("已在目标位置")
            return

        # Setup interpolation
        duration = self._joint_duration_spin.value()
        tick_ms = 20  # 50Hz, same as render timer
        total_steps = max(1, int(duration * 1000 / tick_ms))

        self._joint_move_start_q = q_start.copy()
        self._joint_move_target_q = q_target.copy()
        self._joint_move_target_gripper = gripper_target
        self._joint_move_start_gripper = gripper_start
        self._joint_move_step = 0
        self._joint_move_total_steps = total_steps
        # A2: 逐点下发时，用上一 tick 的插值作为下一段的 start_q
        self._joint_move_prev_q = q_start.copy()

        self._status_bar.showMessage(f"关节运动中 ({duration:.1f}s)...")
        self._joint_move_timer = QTimer(self)
        self._joint_move_timer.timeout.connect(self._joint_move_tick)
        self._joint_move_timer.start(tick_ms)

    def _joint_move_tick(self):
        self._joint_move_step += 1
        t = min(self._joint_move_step / self._joint_move_total_steps, 1.0)
        # Smooth ease-in-out
        t_smooth = t * t * (3 - 2 * t)

        q_interp = self._joint_move_start_q + \
            (self._joint_move_target_q - self._joint_move_start_q) * t_smooth
        gripper_interp = self._joint_move_start_gripper + \
            (self._joint_move_target_gripper - self._joint_move_start_gripper) * t_smooth

        with lock:
            for i, idx in enumerate(arm_qpos_indices):
                self._configuration.data.qpos[idx] = q_interp[i]
                data.qpos[idx] = q_interp[i]
            self._configuration.data.qpos[gripper_qpos_index] = gripper_interp
            data.qpos[gripper_qpos_index] = gripper_interp
            fix_base_position(model, data)
            fix_base_position(self._configuration.model, self._configuration.data)
            mujoco.mj_forward(model, data)
            mujoco.mj_forward(self._configuration.model, self._configuration.data)

        # A2: 每个 tick 用当前插值目标做小段下发，让实机与仿真同步走
        if self._mode == "real" and self._real_manager.is_connected:
            try:
                self._real_manager.send_joint_command(
                    q_interp, start_q=self._joint_move_prev_q,
                )
                # A4: 夹爪 L7 通道也一起下发
                self._real_manager.send_gripper_command(gripper_interp)
            except Exception as e:
                self._status_bar.showMessage(f"实机指令发送失败: {e}")
                self._joint_move_timer.stop()
                self._joint_move_timer.deleteLater()
                self._joint_move_timer = None
                return
        self._joint_move_prev_q = q_interp.copy()

        if t >= 1.0:
            self._joint_move_timer.stop()
            self._joint_move_timer.deleteLater()
            self._joint_move_timer = None
            self._joint_set_controls_from_angles(
                self._joint_move_target_q,
                self._joint_move_target_gripper,
                mark_clean=True,
            )

            self._status_bar.showMessage("关节运动完成")
            self._refresh_position()

    def _joint_go_zero(self):
        if self._viewer is None or not self._viewer.is_running():
            self._status_bar.showMessage("请先启动视图")
            return

        self._joint_updating = True
        for i in range(7):
            lo, hi = JOINT_LIMITS_DEG[i + 1]
            zero_val = 0.0 if lo <= 0.0 <= hi else lo
            self._joint_spinboxes[i].setValue(zero_val)
            self._joint_sliders[i].setValue(self._joint_deg_to_slider(i, zero_val))
        self._joint_updating = False
        self._joint_target_dirty = [True] * 7

        self._joint_send_command()

    def _joint_sync_to_sim(self):
        """Sync sliders/spinboxes to current simulation or real feedback angles."""
        if self._mode == "real" and self._real_manager.is_connected:
            fb = self._real_manager.get_joint_feedback()
            if fb is None:
                self._status_bar.showMessage("无法读取实机当前关节反馈")
                return
            self._joint_set_controls_from_angles(np.asarray(fb, dtype=float))
            self._status_bar.showMessage("已同步实机当前关节角度")
            return

        if self._configuration is None:
            return
        arm_angles = np.array(
            [self._configuration.q[idx] for idx in arm_qpos_indices],
            dtype=float,
        )
        self._joint_set_controls_from_angles(
            arm_angles,
            float(self._configuration.q[gripper_qpos_index]),
        )
        self._status_bar.showMessage("已同步仿真当前关节角度")

    def _update_joint_feedback(self):
        """Update joint feedback labels from sim or real robot."""
        if not self._joint_feedback_labels:
            return

        # Get joint angles
        if self._mode == "real" and self._real_manager.is_connected:
            fb = self._real_manager.get_joint_feedback()
            if fb is not None:
                angles_rad = fb
            else:
                angles_rad = None
        else:
            if self._configuration is not None:
                angles_rad = np.array(
                    [self._configuration.q[idx] for idx in arm_qpos_indices], dtype=float
                )
            else:
                angles_rad = None

        if angles_rad is None:
            return

        for i in range(N_ARM):
            deg = np.degrees(angles_rad[i])
            self._joint_feedback_labels[i].setText(f"{deg:+.1f}°")

        # Gripper feedback
        if self._configuration is not None:
            gripper_deg = np.degrees(self._configuration.q[gripper_qpos_index])
            self._joint_feedback_labels[6].setText(f"{gripper_deg:+.1f}°")

        # Update joint angles rad display
        if self._configuration is not None:
            all_angles = [self._configuration.q[idx] for idx in arm_qpos_indices]
            all_angles.append(self._configuration.q[gripper_qpos_index])
            angles_str = "  ".join(f"L{i+1}={a:+.3f}" for i, a in enumerate(all_angles))
            self._joint_angles_label.setText(angles_str)

    def closeEvent(self, event):
        self._gamepad_timer.stop()
        self._render_timer.stop()
        self._pos_timer.stop()
        if self._joystick is not None:
            self._joystick.disconnect()
        if self._real_manager.is_connected:
            self._real_manager.disconnect()
        if self._viewer is not None:
            try: self._viewer_ctx.__exit__(None, None, None)
            except Exception: pass
        event.accept()


def main():
    app = QApplication(sys.argv)
    app.setStyle("Fusion")
    window = CartesianControlWindow()
    window.show()
    sys.exit(app.exec())


if __name__ == "__main__":
    main()
