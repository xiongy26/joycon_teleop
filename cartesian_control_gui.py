#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Standalone Cartesian control GUI for EL-A3 gripper."""

from __future__ import annotations

import sys
import time
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
    QGroupBox,
    QHBoxLayout,
    QLabel,
    QMainWindow,
    QMessageBox,
    QPushButton,
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

import threading
import struct


class RealRobotManager:
    """Manage EL-A3 real robot connection."""

    def __init__(self):
        self.arm = None
        self._connected = False

    def connect(self, can_name: str = "can0", kp: float = 0.5, kd: float = 0.01):
        from core.real_robot_trajectory import (
            RealRobotOptions,
            _connect_real_arm,
            _ensure_sdk_on_path,
            _ensure_socketcan_ready,
        )
        options = RealRobotOptions(can_name=can_name, kp=kp, kd=kd)
        _ensure_sdk_on_path()
        _ensure_socketcan_ready(options)
        self.arm = _connect_real_arm(options)
        self._connected = True

    def disconnect(self):
        if self.arm is not None:
            try: self.arm.stop_control_loop()
            except Exception: pass
            try: self.arm.DisableArm()
            except Exception: pass
            try: self.arm.DisconnectPort()
            except Exception: pass
            self.arm = None
            self._connected = False

    def get_joint_feedback(self) -> np.ndarray | None:
        if self.arm is None:
            return None
        try:
            joints = self.arm.GetArmJointMsgs()
            if joints is None or joints.timestamp <= 0:
                return None
            values = np.array(joints.to_list(include_gripper=False)[:N_ARM], dtype=float)
            if len(values) != N_ARM or np.any(~np.isfinite(values)):
                return None
            return np.clip(values, ARM_JOINT_LIMITS_LOWER, ARM_JOINT_LIMITS_UPPER)
        except Exception:
            return None

    def send_joint_command(self, q_target: np.ndarray):
        if self.arm is None:
            return
        from core.real_robot_trajectory import build_sdk_queue_trajectory
        start_q = self.get_joint_feedback()
        if start_q is None:
            start_q = np.zeros(N_ARM)
        joint_plan = [start_q, np.array(q_target, dtype=float)]
        trajectory, _ = build_sdk_queue_trajectory(
            joint_plan, sample_period_s=1/200, control_period_s=1/200,
            max_joint_velocity_rad_s=2.0, start_q=start_q,
        )
        if trajectory:
            self.arm._execute_trajectory_async(trajectory, block=False)

    @property
    def is_connected(self):
        return self._connected

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
        self.setWindowTitle("EL-A3 笛卡尔控制 (mink 速度逆运动学)")
        self.setMinimumSize(420, 480)

        self._real_manager = RealRobotManager()
        self._mode = "sim"
        self._step_size = 0.01
        self._rot_step = np.radians(5.0)
        self._dt = 0.05

        self._configuration: mink.Configuration | None = None
        self._ee_task: mink.FrameTask | None = None
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
        self._kp_spin.setRange(0.01, 10.0)
        self._kp_spin.setValue(0.5)
        self._kp_spin.setSingleStep(0.1)
        real_layout.addWidget(self._kp_spin, 0, 3)
        real_layout.addWidget(QLabel("Kd:"), 1, 0)
        self._kd_spin = QDoubleSpinBox()
        self._kd_spin.setRange(0.001, 1.0)
        self._kd_spin.setValue(0.01)
        self._kd_spin.setSingleStep(0.005)
        self._kd_spin.setDecimals(3)
        real_layout.addWidget(self._kd_spin, 1, 1)
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
        gp_layout.addWidget(info, 3, 0, 1, 4)
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

        self._btn_home = QPushButton("回零")
        self._btn_home.setStyleSheet("QPushButton { min-height: 40px; font-size: 13px; }")
        self._btn_home.clicked.connect(self._move_home)
        ctrl_layout.addWidget(self._btn_home, 6, 0, 1, 3)
        tab_motion_layout.addWidget(ctrl_group)

        # Position display
        pos_group = QGroupBox("当前末端执行器位姿")
        pos_layout = QVBoxLayout(pos_group)
        self._pos_label = QLabel("Pos: ---")
        self._pos_label.setStyleSheet("font-family: monospace; font-size: 13px;")
        pos_layout.addWidget(self._pos_label)
        self._rpy_label = QLabel("RPY: ---")
        self._rpy_label.setStyleSheet("font-family: monospace; font-size: 13px;")
        pos_layout.addWidget(self._rpy_label)
        tab_motion_layout.addWidget(pos_group)
        tab_motion_layout.addStretch()

        self._tabs.addTab(tab_motion, "运动控制")

        # Status bar
        self._status_bar = QStatusBar()
        self.setStatusBar(self._status_bar)
        self._status_bar.showMessage("就绪")

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

        # 根据命令类型设置目标 SE3（位置和方向均使用 1.0 cost 锁定）
        if target_pos is not None and target_rot is not None:
            mode = "pos+rot"
            T = np.eye(4)
            T[:3, :3] = target_rot
            T[:3, 3] = target_pos
            target_se3 = mink.SE3.from_matrix(T)
            self._ee_task.set_position_cost(1.0)
            self._ee_task.set_orientation_cost(1.0)
        elif target_rot is not None:
            mode = "rot_only"
            # 纯旋转：保持当前位置，只更新朝向
            T = np.eye(4)
            T[:3, :3] = target_rot
            T[:3, 3] = current_pos
            target_se3 = mink.SE3.from_matrix(T)
            self._ee_task.set_position_cost(1.0)
            self._ee_task.set_orientation_cost(1.0)
        else:
            mode = "pos_only"
            # 纯平移：保持当前朝向，只更新位置
            T = np.eye(4)
            T[:3, :3] = current_rot
            T[:3, 3] = target_pos
            target_se3 = mink.SE3.from_matrix(T)
            self._ee_task.set_position_cost(1.0)
            self._ee_task.set_orientation_cost(1.0)
        self._ee_task.set_target(target_se3)

        for i in range(n_steps):
            vel = mink.solve_ik(self._configuration, tasks, self._dt, self._solver)

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

            if target_pos is not None:
                actual_pos = self._configuration.data.site_xpos[ee_site_id]
                if np.linalg.norm(actual_pos - target_pos) < 0.002:
                    self._log_pose(mode, caller, current_pos, current_quat, target_pos, target_rot, i + 1, n_steps)
                    return True

        self._log_pose(mode, caller, current_pos, current_quat, target_pos, target_rot, n_steps, n_steps)
        return True

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
        self._real_group.setVisible(self._mode == "real")

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
        can_name = self._can_combo.currentText().strip()
        kp = self._kp_spin.value()
        kd = self._kd_spin.value()
        self._status_bar.showMessage(f"正在连接实机 ({can_name}) ...")
        self._start_btn.setEnabled(False)
        try:
            self._real_manager.connect(can_name=can_name, kp=kp, kd=kd)
        except Exception as e:
            self._start_btn.setEnabled(True)
            self._status_bar.showMessage(f"实机连接失败: {e}")
            QMessageBox.critical(self, "连接失败", str(e))
            return

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
            self._real_manager.disconnect()
            self._start_btn.setEnabled(True)
            self._status_bar.showMessage(f"视图启动失败: {e}")
            QMessageBox.critical(self, "视图启动失败", str(e))

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
        pos = self._get_current_ee_pos()
        cur_quat = self._get_current_quat()  # [x, y, z, w]

        cur_rot = self._get_current_rot_matrix()
        target_pos = pos + cur_rot @ np.array([vx, vy, vz]) * dt
        # 用角速度构造增量四元数，避免 RPY 万向锁
        wvec = np.array([w_roll, w_pitch, w_yaw]) * dt
        angle = np.linalg.norm(wvec)
        if angle > 1e-8:
            dq = Rotation.from_rotvec(wvec).as_quat()
            new_quat = Rotation.from_quat(cur_quat) * Rotation.from_quat(dq)
            target_rot = new_quat.as_matrix()
        else:
            target_rot = self._get_current_rot_matrix()

        ok = self._solve_velocity_ik(target_pos, target_rot=target_rot, n_steps=5, caller="gamepad")
        if not ok:
            return

        self._sync_config_to_global()

        if self._mode == "real" and self._real_manager.is_connected:
            q = self._get_current_joint_angles()
            try:
                self._real_manager.send_joint_command(q)
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
        if self._real_manager.is_connected:
            self._real_manager.disconnect()
        self._set_buttons_enabled(False)
        self._start_btn.setEnabled(True)
        self._stop_btn.setEnabled(False)
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
        target_pos = pos + rot @ delta

        ok = self._solve_velocity_ik(target_pos, target_rot=None, n_steps=5, caller=f"move_{axis}")
        if not ok:
            self._status_bar.showMessage("逆运动学求解失败")
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
        delta_angle = direction * self._rot_step
        dq = self._delta_quat(axis, delta_angle)
        # 四元数乘法：cur_quat 再 dq（局部坐标系旋转）
        new_quat = Rotation.from_quat(cur_quat) * Rotation.from_quat(dq)
        target_rot = new_quat.as_matrix()

        ok = self._solve_velocity_ik(None, target_rot=target_rot, n_steps=5, caller=f"rotate_{axis}")
        if not ok:
            self._status_bar.showMessage("逆运动学求解失败")
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
