"""后台工作线程：封装 ELA3Interface，50Hz 数据采集 + 线程安全命令队列"""

import sys
import os
import time
import math
import logging
import traceback
from pathlib import Path
from typing import List, Optional
from queue import Queue, Empty

from PyQt6.QtCore import QThread, pyqtSignal

from MotorStudio.backend.data_buffer import DataBuffer

logger = logging.getLogger("MotorStudio.worker")

# Sentinel for typing
try:
    from el_a3_sdk import ELA3Interface, ArmJointStates, ArmEndPose, ArmStatus
    from el_a3_sdk.protocol import DEFAULT_JOINT_LIMITS, DEFAULT_MOTOR_TYPE_MAP, ParamIndex
except ImportError:
    ELA3Interface = None


class ArmWorker(QThread):
    """
    后台线程：
    - 50Hz 轮询 SDK 反馈
    - 通过 Signal 将数据发送到 UI 线程
    - 通过命令队列接收 UI 的控制指令
    """

    joints_updated = pyqtSignal(object)
    velocities_updated = pyqtSignal(object)
    efforts_updated = pyqtSignal(object)
    status_updated = pyqtSignal(object)
    end_pose_updated = pyqtSignal(object)
    motor_feedback_updated = pyqtSignal(object)
    error_occurred = pyqtSignal(str)
    connected_changed = pyqtSignal(bool)
    enabled_changed = pyqtSignal(bool)
    control_loop_changed = pyqtSignal(bool)
    log_message = pyqtSignal(str)
    can_fps_updated = pyqtSignal(float)
    zero_sta_verified = pyqtSignal(list)
    motor_scan_result = pyqtSignal(list)
    move_j_done = pyqtSignal()

    def __init__(self, parent=None):
        super().__init__(parent)
        self.arm: Optional[object] = None
        self.data_buffer = DataBuffer(max_samples=500, num_channels=7)
        self._running = False
        self._connected = False
        self._enabled = False
        self._cmd_queue: Queue = Queue()
        self._poll_rate_hz = 50.0
        self._sim_mode = False

        self._slow_poll_counter = 0

        # Simulation state
        self._sim_positions = [0.0] * 7
        self._sim_velocities = [0.0] * 7
        self._sim_torques = [0.0] * 7
        self._sim_target = [0.0] * 7

    @property
    def is_connected(self):
        return self._connected

    @property
    def is_enabled(self):
        return self._enabled

    def submit_command(self, cmd: str, *args, **kwargs):
        self._cmd_queue.put((cmd, args, kwargs))

    def run(self):
        self._running = True
        interval = 1.0 / self._poll_rate_hz
        while self._running:
            t0 = time.time()
            try:
                self._process_commands()
                if self._connected:
                    self._poll_feedback()
            except Exception as e:
                logger.error(f"Worker error: {e}\n{traceback.format_exc()}")
                self.error_occurred.emit(str(e))
            elapsed = time.time() - t0
            sleep_time = interval - elapsed
            if sleep_time > 0:
                time.sleep(sleep_time)

    def stop(self):
        self._running = False
        self.wait(3000)

    def _process_commands(self):
        while True:
            try:
                cmd, args, kwargs = self._cmd_queue.get_nowait()
            except Empty:
                break
            try:
                self._execute_command(cmd, args, kwargs)
            except Exception as e:
                logger.error(f"Command '{cmd}' failed: {e}")
                self.error_occurred.emit(f"命令 {cmd} 失败: {e}")

    def _execute_command(self, cmd: str, args, kwargs):
        if cmd == "connect":
            self._do_connect(*args, **kwargs)
        elif cmd == "disconnect":
            self._do_disconnect()
        elif cmd == "enable":
            self._do_enable()
        elif cmd == "disable":
            self._do_disable()
        elif cmd == "emergency_stop":
            self._do_emergency_stop()
        elif cmd == "joint_ctrl":
            self._do_joint_ctrl(*args)
        elif cmd == "move_j":
            self._do_move_j(*args, **kwargs)
        elif cmd == "move_l":
            self._do_move_l(*args, **kwargs)
        elif cmd == "end_pose_ctrl":
            self._do_end_pose_ctrl(*args, **kwargs)
        elif cmd == "cancel_motion":
            self._do_cancel_motion()
        elif cmd == "zero_torque":
            self._do_zero_torque(*args, **kwargs)
        elif cmd == "zero_torque_gravity":
            self._do_zero_torque_gravity(*args, **kwargs)
        elif cmd == "gripper_ctrl":
            self._do_gripper_ctrl(*args)
        elif cmd == "set_zero_position":
            self._do_set_zero(*args)
        elif cmd == "verify_zero_sta":
            self._do_verify_zero_sta()
        elif cmd == "set_all_zero_sta":
            self._do_set_all_zero_sta()
        elif cmd == "scan_motors":
            self._do_scan_motors()
        elif cmd == "read_motor_param":
            self._do_read_param(*args)
        elif cmd == "write_motor_param":
            self._do_write_param(*args)
        elif cmd == "start_control_loop":
            self._do_start_control_loop(*args)
        elif cmd == "stop_control_loop":
            self._do_stop_control_loop()
        elif cmd == "move_j_block":
            self._do_move_j_block(*args, **kwargs)
        elif cmd == "set_smoothing_alpha":
            if self.arm and not self._sim_mode:
                self.arm.SetSmoothingAlpha(args[0])

    def _do_connect(self, can_name="can0", sim_mode=False,
                    backend="socketcan", serial_port=None,
                    serial_baudrate=2000000, can_bitrate=1000000):
        if self._connected:
            return
        self._sim_mode = sim_mode
        if sim_mode:
            self._connected = True
            self.connected_changed.emit(True)
            self.log_message.emit("已连接（模拟模式）")
            return
        if ELA3Interface is None:
            self.error_occurred.emit("el_a3_sdk 未安装")
            return

        if backend != "slcan":
            from MotorStudio.utils.can_utils import get_can_state
            state = get_can_state(can_name)
            if state != "UP":
                self.error_occurred.emit(
                    f"CAN 接口 {can_name} 未开启（当前状态: {state}），请先在工具栏中开启"
                )
                return

        try:
            if getattr(sys, "frozen", False):
                sdk_root = Path(sys._MEIPASS)
            else:
                try:
                    import el_a3_sdk as _el_a3_sdk_pkg
                    sdk_root = Path(_el_a3_sdk_pkg.__file__).resolve().parent.parent
                except Exception:
                    sdk_root = Path(__file__).resolve().parent.parent.parent
            inertia_path = sdk_root / "resources" / "config" / "inertia_params.yaml"
            legacy_urdf_path = sdk_root / "resources" / "urdf" / "el_a3_legacy.urdf"
            kwargs = dict(can_name=can_name)
            if inertia_path.exists():
                kwargs["inertia_config_path"] = str(inertia_path)
            if legacy_urdf_path.exists():
                kwargs["urdf_path"] = str(legacy_urdf_path)
            kwargs["per_joint_kd_min"] = {4: 0.005, 5: 0.005, 6: 0.005, 7: 0.02}
            kwargs["per_joint_kd_max"] = {4: 0.10, 5: 0.05, 6: 0.05, 7: 0.10}
            kwargs["gravity_joint_scale"] = {4: 2.0}
            if backend == "slcan":
                kwargs["backend"] = "slcan"
                kwargs["serial_port"] = serial_port or can_name
                kwargs["serial_baudrate"] = serial_baudrate
                kwargs["can_bitrate"] = can_bitrate
            self.arm = ELA3Interface(**kwargs)
            self.arm.ConnectPort()
            self._connected = True
            self.connected_changed.emit(True)
            display_name = f"{serial_port or can_name} (SLCAN)" if backend == "slcan" else can_name
            self.log_message.emit(f"已连接到 {display_name}")
        except Exception as e:
            self.arm = None
            self.error_occurred.emit(f"连接失败: {e}")

    def _do_disconnect(self):
        if not self._connected:
            return
        if self._sim_mode:
            self._connected = False
            self._enabled = False
            self.connected_changed.emit(False)
            self.enabled_changed.emit(False)
            self.log_message.emit("已断开（模拟模式）")
            return
        try:
            if self.arm:
                if self._enabled:
                    self.arm.DisableArm()
                self.arm.DisconnectPort()
                self.arm = None
            self._connected = False
            self._enabled = False
            self.connected_changed.emit(False)
            self.enabled_changed.emit(False)
            self.log_message.emit("已断开连接")
        except Exception as e:
            self.error_occurred.emit(f"断开失败: {e}")

    def _do_enable(self):
        if not self._connected:
            return
        if self._sim_mode:
            self._enabled = True
            self.enabled_changed.emit(True)
            self.log_message.emit("电机已使能（模拟模式）")
            return
        try:
            self.arm.EnableArm()
            self._enabled = True
            self.enabled_changed.emit(True)
            self.log_message.emit("电机已使能")
        except Exception as e:
            self.error_occurred.emit(f"使能失败: {e}")

    def _do_disable(self):
        if self._sim_mode:
            self._enabled = False
            self.enabled_changed.emit(False)
            self.log_message.emit("电机已失能（模拟模式）")
            return
        try:
            if self.arm:
                self.arm.DisableArm()
            self._enabled = False
            self.enabled_changed.emit(False)
            self.log_message.emit("电机已失能")
        except Exception as e:
            self.error_occurred.emit(f"失能失败: {e}")

    def _do_emergency_stop(self):
        if self._sim_mode:
            self._enabled = False
            self.enabled_changed.emit(False)
            self.log_message.emit("急停已触发（模拟模式）")
            return
        try:
            if self.arm:
                self.arm.EmergencyStop()
            self._enabled = False
            self.enabled_changed.emit(False)
            self.log_message.emit("急停已触发")
        except Exception as e:
            self.error_occurred.emit(f"急停失败: {e}")

    def _ensure_control_loop(self):
        """运动命令前自动启动控制循环（200Hz EMA 平滑 + 速度前馈 + 重力补偿）"""
        if self.arm and not self.arm.control_loop_running:
            self.arm.start_control_loop(rate_hz=200.0)
            self.control_loop_changed.emit(True)
            self.log_message.emit("控制循环已自动启动 (200Hz)")

    def _do_joint_ctrl(self, positions: List[float]):
        if self._sim_mode:
            self._sim_target = list(positions[:7])
            return
        if self.arm and self._enabled:
            self._ensure_control_loop()
            self.arm.JointCtrlList(positions[:6])

    def _do_move_j(self, positions, duration=2.0, block=False):
        if self._sim_mode:
            self._sim_target = list(positions[:7]) + [0.0] * (7 - len(positions))
            self.log_message.emit(f"MoveJ 执行（模拟）duration={duration}s")
            return
        if self.arm and self._enabled:
            self._ensure_control_loop()
            self.arm.MoveJ(positions, duration=duration, block=block)
            self.log_message.emit(f"MoveJ 执行中 duration={duration}s")

    def _do_move_j_block(self, positions, duration=2.0):
        """阻塞式 MoveJ，完成后发出 move_j_done 信号（供标定流程使用）。"""
        if self._sim_mode:
            self._sim_target = list(positions[:7]) + [0.0] * (7 - len(positions))
            import time as _t
            _t.sleep(min(duration, 2.0))
            self.move_j_done.emit()
            return
        if self.arm and self._enabled:
            self._ensure_control_loop()
            self.arm.MoveJ(positions, duration=duration, block=True)
            self.move_j_done.emit()

    def _do_move_l(self, target_pose, duration=2.0, block=False):
        if self._sim_mode:
            self.log_message.emit(f"MoveL 执行（模拟）duration={duration}s")
            return
        if self.arm and self._enabled:
            self._ensure_control_loop()
            self.arm.MoveL(target_pose, duration=duration, block=block)
            self.log_message.emit(f"MoveL 执行中 duration={duration}s")

    def _do_end_pose_ctrl(self, x, y, z, rx, ry, rz, duration=2.0):
        if self._sim_mode:
            self.log_message.emit(f"EndPoseCtrl（模拟）[{x:.3f},{y:.3f},{z:.3f}]")
            return
        if self.arm and self._enabled:
            self._ensure_control_loop()
            self.arm.EndPoseCtrl(x, y, z, rx, ry, rz, duration=duration)

    def _do_cancel_motion(self):
        if self._sim_mode:
            return
        if self.arm:
            self.arm.cancel_motion()
            self.log_message.emit("运动已取消")

    _ZERO_TORQUE_KD = [0.05, 0.05, 0.05, 0.05, 0.0125, 0.0125, 0.05]

    def _do_zero_torque(self, enable):
        if self._sim_mode:
            state = "开启" if enable else "关闭"
            self.log_message.emit(f"零力矩模式{state}（模拟）")
            return
        if self.arm:
            kd = self._ZERO_TORQUE_KD[0] if enable else 1.0
            self.arm.ZeroTorqueMode(enable, kd=kd)
            state = "开启" if enable else "关闭"
            self.log_message.emit(f"零力矩模式{state}")

    def _do_zero_torque_gravity(self, enable):
        if self._sim_mode:
            state = "开启" if enable else "关闭"
            self.log_message.emit(f"重力补偿零力矩{state}（模拟）")
            return
        if self.arm:
            self.arm.ZeroTorqueModeWithGravity(
                enable, kd=self._ZERO_TORQUE_KD, update_rate=100.0)
            state = "开启" if enable else "关闭"
            self.log_message.emit(f"重力补偿零力矩{state}")

    def _do_gripper_ctrl(self, angle):
        if self._sim_mode:
            self._sim_target[6] = angle
            return
        if self.arm and self._enabled:
            self.arm.GripperCtrl(angle)

    def _do_set_zero(self, motor_num=0xFF):
        if self._sim_mode:
            self._sim_positions = [0.0] * 7
            self.log_message.emit("零位已设置（模拟）")
            return
        if not self.arm:
            self.log_message.emit("⚠ 未连接，无法设零")
            return

        import time
        from el_a3_sdk.protocol import ParamIndex, RunMode

        driver = self.arm._driver
        motor_ids = list(range(1, 8)) if motor_num == 0xFF else [motor_num]

        for mid in motor_ids:
            is_gripper = (mid == self.arm.NUM_JOINTS)
            self.log_message.emit(f"===== 电机 {mid} 设零 =====")

            # Step 1: 读取初始状态
            fb_before = driver.get_feedback(mid)
            pos_before = fb_before.position if fb_before and fb_before.is_valid else None
            self.log_message.emit(f"  [1] 初始位置: {pos_before:.4f}" if pos_before else "  [1] 初始位置: 无反馈")

            mode_r = self.arm.ReadMotorParameter(mid, ParamIndex.RUN_MODE)
            mode_before = mode_r.value_uint8 if mode_r and mode_r.success else None
            self.log_message.emit(f"  [1] 初始模式: {mode_before}  (0=运控, 1=PP)")

            # Step 2: 夹爪电机需切到运控模式才能设零
            #         使能状态下无法切换模式，必须先失能
            if is_gripper and mode_before == 1:
                self.log_message.emit(f"  [2] 失能电机 ...")
                ok = driver.disable_motor(mid)
                self.log_message.emit(f"  [2] disable 返回: {ok}")
                time.sleep(0.05)

                self.log_message.emit(f"  [2] 切换到运控模式 ...")
                ok = driver.set_run_mode(mid, RunMode.MOTION_CONTROL)
                self.log_message.emit(f"  [2] set_run_mode 返回: {ok}")
                time.sleep(0.05)

                self.log_message.emit(f"  [2] 重新使能 ...")
                ok = driver.enable_motor(mid)
                self.log_message.emit(f"  [2] enable 返回: {ok}")
                time.sleep(0.1)

                mode_r2 = self.arm.ReadMotorParameter(mid, ParamIndex.RUN_MODE)
                mode_now = mode_r2.value_uint8 if mode_r2 and mode_r2.success else "读取失败"
                self.log_message.emit(f"  [2] 当前模式: {mode_now}")

            # Step 3: 发送 SET_ZERO
            self.log_message.emit(f"  [3] 发送 SET_ZERO (Type 6) ...")
            ok = driver.set_zero_position(mid)
            self.log_message.emit(f"  [3] 发送结果: {ok}")
            time.sleep(0.2)

            # Step 4: 读位置
            fb_after = driver.get_feedback(mid)
            pos_after = fb_after.position if fb_after and fb_after.is_valid else None
            if pos_after is not None and pos_before is not None:
                delta = pos_after - pos_before
                if abs(pos_after) < 0.05:
                    self.log_message.emit(f"  [4] 位置: {pos_after:.4f}  ✓ 已归零")
                else:
                    self.log_message.emit(f"  [4] 位置: {pos_after:.4f}  ✗ 未归零 (变化 {delta:+.4f})")

            # Step 5: 夹爪切回 PP 模式
            if is_gripper and mode_before == 1:
                self.log_message.emit(f"  [5] 失能电机 ...")
                driver.disable_motor(mid)
                time.sleep(0.05)

                self.log_message.emit(f"  [5] 切回 PP 模式 ...")
                driver.set_run_mode(mid, RunMode.POSITION_PP)
                time.sleep(0.05)

                self.log_message.emit(f"  [5] 重新使能 ...")
                driver.enable_motor(mid)
                time.sleep(0.1)

                mode_r3 = self.arm.ReadMotorParameter(mid, ParamIndex.RUN_MODE)
                mode_final = mode_r3.value_uint8 if mode_r3 and mode_r3.success else "读取失败"
                self.log_message.emit(f"  [5] 最终模式: {mode_final}")

            self.log_message.emit(f"===== 电机 {mid} 设零完成 =====")

    def _do_verify_zero_sta(self):
        from el_a3_sdk.protocol import ParamIndex
        results = []
        if not self.arm:
            return
        self.log_message.emit("开始校验 ZERO_STA 参数...")
        all_ok = True
        for mid in range(1, 8):
            result = self.arm.ReadMotorParameter(mid, ParamIndex.ZERO_STA)
            if result and result.success:
                val = result.value_uint8
                ok = (val == 1)
                results.append((mid, val, True))
                status = "✓" if ok else "✗"
                raw_hex = result.raw_bytes.hex() if result.raw_bytes else ""
                self.log_message.emit(
                    f"  电机{mid} ZERO_STA = {val} {status}  (raw: {raw_hex})")
                if not ok:
                    all_ok = False
            else:
                results.append((mid, 0, False))
                self.log_message.emit(f"  电机{mid} ZERO_STA 读取失败")
                all_ok = False
        self.zero_sta_verified.emit(results)
        if all_ok:
            self.log_message.emit("ZERO_STA 校验完成: 全部通过")
        else:
            self.log_message.emit("ZERO_STA 校验完成: 存在异常")

    def _do_set_all_zero_sta(self):
        from el_a3_sdk.protocol import ParamIndex
        if self._sim_mode:
            self.log_message.emit("一键设置 ZERO_STA=1（模拟）")
            return
        if not self.arm or not self._connected:
            self.log_message.emit("未连接，无法设置 ZERO_STA")
            return
        self.log_message.emit("开始设置全部电机 ZERO_STA=1 ...")
        fail_count = 0
        for mid in range(1, 8):
            ok = self.arm.WriteMotorParameterInt(mid, ParamIndex.ZERO_STA, 1)
            if ok:
                self.log_message.emit(f"  电机{mid} ZERO_STA 已设置为 1")
            else:
                self.log_message.emit(f"  电机{mid} ZERO_STA 设置失败")
                fail_count += 1
        if fail_count == 0:
            self.log_message.emit("全部电机 ZERO_STA 设置完成，保存参数到 Flash ...")
            import time as _t
            _t.sleep(0.05)
            self.arm.SaveParameters(0xFF)
            self.log_message.emit("参数已保存")
        else:
            self.log_message.emit(
                f"ZERO_STA 设置完成: {fail_count} 个电机失败，跳过保存")
        self._do_verify_zero_sta()

    def _do_scan_motors(self):
        results = []
        if self._sim_mode:
            results = [(mid, True, "v1.0.0-sim", 24.0) for mid in range(1, 8)]
            self.motor_scan_result.emit(results)
            self.log_message.emit("电机扫描完成（模拟）: 7/7 在线")
            return
        if not self.arm:
            return
        self.log_message.emit("开始扫描电机...")
        online_count = 0
        for mid in range(1, 8):
            fw = self.arm.GetFirmwareVersion(mid)
            voltage = self.arm.GetMotorVoltage(mid)
            online = fw is not None or voltage is not None
            fw_str = fw.version_str if fw else ""
            if online:
                online_count += 1
                v_str = f" {voltage:.1f}V" if voltage is not None else ""
                self.log_message.emit(
                    f"  电机{mid}: 在线  固件={fw_str or '—'}{v_str}")
            else:
                self.log_message.emit(f"  电机{mid}: 离线")
            results.append((mid, online, fw_str, voltage))
        self.motor_scan_result.emit(results)
        self.log_message.emit(f"电机扫描完成: {online_count}/7 在线")

    def _do_read_param(self, motor_id, param_index):
        if self._sim_mode:
            self.log_message.emit(f"读取参数（模拟）motor={motor_id} param=0x{param_index:04X}")
            return
        if self.arm:
            result = self.arm.ReadMotorParameter(motor_id, param_index)
            self.log_message.emit(
                f"电机{motor_id} 参数0x{param_index:04X} = {result}"
            )

    def _do_write_param(self, motor_id, param_index, value):
        if self._sim_mode:
            self.log_message.emit(
                f"写入参数（模拟）motor={motor_id} param=0x{param_index:04X} val={value}"
            )
            return
        if self.arm:
            self.arm.WriteMotorParameter(motor_id, param_index, value)
            self.log_message.emit(
                f"电机{motor_id} 参数0x{param_index:04X} 已写入 {value}"
            )

    def _do_start_control_loop(self, rate_hz=200.0):
        if self._sim_mode:
            self.control_loop_changed.emit(True)
            self.log_message.emit(f"控制循环已启动（模拟）{rate_hz}Hz")
            return
        if self.arm:
            self.arm.start_control_loop(rate_hz=rate_hz)
            self.control_loop_changed.emit(True)
            self.log_message.emit(f"控制循环已启动 {rate_hz}Hz")

    def _do_stop_control_loop(self):
        if self._sim_mode:
            self.control_loop_changed.emit(False)
            self.log_message.emit("控制循环已停止（模拟）")
            return
        if self.arm:
            self.arm.stop_control_loop()
            self.control_loop_changed.emit(False)
            self.log_message.emit("控制循环已停止")

    def _poll_feedback(self):
        now = time.time()
        self._slow_poll_counter += 1
        do_slow_poll = (self._slow_poll_counter % 10 == 0)  # ~5Hz for heavy queries

        if self._sim_mode:
            self._update_sim_state()
            positions = list(self._sim_positions)
            velocities = list(self._sim_velocities)
            torques = list(self._sim_torques)
            temperatures = [25.0 + i * 0.5 for i in range(7)]
        else:
            try:
                joint_msg = self.arm.GetArmJointMsgs()
                vel_msg = self.arm.GetArmJointVelocities()
                eff_msg = self.arm.GetArmJointEfforts()

                positions = joint_msg.to_list()
                velocities = vel_msg.to_list()
                torques = eff_msg.to_list()
                temperatures = [0.0] * 7

                self.joints_updated.emit(joint_msg)
                self.velocities_updated.emit(vel_msg)
                self.efforts_updated.emit(eff_msg)

                if do_slow_poll:
                    try:
                        status = self.arm.GetArmStatus()
                        self.status_updated.emit(status)
                    except Exception:
                        pass

                    try:
                        end_pose = self.arm.GetArmEndPoseMsgs()
                        self.end_pose_updated.emit(end_pose)
                    except Exception:
                        pass

                    try:
                        fps = self.arm.GetCanFps()
                        self.can_fps_updated.emit(fps)
                    except Exception:
                        pass

                    motor_fb_list = []
                    try:
                        states = self.arm.GetMotorStates()
                        if states:
                            for mid in range(1, 8):
                                fb = states.get(mid)
                                if fb is not None:
                                    motor_fb_list.append(fb)
                                    if hasattr(fb, 'temperature'):
                                        temperatures[mid - 1] = fb.temperature
                    except Exception:
                        pass

                    if motor_fb_list:
                        self.motor_feedback_updated.emit(motor_fb_list)

            except Exception as e:
                logger.debug(f"Poll error: {e}")
                return

        self.data_buffer.append(
            now, positions, velocities, torques, temperatures
        )

        if self._sim_mode:
            from el_a3_sdk.data_types import ArmJointStates
            js = ArmJointStates.from_list(positions, timestamp=now)
            js.hz = self._poll_rate_hz
            self.joints_updated.emit(js)
            vs = ArmJointStates.from_list(velocities, timestamp=now)
            self.velocities_updated.emit(vs)
            es = ArmJointStates.from_list(torques, timestamp=now)
            self.efforts_updated.emit(es)
            self.can_fps_updated.emit(200.0)

    def _update_sim_state(self):
        """简单的一阶位置模拟"""
        alpha = 0.05
        for i in range(7):
            diff = self._sim_target[i] - self._sim_positions[i]
            self._sim_velocities[i] = diff * 2.0
            self._sim_positions[i] += diff * alpha
            self._sim_torques[i] = diff * 10.0
            if abs(diff) < 0.001:
                self._sim_velocities[i] = 0.0
                self._sim_torques[i] = 0.0
