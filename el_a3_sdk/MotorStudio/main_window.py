"""主窗口：QDockWidget 布局 + 信号连接"""

import time
import logging
from PyQt6.QtWidgets import (
    QMainWindow, QDockWidget, QTabWidget,
    QWidget, QVBoxLayout, QTextEdit, QApplication,
)
from PyQt6.QtCore import Qt, QTimer

from MotorStudio.backend.arm_worker import ArmWorker
from MotorStudio.widgets.toolbar_panel import ToolbarPanel
from MotorStudio.widgets.joint_control_panel import JointControlPanel
from MotorStudio.widgets.monitoring_window import MonitoringWindow
from MotorStudio.widgets.trajectory_panel import TrajectoryPanel
from MotorStudio.widgets.teaching_panel import TeachingPanel
from MotorStudio.widgets.diagnostics_panel import DiagnosticsPanel
from MotorStudio.widgets.gripper_panel import GripperPanel
from MotorStudio.widgets.gamepad_panel import GamepadPanel
from MotorStudio.widgets.viewer_3d import Viewer3DPanel
from MotorStudio.utils.i18n import tr
from MotorStudio.utils.theme_manager import ThemeManager

logger = logging.getLogger("MotorStudio")


class MainWindow(QMainWindow):
    """EL-A3 调试上位机主窗口"""

    UI_UPDATE_INTERVAL_S = 0.05  # 20 Hz UI refresh cap

    def __init__(self, urdf_path=None, mesh_dir=None):
        super().__init__()
        self.setWindowTitle(tr("win.title"))
        self.setMinimumSize(1280, 800)
        self.resize(1600, 960)

        self._urdf_path = urdf_path
        self._mesh_dir = mesh_dir
        self._last_joint_states = None
        self._last_effort_states = None
        self._last_ui_update_time = 0.0

        self._init_worker()
        self._init_ui()
        self._connect_signals()

        tm = ThemeManager.instance()
        tm.language_changed.connect(lambda _: self._retranslate_ui())
        tm.theme_changed.connect(lambda _: self.viewer_3d.apply_theme())
        tm.theme_changed.connect(
            lambda _: self.monitoring_window.panel.apply_theme()
        )
        tm.theme_changed.connect(lambda _: self.diagnostics_panel.retranslate_ui())
        tm.theme_changed.connect(lambda _: self.gripper_panel.retranslate_ui())
        tm.theme_changed.connect(lambda _: self.gamepad_panel.retranslate_ui())

        QTimer.singleShot(500, self._init_3d_model)

    def _init_worker(self):
        self.worker = ArmWorker()
        self.worker.start()

    def _init_ui(self):
        # --- 顶部工具栏（单行固定高度） ---
        toolbar_widget = ToolbarPanel()
        self.toolbar = toolbar_widget
        self.toolbar_dock = QDockWidget(tr("win.toolbar"), self)
        self.toolbar_dock.setWidget(toolbar_widget)
        self.toolbar_dock.setFeatures(QDockWidget.DockWidgetFeature.NoDockWidgetFeatures)
        empty_title = QWidget()
        empty_title.setFixedHeight(0)
        self.toolbar_dock.setTitleBarWidget(empty_title)
        self.toolbar_dock.setStyleSheet("QDockWidget { border: none; }")
        self.addDockWidget(Qt.DockWidgetArea.TopDockWidgetArea, self.toolbar_dock)

        # --- 左侧：3D 可视化 ---
        self.viewer_3d = Viewer3DPanel(
            urdf_path=self._urdf_path,
            mesh_dir=self._mesh_dir,
        )
        self.viewer_dock = QDockWidget(tr("win.viewer"), self)
        self.viewer_dock.setWidget(self.viewer_3d)
        self.viewer_dock.setFeatures(QDockWidget.DockWidgetFeature.NoDockWidgetFeatures)
        _hide = QWidget(); _hide.setFixedHeight(0)
        self.viewer_dock.setTitleBarWidget(_hide)
        self.addDockWidget(Qt.DockWidgetArea.LeftDockWidgetArea, self.viewer_dock)

        # --- 右侧：功能面板 Tab ---
        self.tabs = QTabWidget()

        self.joint_panel = JointControlPanel()
        self.tabs.addTab(self.joint_panel, tr("tab.joint"))

        self.trajectory_panel = TrajectoryPanel()
        self.tabs.addTab(self.trajectory_panel, tr("tab.trajectory"))

        self.teaching_panel = TeachingPanel()
        self.tabs.addTab(self.teaching_panel, tr("tab.teaching"))

        self.diagnostics_panel = DiagnosticsPanel()
        self.tabs.addTab(self.diagnostics_panel, tr("tab.diagnostics"))

        self.gripper_panel = GripperPanel()
        self.tabs.addTab(self.gripper_panel, tr("tab.gripper"))

        self.gamepad_panel = GamepadPanel()
        self.tabs.addTab(self.gamepad_panel, tr("tab.gamepad"))

        self.tabs_dock = QDockWidget(tr("win.panels"), self)
        self.tabs_dock.setWidget(self.tabs)
        self.tabs_dock.setFeatures(QDockWidget.DockWidgetFeature.NoDockWidgetFeatures)
        _hide2 = QWidget(); _hide2.setFixedHeight(0)
        self.tabs_dock.setTitleBarWidget(_hide2)
        self.addDockWidget(Qt.DockWidgetArea.RightDockWidgetArea, self.tabs_dock)

        # --- 底部日志 ---
        self.log_console = QTextEdit()
        self.log_console.setObjectName("logConsole")
        self.log_console.setReadOnly(True)
        self.log_console.setFixedHeight(100)
        self.log_dock = QDockWidget(tr("win.log"), self)
        self.log_dock.setWidget(self.log_console)
        self.log_dock.setFeatures(QDockWidget.DockWidgetFeature.NoDockWidgetFeatures)
        _hide3 = QWidget(); _hide3.setFixedHeight(0)
        self.log_dock.setTitleBarWidget(_hide3)
        self.addDockWidget(Qt.DockWidgetArea.BottomDockWidgetArea, self.log_dock)

        # --- 实时监控弹出窗口（按需打开） ---
        self.monitoring_window = MonitoringWindow(self.worker.data_buffer, parent=self)

        self.statusBar().showMessage(tr("win.ready"))

        QTimer.singleShot(0, lambda: self._adjust_dock_sizes(self.viewer_dock, self.tabs_dock))

    def _connect_signals(self):
        tb = self.toolbar
        tb.connect_requested.connect(self._on_connect)
        tb.disconnect_requested.connect(lambda: self.worker.submit_command("disconnect"))
        tb.enable_requested.connect(lambda: self.worker.submit_command("enable"))
        tb.disable_requested.connect(lambda: self.worker.submit_command("disable"))
        tb.emergency_stop_requested.connect(
            lambda: self.worker.submit_command("emergency_stop")
        )
        tb.open_monitor_requested.connect(self._open_monitoring)

        self.worker.connected_changed.connect(tb.set_connected)
        self.worker.enabled_changed.connect(tb.set_enabled)
        self.worker.enabled_changed.connect(self.joint_panel.set_enabled)
        self.worker.enabled_changed.connect(self.viewer_3d.set_enabled)
        self.worker.error_occurred.connect(self._on_error)
        self.worker.log_message.connect(self._append_log)
        self.worker.can_fps_updated.connect(tb.set_fps)
        self.worker.can_fps_updated.connect(self.diagnostics_panel.update_can_stats)

        self.worker.joints_updated.connect(self._on_joints_updated)
        self.worker.efforts_updated.connect(self._on_efforts_updated)
        self.worker.motor_feedback_updated.connect(
            self.diagnostics_panel.update_motor_feedback
        )

        self.joint_panel.joint_command.connect(
            lambda pos, dur: self.worker.submit_command("move_j", pos, dur)
        )

        tp = self.trajectory_panel
        tp.move_j_requested.connect(
            lambda pos, dur: self.worker.submit_command("move_j", pos, dur)
        )
        tp.end_pose_requested.connect(
            lambda x, y, z, rx, ry, rz, d: self.worker.submit_command(
                "end_pose_ctrl", x, y, z, rx, ry, rz, d
            )
        )
        tp.cancel_requested.connect(
            lambda: self.worker.submit_command("cancel_motion")
        )

        teach = self.teaching_panel
        teach.zero_torque_requested.connect(
            lambda en: self.worker.submit_command("zero_torque", en)
        )
        teach.zero_torque_gravity_requested.connect(
            lambda en: self.worker.submit_command("zero_torque_gravity", en)
        )
        teach.move_j_requested.connect(
            lambda pos, dur: self.worker.submit_command("move_j", pos, dur)
        )

        gp = self.gripper_panel
        gp.gripper_command.connect(
            lambda angle: self.worker.submit_command("gripper_ctrl", angle)
        )
        gp.set_zero_requested.connect(
            lambda: self.worker.submit_command("set_zero_position", 7)
        )

        diag = self.diagnostics_panel
        diag.read_param_requested.connect(
            lambda mid, pidx: self.worker.submit_command("read_motor_param", mid, pidx)
        )
        diag.write_param_requested.connect(
            lambda mid, pidx, val: self.worker.submit_command(
                "write_motor_param", mid, pidx, val
            )
        )
        diag.set_zero_requested.connect(
            lambda m: self.worker.submit_command("set_zero_position", m)
        )
        diag.verify_zero_sta_requested.connect(
            lambda: self.worker.submit_command("verify_zero_sta")
        )
        diag.set_all_zero_sta_requested.connect(
            lambda: self.worker.submit_command("set_all_zero_sta")
        )
        self.worker.zero_sta_verified.connect(diag.update_zero_sta_result)
        diag.scan_motors_requested.connect(
            lambda: self.worker.submit_command("scan_motors")
        )
        self.worker.motor_scan_result.connect(diag.update_scan_result)

        tp.drag_mode_toggled.connect(self.viewer_3d.set_drag_mode)
        tp.sync_feedback_requested.connect(self.viewer_3d.sync_to_feedback)
        self.viewer_3d.drag_angles_changed.connect(tp.update_drag_angles)

        self.viewer_3d.home_position_requested.connect(
            lambda: self.worker.submit_command("move_j", [0.0] * 6, 3.0)
        )

        # --- Calibration panel ---
        calib = self.teaching_panel.calibration_panel
        calib.move_j_requested.connect(
            lambda pos, dur, block: self.worker.submit_command(
                "move_j_block" if block else "move_j", pos, dur
            )
        )
        self.worker.move_j_done.connect(calib.notify_move_done)

        self.gamepad_panel.gamepad_log.connect(self._append_log)
        self.worker.connected_changed.connect(self._on_connected_for_gamepad)

    # ---- retranslate ----

    def _retranslate_ui(self):
        self.setWindowTitle(tr("win.title"))
        self.toolbar_dock.setWindowTitle(tr("win.toolbar"))
        self.viewer_dock.setWindowTitle(tr("win.viewer"))
        self.tabs_dock.setWindowTitle(tr("win.panels"))
        self.log_dock.setWindowTitle(tr("win.log"))
        self.statusBar().showMessage(tr("win.ready"))

        self.tabs.setTabText(0, tr("tab.joint"))
        self.tabs.setTabText(1, tr("tab.trajectory"))
        self.tabs.setTabText(2, tr("tab.teaching"))
        self.tabs.setTabText(3, tr("tab.diagnostics"))
        self.tabs.setTabText(4, tr("tab.gripper"))
        self.tabs.setTabText(5, tr("tab.gamepad"))

        for panel in (self.toolbar, self.joint_panel, self.trajectory_panel,
                      self.teaching_panel, self.diagnostics_panel,
                      self.gripper_panel, self.gamepad_panel, self.viewer_3d):
            if hasattr(panel, "retranslate_ui"):
                panel.retranslate_ui()
        self.monitoring_window.retranslate_ui()

    # ---- helpers ----

    def _adjust_dock_sizes(self, viewer_dock, tabs_dock):
        w = self.width()
        left_w = int(w * 0.50)
        right_w = w - left_w
        self.resizeDocks(
            [viewer_dock, tabs_dock], [left_w, right_w], Qt.Orientation.Horizontal
        )

    def _open_monitoring(self):
        mw = self.monitoring_window
        if mw.isVisible():
            mw.raise_()
            mw.activateWindow()
        else:
            mw.show()
            mw.raise_()

    def _on_connected_for_gamepad(self, connected: bool):
        if connected and self.worker.arm is not None:
            self.gamepad_panel.set_arm(self.worker.arm)
        elif not connected:
            self.gamepad_panel.set_arm(None)

    def _on_connect(self, can_name: str, connect_kwargs: dict):
        self.worker.submit_command("connect", can_name, **connect_kwargs)

    def _on_joints_updated(self, joint_states):
        self._last_joint_states = joint_states
        self.teaching_panel.update_positions(joint_states)
        self.trajectory_panel.update_current_positions(joint_states)

        positions = joint_states.to_list(include_gripper=False)
        self.teaching_panel.calibration_panel.feed_positions(positions)

        now = time.monotonic()
        if now - self._last_ui_update_time < self.UI_UPDATE_INTERVAL_S:
            return
        self._last_ui_update_time = now

        self.joint_panel.update_feedback(joint_states)
        self.viewer_3d.update_joint_angles(joint_states)
        self.gripper_panel.update_feedback(joint_states, self._last_effort_states)
        self.diagnostics_panel.update_motor_states(
            joint_states, None, self._last_effort_states
        )

    def _on_efforts_updated(self, effort_states):
        self._last_effort_states = effort_states
        efforts = effort_states.to_list(include_gripper=False)
        self.teaching_panel.calibration_panel.feed_efforts(efforts)

    def _on_error(self, msg: str):
        self.toolbar.set_error(msg)
        self._append_log(tr("win.error", msg=msg))

    def _append_log(self, msg: str):
        timestamp = time.strftime("%H:%M:%S")
        self.log_console.append(f"[{timestamp}] {msg}")
        scrollbar = self.log_console.verticalScrollBar()
        scrollbar.setValue(scrollbar.maximum())

    def _init_3d_model(self):
        success = self.viewer_3d.initialize_model()
        if success:
            self._append_log(tr("win.model_ok"))
        else:
            self._append_log(tr("win.model_fail"))

    def closeEvent(self, event):
        self._append_log(tr("win.closing"))
        self.gamepad_panel.cleanup()
        if self.monitoring_window.isVisible():
            self.monitoring_window.close()
        if self.worker.is_connected:
            self.worker.submit_command("disconnect")
        self.worker.stop()
        super().closeEvent(event)
