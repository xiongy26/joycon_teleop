"""High-level EL-A3 motion manager."""

from __future__ import annotations

from typing import Optional, Sequence

import numpy as np

from .backend import ELA3Backend
from .deployment import prepare_trajectory_for_deployment
from .options import ELA3MotionOptions, FinishPolicy, StartPolicy
from .queue import build_sdk_queue_trajectory
from .safety import densify_arm_joint_plan, validate_arm_joint_plan
from .types import ELA3Trajectory, MotionExecutionResult


class ELA3MotionManager:
    """Deploy MuJoCo-developed EL-A3 joint motion to real hardware."""

    def __init__(
        self,
        options: Optional[ELA3MotionOptions] = None,
        *,
        backend: Optional[ELA3Backend] = None,
    ) -> None:
        self.options = options or ELA3MotionOptions()
        self._backend = backend

    @classmethod
    def from_project_config(cls, **overrides) -> "ELA3MotionManager":
        return cls(ELA3MotionOptions.from_project_config(**overrides))

    def execute_joint_plan(
        self,
        joint_plan: Sequence[Sequence[float] | np.ndarray],
        *,
        dt: Optional[float] = None,
        gripper_plan: Optional[Sequence[float]] = None,
        confirm_real: bool = False,
        dry_run: Optional[bool] = None,
        start_policy: StartPolicy = StartPolicy.BRIDGE_FROM_REAL,
        finish_policy: FinishPolicy = FinishPolicy.HOLD,
        metadata: Optional[dict] = None,
    ) -> MotionExecutionResult:
        trajectory = ELA3Trajectory.from_joint_plan(
            joint_plan,
            dt=dt if dt is not None else 1.0 / float(self.options.command_rate_hz),
            gripper_plan=gripper_plan,
            metadata=metadata,
        )
        return self.execute_trajectory(
            trajectory,
            confirm_real=confirm_real,
            dry_run=dry_run,
            start_policy=start_policy,
            finish_policy=finish_policy,
        )

    def execute_trajectory(
        self,
        trajectory: ELA3Trajectory,
        *,
        confirm_real: bool = False,
        dry_run: Optional[bool] = None,
        start_policy: StartPolicy = StartPolicy.BRIDGE_FROM_REAL,
        finish_policy: FinishPolicy = FinishPolicy.HOLD,
        _backend: Optional[ELA3Backend] = None,
        _start_q: Optional[np.ndarray] = None,
        _backend_owned: bool = False,
    ) -> MotionExecutionResult:
        effective_dry_run = self.options.dry_run if dry_run is None else bool(dry_run)
        start_policy = StartPolicy(start_policy)
        finish_policy = FinishPolicy(finish_policy)

        if not effective_dry_run and not confirm_real:
            raise RuntimeError("real EL-A3 execution requires confirm_real=True")

        backend = _backend or self._backend
        backend_owned = _backend_owned
        start_q = None if _start_q is None else np.asarray(_start_q, dtype=float)
        diagnostics = {
            "input_points": len(trajectory.arm),
            "dt": trajectory.dt,
            "start_policy": start_policy.value,
            "finish_policy": finish_policy.value,
        }
        if trajectory.gripper is not None:
            diagnostics["gripper_plan_received"] = True
            diagnostics["gripper_plan_status"] = "stored_only_not_executed"

        try:
            if not effective_dry_run and start_q is None:
                if backend is None:
                    backend = ELA3Backend(self.options)
                    backend_owned = True
                backend.connect_enable_and_start()
                start_q = backend.wait_arm_joints(self.options.feedback_timeout_s)

            lower_limits, upper_limits = self._joint_limits()
            prepared = prepare_trajectory_for_deployment(
                trajectory,
                options=self.options,
                start_q=start_q,
                start_policy=start_policy,
                dry_run=effective_dry_run,
                lower_limits=lower_limits,
                upper_limits=upper_limits,
            )
            diagnostics.update(prepared.diagnostics)

            if effective_dry_run:
                return MotionExecutionResult(
                    dry_run=True,
                    interrupted=False,
                    execution_mode="dry_run_sdk_queue",
                    sent_points=len(prepared.queue.sdk_points),
                    duration_s=prepared.queue.duration_s,
                    queue_waypoints=prepared.queue.queue_waypoints,
                    queue_points=prepared.queue.queue_points,
                    start_q=None,
                    first_target_q=trajectory.arm[0].tolist(),
                    final_q=prepared.dense_plan[-1].tolist(),
                    bridge_inserted=prepared.bridge_inserted,
                    diagnostics=diagnostics,
                )

            assert backend is not None
            exec_result = self._execute_backend_queue(
                backend,
                prepared.queue.sdk_points,
                trajectory=trajectory,
            )
            diagnostics.update(exec_result.diagnostics)

            if finish_policy == FinishPolicy.HOLD:
                backend.hold_until_interrupt(
                    target_q=prepared.dense_plan[-1],
                    log_interval_s=self.options.feedback_log_interval_s,
                )
            elif finish_policy == FinishPolicy.HOME_THEN_DISABLE:
                home_queue = self._build_home_sdk_queue(
                    prepared.dense_plan[-1],
                    sample_period_s=trajectory.dt,
                )
                home_exec_result = self._execute_backend_queue(
                    backend,
                    home_queue.sdk_points,
                    trajectory=None,
                )
                diagnostics["home_then_disable"] = True
                diagnostics["home_queue_waypoints"] = home_queue.queue_waypoints
                diagnostics["home_queue_points"] = home_queue.queue_points
                diagnostics["home_duration_s"] = home_exec_result.duration_s
            elif finish_policy == FinishPolicy.STAY_CONNECTED:
                backend_owned = False

            return MotionExecutionResult(
                dry_run=False,
                interrupted=exec_result.interrupted,
                execution_mode="sdk_trajectory_queue",
                sent_points=exec_result.sent_points + int(
                    diagnostics.get("home_queue_points", 0)
                ),
                duration_s=exec_result.duration_s + float(
                    diagnostics.get("home_duration_s", 0.0)
                ),
                queue_waypoints=prepared.queue.queue_waypoints,
                queue_points=prepared.queue.queue_points,
                start_q=start_q.tolist() if start_q is not None else None,
                first_target_q=trajectory.arm[0].tolist(),
                final_q=prepared.dense_plan[-1].tolist(),
                bridge_inserted=prepared.bridge_inserted,
                diagnostics=diagnostics,
            )
        finally:
            if not effective_dry_run and backend_owned and backend is not None:
                backend.close(disable=self.options.disable_on_finish)

    def execute_cartesian_path(
        self,
        points: Sequence[Sequence[float] | np.ndarray],
        *,
        ik_method: str = "velocity",
        speed_mode: str = "normal",
        gripper_plan: Optional[Sequence[float]] = None,
        confirm_real: bool = False,
        dry_run: Optional[bool] = None,
        start_policy: StartPolicy = StartPolicy.SEED_FROM_REAL,
        finish_policy: FinishPolicy = FinishPolicy.HOLD,
        control_orientation: bool = False,
        energy_reg: float = 0.0,
        dt: Optional[float] = None,
        metadata: Optional[dict] = None,
    ) -> MotionExecutionResult:
        path = self._normalize_cartesian_path(points)
        return self.execute_strokes(
            [path],
            ik_method=ik_method,
            speed_mode=speed_mode,
            gripper_plan=gripper_plan,
            confirm_real=confirm_real,
            dry_run=dry_run,
            start_policy=start_policy,
            finish_policy=finish_policy,
            control_orientation=control_orientation,
            energy_reg=energy_reg,
            dt=dt,
            metadata={
                **(metadata or {}),
                "source": "cartesian_path",
                "cartesian_points": len(path),
            },
        )

    def execute_strokes(
        self,
        strokes_list: Sequence[Sequence[Sequence[float] | np.ndarray] | np.ndarray],
        *,
        ik_method: str = "velocity",
        speed_mode: str = "normal",
        gripper_plan: Optional[Sequence[float]] = None,
        confirm_real: bool = False,
        dry_run: Optional[bool] = None,
        start_policy: StartPolicy = StartPolicy.SEED_FROM_REAL,
        finish_policy: FinishPolicy = FinishPolicy.HOLD,
        control_orientation: bool = False,
        energy_reg: float = 0.0,
        dt: Optional[float] = None,
        metadata: Optional[dict] = None,
    ) -> MotionExecutionResult:
        strokes = self._normalize_strokes(strokes_list)
        effective_dry_run = self.options.dry_run if dry_run is None else bool(dry_run)
        if not effective_dry_run and not confirm_real:
            raise RuntimeError("real EL-A3 execution requires confirm_real=True")

        backend = self._backend
        backend_owned = False
        seed_q = None
        try:
            if not effective_dry_run:
                if backend is None:
                    backend = ELA3Backend(self.options)
                    backend_owned = True
                backend.connect_enable_and_start()
                seed_q = backend.wait_arm_joints(self.options.feedback_timeout_s)

            plan = self._build_strokes_motion_plan(
                strokes,
                ik_method=ik_method,
                speed_mode=speed_mode,
                seed_q=seed_q,
                control_orientation=control_orientation,
                energy_reg=energy_reg,
                dt=dt if dt is not None else 1.0 / float(self.options.command_rate_hz),
            )
            trajectory = ELA3Trajectory.from_joint_plan(
                plan.joint_plan,
                dt=float(plan.sample_period_s),
                gripper_plan=gripper_plan,
                metadata={
                    **(metadata or {}),
                    "source": "strokes",
                    "ik_method": ik_method,
                    "speed_mode": speed_mode,
                    "stroke_count": len(strokes),
                    "stroke_points": sum(len(stroke) for stroke in strokes),
                    "visual_strokes": strokes,
                },
            )

            transfer_backend_owned = backend_owned
            backend_owned = False
            result = self.execute_trajectory(
                trajectory,
                confirm_real=confirm_real,
                dry_run=effective_dry_run,
                start_policy=start_policy,
                finish_policy=finish_policy,
                _backend=backend,
                _start_q=seed_q,
                _backend_owned=transfer_backend_owned,
            )
            result.diagnostics["stroke_count"] = len(strokes)
            result.diagnostics["stroke_points"] = sum(len(stroke) for stroke in strokes)
            result.diagnostics["strokes_ik_method"] = ik_method
            if metadata and metadata.get("source") == "cartesian_path":
                result.diagnostics["cartesian_points"] = int(metadata["cartesian_points"])
                result.diagnostics["cartesian_ik_method"] = ik_method
            return result
        finally:
            if not effective_dry_run and backend_owned and backend is not None:
                backend.close(disable=self.options.disable_on_finish)

    def connect(self) -> None:
        if self._backend is None:
            self._backend = ELA3Backend(self.options)
        self._backend.connect_enable_and_start()

    def read_real_joints(self) -> np.ndarray:
        if self._backend is None:
            raise RuntimeError("manager is not connected")
        return self._backend.wait_arm_joints(self.options.feedback_timeout_s)

    def close(self) -> None:
        if self._backend is not None:
            self._backend.close(disable=self.options.disable_on_finish)

    def _build_cartesian_motion_plan(
        self,
        path: np.ndarray,
        *,
        ik_method: str,
        speed_mode: str,
        seed_q: Optional[np.ndarray],
        control_orientation: bool,
        energy_reg: float,
        dt: float,
    ):
        return self._build_strokes_motion_plan(
            [path],
            ik_method=ik_method,
            speed_mode=speed_mode,
            seed_q=seed_q,
            control_orientation=control_orientation,
            energy_reg=energy_reg,
            dt=dt,
        )

    def _build_strokes_motion_plan(
        self,
        strokes: Sequence[np.ndarray],
        *,
        ik_method: str,
        speed_mode: str,
        seed_q: Optional[np.ndarray],
        control_orientation: bool,
        energy_reg: float,
        dt: float,
    ):
        from core.motion_planning import MotionPlanningOptions, build_motion_plan

        planning_options = MotionPlanningOptions(
            command_rate_hz=1.0 / float(dt),
            max_joint_step_rad=self.options.max_joint_step_rad,
            ik_tolerance_m=self.options.ik_tolerance_m,
            lift_height_m=self.options.lift_height_m,
        )
        return build_motion_plan(
            list(strokes),
            ik_method=ik_method,
            speed_mode=speed_mode,
            control_orientation=control_orientation,
            energy_reg=energy_reg,
            dt=dt,
            seed_q=seed_q,
            options=planning_options,
        )

    def _build_home_sdk_queue(
        self,
        final_q: np.ndarray,
        *,
        sample_period_s: float,
    ):
        home_plan = self._build_home_joint_plan(
            final_q,
            sample_period_s=sample_period_s,
        )
        if not home_plan:
            raise RuntimeError("home trajectory is empty")
        dense_home = densify_arm_joint_plan(
            home_plan,
            max_joint_step_rad=self.options.max_joint_step_rad,
        )
        lower, upper = self._joint_limits()
        validate_arm_joint_plan(
            dense_home,
            lower_limits=lower,
            upper_limits=upper,
            max_joint_step_rad=self.options.max_joint_step_rad,
        )
        return build_sdk_queue_trajectory(
            dense_home,
            sample_period_s=sample_period_s,
            control_period_s=1.0 / float(self.options.control_rate_hz),
            max_joint_velocity_rad_s=self.options.max_joint_velocity_rad_s,
            start_q=None,
        )

    def _build_home_joint_plan(
        self,
        final_q: np.ndarray,
        *,
        sample_period_s: float,
    ) -> list[np.ndarray]:
        from core.home_sequence import build_home_joint_plan

        return build_home_joint_plan(
            final_q,
            lift_height_m=self.options.lift_height_m,
            sample_period_s=sample_period_s,
        )

    def _execute_backend_queue(
        self,
        backend: ELA3Backend,
        sdk_points: list,
        *,
        trajectory: Optional[ELA3Trajectory],
    ) -> MotionExecutionResult:
        if not self.options.mujoco_viewer:
            return backend.execute_queue(
                sdk_points,
                block=True,
                feedback_log_interval_s=self.options.feedback_log_interval_s,
            )
        return self._execute_backend_queue_with_mujoco_mirror(
            backend,
            sdk_points,
            trajectory=trajectory,
        )

    def _execute_backend_queue_with_mujoco_mirror(
        self,
        backend: ELA3Backend,
        sdk_points: list,
        *,
        trajectory: Optional[ELA3Trajectory],
    ) -> MotionExecutionResult:
        import mujoco
        import mujoco.viewer

        from core.ik_core import data, model

        visual_strokes = None
        if trajectory is not None:
            visual_strokes = trajectory.metadata.get("visual_strokes")

        with mujoco.viewer.launch_passive(
            model=model,
            data=data,
            show_left_ui=False,
            show_right_ui=False,
        ) as viewer:
            mujoco.mjv_defaultFreeCamera(model, viewer.cam)
            callback = self._make_mujoco_mirror_callback(
                sdk_points,
                viewer=viewer,
                visual_strokes=visual_strokes,
            )
            callback(0.0)
            return backend.execute_queue(
                sdk_points,
                block=True,
                feedback_log_interval_s=self.options.feedback_log_interval_s,
                progress_callback=callback,
            )

    def _make_mujoco_mirror_callback(
        self,
        sdk_points: list,
        *,
        viewer,
        visual_strokes,
    ):
        import mujoco

        from core.dls_trajectory import GRIPPER_JOINT_MAX, GRIPPER_JOINT_MIN
        from core.ik_core import (
            data,
            ee_site_id,
            gripper_qpos_index,
            lock,
            model,
            set_arm_q_vector,
        )
        from core.sim_motion_executor import _split_path_by_z
        from core.trajectory_visualization import render_trajectory_scene

        traj_idx = 0
        executed_path: list[np.ndarray] = []
        paper_z = None
        if visual_strokes:
            for stroke in visual_strokes:
                if len(stroke) > 0:
                    paper_z = float(stroke[0][2])
                    break

        def sync(elapsed_s: float) -> bool:
            nonlocal traj_idx
            if not viewer.is_running():
                return False
            while (
                traj_idx < len(sdk_points) - 1
                and sdk_points[traj_idx + 1].time <= elapsed_s
            ):
                traj_idx += 1
            q_arr = np.asarray(sdk_points[traj_idx].positions[:6], dtype=float)
            with lock:
                set_arm_q_vector(q_arr)
                data.qpos[gripper_qpos_index] = np.clip(
                    data.qpos[gripper_qpos_index],
                    GRIPPER_JOINT_MIN,
                    GRIPPER_JOINT_MAX,
                )
                mujoco.mj_forward(model, data)
                ee_pos = data.site_xpos[ee_site_id].copy()
            if (
                not executed_path
                or np.linalg.norm(ee_pos - executed_path[-1]) > 1e-5
            ):
                executed_path.append(ee_pos)
            mujoco.mj_camlight(model, data)
            if paper_z is not None:
                writing_segs, lift_segs = _split_path_by_z(executed_path, paper_z)
            else:
                writing_segs = [executed_path] if executed_path else []
                lift_segs = []
            render_trajectory_scene(
                viewer.user_scn,
                preview_strokes=visual_strokes,
                writing_strokes=writing_segs if writing_segs else None,
                lift_strokes=lift_segs if lift_segs else None,
                writing_rgba=np.array([1.0, 0.0, 0.0, 0.8]),
                lift_rgba=np.array([0.7, 0.5, 1.0, 0.25]),
            )
            viewer.sync()
            return True

        return sync

    @staticmethod
    def _normalize_cartesian_path(
        points: Sequence[Sequence[float] | np.ndarray],
    ) -> np.ndarray:
        if not points:
            raise ValueError("cartesian path is empty")
        path = np.asarray(points, dtype=float)
        if path.ndim != 2 or path.shape[1] != 3:
            raise ValueError(f"cartesian path must have shape N x 3, got {path.shape}")
        if np.any(~np.isfinite(path)):
            raise ValueError("cartesian path contains NaN/Inf")
        return path.copy()

    @staticmethod
    def _normalize_strokes(
        strokes_list: Sequence[Sequence[Sequence[float] | np.ndarray] | np.ndarray],
    ) -> list[np.ndarray]:
        if not strokes_list:
            raise ValueError("strokes_list is empty")
        strokes: list[np.ndarray] = []
        for idx, stroke in enumerate(strokes_list):
            arr = np.asarray(stroke, dtype=float)
            if arr.ndim != 2 or arr.shape[1] != 3:
                raise ValueError(
                    f"stroke {idx} must have shape N x 3, got {arr.shape}"
                )
            if len(arr) == 0:
                raise ValueError(f"stroke {idx} is empty")
            if np.any(~np.isfinite(arr)):
                raise ValueError(f"stroke {idx} contains NaN/Inf")
            strokes.append(arr.copy())
        return strokes

    @staticmethod
    def _joint_limits() -> tuple[np.ndarray, np.ndarray]:
        from core.ik_core import ARM_JOINT_LIMITS_LOWER, ARM_JOINT_LIMITS_UPPER

        return ARM_JOINT_LIMITS_LOWER, ARM_JOINT_LIMITS_UPPER
