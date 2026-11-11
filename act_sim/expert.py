"""Scripted IK expert that generates pick-and-lift demonstrations."""

from __future__ import annotations

from dataclasses import dataclass

import mujoco
import numpy as np

from act_sim.env import CUBE_Z


def solve_ik(
    model,
    data,
    site_id: int,
    target_xyz: np.ndarray,
    *,
    q_seed: np.ndarray,
    arm_qpos_adr: np.ndarray,
    arm_dof_adr: np.ndarray,
    arm_jnt_range: np.ndarray,
    tolerance: float = 2e-3,
    max_iters: int = 400,
    step_scale: float = 0.4,
) -> tuple[np.ndarray, bool, float]:
    """Damped-least-squares position IK for the gripper tip site.

    Uses a scratch MjData so it never disturbs live simulation state.
    ``arm_qpos_adr``/``arm_dof_adr``/``arm_jnt_range`` come from
    ELA3ReachEnv (resolved by joint name). Returns
    (qpos, converged, final_position_error).
    """

    arm_qpos_adr = np.asarray(arm_qpos_adr, dtype=int)
    arm_dof_adr = np.asarray(arm_dof_adr, dtype=int)
    lower = arm_jnt_range[:, 0]
    upper = arm_jnt_range[:, 1]
    scratch = mujoco.MjData(model) if data is None else data
    q = np.clip(np.asarray(q_seed, dtype=np.float64).copy(), lower, upper)
    target = np.asarray(target_xyz, dtype=np.float64)

    err_norm = float("inf")
    jac = np.zeros((3, model.nv))
    for _ in range(max_iters):
        scratch.qpos[arm_qpos_adr] = q
        mujoco.mj_forward(model, scratch)  # mj_forward updates cdof for the jacobian
        err = target - scratch.site(site_id).xpos
        err_norm = float(np.linalg.norm(err))
        if err_norm < tolerance:
            return q, True, err_norm
        mujoco.mj_jacSite(model, scratch, jac, None, site_id)
        jac_arm = jac[:, arm_dof_adr]
        dq = jac_arm.T @ np.linalg.solve(jac_arm @ jac_arm.T + 1e-6 * np.eye(3), err)
        scale = min(1.0, step_scale / max(float(np.abs(dq).max()), 1e-9))
        q = np.clip(q + scale * dq, lower, upper)
    scratch.qpos[arm_qpos_adr] = q
    mujoco.mj_forward(model, scratch)
    err_norm = float(np.linalg.norm(target - scratch.site(site_id).xpos))
    return q, err_norm < tolerance, err_norm


def min_jerk_trajectory(start: np.ndarray, end: np.ndarray, duration: float, rate_hz: float) -> np.ndarray:
    """Joint-space minimum-jerk interpolation sampled at rate_hz (includes endpoints)."""

    steps = max(2, int(round(duration * rate_hz)) + 1)
    t = np.linspace(0.0, 1.0, steps)
    s = 10 * t**3 - 15 * t**4 + 6 * t**5
    return start[None, :] + s[:, None] * (end - start)[None, :]


@dataclass
class ExpertPlan:
    """Reference joint trajectory (one row per control step)."""

    refs: np.ndarray  # [T, 7]

    @property
    def num_steps(self) -> int:
        return int(self.refs.shape[0])


class ReachExpert:
    """Plans an open -> approach -> close -> lift grasp trajectory."""

    def __init__(
        self,
        env,
        *,
        rate_hz: float = 20.0,
        hover_height: float = 0.17,
        touch_height: float = 0.025,
        lift_height: float = 0.17,
        max_joint_speed: float = 0.45,
        min_segment_duration: float = 0.9,
        close_steps: int = 16,
        lift_hold_steps: int = 16,
        open_gripper: float = -1.35,
        close_gripper: float = -0.60,
    ):
        self.env = env
        self.model = env.model
        self.site_id = int(env._site_id)
        self.arm_qpos_adr = env.arm_qpos_adr
        self.arm_dof_adr = env.arm_dof_adr
        self.arm_jnt_range = env.arm_jnt_range
        self.cube_qpos_adr = env.cube_qpos_adr
        self.rate_hz = float(rate_hz)
        self.hover_height = float(hover_height)
        self.touch_height = float(touch_height)
        self.lift_height = float(lift_height)
        self.max_joint_speed = float(max_joint_speed)
        self.min_segment_duration = float(min_segment_duration)
        self.close_steps = int(close_steps)
        self.lift_hold_steps = int(lift_hold_steps)
        self.open_gripper = float(open_gripper)
        self.close_gripper = float(close_gripper)
        self._scratch = None

    def _scratch_data(self):
        if self._scratch is None:
            self._scratch = mujoco.MjData(self.model)
            # keep the cube free joint valid in the scratch state
            self._scratch.qpos[self.cube_qpos_adr + 3] = 1.0
        return self._scratch

    def _ik(self, target_xyz: np.ndarray, q_seed: np.ndarray) -> np.ndarray:
        q, ok, err = solve_ik(
            self.model,
            self._scratch_data(),
            self.site_id,
            target_xyz,
            q_seed=q_seed,
            arm_qpos_adr=self.arm_qpos_adr,
            arm_dof_adr=self.arm_dof_adr,
            arm_jnt_range=self.arm_jnt_range,
        )
        if not ok:
            raise RuntimeError(
                f"IK failed for target {np.round(target_xyz, 3).tolist()} (err={err:.4f})"
            )
        return q

    def _segment(self, start: np.ndarray, end: np.ndarray) -> np.ndarray:
        duration = max(
            self.min_segment_duration,
            float(np.abs(end - start).max()) / self.max_joint_speed,
        )
        return min_jerk_trajectory(start, end, duration, self.rate_hz)

    def plan(self, cube_xyz: np.ndarray, *, start_qpos: np.ndarray) -> ExpertPlan:
        cube = np.asarray(cube_xyz, dtype=np.float64)[:3]
        hover_target = np.array([cube[0], cube[1], CUBE_Z + self.hover_height])
        touch_target = np.array([cube[0], cube[1], CUBE_Z + self.touch_height])
        lift_target = np.array([cube[0], cube[1], CUBE_Z + self.lift_height])

        q_start = np.asarray(start_qpos, dtype=np.float64).copy()
        q_open = q_start.copy()
        q_open[6] = self.open_gripper
        q_hover = self._ik(hover_target, q_seed=q_start)
        q_hover[6] = self.open_gripper
        q_touch = self._ik(touch_target, q_seed=q_hover)
        q_touch[6] = self.open_gripper
        q_close = q_touch.copy()
        q_close[6] = self.close_gripper
        q_lift = self._ik(lift_target, q_seed=q_touch)
        q_lift[6] = self.close_gripper

        refs = [self._segment(q_start, q_open)]
        refs.append(self._segment(q_open, q_hover))
        refs.append(self._segment(q_hover, q_touch))
        refs.append(self._segment(q_touch, q_close))
        refs.append(np.tile(q_close, (self.close_steps, 1)))
        refs.append(self._segment(q_close, q_lift))
        refs.append(np.tile(q_lift, (self.lift_hold_steps, 1)))
        return ExpertPlan(refs=np.vstack(refs))
