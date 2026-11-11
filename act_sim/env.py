"""MuJoCo sandbox for the EL-A3 ACT pick-and-lift task.

The red cube spawns on a table. Demonstrations open the two-finger gripper,
approach the cube, close the gripper, and lift it. The small grasp attachment
keeps the free body stable once finger contact is detected; this makes the
demonstration deterministic while preserving normal MuJoCo contact dynamics
for approach and release.
"""

from __future__ import annotations

import os

# Offscreen rendering backend must be selected before mujoco is imported.
# EGL works headless; override by exporting MUJOCO_GL before running.
os.environ.setdefault("MUJOCO_GL", "egl")

from pathlib import Path

import numpy as np

TASK_XML_PATH = Path(__file__).resolve().parent / "el_a3_pick_scene.xml"

ARM_JOINT_NAMES = ["L1_joint", "L2_joint", "L3_joint", "L4_joint", "L5_joint", "L6_joint"]
GRIPPER_JOINT_NAME = "L7_joint"
EE_SITE_NAME = "gripper_tip_site"
CUBE_BODY_NAME = "cube"
CAMERA_NAMES = ["cam_high", "cam_wrist"]

# Start pose: gripper tip hovering inside the cam_high view above the workspace.
HOME_QPOS = np.array([1.04, 1.355, -1.15, 0.243, -1.096, 0.0, 0.0])

# Cube spawn window on the table (verified reachable by the IK expert).
CUBE_X_RANGE = (-0.10, 0.10)
CUBE_Y_RANGE = (-0.42, -0.26)
CUBE_Z = 0.102  # table top (0.08) + cube half size (0.02) + 2 mm clearance

SUCCESS_DISTANCE = 0.035  # meters, gripper tip site to cube center
GRASP_DISTANCE = 0.045  # meters, contact gate before attaching the cube
GRASP_CLOSE_TARGET = -1.0  # L7 <= this means the fingers are closing
GRASP_LIFT_HEIGHT = 0.04  # meters above the settled table pose


def sample_cube_position(rng: np.random.Generator) -> np.ndarray:
    x = rng.uniform(*CUBE_X_RANGE)
    y = rng.uniform(*CUBE_Y_RANGE)
    yaw = rng.uniform(-np.pi, np.pi)
    quat = np.array([np.cos(yaw / 2), 0.0, 0.0, np.sin(yaw / 2)])
    return np.array([x, y, CUBE_Z, *quat])


class ELA3ReachEnv:
    """Position-controlled EL-A3 pick task with two task cameras.

    Control rate is ``control_hz``; every control step advances
    ``round(model_timestep_interval)`` physics substeps so episode data is
    time-consistent with the recorded sample rate.
    """

    def __init__(
        self,
        *,
        image_size: int = 128,
        control_hz: float = 20.0,
    ):
        import mujoco

        if image_size <= 0:
            raise ValueError("image_size must be positive")
        if control_hz <= 0:
            raise ValueError("control_hz must be positive")

        self._mujoco = mujoco
        self.model = mujoco.MjModel.from_xml_path(TASK_XML_PATH.as_posix())
        self.data = mujoco.MjData(self.model)

        self.control_hz = float(control_hz)
        # physics substeps per control step (control period / model timestep)
        self.substeps = max(1, int(round(1.0 / control_hz / self.model.opt.timestep)))

        self._site_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_SITE, EE_SITE_NAME)
        self._cube_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_BODY, CUBE_BODY_NAME)
        self._cam_ids = [
            mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_CAMERA, name)
            for name in CAMERA_NAMES
        ]
        if min(self._site_id, self._cube_id, *self._cam_ids) < 0:
            raise RuntimeError("task XML is missing expected site/body/camera names")

        # Resolve controlled-joint addresses by name: XML element order must not matter.
        joint_name_to_id = {
            self.model.joint(i).name: i for i in range(self.model.njnt)
        }
        controlled_joint_names = ARM_JOINT_NAMES + [GRIPPER_JOINT_NAME]
        arm_joint_ids = [joint_name_to_id[name] for name in controlled_joint_names]
        self.arm_qpos_adr = np.array(
            [self.model.jnt_qposadr[i] for i in arm_joint_ids], dtype=int
        )
        self.arm_dof_adr = np.array(
            [self.model.jnt_dofadr[i] for i in arm_joint_ids], dtype=int
        )
        self.arm_jnt_range = self.model.jnt_range[arm_joint_ids].copy()
        cube_jnt_id = joint_name_to_id["cube_joint"]
        self.cube_qpos_adr = int(self.model.jnt_qposadr[cube_jnt_id])
        self.cube_dof_adr = int(self.model.jnt_dofadr[cube_jnt_id])

        self.qpos_limit_lower = self.arm_jnt_range[:, 0].copy()
        self.qpos_limit_upper = self.arm_jnt_range[:, 1].copy()

        self._renderers = None
        self._image_size = int(image_size)
        self._grasped = False
        self._grasp_offset = np.zeros(3, dtype=np.float64)
        self._cube_start_z = CUBE_Z

    # -- state helpers -------------------------------------------------

    @property
    def qpos(self) -> np.ndarray:
        return self.data.qpos[self.arm_qpos_adr].copy()

    @property
    def sim_time(self) -> float:
        return float(self.data.time)

    @property
    def tip_position(self) -> np.ndarray:
        return np.asarray(self.data.site(self._site_id).xpos, dtype=np.float64).copy()

    @property
    def cube_position(self) -> np.ndarray:
        return np.asarray(self.data.body(self._cube_id).xpos, dtype=np.float64).copy()

    @property
    def tip_cube_distance(self) -> float:
        return float(np.linalg.norm(self.tip_position - self.cube_position))

    @property
    def grasped(self) -> bool:
        """Whether the scripted/learned controller currently holds the cube."""

        return bool(self._grasped)

    @property
    def cube_lift_height(self) -> float:
        """Cube height above the settled table reference."""

        return float(self.cube_position[2] - self._cube_start_z)

    def _maybe_grasp(self) -> None:
        """Attach the cube after a closed gripper reaches it."""

        if self._grasped or float(self.data.ctrl[6]) > GRASP_CLOSE_TARGET:
            return
        if self.tip_cube_distance > GRASP_DISTANCE:
            return
        mujoco = self._mujoco
        cube_geom_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_GEOM, "cube_geom")
        finger_ids = {
            mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_GEOM, name)
            for name in ("finger_left_visual", "finger_right_visual")
        }
        touching = any(
            cube_geom_id in {self.data.contact[i].geom1, self.data.contact[i].geom2}
            and bool({self.data.contact[i].geom1, self.data.contact[i].geom2} & finger_ids)
            for i in range(int(self.data.ncon))
        )
        if not touching:
            return
        self._grasped = True
        self._grasp_offset = self.cube_position - self.tip_position

    def _carry_cube(self) -> None:
        if not self._grasped:
            return
        start = self.cube_qpos_adr
        self.data.qpos[start : start + 3] = self.tip_position + self._grasp_offset
        self.data.qvel[self.cube_dof_adr : self.cube_dof_adr + 6] = 0.0
        self._mujoco.mj_forward(self.model, self.data)

    # -- simulation ----------------------------------------------------

    def set_ctrl(
        self,
        qpos_target: np.ndarray,
        *,
        max_delta: float | None = None,
    ) -> np.ndarray:
        """Set a checked position target and return the applied target."""

        target = np.asarray(qpos_target, dtype=np.float64)
        if target.shape != (self.model.nu,):
            raise ValueError(
                f"qpos_target must have shape ({self.model.nu},), got {target.shape}"
            )
        if np.any(~np.isfinite(target)):
            raise ValueError("qpos_target contains NaN/Inf")
        if max_delta is not None:
            if max_delta < 0:
                raise ValueError("max_delta must be non-negative")
            current = self.qpos
            target = current + np.clip(target - current, -max_delta, max_delta)
        target = np.clip(target, self.qpos_limit_lower, self.qpos_limit_upper)
        if self._grasped and target[6] <= -1.0:
            self._grasped = False
        self.data.ctrl[:] = target
        return target.copy()

    def step(self) -> None:
        for _ in range(self.substeps):
            self._mujoco.mj_step(self.model, self.data)
            self._maybe_grasp()
            self._carry_cube()

    def reset(
        self,
        *,
        cube_pos: np.ndarray | None = None,
        qpos_noise: float = 0.0,
        rng: np.random.Generator | None = None,
        settle_seconds: float = 0.4,
    ) -> None:
        """Reset the arm to HOME (+ noise) and place the cube, then settle."""

        mujoco = self._mujoco
        mujoco.mj_resetData(self.model, self.data)
        self._grasped = False
        self._grasp_offset[:] = 0.0
        qpos = HOME_QPOS.copy()
        if qpos_noise > 0.0:
            rng = rng or np.random.default_rng()
            qpos += rng.normal(0.0, qpos_noise, size=7)
        qpos = np.clip(qpos, self.qpos_limit_lower, self.qpos_limit_upper)
        self.data.qpos[self.arm_qpos_adr] = qpos

        if rng is None:
            rng = np.random.default_rng()
        cube = sample_cube_position(rng) if cube_pos is None else np.asarray(cube_pos, dtype=np.float64)
        self.data.qpos[self.cube_qpos_adr : self.cube_qpos_adr + 3] = cube[:3]
        quat = cube[3:7] if len(cube) >= 7 else np.array([1.0, 0.0, 0.0, 0.0])
        self.data.qpos[self.cube_qpos_adr + 3 : self.cube_qpos_adr + 7] = quat
        self._cube_start_z = float(self.data.qpos[self.cube_qpos_adr + 2])

        self.data.ctrl[:] = np.clip(HOME_QPOS, self.qpos_limit_lower, self.qpos_limit_upper)
        mujoco.mj_forward(self.model, self.data)
        steps = int(round(settle_seconds * self.control_hz)) * self.substeps
        for _ in range(steps):
            mujoco.mj_step(self.model, self.data)

    # -- rendering -----------------------------------------------------

    def _ensure_renderers(self):
        if self._renderers is None:
            import mujoco

            self._renderers = [
                mujoco.Renderer(self.model, self._image_size, self._image_size)
                for _ in CAMERA_NAMES
            ]
        return self._renderers

    def render(self) -> dict[str, np.ndarray]:
        frames = {}
        for name, cam_id, renderer in zip(CAMERA_NAMES, self._cam_ids, self._ensure_renderers()):
            renderer.update_scene(self.data, camera=cam_id)
            frames[name] = renderer.render().copy()
        return frames

    def render_pair(self) -> tuple[np.ndarray, np.ndarray]:
        frames = self.render()
        return frames[CAMERA_NAMES[0]], frames[CAMERA_NAMES[1]]

    def close(self) -> None:
        if self._renderers is not None:
            for renderer in self._renderers:
                renderer.close()
            self._renderers = None
