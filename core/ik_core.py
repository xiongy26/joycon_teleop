#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Minimal ik_core for standalone Cartesian control GUI."""

from pathlib import Path
import json
import threading

import numpy as np
import mujoco

PROJECT_ROOT = Path(__file__).resolve().parent.parent
MODEL_PATH = PROJECT_ROOT / "robot" / "el_a3_description" / "urdf" / "scene_gripper.xml"

# Load config
CONFIG_PATH = PROJECT_ROOT / "config" / "config.json"
try:
    if CONFIG_PATH.is_file():
        with open(CONFIG_PATH.as_posix(), "r", encoding="utf-8") as _f:
            _cfg = json.load(_f)
    else:
        _cfg = {}
except Exception:
    _cfg = {}

ik_cfg = _cfg.get("ik", {})
JOINT_MIN = float(ik_cfg.get("joint_min", -2.79253))
JOINT_MAX = float(ik_cfg.get("joint_max", 2.79253))

# End effector site
EE_SITE_NAME = "gripper_tip_site"

# Joint names
ARM_JOINT_NAMES = [
    "L1_joint", "L2_joint", "L3_joint",
    "L4_joint", "L5_joint", "L6_joint",
]
GRIPPER_JOINT_NAME = "L7_joint"

# Load model
if not MODEL_PATH.is_file():
    raise FileNotFoundError(f"Model not found: {MODEL_PATH}")

model = mujoco.MjModel.from_xml_path(MODEL_PATH.as_posix())
data = mujoco.MjData(model)

# End effector site ID
ee_site_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_SITE, EE_SITE_NAME)
if ee_site_id < 0:
    raise RuntimeError(f"Site not found: {EE_SITE_NAME}")

# Joint indices
joint_name_to_id = {model.joint(i).name: i for i in range(model.njnt)}

arm_joint_ids = []
arm_qpos_indices = []
arm_dof_indices = []

for name in ARM_JOINT_NAMES:
    if name not in joint_name_to_id:
        raise RuntimeError(f"Joint not found: {name}")
    j_id = joint_name_to_id[name]
    arm_joint_ids.append(j_id)
    arm_qpos_indices.append(int(model.jnt_qposadr[j_id]))
    arm_dof_indices.append(int(model.jnt_dofadr[j_id]))

if GRIPPER_JOINT_NAME not in joint_name_to_id:
    raise RuntimeError(f"Gripper joint not found: {GRIPPER_JOINT_NAME}")
gripper_joint_id = joint_name_to_id[GRIPPER_JOINT_NAME]
gripper_qpos_index = int(model.jnt_qposadr[gripper_joint_id])
gripper_dof_index = int(model.jnt_dofadr[gripper_joint_id])

free_joint_ids = [
    i for i in range(model.njnt)
    if int(model.jnt_type[i]) == mujoco.mjtJoint.mjJNT_FREE
]
base_free_joint_id = free_joint_ids[0] if free_joint_ids else None
base_free_qpos_adr = (
    int(model.jnt_qposadr[base_free_joint_id])
    if base_free_joint_id is not None else None
)
base_free_dof_adr = (
    int(model.jnt_dofadr[base_free_joint_id])
    if base_free_joint_id is not None else None
)

N_ARM = len(arm_qpos_indices)
ARM_JOINT_LIMITS_LOWER = np.maximum(
    model.jnt_range[arm_joint_ids, 0].astype(float),
    np.full(N_ARM, JOINT_MIN, dtype=float),
)
ARM_JOINT_LIMITS_UPPER = np.minimum(
    model.jnt_range[arm_joint_ids, 1].astype(float),
    np.full(N_ARM, JOINT_MAX, dtype=float),
)

lock = threading.Lock()


def fix_base_position(model_arg=None, data_arg=None):
    """Fix base position to prevent falling through floor."""
    mdl = model_arg if model_arg is not None else model
    dat = data_arg if data_arg is not None else data
    try:
        if mdl is model:
            qpos_adr = base_free_qpos_adr
            dof_adr = base_free_dof_adr
        else:
            free_ids = [
                i for i in range(mdl.njnt)
                if int(mdl.jnt_type[i]) == mujoco.mjtJoint.mjJNT_FREE
            ]
            qpos_adr = int(mdl.jnt_qposadr[free_ids[0]]) if free_ids else None
            dof_adr = int(mdl.jnt_dofadr[free_ids[0]]) if free_ids else None

        if qpos_adr is not None and dat.qpos.size >= qpos_adr + 7:
            dat.qpos[qpos_adr:qpos_adr + 3] = [0.0, 0.0, 0.0]
            dat.qpos[qpos_adr + 3:qpos_adr + 7] = [1.0, 0.0, 0.0, 0.0]
    except Exception:
        pass
    try:
        if dof_adr is not None and dat.qvel.size >= dof_adr + 6:
            dat.qvel[dof_adr:dof_adr + 6] = 0.0
    except Exception:
        pass


def rotation_matrix_to_quat(R):
    """3x3 rotation matrix -> quaternion [w, x, y, z]."""
    tr = R[0, 0] + R[1, 1] + R[2, 2]
    if tr > 0:
        S = np.sqrt(tr + 1.0) * 2.0
        qw = 0.25 * S
        qx = (R[2, 1] - R[1, 2]) / S
        qy = (R[0, 2] - R[2, 0]) / S
        qz = (R[1, 0] - R[0, 1]) / S
    elif R[0, 0] > R[1, 1] and R[0, 0] > R[2, 2]:
        S = np.sqrt(1.0 + R[0, 0] - R[1, 1] - R[2, 2]) * 2.0
        qw = (R[2, 1] - R[1, 2]) / S
        qx = 0.25 * S
        qy = (R[0, 1] + R[1, 0]) / S
        qz = (R[0, 2] + R[2, 0]) / S
    elif R[1, 1] > R[2, 2]:
        S = np.sqrt(1.0 + R[1, 1] - R[0, 0] - R[2, 2]) * 2.0
        qw = (R[0, 2] - R[2, 0]) / S
        qx = (R[0, 1] + R[1, 0]) / S
        qy = 0.25 * S
        qz = (R[1, 2] + R[2, 1]) / S
    else:
        S = np.sqrt(1.0 + R[2, 2] - R[0, 0] - R[1, 1]) * 2.0
        qw = (R[1, 0] - R[0, 1]) / S
        qx = (R[0, 2] + R[2, 0]) / S
        qy = (R[1, 2] + R[2, 1]) / S
        qz = 0.25 * S
    return np.array([qw, qx, qy, qz], dtype=float)


def set_arm_q_vector(q):
    """Set arm joint positions in global data."""
    with lock:
        for i, idx in enumerate(arm_qpos_indices):
            data.qpos[idx] = q[i]
        fix_base_position()
        mujoco.mj_forward(model, data)


def get_arm_q_vector():
    """Get arm joint positions from global data."""
    with lock:
        return np.array([data.qpos[idx] for idx in arm_qpos_indices])
