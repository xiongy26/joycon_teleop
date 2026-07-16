#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""EL-A3 准解析逆运动学。

几何依据(零位世界系,MuJoCo 提取):
  L1 腰:  anchor (0,0,0.06),  axis (0,0,1)      竖直
  L2 肩:  anchor (-0.0096,0,0.129), axis (0,-1,0) 水平
  L3 肘:  anchor (0.1804,0,0.129)                 与 L2 平行
  L4 :    anchor (-0.0046,0,0.193), axis (0,-1,0) 与 L2/L3 平行
  腕心(L5∩L6): (-0.0538,0,0.192)   仅依赖 q1..q4
  末端 site:    (-0.241,0,0.192)    末端->腕心偏置 0.1872 m

结构:
  - L2/L3/L4 三轴平行 -> 满足 Pieper 条件,位置可解析
  - L4-L5 有 4.92 cm 偏置 -> 非标准球形腕,位置/姿态耦合
  - L5⊥L6 且相交 -> 腕心 = L5∩L6

求解(准解析):
  1. 腕心目标 p_w = target_pos + R_target @ r_wrist_offset_site
  2. 位置解析(固定 q4=q_ref):L1(腰)+L2/L3(平行双臂)封闭解
     - q1 由腕心水平方位定(腰前/腰后两解)
     - q2 = 大臂仰角(2R: α - es·γ),符号由 MuJoCo 标定
     - q3 = 小臂相对大臂角(2R: es·β),符号/零位由 MuJoCo 标定
  3. 姿态精修:固定 q1,q2,q3,用 q4,q5,q6 雅可比 DLS 迭代降残差
  4. 多解(腰×肘=4)择近 q_ref,限位夹紧

仅依赖 core.ik_core 已加载的 mujoco model/data,不重复建模型。
"""

from __future__ import annotations

import numpy as np
import mujoco

from core.ik_core import (
    model,
    data,
    ARM_JOINT_LIMITS_LOWER,
    ARM_JOINT_LIMITS_UPPER,
    ee_site_id,
    lock,
    fix_base_position,
)

# ------------------------------------------------------------------
# 关节索引辅助
# ------------------------------------------------------------------
_JOINT_NAMES = [
    "L1_joint", "L2_joint", "L3_joint",
    "L4_joint", "L5_joint", "L6_joint",
]
_jid = {model.joint(i).name: i for i in range(model.njnt)}
_joint_ids = [_jid[n] for n in _JOINT_NAMES]
_qpos_adr = [int(model.jnt_qposadr[j]) for j in _joint_ids]
_dof_adr = [int(model.jnt_dofadr[j]) for j in _joint_ids]
N_ARM = len(_JOINT_NAMES)

# 末端 site 在 site 系下到腕心的偏置(FK 验证,常量,沿 site +z)
# rot.T @ (pw-pe) = (0, 0, 0.18715)  对所有 q6 成立
_R_WRIST_OFFSET_SITE = np.array([0.0, 0.0, 0.18715])

# 连杆几何(零位提取)
_BIG_ARM = 0.19          # |p3 - p2|
_SHOULDER_R = 0.0096     # 肩点到 L1 轴的水平距离
_SHOULDER_Z = 0.069      # 肩点相对 L1 锚点的竖直偏置 (0.129-0.06)
_L1_ANCHOR = np.array([0.0, 0.0, 0.06])
# 注意:_FORE_ARM(=|wc-p3|) 与小臂相对角零位 _THETA_FORE0 随 q4 变化,
# 故在 _solve_position 内按 q4_ref 动态计算,不在此固定。

_Q_LOWER = ARM_JOINT_LIMITS_LOWER[:N_ARM].astype(float)
_Q_UPPER = ARM_JOINT_LIMITS_UPPER[:N_ARM].astype(float)


# ------------------------------------------------------------------
# FK 辅助
# ------------------------------------------------------------------
def _set_q(q: np.ndarray):
    for i, idx in enumerate(_qpos_adr):
        data.qpos[idx] = q[i]
    fix_base_position(model, data)
    mujoco.mj_forward(model, data)


def _ee_pose():
    pos = data.site_xpos[ee_site_id].copy()
    rot = data.site_xmat[ee_site_id].reshape(3, 3).copy()
    return pos, rot


def _wrist_center():
    j5, j6 = _jid["L5_joint"], _jid["L6_joint"]
    p5, a5 = data.xanchor[j5].copy(), data.xaxis[j5].copy()
    p6, a6 = data.xanchor[j6].copy(), data.xaxis[j6].copy()
    n = np.cross(a5, a6)
    t = np.dot(np.cross(p6 - p5, a6), n) / np.dot(n, n)
    return p5 + t * a5


def _jacobian():
    jacp = np.zeros((3, model.nv))
    jacr = np.zeros((3, model.nv))
    mujoco.mj_jacSite(model, data, jacp, jacr, ee_site_id)
    J = np.vstack([jacp[:, _dof_adr], jacr[:, _dof_adr]])  # (6,6) [v; w]
    return J


def _signed_angle(u, v, axis):
    """绕 axis 从 u 到 v 的有符号角。"""
    return float(np.arctan2(np.dot(axis, np.cross(u, v)), np.dot(u, v)))


def _wrap(a: float) -> float:
    return float((a + np.pi) % (2 * np.pi) - np.pi)


# ------------------------------------------------------------------
# q2/q3 标定:大臂仰角 / 小臂相对角 与关节角的线性映射
# ------------------------------------------------------------------
def _arm_dirs(q):
    """返回 (d_arm, d_fore, a2, e_arm)。"""
    _set_q(q)
    p2 = data.xanchor[_jid["L2_joint"]].copy()
    p3 = data.xanchor[_jid["L3_joint"]].copy()
    wc = _wrist_center()
    d_arm = p3 - p2
    d_fore = wc - p3
    a2 = data.xaxis[_jid["L2_joint"]].copy()
    arm_xy = d_arm[:2]
    nrm = np.linalg.norm(arm_xy)
    if nrm > 1e-9:
        e_arm = np.array([arm_xy[0] / nrm, arm_xy[1] / nrm, 0.0])
    else:
        e_arm = np.array([1.0, 0.0, 0.0])
    return d_arm, d_fore, a2, e_arm


def _calibrate():
    base = np.zeros(N_ARM)
    d_arm0, d_fore0, a20, e_arm0 = _arm_dirs(base)
    theta_arm0 = _signed_angle(e_arm0, d_arm0, a20)      # 大臂仰角零位 ~0
    theta_fore0 = _signed_angle(d_arm0, d_fore0, a20)    # 小臂相对角零位 ~2.879

    q2p = base.copy()
    q2p[1] = 0.1
    d_arm2, _, a22, e_arm2 = _arm_dirs(q2p)
    d_angle2 = _signed_angle(e_arm2, d_arm2, a22) - theta_arm0
    sign2 = 1.0 if d_angle2 >= 0 else -1.0

    q3p = base.copy()
    q3p[2] = 0.1
    _, d_fore3, a23, _ = _arm_dirs(q3p)
    # q3 扰动时大臂不动(d_arm 保持 base),小臂相对角变化
    theta_fore3 = _signed_angle(d_arm0, d_fore3, a23)
    sign3 = 1.0 if (theta_fore3 - theta_fore0) >= 0 else -1.0
    return sign2, theta_arm0, sign3, theta_fore0


_SIGN2, _THETA_ARM0, _SIGN3, _THETA_FORE0 = _calibrate()


# ------------------------------------------------------------------
# 位置解析(固定 q4)
# ------------------------------------------------------------------
def _solve_position(pw_target: np.ndarray, q4_ref: float):
    """固定 q4,解析解 q1,q2,q3 使腕心=pw_target。返回候选 [(q1,q2,q3),...](至多 4)。"""
    # 按 q4_ref 动态提取小臂几何:fore_arm 长度、小臂相对角零位
    _set_q(np.array([0.0, 0.0, 0.0, q4_ref, 0.0, 0.0]))
    p2_0 = data.xanchor[_jid["L2_joint"]].copy()
    p3_0 = data.xanchor[_jid["L3_joint"]].copy()
    wc_0 = _wrist_center()
    d_arm0 = p3_0 - p2_0
    d_fore0 = wc_0 - p3_0
    a2_0 = data.xaxis[_jid["L2_joint"]].copy()
    fore_arm = float(np.linalg.norm(d_fore0))
    theta_fore0 = _signed_angle(d_arm0, d_fore0, a2_0)

    dx, dy, dz = pw_target - _L1_ANCHOR
    r_xy = float(np.hypot(dx, dy))
    H = float(dz)
    a, b = _BIG_ARM, fore_arm

    cands = []
    # 腰前: e_arm 朝 -pw_xy, 腕心径向 -r; 腰后: e_arm 朝 +pw_xy, 腕心径向 +r
    for q1, radial in (
        (_wrap(np.arctan2(dy, dx) - np.pi), -r_xy),
        (_wrap(np.arctan2(dy, dx)), +r_xy),
    ):
        # 平面内肩->腕:肩点平面坐标 (-_SHOULDER_R, _SHOULDER_Z)
        dr = radial - (-_SHOULDER_R)
        dh = H - _SHOULDER_Z
        d2 = dr * dr + dh * dh
        d = np.sqrt(d2)
        if d > a + b + 1e-9 or d < abs(a - b) - 1e-9:
            continue
        cos_beta = (d2 - a * a - b * b) / (2.0 * a * b)
        beta = np.arccos(float(np.clip(cos_beta, -1.0, 1.0)))
        alpha = np.arctan2(dh, dr)
        gamma = np.arctan2(b * np.sin(beta), a + b * np.cos(beta))
        for es in (+1.0, -1.0):
            theta_arm = alpha - es * gamma      # 大臂仰角(零位 0)
            theta_fore = es * beta              # 小臂相对角(零位 theta_fore0)
            q2 = _wrap(_SIGN2 * (theta_arm - _THETA_ARM0))
            q3 = _wrap(_SIGN3 * (theta_fore - theta_fore0))
            cands.append((q1, q2, q3))
    return cands


# ------------------------------------------------------------------
# 姿态精修(q4,q5,q6 雅可比 DLS)
# ------------------------------------------------------------------
def _refine_orientation(q1, q2, q3, R_target, target_pos,
                        q_ref, max_iter=10, eps=1e-7):
    """固定 q1,q2,q3,用 q4,q5,q6 雅可比 DLS 降 6 维残差。"""
    q = np.array([q1, q2, q3, q_ref[3], q_ref[4], q_ref[5]], dtype=float)
    for _ in range(max_iter):
        _set_q(q)
        pos, rot = _ee_pose()
        e_pos = target_pos - pos
        dR = R_target @ rot.T
        e_rot = 0.5 * np.array([
            dR[2, 1] - dR[1, 2], dR[0, 2] - dR[2, 0], dR[1, 0] - dR[0, 1]
        ])
        err = np.concatenate([e_pos, e_rot])
        if np.linalg.norm(err) < eps:
            break
        J = _jacobian()[:, 3:6]
        lam = 1e-3
        if abs(q[4]) < 0.1:
            lam = max(lam, 0.05)
        A = J.T @ J + lam * np.eye(3)
        dq = np.linalg.solve(A, J.T @ err)
        nq = np.linalg.norm(dq)
        if nq > 0.3:
            dq *= 0.3 / nq
        q[3:6] += dq
        q[3:6] = np.clip(q[3:6], _Q_LOWER[3:6], _Q_UPPER[3:6])
    _set_q(q)
    pos, rot = _ee_pose()
    dR = R_target @ rot.T
    e_rot_f = 0.5 * np.array([
        dR[2, 1] - dR[1, 2], dR[0, 2] - dR[2, 0], dR[1, 0] - dR[0, 1]
    ])
    err = np.concatenate([target_pos - pos, e_rot_f])
    return q, float(np.linalg.norm(err))


# ------------------------------------------------------------------
# 入口
# ------------------------------------------------------------------
def solve(target_pos, target_rot, q_ref):
    """准解析 IK。返回 (q, err_norm) 或 None。"""
    target_pos = np.asarray(target_pos, dtype=float).copy()
    R_target = np.asarray(target_rot, dtype=float)
    q_ref = np.asarray(q_ref, dtype=float)
    pw_target = target_pos + R_target @ _R_WRIST_OFFSET_SITE

    cands = _solve_position(pw_target, float(q_ref[3]))
    if not cands:
        return None

    best = None
    for q1, q2, q3 in cands:
        q_full, err = _refine_orientation(
            q1, q2, q3, R_target, target_pos, q_ref
        )
        lo_viol = np.any(q_full < _Q_LOWER - 1e-6)
        hi_viol = np.any(q_full > _Q_UPPER + 1e-6)
        if lo_viol or hi_viol:
            q_full = np.clip(q_full, _Q_LOWER, _Q_UPPER)
        dist = float(np.linalg.norm(q_full - q_ref))
        # 主排序:误差小;同误差择近
        key = (err, dist)
        if best is None or key < best[0]:
            best = (key, q_full, err)
    if best is None:
        return None
    return best[1], best[2]


# ------------------------------------------------------------------
# self_test
# ------------------------------------------------------------------
def self_test(n=1000, seed=42):
    rng = np.random.default_rng(seed)
    rng2 = np.random.default_rng(seed + 1)
    pos_errs, rot_errs, fails = [], [], 0
    lo = _Q_LOWER + 0.1
    hi = _Q_UPPER - 0.1
    for _ in range(n):
        q_true = rng2.uniform(lo, hi)
        with lock:
            _set_q(q_true)
            pos, rot = _ee_pose()
        q_noise = rng.normal(0, 0.05, N_ARM)
        q_ref = np.clip(q_true + q_noise, _Q_LOWER, _Q_UPPER)
        sol = solve(pos, rot, q_ref)
        if sol is None:
            fails += 1
            continue
        q_sol = sol[0]
        with lock:
            _set_q(q_sol)
            pos2, rot2 = _ee_pose()
        pos_errs.append(np.linalg.norm(pos2 - pos))
        dR = rot @ rot2.T
        rot_errs.append(np.arccos(np.clip((np.trace(dR) - 1) / 2, -1, 1)))
    pos_errs = np.array(pos_errs)
    rot_errs = np.array(rot_errs)
    print(f"self_test n={n} fails={fails} solved={pos_errs.size}")
    if pos_errs.size:
        p99p = np.percentile(pos_errs, 99)
        p99r = np.percentile(rot_errs, 99)
        print(f"  pos err [m]:  max={pos_errs.max():.2e}"
              f" mean={pos_errs.mean():.2e} p99={p99p:.2e}")
        print(f"  rot err [rad]: max={rot_errs.max():.2e}"
              f" mean={rot_errs.mean():.2e} p99={p99r:.2e}")


if __name__ == "__main__":
    self_test()
