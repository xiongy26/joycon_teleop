#!/usr/bin/env python
"""把 yam_arm_mujoco_standalone 的 YAM 模型与本项目 EL-A3 模型叠加进同一个
MuJoCo 场景，驱动同样的逻辑关节角，目视验证两者结构是否一致。

用法（在仓库根目录执行）:
  python scripts/view_model_overlay.py --solve-report   # 只打印对齐/映射与残差报告
  python scripts/view_model_overlay.py                  # 打开 3D 窗口，自动演示扫掠
  python scripts/view_model_overlay.py --gui            # 3D 窗口 + 滑条手动驱动
  python scripts/view_model_overlay.py --solid-yam      # yam 不半透明（默认半透明便于叠看）
  python scripts/view_model_overlay.py --render PREFIX  # 离屏渲染若干位姿存 PNG（无窗口）
  python scripts/view_model_overlay.py --smoke          # 无头自检: 构建+演示 120 帧后退出

原理:
  * 运行时把两份 MJCF 合并成一个模型: yam 子树整体挂到求解出的基座对齐变换上，
    所有 yam 对象名加 "yam_" 前缀；网格全部内嵌（assets dict），不落盘。
  * 基座对齐: 以 joint1(底座竖直轴) 锚点为原点、轴为 z，用 joint1→joint2 锚点
    方向定 x，构造两侧基座标架，T = F_el ∘ F_yam⁻¹（--align all 则改用全部
    关节锚点 Kabsch 最小二乘）。
  * 关节映射: 按链条顺序枚举 6 选 7 的对应关系并打分；逐关节符号 s_i 由对齐后
    轴向点积给出；逻辑角约定 = EL-A3 关节角，yam_q = s_i * θ（无偏置假设在
    报告里用零位残差验证）。
  * 驱动为纯运动学回放：直接写两侧 qpos（同一逻辑角），不经过物理/执行器，
    叠合差异纯粹反映运动学结构本身，不受跟踪误差干扰。
"""
import argparse
import itertools
import os
import sys
import time

import numpy as np
import mujoco

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
PROJ_XML = os.path.join(REPO, "robot", "el_a3_description", "urdf", "scene_gripper.xml")
PROJ_MESH_DIR = os.path.join(REPO, "robot", "el_a3_description", "meshes")
YAM_XML = os.path.join(REPO, "yam_arm_mujoco_standalone", "yam_arm_standalone", "yam", "yam_full.xml")
YAM_MESH_DIR = os.path.join(REPO, "yam_arm_mujoco_standalone", "yam_arm_standalone", "yam", "assets")

YAM_PREFIX = "yam_"
YAM_ARM_JOINTS = [f"joint{i}" for i in range(1, 7)]           # yam 侧关节名（合并前）
PROJ_ARM_JOINTS = [f"L{i}_joint" for i in range(1, 7)]        # EL-A3 侧关节名
YAM_GRIP_JOINTS = ["joint7", "joint8"]                        # yam 夹爪滑动关节
PROJ_EXTRA_JOINT = "L7_joint"                                 # EL-A3 夹爪滚转（yam 无对应）
PROJ_ALL = PROJ_ARM_JOINTS + [PROJ_EXTRA_JOINT]               # L1..L7，用于 σ 枚举

# ---------------------------------------------------------------- 纯函数工具

def xmat(d, bid):
    return d.xmat[bid].reshape(3, 3)


def joint_frames(m, d):
    """返回 {关节名: (世界锚点, 世界轴, 子body世界原点, range)}，仅铰链。"""
    out = {}
    for j in range(m.njnt):
        if m.jnt_type[j] != mujoco.mjtJoint.mjJNT_HINGE:
            continue
        name = mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_JOINT, j)
        bid = m.jnt_bodyid[j]
        R = xmat(d, bid)
        out[name] = (
            d.xpos[bid] + R @ m.jnt_pos[j],
            R @ m.jnt_axis[j],
            np.array(d.xpos[bid]),
            np.array(m.jnt_range[j]),
        )
    return out


def frame_from(a1, ax1, a2):
    """由锚点 a1/轴 ax1/锚点 a2 构造世界标架 F（列 = x,y,z）。z 取 ax1 方向，
    x 取 a1→a2 在垂直于 z 平面上的投影。"""
    z = ax1 / np.linalg.norm(ax1)
    v = a2 - a1
    v = v - (v @ z) * z
    n = np.linalg.norm(v)
    if n < 1e-9:
        raise ValueError("joint1 与 joint2 锚点连线与轴共线，无法构标架")
    x = v / n
    y = np.cross(z, x)
    return np.column_stack([x, y, z])


def kabsch(src, dst):
    """R,t 使 dst ≈ R @ src + t（行列式 +1）。"""
    cs, cd = src.mean(0), dst.mean(0)
    H = (src - cs).T @ (dst - cd)
    U, _, Vt = np.linalg.svd(H)
    R = Vt.T @ U.T
    if np.linalg.det(R) < 0:
        Vt[-1] *= -1
        R = Vt.T @ U.T
    return R, cd - R @ cs


def quat_of(R):
    q = np.zeros(4)
    mujoco.mju_mat2Quat(q, R.reshape(-1))
    return q


# ---------------------------------------------------------------- 对齐求解

def solve_alignment(align="base"):
    """求解基座变换 T（yam 世界 → EL-A3 世界）与逐关节符号 s_i。
    返回 (R, t, signs, report)。逻辑角约定 = EL-A3 关节角。"""
    mp = mujoco.MjModel.from_xml_path(PROJ_XML)
    my = mujoco.MjModel.from_xml_path(YAM_XML)
    dp, dy = mujoco.MjData(mp), mujoco.MjData(my)
    mujoco.mj_forward(mp, dp)
    mujoco.mj_forward(my, dy)
    fp = joint_frames(mp, dp)
    fy = joint_frames(my, dy)
    ap = [fp[n] for n in PROJ_ALL]   # 7 个（L1..L7）
    ay = [fy[n] for n in YAM_ARM_JOINTS]    # 6 个
    report = {"fp": fp, "fy": fy}

    # --- 候选对应关系打分（信息用）：链条顺序保持，6 选 7
    cand_rows = []
    for sel in itertools.combinations(range(7), 6):
        dst = np.array([ap[i][0] for i in sel])
        src = np.array([a[0] for a in ay])
        R, t = kabsch(src, dst)
        res = np.linalg.norm(dst - (R @ src.T).T - t, axis=1)
        ax_err = 0.0
        for k, i in enumerate(sel):
            c = abs((R @ ay[k][1]) @ ap[i][1])
            ax_err += np.degrees(np.arccos(np.clip(c, -1, 1)))
        cand_rows.append((sel, res.mean(), ax_err))
    cand_rows.sort(key=lambda r: (r[1], r[2]))
    report["candidates"] = cand_rows

    # --- 基座对齐 T
    best = cand_rows[0][0]
    if align == "base":
        # z = joint1 轴；x = 零位时 joint1→末端 的水平投影方向
        # （两臂零位指向同一方向 = "按同样方式安装"）
        tip_p = np.array(dp.site_xpos[mujoco.mj_name2id(mp, mujoco.mjtObj.mjOBJ_SITE, "gripper_tip_site")])
        tip_y = np.array(dy.site_xpos[mujoco.mj_name2id(my, mujoco.mjtObj.mjOBJ_SITE, "grip")])
        Fp = frame_from(ap[0][0], ap[0][1], tip_p)
        Fy = frame_from(ay[0][0], ay[0][1], tip_y)
        if (Fp[:, 2] @ ap[0][1]) * (Fy[:, 2] @ ay[0][1]) < 0:
            Fy[:, 2] *= -1
            Fy[:, 1] *= -1
        R = Fp @ Fy.T
        t = ap[0][0] - R @ ay[0][0]
    else:  # all: Kabsch 于最优候选
        src = np.array([a[0] for a in ay])
        dst = np.array([ap[i][0] for i in best])
        R, t = kabsch(src, dst)
    report["align"] = align
    report["sigma"] = cand_rows[0][0]

    # --- 逐关节符号与零位残差
    signs, zero_res = [], []
    for k, name_y in enumerate(YAM_ARM_JOINTS):
        name_p = PROJ_ALL[report["sigma"][k]]
        s = 1.0 if (R @ fy[name_y][1]) @ fp[name_p][1] >= 0 else -1.0
        signs.append(s)
        zero_res.append(float(np.linalg.norm(fp[name_p][0] - (R @ fy[name_y][0] + t))))
    report["signs"] = signs
    report["zero_res"] = zero_res
    report["R"], report["t"] = R, t
    return R, t, signs, report


# ---------------------------------------------------------------- 合并模型

def _collect_names(elem, acc):
    if elem.get("name"):
        acc.add(elem.get("name"))
    for ch in elem:
        _collect_names(ch, acc)


def _prefix_refs(elem, names):
    for k in ("name", "mesh", "joint", "body", "site", "body1", "body2",
              "geom", "camera", "material", "subtree"):
        v = elem.get(k)
        if v and v in names:
            elem.set(k, YAM_PREFIX + v)
    for ch in elem:
        _prefix_refs(ch, names)


def build_overlay(R, t, solid_yam=False):
    """返回 (MjModel, meta)。合并后的模型不含执行器（纯运动学回放）。

    yam 子树保持独立包原生摆放、不做几何互换，但两指滑动轴与
    scene_gripper.xml 同步做了符号对调（2026-11-11 用户要求的开合方向翻转）：
    solve_alignment 的基座对齐与关节符号映射已把两臂夹爪标架在世界里摆成
    同向，两模型手指几何数值完全相同，故四指天然成对、同向、同侧。"""
    import xml.etree.ElementTree as ET

    proj = ET.parse(PROJ_XML).getroot()
    yam = ET.parse(YAM_XML).getroot()

    # yam 子树收集原名并加前缀
    yam_names = set()
    _collect_names(yam, yam_names)
    _prefix_refs(yam, yam_names)

    # yam 子树两指滑动轴做与 scene_gripper.xml 同款的符号对调：
    # EL-A3 集成夹爪的张开方向已按用户要求(2026-11-11)翻转，这里同步翻转
    # yam 侧，保证合并场景四根手指依旧成对、同向、同步开合。
    for jn in ("yam_joint7", "yam_joint8"):
        j = yam.find(f".//joint[@name='{jn}']")
        if j is not None:
            ax = [float(v) for v in j.get("axis").split()]
            j.set("axis", " ".join(f"{-v:g}" for v in ax))

    # yam 网格文件路径改成平铺文件名（键即 assets dict 键）
    for mesh in yam.iter("mesh"):
        f = mesh.get("file")
        mesh.set("file", os.path.basename(f))

    # ---- 基树 = EL-A3 场景：去掉执行器与 mocap 目标球
    for act in proj.findall("actuator"):
        proj.remove(act)
    wb = proj.find("worldbody")
    tgt = wb.find("body[@name='target']")
    if tgt is not None:
        wb.remove(tgt)

    vis = proj.find("visual")
    glob = vis.find("global")
    if glob is None:
        glob = ET.SubElement(vis, "global")
    glob.set("offwidth", "1600")
    glob.set("offheight", "1000")

    # ---- 资产: yam 网格并入（_prefix_refs 已把 mesh name/ref 统一加前缀）
    # scene_gripper 已内置同名 yam 夹爪网格（yam_tip_left 等），跳过避免重名
    existing_meshes = {m.get("name") for m in proj.find("asset").iter("mesh")}
    for mesh in yam.find("asset").findall("mesh"):
        if mesh.get("name") in existing_meshes:
            continue
        proj.find("asset").append(mesh)

    # ---- yam base 挂到对齐变换上，geom 半透明
    yam_base = yam.find("worldbody").find("body")
    yam_base.set("pos", " ".join(f"{v:.6f}" for v in t))
    yam_base.set("quat", " ".join(f"{v:.6f}" for v in quat_of(R)))
    if not solid_yam:
        for g in yam_base.iter("geom"):
            r, g_, b, *_ = (g.get("rgba", "0.5 0.5 0.5 1")).split()
            g.set("rgba", f"{r} {g_} {b} 0.45")
    wb.append(yam_base)

    # ---- 相机: 正面与斜视
    def add_cam(name, pos, lookat):
        pos, lookat = np.array(pos, float), np.array(lookat, float)
        fwd = lookat - pos
        fwd /= np.linalg.norm(fwd)
        right = np.cross(fwd, [0, 0, 1.0])
        right /= np.linalg.norm(right)
        up = np.cross(right, fwd)
        c = ET.SubElement(wb, "camera")
        c.set("name", name)
        c.set("pos", " ".join(f"{v:.3f}" for v in pos))
        c.set("xyaxes", " ".join(f"{v:.3f}" for v in np.r_[right, up]))
    add_cam("front", (0.0, -1.05, 0.22), (0.0, 0.0, 0.16))
    add_cam("iso", (0.85, -0.85, 0.62), (0.0, 0.0, 0.16))

    # ---- assets 字典（读网格字节；scene_gripper 已内置 yam 夹爪网格，按文件名去重）
    assets = {}
    for mesh in proj.find("asset").iter("mesh"):
        f = os.path.basename(mesh.get("file"))
        if f in assets:
            continue
        # mesh 名以 yam_ 开头的未必来自 yam 包（如本项目集成的 yam_gripper_rail.stl），
        # 按文件实际所在目录解析
        p_proj = os.path.join(PROJ_MESH_DIR, f)
        p_yam = os.path.join(YAM_MESH_DIR, f)
        assets[f] = open(p_proj if os.path.exists(p_proj) else p_yam, "rb").read()

    xml = ET.tostring(proj, encoding="unicode")
    model = mujoco.MjModel.from_xml_string(xml, assets)

    meta = {
        "proj_ids": [mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, n) for n in PROJ_ARM_JOINTS],
        "yam_ids": [mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, YAM_PREFIX + n) for n in YAM_ARM_JOINTS],
        "proj_qadr": [], "yam_qadr": [],
        "proj_range": [], "yam_range": [],
        "extra_qadr": mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, PROJ_EXTRA_JOINT),
        "grip_qadr": [mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, YAM_PREFIX + n) for n in YAM_GRIP_JOINTS],
        # EL-A3 自身的双指滑关节（scene_gripper 集成 yam 夹爪后存在；纯运动学回放需手动驱动）
        "proj_grip_qadr": [q for q in (mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, n)
                                       for n in ("finger_left_joint", "finger_right_joint")) if q >= 0],
    }
    for jid in meta["proj_ids"]:
        meta["proj_qadr"].append(model.jnt_qposadr[jid])
        meta["proj_range"].append(model.jnt_range[jid])
    for jid in meta["yam_ids"]:
        meta["yam_qadr"].append(model.jnt_qposadr[jid])
        meta["yam_range"].append(model.jnt_range[jid])
    meta["proj_range"] = np.array(meta["proj_range"])
    meta["yam_range"] = np.array(meta["yam_range"])
    meta["extra_range"] = model.jnt_range[meta["extra_qadr"]]
    return model, meta


GRIP_COUPLE_COEF = -0.0299                                    # q_finger = coef * L7（与 scene_gripper.xml equality 一致）
GRIP_QRANGE = (0.0, 0.04695)                                  # 手指滑关节行程（两侧相同）


def apply_logical(model, data, meta, signs, theta, l7=0.0, grip=None):
    """同一逻辑角 θ(6,) 写入两侧 qpos；l7 = 夹爪通道（-90°开 / 0°合）。

    两侧夹爪手指共用同一开合量 q = clip(-0.0299*l7, 0, 0.04695)：
    EL-A3 侧手写 qpos（纯运动学回放不走 equality 求解器），yam 侧
    joint7/8 用同一 q —— 两个夹爪同步开合，运动方向严格一致。
    grip 仅用于强制覆盖开合量（调试用），默认随 l7 联动。"""
    for i in range(6):
        data.qpos[meta["proj_qadr"][i]] = theta[i]
        data.qpos[meta["yam_qadr"][i]] = signs[i] * theta[i]
    data.qpos[meta["extra_qadr"]] = l7
    if grip is None:
        grip = float(np.clip(GRIP_COUPLE_COEF * l7, *GRIP_QRANGE))
    for adr in meta["grip_qadr"]:
        data.qpos[adr] = grip
    for adr in meta.get("proj_grip_qadr", []):
        data.qpos[adr] = grip
    mujoco.mj_forward(model, data)


def logical_ranges(meta, signs):
    """每对关节: (EL-A3 范围, 映射到逻辑角的 yam 范围, 交集或 None)。
    交集为空 = 弯曲方向/范围不兼容。"""
    out = []
    for i in range(6):
        pr = np.sort(meta["proj_range"][i])
        yr_l = np.sort(meta["yam_range"][i] / signs[i])
        lo, hi = max(pr[0], yr_l[0]), min(pr[1], yr_l[1])
        out.append((pr, yr_l, (lo, hi) if lo <= hi else None))
    return out


# ---------------------------------------------------------------- 报告

def print_report(align, R, t, signs, report, meta, model):
    fp, fy = report["fp"], report["fy"]
    print("=" * 78)
    print("基座对齐变换 T（yam 世界系 → EL-A3 世界系）  align=%s" % align)
    print("  R =\n", np.round(R, 5))
    print("  t =", np.round(t, 5))
    print("-" * 78)
    print("关节锚点距离矩阵（链条顺序，frame 无关）")
    for tag, frames, names in (("EL-A3", fp, PROJ_ARM_JOINTS), ("yam  ", fy, YAM_ARM_JOINTS)):
        pts = [frames[n][0] for n in names]
        d = [np.linalg.norm(pts[i + 1] - pts[i]) for i in range(len(pts) - 1)]
        print(f"  {tag} 相邻锚距: " + "  ".join(f"{v:.4f}" for v in d))
    print("-" * 78)
    print("对应关系候选（Kabsch 平均锚点残差 | 轴向角误差和）:")
    for sel, res, ax in report["candidates"]:
        names = ",".join(PROJ_ALL[i].split("_")[0] for i in sel)
        print(f"  σ={names:20s} {res * 1000:8.1f} mm | {ax:6.1f}°")
    print("-" * 78)
    print("逐关节: 符号 s_i（逻辑角→yam 角）| 零位锚点残差 | 轴向夹角(符号修正后) | 公共角范围")
    lr = logical_ranges(meta, signs)
    for k, name_y in enumerate(YAM_ARM_JOINTS):
        name_p = PROJ_ALL[report["sigma"][k]]
        ay_ax = signs[k] * (R @ fy[name_y][1])
        c = float(np.clip(ay_ax @ fp[name_p][1], -1, 1))
        common = lr[k][2]
        cs = f"[{common[0]:+.3f},{common[1]:+.3f}]" if common else "空(方向相反)"
        print(f"  L{k+1}↔j{k+1}: s={signs[k]:+.0f}  零位残差={report['zero_res'][k] * 1000:7.1f} mm  "
              f"轴夹角={np.degrees(np.arccos(c)):5.2f}°  公共={cs}")
    print("-" * 78)
    print("逻辑角扫掠下的末端残差（EL-A3 gripper_tip_site vs T∘yam grip）")
    data = mujoco.MjData(model)
    tip_p = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_SITE, "gripper_tip_site")
    tip_y = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_SITE, YAM_PREFIX + "grip")
    worst = 0.0
    for i in range(6):
        res_max, at = 0.0, 0.0
        pr = meta["proj_range"][i]
        for th in np.linspace(pr[0], pr[1], 25):
            theta = np.zeros(6)
            theta[i] = th
            apply_logical(model, data, meta, signs, theta)
            r = float(np.linalg.norm(data.site_xpos[tip_p] - data.site_xpos[tip_y]))
            if r > res_max:
                res_max, at = r, th
        worst = max(worst, res_max)
        print(f"  仅扫 L{i+1} ({pr[0]:+.2f}→{pr[1]:+.2f}): 末端最大残差 {res_max * 1000:7.1f} mm @ θ={at:+.2f}")
    rng = np.random.default_rng(0)
    res_all = []
    for _ in range(50):
        theta = np.array([rng.uniform(*meta["proj_range"][i]) for i in range(6)])
        apply_logical(model, data, meta, signs, theta)
        res_all.append(np.linalg.norm(data.site_xpos[tip_p] - data.site_xpos[tip_y]))
    res_all = np.array(res_all)
    print(f"  随机 50 姿态: 平均 {res_all.mean() * 1000:.1f} mm, 最大 {res_all.max() * 1000:.1f} mm")
    print("=" * 78)


def prepare(align, solid_yam):
    """求解对齐 + 构建合并模型，一次打包所有模式需要的东西。"""
    R, t, signs, rep = solve_alignment(align)
    model, meta = build_overlay(R, t, solid_yam=solid_yam)
    return model, meta, signs, rep


# ---------------------------------------------------------------- 演示序列

def demo_theta(t, lr):
    """顺序扫掠每个关节（各 3.5 s），再整体正弦组合，循环。"""
    theta = np.zeros(6)
    seg = 3.5
    k = int(t // seg) % 7
    ph = (t % seg) / seg * 2 * np.pi
    if k < 6:
        pr = np.array(lr[k][0])
        amp = 0.45 * (pr[1] - pr[0])
        mid = 0.5 * (pr[0] + pr[1])
        theta[k] = mid + amp * (np.sin(ph - np.pi / 2) + 1) / 2
    else:
        for i in range(6):
            pr = np.array(lr[i][0])
            amp = 0.3 * (pr[1] - pr[0])
            mid = 0.5 * (pr[0] + pr[1])
            theta[i] = mid + amp * np.sin(0.7 * t + 1.3 * i)
    return theta


# ---------------------------------------------------------------- 模式入口

def _viewer_loop(model, data, meta, signs, lr, state, stop, smoke):
    import mujoco.viewer
    try:
        with mujoco.viewer.launch_passive(model, data) as v:
            v.cam.lookat[:] = [0, 0, 0.16]
            v.cam.distance = 1.15
            v.cam.azimuth = 140
            v.cam.elevation = -25
            t0 = time.time()
            n = 0
            while v.is_running() and not stop["flag"]:
                t = time.time() - t0
                if state is not None:
                    apply_logical(model, data, meta, signs, np.array(state["theta"]),
                                  l7=state["l7"])
                else:
                    th = demo_theta(t, lr)
                    # 两侧夹爪同一开合信号：l7 摆动 → 双方手指同步开合（方向一致）
                    apply_logical(model, data, meta, signs, th,
                                  l7=-0.8 * (np.sin(0.5 * t) + 1) / 2)
                v.sync()
                n += 1
                if smoke and n >= 120:
                    break
                time.sleep(0.016)
    finally:
        stop["flag"] = True

def run_viewer(model, meta, signs, gui=False, smoke=False):
    data = mujoco.MjData(model)
    lr = logical_ranges(meta, signs)
    apply_logical(model, data, meta, signs, np.zeros(6))
    stop = {"flag": False}

    if not gui:
        _viewer_loop(model, data, meta, signs, lr, None, stop, smoke)
        return

    # GUI 模式: tkinter 占主线程（Tcl 要求），MuJoCo 循环放子线程，
    # 滑条经回调把浮点值写进普通 dict，两边不互跨线程触碰对方 API。
    import threading
    import tkinter as tk
    state = {"theta": [0.0] * 6, "l7": -1.5708}
    t = threading.Thread(target=_viewer_loop,
                         args=(model, data, meta, signs, lr, state, stop, smoke),
                         daemon=True)
    t.start()

    root = tk.Tk()
    root.title("overlay 逻辑角（= EL-A3 关节角 rad）")
    rows = [(f"L{i+1}", lr[i][0], (lambda i: lambda v: state["theta"].__setitem__(i, float(v)))(i))
            for i in range(6)]
    # 单一夹爪滑条：同一信号驱动两个模型的夹爪（负=开 / 0=合），方向严格一致
    rows.append(("夹爪 L7 (-90开/0合)", (-1.5708, 0.0), lambda v: state.__setitem__("l7", float(v))))
    for r, (label, rng, cmd) in enumerate(rows):
        tk.Label(root, text=label).grid(row=r, column=0, sticky="e")
        s = tk.Scale(root, from_=float(rng[0]), to=float(rng[1]), resolution=0.005,
                     orient="horizontal", length=320, command=cmd)
        s.set(-1.5708 if "夹爪" in label else 0.0)
        s.grid(row=r, column=1)

    def poll():  # Tk 线程内轮询停止标记（smoke 用；关闭 MuJoCo 窗口后自动收尾）
        if stop["flag"]:
            root.destroy()
            return
        root.after(100, poll)
    poll()
    root.mainloop()
    stop["flag"] = True
    t.join(timeout=3)


def run_render(model, meta, signs, prefix):
    data = mujoco.MjData(model)
    renderer = mujoco.Renderer(model, height=900, width=1280)
    poses = [
        ("zero", np.zeros(6)),
        ("bent", np.array([0.4, 0.5, -1.0, 0.6, 0.4, 0.3])),
        ("elbow_neg", np.array([0.0, 0.0, +1.0, 0.0, 0.0, 0.0])),
    ]
    for cam in ("front", "iso"):
        cid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_CAMERA, cam)
        for name, th in poses:
            apply_logical(model, data, meta, signs, th)
            renderer.update_scene(data, camera=cid)
            img = renderer.render()
            path = f"{prefix}_{name}_{cam}.png"
            from PIL import Image
            Image.fromarray(img).save(path)
            print("saved", path)


# ---------------------------------------------------------------- main

def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--align", choices=["base", "all"], default="base")
    ap.add_argument("--solve-report", action="store_true")
    ap.add_argument("--gui", action="store_true")
    ap.add_argument("--solid-yam", action="store_true")
    ap.add_argument("--render", metavar="PREFIX")
    ap.add_argument("--smoke", action="store_true")
    args = ap.parse_args()

    if args.solve_report:
        model, meta, signs, rep = prepare(args.align, solid_yam=True)
        print_report(args.align, rep.get("R"), rep.get("t"), signs, rep, meta, model)
        return
    model, meta, signs, rep = prepare(args.align, solid_yam=args.solid_yam)
    if args.render:
        run_render(model, meta, signs, args.render)
        return
    run_viewer(model, meta, signs, gui=args.gui, smoke=args.smoke)


if __name__ == "__main__":
    main()
