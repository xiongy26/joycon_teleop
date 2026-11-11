#!/usr/bin/env python
"""EL-A3 + yam 可动夹爪查看/控制工具。

scene_gripper.xml 已把 yam 的双指夹爪焊接到 EL-A3 末端法兰：
  * 两指各为滑动关节，经 <equality> 与 L7（=实机 7 号夹爪电机）耦合：
    L7 = -90° → 全开，L7 = 0° → 并拢，>0 → 保持夹紧。
  * 指尖可碰撞，能真实夹取物体；gripper_tip_site 移到新指尖中心。

用法（仓库根目录）:
  python scripts/view_el_a3_gripper.py                 # 3D 窗口自动演示开合
  python scripts/view_el_a3_gripper.py --gui           # 滑条手动驱动 L1..L6 + 夹爪
  python scripts/view_el_a3_gripper.py --render PREFIX # 离屏渲染证据图（开/合/抓方块）
  python scripts/view_el_a3_gripper.py --smoke         # 无头自检 300 帧后退出
"""
import argparse
import os
import sys
import time
import xml.etree.ElementTree as ET

import numpy as np
import mujoco

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MODEL_XML = os.path.join(REPO, "robot", "el_a3_description", "urdf", "scene_gripper.xml")

ARM_JOINTS = [f"L{i}_joint" for i in range(1, 7)]
ARM_ACTS = [f"L{i}" for i in range(1, 7)]
L7_ACT = "L7"
L7_QMIN, L7_QMAX = -1.5708, 1.5708          # 关节硬限位（GUI 夹爪滑条只用 -90°..0°）

# 演示用臂姿态（弧度，= 各 L 关节目标）
POSE_DEMO = np.array([0.0, 0.35, -0.9, 0.5, 0.3, 0.0])
CUBE_HALF = 0.02                              # 抓取演示方块半宽 20mm


def jid(m, name, obj=mujoco.mjtObj.mjOBJ_JOINT):
    return mujoco.mj_name2id(m, obj, name)


def aid(m, name):
    return mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_ACTUATOR, name)


def _with_offscreen(proj, cameras=None):
    """离屏渲染需要大于默认 640px 的帧缓冲；可注入显式相机（pos/xyaxes）。"""
    vis = proj.find("visual")
    glob = vis.find("global")
    if glob is None:
        glob = ET.SubElement(vis, "global")
    glob.set("offwidth", "1600")
    glob.set("offheight", "1000")
    if cameras:
        wb = proj.find("worldbody")
        for name, pos, xyaxes in cameras:
            ET.SubElement(wb, "camera", name=name, pos=pos, xyaxes=xyaxes)
    return proj


def load_model(with_cube=False, cameras=None):
    proj = ET.parse(MODEL_XML).getroot()
    if with_cube:
        # 运行时注入一块可夹取的方块（不落盘，不影响原模型文件）
        wb = proj.find("worldbody")
        body = ET.SubElement(wb, "body", name="demo_cube", pos="0.2 0 0.2")
        ET.SubElement(body, "freejoint", name="demo_cube_joint")
        ET.SubElement(body, "inertial", pos="0 0 0", mass="0.05",
                      diaginertia="1.33e-5 1.33e-5 1.33e-5")
        ET.SubElement(body, "geom", name="demo_cube_geom", type="box",
                      size=f"{CUBE_HALF} {CUBE_HALF} {CUBE_HALF}",
                      rgba="0.85 0.15 0.12 1", contype="1", conaffinity="1",
                      friction="0.8 0.02 0.001")
    # from_xml_string 不解析相对路径：mesh 全部平铺内嵌
    xml_dir = os.path.dirname(MODEL_XML)
    assets = {}
    for mesh in proj.find("asset").iter("mesh"):
        f = os.path.basename(mesh.get("file"))
        if f not in assets:
            with open(os.path.join(xml_dir, mesh.get("file")), "rb") as fh:
                assets[f] = fh.read()
        mesh.set("file", f)
    return mujoco.MjModel.from_xml_string(
        ET.tostring(_with_offscreen(proj, cameras), encoding="unicode"), assets)


def set_pose(m, d, theta):
    """写臂关节 qpos 并让位置执行器目标同步（gravity 已禁用，臂保持姿态）。"""
    for i, n in enumerate(ARM_JOINTS):
        q = float(np.clip(theta[i], *m.jnt_range[jid(m, n)]))
        d.qpos[m.jnt_qposadr[jid(m, n)]] = q
        d.ctrl[aid(m, ARM_ACTS[i])] = q
    mujoco.mj_forward(m, d)


def grasp_center(m, d):
    """指尖抓取中心的世界坐标与夹爪轴方向。"""
    bid = jid(m, "yam_gripper_mount", mujoco.mjtObj.mjOBJ_BODY)
    R = d.xmat[bid].reshape(3, 3)
    site = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_SITE, "gripper_tip_site")
    return d.site_xpos[site], R[:, 2]           # 夹爪 +Z = 指向指尖方向


def close_gripper(m, d, l7_ctrl, steps=1500):
    d.ctrl[aid(m, L7_ACT)] = l7_ctrl
    for _ in range(steps):
        mujoco.mj_step(m, d)


def run_viewer(m, gui=False, smoke=False):
    import mujoco.viewer
    d = mujoco.MjData(m)
    set_pose(m, d, np.zeros(6))
    d.ctrl[aid(m, L7_ACT)] = -1.5708
    state = None
    stop = {"flag": False}

    def loop():
        t0 = time.time()
        n = 0
        while v.is_running() and not stop["flag"]:
            t = time.time() - t0
            if state is None:  # 自动演示：姿态缓慢摆动 + 夹爪周期开合
                th = POSE_DEMO * (0.5 + 0.5 * np.sin(0.35 * t))
                set_pose(m, d, th)
                phase = (np.sin(0.8 * t - np.pi / 2) + 1) / 2
                d.ctrl[aid(m, L7_ACT)] = -1.5708 * (1 - phase)
            else:              # GUI 模式
                set_pose(m, d, np.array(state["theta"]))
                d.ctrl[aid(m, L7_ACT)] = state["l7"]
            mujoco.mj_step(m, d)
            v.sync()
            n += 1
            if smoke and n >= 300:
                break
            time.sleep(0.002)

    try:
        with mujoco.viewer.launch_passive(m, d) as v:
            v.cam.lookat[:] = d.site_xpos[mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_SITE, "gripper_tip_site")]
            v.cam.distance = 0.85
            v.cam.azimuth = 130
            v.cam.elevation = -18
            if not gui:
                loop()
                return
            import threading
            import tkinter as tk
            state = {"theta": [0.0] * 6, "l7": -1.5708}
            t = threading.Thread(target=loop, daemon=True)
            t.start()
            root = tk.Tk()
            root.title("EL-A3 + yam 夹爪（rad / 夹爪通道）")
            for r in range(6):
                tk.Label(root, text=f"L{r+1}").grid(row=r, column=0, sticky="e")
                rng = m.jnt_range[jid(m, ARM_JOINTS[r])]
                s = tk.Scale(root, from_=float(rng[0]), to=float(rng[1]), resolution=0.005,
                             orient="horizontal", length=320,
                             command=(lambda i: lambda val: state["theta"].__setitem__(i, float(val)))(r))
                s.set(0.0)
                s.grid(row=r, column=1)
            tk.Label(root, text="L7 夹爪(-90开/0合)").grid(row=6, column=0, sticky="e")
            s7 = tk.Scale(root, from_=-1.5708, to=0.6, resolution=0.005,
                          orient="horizontal", length=320,
                          command=lambda val: state.__setitem__("l7", float(val)))
            s7.set(-1.5708)
            s7.grid(row=6, column=1)

            def poll():
                if stop["flag"]:
                    root.destroy()
                    return
                root.after(100, poll)
            poll()
            root.mainloop()
            stop["flag"] = True
            t.join(timeout=3)
    finally:
        stop["flag"] = True


def _cameras_from_gripper_frame(m, d, theta):
    """按演示姿态求夹爪标架，生成三台显式相机（正视指间 / 侧面行程 / 全景斜视）。

    夹爪系: +Z 指向指尖，双指沿 ±X 开合。相机 xyaxes = (右, 上)，视线 = 右×上。"""
    set_pose(m, d, theta)
    bid = jid(m, "yam_gripper_mount", mujoco.mjtObj.mjOBJ_BODY)
    p_site, z = grasp_center(m, d)
    R = d.xmat[bid].reshape(3, 3)
    x_g, y_g = R[:, 0], R[:, 1]
    tip = p_site + z * (0.1468 - 0.141)         # 指尖平面中心

    def cam_spec(eye, lookat, up):
        f = np.asarray(lookat) - np.asarray(eye)
        f /= np.linalg.norm(f)
        zc = -f
        up = np.asarray(up, float)
        x_c = up - (up @ zc) * zc
        x_c /= np.linalg.norm(x_c)
        y_c = np.cross(zc, x_c)
        return " ".join(f"{v:.4f}" for v in eye), " ".join(f"{v:.4f}" for v in np.r_[x_c, y_c])

    return [
        ("cam_front", *cam_spec(tip + z * 0.30, tip, y_g)),
        ("cam_side", *cam_spec(tip + x_g * 0.06 + y_g * 0.34, tip, z)),
        ("cam_iso", *cam_spec(tip + z * 0.38 + x_g * 0.30 + y_g * 0.30, tip, np.array([0.0, 0, 1.0]))),
        # 俯视：世界正上方往下看，右=+X、上=+Y
        ("cam_top", " ".join(f"{v:.4f}" for v in (tip[0], tip[1], tip[2] + 0.45)), "1 0 0 0 1 0"),
    ]


def run_render(m, prefix):
    from PIL import Image

    # 先在无相机模型上求演示姿态的夹爪标架，再把相机注入重编译
    d0 = mujoco.MjData(m)
    cams = _cameras_from_gripper_frame(m, d0, POSE_DEMO)
    m = load_model(cameras=cams)
    d = mujoco.MjData(m)
    renderer = mujoco.Renderer(m, height=900, width=1280)
    m2 = load_model(with_cube=True, cameras=cams)
    d2 = mujoco.MjData(m2)
    renderer2 = mujoco.Renderer(m2, height=900, width=1280)  # 必须绑带方块的模型

    def snap(name, cam_name, data=None, model=None):
        model = model or m
        rend = renderer2 if model is m2 else renderer
        dd = data if data is not None else d
        cid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_CAMERA, cam_name)
        rend.update_scene(dd, camera=cid)
        path = f"{prefix}_{name}.png"
        Image.fromarray(rend.render()).save(path)
        print("saved", path)

    # --- 1/2: 全开与并拢（指尖从全开状态纯物理合拢）
    for cmd, tag in ((-1.5708, "open"), (0.0, "closed")):
        mujoco.mj_resetData(m, d)
        set_pose(m, d, POSE_DEMO)
        l7 = jid(m, "L7_joint")
        open_q = -0.0299 * -1.5708
        d.qpos[m.jnt_qposadr[l7]] = -1.5708       # 先摆到全开，避免生成时穿插
        for fn in ("finger_left_joint", "finger_right_joint"):
            fj = jid(m, fn)
            d.qpos[m.jnt_qposadr[fj]] = open_q
        mujoco.mj_forward(m, d)
        close_gripper(m, d, cmd)
        snap(f"{tag}_front", "cam_front")
        snap(f"{tag}_side", "cam_side")
        snap(f"{tag}_top", "cam_top")

    # --- 3: 夹方块：先全开 → 方块放入指间 → 再合拢轻捏（40mm 方块 → L7≈-34°捏住）
    set_pose(m2, d2, POSE_DEMO)
    p2, z2 = grasp_center(m2, d2)
    cj = jid(m2, "demo_cube_joint")
    base = m2.jnt_qposadr[cj]
    l7 = jid(m2, "L7_joint")
    d2.qpos[m2.jnt_qposadr[l7]] = -1.5708
    for fn in ("finger_left_joint", "finger_right_joint"):
        fj = jid(m2, fn)
        d2.qpos[m2.jnt_qposadr[fj]] = -0.0299 * -1.5708
    d2.qpos[base:base + 3] = p2 + z2 * (0.147 - 0.135)
    d2.qpos[base + 3:base + 7] = (1, 0, 0, 0)   # free joint 必须给有效四元数
    mujoco.mj_forward(m2, d2)
    close_gripper(m2, d2, -0.60)
    print("cube@grasp:", np.round(d2.qpos[base:base + 3], 4),
          "fingers mm:", round(d2.qpos[m2.jnt_qposadr[jid(m2, 'finger_left_joint')]] * 1000, 1))
    snap("grasp_front", "cam_front", d2, m2)
    snap("grasp_iso", "cam_iso", d2, m2)


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--gui", action="store_true")
    ap.add_argument("--render", metavar="PREFIX")
    ap.add_argument("--smoke", action="store_true")
    args = ap.parse_args()
    m = load_model()
    if args.render:
        run_render(m, args.render)
        return
    run_viewer(m, gui=args.gui, smoke=args.smoke)


if __name__ == "__main__":
    main()
