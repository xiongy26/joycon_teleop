"""Headless self-test: load both models and verify the arm actually works.

Run:  python test_load.py
Exits non-zero if anything fails, so it can be used as a smoke test after
setting up a fresh environment.

yam_full.xml is verified for loading/structure; scene.xml is verified for a
full physics rollout (it carries the physics-enablement tweaks — see
yam/scene.xml and README.md: base/link1 collision exclude, home_open
keyframe, joint damping default).
"""
import os
import sys

import numpy as np
import mujoco

HERE = os.path.dirname(os.path.abspath(__file__))
ARM = os.path.join(HERE, 'yam', 'yam_full.xml')
SCENE = os.path.join(HERE, 'yam', 'scene.xml')


def check(cond, msg):
    if not cond:
        print(f'FAIL: {msg}')
        sys.exit(1)
    print(f'ok: {msg}')


def structure(model, label):
    joints = [mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_JOINT, j)
              for j in range(model.njnt)]
    actuators = [mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_ACTUATOR, a)
                 for a in range(model.nu)]
    print(f'\n[{label}] nq={model.nq} nv={model.nv} nu={model.nu} '
          f'nbody={model.nbody} ngeom={model.ngeom}')
    print('  joints:   ', ', '.join(joints))
    print('  actuators:', ', '.join(actuators))


print(f'mujoco {mujoco.__version__}')

# --- 1. the bare arm model loads standalone -------------------------------
arm_model = mujoco.MjModel.from_xml_path(ARM)
structure(arm_model, 'yam_full.xml (load only)')
check(arm_model.njnt == 8 and arm_model.nu == 8,
      'yam_full.xml: 8 joints / 8 actuators as documented')

# --- 2. the physics-ready scene: full rollout -----------------------------
model = mujoco.MjModel.from_xml_path(SCENE)
data = mujoco.MjData(model)
structure(model, 'scene.xml (physics rollout)')

check(model.dof_damping.min() > 0,
      f'scene.xml: joint damping inherited from default '
      f'(min {model.dof_damping.min():.1f})')

kid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_KEY, 'home_open')
check(kid >= 0, 'scene.xml: keyframe home_open exists')
mujoco.mj_resetDataKeyframe(model, data, kid)
mujoco.mj_forward(model, data)

# phase 1: track a small command on each arm joint and settle
commands = [0.3, 0.4, 0.3, -0.2, 0.2, 0.2]     # within every joint's range
data.ctrl[:6] = commands
data.ctrl[6:8] = 0.012                          # close gripper on empty space
snap = None
for i in range(6):                              # 3 s total
    for _ in range(250):
        mujoco.mj_step(model, data)
    snap = data.qpos[:6].copy()
check(np.all(np.isfinite(data.qpos)), 'scene.xml: qpos finite')
errs = np.abs(snap - np.array(commands))
check(np.all(errs < 0.2),
      f'scene.xml: all six arm joints track commands '
      f'(max err {errs.max():.3f} rad — residual is gravity sag at stock kp)')
print('   qpos[0:6]:', np.round(snap, 3))

# phase 2: steady state — the damped arm must hold still
for _ in range(250):                            # +0.5 s
    mujoco.mj_step(model, data)
drift = np.abs(data.qpos[:6] - snap).max()
check(drift < 0.01,
      f'scene.xml: arm holds still at steady state (1 s drift {drift:.5f} rad)')

# phase 3: gripper closed on empty space settles where the jaws touch
q7 = data.qpos[model.joint('joint7').qposadr[0]]
check(0.015 < q7 < 0.035,
      f'scene.xml: gripper holds closed stably, joint7={q7:.4f} m '
      '(jaws touching, no jam)')
print(f'   grip site world pos: {np.round(data.site("grip").xpos, 3)}')

print('\nALL CHECKS PASSED')
