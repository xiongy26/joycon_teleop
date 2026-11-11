"""Interactive viewer for the standalone YAM scene.

Run:  python view.py          (needs a display / OpenGL context)

Resets to the "home_open" keyframe, then drives a gentle sinusoidal wave on
the six arm joints while the gripper opens and closes, so you immediately see
the model moving in real physics. Stop with the viewer window close button
or Ctrl+C.

scene.xml already carries the physics-enablement tweaks (collision exclude,
keyframe, joint damping) — no model edits needed here.
"""
import os
import time

import numpy as np
import mujoco
import mujoco.viewer

HERE = os.path.dirname(os.path.abspath(__file__))
SCENE = os.path.join(HERE, 'yam', 'scene.xml')

model = mujoco.MjModel.from_xml_path(SCENE)
data = mujoco.MjData(model)
mujoco.mj_resetDataKeyframe(model, data, 0)   # home_open
mujoco.mj_forward(model, data)

with mujoco.viewer.launch_passive(model, data) as viewer:
    t0 = time.time()
    while viewer.is_running():
        t = time.time() - t0
        # six arm joints: small phase-shifted sinusoids
        for i in range(6):
            data.ctrl[i] = 0.3 * np.sin(0.8 * t + 0.7 * i)
        # gripper: slide joints between closed-on-empty (~0.024) and open
        grip = 0.012 + 0.012 * (1 + np.sin(1.2 * t))    # 0.024 .. 0.036
        data.ctrl[6] = data.ctrl[7] = grip
        mujoco.mj_step(model, data)
        viewer.sync()
        time.sleep(0.002)
