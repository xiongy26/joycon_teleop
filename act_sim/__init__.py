"""MuJoCo simulation sandbox for EL-A3 ACT pipeline validation."""

from __future__ import annotations

import os

# Offscreen rendering backend must be selected before mujoco is imported.
# EGL works headless; override by exporting MUJOCO_GL before running.
os.environ.setdefault("MUJOCO_GL", "egl")
