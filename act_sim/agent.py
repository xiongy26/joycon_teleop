"""Closed-loop, GPT-Policy-inspired demonstration agent for MuJoCo.

GPT-Policy's useful idea for this task is the control boundary: observe fresh
camera/state feedback, emit one structured action, execute it through a checked
controller, then observe again.  This module keeps that boundary local and
deterministic for dataset generation.  It does not pretend to call a cloud VLM;
the bounded stochastic planner is the teacher that supplies varied ACT demos.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np

from act_sim.expert import ExpertPlan, ReachExpert


@dataclass
class AgentObservation:
    """The state a VLM-style controller would receive on one decision turn."""

    step: int
    qpos: np.ndarray
    tip_xyz: np.ndarray
    cube_xyz: np.ndarray
    tip_cube_distance_m: float
    grasped: bool
    cube_lift_height_m: float

    def as_dict(self) -> dict[str, Any]:
        return {
            "step": self.step,
            "qpos": self.qpos.astype(float).tolist(),
            "tip_xyz": self.tip_xyz.astype(float).tolist(),
            "cube_xyz": self.cube_xyz.astype(float).tolist(),
            "tip_cube_distance_m": float(self.tip_cube_distance_m),
            "grasped": bool(self.grasped),
            "cube_lift_height_m": float(self.cube_lift_height_m),
        }


class GPTPolicySimAgent:
    """A local observe-decide-execute wrapper around the IK teacher."""

    def __init__(self, env, *, rate_hz: float, rng: np.random.Generator):
        self.env = env
        self.rng = rng
        self.expert = ReachExpert(env, rate_hz=rate_hz)
        self.plan: ExpertPlan | None = None

    def plan_episode(self, cube_xyz: np.ndarray, *, start_qpos: np.ndarray) -> ExpertPlan:
        self.plan = self.expert.plan(
            cube_xyz,
            start_qpos=start_qpos,
            rng=self.rng,
            humanize=True,
        )
        return self.plan

    def observe(self, step: int) -> AgentObservation:
        """Read measured state immediately before a new structured action."""

        return AgentObservation(
            step=int(step),
            qpos=self.env.qpos,
            tip_xyz=self.env.tip_position,
            cube_xyz=self.env.cube_position,
            tip_cube_distance_m=self.env.tip_cube_distance,
            grasped=self.env.grasped,
            cube_lift_height_m=self.env.cube_lift_height,
        )

    def decide(self, step: int, target_qpos: np.ndarray) -> dict[str, Any]:
        """Emit one JSON-like tool call for the current phase."""

        if self.plan is None:
            raise RuntimeError("plan_episode must be called before decide")
        phase = next(
            (item for item in self.plan.phases if item["start"] <= step < item["end"]),
            self.plan.phases[-1],
        )
        tool = "set_gripper" if phase["name"] in {"close", "hold", "settle"} else "move_eef_chunk"
        args: dict[str, Any] = {
            "target_qpos": np.asarray(target_qpos, dtype=np.float64).tolist(),
            "note": phase["note"],
        }
        if tool == "set_gripper":
            args["gripper_target"] = float(target_qpos[6])
        return {"name": tool, "phase": phase["name"], "arguments": args}
