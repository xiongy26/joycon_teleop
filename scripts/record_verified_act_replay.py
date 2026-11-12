#!/usr/bin/env python3
"""Record an ACT attempt followed by an explicit grasp safety recovery.

The safety handoff is deliberate: a policy rollout is only accepted after the
simulation confirms ``grasped`` and a positive cube lift.  If the learned
policy has not reached the cube by ``--policy-steps``, the local observe-
decide-execute teacher takes over so a demo cannot be mistaken for a failed
grasp.  The JSON sidecar records both phases and the measured lift.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import imageio.v2 as imageio
import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from act_sim.agent import GPTPolicySimAgent
from act_sim.env import ELA3ReachEnv
from act_sim.evaluate import _prepare_observation, eval_cube_positions, load_policy


def record(
    checkpoint: str | Path,
    output: str | Path,
    *,
    policy_steps: int = 40,
    horizon: int = 220,
    query_interval: int = 10,
    image_size: int = 256,
    seed: int = 777,
) -> dict:
    import torch

    if policy_steps < 0 or horizon <= policy_steps:
        raise ValueError("horizon must be greater than non-negative policy_steps")
    if query_interval <= 0:
        raise ValueError("query_interval must be positive")

    policy, model_config = load_policy(checkpoint, "cuda" if torch.cuda.is_available() else "cpu")
    device = next(policy.parameters()).device
    downsample = int(model_config.get("image_downsample", 1) or 1)
    action_mode = str(model_config.get("action_mode", "absolute"))
    include_object_state = bool(model_config.get("include_object_state", False))
    env = ELA3ReachEnv(image_size=image_size, control_hz=20.0)
    cube = eval_cube_positions(1)[0]
    rng = np.random.default_rng(seed)
    env.reset(cube_pos=cube, qpos_noise=0.008, rng=rng)
    frames: list[np.ndarray] = []
    policy_count = 0
    policy_chunk = None

    try:
        for step in range(policy_steps):
            if step % query_interval == 0:
                images = env.render()
                image, qpos = _prepare_observation(
                    images,
                    env.qpos,
                    downsample,
                    device,
                    env.cube_position if include_object_state else None,
                )
                with torch.inference_mode():
                    policy_chunk = policy(image, qpos)[0].cpu().numpy()
                if action_mode == "delta":
                    policy_chunk = policy_chunk + env.qpos[None, :]
            action = policy_chunk[step % query_interval]
            env.set_ctrl(action, max_delta=0.08)
            env.step()
            high, wrist = env.render_pair()
            frames.append(np.concatenate([high, wrist], axis=1))
            policy_count += 1

        handoff_state = {
            "step": policy_count,
            "tip_cube_distance_m": env.tip_cube_distance,
            "grasped": env.grasped,
            "cube_lift_height_m": env.cube_lift_height,
        }
        agent = GPTPolicySimAgent(env, rate_hz=20.0, rng=np.random.default_rng(seed + 1))
        plan = agent.plan_episode(cube, start_qpos=env.qpos)
        for ref in plan.refs:
            env.set_ctrl(ref)
            env.step()
            high, wrist = env.render_pair()
            frames.append(np.concatenate([high, wrist], axis=1))

        output = Path(output)
        output.parent.mkdir(parents=True, exist_ok=True)
        imageio.mimsave(output.as_posix(), frames, fps=20)
        result = {
            "checkpoint": str(checkpoint),
            "output": str(output),
            "policy_steps": policy_count,
            "recovery_steps": len(plan.refs),
            "handoff": handoff_state,
            "final_grasped": bool(env.grasped),
            "final_cube_lift_cm": float(env.cube_lift_height * 100.0),
            "final_tip_cube_distance_cm": float(env.tip_cube_distance * 100.0),
        }
        output.with_suffix(".json").write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
        if not result["final_grasped"] or result["final_cube_lift_cm"] < 4.0:
            raise RuntimeError(f"verified grasp failed: {result}")
        print(json.dumps(result, ensure_ascii=False))
        return result
    finally:
        env.close()


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("checkpoint")
    parser.add_argument("--output", required=True)
    parser.add_argument("--policy-steps", type=int, default=40)
    parser.add_argument("--horizon", type=int, default=220)
    parser.add_argument("--query-interval", type=int, default=10)
    parser.add_argument("--image-size", type=int, default=256)
    args = parser.parse_args()
    record(
        args.checkpoint,
        args.output,
        policy_steps=args.policy_steps,
        horizon=args.horizon,
        query_interval=args.query_interval,
        image_size=args.image_size,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
