"""Automatic ACT demonstration collection inside MuJoCo simulation."""

from __future__ import annotations

import sys
import time
import json
from pathlib import Path

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from act_data.episode_writer import ACTEpisodeWriter, EpisodeMetadata
from act_sim.env import CAMERA_NAMES, ELA3ReachEnv, sample_cube_position
from act_sim.agent import GPTPolicySimAgent


def _next_episode_path(output_dir: Path, task_name: str) -> tuple[int, Path]:
    output_dir.mkdir(parents=True, exist_ok=True)
    prefix = f"episode_{task_name}_"
    existing = sorted(output_dir.glob(f"{prefix}*.hdf5"))
    max_id = -1
    for path in existing:
        try:
            max_id = max(max_id, int(path.stem.rsplit("_", 1)[1]))
        except (IndexError, ValueError):
            continue
    episode_id = max_id + 1
    return episode_id, output_dir / f"{prefix}{episode_id:06d}.hdf5"


def collect_dataset(
    output_dir: str | Path,
    *,
    num_episodes: int,
    task_name: str = "sim_pick",
    image_size: int = 128,
    control_hz: float = 20.0,
    seed: int = 1000,
    qpos_noise: float = 0.008,
    verbose: bool = True,
) -> list[Path]:
    """Collect varied, closed-loop demonstrations into ACT HDF5 episodes.

    Each episode: arm settles at HOME with small joint noise, then the IK
    teacher observes the simulation state, emits structured move/gripper
    decisions, and executes the checked qpos targets.  The teacher adds bounded
    timing and waypoint variation so ACT does not learn one identical path.
    Recorded qpos is the measured joint state; ``object_qpos`` stores the cube
    trajectory so browser replays show the grasp.
    """

    if num_episodes <= 0:
        raise ValueError("num_episodes must be positive")
    if control_hz <= 0:
        raise ValueError("control_hz must be positive")
    if image_size <= 0:
        raise ValueError("image_size must be positive")

    env = ELA3ReachEnv(image_size=image_size, control_hz=control_hz)
    output_dir = Path(output_dir)
    trace_path = output_dir / "agent_trace.jsonl"

    paths: list[Path] = []
    try:
        for episode in range(num_episodes):
            rng = np.random.default_rng(seed + episode)
            cube = sample_cube_position(rng)
            env.reset(cube_pos=cube, qpos_noise=qpos_noise, rng=rng)
            agent = GPTPolicySimAgent(env, rate_hz=control_hz, rng=rng)

            episode_id, output_path = _next_episode_path(output_dir, task_name)
            metadata = EpisodeMetadata(
                task_name=task_name,
                episode_id=episode_id,
                sample_rate_hz=control_hz,
                camera_names=list(CAMERA_NAMES),
                qpos_dim=7,
                action_dim=7,
                sim=True,
                action_semantics="next measured qpos; closed-loop GPT-Policy-style teacher in MuJoCo",
                cube_start=[float(cube[0]), float(cube[1]), float(cube[2])],
                controller="gpt_policy_sim_agent",
            )

            plan = agent.plan_episode(cube, start_qpos=env.qpos)
            metadata.variation = plan.variation
            decisions = []
            with ACTEpisodeWriter(output_path, metadata) as writer:
                phase_starts = {int(phase["start"]): phase for phase in plan.phases}
                for step, ref in enumerate(plan.refs):
                    if step in phase_starts:
                        observation = agent.observe(step)
                        decision = agent.decide(step, ref)
                        decisions.append(
                            {
                                "observation": observation.as_dict(),
                                "decision": decision,
                                "camera_observations": list(CAMERA_NAMES),
                            }
                        )
                    env.set_ctrl(ref)
                    env.step()
                    cube_qpos = env.data.qpos[env.cube_qpos_adr : env.cube_qpos_adr + 7]
                    writer.append(
                        qpos=env.qpos,
                        object_qpos=cube_qpos,
                        images=env.render(),
                        timestamp=env.sim_time,
                    )
            paths.append(output_path)
            with trace_path.open("a", encoding="utf-8") as trace:
                trace.write(
                    json.dumps(
                        {
                            "episode_id": episode_id,
                            "path": str(output_path),
                            "controller": "gpt_policy_sim_agent",
                            "cube_start": cube[:3].astype(float).tolist(),
                            "variation": plan.variation,
                            "decisions": decisions,
                        },
                        ensure_ascii=False,
                    )
                    + "\n"
                )
            if verbose:
                print(
                    f"episode {episode_id:06d}: {writer.count} steps, "
                    f"cube at [{cube[0]:+.3f}, {cube[1]:+.3f}], lift "
                    f"{env.cube_lift_height * 100:.1f} cm, grasped={env.grasped}",
                    flush=True,
                )
    finally:
        env.close()
    return paths


def main(argv: list[str] | None = None) -> int:
    import argparse

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", default="data/act_sim")
    parser.add_argument("--episodes", type=int, default=24)
    parser.add_argument("--task-name", default="sim_pick")
    parser.add_argument("--image-size", type=int, default=128)
    parser.add_argument("--rate", type=float, default=20.0, help="control/collection rate in Hz")
    parser.add_argument("--seed", type=int, default=1000)
    parser.add_argument("--qpos-noise", type=float, default=0.008)
    args = parser.parse_args(argv)

    start = time.perf_counter()
    paths = collect_dataset(
        args.output_dir,
        num_episodes=args.episodes,
        task_name=args.task_name,
        image_size=args.image_size,
        control_hz=args.rate,
        seed=args.seed,
        qpos_noise=args.qpos_noise,
    )
    elapsed = time.perf_counter() - start
    print(f"collected {len(paths)} episodes in {elapsed:.1f}s -> {args.output_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
