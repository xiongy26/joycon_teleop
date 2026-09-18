"""Closed-loop evaluation of a trained ACT policy inside MuJoCo simulation."""

from __future__ import annotations

import json
import sys
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from act_sim.env import (
    CAMERA_NAMES,
    ELA3ReachEnv,
    HOME_QPOS,
    SUCCESS_DISTANCE,
    sample_cube_position,
)

# Deterministic held-out cube positions (4 x 3 grid inside the training window).
EVAL_GRID_X = (-0.09, -0.03, 0.03, 0.09)
EVAL_GRID_Y = (-0.40, -0.34, -0.28)


def eval_cube_positions(count: int) -> list[np.ndarray]:
    positions = []
    for i in range(count):
        x = EVAL_GRID_X[i % len(EVAL_GRID_X)]
        y = EVAL_GRID_Y[(i // len(EVAL_GRID_X)) % len(EVAL_GRID_Y)]
        yaw = ((i * 137) % 360 - 180) * np.pi / 180.0
        quat = np.array([np.cos(yaw / 2), 0.0, 0.0, np.sin(yaw / 2)])
        positions.append(np.array([x, y, 0.102, *quat]))
    return positions


@dataclass
class EpisodeMetrics:
    episode: int
    cube_x: float
    cube_y: float
    final_distance_cm: float
    min_last20_distance_cm: float
    success: bool
    horizon_steps: int


def load_policy(checkpoint_path: str | Path, device: str):
    """Load a checkpoint saved by scripts/train_act_policy.py."""

    import torch

    from act_data.policy import ACTPolicy

    checkpoint_path = Path(checkpoint_path)
    payload = torch.load(checkpoint_path, map_location="cpu")
    model_config = dict(payload.get("model_config") or {})
    train_config_path = checkpoint_path.parent / "train_config.json"
    if train_config_path.exists():
        train_config = json.loads(train_config_path.read_text())
        model_config = {**train_config.get("model_config", {}), **model_config}
    required = (
        "qpos_dim",
        "action_dim",
        "num_cameras",
        "chunk_size",
        "hidden_dim",
        "nheads",
        "num_encoder_layers",
        "num_decoder_layers",
        "dropout",
    )
    missing = [key for key in required if key not in model_config]
    if missing:
        raise RuntimeError(f"checkpoint missing model config keys: {missing}")
    policy = ACTPolicy(**{key: model_config[key] for key in required})
    policy.load_state_dict(payload["model"])
    policy.to(device)
    policy.eval()
    return policy, model_config


def _prepare_observation(images: dict[str, np.ndarray], qpos: np.ndarray, downsample: int, device):
    import torch

    tensors = []
    for name in CAMERA_NAMES:
        image = images[name]
        if downsample > 1:
            image = image[::downsample, ::downsample]
        tensors.append(np.ascontiguousarray(image.transpose(2, 0, 1)))
    batch = np.stack(tensors, axis=0).astype(np.float32) / 255.0
    image_tensor = torch.from_numpy(batch).unsqueeze(0).to(device)
    qpos_tensor = torch.from_numpy(np.asarray(qpos, dtype=np.float32)).unsqueeze(0).to(device)
    return image_tensor, qpos_tensor


def run_rollout(
    env: ELA3ReachEnv,
    policy,
    *,
    cube_pos: np.ndarray,
    horizon: int,
    query_interval: int,
    device: str,
    image_downsample: int,
    seed: int = 0,
    record_frames: bool = False,
) -> tuple[EpisodeMetrics, list[tuple[np.ndarray, np.ndarray]], dict]:
    """One closed-loop episode: replan every query_interval steps, execute chunk."""

    import torch

    rng = np.random.default_rng(seed)
    env.reset(cube_pos=cube_pos, qpos_noise=0.008, rng=rng)

    chunk: np.ndarray | None = None
    distances: list[float] = []
    ctrl_log: list[list[float]] = []
    qpos_log: list[list[float]] = []
    frames: list[tuple[np.ndarray, np.ndarray]] = []

    for step in range(horizon):
        if step % query_interval == 0:
            images = env.render()
            image_tensor, qpos_tensor = _prepare_observation(
                images, env.qpos, image_downsample, device
            )
            with torch.inference_mode():
                chunk = policy(image_tensor, qpos_tensor)[0].cpu().numpy()
        action = chunk[min(step % query_interval, chunk.shape[0] - 1)]
        env.set_ctrl(action)
        env.step()
        distances.append(env.tip_cube_distance)
        ctrl_log.append(np.asarray(action, dtype=np.float32).tolist())
        qpos_log.append(env.qpos.astype(np.float32).tolist())
        if record_frames:
            frames.append(env.render_pair())

    distances = np.asarray(distances)
    tail = distances[int(len(distances) * 0.8):]
    metrics = EpisodeMetrics(
        episode=-1,
        cube_x=float(cube_pos[0]),
        cube_y=float(cube_pos[1]),
        final_distance_cm=float(distances[-1] * 100),
        min_last20_distance_cm=float(tail.min() * 100) if len(tail) else float(distances.min() * 100),
        success=bool(tail.min() < SUCCESS_DISTANCE) if len(tail) else bool(distances[-1] < SUCCESS_DISTANCE),
        horizon_steps=int(horizon),
    )
    trace = {
        "ctrl": np.asarray(ctrl_log, dtype=np.float32),
        "qpos": np.asarray(qpos_log, dtype=np.float32),
        "dist": distances.astype(np.float32),
    }
    return metrics, frames, trace


def evaluate_checkpoint(
    checkpoint_path: str | Path,
    *,
    output_dir: str | Path,
    num_episodes: int = 12,
    horizon: int = 130,
    query_interval: int = 10,
    device: str | None = None,
    image_size: int = 128,
    control_hz: float = 20.0,
    record_video_episodes: int = 2,
) -> dict:
    """Run the full eval suite and write metrics + optional rollout videos."""

    import torch

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"

    policy, model_config = load_policy(checkpoint_path, device)
    downsample = int(model_config.get("image_downsample", 1) or 1)

    env = ELA3ReachEnv(image_size=image_size, control_hz=control_hz)
    results: list[EpisodeMetrics] = []
    traces: dict[str, np.ndarray] = {}
    try:
        for i, cube_pos in enumerate(eval_cube_positions(num_episodes)):
            record = i < record_video_episodes
            metrics, frames, trace = run_rollout(
                env,
                policy,
                cube_pos=cube_pos,
                horizon=horizon,
                query_interval=query_interval,
                device=device,
                image_downsample=downsample,
                seed=777 + i,
                record_frames=record,
            )
            metrics.episode = i
            results.append(metrics)
            traces[f"cube_{i}"] = np.asarray(cube_pos[:3], dtype=np.float32)
            traces[f"ctrl_{i}"] = trace["ctrl"]
            traces[f"qpos_{i}"] = trace["qpos"]
            traces[f"dist_{i}"] = trace["dist"]
            print(
                f"eval episode {i:02d}: cube=({metrics.cube_x:+.2f},{metrics.cube_y:+.2f}) "
                f"final={metrics.final_distance_cm:.1f}cm "
                f"tail_min={metrics.min_last20_distance_cm:.1f}cm "
                f"{'SUCCESS' if metrics.success else 'FAIL'}",
                flush=True,
            )
            if record:
                _write_video(output_dir / f"rollout_{i:02d}.mp4", frames, fps=int(control_hz))
    finally:
        env.close()

    successes = [r.success for r in results]
    finals = np.array([r.final_distance_cm for r in results])
    summary = {
        "checkpoint": str(checkpoint_path),
        "device": device,
        "num_episodes": len(results),
        "success_rate": float(np.mean(successes)),
        "successes": int(np.sum(successes)),
        "success_distance_cm": SUCCESS_DISTANCE * 100,
        "mean_final_distance_cm": float(finals.mean()),
        "median_final_distance_cm": float(np.median(finals)),
        "worst_final_distance_cm": float(finals.max()),
        "horizon": horizon,
        "query_interval": query_interval,
        "episodes": [asdict(r) for r in results],
    }
    (output_dir / "eval_results.json").write_text(json.dumps(summary, indent=2) + "\n")
    np.savez_compressed(output_dir / "rollout_traces.npz", **traces)
    return summary


def _write_video(path: Path, frames: list[tuple[np.ndarray, np.ndarray]], fps: int) -> None:
    """Side-by-side cam_high | cam_wrist mp4; falls back to a PNG contact sheet."""

    if not frames:
        return
    try:
        import imageio

        stacked = [np.concatenate([high, wrist], axis=1) for high, wrist in frames]
        imageio.mimsave(path.as_posix(), stacked, fps=fps)
        return
    except Exception as exc:  # pragma: no cover - environment dependent
        print(f"video write failed ({exc}); saving contact sheet instead")
    import PIL.Image

    middle = frames[len(frames) // 2]
    sheet = np.concatenate([middle[0], middle[1]], axis=1)
    PIL.Image.fromarray(sheet).save(path.with_suffix(".png").as_posix())


def main(argv: list[str] | None = None) -> int:
    import argparse

    parser = argparse.ArgumentParser(description="Evaluate a trained ACT policy in MuJoCo simulation.")
    parser.add_argument("checkpoint", help="Checkpoint path, e.g. runs/act_sim_x/ckpt/policy_best.pt")
    parser.add_argument("--output-dir", default=None, help="Defaults to <checkpoint parent>/../eval")
    parser.add_argument("--episodes", type=int, default=12)
    parser.add_argument("--horizon", type=int, default=130)
    parser.add_argument("--query-interval", type=int, default=10)
    parser.add_argument("--device", default=None)
    parser.add_argument("--image-size", type=int, default=128)
    parser.add_argument("--rate", type=float, default=20.0)
    parser.add_argument("--record-video-episodes", type=int, default=2)
    args = parser.parse_args(argv)

    output_dir = Path(args.output_dir) if args.output_dir else Path(args.checkpoint).parent.parent / "eval"
    summary = evaluate_checkpoint(
        args.checkpoint,
        output_dir=output_dir,
        num_episodes=args.episodes,
        horizon=args.horizon,
        query_interval=args.query_interval,
        device=args.device,
        image_size=args.image_size,
        control_hz=args.rate,
        record_video_episodes=args.record_video_episodes,
    )
    print(
        f"success {summary['successes']}/{summary['num_episodes']} "
        f"({summary['success_rate'] * 100:.0f}%), mean final "
        f"{summary['mean_final_distance_cm']:.1f} cm -> {output_dir}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
