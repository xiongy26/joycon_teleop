#!/usr/bin/env python3
"""One-command ACT validation pipeline in MuJoCo simulation.

Runs, in order:
  1. closed-loop demonstration collection with a GPT-Policy-style local teacher,
  2. dataset inspection (existing inspect_act_dataset.py),
  3. ACT training (existing scripts/train_act_policy.py, unmodified),
  4. closed-loop policy rollouts in MuJoCo with success-rate reporting.

Everything lands under --out-dir (default: runs/act_sim_<timestamp>).
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
import time
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

CAMERAS = "cam_high,cam_wrist"


def _run(cmd: list[str], log_path: Path | None = None) -> None:
    print(f"\n$ {' '.join(str(c) for c in cmd)}", flush=True)
    if log_path is None:
        subprocess.run([str(c) for c in cmd], check=True)
        return
    with log_path.open("w") as log:
        proc = subprocess.run([str(c) for c in cmd], stdout=log, stderr=subprocess.STDOUT)
    if proc.returncode != 0:
        print(log_path.read_text()[-4000:])
        raise RuntimeError(f"command failed, see {log_path}")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out-dir", default=None, help="Pipeline output directory")
    parser.add_argument("--episodes", type=int, default=80, help="Demonstrations to collect")
    parser.add_argument("--epochs", type=int, default=150)
    parser.add_argument("--chunk-size", type=int, default=20)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--image-size", type=int, default=128)
    parser.add_argument("--image-downsample", type=int, default=1)
    parser.add_argument("--rate", type=float, default=20.0)
    parser.add_argument("--eval-episodes", type=int, default=12)
    parser.add_argument(
        "--horizon",
        type=int,
        default=None,
        help="evaluation control steps (default: max collected episode length + 10)",
    )
    parser.add_argument("--query-interval", type=int, default=10)
    parser.add_argument("--max-target-delta", type=float, default=0.08)
    parser.add_argument("--device", default=None, help="torch device (default: cuda if available)")
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--seed", type=int, default=1000)
    parser.add_argument(
        "--quick",
        action="store_true",
        help="Tiny smoke-test configuration (few episodes/epochs/evals)",
    )
    parser.add_argument("--skip-collect", action="store_true")
    parser.add_argument("--skip-train", action="store_true")
    args = parser.parse_args(argv)

    if args.quick:
        args.episodes = min(args.episodes, 3)
        args.epochs = min(args.epochs, 3)
        args.eval_episodes = min(args.eval_episodes, 2)
        args.num_workers = 0

    if args.episodes <= 0:
        parser.error("--episodes must be positive")
    if args.epochs <= 0:
        parser.error("--epochs must be positive")
    if args.eval_episodes <= 0:
        parser.error("--eval-episodes must be positive")
    if args.horizon is not None and args.horizon <= 0:
        parser.error("--horizon must be positive")
    if args.chunk_size <= 0:
        parser.error("--chunk-size must be positive")
    if args.batch_size <= 0:
        parser.error("--batch-size must be positive")
    if args.rate <= 0:
        parser.error("--rate must be positive")
    if args.query_interval <= 0:
        parser.error("--query-interval must be positive")
    if args.max_target_delta < 0:
        parser.error("--max-target-delta cannot be negative")

    stamp = time.strftime("%Y%m%d_%H%M%S")
    out_dir = Path(args.out_dir) if args.out_dir else Path("runs") / f"act_sim_{stamp}"
    data_dir = out_dir / "data"
    ckpt_dir = out_dir / "ckpt"
    eval_dir = out_dir / "eval"
    for path in (data_dir, ckpt_dir, eval_dir):
        path.mkdir(parents=True, exist_ok=True)
    device = args.device
    if device is None:
        import torch

        device = "cuda" if torch.cuda.is_available() else "cpu"

    print(f"pipeline output: {out_dir.resolve()} (device={device})")

    # 1. automatic data collection ------------------------------------
    if not args.skip_collect:
        from act_sim.collect import collect_dataset

        start = time.perf_counter()
        paths = collect_dataset(
            data_dir,
            num_episodes=args.episodes,
            image_size=args.image_size,
            control_hz=args.rate,
            seed=args.seed,
        )
        print(f"collected {len(paths)} episodes in {time.perf_counter() - start:.1f}s")
    else:
        paths = sorted(data_dir.glob("episode_*.hdf5"))
        if not paths:
            raise RuntimeError(f"--skip-collect but no episodes found in {data_dir}")

    if args.horizon is None:
        import h5py

        with h5py.File(paths[0], "r") as h5:
            max_episode_steps = int(h5["action"].shape[0])
        # Humanized plans can be longer than the old fixed 130-step horizon.
        # Use all collected steps plus a short settling tail.
        for path in paths[1:]:
            with h5py.File(path, "r") as h5:
                max_episode_steps = max(max_episode_steps, int(h5["action"].shape[0]))
        args.horizon = max(130, max_episode_steps + 10)

    # 2. dataset inspection (existing tooling) ------------------------
    _run(
        [sys.executable, PROJECT_ROOT / "scripts" / "inspect_act_dataset.py", paths[0]],
        log_path=out_dir / "inspect.log",
    )

    # 3. training (existing script, unmodified) -----------------------
    checkpoint = ckpt_dir / "policy_best.pt"
    if not args.skip_train:
        train_cmd = [
            sys.executable,
            PROJECT_ROOT / "scripts" / "train_act_policy.py",
            "--dataset",
            data_dir,
            "--output-dir",
            ckpt_dir,
            "--cameras",
            CAMERAS,
            "--chunk-size",
            args.chunk_size,
            "--image-downsample",
            args.image_downsample,
            "--batch-size",
            args.batch_size,
            "--epochs",
            args.epochs,
            "--num-workers",
            args.num_workers,
            "--device",
            device,
            "--save-every",
            max(10, args.epochs // 4),
        ]
        _run(train_cmd, log_path=out_dir / "train.log")
    if not checkpoint.exists():
        raise RuntimeError(f"training did not produce {checkpoint}")

    # 4. closed-loop evaluation ---------------------------------------
    from act_sim.evaluate import evaluate_checkpoint

    summary = evaluate_checkpoint(
        checkpoint,
        output_dir=eval_dir,
        num_episodes=args.eval_episodes,
        horizon=args.horizon,
        query_interval=args.query_interval,
        max_target_delta=args.max_target_delta,
        device=device,
        image_size=args.image_size,
        control_hz=args.rate,
    )

    report = {
        "generated_at": time.strftime("%Y-%m-%d %H:%M:%S"),
        "out_dir": str(out_dir.resolve()),
        "config": {
            "episodes": args.episodes if not args.skip_collect else "skipped",
            "epochs": args.epochs if not args.skip_train else "skipped",
            "chunk_size": args.chunk_size,
            "image_size": args.image_size,
            "control_hz": args.rate,
            "query_interval": args.query_interval,
            "horizon": args.horizon,
            "max_target_delta": args.max_target_delta,
            "device": device,
        },
        "eval": summary,
    }
    (out_dir / "pipeline_report.json").write_text(json.dumps(report, indent=2) + "\n")
    _write_markdown_report(out_dir, report)

    # 5. export replay data for the MuJoCo-WASM web viewer (best effort) ----
    try:
        from scripts.export_act_sim_web import export_for_viewer

        web_summary = export_for_viewer(out_dir)
        report["web_export"] = web_summary
        (out_dir / "pipeline_report.json").write_text(json.dumps(report, indent=2) + "\n")
        print(f"web viewer data exported -> web/data ({web_summary['episodes']} episodes, "
              f"{web_summary['rollouts']} rollouts)")
    except Exception as exc:
        print(f"web viewer export skipped: {exc}")

    print(
        f"\n==== RESULT: success {summary['successes']}/{summary['num_episodes']} "
        f"({summary['success_rate'] * 100:.0f}%), "
        f"mean final distance {summary['mean_final_distance_cm']:.1f} cm ===="
    )
    print(f"report: {out_dir / 'report.md'}")
    return 0


def _write_markdown_report(out_dir: Path, report: dict) -> None:
    ev = report["eval"]
    lines = [
        "# ACT MuJoCo 仿真验证报告",
        "",
        f"- 生成时间: {report['generated_at']}",
        f"- 输出目录: `{report['out_dir']}`",
        "",
        "## 管线配置",
        "",
    ]
    for key, value in report["config"].items():
        lines.append(f"- {key}: {value}")
    lines += [
        "",
        "## 数据与训练",
        "",
        f"- 演示数据: `{report['out_dir']}/data` (观测→决策→执行的仿真教师采集)",
        f"- 决策轨迹: `{report['out_dir']}/data/agent_trace.jsonl` (每个阶段的状态与结构化动作)",
        f"- 训练日志: `train.log`, 最优 checkpoint: `ckpt/policy_best.pt`",
        "",
        "## 闭环评估结果",
        "",
        f"- 成功率: **{ev['successes']}/{ev['num_episodes']} ({ev['success_rate'] * 100:.0f}%)**"
        f"（方块抬升至少 4 cm 且保持夹住）",
        f"- 平均最终距离: {ev['mean_final_distance_cm']:.2f} cm",
        f"- 中位最终距离: {ev['median_final_distance_cm']:.2f} cm",
        f"- 最差最终距离: {ev['worst_final_distance_cm']:.2f} cm",
        "",
        "| episode | cube_x | cube_y | final (cm) | tail min (cm) | tail under threshold | result |",
        "|---|---|---|---|---|---:|---|",
    ]
    for ep in ev["episodes"]:
        lines.append(
            f"| {ep['episode']} | {ep['cube_x']:+.2f} | {ep['cube_y']:+.2f} "
            f"| {ep['final_distance_cm']:.2f} | {ep['min_last20_distance_cm']:.2f} "
            f"| {ep['tail_success_ratio'] * 100:.0f}% "
            f"| {'成功' if ep['success'] else '失败'} |"
        )
    lines += [
        "",
        "评估视频: `eval/rollout_00.mp4`, `eval/rollout_01.mp4` (左 cam_high / 右 cam_wrist)",
        "",
    ]
    (out_dir / "report.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


if __name__ == "__main__":
    raise SystemExit(main())
