#!/usr/bin/env python3
"""Export ACT-sim episodes and policy rollouts to JSON for the web viewer."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

WEB_DATA_DIR = PROJECT_ROOT / "web" / "data"


def _round7(rows) -> list[list[float]]:
    return [[round(float(v), 5) for v in row] for row in rows]


def export_episodes(run_dir: Path, seed: int = 1000) -> list[dict]:
    import h5py

    from act_sim.env import sample_cube_position

    episodes = []
    for path in sorted((run_dir / "data").glob("episode_*.hdf5")):
        with h5py.File(path, "r") as h5:
            qpos = np.asarray(h5["observations/qpos"], dtype=np.float32)
            action = np.asarray(h5["action"], dtype=np.float32)
            object_qpos = (
                np.asarray(h5["observations/object_qpos"], dtype=np.float32)
                if "observations/object_qpos" in h5
                else None
            )
            meta = json.loads(h5.attrs.get("metadata_json", "{}"))
        cube_meta = meta.get("cube_start")
        if cube_meta:
            cube = [round(float(v), 3) for v in cube_meta[:2]]
        else:
            # episodes collected before cube_start was recorded: cube positions
            # are deterministic given (seed + episode_id)
            rng = np.random.default_rng(seed + int(meta.get("episode_id", 0)))
            cube = [round(float(v), 3) for v in sample_cube_position(rng)[:2]]
        episodes.append(
            {
                "id": meta.get("episode_id", int(path.stem.rsplit("_", 1)[1])),
                "task": meta.get("task_name", "sim_reach"),
                "cube": cube,
                "T": int(action.shape[0]),
                "rate": float(meta.get("sample_rate_hz", 20.0)),
                # ctrl[t] = action[t] = qpos[t+1]: the expert reference command
                "ctrl": _round7(action),
                "qpos": _round7(qpos),
                **({"object_qpos": _round7(object_qpos)} if object_qpos is not None else {}),
            }
        )
    return episodes


def export_rollouts(run_dir: Path) -> list[dict]:
    traces_path = run_dir / "eval" / "rollout_traces.npz"
    results_path = run_dir / "eval" / "eval_results.json"
    if not traces_path.exists():
        return []
    traces = np.load(traces_path)
    results = json.loads(results_path.read_text()) if results_path.exists() else {}
    meta = {ep["episode"]: ep for ep in results.get("episodes", [])}
    rollouts = []
    # Filter before sorting: rollout_traces also contains cube/qpos/dist keys,
    # whose second token is not an integer (e.g. ``qpos_0``).
    ctrl_keys = [key for key in traces.files if key.startswith("ctrl_")]
    for key in sorted(ctrl_keys, key=lambda s: int(s.split("_")[1])):
        i = int(key.split("_")[1])
        ep = meta.get(i, {})
        rollouts.append(
            {
                "i": i,
                "cube": [round(float(v), 3) for v in traces[f"cube_{i}"]],
                "success": bool(ep.get("success", False)),
                "T": int(traces[key].shape[0]),
                "rate": 20.0,
                "ctrl": _round7(traces[key]),
                "qpos": _round7(traces[f"qpos_{i}"]),
                "object_qpos": _round7(traces[f"cube_qpos_{i}"]) if f"cube_qpos_{i}" in traces else [],
                "dist": [round(float(v), 4) for v in traces[f"dist_{i}"]],
            }
        )
    return rollouts


def export_for_viewer(run_dir: str | Path, out_dir: str | Path = WEB_DATA_DIR, seed: int = 1000) -> dict:
    run_dir = Path(run_dir)
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    episodes = export_episodes(run_dir, seed=seed)
    rollouts = export_rollouts(run_dir)

    (out_dir / "episodes.json").write_text(
        json.dumps({"run_dir": str(run_dir), "episodes": episodes}, ensure_ascii=False)
    )
    (out_dir / "rollouts.json").write_text(
        json.dumps({"run_dir": str(run_dir), "rollouts": rollouts}, ensure_ascii=False)
    )
    summary = {
        "run_dir": str(run_dir),
        "episodes": len(episodes),
        "rollouts": len(rollouts),
        "out_dir": str(out_dir),
    }
    if (run_dir / "eval" / "eval_results.json").exists():
        results = json.loads((run_dir / "eval" / "eval_results.json").read_text())
        summary["success_rate"] = results.get("success_rate")
    print(json.dumps(summary, ensure_ascii=False))
    return summary


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("run_dir", help="Pipeline run directory (contains data/, eval/)")
    parser.add_argument("--out-dir", default=str(WEB_DATA_DIR))
    args = parser.parse_args(argv)
    export_for_viewer(args.run_dir, args.out_dir)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
