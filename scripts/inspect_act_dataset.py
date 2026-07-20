#!/usr/bin/env python3
"""Inspect an ACT HDF5 episode for basic shape and timing issues."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np


def inspect(path: Path) -> int:
    try:
        import h5py
    except ImportError as exc:
        raise RuntimeError("h5py is required. Install with: pip install h5py") from exc

    with h5py.File(path, "r") as h5:
        qpos = h5["observations/qpos"]
        qvel = h5["observations/qvel"]
        action = h5["action"]
        timestamps = np.asarray(h5["timestamps"])
        image_group = h5["observations/images"]
        metadata = json.loads(h5.attrs.get("metadata_json", "{}"))

        print(f"file: {path}")
        print(f"task: {metadata.get('task_name', '<unknown>')}")
        print(f"samples: {qpos.shape[0]}")
        print(f"qpos: {qpos.shape} {qpos.dtype}")
        print(f"qvel: {qvel.shape} {qvel.dtype}")
        print(f"action: {action.shape} {action.dtype}")
        for name, ds in image_group.items():
            print(f"image/{name}: {ds.shape} {ds.dtype}")

        issues = []
        if qpos.shape != action.shape:
            issues.append(f"qpos/action shape mismatch: {qpos.shape} vs {action.shape}")
        if qvel.shape != qpos.shape:
            issues.append(f"qvel/qpos shape mismatch: {qvel.shape} vs {qpos.shape}")
        if len(timestamps) != qpos.shape[0]:
            issues.append("timestamps length does not match qpos")
        if len(timestamps) > 1:
            dt = np.diff(timestamps)
            bad = np.where(dt <= 0)[0]
            if len(bad):
                issues.append(f"non-increasing timestamps at indices {bad[:10].tolist()}")
            print(f"duration_s: {timestamps[-1]:.3f}")
            print(f"rate_hz mean/min/max: {1 / np.mean(dt):.2f} / {1 / np.max(dt):.2f} / {1 / np.min(dt):.2f}")

        qpos_np = np.asarray(qpos)
        action_np = np.asarray(action)
        if np.any(~np.isfinite(qpos_np)):
            issues.append("qpos contains NaN/Inf")
        if np.any(~np.isfinite(action_np)):
            issues.append("action contains NaN/Inf")

        if issues:
            print("\nissues:")
            for issue in issues:
                print(f"- {issue}")
            return 1

        print("\nOK")
        return 0


def main() -> int:
    parser = argparse.ArgumentParser(description="Inspect an ACT HDF5 episode.")
    parser.add_argument("path", type=Path)
    args = parser.parse_args()
    return inspect(args.path)


if __name__ == "__main__":
    raise SystemExit(main())
