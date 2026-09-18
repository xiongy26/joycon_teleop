"""HDF5 writer for ACT/ALOHA-style demonstration episodes."""

from __future__ import annotations

import json
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Dict, Optional

import numpy as np


@dataclass
class EpisodeMetadata:
    """Metadata stored as HDF5 attributes and JSON for each episode."""

    task_name: str
    episode_id: int
    sample_rate_hz: float
    camera_names: list[str]
    qpos_dim: int = 7
    action_dim: int = 7
    created_at: str = field(
        default_factory=lambda: time.strftime("%Y-%m-%d %H:%M:%S")
    )
    robot: str = "EL-A3"
    sim: bool = False
    action_semantics: str = "next qpos target; final action duplicates final qpos"
    cube_start: Optional[list] = None


class ACTEpisodeWriter:
    """Stream an ACT episode to HDF5 with extendable datasets."""

    def __init__(self, path: str | Path, metadata: EpisodeMetadata):
        self.path = Path(path)
        self.metadata = metadata
        self._h5 = None
        self._qpos_ds = None
        self._timestamps_ds = None
        self._image_datasets: Dict[str, object] = {}
        self._qpos_buffer: list[np.ndarray] = []
        self._t0: Optional[float] = None
        self._count = 0

    def __enter__(self) -> "ACTEpisodeWriter":
        self.open()
        return self

    def __exit__(self, exc_type, _exc, _tb) -> None:
        if exc_type is None:
            self.finalize()
        else:
            self.close()

    @property
    def count(self) -> int:
        return self._count

    def open(self) -> None:
        try:
            import h5py
        except ImportError as exc:
            raise RuntimeError(
                "h5py is required for ACT dataset collection. Install with: pip install h5py"
            ) from exc

        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._h5 = h5py.File(self.path, "w")
        self._h5.attrs["sim"] = self.metadata.sim
        self._h5.attrs["compress"] = False
        self._h5.attrs["metadata_json"] = json.dumps(
            asdict(self.metadata), ensure_ascii=False
        )
        for key, value in asdict(self.metadata).items():
            if isinstance(value, (list, dict)):
                self._h5.attrs[key] = json.dumps(value, ensure_ascii=False)
            else:
                self._h5.attrs[key] = value

        obs = self._h5.create_group("observations")
        obs.create_group("images")
        self._qpos_ds = obs.create_dataset(
            "qpos",
            shape=(0, self.metadata.qpos_dim),
            maxshape=(None, self.metadata.qpos_dim),
            chunks=(1, self.metadata.qpos_dim),
            dtype="f4",
        )
        self._timestamps_ds = self._h5.create_dataset(
            "timestamps",
            shape=(0,),
            maxshape=(None,),
            chunks=(1024,),
            dtype="f8",
        )

    def append(self, *, qpos: np.ndarray, images: Dict[str, np.ndarray], timestamp: Optional[float] = None) -> None:
        if self._h5 is None or self._qpos_ds is None or self._timestamps_ds is None:
            raise RuntimeError("episode writer is not open")

        qpos = np.asarray(qpos, dtype=np.float32)
        if qpos.shape != (self.metadata.qpos_dim,):
            raise ValueError(
                f"qpos must have shape ({self.metadata.qpos_dim},), got {qpos.shape}"
            )
        if np.any(~np.isfinite(qpos)):
            raise ValueError("qpos contains NaN/Inf")

        if set(images) != set(self.metadata.camera_names):
            raise ValueError(
                f"image cameras {sorted(images)} do not match metadata "
                f"{sorted(self.metadata.camera_names)}"
            )

        idx = self._count
        if timestamp is not None:
            stamp = float(timestamp)
        elif self._t0 is None:
            stamp = 0.0
            self._t0 = time.perf_counter()
        else:
            stamp = time.perf_counter() - self._t0

        self._qpos_ds.resize((idx + 1, self.metadata.qpos_dim))
        self._qpos_ds[idx] = qpos
        self._timestamps_ds.resize((idx + 1,))
        self._timestamps_ds[idx] = stamp

        image_group = self._h5["observations"]["images"]
        for name, image in images.items():
            arr = np.asarray(image)
            if arr.ndim != 3 or arr.shape[2] != 3:
                raise ValueError(
                    f"camera {name!r} image must have shape HxWx3, got {arr.shape}"
                )
            if arr.dtype != np.uint8:
                arr = np.clip(arr, 0, 255).astype(np.uint8)

            ds = self._image_datasets.get(name)
            if ds is None:
                ds = image_group.create_dataset(
                    name,
                    shape=(0, arr.shape[0], arr.shape[1], 3),
                    maxshape=(None, arr.shape[0], arr.shape[1], 3),
                    chunks=(1, arr.shape[0], arr.shape[1], 3),
                    dtype="uint8",
                )
                self._image_datasets[name] = ds
            elif ds.shape[1:] != arr.shape:
                raise ValueError(
                    f"camera {name!r} image shape changed from {ds.shape[1:]} "
                    f"to {arr.shape}"
                )
            ds.resize((idx + 1, arr.shape[0], arr.shape[1], 3))
            ds[idx] = arr

        self._qpos_buffer.append(qpos.copy())
        self._count += 1

    def finalize(self) -> None:
        if self._h5 is None:
            return
        if self._count == 0:
            self.close()
            raise RuntimeError("cannot finalize an empty ACT episode")

        qpos = np.vstack(self._qpos_buffer).astype(np.float32)
        action = np.empty((self._count, self.metadata.action_dim), dtype=np.float32)
        action[:-1] = qpos[1:, : self.metadata.action_dim]
        action[-1] = qpos[-1, : self.metadata.action_dim]

        timestamps = np.asarray(self._h5["timestamps"], dtype=np.float64)
        qvel = np.zeros_like(qpos, dtype=np.float32)
        if self._count > 1:
            dt = np.diff(timestamps)
            dt[dt <= 1e-6] = 1.0 / float(self.metadata.sample_rate_hz)
            qvel[1:] = (qpos[1:] - qpos[:-1]) / dt[:, None]
            qvel[0] = qvel[1]

        obs = self._h5["observations"]
        obs.create_dataset("qvel", data=qvel, dtype="f4")
        self._h5.create_dataset("action", data=action, dtype="f4")
        self._h5.attrs["num_samples"] = self._count
        self._h5.attrs["duration_s"] = float(timestamps[-1]) if len(timestamps) else 0.0
        self._h5.flush()
        self.close()

    def close(self) -> None:
        if self._h5 is not None:
            self._h5.close()
            self._h5 = None
