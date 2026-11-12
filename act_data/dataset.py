"""PyTorch datasets for ACT-style EL-A3 HDF5 episodes."""

from __future__ import annotations

import random
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import numpy as np


@dataclass(frozen=True)
class EpisodeIndex:
    path: Path
    timestep: int
    length: int


def discover_episode_files(paths: list[str | Path]) -> list[Path]:
    """Expand files/directories into sorted HDF5 episode paths."""

    episodes: list[Path] = []
    for raw in paths:
        path = Path(raw)
        if path.is_dir():
            episodes.extend(sorted(path.glob("*.hdf5")))
            episodes.extend(sorted(path.glob("*.h5")))
        elif path.is_file():
            episodes.append(path)
        else:
            raise FileNotFoundError(f"dataset path does not exist: {path}")
    unique = sorted(set(episodes))
    if not unique:
        raise FileNotFoundError(f"no .hdf5/.h5 episodes found in {paths}")
    return unique


class ACTHDF5Dataset:
    """Map-style dataset returning one timestep and a padded action chunk."""

    def __init__(
        self,
        episode_paths: list[str | Path],
        *,
        chunk_size: int = 50,
        camera_names: Optional[list[str]] = None,
        image_downsample: int = 1,
        normalize_images: bool = True,
        action_mode: str = "absolute",
        include_object_state: bool = False,
        seed: int = 0,
    ):
        if chunk_size <= 0:
            raise ValueError("chunk_size must be positive")
        if image_downsample <= 0:
            raise ValueError("image_downsample must be positive")
        if action_mode not in {"absolute", "delta"}:
            raise ValueError("action_mode must be 'absolute' or 'delta'")

        self.episode_paths = discover_episode_files(episode_paths)
        self.chunk_size = int(chunk_size)
        self.camera_names = camera_names
        self.image_downsample = int(image_downsample)
        self.normalize_images = normalize_images
        self.action_mode = action_mode
        self.include_object_state = bool(include_object_state)
        self.seed = int(seed)
        self.index: list[EpisodeIndex] = []
        self._episode_lengths: dict[Path, int] = {}
        self._build_index()

    def _build_index(self) -> None:
        import h5py

        for path in self.episode_paths:
            with h5py.File(path, "r") as h5:
                length = int(h5["action"].shape[0])
                if length <= 0:
                    continue
                available_cameras = sorted(h5["observations/images"].keys())
                if self.camera_names is None:
                    self.camera_names = available_cameras
                missing = sorted(set(self.camera_names) - set(available_cameras))
                if missing:
                    raise KeyError(f"{path} missing camera datasets: {missing}")
                self._episode_lengths[path] = length
                for timestep in range(length):
                    self.index.append(EpisodeIndex(path, timestep, length))
        if not self.index:
            raise RuntimeError("no valid samples found in ACT dataset")

    def train_val_split(self, val_ratio: float) -> tuple[list[int], list[int]]:
        """Return sample indices split by episode, not by timestep."""

        if not 0.0 <= val_ratio < 1.0:
            raise ValueError("val_ratio must be in [0, 1)")
        rng = random.Random(self.seed)
        paths = list(self.episode_paths)
        rng.shuffle(paths)
        val_count = int(round(len(paths) * val_ratio))
        val_paths = set(paths[:val_count])
        train_indices = []
        val_indices = []
        for idx, item in enumerate(self.index):
            if item.path in val_paths:
                val_indices.append(idx)
            else:
                train_indices.append(idx)
        if not train_indices:
            raise RuntimeError("validation split consumed all samples")
        return train_indices, val_indices

    def __len__(self) -> int:
        return len(self.index)

    def __getitem__(self, idx: int) -> dict:
        import h5py
        import torch

        item = self.index[idx]
        with h5py.File(item.path, "r") as h5:
            qpos = np.asarray(h5["observations/qpos"][item.timestep], dtype=np.float32)
            if self.include_object_state:
                if "observations/object_qpos" not in h5:
                    raise KeyError(f"{item.path} has no observations/object_qpos")
                object_qpos = np.asarray(
                    h5["observations/object_qpos"][item.timestep], dtype=np.float32
                )
                qpos = np.concatenate([qpos, object_qpos[:3]], axis=0)
            images = []
            assert self.camera_names is not None
            for name in self.camera_names:
                image = np.asarray(h5[f"observations/images/{name}"][item.timestep])
                if self.image_downsample > 1:
                    image = image[:: self.image_downsample, :: self.image_downsample]
                image = np.ascontiguousarray(image.transpose(2, 0, 1))
                images.append(image)

            actions_ds = h5["action"]
            end = min(item.timestep + self.chunk_size, item.length)
            valid = end - item.timestep
            action = np.zeros((self.chunk_size, actions_ds.shape[1]), dtype=np.float32)
            action[:valid] = np.asarray(actions_ds[item.timestep:end], dtype=np.float32)
            if self.action_mode == "delta":
                action[:valid] -= qpos[:7][None, :]
            if valid < self.chunk_size:
                action[valid:] = action[valid - 1]
            is_pad = np.ones((self.chunk_size,), dtype=bool)
            is_pad[:valid] = False

        image_tensor = torch.from_numpy(np.stack(images, axis=0)).float()
        if self.normalize_images:
            image_tensor = image_tensor / 255.0
        return {
            "image": image_tensor,
            "qpos": torch.from_numpy(qpos).float(),
            "action": torch.from_numpy(action).float(),
            "is_pad": torch.from_numpy(is_pad),
        }
