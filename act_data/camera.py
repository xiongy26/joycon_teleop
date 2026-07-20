"""OpenCV camera capture helpers for ACT dataset collection."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Iterable, List

import numpy as np


@dataclass(frozen=True)
class CameraConfig:
    """Configuration for one OpenCV camera stream."""

    name: str
    source: int | str
    width: int = 640
    height: int = 480
    fps: float = 30.0


def parse_camera_specs(specs: Iterable[str]) -> List[CameraConfig]:
    """Parse camera specs formatted as name=source or name=source:widthxheight."""

    cameras: List[CameraConfig] = []
    for raw in specs:
        if "=" not in raw:
            raise ValueError(
                f"camera spec must be name=source[:WIDTHxHEIGHT], got {raw!r}"
            )
        name, rest = raw.split("=", 1)
        name = name.strip()
        if not name:
            raise ValueError(f"camera name is empty in spec {raw!r}")

        source_part = rest
        width = 640
        height = 480
        if ":" in rest:
            source_part, size_part = rest.rsplit(":", 1)
            if "x" not in size_part.lower():
                raise ValueError(f"camera size must be WIDTHxHEIGHT, got {size_part!r}")
            w_text, h_text = size_part.lower().split("x", 1)
            width = int(w_text)
            height = int(h_text)

        source_text = source_part.strip()
        if not source_text:
            raise ValueError(f"camera source is empty in spec {raw!r}")
        source: int | str
        try:
            source = int(source_text)
        except ValueError:
            source = source_text
        cameras.append(CameraConfig(name=name, source=source, width=width, height=height))

    names = [camera.name for camera in cameras]
    if len(names) != len(set(names)):
        raise ValueError(f"camera names must be unique, got {names}")
    return cameras


class MultiCameraCapture:
    """Capture synchronized-ish RGB frames from multiple OpenCV cameras."""

    def __init__(self, configs: Iterable[CameraConfig]):
        self.configs = list(configs)
        if not self.configs:
            raise ValueError("at least one camera is required")
        self._captures = {}

    def __enter__(self) -> "MultiCameraCapture":
        self.open()
        return self

    def __exit__(self, _exc_type, _exc, _tb) -> None:
        self.close()

    @property
    def names(self) -> list[str]:
        return [config.name for config in self.configs]

    def open(self) -> None:
        try:
            import cv2
        except ImportError as exc:
            raise RuntimeError(
                "OpenCV is required for ACT camera collection. Install with: "
                "pip install opencv-python"
            ) from exc

        for config in self.configs:
            cap = cv2.VideoCapture(config.source)
            if not cap.isOpened():
                self.close()
                raise RuntimeError(
                    f"failed to open camera {config.name!r} from {config.source!r}"
                )
            cap.set(cv2.CAP_PROP_FRAME_WIDTH, float(config.width))
            cap.set(cv2.CAP_PROP_FRAME_HEIGHT, float(config.height))
            cap.set(cv2.CAP_PROP_FPS, float(config.fps))
            self._captures[config.name] = cap

    def read(self) -> Dict[str, np.ndarray]:
        try:
            import cv2
        except ImportError as exc:
            raise RuntimeError("OpenCV is required for camera reads") from exc

        frames: Dict[str, np.ndarray] = {}
        for name, cap in self._captures.items():
            ok, bgr = cap.read()
            if not ok or bgr is None:
                raise RuntimeError(f"failed to read frame from camera {name!r}")
            rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
            frames[name] = np.ascontiguousarray(rgb)
        return frames

    def close(self) -> None:
        for cap in self._captures.values():
            cap.release()
        self._captures.clear()
