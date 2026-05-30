from dataclasses import dataclass, field
from typing import Dict, List

import numpy as np


@dataclass
class SubmapRecord:
    anchor_key: str
    frame_keys: List[str]
    local_points: np.ndarray
    colors: np.ndarray
    frame_point_ids: Dict[str, np.ndarray]
    scale: float = 1.0
    unique_point_ids: np.ndarray = field(default_factory=lambda: np.empty((0,), dtype=np.int64))
    publish_point_mask: np.ndarray = field(default_factory=lambda: np.empty((0,), dtype=bool))
    publish_downsample_ratio: float = -1.0

    def __post_init__(self) -> None:
        self.local_points = np.ascontiguousarray(self.local_points.astype(np.float32, copy=False))
        self.colors = np.ascontiguousarray(self.colors.astype(np.uint8, copy=False))
        self.scale = float(self.scale)
        self.publish_downsample_ratio = float(self.publish_downsample_ratio)
        if self.local_points.ndim != 2 or self.local_points.shape[1] != 4:
            raise ValueError(f"local_points must have shape (N,4), got {self.local_points.shape}")
        if self.colors.ndim != 2 or self.colors.shape[1] != 3:
            raise ValueError(f"colors must have shape (N,3), got {self.colors.shape}")
        if self.local_points.shape[0] != self.colors.shape[0]:
            raise ValueError(
                f"Point/color count mismatch: points={self.local_points.shape[0]}, colors={self.colors.shape[0]}"
            )
        if self.unique_point_ids.size == 0 and self.local_points.shape[0] > 0:
            self.unique_point_ids = np.arange(self.local_points.shape[0], dtype=np.int64)
        else:
            self.unique_point_ids = np.ascontiguousarray(
                self.unique_point_ids.astype(np.int64, copy=False).reshape(-1)
            )
        if self.publish_point_mask.size == 0:
            self.publish_point_mask = np.empty((0,), dtype=bool)
        else:
            self.publish_point_mask = np.ascontiguousarray(
                self.publish_point_mask.astype(bool, copy=False).reshape(-1)
            )
            if self.publish_point_mask.shape[0] != self.local_points.shape[0]:
                raise ValueError(
                    "publish_point_mask length must match local_points: "
                    f"mask={self.publish_point_mask.shape[0]}, points={self.local_points.shape[0]}"
                )
