from dataclasses import dataclass
from typing import Dict, List, Mapping

import numpy as np


@dataclass
class FisheyeCamera:
    T_cam_imu: np.ndarray
    intrinsics: List[float]
    distortion_coeffs: List[float]
    camera_model: str
    distortion_model: str
    resolution: List[int]

    def __repr__(self) -> str:
        return (
            f"FisheyeCamera(\n"
            f"  T_cam_imu=\n{self.T_cam_imu},\n"
            f"  intrinsics={self.intrinsics},\n"
            f"  distortion_coeffs={self.distortion_coeffs},\n"
            f"  camera_model='{self.camera_model}',\n"
            f"  distortion_model='{self.distortion_model}',\n"
            f"  resolution={self.resolution}\n"
            f")"
        )


@dataclass
class PinholeCamera:
    intrinsics: List[float]
    resolution: List[int]
    intrinsics_mtx: np.ndarray | None = None
    camera_model: str = "pinhole"
    distortion_model: str = "none"

    def __post_init__(self) -> None:
        if len(self.intrinsics) != 4:
            raise ValueError(
                f"pinhole intrinsics must contain [fx, fy, cx, cy], got {self.intrinsics}"
            )
        if len(self.resolution) != 2:
            raise ValueError(
                f"pinhole resolution must contain [width, height], got {self.resolution}"
            )
        if self.intrinsics_mtx is None:
            fx, fy, cx, cy = self.intrinsics
            self.intrinsics_mtx = np.array(
                [
                    [fx, 0, cx],
                    [0, fy, cy],
                    [0, 0, 1],
                ],
                dtype=float,
            )

    def __repr__(self) -> str:
        return (
            f"PinholeCamera(\n"
            f"  intrinsics={self.intrinsics},\n"
            f"  intrinsics_mtx=\n{self.intrinsics_mtx},\n"
            f"  camera_model='{self.camera_model}',\n"
            f"  distortion_model='{self.distortion_model}',\n"
            f"  resolution={self.resolution}\n"
            f")"
        )


@dataclass
class RotateParam:
    cam: str = "cam0"
    yaw: int = 0
    pitch: int = 0
    roll: int = 0


def load_fisheye_cameras_from_config(config: Mapping[str, object]) -> Dict[str, FisheyeCamera]:
    import re

    fisheye_cameras: Dict[str, FisheyeCamera] = {}
    camera_key_pattern = re.compile(r"^(?:fisheye_)?(cam\d+)$")

    for config_cam_id, cam in config.items():
        match = camera_key_pattern.match(str(config_cam_id))
        if not match:
            continue
        cam_id = match.group(1)
        if not isinstance(cam, Mapping):
            raise ValueError(f"{config_cam_id} config must be a mapping")

        missing_fields = [
            field_name
            for field_name in (
                "T_cam_imu",
                "intrinsics",
                "distortion_coeffs",
                "camera_model",
                "distortion_model",
                "resolution",
            )
            if field_name not in cam
        ]
        if missing_fields:
            raise KeyError(f"{config_cam_id} missing required fields in config: {missing_fields}")
        if str(cam["camera_model"]).lower() != "pinhole":
            raise ValueError(f"{config_cam_id}.camera_model must be 'pinhole'")
        if str(cam["distortion_model"]).lower() != "equidistant":
            raise ValueError(f"{config_cam_id}.distortion_model must be 'equidistant'")
        if len(cam["intrinsics"]) != 4:
            raise ValueError(f"{config_cam_id}.intrinsics must contain [fx, fy, cx, cy]")
        if len(cam["distortion_coeffs"]) != 4:
            raise ValueError(f"{config_cam_id}.distortion_coeffs must contain [k1, k2, k3, k4]")
        if len(cam["resolution"]) != 2:
            raise ValueError(f"{config_cam_id}.resolution must contain [width, height]")

        fisheye_cameras[cam_id] = FisheyeCamera(
            T_cam_imu=np.array(cam["T_cam_imu"], dtype=float),
            intrinsics=list(cam["intrinsics"]),
            distortion_coeffs=list(cam["distortion_coeffs"]),
            camera_model=str(cam["camera_model"]).lower(),
            distortion_model=str(cam["distortion_model"]).lower(),
            resolution=list(cam["resolution"]),
        )

    return fisheye_cameras
