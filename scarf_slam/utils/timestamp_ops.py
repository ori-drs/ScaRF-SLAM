from typing import Dict, Optional

import numpy as np

from scarf_slam.core.timestamp import MappingTimestamp
from scarf_slam.core.pose import MappingPose


def pose_dicts_equal(
    poses_a: Dict[str, MappingPose],
    poses_b: Dict[str, MappingPose],
    position_tol_m: float = 1e-8,
    rotation_tol_deg: Optional[float] = None,
    batch_size: int = 512,
) -> bool:
    missing_keys = poses_a.keys() - poses_b.keys()
    if missing_keys:
        raise KeyError(
            "poses_b does not enclose poses_a keys: "
            f"missing_keys={sorted(missing_keys)[:10]}"
            f"{'...' if len(missing_keys) > 10 else ''}"
        )

    if batch_size <= 0:
        raise ValueError(f"batch_size must be positive, got {batch_size}")

    ts_keys = list(poses_a.keys())
    for start_idx in range(0, len(ts_keys), batch_size):
        batch_keys = ts_keys[start_idx : start_idx + batch_size]

        pos_a = np.asarray([poses_a[k].pos for k in batch_keys], dtype=float)
        pos_b = np.asarray([poses_b[k].pos for k in batch_keys], dtype=float)

        position_deltas_m = np.linalg.norm(pos_a - pos_b, axis=1)
        bad_pos = np.flatnonzero(position_deltas_m > position_tol_m)
        if bad_pos.size:
            idx = int(bad_pos[0])
            print("Trajectory Changed:")
            print(pos_a[idx])
            print(pos_b[idx])
            print(f"position_delta_m={position_deltas_m[idx]}")
            return False

        if rotation_tol_deg is None:
            continue

        quat_a = np.asarray([poses_a[k].quat for k in batch_keys], dtype=float)
        quat_b = np.asarray([poses_b[k].quat for k in batch_keys], dtype=float)

        norm_a = np.linalg.norm(quat_a, axis=1, keepdims=True)
        norm_b = np.linalg.norm(quat_b, axis=1, keepdims=True)
        if np.any(norm_a == 0.0) or np.any(norm_b == 0.0):
            bad_idx = np.flatnonzero((norm_a[:, 0] == 0.0) | (norm_b[:, 0] == 0.0))
            raise ValueError(f"Invalid zero-norm quaternion at timestamp {batch_keys[int(bad_idx[0])]}")

        quat_a = quat_a / norm_a
        quat_b = quat_b / norm_b

        dots = np.sum(quat_a * quat_b, axis=1)
        dots = np.clip(np.abs(dots), -1.0, 1.0)
        rotation_deltas_deg = np.degrees(2.0 * np.arccos(dots))

        bad_rot = np.flatnonzero(rotation_deltas_deg > rotation_tol_deg)
        if bad_rot.size:
            idx = int(bad_rot[0])
            print("Trajectory Changed:")
            print(quat_a[idx])
            print(quat_b[idx])
            print(f"rotation_delta_deg={rotation_deltas_deg[idx]}")
            return False

    return True


def timestamp_key_to_seconds(timestamp: str) -> float:
    sec_str, nsec_str = timestamp.split("_", 1)
    return int(sec_str) + int(nsec_str) * 1e-9


def timestamp_key_to_timestamp(ts_key: str) -> MappingTimestamp:
    sec_str, nsec_str = ts_key.split("_", 1)
    return MappingTimestamp(int(sec_str), int(nsec_str))


def timestamp_nsec_to_key(timestamp_nsec: int) -> str:
    sec, nsec = divmod(int(timestamp_nsec), 1_000_000_000)
    return f"{sec:010d}_{nsec:09d}"


def timestamp_key_to_nsec(ts_key: str, context: Optional[str] = None) -> int:
    timestamp = timestamp_key_to_timestamp(str(ts_key))
    if timestamp.nsec < 0 or timestamp.nsec >= 1_000_000_000:
        label = context or f"Timestamp key '{ts_key}'"
        raise ValueError(f"{label} has invalid nanoseconds: {timestamp.nsec}")
    return timestamp.total_nanoseconds
