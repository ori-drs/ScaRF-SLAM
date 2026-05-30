import math
from typing import List, Sequence, Tuple

import numpy as np

from scarf_slam.core.pose import MappingPose


def normalized_vector(vec: Sequence[float]) -> np.ndarray:
    arr = np.asarray(vec, dtype=float)
    norm = np.linalg.norm(arr)
    return arr / norm if norm > 0 else arr


def angle_between_vectors(vec_a: Sequence[float], vec_b: Sequence[float]) -> float:
    a = normalized_vector(vec_a)
    b = normalized_vector(vec_b)
    return float(np.arccos(np.clip(np.dot(a, b), -1.0, 1.0)))


def euclidean_distance(pos_a: Sequence[float], pos_b: Sequence[float]) -> float:
    return math.sqrt(
        (pos_a[0] - pos_b[0]) ** 2
        + (pos_a[1] - pos_b[1]) ** 2
        + (pos_a[2] - pos_b[2]) ** 2
    )


def view_vector_from_quat(quat: Sequence[float]) -> np.ndarray:
    x, y, z, w = quat
    qv = np.array([x, y, z], float)
    v = np.array([0.0, 0.0, 1.0], float)
    t = 2.0 * np.cross(qv, v)
    return normalized_vector(v + w * t + np.cross(qv, t))


def motion_exceeds_threshold(
    previous_pose: MappingPose,
    current_pose: MappingPose,
    kf_distance: float,
    kf_angle_rad: float,
) -> bool:
    dist_cond = euclidean_distance(previous_pose.pos, current_pose.pos) >= kf_distance
    prev_view = view_vector_from_quat(previous_pose.quat)
    curr_view = view_vector_from_quat(current_pose.quat)
    angle_cond = angle_between_vectors(prev_view, curr_view) >= kf_angle_rad
    return dist_cond or angle_cond


def validate_submap_window_config(
    batch_size: int,
    submap_size: int,
    overlap: int,
) -> Tuple[int, int]:
    if batch_size <= 0:
        raise ValueError("num_ref_poses_per_batch must be positive.")
    if overlap < 0:
        raise ValueError("overlap_ref_views must be non-negative.")
    if overlap >= submap_size:
        raise ValueError(
            "overlap_ref_views must be smaller than num_ref_poses_per_submap. "
            f"Got overlap_ref_views={overlap}, num_ref_poses_per_submap={submap_size}."
        )
    if submap_size > batch_size:
        raise ValueError(
            "num_ref_poses_per_submap cannot exceed num_ref_poses_per_batch. "
            f"Got num_ref_poses_per_submap={submap_size}, num_ref_poses_per_batch={batch_size}."
        )

    submap_stride = submap_size - overlap
    batch_minus_submap = batch_size - submap_size
    if batch_minus_submap % submap_stride != 0:
        raise ValueError(
            "Expected num_ref_poses_per_batch = num_ref_poses_per_submap + N * "
            "(num_ref_poses_per_submap - overlap_ref_views). "
            f"Got num_ref_poses_per_batch={batch_size}, "
            f"num_ref_poses_per_submap={submap_size}, overlap_ref_views={overlap}."
        )

    return 1 + (batch_minus_submap // submap_stride), submap_stride


def get_submap_ref_pose_windows(
    num_ref_poses_in_batch: int,
    num_ref_poses_per_batch: int,
    num_submaps_per_batch: int,
    submap_ref_pose_stride: int,
    num_ref_poses_per_submap: int,
) -> List[Tuple[int, int]]:
    if num_ref_poses_in_batch != num_ref_poses_per_batch:
        raise ValueError(
            "Unexpected num_ref_poses_in_batch. "
            f"Expected {num_ref_poses_per_batch}, got {num_ref_poses_in_batch}."
        )

    windows = [
        (
            submap_idx * submap_ref_pose_stride,
            submap_idx * submap_ref_pose_stride + num_ref_poses_per_submap,
        )
        for submap_idx in range(num_submaps_per_batch)
    ]
    if windows[-1][1] != num_ref_poses_in_batch:
        raise ValueError(
            "Submap windows do not fully cover the batch. "
            f"Last window={windows[-1]}, num_ref_poses_in_batch={num_ref_poses_in_batch}."
        )
    return windows
