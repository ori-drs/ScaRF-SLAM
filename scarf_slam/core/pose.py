from dataclasses import dataclass, field
from typing import List

import math
import numpy as np


@dataclass
class MappingPose:
    pos: List[float] = field(default_factory=lambda: [0.0, 0.0, 0.0])
    quat: List[float] = field(default_factory=lambda: [0.0, 0.0, 0.0, 1.0])

    def __repr__(self) -> str:
        return f"MappingPose(pos={self.pos}, quat={self.quat})"


class MappingTransforms:
    @staticmethod
    def matrix_to_pose(pose_matrix: np.ndarray) -> MappingPose:
        quat = MappingTransforms.quaternion_from_matrix(pose_matrix)
        xyz = [
            float(pose_matrix[0, 3]),
            float(pose_matrix[1, 3]),
            float(pose_matrix[2, 3]),
        ]
        return MappingPose(xyz, quat.tolist())

    @staticmethod
    def pose_to_matrix(pose: MappingPose) -> np.ndarray:
        matrix = MappingTransforms.quaternion_matrix(pose.quat)
        matrix[0, 3] = pose.pos[0]
        matrix[1, 3] = pose.pos[1]
        matrix[2, 3] = pose.pos[2]
        return matrix

    @staticmethod
    def quaternion_from_matrix(matrix: np.ndarray) -> np.ndarray:
        q = np.empty((4,), dtype=np.float64)
        m = np.array(matrix, dtype=np.float64)[:4, :4]
        trace = np.trace(m)
        if trace > m[3, 3]:
            q[3] = trace
            q[2] = m[1, 0] - m[0, 1]
            q[1] = m[0, 2] - m[2, 0]
            q[0] = m[2, 1] - m[1, 2]
        else:
            i, j, k = 0, 1, 2
            if m[1, 1] > m[0, 0]:
                i, j, k = 1, 2, 0
            if m[2, 2] > m[i, i]:
                i, j, k = 2, 0, 1
            trace = m[i, i] - (m[j, j] + m[k, k]) + m[3, 3]
            q[i] = trace
            q[j] = m[i, j] + m[j, i]
            q[k] = m[k, i] + m[i, k]
            q[3] = m[k, j] - m[j, k]
        q *= 0.5 / math.sqrt(trace * m[3, 3])
        return q

    @staticmethod
    def quaternion_matrix(quaternion: List[float]) -> np.ndarray:
        eps = 2.220446049250313e-16
        q = np.array(quaternion[:4], dtype=np.float64, copy=True)
        norm_sq = np.dot(q, q)
        if norm_sq < eps:
            return np.identity(4)
        q *= math.sqrt(2.0 / norm_sq)
        q = np.outer(q, q)
        return np.array(
            (
                (1.0 - q[1, 1] - q[2, 2], q[0, 1] - q[2, 3], q[0, 2] + q[1, 3], 0.0),
                (q[0, 1] + q[2, 3], 1.0 - q[0, 0] - q[2, 2], q[1, 2] - q[0, 3], 0.0),
                (q[0, 2] - q[1, 3], q[1, 2] + q[0, 3], 1.0 - q[0, 0] - q[1, 1], 0.0),
                (0.0, 0.0, 0.0, 1.0),
            ),
            dtype=np.float64,
        )
