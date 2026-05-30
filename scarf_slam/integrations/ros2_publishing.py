import math
from typing import Optional

import cv2
import numpy as np

from scarf_slam.core.timestamp import MappingTimestamp
from scarf_slam.integrations.ros_pointcloud import point_cloud_xyzrgb

try:
    import rclpy
    from rclpy.node import Node
    from rclpy.qos import DurabilityPolicy, HistoryPolicy, QoSProfile, ReliabilityPolicy
    from geometry_msgs.msg import PoseStamped
    from nav_msgs.msg import Path as PathMsg
    from sensor_msgs.msg import PointCloud2, Image as ImageMsg
except ImportError:
    rclpy = None
    Node = None
    DurabilityPolicy = None
    HistoryPolicy = None
    QoSProfile = None
    ReliabilityPolicy = None
    PoseStamped = None
    PathMsg = None
    PointCloud2 = None
    ImageMsg = None


def timestamp_plus_seconds(timestamp: MappingTimestamp, duration_sec: float) -> MappingTimestamp:
    if not math.isfinite(duration_sec):
        raise ValueError(f"duration_sec must be finite, got {duration_sec}")

    total_nsec = int(timestamp.sec) * 1_000_000_000 + int(timestamp.nsec)
    total_nsec += int(round(duration_sec * 1_000_000_000))
    sec, nsec = divmod(total_nsec, 1_000_000_000)
    return MappingTimestamp(int(sec), int(nsec))


def concat_images_n_by_3(images: np.ndarray) -> np.ndarray:
    images = np.asarray(images)
    if images.ndim != 4 or images.shape[-1] != 3:
        raise ValueError(f"Expected processed images with shape [N, H, W, 3], got {images.shape}")
    if images.shape[0] == 0:
        raise ValueError("Cannot publish an empty processed image batch")

    images = np.ascontiguousarray(np.clip(images, 0, 255).astype(np.uint8, copy=False))
    num_images, height, width, channels = images.shape
    num_cols = 3
    num_rows = int(math.ceil(num_images / num_cols))
    padded_count = num_rows * num_cols

    if padded_count != num_images:
        padding = np.zeros(
            (padded_count - num_images, height, width, channels),
            dtype=np.uint8,
        )
        images = np.concatenate([images, padding], axis=0)

    grid = images.reshape(num_rows, num_cols, height, width, channels)
    grid = grid.transpose(0, 2, 1, 3, 4).reshape(num_rows * height, num_cols * width, channels)
    return np.ascontiguousarray(grid)


def ensure_ros2_available(
    publish_pointcloud: bool,
    publish_path: bool,
    publish_images: bool,
) -> None:
    if not publish_pointcloud and not publish_path and not publish_images:
        return
    if (
        rclpy is None
        or Node is None
        or QoSProfile is None
        or HistoryPolicy is None
        or ReliabilityPolicy is None
        or DurabilityPolicy is None
    ):
        raise ImportError("ROS2 publishing requested, but ROS2 Python dependencies are unavailable")
    if publish_pointcloud and PointCloud2 is None:
        raise ImportError("ROS2 point cloud publishing requested, but sensor_msgs/PointCloud2 is unavailable")
    if publish_path and (PoseStamped is None or PathMsg is None):
        raise ImportError("ROS2 path publishing requested, but nav_msgs/Path or geometry_msgs/PoseStamped is unavailable")
    if publish_images and ImageMsg is None:
        raise ImportError("ROS2 image publishing requested, but sensor_msgs/Image is unavailable")


def create_qos_profile():
    return QoSProfile(
        history=HistoryPolicy.KEEP_LAST,
        depth=1,
        reliability=ReliabilityPolicy.RELIABLE,
        durability=DurabilityPolicy.TRANSIENT_LOCAL,
    )


def set_ros_header_stamp(node, header, header_timestamp: Optional[MappingTimestamp]) -> None:
    if header_timestamp is None:
        header.stamp = node.get_clock().now().to_msg()
        return

    header.stamp.sec = int(header_timestamp.sec)
    header.stamp.nanosec = int(header_timestamp.nsec)


__all__ = [
    "DurabilityPolicy",
    "HistoryPolicy",
    "ImageMsg",
    "Node",
    "PathMsg",
    "PointCloud2",
    "PoseStamped",
    "QoSProfile",
    "ReliabilityPolicy",
    "concat_images_n_by_3",
    "create_qos_profile",
    "cv2",
    "ensure_ros2_available",
    "point_cloud_xyzrgb",
    "rclpy",
    "set_ros_header_stamp",
    "timestamp_plus_seconds",
]
