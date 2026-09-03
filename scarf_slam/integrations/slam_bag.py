from __future__ import annotations

import io
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Mapping, Optional, Tuple

import imageio.v2 as imageio
import numpy as np

from scarf_slam.core.pose import MappingPose
from scarf_slam.integrations.slam_bag_time_sync import (
    _collect_image_files,
    _compressed_image_payload,
    _path_header_target_nsec,
    _stamp_to_nsec,
    _timestamp_nsec_key,
    read_pose_file,
)


DEFAULT_TRAJECTORY_TOPIC = "/scarf_slam/input/trajectory"
DEFAULT_FINAL_TRAJECTORY_TOPIC = "/scarf_slam/input/trajectory_final"
DEFAULT_ODOMETRY_TOPIC = "/scarf_slam/input/odometry"
DEFAULT_IMAGE_TOPIC = "/scarf_slam/input/cam0/image_raw/compressed"


def _stamp_key(stamp) -> str:
    nsec = getattr(stamp, "nanosec", getattr(stamp, "nsec", None))
    if nsec is None:
        raise AttributeError(f"ROS timestamp has no nanosecond field: {stamp!r}")
    return f"{int(stamp.sec):010d}_{int(nsec):09d}"


def _pose_to_mapping_pose(pose) -> MappingPose:
    return MappingPose(
        [
            float(pose.position.x),
            float(pose.position.y),
            float(pose.position.z),
        ],
        [
            float(pose.orientation.x),
            float(pose.orientation.y),
            float(pose.orientation.z),
            float(pose.orientation.w),
        ],
    )


def _path_msg_to_pose_dict(path_msg) -> Dict[str, MappingPose]:
    return {
        _stamp_key(pose_stamped.header.stamp): _pose_to_mapping_pose(pose_stamped.pose)
        for pose_stamped in path_msg.poses
    }


def _path_msg_snapshot_key(path_msg) -> str:
    if path_msg.poses:
        return _stamp_key(path_msg.poses[-1].header.stamp)
    return _stamp_key(path_msg.header.stamp)


def _synced_path_snapshot(
    path_msg,
    timestamp_map: Mapping[int, int],
) -> Optional[Tuple[int, str, Dict[str, MappingPose]]]:
    """Return a loaded Path snapshot with pose timestamps mapped to image time."""
    pose_timestamps_nsec = [
        _stamp_to_nsec(pose_stamped.header.stamp) for pose_stamped in path_msg.poses
    ]
    pose_dict: Dict[str, MappingPose] = {}
    for pose_stamped, source_nsec in zip(path_msg.poses, pose_timestamps_nsec):
        target_nsec = timestamp_map.get(source_nsec)
        if target_nsec is None:
            continue
        pose_dict[_timestamp_nsec_key(target_nsec)] = _pose_to_mapping_pose(pose_stamped.pose)
    if not pose_dict:
        return None
    header_nsec = _path_header_target_nsec(
        _stamp_to_nsec(path_msg.header.stamp),
        pose_timestamps_nsec,
        timestamp_map,
    )
    return header_nsec, _timestamp_nsec_key(header_nsec), pose_dict


def _array_or_bytes_to_bytes(data) -> bytes:
    if isinstance(data, bytes):
        return data
    if isinstance(data, bytearray):
        return bytes(data)
    if isinstance(data, memoryview):
        return data.tobytes()
    return np.asarray(data, dtype=np.uint8).tobytes()


@dataclass
class SlamBagData:
    bag_path: Path
    trajectory_snapshots: Dict[str, Dict[str, MappingPose]]
    odometry: Dict[str, MappingPose]
    compressed_images: Dict[str, bytes]
    image_formats: Dict[str, str] = field(default_factory=dict)
    image_camera: str = "cam0"
    _decoded_images: Dict[str, np.ndarray] = field(default_factory=dict, init=False, repr=False)

    @property
    def trajectory_snapshot_timestamps(self) -> List[str]:
        return sorted(self.trajectory_snapshots.keys())

    @property
    def image_timestamps(self) -> List[str]:
        return sorted(self.compressed_images.keys())

    def get_trajectory_snapshot(self, timestamp: str) -> Dict[str, MappingPose]:
        if timestamp not in self.trajectory_snapshots:
            raise FileNotFoundError(
                f"Trajectory snapshot {timestamp} was not found in {self.bag_path}"
            )
        return self.trajectory_snapshots[timestamp].copy()

    def decode_image(self, timestamp: str, cam_name: str = "cam0") -> np.ndarray:
        if cam_name != self.image_camera:
            raise FileNotFoundError(
                f"Bag image topic only provides {self.image_camera}; requested {cam_name}. "
                "Support for loading two camera image topics is not implemented yet."
            )
        if timestamp not in self.compressed_images:
            raise FileNotFoundError(
                f"Compressed image for {cam_name} at {timestamp} was not found in {self.bag_path}"
            )
        if timestamp not in self._decoded_images:
            self._decoded_images[timestamp] = np.asarray(
                imageio.imread(io.BytesIO(self.compressed_images[timestamp]))
            )
        return self._decoded_images[timestamp]


def load_slam_bag(
    bag_path: str | Path,
    *,
    use_slam: bool = True,
    trajectory_topic: Optional[str] = DEFAULT_TRAJECTORY_TOPIC,
    final_trajectory_topic: Optional[str] = DEFAULT_FINAL_TRAJECTORY_TOPIC,
    odometry_topic: Optional[str] = DEFAULT_ODOMETRY_TOPIC,
    image_topic: Optional[str] = DEFAULT_IMAGE_TOPIC,
    pose_timestamp_map: Optional[Mapping[int, int]] = None,
) -> SlamBagData:
    try:
        from rosbags.highlevel import AnyReader
        from rosbags.typesys import Stores, get_typestore
    except ImportError as exc:
        raise ImportError(
            "Reading a SLAM bag requires the 'rosbags' Python package. "
            "Install rosbags to read ROS2 MCAP bags without a ROS2 installation."
        ) from exc

    if image_topic is None:
        raise ValueError("slam_image_topic must not be null")
    if final_trajectory_topic is None:
        raise ValueError("slam_final_trajectory_topic must not be null")
    if use_slam and trajectory_topic is None:
        raise ValueError("slam_trajectory_topic must not be null when use_slam is true")
    if use_slam and odometry_topic is None:
        raise ValueError("slam_odometry_topic must not be null when use_slam is true")

    resolved_bag_path = Path(bag_path).expanduser()
    if not resolved_bag_path.exists():
        raise FileNotFoundError(f"SLAM bag was not found: {resolved_bag_path}")

    required_topics = {
        image_topic: "sensor_msgs/msg/CompressedImage",
        final_trajectory_topic: "nav_msgs/msg/Path",
    }
    if use_slam:
        required_topics[trajectory_topic] = "nav_msgs/msg/Path"
        required_topics[odometry_topic] = "nav_msgs/msg/Odometry"
    selected_topics = set(required_topics)

    trajectory_snapshots: Dict[str, Dict[str, MappingPose]] = {}
    final_trajectory_snapshots: List[Tuple[int, str, Dict[str, MappingPose]]] = []
    odometry: Dict[str, MappingPose] = {}
    compressed_images: Dict[str, bytes] = {}
    image_formats: Dict[str, str] = {}

    typestore = get_typestore(Stores.ROS2_JAZZY)
    with AnyReader([resolved_bag_path], default_typestore=typestore) as reader:
        topic_types = {connection.topic: connection.msgtype for connection in reader.connections}
        missing_topics = [topic for topic in required_topics if topic not in topic_types]
        if missing_topics:
            available = ", ".join(sorted(topic_types)) or "<none>"
            raise FileNotFoundError(
                f"Missing required SLAM bag topics: {missing_topics}. "
                f"Available topics: {available}"
            )

        for topic, expected_type in required_topics.items():
            if topic in topic_types and topic_types[topic] != expected_type:
                raise TypeError(
                    f"Topic {topic} has type {topic_types[topic]}, expected {expected_type}"
                )

        connections = [
            connection
            for connection in reader.connections
            if connection.topic in selected_topics
        ]

        for connection, timestamp, rawdata in reader.messages(connections=connections):
            msg = reader.deserialize(rawdata, connection.msgtype)
            if connection.topic == trajectory_topic:
                if pose_timestamp_map is None:
                    trajectory_snapshots[_path_msg_snapshot_key(msg)] = _path_msg_to_pose_dict(msg)
                else:
                    snapshot = _synced_path_snapshot(msg, pose_timestamp_map)
                    if snapshot is not None:
                        trajectory_snapshots[snapshot[1]] = snapshot[2]
            elif connection.topic == final_trajectory_topic:
                if pose_timestamp_map is None:
                    final_trajectory_snapshots.append(
                        (int(timestamp), _path_msg_snapshot_key(msg), _path_msg_to_pose_dict(msg))
                    )
                else:
                    snapshot = _synced_path_snapshot(msg, pose_timestamp_map)
                    if snapshot is not None:
                        final_trajectory_snapshots.append(snapshot)
            elif connection.topic == odometry_topic:
                if pose_timestamp_map is None:
                    odometry[_stamp_key(msg.header.stamp)] = _pose_to_mapping_pose(msg.pose.pose)
                else:
                    target_nsec = pose_timestamp_map.get(_stamp_to_nsec(msg.header.stamp))
                    if target_nsec is not None:
                        odometry[_timestamp_nsec_key(target_nsec)] = _pose_to_mapping_pose(msg.pose.pose)
            elif connection.topic == image_topic:
                key = _stamp_key(msg.header.stamp)
                compressed_images[key] = _array_or_bytes_to_bytes(msg.data)
                image_formats[key] = str(getattr(msg, "format", ""))

    if use_slam:
        for _, timestamp, pose_dict in final_trajectory_snapshots:
            # Apply final snapshots after regular snapshots so matching timestamps use
            # the configured final trajectory topic.
            trajectory_snapshots[timestamp] = pose_dict
    else:
        if not final_trajectory_snapshots:
            raise FileNotFoundError(
                f"No final trajectory messages were found on {final_trajectory_topic} "
                f"in {resolved_bag_path}"
            )
        _, latest_timestamp, latest_pose_dict = max(
            final_trajectory_snapshots,
            key=lambda item: item[0],
        )
        trajectory_snapshots = {latest_timestamp: latest_pose_dict}
        odometry = latest_pose_dict.copy()

    if not trajectory_snapshots:
        raise FileNotFoundError(f"No trajectory snapshots were found in {resolved_bag_path}")
    if use_slam and not odometry:
        raise FileNotFoundError(f"No odometry messages were found in {resolved_bag_path}")
    if not compressed_images:
        raise FileNotFoundError(f"No compressed images were found in {resolved_bag_path}")

    return SlamBagData(
        bag_path=resolved_bag_path,
        trajectory_snapshots=trajectory_snapshots,
        odometry=odometry,
        compressed_images=compressed_images,
        image_formats=image_formats,
    )


def load_image_folder_and_poses(
    image_folder: str | Path,
    poses_path: str | Path,
    *,
    pose_timestamp_map: Optional[Mapping[int, int]] = None,
) -> SlamBagData:
    """Load image-folder and pose-file input."""
    resolved_image_folder = Path(image_folder).expanduser()
    image_files = _collect_image_files(resolved_image_folder)
    poses = read_pose_file(poses_path)

    compressed_images: Dict[str, bytes] = {}
    image_formats: Dict[str, str] = {}
    for timestamp_nsec, image_path in image_files:
        image_format, payload = _compressed_image_payload(image_path)
        key = _timestamp_nsec_key(timestamp_nsec)
        compressed_images[key] = payload.tobytes()
        image_formats[key] = image_format

    pose_dict: Dict[str, MappingPose] = {}
    last_pose_nsec: Optional[int] = None
    for sec, nsec, x, y, z, qx, qy, qz, qw in poses:
        source_nsec = int(sec) * 1_000_000_000 + int(nsec)
        if pose_timestamp_map is None:
            target_nsec = source_nsec
        else:
            target_nsec = pose_timestamp_map.get(source_nsec)
            if target_nsec is None:
                continue
        pose_dict[_timestamp_nsec_key(target_nsec)] = MappingPose(
            [float(x), float(y), float(z)],
            [float(qx), float(qy), float(qz), float(qw)],
        )
        last_pose_nsec = target_nsec
    if not pose_dict or last_pose_nsec is None:
        raise FileNotFoundError(
            f"No poses in {poses_path} matched an image timestamp in {resolved_image_folder}"
        )

    snapshot_key = _timestamp_nsec_key(last_pose_nsec)
    return SlamBagData(
        bag_path=resolved_image_folder,
        trajectory_snapshots={snapshot_key: pose_dict},
        odometry=pose_dict.copy(),
        compressed_images=compressed_images,
        image_formats=image_formats,
    )
