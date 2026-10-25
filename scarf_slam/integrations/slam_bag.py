from __future__ import annotations

import io
import os
from dataclasses import dataclass, field
from pathlib import Path
from collections import OrderedDict
from typing import Dict, Iterator, List, Mapping, Optional, Tuple

import cv2
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


class _ImageFilePayloads(Mapping):
    """The image-folder input as a mapping of the same shape as the bag's compressed images, read from disk on demand.
    Holding every payload costs ~2.8 MB per frame, i.e. tens of GB on a long sequence, for data the app reads once."""

    def __init__(self, paths: Dict[str, Path]): self._paths = dict(paths)

    def __getitem__(self, key: str) -> bytes: return _compressed_image_payload(self._paths[key])[1].tobytes()

    def __iter__(self) -> Iterator[str]: return iter(self._paths)

    def __len__(self) -> int: return len(self._paths)


@dataclass
class SlamBagData:
    bag_path: Path
    trajectory_snapshots: Dict[str, Dict[str, MappingPose]]
    odometry: Dict[str, MappingPose]
    compressed_images: Dict[str, bytes]
    image_formats: Dict[str, str] = field(default_factory=dict)
    image_camera: str = "cam0"
    _decoded_images: "OrderedDict[str, np.ndarray]" = field(default_factory=OrderedDict, init=False, repr=False)
    # A decoded frame is ~11 MB at 1920x1920, so caching every frame of a long sequence costs tens of GB. Measured over a
    # 5391-frame run: 500 reads covered 417 distinct frames and every hit was on the frame read immediately before
    # (reuse distance 1), so a handful of entries captures all of the reuse. SCARF_IMAGE_CACHE=0 restores the old behaviour.
    image_cache_size: int = int(os.environ.get("SCARF_IMAGE_CACHE", 8))

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
            # cv2 (libjpeg-turbo) decodes the same pixels as imageio/Pillow for these JPEGs, ~1.6x faster (verified on the UniPark bags)
            buf = np.frombuffer(self.compressed_images[timestamp], dtype=np.uint8)
            img = cv2.imdecode(buf, cv2.IMREAD_UNCHANGED)
            if img is None:
                img = np.asarray(imageio.imread(io.BytesIO(self.compressed_images[timestamp])))
            elif img.ndim == 3 and img.shape[2] == 3:
                img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            elif img.ndim == 3 and img.shape[2] == 4:
                img = cv2.cvtColor(img, cv2.COLOR_BGRA2RGBA)
            self._decoded_images[timestamp] = img
            if self.image_cache_size:
                while len(self._decoded_images) > self.image_cache_size: self._decoded_images.popitem(last=False)
        else:
            self._decoded_images.move_to_end(timestamp)
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

    image_paths: Dict[str, Path] = {}
    image_formats: Dict[str, str] = {}
    for timestamp_nsec, image_path in image_files:
        key = _timestamp_nsec_key(timestamp_nsec)
        image_paths[key] = image_path
        image_formats[key] = image_path.suffix.lstrip(".").lower() or "jpeg"
    compressed_images = _ImageFilePayloads(image_paths)   # read on demand: the folder is already on disk

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


# ---------------------------------------------------------------------------------------------------------------------
# Bag images + pose file (2026-10-01): the frames come straight from the recording's mcap (read on demand, one message at a
# time) and the poses from a TUM file at any rate, interpolated to the frame stamps. Removes the need to extract the bag's
# JPEGs into an image folder (a 1:1 copy, 200 GB for the CityPark Spot sessions) when the trajectory is a VIO / PGO text file.
# ---------------------------------------------------------------------------------------------------------------------
import functools


@functools.lru_cache(maxsize=8)
def index_bag_images(bag_path: str, image_topic: str) -> Tuple[Tuple[int, int, str], ...]:
    """One pass over the image topic: (header stamp ns, bag log time ns, format) per message, sorted by header stamp.
    Cached per (bag, topic) so the timestamp sync and the loader share it."""
    from rosbags.highlevel import AnyReader
    from rosbags.typesys import Stores, get_typestore
    from scarf_slam.integrations.slam_bag_time_sync import _stamp_to_nsec

    resolved = Path(bag_path).expanduser()
    if not resolved.exists():
        raise FileNotFoundError(f"Input bag was not found: {resolved}")
    typestore = get_typestore(Stores.ROS2_JAZZY)
    rows: List[Tuple[int, int, str]] = []
    with AnyReader([resolved], default_typestore=typestore) as reader:
        connections = [c for c in reader.connections if c.topic == image_topic]
        if not connections:
            available = ", ".join(sorted(c.topic for c in reader.connections)) or "<none>"
            raise FileNotFoundError(f"Missing image topic {image_topic!r} in {resolved}. Available topics: {available}")
        if connections[0].msgtype != "sensor_msgs/msg/CompressedImage":
            raise TypeError(f"Image topic {image_topic!r} has type {connections[0].msgtype}, expected sensor_msgs/msg/CompressedImage")
        for connection, log_nsec, rawdata in reader.messages(connections=connections):
            msg = reader.deserialize(rawdata, connection.msgtype)
            rows.append((_stamp_to_nsec(msg.header.stamp), int(log_nsec), str(getattr(msg, "format", "") or "jpeg")))
    if not rows:
        raise FileNotFoundError(f"No image messages were found on {image_topic} in {resolved}")
    return tuple(sorted(rows))


class _BagImagePayloads(Mapping):
    """Compressed frames of a bag as a mapping keyed like the other inputs, each message read from the mcap when asked for
    (the app reads frames once, in order; rosbags' start/stop window makes a single-message read cheap)."""

    def __init__(self, bag_path: Path, image_topic: str, log_nsec_by_key: Dict[str, int], header_nsec_by_key: Dict[str, int]):
        self._bag_path = bag_path; self._image_topic = image_topic
        self._log = dict(log_nsec_by_key); self._hdr = dict(header_nsec_by_key); self._reader = None; self._connections = None

    def _open(self):
        if self._reader is None:
            from rosbags.highlevel import AnyReader
            from rosbags.typesys import Stores, get_typestore
            self._reader = AnyReader([self._bag_path], default_typestore=get_typestore(Stores.ROS2_JAZZY)); self._reader.open()
            self._connections = [c for c in self._reader.connections if c.topic == self._image_topic]
        return self._reader

    def __getitem__(self, key: str) -> bytes:
        from scarf_slam.integrations.slam_bag_time_sync import _stamp_to_nsec
        if key not in self._log:
            raise KeyError(key)
        reader = self._open(); t = self._log[key]; want = self._hdr[key]
        for connection, _, rawdata in reader.messages(connections=self._connections, start=t - 1000, stop=t + 1000):   # rosbags' start is exclusive: a 1 us window around the log time
            msg = reader.deserialize(rawdata, connection.msgtype)
            if _stamp_to_nsec(msg.header.stamp) == want:
                return _array_or_bytes_to_bytes(msg.data)
        raise KeyError(f"Image {key} not found in {self._bag_path} at log time {t}")

    def __iter__(self) -> Iterator[str]: return iter(self._log)

    def __len__(self) -> int: return len(self._log)

    def __del__(self):
        try:
            if self._reader is not None: self._reader.close()
        except Exception: pass


def select_bag_frames(bag_path: str | Path, image_topic: str, frame_stride: int = 1) -> List[Tuple[int, int, str]]:
    """Every `frame_stride`-th image message (header ns, log ns, format), as the extraction scripts' --step did."""
    rows = index_bag_images(str(Path(bag_path).expanduser()), image_topic)
    return list(rows[:: max(1, int(frame_stride))])


def interpolate_poses_at(poses: List[Tuple], stamps_nsec: List[int]) -> Dict[int, MappingPose]:
    """TUM poses (sec, nsec, x, y, z, qx, qy, qz, qw) interpolated at the given stamps: positions linearly, rotations by
    slerp; stamps outside the pose span are left out (as spot_extract.py / spot_serial_poses.py do)."""
    from scipy.spatial.transform import Rotation, Slerp

    t = np.array([int(p[0]) + int(p[1]) * 1e-9 for p in poses], dtype=np.float64)
    keep = np.concatenate([[True], np.diff(t) > 1e-9]); t = t[keep]
    P = np.array([[p[2], p[3], p[4]] for p in poses], dtype=np.float64)[keep]
    Q = np.array([[p[5], p[6], p[7], p[8]] for p in poses], dtype=np.float64)[keep]
    slerp = Slerp(t, Rotation.from_quat(Q))
    out: Dict[int, MappingPose] = {}
    for s in stamps_nsec:
        ts = s * 1e-9
        if ts < t[0] or ts > t[-1]:
            continue
        pos = [float(np.interp(ts, t, P[:, i])) for i in range(3)]
        quat = slerp([ts]).as_quat()[0]
        out[s] = MappingPose(pos, [float(quat[0]), float(quat[1]), float(quat[2]), float(quat[3])])
    return out


def load_bag_images_and_poses(
    bag_path: str | Path,
    poses_path: str | Path,
    *,
    image_topic: str,
    frame_stride: int = 1,
    pose_timestamp_map: Optional[Mapping[int, int]] = None,
) -> SlamBagData:
    """Bag images (on demand) + TUM pose file interpolated at the selected frames' stamps (use_slam: false only)."""
    from scarf_slam.integrations.slam_bag_time_sync import _timestamp_nsec_key, read_pose_file

    resolved_bag_path = Path(bag_path).expanduser()
    frames = select_bag_frames(resolved_bag_path, image_topic, frame_stride)
    poses = read_pose_file(poses_path)
    interpolated = interpolate_poses_at(poses, [hdr for hdr, _, _ in frames])
    if not interpolated:
        raise FileNotFoundError(f"No frame of {resolved_bag_path} falls inside the pose span of {poses_path}")
    log_by_key: Dict[str, int] = {}; hdr_by_key: Dict[str, int] = {}; image_formats: Dict[str, str] = {}; pose_dict: Dict[str, MappingPose] = {}
    last_nsec: Optional[int] = None
    for hdr, log_nsec, fmt in frames:
        if hdr not in interpolated:
            continue
        target_nsec = hdr if pose_timestamp_map is None else pose_timestamp_map.get(hdr, hdr)
        key = _timestamp_nsec_key(target_nsec)
        log_by_key[key] = log_nsec; hdr_by_key[key] = hdr; image_formats[key] = fmt.lower() or "jpeg"
        pose_dict[key] = interpolated[hdr]; last_nsec = target_nsec
    print(f"[slam_bag] bag images + pose file: {len(frames)} frames (stride {frame_stride}) of {resolved_bag_path.name}, "
          f"{len(pose_dict)} inside the pose span of {Path(poses_path).name}")
    return SlamBagData(
        bag_path=resolved_bag_path,
        trajectory_snapshots={_timestamp_nsec_key(last_nsec): pose_dict},
        odometry=pose_dict.copy(),
        compressed_images=_BagImagePayloads(resolved_bag_path, image_topic, log_by_key, hdr_by_key),
        image_formats=image_formats,
    )
