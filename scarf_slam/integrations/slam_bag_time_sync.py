from __future__ import annotations

import csv
import io
import re
from bisect import bisect_left
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Set, Tuple

import numpy as np


PATH_MSG_TYPE = "nav_msgs/msg/Path"
ODOMETRY_MSG_TYPE = "nav_msgs/msg/Odometry"
COMPRESSED_IMAGE_MSG_TYPE = "sensor_msgs/msg/CompressedImage"
IMAGE_FILENAME_RE = re.compile(r"^image_(\d+)_(\d+)(?:\.[^.]+)?$")

PoseRow = Tuple[int, int, float, float, float, float, float, float, float]


@dataclass(frozen=True)
class BagTimestampSyncResult:
    bag_path: Path
    sync_applied: bool
    unique_pose_timestamps: int
    exact_matches: int
    rewritten_timestamps: int
    path_header_updates: int
    max_delta_sec: float
    timestamp_map: Dict[int, int] = field(default_factory=dict)
    current_session_start_image_timestamp_nsec: Optional[int] = None
    skipped_pose_timestamps: int = 0
    current_session_skipped_pose_timestamps: int = 0
    previous_session_skipped_pose_timestamps: int = 0


def _stamp_to_nsec(stamp) -> int:
    nsec = getattr(stamp, "nanosec", getattr(stamp, "nsec", None))
    if nsec is None:
        raise AttributeError(f"ROS timestamp has no nanosecond field: {stamp!r}")
    return int(stamp.sec) * 1_000_000_000 + int(nsec)


def _timestamp_nsec_key(timestamp_nsec: int) -> str:
    sec, nsec = divmod(int(timestamp_nsec), 1_000_000_000)
    return f"{sec:010d}_{nsec:09d}"


def _nearest_timestamp(
    timestamp_nsec: int,
    sorted_targets_nsec: Sequence[int],
    tolerance_nsec: int,
) -> Optional[Tuple[int, int]]:
    idx = bisect_left(sorted_targets_nsec, timestamp_nsec)
    candidates: List[int] = []
    if idx < len(sorted_targets_nsec):
        candidates.append(sorted_targets_nsec[idx])
    if idx > 0:
        candidates.append(sorted_targets_nsec[idx - 1])
    if not candidates:
        raise ValueError("Cannot synchronize poses because the image topic has no timestamps.")

    best = min(candidates, key=lambda value: (abs(value - timestamp_nsec), value))
    delta = abs(best - timestamp_nsec)
    if delta > tolerance_nsec:
        return None
    return best, delta


def _selected_pose_topics(
    *,
    trajectory_topic: Optional[str],
    final_trajectory_topic: Optional[str],
    odometry_topic: Optional[str],
) -> Tuple[Set[str], Set[str]]:
    path_topics = {
        topic
        for topic in (trajectory_topic, final_trajectory_topic)
        if topic is not None
    }
    odometry_topics = {odometry_topic} if odometry_topic is not None else set()
    return path_topics, odometry_topics


def _collect_pose_timestamps_nsec(msg, msgtype: str) -> List[int]:
    if msgtype == PATH_MSG_TYPE:
        return [_stamp_to_nsec(pose_stamped.header.stamp) for pose_stamped in msg.poses]
    if msgtype == ODOMETRY_MSG_TYPE:
        return [_stamp_to_nsec(msg.header.stamp)]
    return []


def _path_header_target_nsec(
    header_nsec: int,
    pose_timestamps_nsec: Sequence[int],
    timestamp_map: Dict[int, int],
) -> Optional[int]:
    for pose_nsec in reversed(pose_timestamps_nsec):
        target_nsec = timestamp_map.get(pose_nsec)
        if target_nsec is not None:
            return target_nsec
    return timestamp_map.get(header_nsec)


def _match_pose_timestamps_to_images(
    pose_timestamps_nsec: Iterable[int],
    image_timestamps_nsec: Iterable[int],
    tolerance_nsec: int,
) -> Tuple[Dict[int, int], List[int]]:
    sorted_images_nsec = sorted(image_timestamps_nsec)
    timestamp_map: Dict[int, int] = {}
    deltas_nsec: List[int] = []

    for pose_nsec in sorted(pose_timestamps_nsec):
        nearest = _nearest_timestamp(
            pose_nsec,
            sorted_images_nsec,
            tolerance_nsec,
        )
        if nearest is None:
            continue
        matched_nsec, delta_nsec = nearest
        timestamp_map[pose_nsec] = matched_nsec
        deltas_nsec.append(delta_nsec)

    return timestamp_map, deltas_nsec


def _ensure_unique_image_matches(timestamp_map: Dict[int, int]) -> None:
    inverse_map: Dict[int, int] = {}
    for pose_nsec, image_nsec in timestamp_map.items():
        existing_pose_nsec = inverse_map.setdefault(image_nsec, pose_nsec)
        if existing_pose_nsec != pose_nsec:
            raise ValueError(
                "Cannot synchronize bag timestamps because multiple distinct pose "
                "timestamps match the same image timestamp: "
                f"pose_nsec={existing_pose_nsec}, other_pose_nsec={pose_nsec}, image_nsec={image_nsec}"
            )


def _count_path_header_updates(
    path_message_timestamps_nsec: Iterable[Tuple[int, List[int]]],
    timestamp_map: Dict[int, int],
) -> int:
    return sum(
        1
        for header_nsec, message_pose_timestamps_nsec in path_message_timestamps_nsec
        if (
            target_nsec := _path_header_target_nsec(
                header_nsec,
                message_pose_timestamps_nsec,
                timestamp_map,
            )
        ) is not None
        and target_nsec != header_nsec
    )


def read_bag_start_image_timestamp_nsec(
    input_bag: str | Path,
    *,
    image_topic: Optional[str],
) -> int:
    if image_topic is None:
        raise ValueError("slam_image_topic must not be null when checking previous-session ordering")

    try:
        from rosbags.highlevel import AnyReader
        from rosbags.typesys import Stores, get_typestore
    except ImportError as exc:
        raise ImportError(
            "Reading image timestamps requires the 'rosbags' Python package."
        ) from exc

    resolved_input_bag = Path(input_bag).expanduser()
    if not resolved_input_bag.exists():
        raise FileNotFoundError(f"Input bag was not found: {resolved_input_bag}")

    typestore = get_typestore(Stores.ROS2_JAZZY)
    with AnyReader([resolved_input_bag], default_typestore=typestore) as reader:
        image_connections = [
            connection
            for connection in reader.connections
            if connection.topic == image_topic
        ]
        if not image_connections:
            available = ", ".join(sorted(connection.topic for connection in reader.connections)) or "<none>"
            raise FileNotFoundError(
                f"Missing image topic {image_topic!r} while reading current-session start time. "
                f"Available topics: {available}"
            )
        for connection in image_connections:
            if connection.msgtype != COMPRESSED_IMAGE_MSG_TYPE:
                raise TypeError(
                    f"Configured image topic {image_topic!r} has type "
                    f"{connection.msgtype!r}, expected {COMPRESSED_IMAGE_MSG_TYPE!r}"
                )

        first_image_timestamp_nsec: Optional[int] = None
        for connection, _, rawdata in reader.messages(connections=image_connections):
            msg = reader.deserialize(rawdata, connection.msgtype)
            image_timestamp_nsec = _stamp_to_nsec(msg.header.stamp)
            if (
                first_image_timestamp_nsec is None
                or image_timestamp_nsec < first_image_timestamp_nsec
            ):
                first_image_timestamp_nsec = image_timestamp_nsec
        if first_image_timestamp_nsec is not None:
            return first_image_timestamp_nsec

    raise FileNotFoundError(
        f"No image messages were found on {image_topic!r} in {resolved_input_bag}"
    )


def read_folder_start_image_timestamp_nsec(image_folder: str | Path) -> int:
    return _collect_image_files(image_folder)[0][0]


def _parse_decimal_timestamp(timestamp_text: str) -> Tuple[int, int]:
    whole, dot, fractional = timestamp_text.strip().partition(".")
    if not whole or whole.startswith("-"):
        raise ValueError(f"Invalid non-negative timestamp: {timestamp_text!r}")
    sec = int(whole)
    nsec = int(((fractional if dot else "") + "000000000")[:9])
    return sec, nsec


def _parse_pose_csv_row(fields: Sequence[str], row_number: int) -> Optional[PoseRow]:
    values = [field.strip() for field in fields]
    if not values or not values[0] or values[0].startswith("#"):
        return None
    if values[0].lower() in {"counter", "timestamp"}:
        return None

    try:
        if len(values) >= 10:
            sec = int(values[1])
            nsec = int(values[2])
            pose_values = [float(value) for value in values[3:10]]
            return (sec, nsec, *pose_values)

        if len(values) == 8:
            sec, nsec = _parse_decimal_timestamp(values[0])
            pose_values = [float(value) for value in values[1:8]]
            return (sec, nsec, *pose_values)
    except ValueError as exc:
        raise ValueError(f"Invalid pose row {row_number}: {fields}") from exc

    raise ValueError(
        f"Invalid pose row {row_number}: expected either "
        "counter,sec,nsec,x,y,z,qx,qy,qz,qw or TUM timestamp,x,y,z,qx,qy,qz,qw"
    )


def read_pose_file(poses_path: str | Path) -> List[PoseRow]:
    resolved_poses_path = Path(poses_path).expanduser()
    if not resolved_poses_path.exists():
        raise FileNotFoundError(f"Pose file was not found: {resolved_poses_path}")

    poses: List[PoseRow] = []
    with resolved_poses_path.open(newline="") as poses_file:
        reader = csv.reader(poses_file)
        for row_number, row in enumerate(reader, start=1):
            if len(row) == 1:
                stripped = row[0].strip()
                if stripped and not stripped.startswith("#"):
                    row = stripped.split()
            pose = _parse_pose_csv_row(row, row_number)
            if pose is not None:
                poses.append(pose)

    if not poses:
        raise ValueError(f"No poses were found in {resolved_poses_path}")

    return sorted(poses, key=lambda pose: (pose[0], pose[1]))


def _collect_image_files(image_folder: str | Path) -> List[Tuple[int, Path]]:
    resolved_image_folder = Path(image_folder).expanduser()
    if not resolved_image_folder.is_dir():
        raise FileNotFoundError(f"Image folder was not found: {resolved_image_folder}")

    image_files: List[Tuple[int, Path]] = []
    seen_timestamps: Dict[int, Path] = {}
    for image_path in sorted(path for path in resolved_image_folder.iterdir() if path.is_file()):
        match = IMAGE_FILENAME_RE.match(image_path.name)
        if match is None:
            continue
        sec = int(match.group(1))
        nsec = int(match.group(2))
        if nsec < 0 or nsec >= 1_000_000_000:
            raise ValueError(f"Image filename has invalid nanoseconds: {image_path}")
        timestamp_nsec = sec * 1_000_000_000 + nsec
        previous_path = seen_timestamps.setdefault(timestamp_nsec, image_path)
        if previous_path != image_path:
            raise ValueError(
                "Multiple images have the same timestamp: "
                f"{previous_path} and {image_path}"
            )
        image_files.append((timestamp_nsec, image_path))

    if not image_files:
        raise ValueError(
            f"No images named image_<sec>_<nsec>[.<ext>] were found in {resolved_image_folder}"
        )

    return sorted(image_files, key=lambda item: item[0])


def _compressed_image_payload(image_path: Path) -> Tuple[str, np.ndarray]:
    suffix = image_path.suffix.lower()
    if suffix in {".jpg", ".jpeg"}:
        return "jpeg", np.frombuffer(image_path.read_bytes(), dtype=np.uint8)
    if suffix == ".png":
        return "png", np.frombuffer(image_path.read_bytes(), dtype=np.uint8)

    try:
        import imageio.v2 as imageio
    except ImportError as exc:
        raise ImportError(
            "Encoding non-JPEG/PNG image files requires the 'imageio' Python package."
        ) from exc

    image = imageio.imread(image_path)
    buffer = io.BytesIO()
    imageio.imwrite(buffer, image, format="png")
    return "png", np.frombuffer(buffer.getvalue(), dtype=np.uint8)


def _build_sync_result(
    source_path: Path,
    *,
    image_timestamps_nsec: Set[int],
    pose_timestamps_nsec: Set[int],
    path_message_timestamps_nsec: List[Tuple[int, List[int]]],
    tolerance_sec: float,
    previous_session_image_timestamps_nsec_set: Set[int],
) -> BagTimestampSyncResult:
    tolerance_nsec = int(round(float(tolerance_sec) * 1_000_000_000))
    all_image_timestamps_nsec = image_timestamps_nsec | previous_session_image_timestamps_nsec_set
    timestamp_map, deltas_nsec = _match_pose_timestamps_to_images(
        pose_timestamps_nsec,
        all_image_timestamps_nsec,
        tolerance_nsec,
    )
    _ensure_unique_image_matches(timestamp_map)

    matched_previous_image_timestamps_nsec = set(timestamp_map.values()) & previous_session_image_timestamps_nsec_set
    missing_previous_image_timestamps_nsec = sorted(
        previous_session_image_timestamps_nsec_set - matched_previous_image_timestamps_nsec
    )
    if missing_previous_image_timestamps_nsec:
        preview = ", ".join(
            _timestamp_nsec_key(timestamp_nsec)
            for timestamp_nsec in missing_previous_image_timestamps_nsec[:10]
        )
        suffix = "..." if len(missing_previous_image_timestamps_nsec) > 10 else ""
        raise ValueError(
            "Previous-session image timestamp(s) have no matching pose within "
            f"{tolerance_sec:.9f}s: count={len(missing_previous_image_timestamps_nsec)}, "
            f"first={preview}{suffix}"
        )

    exact_matches = sum(1 for pose_nsec, image_nsec in timestamp_map.items() if pose_nsec == image_nsec)
    rewritten_timestamps = len(timestamp_map) - exact_matches
    skipped_pose_timestamps_nsec = pose_timestamps_nsec - set(timestamp_map)
    skipped_pose_timestamps = len(skipped_pose_timestamps_nsec)
    current_session_start_image_timestamp_nsec = min(image_timestamps_nsec)
    current_session_skipped_pose_timestamps_nsec = sorted(
        pose_nsec
        for pose_nsec in skipped_pose_timestamps_nsec
        if (
            not previous_session_image_timestamps_nsec_set
            or pose_nsec >= current_session_start_image_timestamp_nsec
        )
    )
    if current_session_skipped_pose_timestamps_nsec:
        preview = ", ".join(
            _timestamp_nsec_key(timestamp_nsec)
            for timestamp_nsec in current_session_skipped_pose_timestamps_nsec[:10]
        )
        suffix = "..." if len(current_session_skipped_pose_timestamps_nsec) > 10 else ""
        raise ValueError(
            "Current-session pose timestamp(s) have no matching image within "
            f"{tolerance_sec:.9f}s: count={len(current_session_skipped_pose_timestamps_nsec)}, "
            f"first={preview}{suffix}"
        )
    previous_session_skipped_pose_timestamps = (
        sum(
            1
            for pose_nsec in skipped_pose_timestamps_nsec
            if pose_nsec < current_session_start_image_timestamp_nsec
        )
        if previous_session_image_timestamps_nsec_set
        else 0
    )
    current_session_skipped_pose_timestamps = (
        skipped_pose_timestamps - previous_session_skipped_pose_timestamps
    )
    path_header_updates = _count_path_header_updates(
        path_message_timestamps_nsec,
        timestamp_map,
    )
    max_delta_sec = (max(deltas_nsec) * 1e-9) if deltas_nsec else 0.0
    sync_applied = not (
        rewritten_timestamps == 0
        and path_header_updates == 0
        and current_session_skipped_pose_timestamps == 0
    )
    return BagTimestampSyncResult(
        bag_path=source_path,
        sync_applied=sync_applied,
        unique_pose_timestamps=len(timestamp_map),
        exact_matches=exact_matches,
        rewritten_timestamps=rewritten_timestamps,
        path_header_updates=path_header_updates,
        max_delta_sec=max_delta_sec,
        timestamp_map=dict(timestamp_map),
        current_session_start_image_timestamp_nsec=current_session_start_image_timestamp_nsec,
        skipped_pose_timestamps=skipped_pose_timestamps,
        current_session_skipped_pose_timestamps=current_session_skipped_pose_timestamps,
        previous_session_skipped_pose_timestamps=previous_session_skipped_pose_timestamps,
    )


def compute_bag_image_pose_timestamp_sync(
    input_bag: str | Path,
    *,
    image_topic: str,
    trajectory_topic: Optional[str],
    final_trajectory_topic: Optional[str],
    odometry_topic: Optional[str],
    tolerance_sec: float,
    previous_session_image_timestamps_nsec: Iterable[int] = (),
) -> BagTimestampSyncResult:
    if image_topic is None:
        raise ValueError("slam_image_topic must not be null")
    if tolerance_sec < 0:
        raise ValueError(f"image_pose_timestamp_tolerance_sec must be non-negative, got {tolerance_sec}")

    try:
        from rosbags.highlevel import AnyReader
        from rosbags.typesys import Stores, get_typestore
    except ImportError as exc:
        raise ImportError(
            "Synchronizing bag timestamps requires the 'rosbags' Python package."
        ) from exc

    resolved_input_bag = Path(input_bag).expanduser()
    if not resolved_input_bag.exists():
        raise FileNotFoundError(f"Input bag was not found: {resolved_input_bag}")

    previous_session_image_timestamps_nsec_set = {
        int(timestamp_nsec)
        for timestamp_nsec in previous_session_image_timestamps_nsec
    }
    path_topics, odometry_topics = _selected_pose_topics(
        trajectory_topic=trajectory_topic,
        final_trajectory_topic=final_trajectory_topic,
        odometry_topic=odometry_topic,
    )
    pose_topics = path_topics | odometry_topics
    selected_topics = {image_topic} | pose_topics
    typestore = get_typestore(Stores.ROS2_JAZZY)

    image_timestamps_nsec: Set[int] = set()
    pose_timestamps_nsec: Set[int] = set()
    path_message_timestamps_nsec: List[Tuple[int, List[int]]] = []

    with AnyReader([resolved_input_bag], default_typestore=typestore) as reader:
        topic_types = {connection.topic: connection.msgtype for connection in reader.connections}
        if image_topic not in topic_types:
            available = ", ".join(sorted(topic_types)) or "<none>"
            raise FileNotFoundError(
                f"Missing image topic {image_topic!r} while synchronizing bag timestamps. "
                f"Available topics: {available}"
            )
        missing_pose_topics = [topic for topic in pose_topics if topic not in topic_types]
        if missing_pose_topics:
            available = ", ".join(sorted(topic_types)) or "<none>"
            raise FileNotFoundError(
                "Missing pose topic(s) while synchronizing bag timestamps: "
                f"{missing_pose_topics}. Available topics: {available}"
            )
        invalid_path_topics = [
            topic for topic in path_topics if topic_types[topic] != PATH_MSG_TYPE
        ]
        if invalid_path_topics:
            raise TypeError(
                "Configured trajectory topic(s) must have type "
                f"{PATH_MSG_TYPE!r}: "
                f"{[(topic, topic_types[topic]) for topic in invalid_path_topics]}"
            )
        invalid_odometry_topics = [
            topic for topic in odometry_topics if topic_types[topic] != ODOMETRY_MSG_TYPE
        ]
        if invalid_odometry_topics:
            raise TypeError(
                "Configured odometry topic(s) must have type "
                f"{ODOMETRY_MSG_TYPE!r}: "
                f"{[(topic, topic_types[topic]) for topic in invalid_odometry_topics]}"
            )

        connections = [
            connection
            for connection in reader.connections
            if connection.topic in selected_topics
        ]

        for connection, _, rawdata in reader.messages(connections=connections):
            msg = reader.deserialize(rawdata, connection.msgtype)
            if connection.topic == image_topic:
                image_timestamps_nsec.add(_stamp_to_nsec(msg.header.stamp))
            elif connection.topic in pose_topics:
                message_pose_timestamps_nsec = _collect_pose_timestamps_nsec(msg, connection.msgtype)
                pose_timestamps_nsec.update(message_pose_timestamps_nsec)
                if connection.msgtype == PATH_MSG_TYPE:
                    path_message_timestamps_nsec.append(
                        (_stamp_to_nsec(msg.header.stamp), message_pose_timestamps_nsec)
                    )

    if not image_timestamps_nsec:
        raise FileNotFoundError(f"No image messages were found on {image_topic} in {resolved_input_bag}")
    if not pose_timestamps_nsec:
        raise FileNotFoundError(
            "No pose timestamps were found on the configured trajectory/odometry topics "
            f"in {resolved_input_bag}"
        )

    return _build_sync_result(
        resolved_input_bag,
        image_timestamps_nsec=image_timestamps_nsec,
        pose_timestamps_nsec=pose_timestamps_nsec,
        path_message_timestamps_nsec=path_message_timestamps_nsec,
        tolerance_sec=tolerance_sec,
        previous_session_image_timestamps_nsec_set=previous_session_image_timestamps_nsec_set,
    )


def compute_folder_image_pose_timestamp_sync(
    image_folder: str | Path,
    poses_path: str | Path,
    *,
    tolerance_sec: float,
    previous_session_image_timestamps_nsec: Iterable[int] = (),
) -> BagTimestampSyncResult:
    if tolerance_sec < 0:
        raise ValueError(f"image_pose_timestamp_tolerance_sec must be non-negative, got {tolerance_sec}")

    image_files = _collect_image_files(image_folder)
    poses = read_pose_file(poses_path)
    image_timestamps_nsec = {timestamp_nsec for timestamp_nsec, _ in image_files}
    pose_timestamps_list = [
        int(pose[0]) * 1_000_000_000 + int(pose[1]) for pose in poses
    ]
    last_pose_nsec = pose_timestamps_list[-1]

    return _build_sync_result(
        Path(image_folder).expanduser(),
        image_timestamps_nsec=image_timestamps_nsec,
        pose_timestamps_nsec=set(pose_timestamps_list),
        path_message_timestamps_nsec=[(last_pose_nsec, pose_timestamps_list)],
        tolerance_sec=tolerance_sec,
        previous_session_image_timestamps_nsec_set={
            int(timestamp_nsec) for timestamp_nsec in previous_session_image_timestamps_nsec
        },
    )
