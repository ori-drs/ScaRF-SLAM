from __future__ import annotations

import csv
import hashlib
import io
import re
import shutil
from bisect import bisect_left
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Set, Tuple

import numpy as np


PATH_MSG_TYPE = "nav_msgs/msg/Path"
ODOMETRY_MSG_TYPE = "nav_msgs/msg/Odometry"
COMPRESSED_IMAGE_MSG_TYPE = "sensor_msgs/msg/CompressedImage"
SYNCED_BAGS_DIRNAME = "synced_bags"
IMAGE_FILENAME_RE = re.compile(r"^image_(\d+)_(\d+)(?:\.[^.]+)?$")

PoseRow = Tuple[int, int, float, float, float, float, float, float, float]


@dataclass(frozen=True)
class BagTimestampSyncResult:
    bag_path: Path
    original_bag_path: Path
    synchronized: bool
    unique_pose_timestamps: int
    exact_matches: int
    rewritten_timestamps: int
    path_header_updates: int
    max_delta_sec: float
    current_session_start_image_timestamp_nsec: Optional[int] = None
    skipped_pose_timestamps: int = 0
    current_session_skipped_pose_timestamps: int = 0
    previous_session_skipped_pose_timestamps: int = 0
    output_bag_path: Optional[Path] = None


def _stamp_to_nsec(stamp) -> int:
    nsec = getattr(stamp, "nanosec", getattr(stamp, "nsec", None))
    if nsec is None:
        raise AttributeError(f"ROS timestamp has no nanosecond field: {stamp!r}")
    return int(stamp.sec) * 1_000_000_000 + int(nsec)


def _timestamp_nsec_key(timestamp_nsec: int) -> str:
    sec, nsec = divmod(int(timestamp_nsec), 1_000_000_000)
    return f"{sec:010d}_{nsec:09d}"


def _set_stamp_from_nsec(stamp, timestamp_nsec: int) -> None:
    sec, nsec = divmod(int(timestamp_nsec), 1_000_000_000)
    stamp.sec = int(sec)
    if hasattr(stamp, "nanosec"):
        stamp.nanosec = int(nsec)
    elif hasattr(stamp, "nsec"):
        stamp.nsec = int(nsec)
    else:
        raise AttributeError(f"ROS timestamp has no nanosecond field: {stamp!r}")


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


def _rewrite_pose_message_timestamps(msg, msgtype: str, timestamp_map: Dict[int, int]) -> Optional[int]:
    if msgtype == PATH_MSG_TYPE:
        header_nsec = _stamp_to_nsec(msg.header.stamp)
        pose_timestamps_nsec = [
            _stamp_to_nsec(pose_stamped.header.stamp)
            for pose_stamped in msg.poses
        ]
        matched_poses = []
        for pose_stamped in msg.poses:
            source_nsec = _stamp_to_nsec(pose_stamped.header.stamp)
            target_nsec = timestamp_map.get(source_nsec)
            if target_nsec is None:
                continue
            _set_stamp_from_nsec(pose_stamped.header.stamp, target_nsec)
            matched_poses.append(pose_stamped)
        if not matched_poses:
            return None
        msg.poses = matched_poses

        target_header_nsec = _path_header_target_nsec(
            header_nsec,
            pose_timestamps_nsec,
            timestamp_map,
        )
        if target_header_nsec is None:
            return None
        if target_header_nsec != header_nsec:
            _set_stamp_from_nsec(msg.header.stamp, target_header_nsec)
        return target_header_nsec

    if msgtype == ODOMETRY_MSG_TYPE:
        source_nsec = _stamp_to_nsec(msg.header.stamp)
        target_nsec = timestamp_map.get(source_nsec)
        if target_nsec is None:
            return None
        _set_stamp_from_nsec(msg.header.stamp, target_nsec)
        return target_nsec

    raise ValueError(f"Unsupported pose message type for timestamp rewrite: {msgtype}")


def _make_output_bag_path(
    input_bag: Path,
    tmp_root: Path,
    *,
    tolerance_sec: float,
    topics: Iterable[Optional[str]],
    extra_image_timestamps_nsec: Iterable[int] = (),
) -> Path:
    extra_images_hash = hashlib.sha1(
        ",".join(
            str(int(timestamp))
            for timestamp in sorted(extra_image_timestamps_nsec)
        ).encode("utf-8")
    ).hexdigest()[:12]
    hash_input = "|".join(
        [
            "sync_v4_drop_unmatched_poses",
            str(input_bag.resolve()),
            str(input_bag.stat().st_mtime_ns),
            f"{tolerance_sec:.9f}",
            *[topic or "<none>" for topic in topics],
            extra_images_hash,
        ]
    )
    digest = hashlib.sha1(hash_input.encode("utf-8")).hexdigest()[:12]
    return tmp_root / SYNCED_BAGS_DIRNAME / f"{input_bag.name}_synced_{digest}"


def cleanup_synced_bag_temporary_data(tmp_root: str | Path) -> None:
    synced_bags_dir = Path(tmp_root).expanduser() / SYNCED_BAGS_DIRNAME
    if synced_bags_dir.exists():
        shutil.rmtree(synced_bags_dir)


def read_current_session_start_image_timestamp_nsec(
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


def _make_header(typestore, timestamp_nsec: int, frame_id: str):
    time_cls = typestore.types["builtin_interfaces/msg/Time"]
    header_cls = typestore.types["std_msgs/msg/Header"]
    sec, nsec = divmod(int(timestamp_nsec), 1_000_000_000)
    return header_cls(time_cls(int(sec), int(nsec)), frame_id)


def _make_pose_stamped(typestore, pose: PoseRow, frame_id: str):
    sec, nsec, x, y, z, qx, qy, qz, qw = pose
    pose_stamped_cls = typestore.types["geometry_msgs/msg/PoseStamped"]
    pose_cls = typestore.types["geometry_msgs/msg/Pose"]
    point_cls = typestore.types["geometry_msgs/msg/Point"]
    quaternion_cls = typestore.types["geometry_msgs/msg/Quaternion"]
    timestamp_nsec = int(sec) * 1_000_000_000 + int(nsec)
    return pose_stamped_cls(
        _make_header(typestore, timestamp_nsec, frame_id),
        pose_cls(
            point_cls(float(x), float(y), float(z)),
            quaternion_cls(float(qx), float(qy), float(qz), float(qw)),
        ),
    )


def _make_file_input_bag_path(
    image_folder: Path,
    poses_path: Path,
    tmp_root: Path,
    *,
    image_topic: str,
    poses_topic: str,
) -> Path:
    image_files = _collect_image_files(image_folder)
    hash_input = "|".join(
        [
            str(image_folder.resolve()),
            str(poses_path.resolve()),
            str(poses_path.stat().st_mtime_ns),
            str(len(image_files)),
            str(max(path.stat().st_mtime_ns for _, path in image_files)),
            image_topic,
            poses_topic,
        ]
    )
    digest = hashlib.sha1(hash_input.encode("utf-8")).hexdigest()[:12]
    return tmp_root / SYNCED_BAGS_DIRNAME / f"{image_folder.name}_images_poses_{digest}"


def create_bag_from_image_folder_and_poses(
    *,
    image_folder: str | Path,
    poses_path: str | Path,
    image_topic: str,
    poses_topic: str,
    tmp_root: str | Path,
    frame_id: str = "map",
) -> Path:
    if image_topic is None:
        raise ValueError("slam_image_topic must not be null")
    if poses_topic is None:
        raise ValueError("slam_final_trajectory_topic must not be null for --poses input")

    try:
        from rosbags.rosbag2 import StoragePlugin, Writer
        from rosbags.typesys import Stores, get_typestore
    except ImportError as exc:
        raise ImportError(
            "Creating a temporary bag from images and poses requires the 'rosbags' Python package."
        ) from exc

    resolved_image_folder = Path(image_folder).expanduser()
    resolved_poses_path = Path(poses_path).expanduser()
    image_files = _collect_image_files(resolved_image_folder)
    poses = read_pose_file(resolved_poses_path)
    print(
        "Preparing temporary image/pose bag: "
        f"images={len(image_files)}, poses={len(poses)}, "
        f"image_folder={resolved_image_folder}, poses={resolved_poses_path}"
    )
    output_bag = _make_file_input_bag_path(
        resolved_image_folder,
        resolved_poses_path,
        Path(tmp_root).expanduser(),
        image_topic=image_topic,
        poses_topic=poses_topic,
    )

    if output_bag.exists():
        print(f"Reusing existing temporary image/pose bag: {output_bag}")
        return output_bag

    output_bag.parent.mkdir(parents=True, exist_ok=True)
    typestore = get_typestore(Stores.ROS2_JAZZY)
    compressed_image_cls = typestore.types[COMPRESSED_IMAGE_MSG_TYPE]
    path_cls = typestore.types[PATH_MSG_TYPE]

    last_pose_nsec = int(poses[-1][0]) * 1_000_000_000 + int(poses[-1][1])
    path_msg = path_cls(
        _make_header(typestore, last_pose_nsec, frame_id),
        [_make_pose_stamped(typestore, pose, frame_id) for pose in poses],
    )
    serialized_path = typestore.serialize_cdr(path_msg, PATH_MSG_TYPE)

    with Writer(
        output_bag,
        version=Writer.VERSION_LATEST,
        storage_plugin=StoragePlugin.MCAP,
    ) as writer:
        image_connection = writer.add_connection(
            image_topic,
            COMPRESSED_IMAGE_MSG_TYPE,
            typestore=typestore,
        )
        poses_connection = writer.add_connection(
            poses_topic,
            PATH_MSG_TYPE,
            typestore=typestore,
        )

        path_written = False
        progress_interval = max(1, min(100, len(image_files) // 10))
        for image_idx, (image_timestamp_nsec, image_path) in enumerate(image_files, start=1):
            if not path_written and image_timestamp_nsec > last_pose_nsec:
                print(
                    "Writing pose path to temporary bag: "
                    f"poses={len(poses)}, timestamp_nsec={last_pose_nsec}"
                )
                writer.write(poses_connection, last_pose_nsec, serialized_path)
                path_written = True

            image_format, image_data = _compressed_image_payload(image_path)
            image_msg = compressed_image_cls(
                _make_header(typestore, image_timestamp_nsec, frame_id),
                image_format,
                image_data,
            )
            writer.write(
                image_connection,
                image_timestamp_nsec,
                typestore.serialize_cdr(image_msg, COMPRESSED_IMAGE_MSG_TYPE),
            )
            if (
                image_idx == 1
                or image_idx == len(image_files)
                or image_idx % progress_interval == 0
            ):
                print(
                    "Writing images to temporary bag: "
                    f"{image_idx}/{len(image_files)} "
                    f"({image_path.name})"
                )

        if not path_written:
            print(
                "Writing pose path to temporary bag: "
                f"poses={len(poses)}, timestamp_nsec={last_pose_nsec}"
            )
            writer.write(poses_connection, last_pose_nsec, serialized_path)

    print(
        "Created temporary image/pose bag: "
        f"{output_bag} "
        f"(images={len(image_files)}, poses={len(poses)}, image_topic={image_topic}, poses_topic={poses_topic})"
    )
    return output_bag


def ensure_bag_image_pose_timestamps_synchronized(
    input_bag: str | Path,
    *,
    image_topic: str,
    trajectory_topic: Optional[str],
    final_trajectory_topic: Optional[str],
    odometry_topic: Optional[str],
    tolerance_sec: float,
    tmp_root: str | Path,
    extra_image_timestamps_nsec: Iterable[int] = (),
) -> BagTimestampSyncResult:
    if image_topic is None:
        raise ValueError("slam_image_topic must not be null")
    if tolerance_sec < 0:
        raise ValueError(f"image_pose_timestamp_tolerance_sec must be non-negative, got {tolerance_sec}")

    try:
        from rosbags.highlevel import AnyReader
        from rosbags.rosbag2 import StoragePlugin, Writer
        from rosbags.typesys import Stores, get_typestore
    except ImportError as exc:
        raise ImportError(
            "Synchronizing bag timestamps requires the 'rosbags' Python package."
        ) from exc

    resolved_input_bag = Path(input_bag).expanduser()
    if not resolved_input_bag.exists():
        raise FileNotFoundError(f"Input bag was not found: {resolved_input_bag}")

    extra_image_timestamps_nsec_set = {
        int(timestamp_nsec)
        for timestamp_nsec in extra_image_timestamps_nsec
    }
    tolerance_nsec = int(round(float(tolerance_sec) * 1_000_000_000))
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
    topic_types: Dict[str, str] = {}

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

    all_image_timestamps_nsec = image_timestamps_nsec | extra_image_timestamps_nsec_set
    timestamp_map, deltas_nsec = _match_pose_timestamps_to_images(
        pose_timestamps_nsec,
        all_image_timestamps_nsec,
        tolerance_nsec,
    )
    _ensure_unique_image_matches(timestamp_map)

    matched_previous_image_timestamps_nsec = set(timestamp_map.values()) & extra_image_timestamps_nsec_set
    missing_previous_image_timestamps_nsec = sorted(
        extra_image_timestamps_nsec_set - matched_previous_image_timestamps_nsec
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
    previous_session_skipped_pose_timestamps = (
        sum(
            1
            for pose_nsec in skipped_pose_timestamps_nsec
            if pose_nsec < current_session_start_image_timestamp_nsec
        )
        if extra_image_timestamps_nsec_set
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
    if (
        rewritten_timestamps == 0
        and path_header_updates == 0
        and current_session_skipped_pose_timestamps == 0
    ):
        return BagTimestampSyncResult(
            bag_path=resolved_input_bag,
            original_bag_path=resolved_input_bag,
            synchronized=False,
            unique_pose_timestamps=len(timestamp_map),
            exact_matches=exact_matches,
            rewritten_timestamps=0,
            path_header_updates=0,
            max_delta_sec=max_delta_sec,
            current_session_start_image_timestamp_nsec=current_session_start_image_timestamp_nsec,
            skipped_pose_timestamps=skipped_pose_timestamps,
            current_session_skipped_pose_timestamps=current_session_skipped_pose_timestamps,
            previous_session_skipped_pose_timestamps=previous_session_skipped_pose_timestamps,
        )

    output_bag = _make_output_bag_path(
        resolved_input_bag,
        Path(tmp_root).expanduser(),
        tolerance_sec=tolerance_sec,
        topics=(image_topic, trajectory_topic, final_trajectory_topic, odometry_topic),
        extra_image_timestamps_nsec=extra_image_timestamps_nsec_set,
    )
    if output_bag.exists():
        return BagTimestampSyncResult(
            bag_path=output_bag,
            original_bag_path=resolved_input_bag,
            synchronized=True,
            unique_pose_timestamps=len(timestamp_map),
            exact_matches=exact_matches,
            rewritten_timestamps=rewritten_timestamps,
            path_header_updates=path_header_updates,
            max_delta_sec=max_delta_sec,
            current_session_start_image_timestamp_nsec=current_session_start_image_timestamp_nsec,
            skipped_pose_timestamps=skipped_pose_timestamps,
            current_session_skipped_pose_timestamps=current_session_skipped_pose_timestamps,
            previous_session_skipped_pose_timestamps=previous_session_skipped_pose_timestamps,
            output_bag_path=output_bag,
        )

    output_bag.parent.mkdir(parents=True, exist_ok=True)

    with AnyReader([resolved_input_bag], default_typestore=typestore) as reader, Writer(
        output_bag,
        version=Writer.VERSION_LATEST,
        storage_plugin=StoragePlugin.MCAP,
    ) as writer:
        output_connections = {}
        for connection in reader.connections:
            if connection.topic not in selected_topics:
                continue
            output_connections[connection.id] = writer.add_connection(
                connection.topic,
                connection.msgtype,
                msgdef=connection.msgdef.data,
                rihs01=connection.digest,
                serialization_format=connection.ext.serialization_format,
                offered_qos_profiles=connection.ext.offered_qos_profiles,
            )

        filtered_connections = [
            connection
            for connection in reader.connections
            if connection.topic in selected_topics
        ]
        for connection, bag_timestamp_nsec, rawdata in reader.messages(connections=filtered_connections):
            out_timestamp_nsec = int(bag_timestamp_nsec)
            out_data = rawdata
            if connection.topic in pose_topics:
                if connection.msgtype not in {PATH_MSG_TYPE, ODOMETRY_MSG_TYPE}:
                    raise TypeError(
                        f"Configured pose topic {connection.topic!r} has type "
                        f"{connection.msgtype!r}, expected {PATH_MSG_TYPE!r} or {ODOMETRY_MSG_TYPE!r}"
                    )
                msg = reader.deserialize(rawdata, connection.msgtype)
                out_timestamp_nsec = _rewrite_pose_message_timestamps(
                    msg,
                    connection.msgtype,
                    timestamp_map,
                )
                if out_timestamp_nsec is None:
                    continue
                out_data = typestore.serialize_cdr(msg, connection.msgtype)

            writer.write(output_connections[connection.id], out_timestamp_nsec, out_data)

    return BagTimestampSyncResult(
        bag_path=output_bag,
        original_bag_path=resolved_input_bag,
        synchronized=True,
        unique_pose_timestamps=len(timestamp_map),
        exact_matches=exact_matches,
        rewritten_timestamps=rewritten_timestamps,
        path_header_updates=path_header_updates,
        max_delta_sec=max_delta_sec,
        current_session_start_image_timestamp_nsec=current_session_start_image_timestamp_nsec,
        skipped_pose_timestamps=skipped_pose_timestamps,
        current_session_skipped_pose_timestamps=current_session_skipped_pose_timestamps,
        previous_session_skipped_pose_timestamps=previous_session_skipped_pose_timestamps,
        output_bag_path=output_bag,
    )
