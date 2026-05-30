#!/usr/bin/env python3
"""Create a ScaRF-SLAM-compatible ROS 2 bag from an input bag plus a trajectory.

The output bag is intended for ScaRF-SLAM's ``use_slam: false`` mode.
By default it writes:

  /insta/cam0/image_raw/compressed   sensor_msgs/msg/CompressedImage
  /insta/poses_est                   nav_msgs/msg/Path

and uses MCAP storage so Python rosbags.highlevel.AnyReader can open it.
"""

from __future__ import annotations

import argparse
import csv
import shutil
from pathlib import Path
from typing import Iterable, List, Optional, Sequence, Tuple

import rosbag2_py
from geometry_msgs.msg import PoseStamped
from nav_msgs.msg import Path as PathMsg
from rclpy.serialization import serialize_message


DEFAULT_TRAJ_TOPIC = "/insta/poses_est"
DEFAULT_IMAGE_TOPIC = "/insta/cam0/image_raw/compressed"
DEFAULT_FRAME_ID = "map"
DEFAULT_COPY_TOPICS = (
    DEFAULT_IMAGE_TOPIC,
    "/insta/imu/data_raw",
)
PATH_MSG_TYPE = "nav_msgs/msg/Path"
SERIALIZATION_FORMAT = "cdr"
DEFAULT_OUTPUT_STORAGE_ID = "mcap"


PoseRow = Tuple[int, int, float, float, float, float, float, float, float]


def _timestamp_to_nsec(sec: int, nsec: int) -> int:
    return int(sec) * 1_000_000_000 + int(nsec)


def _nsec_to_sec_nsec(timestamp_nsec: int) -> Tuple[int, int]:
    return divmod(timestamp_nsec, 1_000_000_000)


def _parse_timestamp_to_nsec(timestamp: str) -> int:
    timestamp_text = timestamp.strip()
    if not timestamp_text:
        raise ValueError("Empty timestamp")
    if timestamp_text.startswith("-"):
        raise ValueError(f"Negative timestamp is not supported: {timestamp}")

    if "." in timestamp_text:
        sec_text, nsec_text = timestamp_text.split(".", 1)
    else:
        sec_text, nsec_text = timestamp_text, ""

    if sec_text == "":
        sec_text = "0"
    if not sec_text.isdigit():
        raise ValueError(f"Invalid seconds in timestamp: {timestamp}")
    if nsec_text and not nsec_text.isdigit():
        raise ValueError(f"Invalid fractional seconds in timestamp: {timestamp}")

    sec = int(sec_text)
    nsec = int((nsec_text + "0" * 9)[:9]) if nsec_text else 0
    return sec * 1_000_000_000 + nsec


def _parse_sec_nsec_to_nsec(sec_value: str, nsec_value: str) -> int:
    sec_text = sec_value.strip()
    nsec_text = nsec_value.strip()
    if not sec_text or not nsec_text:
        raise ValueError("Empty sec/nsec timestamp field")
    if sec_text.startswith("-") or nsec_text.startswith("-"):
        raise ValueError(f"Negative timestamp is not supported: {sec_value}, {nsec_value}")
    if not sec_text.isdigit() or not nsec_text.isdigit():
        raise ValueError(f"Invalid sec/nsec timestamp: {sec_value}, {nsec_value}")

    sec = int(sec_text)
    nsec = int(nsec_text)
    if nsec >= 1_000_000_000:
        raise ValueError(f"Nanoseconds field must be less than 1000000000: {nsec_value}")
    return sec * 1_000_000_000 + nsec


def _parse_pose_row_from_parts(
    parts: Sequence[str],
    row_number: int,
    *,
    csv_row: bool,
) -> Optional[PoseRow]:
    values = [part.strip() for part in parts]
    if not values or not values[0] or values[0].startswith("#"):
        return None

    try:
        if csv_row:
            if values[0].lower() == "counter":
                return None
            if len(values) != 10:
                raise ValueError(
                    "expected counter, sec, nsec, x, y, z, qx, qy, qz, qw"
                )

            timestamp_nsec = _parse_sec_nsec_to_nsec(values[1], values[2])
            sec, nsec = _nsec_to_sec_nsec(timestamp_nsec)
            pose_values = [float(value) for value in values[3:10]]
            return (sec, nsec, *pose_values)

        if len(values) < 8:
            raise ValueError("expected timestamp x y z qx qy qz qw")

        timestamp_nsec = _parse_timestamp_to_nsec(values[0])
        sec, nsec = _nsec_to_sec_nsec(timestamp_nsec)
        pose_values = [float(value) for value in values[1:8]]
        return (sec, nsec, *pose_values)
    except ValueError as exc:
        format_name = "CSV trajectory" if csv_row else "TUM trajectory"
        raise ValueError(f"Invalid {format_name} row at line {row_number}: {parts}") from exc


def read_trajectory_poses(poses_path: Path) -> List[PoseRow]:
    poses: List[PoseRow] = []
    with poses_path.open("r", encoding="utf-8", newline="") as poses_file:
        for row_number, line in enumerate(poses_file, start=1):
            stripped = line.strip()
            if not stripped or stripped.startswith("#"):
                continue

            if "," in stripped:
                parts = next(csv.reader([stripped], skipinitialspace=True))
                pose = _parse_pose_row_from_parts(parts, row_number, csv_row=True)
            else:
                parts = stripped.split()
                pose = _parse_pose_row_from_parts(parts, row_number, csv_row=False)

            if pose is not None:
                poses.append(pose)

    if not poses:
        raise ValueError(f"No trajectory poses found in {poses_path}")

    return sorted(poses, key=lambda pose: (pose[0], pose[1]))


def make_trajectory_path_message(
    poses: Iterable[PoseRow],
    *,
    frame_id: str,
    header_sec: int,
    header_nsec: int,
) -> PathMsg:
    path_msg = PathMsg()
    path_msg.header.frame_id = frame_id
    path_msg.header.stamp.sec = int(header_sec)
    path_msg.header.stamp.nanosec = int(header_nsec)

    for sec, nsec, x, y, z, qx, qy, qz, qw in poses:
        pose_msg = PoseStamped()
        pose_msg.header.frame_id = frame_id
        pose_msg.header.stamp.sec = int(sec)
        pose_msg.header.stamp.nanosec = int(nsec)
        pose_msg.pose.position.x = float(x)
        pose_msg.pose.position.y = float(y)
        pose_msg.pose.position.z = float(z)
        pose_msg.pose.orientation.x = float(qx)
        pose_msg.pose.orientation.y = float(qy)
        pose_msg.pose.orientation.z = float(qz)
        pose_msg.pose.orientation.w = float(qw)
        path_msg.poses.append(pose_msg)

    return path_msg


def _make_topic_metadata(
    name: str,
    msg_type: str,
    serialization_format: str = SERIALIZATION_FORMAT,
    offered_qos_profiles=None,
    type_description_hash: str = "",
):
    if offered_qos_profiles is None:
        offered_qos_profiles = []

    try:
        return rosbag2_py.TopicMetadata(
            0,
            name,
            msg_type,
            serialization_format,
            offered_qos_profiles,
            type_description_hash,
        )
    except TypeError:
        return rosbag2_py.TopicMetadata(
            0,
            name,
            msg_type,
            serialization_format,
            offered_qos_profiles,
        )


def _copy_topic_metadata(topic_metadata, *, keep_type_hash: bool = False):
    return _make_topic_metadata(
        topic_metadata.name,
        topic_metadata.type,
        topic_metadata.serialization_format,
        topic_metadata.offered_qos_profiles,
        getattr(topic_metadata, "type_description_hash", "") if keep_type_hash else "",
    )


def _metadata_by_topic(metadata) -> dict:
    return {
        topic_info.topic_metadata.name: topic_info.topic_metadata
        for topic_info in metadata.topics_with_message_count
    }


def _remove_existing_output(path: Path, *, force: bool) -> None:
    if not path.exists():
        return
    if not force:
        raise FileExistsError(
            f"Output bag already exists: {path}. Use --force to overwrite it."
        )
    if path.is_dir():
        shutil.rmtree(path)
    else:
        path.unlink()


def write_traj_to_output_bag(
    input_bag: Path,
    output_bag: Path,
    poses: Sequence[PoseRow],
    *,
    topic: str,
    frame_id: str,
    topics_to_copy: Sequence[str],
    output_storage_id: str,
    force: bool,
    keep_type_hash: bool,
) -> Tuple[int, int]:
    metadata = rosbag2_py.Info().read_metadata(str(input_bag), "")
    topic_metadata_by_name = _metadata_by_topic(metadata)

    if topic in topics_to_copy:
        raise ValueError(
            f"Cannot copy {topic} from the input bag and write trajectory poses to the same topic."
        )

    missing_topics = [
        topic_name for topic_name in topics_to_copy if topic_name not in topic_metadata_by_name
    ]
    if missing_topics:
        available_topics = ", ".join(sorted(topic_metadata_by_name)) or "<none>"
        raise ValueError(
            f"Input bag is missing requested topics: {missing_topics}. "
            f"Available topics: {available_topics}"
        )

    _remove_existing_output(output_bag, force=force)

    last_sec, last_nsec = poses[-1][0], poses[-1][1]
    trajectory_timestamp_nsec = _timestamp_to_nsec(last_sec, last_nsec)

    trajectory_msg = make_trajectory_path_message(
        poses,
        frame_id=frame_id,
        header_sec=last_sec,
        header_nsec=last_nsec,
    )
    serialized_trajectory_msg = serialize_message(trajectory_msg)

    reader = rosbag2_py.SequentialReader()
    reader.open(
        rosbag2_py.StorageOptions(
            uri=str(input_bag),
            storage_id=metadata.storage_identifier,
        ),
        rosbag2_py.ConverterOptions("", ""),
    )

    writer = rosbag2_py.SequentialWriter()
    writer.open(
        rosbag2_py.StorageOptions(
            uri=str(output_bag),
            storage_id=output_storage_id,
        ),
        rosbag2_py.ConverterOptions("", ""),
    )

    selected_topics = set(topics_to_copy)

    for topic_name in topics_to_copy:
        writer.create_topic(
            _copy_topic_metadata(
                topic_metadata_by_name[topic_name],
                keep_type_hash=keep_type_hash,
            )
        )

    writer.create_topic(_make_topic_metadata(topic, PATH_MSG_TYPE))

    trajectory_written = False
    copied_count = 0

    while reader.has_next():
        topic_name, data, timestamp_nsec = reader.read_next()
        if topic_name not in selected_topics:
            continue

        if not trajectory_written and timestamp_nsec > trajectory_timestamp_nsec:
            writer.write(topic, serialized_trajectory_msg, trajectory_timestamp_nsec)
            trajectory_written = True

        writer.write(topic_name, data, timestamp_nsec)
        copied_count += 1

    if not trajectory_written:
        writer.write(topic, serialized_trajectory_msg, trajectory_timestamp_nsec)

    writer.close()
    reader.close()

    return copied_count, len(poses)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Copy selected topics from a ROS 2 bag and add a trajectory file "
            "as one nav_msgs/Path message for ScaRF-SLAM use_slam:false mode."
        )
    )
    parser.add_argument(
        "--in-bag",
        type=Path,
        required=True,
        help="Input ROS 2 bag directory.",
    )
    parser.add_argument(
        "--in-poses",
        type=Path,
        required=True,
        help=(
            "Input trajectory in TUM format or ScaRF-SLAM "
            "counter,sec,nsec,x,y,z,qx,qy,qz,qw CSV format."
        ),
    )
    parser.add_argument(
        "--out-bag",
        type=Path,
        required=True,
        help="Output ROS 2 bag directory.",
    )
    parser.add_argument(
        "--topics",
        nargs="+",
        default=list(DEFAULT_COPY_TOPICS),
        help=f"Input topics to copy. Default: {', '.join(DEFAULT_COPY_TOPICS)}.",
    )
    parser.add_argument("--topic", default=DEFAULT_TRAJ_TOPIC, help="Output trajectory topic.")
    parser.add_argument("--frame-id", default=DEFAULT_FRAME_ID, help="Path frame_id.")
    parser.add_argument(
        "--storage-id",
        default=DEFAULT_OUTPUT_STORAGE_ID,
        choices=("mcap", "sqlite3"),
        help="Output bag storage backend. Default: mcap.",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Overwrite the output bag directory if it already exists.",
    )
    parser.add_argument(
        "--keep-type-hash",
        action="store_true",
        help="Preserve source type_description_hash metadata on copied topics.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    input_bag = args.in_bag.expanduser().resolve()
    if not input_bag.exists():
        raise FileNotFoundError(f"Input bag does not exist: {input_bag}")

    poses_path = args.in_poses.expanduser().resolve()
    if not poses_path.exists():
        raise FileNotFoundError(f"Trajectory file does not exist: {poses_path}")

    poses = read_trajectory_poses(poses_path)

    output_bag = args.out_bag.expanduser().resolve()

    copied_count, trajectory_pose_count = write_traj_to_output_bag(
        input_bag,
        output_bag,
        poses,
        topic=args.topic,
        frame_id=args.frame_id,
        topics_to_copy=args.topics,
        output_storage_id=args.storage_id,
        force=args.force,
        keep_type_hash=args.keep_type_hash,
    )

    print(
        f"Wrote ScaRF-SLAM-compatible {args.storage_id} bag: {output_bag}\n"
        f"  copied messages: {copied_count} from {len(args.topics)} topic(s)\n"
        f"  trajectory poses: {trajectory_pose_count} on {args.topic}\n"
        f"  copied topics:   {', '.join(args.topics)}"
    )


if __name__ == "__main__":
    main()
