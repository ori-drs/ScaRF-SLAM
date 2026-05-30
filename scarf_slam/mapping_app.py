#!/usr/bin/env python3
# Optional config for better memory efficiency
# -------------------------
# Standard Library Imports
# -------------------------
import argparse
import json
import os
import pathlib
import re
import sys
from bisect import bisect_right
from types import SimpleNamespace
from typing import List, Sequence, Union, Optional, Dict, Tuple, Set
import shutil
import math
import time
from pathlib import Path

# Environment variables
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"

# -------------------------
# Third-Party Imports
# -------------------------
import numpy as np
import torch
import gc
import yaml

# -------------------------
# Local Package Imports
# -------------------------
from scarf_slam.core.timestamp import MappingTimestamp
from scarf_slam.core.pose import MappingPose, MappingTransforms

PACKAGE_DIR = pathlib.Path(__file__).resolve().parent
REPO_ROOT = PACKAGE_DIR.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from scarf_slam.mapping import fusion as map_fusion
from scarf_slam.mapping import feature_matching
from scarf_slam.mapping import scale_optimization as gtsam_optimization
from scarf_slam.core.covisibility_graph import FrameSubmapCovisibilityGraph
from scarf_slam.core.submap import SubmapRecord
from scarf_slam.core.camera import (
    PinholeCamera,
    RotateParam,
    load_fisheye_cameras_from_config,
)
from scarf_slam.utils.pointcloud_ops import (
    depth_to_world_points_vectorized,
    submap_to_world_pointcloud,
    world_points_to_anchor_local,
)
from scarf_slam.mapping import graph_io
from scarf_slam.utils.timestamp_ops import (
    pose_dicts_equal,
    timestamp_key_to_nsec,
    timestamp_key_to_seconds,
    timestamp_key_to_timestamp,
    timestamp_nsec_to_key,
)
from scarf_slam.utils import keyframe_selection
from scarf_slam.integrations.slam_bag_time_sync import (
    cleanup_synced_bag_temporary_data,
    create_bag_from_image_folder_and_poses,
    ensure_bag_image_pose_timestamps_synchronized,
    read_current_session_start_image_timestamp_nsec,
)
from scarf_slam.integrations.slam_bag import (
    DEFAULT_FINAL_TRAJECTORY_TOPIC,
    DEFAULT_IMAGE_TOPIC,
    DEFAULT_ODOMETRY_TOPIC,
    DEFAULT_TRAJECTORY_TOPIC,
    load_slam_bag,
)
from scarf_slam.integrations import ros2_publishing
from scarf_slam.integrations.ros2_publishing import (
    ImageMsg,
    Node,
    PathMsg,
    PointCloud2,
    PoseStamped,
    create_qos_profile,
    ensure_ros2_available,
    point_cloud_xyzrgb,
    rclpy,
    set_ros_header_stamp,
    timestamp_plus_seconds,
)
try:
    from scarf_slam.backends import depthanything as depthanything_insta
except ImportError:
    pass


TIMESTAMP_KEY_RE = re.compile(r"^\d{10}_\d{9}$")
ANSI_GREEN = "\033[92m"
ANSI_YELLOW = "\033[93m"
ANSI_RESET = "\033[0m"


def _iter_manifest_timestamp_keys(value) -> Set[str]:
    timestamp_keys: Set[str] = set()
    if isinstance(value, dict):
        for key, child in value.items():
            key_text = str(key)
            if TIMESTAMP_KEY_RE.match(key_text):
                timestamp_keys.add(key_text)
            timestamp_keys.update(_iter_manifest_timestamp_keys(child))
    elif isinstance(value, list):
        for child in value:
            timestamp_keys.update(_iter_manifest_timestamp_keys(child))
    elif isinstance(value, str) and TIMESTAMP_KEY_RE.match(value):
        timestamp_keys.add(value)
    return timestamp_keys


def _require_open3d():
    try:
        import open3d as o3d
    except ImportError as exc:
        raise ImportError(
            "open3d is required for point cloud export. Install open3d before calling save_*pointcloud* methods."
        ) from exc
    return o3d


def _require_cuda():
    if not torch.cuda.is_available():
        raise RuntimeError(
            "CUDA GPU is required to run ScaRF-SLAM mapping, but no CUDA-capable GPU was detected."
        )
    return torch.device("cuda")


class ScaRFSLAM():
    def __init__(self, slam_folder, input_bag=None, prev_slam_folder=None, image_folder=None, poses=None):
        self.slam_folder = slam_folder
        self.input_bag = input_bag
        self.prev_slam_folder = prev_slam_folder
        self.image_folder = image_folder
        self.poses = poses

        self.transforms = MappingTransforms()
        self.submaps: Dict[str, SubmapRecord] = {}
        self.in_ref_poses_dict: Dict[str, MappingPose] = {}
        self.odom_ref_poses_dict: Dict[str, MappingPose] = {}
        self.previous_session_frame_keys: Set[str] = set()
        self.out_ph_poses_dict: Dict[str, MappingPose] = {}
        self.ph_to_ref_dict: Dict[str, Tuple[MappingTimestamp, MappingPose]] = {}
        self.covisibility_graph = FrameSubmapCovisibilityGraph(self.transforms)
        self.ref_traj_snapshot_timestamps = []
        self.current_traj_timestamp: Optional[str] = None
        self.ref_timestamps = []
        self.ros2_node = None
        self.ros2_pointcloud_publisher = None
        self.ros2_path_publisher = None
        self.ros2_prev_path_publisher = None
        self.ros2_images_publisher = None
        self.publish_ros2_pointcloud = False
        self.ros2_pointcloud_downsample_ratio = 0.0
        self.ros2_pointcloud_topic = "/scarf_slam/clouds"
        self.ros2_pointcloud_frame_id = "map"
        self.ros2_path_topic = "/scarf_slam/slam_poses"
        self.ros2_prev_path_topic = "/scarf_slam/slam_poses_prev"
        self.publish_ros2_images = False
        self.ros2_images_topic = "/scarf_slam/images"
        self.slam_bag_data = None
        self.timestamp_sync_result = None
        self.timestamp_sync_tmp_root = None
        self.is_mono = False
        self.fixed_mono_trajectory_scale: Optional[float] = None

        self.model_inference_time = 0.0
        self.feature_match_time = 0.0
        self.frame_scale_opt_time = 0.0
        self.submap_scale_opt_time = 0.0
        self.fuse_pts_time = 0.0


    def _print_timing_line(self, label: str, duration: float, indent: int = 0, color: Optional[str] = None) -> None:
        prefix = "\t" * indent
        line = f"{prefix}{label}: {duration:.6f} seconds"
        if color is not None:
            print(f"{color}{line}\033[0m")
        else:
            print(line)


    def _add_elapsed_time(self, attr_name: str, start_time: float) -> float:
        duration = time.perf_counter() - start_time
        setattr(self, attr_name, getattr(self, attr_name) + duration)
        return duration


    def _print_runtime_summary(self, num_submaps_processed: int) -> None:
        avg_divisor = max(1, int(num_submaps_processed))
        print("Runtime Summary:")
        self._print_timing_line("Model Inference Time", self.model_inference_time, indent=0)
        self._print_timing_line("Feature Match Time", self.feature_match_time, indent=0)
        self._print_timing_line("Frame Scale Optimization Time", self.frame_scale_opt_time, indent=0)
        self._print_timing_line("Submap Scale Optimization Time", self.submap_scale_opt_time, indent=0)
        self._print_timing_line("Points Fusion Time", self.fuse_pts_time, indent=0)
        print("Average Per Submap:")
        self._print_timing_line("Model Inference Time", self.model_inference_time / avg_divisor, indent=0)
        self._print_timing_line("Feature Match Time", self.feature_match_time / avg_divisor, indent=0)
        self._print_timing_line("Frame Scale Optimization Time", self.frame_scale_opt_time / avg_divisor, indent=0)
        self._print_timing_line("Submap Scale Optimization Time", self.submap_scale_opt_time / avg_divisor, indent=0)
        self._print_timing_line("Points Fusion Time", self.fuse_pts_time / avg_divisor, indent=0)


    def _transform_world_points_to_anchor_local(
        self,
        pts_world: np.ndarray,
        anchor_pose: MappingPose,
    ) -> np.ndarray:
        return world_points_to_anchor_local(pts_world, anchor_pose, self.transforms)


    def _submap_to_world_pointcloud(
        self,
        submap: SubmapRecord,
        point_mask: Optional[np.ndarray] = None,
    ) -> Tuple[np.ndarray, np.ndarray]:
        if submap.anchor_key not in self.out_ph_poses_dict:
            raise KeyError(f"Missing anchor pose for submap '{submap.anchor_key}' in out_ph_poses_dict.")

        return submap_to_world_pointcloud(
            submap,
            self.out_ph_poses_dict[submap.anchor_key],
            self.transforms,
            point_mask=point_mask,
        )


    def _build_publish_point_mask(self, local_points: np.ndarray, downsample_ratio: float) -> np.ndarray:
        if not math.isfinite(downsample_ratio) or downsample_ratio < 0.0 or downsample_ratio > 1.0:
            raise ValueError("downsample_ratio must be between 0 and 1.")

        local_points = np.asarray(local_points)
        if local_points.ndim != 2 or local_points.shape[1] < 4:
            raise ValueError(f"local_points must have shape (N, at least 4), got {local_points.shape}")

        mask = np.zeros(local_points.shape[0], dtype=bool)
        conf = local_points[:, 3]
        valid_indices = np.flatnonzero(np.isfinite(conf) & (conf != 0.0))
        if downsample_ratio == 0.0 or valid_indices.size == 0:
            return np.ascontiguousarray(mask)

        if downsample_ratio == 1.0:
            mask[valid_indices] = True
            return np.ascontiguousarray(mask)

        keep_count = int(math.ceil(valid_indices.size * downsample_ratio))
        rng = np.random.default_rng(0)
        selected_offsets = rng.choice(valid_indices.size, size=keep_count, replace=False)
        mask[valid_indices[selected_offsets]] = True
        return np.ascontiguousarray(mask)


    def _get_publish_point_mask(self, submap: SubmapRecord, downsample_ratio: float) -> np.ndarray:
        if (
            submap.publish_downsample_ratio == downsample_ratio
            and submap.publish_point_mask.shape[0] == submap.local_points.shape[0]
        ):
            return submap.publish_point_mask

        submap.publish_point_mask = self._build_publish_point_mask(submap.local_points, downsample_ratio)
        submap.publish_downsample_ratio = downsample_ratio
        return submap.publish_point_mask


    def _compose_global_pointcloud(self, sample_offset: int = 1) -> Tuple[np.ndarray, np.ndarray]:
        if sample_offset <= 0:
            raise ValueError("sample_offset must be positive.")

        pts_chunks: List[np.ndarray] = []
        color_chunks: List[np.ndarray] = []
        for submap_key in sorted(self.submaps.keys()):
            pts_world, colors = self._submap_to_world_pointcloud(self.submaps[submap_key])
            if pts_world.shape[0] == 0:
                continue
            if sample_offset > 1:
                pts_world = pts_world[::sample_offset]
                colors = colors[::sample_offset]
            pts_chunks.append(pts_world)
            color_chunks.append(colors)

        if not pts_chunks:
            return (
                np.empty((0, 4), dtype=np.float32),
                np.empty((0, 3), dtype=np.uint8),
            )

        return (
            np.ascontiguousarray(np.concatenate(pts_chunks, axis=0)),
            np.ascontiguousarray(np.concatenate(color_chunks, axis=0)),
        )


    def _compose_publish_pointcloud(self, downsample_ratio: float) -> Tuple[np.ndarray, np.ndarray]:
        if not math.isfinite(downsample_ratio) or downsample_ratio < 0.0 or downsample_ratio > 1.0:
            raise ValueError("downsample_ratio must be between 0 and 1.")

        publish_items: List[Tuple[SubmapRecord, np.ndarray, int]] = []
        total_points = 0
        for submap_key in sorted(self.submaps.keys()):
            submap = self.submaps[submap_key]
            point_mask = self._get_publish_point_mask(submap, downsample_ratio)
            num_points = int(np.count_nonzero(point_mask))
            if num_points == 0:
                continue
            publish_items.append((submap, point_mask, num_points))
            total_points += num_points

        if total_points == 0:
            return (
                np.empty((0, 3), dtype=np.float32),
                np.empty((0, 3), dtype=np.uint8),
            )

        points = np.empty((total_points, 3), dtype=np.float32)
        colors_out = np.empty((total_points, 3), dtype=np.uint8)
        write_idx = 0
        for submap, point_mask, num_points in publish_items:
            if submap.anchor_key not in self.out_ph_poses_dict:
                raise KeyError(f"Missing anchor pose for submap '{submap.anchor_key}' in out_ph_poses_dict.")

            local_xyz = np.asarray(submap.local_points[point_mask, :3], dtype=np.float32)
            if local_xyz.shape[0] == 0:
                continue
            end_idx = write_idx + local_xyz.shape[0]
            if local_xyz.shape[0] != num_points:
                raise ValueError(
                    "Publish mask count changed during point cloud composition: "
                    f"expected={num_points}, actual={local_xyz.shape[0]}"
                )
            t_world_anchor = self.transforms.pose_to_matrix(self.out_ph_poses_dict[submap.anchor_key]).astype(np.float32, copy=False)
            scaled_local_xyz = np.float32(submap.scale) * local_xyz
            points[write_idx:end_idx] = (
                (t_world_anchor[:3, :3] @ scaled_local_xyz.T).T + t_world_anchor[:3, 3]
            )
            colors_out[write_idx:end_idx] = submap.colors[point_mask]
            write_idx = end_idx

        if write_idx != total_points:
            points = points[:write_idx]
            colors_out = colors_out[:write_idx]

        return points, colors_out


    def _load_runtime_config(self, config_path: Union[str, Path]) -> dict:
        config_path = Path(config_path).expanduser()
        if not config_path.exists():
            raise FileNotFoundError(f"Config file not found: {config_path}")
        with open(config_path, "r") as f:
            data = yaml.safe_load(f) or {}
        if not isinstance(data, dict):
            raise ValueError(f"Config file must contain a YAML mapping: {config_path}")
        print(f"\n=== Runtime Config Loaded: {config_path} ===")
        print(yaml.dump(data, sort_keys=False, default_flow_style=False))
        return data


    def _setup_ros2_publisher(self) -> None:
        if (
            not self.publish_ros2_pointcloud
            and not self.publish_ros2_path
            and not self.publish_ros2_images
        ):
            return
        ensure_ros2_available(
            publish_pointcloud=self.publish_ros2_pointcloud,
            publish_path=self.publish_ros2_path,
            publish_images=self.publish_ros2_images,
        )
        if not rclpy.ok():
            rclpy.init(args=None)
        if self.ros2_node is None:
            self.ros2_node = Node("depthanything_mapping_publisher")
            qos_profile = create_qos_profile()
            if self.publish_ros2_pointcloud:
                self.ros2_pointcloud_publisher = self.ros2_node.create_publisher(
                    PointCloud2,
                    self.ros2_pointcloud_topic,
                    qos_profile,
                )
            if self.publish_ros2_path:
                self.ros2_path_publisher = self.ros2_node.create_publisher(
                    PathMsg,
                    self.ros2_path_topic,
                    qos_profile,
                )
                self.ros2_prev_path_publisher = self.ros2_node.create_publisher(
                    PathMsg,
                    self.ros2_prev_path_topic,
                    qos_profile,
                )
            if self.publish_ros2_images:
                self.ros2_images_publisher = self.ros2_node.create_publisher(
                    ImageMsg,
                    self.ros2_images_topic,
                    qos_profile,
                )


    def _publish_sampled_global_pointcloud(self, header_timestamp: Optional[MappingTimestamp] = None) -> None:
        if not self.publish_ros2_pointcloud:
            return
        if self.ros2_pointcloud_publisher is None or self.ros2_node is None:
            raise RuntimeError("ROS2 point cloud publisher is not initialized")

        downsample_ratio = self.ros2_pointcloud_downsample_ratio
        if not math.isfinite(downsample_ratio) or downsample_ratio < 0.0 or downsample_ratio > 1.0:
            raise ValueError("ROS2 point cloud downsample ratio must be between 0 and 1")
        if downsample_ratio == 0.0:
            return

        sampled_points, sampled_colors_global = self._compose_publish_pointcloud(
            downsample_ratio=downsample_ratio
        )
        if sampled_points.shape[0] == 0:
            raise ValueError("Global map is empty; cannot publish a ROS2 point cloud")
        if sampled_colors_global.shape[0] != sampled_points.shape[0]:
            raise ValueError(
                "Sampled global point/color size mismatch for ROS2 point cloud publish: "
                f"points={sampled_points.shape[0]}, colors={sampled_colors_global.shape[0]}"
            )

        sampled_points = np.ascontiguousarray(sampled_points.astype(np.float32, copy=False))
        sampled_colors_global = np.ascontiguousarray(sampled_colors_global.astype(np.uint8, copy=False))
        pcd_msg = point_cloud_xyzrgb(
            sampled_points,
            sampled_colors_global,
            self.ros2_pointcloud_frame_id,
        )
        set_ros_header_stamp(self.ros2_node, pcd_msg.header, header_timestamp)
        self.ros2_pointcloud_publisher.publish(pcd_msg)
        rclpy.spin_once(self.ros2_node, timeout_sec=0.0)


    def _publish_out_ph_poses_path(self, header_timestamp: Optional[MappingTimestamp] = None) -> None:
        if not self.publish_ros2_path:
            return
        if (
            self.ros2_path_publisher is None
            or self.ros2_prev_path_publisher is None
            or self.ros2_node is None
        ):
            raise RuntimeError("ROS2 path publisher is not initialized")
        if not self.out_ph_poses_dict:
            raise ValueError("out_ph_poses_dict is empty; cannot publish ROS2 path")

        path_msg = PathMsg()
        path_msg.header.frame_id = self.ros2_pointcloud_frame_id
        set_ros_header_stamp(self.ros2_node, path_msg.header, header_timestamp)

        prev_path_msg = PathMsg()
        prev_path_msg.header.frame_id = self.ros2_pointcloud_frame_id
        set_ros_header_stamp(self.ros2_node, prev_path_msg.header, header_timestamp)

        for ts_key in sorted(self.out_ph_poses_dict.keys()):
            sec_str, nsec_str = ts_key.split("_")
            pose = self.out_ph_poses_dict[ts_key]

            pose_stamped = PoseStamped()
            pose_stamped.header.frame_id = self.ros2_pointcloud_frame_id
            pose_stamped.header.stamp.sec = int(sec_str)
            pose_stamped.header.stamp.nanosec = int(nsec_str)
            pose_stamped.pose.position.x = float(pose.pos[0])
            pose_stamped.pose.position.y = float(pose.pos[1])
            pose_stamped.pose.position.z = float(pose.pos[2])
            pose_stamped.pose.orientation.x = float(pose.quat[0])
            pose_stamped.pose.orientation.y = float(pose.quat[1])
            pose_stamped.pose.orientation.z = float(pose.quat[2])
            pose_stamped.pose.orientation.w = float(pose.quat[3])
            if ts_key in self.previous_session_frame_keys:
                prev_path_msg.poses.append(pose_stamped)
            else:
                path_msg.poses.append(pose_stamped)

        self.ros2_path_publisher.publish(path_msg)
        self.ros2_prev_path_publisher.publish(prev_path_msg)
        rclpy.spin_once(self.ros2_node, timeout_sec=0.0)


    def _publish_loaded_previous_session(self, header_timestamp: MappingTimestamp) -> None:
        self._publish_sampled_global_pointcloud(header_timestamp=header_timestamp)
        self._publish_out_ph_poses_path(header_timestamp=header_timestamp)


    def _publish_processed_images(self, processed_images: np.ndarray, header_timestamp: Optional[MappingTimestamp] = None) -> None:
        if not self.publish_ros2_images:
            return
        if self.ros2_images_publisher is None or self.ros2_node is None:
            raise RuntimeError("ROS2 image publisher is not initialized")

        grid_rgb = ros2_publishing.concat_images_n_by_3(processed_images)
        target_width = 800
        if grid_rgb.shape[1] != target_width:
            target_height = max(1, int(round(grid_rgb.shape[0] * target_width / grid_rgb.shape[1])))
            grid_rgb = ros2_publishing.cv2.resize(
                grid_rgb,
                (target_width, target_height),
                interpolation=(
                    ros2_publishing.cv2.INTER_AREA
                    if target_width < grid_rgb.shape[1]
                    else ros2_publishing.cv2.INTER_LINEAR
                ),
            )
            grid_rgb = np.ascontiguousarray(grid_rgb)
        image_msg = ImageMsg()
        set_ros_header_stamp(self.ros2_node, image_msg.header, header_timestamp)
        image_msg.header.frame_id = self.ros2_pointcloud_frame_id
        image_msg.height = int(grid_rgb.shape[0])
        image_msg.width = int(grid_rgb.shape[1])
        image_msg.encoding = "rgb8"
        image_msg.is_bigendian = 0
        image_msg.step = int(grid_rgb.shape[1] * grid_rgb.shape[2])
        image_msg.data = grid_rgb.tobytes()

        self.ros2_images_publisher.publish(image_msg)
        rclpy.spin_once(self.ros2_node, timeout_sec=0.0)


    def _resolve_prev_graph_dir(self, prev_slam_folder: Union[str, Path]) -> Path:
        root = Path(prev_slam_folder)
        if (root / "manifest.json").exists():
            return root

        candidates = [
            path
            for path in sorted(root.glob("recon/*/opt_graph*"))
            if path.is_dir() and (path / "manifest.json").exists()
        ]
        candidates.extend(
            path
            for path in sorted(root.glob("opt_graph*"))
            if path.is_dir() and (path / "manifest.json").exists()
        )
        if not candidates:
            raise FileNotFoundError(
                "Could not find a saved graph manifest. Pass either the graph directory "
                f"or a SLAM folder containing recon/*/opt_graph*/manifest.json: {root}"
            )
        if len(candidates) > 1:
            raise ValueError(
                "Multiple previous graph artifacts found; pass the exact graph directory: "
                f"{candidates}"
            )
        return candidates[0]


    def _previous_graph_timestamps_nsec(
        self,
        prev_slam_folder: Union[str, Path],
    ) -> Tuple[Set[int], Set[int]]:
        graph_dir = self._resolve_prev_graph_dir(prev_slam_folder)
        manifest = graph_io.load_graph_manifest(graph_dir)
        frames_json = graph_io.require_graph_key(manifest, "frames", "Previous graph manifest")
        if not isinstance(frames_json, dict):
            raise ValueError("Previous graph frames manifest must be a JSON object.")

        image_timestamps_nsec: Set[int] = set()
        data_timestamps_nsec = {
            timestamp_key_to_nsec(
                timestamp_key,
                f"Previous graph manifest timestamp '{timestamp_key}'",
            )
            for timestamp_key in _iter_manifest_timestamp_keys(manifest)
        }
        frames_dir = graph_dir / "frames"
        for frame_key, frame_info in sorted(frames_json.items()):
            if not isinstance(frame_info, dict):
                raise ValueError(f"Invalid frame manifest entry for {frame_key}: {frame_info!r}")
            image_path = graph_io.require_graph_key(frame_info, "image", f"Frame '{frame_key}'")
            resolved_image_path = frames_dir / str(frame_key) / str(image_path)
            if not resolved_image_path.exists():
                raise FileNotFoundError(
                    f"Previous graph frame '{frame_key}' is missing cached image: "
                    f"{resolved_image_path}"
                )

            frame_timestamp_nsec = timestamp_key_to_nsec(
                str(frame_key),
                f"Previous graph frame '{frame_key}'",
            )
            image_timestamps_nsec.add(frame_timestamp_nsec)

        return image_timestamps_nsec, data_timestamps_nsec


    def _validate_previous_graph_before_current_session(
        self,
        *,
        previous_data_timestamps_nsec: Set[int],
        current_session_start_nsec: int,
    ) -> None:
        later_timestamps_nsec = sorted(
            timestamp_nsec
            for timestamp_nsec in previous_data_timestamps_nsec
            if timestamp_nsec > current_session_start_nsec
        )
        if not later_timestamps_nsec:
            return

        first_later_nsec = later_timestamps_nsec[0]
        first_later_sec, first_later_sub_nsec = divmod(first_later_nsec, 1_000_000_000)
        start_sec, start_sub_nsec = divmod(current_session_start_nsec, 1_000_000_000)
        raise ValueError(
            "Previous session data must be earlier than the current session. "
            f"Found {len(later_timestamps_nsec)} previous-session timestamp(s) greater than "
            f"the current session start time from first image "
            f"{start_sec:010d}_{start_sub_nsec:09d}; "
            f"first offending previous timestamp is {first_later_sec:010d}_{first_later_sub_nsec:09d}."
        )


    def _load_graph_submaps(
        self,
        graph_dir: Path,
        manifest: Dict[str, object],
    ) -> Dict[str, SubmapRecord]:
        submaps_json = graph_io.require_graph_key(manifest, "submaps", "Previous graph manifest")
        if not isinstance(submaps_json, dict) or not submaps_json:
            raise ValueError("Previous graph manifest has no submaps.")

        loaded_submaps: Dict[str, SubmapRecord] = {}
        submaps_dir = graph_dir / "submaps"
        for submap_key, submap_info in sorted(submaps_json.items()):
            if not isinstance(submap_info, dict):
                raise ValueError(f"Invalid submap manifest entry for {submap_key}: {submap_info!r}")
            submap_dir = submaps_dir / submap_key
            anchor_key = str(graph_io.require_graph_key(submap_info, "anchor_key", f"Submap '{submap_key}'"))
            frame_keys_json = graph_io.require_graph_key(submap_info, "frame_keys", f"Submap '{submap_key}'")
            if not isinstance(frame_keys_json, list) or not frame_keys_json:
                raise ValueError(f"Submap '{submap_key}' must contain a non-empty frame_keys list.")
            scale = float(graph_io.require_graph_key(submap_info, "scale", f"Submap '{submap_key}'"))
            publish_downsample_ratio = float(
                graph_io.require_graph_key(
                    submap_info,
                    "publish_downsample_ratio",
                    f"Submap '{submap_key}'",
                )
            )
            local_points = graph_io.load_graph_array(
                submap_dir / str(graph_io.require_graph_key(submap_info, "local_points", f"Submap '{submap_key}'")),
                dtype=np.float32,
                ndim=2,
                shape_tail=(4,),
            )
            colors = graph_io.load_graph_array(
                submap_dir / str(graph_io.require_graph_key(submap_info, "colors", f"Submap '{submap_key}'")),
                dtype=np.uint8,
                ndim=2,
                shape_tail=(3,),
            )
            unique_point_ids = graph_io.load_graph_array(
                submap_dir / str(graph_io.require_graph_key(submap_info, "unique_point_ids", f"Submap '{submap_key}'")),
                dtype=np.int64,
                ndim=1,
            )

            publish_point_mask_path = graph_io.require_graph_key(
                submap_info,
                "publish_point_mask",
                f"Submap '{submap_key}'",
            )
            if publish_point_mask_path is None:
                publish_point_mask = np.empty((0,), dtype=bool)
            else:
                publish_point_mask = graph_io.load_graph_array(
                    submap_dir / str(publish_point_mask_path),
                    dtype=bool,
                    ndim=1,
                )

            frame_point_ids_json = graph_io.require_graph_key(
                submap_info,
                "frame_point_ids",
                f"Submap '{submap_key}'",
            )
            if not isinstance(frame_point_ids_json, dict) or not frame_point_ids_json:
                raise ValueError(f"Submap '{submap_key}' has no frame_point_ids.")
            frame_point_ids = {
                frame_key: graph_io.load_graph_array(
                    submap_dir / str(rel_path),
                    dtype=np.int64,
                    ndim=2,
                )
                for frame_key, rel_path in sorted(frame_point_ids_json.items())
            }

            loaded_submaps[submap_key] = SubmapRecord(
                anchor_key=anchor_key,
                frame_keys=[str(v) for v in frame_keys_json],
                local_points=local_points,
                colors=colors,
                frame_point_ids=frame_point_ids,
                scale=scale,
                unique_point_ids=unique_point_ids,
                publish_point_mask=publish_point_mask,
                publish_downsample_ratio=publish_downsample_ratio,
            )

        return loaded_submaps


    def _load_graph_frame_inputs(self, graph_dir: Path, manifest: Dict[str, object]) -> None:
        graph = self.covisibility_graph
        graph._frame_image_dict.clear()
        graph._frame_intrinsics_dict.clear()
        graph._frame_conf_dict.clear()
        graph._frame_matcher_feature_dict.clear()
        graph._frame_matcher_feature_storage_device_dict.clear()

        frames_json = graph_io.require_graph_key(manifest, "frames", "Previous graph manifest")
        if not isinstance(frames_json, dict):
            raise ValueError("Previous graph frames manifest must be a JSON object.")

        frames_dir = graph_dir / "frames"
        for frame_key, frame_info in sorted(frames_json.items()):
            if not isinstance(frame_info, dict):
                raise ValueError(f"Invalid frame manifest entry for {frame_key}: {frame_info!r}")
            frame_dir = frames_dir / frame_key
            image_path = graph_io.require_graph_key(frame_info, "image", f"Frame '{frame_key}'")
            intrinsics_path = graph_io.require_graph_key(frame_info, "intrinsics", f"Frame '{frame_key}'")
            confidence_path = graph_io.require_graph_key(frame_info, "confidence", f"Frame '{frame_key}'")
            graph._frame_image_dict[frame_key] = graph_io.load_graph_array(
                frame_dir / str(image_path),
                dtype=np.uint8,
                ndim=2,
            )
            graph._frame_intrinsics_dict[frame_key] = graph_io.load_graph_array(
                frame_dir / str(intrinsics_path),
                dtype=np.float32,
                ndim=2,
                shape_tail=(3,),
            )
            graph._frame_conf_dict[frame_key] = graph_io.load_graph_array(
                frame_dir / str(confidence_path),
                dtype=np.float32,
                ndim=2,
            )


    def _load_graph_matches(self, graph_dir: Path, manifest: Dict[str, object]) -> None:
        graph = self.covisibility_graph
        graph._frame_match_result_dict.clear()
        graph._frame_match_count_dict.clear()

        matches_json = graph_io.require_graph_key(manifest, "matches", "Previous graph manifest")
        if not isinstance(matches_json, dict):
            raise ValueError("Previous graph matches manifest must be a JSON object.")
        frame_pair_matches = graph_io.require_graph_key(
            matches_json,
            "frame_pair_matches",
            "Previous graph matches manifest",
        )
        if not isinstance(frame_pair_matches, dict):
            raise ValueError("Previous graph frame_pair_matches must be a JSON object.")

        matches_dir = graph_dir / "matches"
        for _, match_info in sorted(frame_pair_matches.items()):
            if not isinstance(match_info, dict):
                raise ValueError(f"Invalid match manifest entry: {match_info!r}")
            frame_key_a = str(graph_io.require_graph_key(match_info, "frame_key_a", "Match manifest entry"))
            frame_key_b = str(graph_io.require_graph_key(match_info, "frame_key_b", "Match manifest entry"))
            match_path = matches_dir / str(graph_io.require_graph_key(match_info, "file", "Match manifest entry"))
            if not match_path.exists():
                raise FileNotFoundError(f"Missing saved graph match file: {match_path}")

            with np.load(match_path, allow_pickle=False) as match_npz:
                matched_points0 = np.ascontiguousarray(
                    np.asarray(match_npz["matched_points0"], dtype=np.float32)
                )
                matched_points1 = np.ascontiguousarray(
                    np.asarray(match_npz["matched_points1"], dtype=np.float32)
                )
                num_raw_matches = int(np.asarray(match_npz["num_raw_matches"]).item())
                num_inlier_matches = int(np.asarray(match_npz["num_inlier_matches"]).item())

            if frame_key_a <= frame_key_b:
                pair_key = (frame_key_a, frame_key_b)
                cached_points0 = matched_points0
                cached_points1 = matched_points1
            else:
                pair_key = (frame_key_b, frame_key_a)
                cached_points0 = matched_points1
                cached_points1 = matched_points0

            graph._frame_match_result_dict[pair_key] = {
                "num_raw_matches": num_raw_matches,
                "num_inlier_matches": num_inlier_matches,
                "matched_points0": cached_points0,
                "matched_points1": cached_points1,
            }
            graph._frame_match_count_dict[pair_key] = num_inlier_matches


    def _load_graph_topology(self, manifest: Dict[str, object]) -> None:
        graph_json = graph_io.require_graph_key(manifest, "graph", "Previous graph manifest")
        if not isinstance(graph_json, dict):
            raise ValueError("Previous graph topology must be a JSON object.")

        submap_frame_keys_json = graph_io.require_graph_key(graph_json, "submap_frame_keys", "Previous graph topology")
        submap_registration_index_json = graph_io.require_graph_key(
            graph_json,
            "submap_registration_index",
            "Previous graph topology",
        )
        frame_to_submaps_json = graph_io.require_graph_key(graph_json, "frame_to_submaps", "Previous graph topology")
        frame_covisibility_graph_json = graph_io.require_graph_key(
            graph_json,
            "frame_covisibility_graph",
            "Previous graph topology",
        )
        submap_covisibility_graph_json = graph_io.require_graph_key(
            graph_json,
            "submap_covisibility_graph",
            "Previous graph topology",
        )
        submap_covisibility_frame_pairs_json = graph_io.require_graph_key(
            graph_json,
            "submap_covisibility_frame_pairs",
            "Previous graph topology",
        )
        for key, value in (
            ("submap_frame_keys", submap_frame_keys_json),
            ("submap_registration_index", submap_registration_index_json),
            ("frame_to_submaps", frame_to_submaps_json),
            ("frame_covisibility_graph", frame_covisibility_graph_json),
            ("submap_covisibility_graph", submap_covisibility_graph_json),
            ("submap_covisibility_frame_pairs", submap_covisibility_frame_pairs_json),
        ):
            if not isinstance(value, dict):
                raise ValueError(f"Previous graph topology '{key}' must be a JSON object.")

        graph = self.covisibility_graph
        graph.submap_frame_keys_dict = {
            str(submap_key): [str(frame_key) for frame_key in frame_keys]
            for submap_key, frame_keys in submap_frame_keys_json.items()
        }
        graph._submap_registration_index = {
            str(submap_key): int(registration_idx)
            for submap_key, registration_idx in submap_registration_index_json.items()
        }
        graph.frame_to_submaps_dict = {
            str(frame_key): {str(submap_key) for submap_key in submap_keys}
            for frame_key, submap_keys in frame_to_submaps_json.items()
        }
        graph.frame_covisibility_graph = {
            str(frame_key): {str(neighbor_key) for neighbor_key in neighbor_keys}
            for frame_key, neighbor_keys in frame_covisibility_graph_json.items()
        }
        graph.submap_covisibility_graph = {
            str(submap_key): {str(neighbor_key) for neighbor_key in neighbor_keys}
            for submap_key, neighbor_keys in submap_covisibility_graph_json.items()
        }
        graph.submap_covisibility_frame_pairs = {}
        for pair_key, frame_pairs in submap_covisibility_frame_pairs_json.items():
            submap_key_a, submap_key_b = graph_io.split_graph_pair_key(str(pair_key))
            graph.submap_covisibility_frame_pairs[(submap_key_a, submap_key_b)] = [
                (str(frame_key_a), str(frame_key_b))
                for frame_key_a, frame_key_b in frame_pairs
            ]


    def _load_graph_ph_to_ref(self, manifest: Dict[str, object]) -> Dict[str, Tuple[MappingTimestamp, MappingPose]]:
        ph_to_ref_json = graph_io.require_graph_key(manifest, "ph_to_ref", "Previous graph manifest")
        if not isinstance(ph_to_ref_json, dict):
            raise ValueError("Previous graph ph_to_ref must be a JSON object.")

        loaded_ph_to_ref: Dict[str, Tuple[MappingTimestamp, MappingPose]] = {}
        for ph_key, ph_info in sorted(ph_to_ref_json.items()):
            if not isinstance(ph_info, dict):
                raise ValueError(f"Invalid ph_to_ref entry for {ph_key}: {ph_info!r}")
            ref_key = str(graph_io.require_graph_key(ph_info, "ref_key", f"ph_to_ref entry '{ph_key}'"))
            ph_to_ref_pose_json = graph_io.require_graph_key(
                ph_info,
                "ph_to_ref_pose",
                f"ph_to_ref entry '{ph_key}'",
            )
            loaded_ph_to_ref[str(ph_key)] = (
                timestamp_key_to_timestamp(ref_key),
                graph_io.graph_json_to_pose(ph_to_ref_pose_json),
            )
        return loaded_ph_to_ref


    def _pose_from_current_trajectory_for_loaded_frame(
        self,
        frame_key: str,
        loaded_ph_to_ref: Dict[str, Tuple[MappingTimestamp, MappingPose]],
    ) -> MappingPose:
        ph_to_ref_entry = loaded_ph_to_ref.get(frame_key)
        if ph_to_ref_entry is None:
            raise KeyError(f"Previous graph frame '{frame_key}' is missing ph_to_ref metadata.")
        ref_time, ph_to_ref_pose = ph_to_ref_entry
        ref_key = f"{ref_time.sec:010d}_{ref_time.nsec:09d}"
        if ref_key not in self.in_ref_poses_dict:
            raise KeyError(
                f"Previous graph frame '{frame_key}' references missing trajectory pose "
                f"'{ref_key}' in ph_to_ref metadata."
            )
        t_world_ref = self.transforms.pose_to_matrix(self.in_ref_poses_dict[ref_key])
        t_ref_ph = self.transforms.pose_to_matrix(ph_to_ref_pose)
        return self.transforms.matrix_to_pose(t_world_ref @ t_ref_ph)


    def _previous_graph_frame_keys(
        self,
        manifest: Dict[str, object],
        loaded_submaps: Dict[str, SubmapRecord],
    ) -> Set[str]:
        frame_keys: Set[str] = set()
        frames_json = graph_io.require_graph_key(manifest, "frames", "Previous graph manifest")
        if not isinstance(frames_json, dict):
            raise ValueError("Previous graph frames manifest must be a JSON object.")
        frame_keys.update(str(key) for key in frames_json.keys())

        graph_json = graph_io.require_graph_key(manifest, "graph", "Previous graph manifest")
        if not isinstance(graph_json, dict):
            raise ValueError("Previous graph topology must be a JSON object.")
        frame_to_submaps = graph_io.require_graph_key(graph_json, "frame_to_submaps", "Previous graph topology")
        if not isinstance(frame_to_submaps, dict):
            raise ValueError("Previous graph topology frame_to_submaps must be a JSON object.")
        frame_keys.update(str(key) for key in frame_to_submaps.keys())

        for submap in loaded_submaps.values():
            frame_keys.update(str(frame_key) for frame_key in submap.frame_keys)
            frame_keys.update(str(frame_key) for frame_key in submap.frame_point_ids.keys())
        return frame_keys


    def _load_prev_graph(self, prev_slam_folder: Union[str, Path]) -> Path:
        graph_dir = self._resolve_prev_graph_dir(prev_slam_folder)
        manifest = graph_io.load_graph_manifest(graph_dir)
        loaded_submaps = self._load_graph_submaps(graph_dir, manifest)
        loaded_ph_to_ref = self._load_graph_ph_to_ref(manifest)
        previous_frame_keys = self._previous_graph_frame_keys(manifest, loaded_submaps)
        self.previous_session_frame_keys = set(previous_frame_keys)

        for frame_key in sorted(previous_frame_keys):
            ph_to_ref_entry = loaded_ph_to_ref.get(frame_key)
            if ph_to_ref_entry is None:
                raise KeyError(f"Previous graph frame '{frame_key}' is missing ph_to_ref metadata.")
            ref_time = ph_to_ref_entry[0]
            ref_key = f"{ref_time.sec:010d}_{ref_time.nsec:09d}"
            if ref_key not in self.in_ref_poses_dict:
                raise KeyError(
                    f"Previous graph frame '{frame_key}' references missing trajectory pose "
                    f"'{ref_key}' in ph_to_ref metadata."
                )

        duplicate_submap_keys = sorted(set(self.submaps.keys()) & set(loaded_submaps.keys()))
        if duplicate_submap_keys:
            raise ValueError(
                "Previous graph submap keys already exist in current map: "
                f"{duplicate_submap_keys[:10]}{'...' if len(duplicate_submap_keys) > 10 else ''}"
            )

        self.submaps.update(loaded_submaps)
        self.ph_to_ref_dict.update(loaded_ph_to_ref)
        for frame_key in sorted(previous_frame_keys):
            pose = self._pose_from_current_trajectory_for_loaded_frame(frame_key, loaded_ph_to_ref)
            self.out_ph_poses_dict[frame_key] = pose

        self._load_graph_topology(manifest)
        self._load_graph_frame_inputs(graph_dir, manifest)
        self._load_graph_matches(graph_dir, manifest)

        graph = self.covisibility_graph
        graph._frame_positions_dict.clear()
        graph._frame_view_dirs_dict.clear()
        graph._clear_spatial_index()

        missing_frame_input_keys = sorted(
            frame_key
            for frame_key in previous_frame_keys
            if (
                frame_key not in graph._frame_image_dict
                or frame_key not in graph._frame_intrinsics_dict
                or frame_key not in graph._frame_conf_dict
            )
        )
        if missing_frame_input_keys:
            raise KeyError(
                "Previous graph frames are missing cached image/intrinsics/confidence inputs: "
                f"{missing_frame_input_keys[:10]}{'...' if len(missing_frame_input_keys) > 10 else ''}"
            )

        missing_frame_topology_keys = sorted(
            frame_key
            for frame_key in previous_frame_keys
            if frame_key not in graph.frame_covisibility_graph or frame_key not in graph.frame_to_submaps_dict
        )
        if missing_frame_topology_keys:
            raise KeyError(
                "Previous graph frames are missing topology entries: "
                f"{missing_frame_topology_keys[:10]}{'...' if len(missing_frame_topology_keys) > 10 else ''}"
            )

        missing_submap_topology_keys = sorted(
            submap_key
            for submap_key in loaded_submaps.keys()
            if (
                submap_key not in graph.submap_frame_keys_dict
                or submap_key not in graph.submap_covisibility_graph
                or submap_key not in graph._submap_registration_index
            )
        )
        if missing_submap_topology_keys:
            raise KeyError(
                "Previous graph submaps are missing topology entries: "
                f"{missing_submap_topology_keys[:10]}{'...' if len(missing_submap_topology_keys) > 10 else ''}"
            )

        for frame_key in sorted(previous_frame_keys):
            graph._cache_frame_pose_features(frame_key, self.out_ph_poses_dict)

        print(
            "Loaded previous global optimization graph: "
            f"path={graph_dir}, "
            f"submaps={len(loaded_submaps)}, "
            f"frames={len(previous_frame_keys)}, "
            f"cached_matches={len(graph._frame_match_result_dict)}"
        )
        return graph_dir


    def _register_submap_frames(self, submap_key: str, frame_keys: Sequence[str]) -> None:
        summary = self.covisibility_graph.register_submap_frames(
            submap_key=submap_key,
            frame_keys=frame_keys,
            pose_dict=self.out_ph_poses_dict,
        )
        print(
            "Frame covisibility updated: "
            f"submap={summary.submap_key}, "
            f"frames={summary.num_frames}, "
            f"new_frame_edges={summary.new_frame_edges}, "
            f"cross_submap_neighbors={summary.cross_submap_neighbors}"
        )


    def _rebuild_frame_covisibility_graph(self) -> None:
        summary = self.covisibility_graph.rebuild(self.out_ph_poses_dict)
        print(
            "Frame covisibility rebuilt: "
            f"frames={summary.num_frames}, "
            f"frame_edges={summary.frame_edges}, "
            f"cross_submap_links={summary.cross_submap_links}"
        )


    def _refresh_frame_covisibility_pose_features(self) -> None:
        summary = self.covisibility_graph.refresh_pose_features(self.out_ph_poses_dict)
        print(
            "Frame covisibility pose features refreshed: "
            f"frames={summary.num_frames}, "
            f"frame_edges={summary.frame_edges}, "
            f"cross_submap_links={summary.cross_submap_links}"
        )


    def _collect_global_covisibility_scale_links(
        self,
        selected_submap_keys: Optional[Set[str]] = None,
        prefetch_matcher_features: bool = False,
    ) -> Tuple[Dict[Tuple[str, str], List[Tuple[str, str]]], Dict[Tuple[str, str], Dict[str, np.ndarray]]]:
        if selected_submap_keys is None:
            selected_submap_keys = set(self.submaps.keys())
        covisible_frame_pairs = self.covisibility_graph.get_cross_submap_covisible_frame_pairs(
            selected_submap_keys=selected_submap_keys,
        )
        if prefetch_matcher_features:
            frame_keys_to_prefetch = {
                frame_key
                for frame_pairs in covisible_frame_pairs.values()
                for frame_pair in frame_pairs
                for frame_key in frame_pair
            }
            for frame_key in sorted(frame_keys_to_prefetch):
                self.covisibility_graph._set_frame_matcher_feature_storage_device(
                    frame_key,
                    self.covisibility_graph.matcher_device,
                )

        frame_pair_match_dict: Dict[Tuple[str, str], Dict[str, np.ndarray]] = {}
        try:
            for frame_pairs in covisible_frame_pairs.values():
                for frame_key_a, frame_key_b in frame_pairs:
                    frame_pair_key = (frame_key_a, frame_key_b)
                    if frame_pair_key in frame_pair_match_dict:
                        continue
                    frame_pair_match_dict[frame_pair_key] = self.covisibility_graph.get_frame_pair_match_result(
                        frame_key_a,
                        frame_key_b,
                    )
        finally:
            if prefetch_matcher_features:
                self.covisibility_graph._rebalance_matcher_feature_cache_storage()
        return covisible_frame_pairs, frame_pair_match_dict


    def _get_ph_poses_from_ref(self):
        in_ph_poses_dict = {}
        for ph_ts_key in self.ph_to_ref_dict:
            ref_ts = self.ph_to_ref_dict[ph_ts_key][0]
            ref_ts_key = f"{ref_ts.sec:010d}_{ref_ts.nsec:09d}"

            ph_to_ref_pose = self.ph_to_ref_dict[ph_ts_key][1]
            T_ref_ph = self.transforms.pose_to_matrix(ph_to_ref_pose)

            if ref_ts_key not in self.in_ref_poses_dict:
                raise KeyError(
                    f"Missing reference pose '{ref_ts_key}' while projecting pinhole poses from reference poses."
                )
            T_world_ref = self.transforms.pose_to_matrix(self.in_ref_poses_dict[ref_ts_key])
            T_world_ph = T_world_ref @ T_ref_ph
            in_ph_poses_dict[ph_ts_key] = self.transforms.matrix_to_pose(T_world_ph)
        return in_ph_poses_dict


    def _load_traj_snapshot(self, traj_timestamp: str) -> Dict[str, MappingPose]:
        if self.slam_bag_data is None:
            raise RuntimeError("Trajectory snapshots are only loaded from the input ROS 2 bag")
        pose_dict = self.slam_bag_data.get_trajectory_snapshot(traj_timestamp)
        if self.is_mono and self.fixed_mono_trajectory_scale is not None:
            return self._scale_pose_dict(pose_dict, self.fixed_mono_trajectory_scale)
        return pose_dict


    @staticmethod
    def _scale_pose(pose: MappingPose, scale: float) -> MappingPose:
        return MappingPose(
            [float(coord) * scale for coord in pose.pos],
            [float(coord) for coord in pose.quat],
        )


    @classmethod
    def _scale_pose_dict(
        cls,
        pose_dict: Dict[str, MappingPose],
        scale: float,
    ) -> Dict[str, MappingPose]:
        return {
            ts_key: cls._scale_pose(pose, scale)
            for ts_key, pose in pose_dict.items()
        }


    def _ensure_fixed_mono_trajectory_scale(self, batch_ref_timestamps: Sequence[str]) -> None:
        if not self.is_mono or self.fixed_mono_trajectory_scale is not None:
            return
        if len(batch_ref_timestamps) < 2:
            raise ValueError(
                "At least two reference poses are required to fix monocular trajectory scale."
            )

        adjacent_translations = []
        for prev_ts, curr_ts in zip(batch_ref_timestamps[:-1], batch_ref_timestamps[1:]):
            if prev_ts not in self.in_ref_poses_dict:
                raise KeyError(f"Missing reference pose for timestamp {prev_ts}")
            if curr_ts not in self.in_ref_poses_dict:
                raise KeyError(f"Missing reference pose for timestamp {curr_ts}")

            prev_pos = np.asarray(self.in_ref_poses_dict[prev_ts].pos, dtype=float)
            curr_pos = np.asarray(self.in_ref_poses_dict[curr_ts].pos, dtype=float)
            adjacent_translations.append(float(np.linalg.norm(curr_pos - prev_pos)))

        mean_translation = float(np.mean(adjacent_translations))
        if not math.isfinite(mean_translation) or mean_translation <= 0.0:
            raise ValueError(
                "Cannot fix monocular trajectory scale from the first input batch because "
                f"mean adjacent translation is {mean_translation}."
            )

        target_mean_translation = 0.5
        self.fixed_mono_trajectory_scale = target_mean_translation / mean_translation
        self.in_ref_poses_dict = self._scale_pose_dict(
            self.in_ref_poses_dict,
            self.fixed_mono_trajectory_scale,
        )
        self.odom_ref_poses_dict = self._scale_pose_dict(
            self.odom_ref_poses_dict,
            self.fixed_mono_trajectory_scale,
        )
        print(
            "Fixed monocular trajectory scale: "
            f"{self.fixed_mono_trajectory_scale:.9f} "
            f"(first batch mean adjacent translation={mean_translation:.9f}, "
            f"target={target_mean_translation:.9f})"
        )


    def _select_traj_timestamp_for_batch(self, batch_ref_timestamps: List[str]) -> Optional[str]:
        if not self.ref_traj_snapshot_timestamps:
            raise ValueError("ref_traj_snapshot_timestamps is empty; cannot select a trajectory snapshot")
        if not batch_ref_timestamps:
            raise ValueError("batch_ref_timestamps is empty; cannot select a trajectory snapshot")

        largest_selected_ts = max(batch_ref_timestamps)
        idx = bisect_right(self.ref_traj_snapshot_timestamps, largest_selected_ts) - 1
        if idx >= 0 and self.ref_traj_snapshot_timestamps[idx] == largest_selected_ts:
            return self.ref_traj_snapshot_timestamps[idx]

        idx += 1
        if idx >= len(self.ref_traj_snapshot_timestamps):
            raise ValueError(
                "No trajectory snapshot exists at or after the batch timestamps: "
                f"largest_batch_ts={largest_selected_ts}, "
                f"last_traj_snapshot={self.ref_traj_snapshot_timestamps[-1]}"
            )
        return self.ref_traj_snapshot_timestamps[idx]


    def _refresh_slam_trajectory_for_batch(self, batch_ref_timestamps: List[str]) -> bool:
        traj_timestamp = self._select_traj_timestamp_for_batch(batch_ref_timestamps)
        if traj_timestamp is None:
            return False

        new_ref_poses_dict = self._load_traj_snapshot(traj_timestamp)
        poses_changed = not pose_dicts_equal(
            self.in_ref_poses_dict,
            new_ref_poses_dict,
        )

        missing_batch_timestamps = [ts for ts in batch_ref_timestamps if ts not in new_ref_poses_dict]
        if missing_batch_timestamps:
            raise KeyError(
                "Selected trajectory snapshot is missing batch timestamps: "
                f"snapshot={traj_timestamp}, missing={missing_batch_timestamps[:10]}"
                f"{'...' if len(missing_batch_timestamps) > 10 else ''}"
            )

        self.current_traj_timestamp = traj_timestamp
        self.in_ref_poses_dict = new_ref_poses_dict

        if not poses_changed:
            return False

        if not self.out_ph_poses_dict:
            return False

        in_ph_poses_dict = self._get_ph_poses_from_ref()
        print("\033[33mUpdate map with loop corrected trajectory\033[0m")
        print(f"\033[33mLoaded trajectory snapshot: {traj_timestamp}\033[0m")
        self.out_ph_poses_dict = {
            k: in_ph_poses_dict[k] if k in in_ph_poses_dict else (_ for _ in ()).throw(KeyError(k))
            for k in self.out_ph_poses_dict
        }
        self._refresh_frame_covisibility_pose_features()
        self._run_submap_scale_optimization(
            latest_n_submaps=None,
            max_points_per_overlap_frame=100,
            log_prefix="loop-closure global",
        )
        return True


    def _validate_submap_window_config(self) -> None:
        self.num_submaps_per_batch, self.submap_ref_pose_stride = keyframe_selection.validate_submap_window_config(
            batch_size=self.num_ref_poses_per_batch,
            submap_size=self.num_ref_poses_per_submap,
            overlap=self.overlap_ref_views,
        )


    def _get_submap_ref_pose_windows(self, num_ref_poses_in_batch: int) -> List[Tuple[int, int]]:
        return keyframe_selection.get_submap_ref_pose_windows(
            num_ref_poses_in_batch=num_ref_poses_in_batch,
            num_ref_poses_per_batch=self.num_ref_poses_per_batch,
            num_submaps_per_batch=self.num_submaps_per_batch,
            submap_ref_pose_stride=self.submap_ref_pose_stride,
            num_ref_poses_per_submap=self.num_ref_poses_per_submap,
        )


    def _slice_prediction_batch(self, predictions, view_start: int, view_end: int):
        if view_start < 0 or view_end < view_start:
            raise ValueError(f"Invalid prediction slice: start={view_start}, end={view_end}.")

        sliced_fields = {}
        for attr_name in ("processed_images", "depth", "conf", "extrinsics", "intrinsics", "mask"):
            if not hasattr(predictions, attr_name):
                continue
            attr_value = getattr(predictions, attr_name)
            if attr_value is None or not hasattr(attr_value, "shape") or len(attr_value.shape) == 0:
                sliced_fields[attr_name] = attr_value
            else:
                sliced_fields[attr_name] = attr_value[view_start:view_end]

        if not sliced_fields:
            raise ValueError("Prediction object has no sliceable batch fields.")

        return SimpleNamespace(**sliced_fields)


    def _overwrite_prediction_camera_params_from_inputs(self, predictions, ph_view_poses_sub):
        num_views = len(ph_view_poses_sub)
        if num_views == 0:
            raise ValueError("Cannot overwrite prediction camera parameters for an empty view batch.")

        if hasattr(predictions, "depth") and predictions.depth is not None:
            prediction_batch_size = int(np.asarray(predictions.depth).shape[0])
            if prediction_batch_size != num_views:
                raise ValueError(
                    "Prediction batch size does not match input view count: "
                    f"predictions={prediction_batch_size}, inputs={num_views}."
                )

        processed_images = np.asarray(predictions.processed_images)
        if processed_images.ndim < 4:
            raise ValueError(
                "predictions.processed_images must have shape [N, H, W, C] "
                f"to scale intrinsics, got {processed_images.shape}."
            )
        if int(processed_images.shape[0]) != num_views:
            raise ValueError(
                "Processed image batch size does not match input view count: "
                f"processed_images={processed_images.shape[0]}, inputs={num_views}."
            )

        processed_h, processed_w = processed_images.shape[1:3]
        input_w, input_h = self.pinhole_camera.resolution
        sx = float(processed_w) / float(input_w)
        sy = float(processed_h) / float(input_h)
        input_intrinsics_one = self.pinhole_camera.intrinsics_mtx.astype(np.float32).copy()
        input_intrinsics_one[0, :] *= sx
        input_intrinsics_one[1, :] *= sy
        input_intrinsics = np.stack([input_intrinsics_one] * num_views, axis=0)
        input_extrinsics = []
        for _, input_pose in ph_view_poses_sub:
            t_world_view = self.transforms.pose_to_matrix(input_pose)
            input_extrinsics.append(np.linalg.inv(t_world_view)[:3, :4])
        input_extrinsics = np.stack(input_extrinsics, axis=0).astype(np.float32)

        predictions.intrinsics = input_intrinsics
        predictions.extrinsics = input_extrinsics
        return predictions


    def _update_max_distance_for_batch(self, batch_ref_timestamps: Sequence[str]) -> None:
        configured_max_distance = self.config.get("max_distance")
        if configured_max_distance != 0 and not self.is_mono:
            self.max_distance = configured_max_distance
            return

        if len(batch_ref_timestamps) < 2:
            raise ValueError(
                "At least two reference poses are required to compute dynamic max_distance."
            )

        adjacent_translations = []
        for prev_ts, curr_ts in zip(batch_ref_timestamps[:-1], batch_ref_timestamps[1:]):
            if prev_ts not in self.in_ref_poses_dict:
                raise KeyError(f"Missing reference pose for timestamp {prev_ts}")
            if curr_ts not in self.in_ref_poses_dict:
                raise KeyError(f"Missing reference pose for timestamp {curr_ts}")

            prev_pos = np.asarray(self.in_ref_poses_dict[prev_ts].pos, dtype=float)
            curr_pos = np.asarray(self.in_ref_poses_dict[curr_ts].pos, dtype=float)
            adjacent_translations.append(float(np.linalg.norm(curr_pos - prev_pos)))

        median_translation = float(np.median(adjacent_translations))
        if not self.is_mono:
            median_translation = max(median_translation, self.kf_distance)
            self.max_distance = min(median_translation * 20.0, 50)
        else:
            self.max_distance = median_translation * 20.0
        print(
            "Dynamic max_distance for batch: "
            f"{self.max_distance:.6f} "
            f"(median adjacent translation={median_translation:.6f})"
        )


    def _run_submap_scale_optimization(
        self,
        latest_n_submaps: Optional[int],
        max_points_per_overlap_frame: int,
        log_prefix: str,
    ) -> None:
        if not self.submap_scale_opt:
            return
        if len(self.submaps) < 2:
            return

        submap_keys = sorted(self.submaps.keys())
        if latest_n_submaps is None:
            selected_submap_keys = set(submap_keys)
        else:
            selected_submap_keys = set(submap_keys[-int(latest_n_submaps) :])
        covisible_frame_pairs, frame_pair_match_dict = self._collect_global_covisibility_scale_links(
            selected_submap_keys=selected_submap_keys,
            prefetch_matcher_features=latest_n_submaps is not None,
        )
        submap_scales_dict = gtsam_optimization.optimize_submap_scales_gtsam(
            submaps=self.submaps,
            out_ph_poses_dict=self.out_ph_poses_dict,
            overlap_frames=self.overlap_ph_views,
            iters=30,
            max_points_per_overlap_frame=max_points_per_overlap_frame,
            robust_delta=0.1,
            reg_weight=0.01,
            normalize_mean=False,
            use_exp_param=True,
            random_seed=0,
            latest_n_submaps=latest_n_submaps,
            covisible_frame_pairs=covisible_frame_pairs,
            frame_pair_match_dict=frame_pair_match_dict,
        )
        for submap_key, scale_value in submap_scales_dict.items():
            if submap_key not in self.submaps:
                raise KeyError(f"Optimized scale returned for unknown submap '{submap_key}'.")
            self.submaps[submap_key].scale = float(scale_value)
        print(f"{log_prefix} optimized submap scales:")
        for submap_key in sorted(submap_scales_dict.keys()):
            print(f"  {submap_key}: {submap_scales_dict[submap_key]:.6f}")


    def _filter_prediction_confidence(self, predictions, conf_thresh_percentile: float):
        prediction_sky = getattr(predictions, "sky", None)
        if prediction_sky is not None:
            sky_mask = np.asarray(prediction_sky).astype(bool)
            predictions.conf = np.where(sky_mask, 0, predictions.conf)
            predictions.depth = np.where(sky_mask, 0, predictions.depth)
            print(
                f"Sky filtering removed {int(np.count_nonzero(sky_mask))} "
                "pixels using prediction sky mask."
            )
        else:
            finite_depth = predictions.depth[np.isfinite(predictions.depth)]
            if finite_depth.size > 0:
                sky_depth_threshold = np.percentile(finite_depth, 99)
                sky_mask = predictions.depth >= sky_depth_threshold
                predictions.conf = np.where(sky_mask, 0, predictions.conf)
                predictions.depth = np.where(sky_mask, 0, predictions.depth)
                print(
                    f"Sky filtering removed {int(np.count_nonzero(sky_mask))} "
                    f"pixels at p99 depth threshold {sky_depth_threshold:.6f}."
                )
            else:
                print("Warning: no finite depth values available for sky filtering.")

        finite_nonzero_depth = predictions.depth[
            np.isfinite(predictions.depth) & (predictions.depth > 0)
        ]
        depth_p50 = float(np.percentile(finite_nonzero_depth, 50))
        max_distance_threshold = max(float(self.max_distance), depth_p50)

        predictions.conf = np.where(predictions.depth > max_distance_threshold, 0, predictions.conf)
        # predictions.depth = np.where(predictions.depth > max_distance_threshold, 0, predictions.depth)

        conf_threshold = np.percentile(predictions.conf[predictions.conf != 0], conf_thresh_percentile)
        predictions.conf[predictions.conf <= conf_threshold] = 0.0
        # predictions.depth[predictions.conf < conf_threshold] = 0.0

        if self.model_name == "da" and self.fisheye_cameras:
            edge_masks = map_fusion.mask_depth_edge_batch(depth=predictions.depth, kernel_size=3, rtol=0.03)
            predictions.conf[edge_masks] = 0.0
            # predictions.depth[edge_masks] = 0.0

        return predictions


    def _is_open_space(self, predictions) -> bool:
        """
        Determine if the LAST frame is outdoor based on predictions.

        Checks:
        - The .sky attribute contains value 1 (if available), OR
        - The 50th percentile of depth values is greater than 20
        """
        i = -1  # last frame

        # Check sky
        prediction_sky = getattr(predictions, "sky", None)
        if prediction_sky is not None:
            sky_frame = np.asarray(prediction_sky[i]).astype(bool)
            if np.any(sky_frame):
                return True

        # Check depth
        if self.is_mono:
            return False

        depth_frame = predictions.depth[i]
        finite_depth = depth_frame[np.isfinite(depth_frame)]
        if finite_depth.size > 0:
            depth_percentile = np.percentile(finite_depth, 50)
            if depth_percentile > 20:
                return True

        return False


    def app_configure(self, config_path: Union[str, Path]):
        # Config the application
        config_path = Path(config_path).expanduser()
        self.config = self._load_runtime_config(config_path)
        self.config_filename = str(config_path)
        self.model_name = self.config.get("model_name")
        self.is_mono = bool(self.config.get("is_mono", False))
        if self.is_mono and self.prev_slam_folder is not None:
            raise ValueError(
                "Multi-session mode not supported for monocular only yet"
            )

        self.device = _require_cuda()

        # --- load fisheye config ---
        self.fisheye_cameras = load_fisheye_cameras_from_config(self.config)
        if self.fisheye_cameras:
            print(f"\n=== Fisheye Cameras Loaded From Config: {self.config_filename} ===")
            for cam_id, cam in self.fisheye_cameras.items():
                print(f"{cam_id}:")
                print(cam)
                print("")
        else:
            print("\n=== No Fisheye Cameras In Config; Input Images Assumed Pinhole ===")

        # --- load pinhole config ---
        missing_pinhole_fields = [
            field_name
            for field_name in ("pinhole_intrinsics", "pinhole_resolution")
            if field_name not in self.config
        ]
        if missing_pinhole_fields:
            raise KeyError(
                f"Config {self.config_filename} is missing required pinhole fields: "
                f"{missing_pinhole_fields}"
            )
        self.pinhole_camera = PinholeCamera(
            intrinsics=self.config["pinhole_intrinsics"],
            resolution=self.config["pinhole_resolution"],
        )
        
        print("\n=== Pinhole Camera Loaded ===")
        print(self.pinhole_camera)

        # --- load trajectories and image timestamps ---
        use_slam = bool(self.config.get("use_slam"))
        using_file_input = self.image_folder is not None or self.poses is not None
        if self.input_bag is None and not using_file_input:
            raise ValueError("Either --input_bag or both --image_folder and --poses must be provided.")
        if self.input_bag is not None and using_file_input:
            raise ValueError("--input_bag cannot be used together with --image_folder/--poses.")
        if using_file_input and (self.image_folder is None or self.poses is None):
            raise ValueError("--image_folder and --poses must be provided together.")
        if using_file_input and use_slam:
            raise ValueError("--image_folder/--poses input requires use_slam: false in the config.")

        slam_odometry_topic = (
            self.config.get("slam_odometry_topic", DEFAULT_ODOMETRY_TOPIC)
            if use_slam
            else None
        )
        slam_trajectory_topic = (
            self.config.get("slam_trajectory_topic", DEFAULT_TRAJECTORY_TOPIC)
            if use_slam
            else None
        )
        slam_final_trajectory_topic = self.config.get(
            "slam_final_trajectory_topic", DEFAULT_FINAL_TRAJECTORY_TOPIC
        )
        slam_image_topic = self.config.get(
            "slam_image_topic", DEFAULT_IMAGE_TOPIC
        )
        timestamp_tolerance_sec = float(
            self.config.get("image_pose_timestamp_tolerance_sec", 0.005)
        )
        timestamp_sync_tmp_root = Path(self.slam_folder).expanduser().parent / "tmp"
        self.timestamp_sync_tmp_root = timestamp_sync_tmp_root
        cleanup_synced_bag_temporary_data(timestamp_sync_tmp_root)
        input_bag_for_sync = self.input_bag
        if using_file_input:
            input_bag_for_sync = create_bag_from_image_folder_and_poses(
                image_folder=self.image_folder,
                poses_path=self.poses,
                image_topic=slam_image_topic,
                poses_topic=slam_final_trajectory_topic,
                tmp_root=timestamp_sync_tmp_root,
                frame_id=self.config.get("ros2_pointcloud_frame_id", "map"),
            )

        previous_image_timestamps_nsec: Set[int] = set()
        if self.prev_slam_folder is not None:
            current_session_start_nsec = read_current_session_start_image_timestamp_nsec(
                input_bag_for_sync,
                image_topic=slam_image_topic,
            )
            (
                previous_image_timestamps_nsec,
                previous_data_timestamps_nsec,
            ) = self._previous_graph_timestamps_nsec(self.prev_slam_folder)
            self._validate_previous_graph_before_current_session(
                previous_data_timestamps_nsec=previous_data_timestamps_nsec,
                current_session_start_nsec=current_session_start_nsec,
            )
            print(
                "Image/pose timestamp sync previous images: "
                f"timestamps={len(previous_image_timestamps_nsec)}"
            )

        sync_result = ensure_bag_image_pose_timestamps_synchronized(
            input_bag_for_sync,
            image_topic=slam_image_topic,
            trajectory_topic=slam_trajectory_topic,
            final_trajectory_topic=slam_final_trajectory_topic,
            odometry_topic=slam_odometry_topic,
            tolerance_sec=timestamp_tolerance_sec,
            tmp_root=timestamp_sync_tmp_root,
            extra_image_timestamps_nsec=previous_image_timestamps_nsec,
        )
        self.timestamp_sync_result = sync_result
        print(
            "Image/pose timestamp sync: "
            f"poses={sync_result.unique_pose_timestamps}, "
            f"exact={sync_result.exact_matches}, "
            f"rewritten={sync_result.rewritten_timestamps}, "
            f"skipped={sync_result.skipped_pose_timestamps}, "
            f"path_header_updates={sync_result.path_header_updates}, "
            f"max_delta_sec={sync_result.max_delta_sec:.9f}"
        )
        current_skipped_color = (
            ANSI_GREEN
            if sync_result.current_session_skipped_pose_timestamps == 0
            else ANSI_YELLOW
        )
        print(
            f"{current_skipped_color}"
            "Image/pose timestamp sync current-session skipped poses: "
            f"{sync_result.current_session_skipped_pose_timestamps}"
            f"{ANSI_RESET}"
        )
        if self.prev_slam_folder is not None:
            previous_skipped_color = (
                ANSI_GREEN
                if sync_result.previous_session_skipped_pose_timestamps == 0
                else ANSI_YELLOW
            )
            print(
                f"{previous_skipped_color}"
                "Image/pose timestamp sync previous-session skipped poses: "
                f"{sync_result.previous_session_skipped_pose_timestamps}"
                f"{ANSI_RESET}"
            )
        if sync_result.synchronized:
            print(f"Using synchronized input bag: {sync_result.bag_path}")
        self.input_bag = str(sync_result.bag_path)
        sync_start_timestamp_nsec = sync_result.current_session_start_image_timestamp_nsec
        if sync_start_timestamp_nsec is not None:
            self.session_start_time = timestamp_nsec_to_key(sync_start_timestamp_nsec)

        self.slam_bag_data = load_slam_bag(
            self.input_bag,
            use_slam=use_slam,
            trajectory_topic=slam_trajectory_topic,
            final_trajectory_topic=slam_final_trajectory_topic,
            odometry_topic=slam_odometry_topic,
            image_topic=slam_image_topic,
        )
        current_session_timestamps = set(self.slam_bag_data.compressed_images)
        if sync_start_timestamp_nsec is None:
            self.session_start_time = min(current_session_timestamps)
        self.ref_traj_snapshot_timestamps = self.slam_bag_data.trajectory_snapshot_timestamps
        if len(self.ref_traj_snapshot_timestamps) == 0:
            raise FileNotFoundError(f"No trajectory snapshots found in {self.slam_bag_data.bag_path}")

        self.odom_ref_poses_dict = self.slam_bag_data.odometry.copy()
        first_traj_timestamp = self.ref_traj_snapshot_timestamps[0]
        self.in_ref_poses_dict = self._load_traj_snapshot(first_traj_timestamp)
        self.current_traj_timestamp = first_traj_timestamp
        self.ref_timestamps = sorted(self.odom_ref_poses_dict.keys())
        self.ref_timestamps = [
            ts for ts in self.ref_timestamps
            if ts in current_session_timestamps
        ]

        if not self.ref_timestamps:
            raise ValueError(
                "No current-session image timestamps matched the loaded trajectory poses. "
                f"image_timestamps={len(current_session_timestamps)}, "
                f"trajectory_poses={len(self.odom_ref_poses_dict)}"
            )

        print(f"session start time: {self.session_start_time}, {self.ref_timestamps[0]}")

        # --- load mapping config ---
        configured_frame_scale_opt = bool(self.config.get("frame_scale_opt", False))
        configured_submap_scale_opt = bool(self.config.get("submap_scale_opt", False))
        configured_point_cloud_fusion = bool(self.config.get("point_cloud_fusion", False))
        self.frame_scale_opt = configured_frame_scale_opt
        self.submap_scale_opt = configured_submap_scale_opt
        self.point_cloud_fusion = configured_point_cloud_fusion and not self.is_mono

        self.vis_matcher = self.config.get("vis_matcher", "superpoint-lightglue")

        self.select_kf_on_time = self.config.get("select_kf_on_time")
        self.sec_skip = self.config.get("sec_skip")

        self.kf_distance = self.config.get("kf_distance") if not self.is_mono else 1e-3
        self.kf_angle_rad = np.deg2rad(self.config.get("kf_angle_deg")) if not self.is_mono else 1e-3
        self.max_distance = self.config.get("max_distance") if not self.is_mono else 0
        self.num_ref_poses_per_batch = self.config.get("num_ref_poses_per_batch")
        self.num_ref_poses_per_submap = self.config.get(
            "num_ref_poses_per_submap",
            self.num_ref_poses_per_batch,
        )
        self.overlap_ref_views = self.config.get("overlap_ref_views")
        self._validate_submap_window_config()
        self.covisibility_graph.configure(
            enabled=True,
            max_distance=math.inf if self.is_mono else 3.0,
            max_angle_deg=30.0,
            min_time_separation_sec=5.0 if self.config.get("use_slam") else 0.0,
            max_old_candidates_per_new_frame=3,
            max_old_edges_per_new_frame=1,
            matcher_name=self.vis_matcher,
            matcher_device="cuda",
            max_num_keypoints=1024,
            ransac_reproj_thresh=3.0,
            min_inlier_matches=30,
            recent_gpu_feature_cache_size=self.num_ref_poses_per_batch,
        )

        self.min_matches_for_edge_drop = self.config.get("min_matches_for_edge_drop", 0)
        self.publish_ros2_pointcloud = self.config.get("publish_ros2_pointcloud", False)
        self.publish_ros2_path = self.config.get("publish_ros2_path", False)
        self.publish_ros2_images = self.config.get("publish_ros2_images", False)
        self.ros2_pointcloud_downsample_ratio = float(
            self.config.get("ros2_pointcloud_downsample_ratio", 0.05)
        )
        self.ros2_pointcloud_frame_id = self.config.get("ros2_pointcloud_frame_id", "map")
        self.ros2_pointcloud_topic = self.config.get("ros2_pointcloud_topic", "/scarf_slam/clouds")
        self.ros2_path_topic = self.config.get("ros2_path_topic", "/scarf_slam/slam_poses")
        self.ros2_prev_path_topic = self.config.get("ros2_prev_path_topic", "/scarf_slam/slam_poses_prev")
        self.ros2_images_topic = self.config.get("ros2_images_topic", "/scarf_slam/images")
        if (
            self.publish_ros2_pointcloud
            and (
                not math.isfinite(self.ros2_pointcloud_downsample_ratio)
                or self.ros2_pointcloud_downsample_ratio < 0.0
                or self.ros2_pointcloud_downsample_ratio > 1.0
            )
        ):
            raise ValueError(
                "ros2_pointcloud_downsample_ratio must be between 0 and 1 when publish_ros2_pointcloud is enabled"
            )
        self._setup_ros2_publisher()
        self.recon_save_folder_name = f"{self.config.get("trajectory")}"
        if self.config.get("use_slam"):
            self.recon_save_folder_name = self.recon_save_folder_name + "_slam"

        # current implementation always assume using forward pinhole image.
        self.ph_views_per_ref_pose = [
            RotateParam(cam="cam0", yaw=0, pitch=0),
        ]
        self.ph_views_per_batch = [self.ph_views_per_ref_pose] * self.num_ref_poses_per_batch
        self.num_ph_views_per_ref_pose = len(self.ph_views_per_ref_pose)
        self.overlap_ph_views = self.overlap_ref_views * self.num_ph_views_per_ref_pose

        # --- load previous graph ---
        if self.prev_slam_folder is not None:
            print(f"Load previous session graph from {self.prev_slam_folder}")
            self._load_prev_graph(self.prev_slam_folder)
            header_timestamp = timestamp_key_to_timestamp(self.ref_timestamps[0])
            self._publish_loaded_previous_session(
                header_timestamp=header_timestamp,
            )

        # --- load model ---
        if self.model_name == "da":
            from depth_anything_3.api import DepthAnything3
            self.model = DepthAnything3.from_pretrained("depth-anything/DA3NESTED-GIANT-LARGE").to(device=self.device)
        else:
            raise ValueError(f"Invalid self.model_name: {self.model_name}")


    def do_processing_outer(self):
        counter = 0
        periodic_submap_opt_interval = 5
        next_periodic_submap_opt_count = periodic_submap_opt_interval
        indices_overlap = [0]

        def _run_periodic_submap_scale_optimization() -> None:
            nonlocal next_periodic_submap_opt_count
            while counter >= next_periodic_submap_opt_count:
                if self.submap_scale_opt and len(self.submaps) >= 2:
                    self._run_submap_scale_optimization(
                        latest_n_submaps=min(10, counter),
                        max_points_per_overlap_frame=200,
                        log_prefix="periodic",
                    )
                next_periodic_submap_opt_count += periodic_submap_opt_interval

        while True:
            indices = indices_overlap

            i = indices_overlap[-1] + 1
            while i < len(self.ref_timestamps) and len(indices) < self.num_ref_poses_per_batch:
                if self.select_kf_on_time:
                    last_ts = self.ref_timestamps[indices[-1]]
                    curr_ts = self.ref_timestamps[i]
                    if (
                        (timestamp_key_to_seconds(curr_ts) - timestamp_key_to_seconds(last_ts)) >= self.sec_skip
                        and keyframe_selection.motion_exceeds_threshold(
                            self.odom_ref_poses_dict[last_ts],
                            self.odom_ref_poses_dict[curr_ts],
                            self.kf_distance,
                            self.kf_angle_rad,
                        )
                    ):
                        indices.append(i)
                else:
                    # Determine new mapping frame using the reference pose (pose of fisheye cam0)
                    t_curr = self.ref_timestamps[i]

                    t_prev = self.ref_timestamps[indices[-1]]
                    if keyframe_selection.motion_exceeds_threshold(
                        self.odom_ref_poses_dict[t_prev],
                        self.odom_ref_poses_dict[t_curr],
                        self.kf_distance,
                        self.kf_angle_rad,
                    ):
                        indices.append(i)
                i += 1

            if not indices or len(indices) < self.num_ref_poses_per_batch:
                print(f"No enough poses left; exiting loop.")
                self._print_runtime_summary(counter)

                updated_traj = False
                if self.config.get("use_slam"):
                    updated_traj = self._refresh_slam_trajectory_for_batch([self.ref_timestamps[-1]])
                if self.submap_scale_opt and not updated_traj:
                    self._run_submap_scale_optimization(
                        latest_n_submaps=None,
                        max_points_per_overlap_frame=100,
                        log_prefix="loop-closure global",
                    )
                sec_str, nsec_str = self.ref_timestamps[-1].split("_", 1)
                header_timestamp = MappingTimestamp(int(sec_str), int(nsec_str))
                self._publish_sampled_global_pointcloud(header_timestamp=header_timestamp)
                self._publish_out_ph_poses_path(header_timestamp=header_timestamp)

                # self.covisibility_graph.plot_trajectory_covisibility(
                #     pose_dict=self.out_ph_poses_dict,
                #     block=True,
                # )
                if len(self.submaps) == 0:
                    raise RuntimeError("No submaps were created; cannot save the final map.")
                print(f"Save map...")
                suffix = (
                    f"_f{int(self.frame_scale_opt)}"
                    f"_s{int(self.submap_scale_opt)}"
                    f"_p{int(self.point_cloud_fusion)}"
                    f"_b{self.num_ref_poses_per_batch}"
                    f"_s{self.num_ref_poses_per_submap}"
                    f"_o{self.overlap_ref_views}"
                )
                recon_dir = Path(self.slam_folder) / "recon" / self.recon_save_folder_name
                recon_dir.mkdir(parents=True, exist_ok=True)
                self.save_out_poses_dict_to_csv(str(recon_dir / f"poses_{self.model_name}.csv"))
                self.save_out_poses_ts_to_csv(str(recon_dir / f"poses_{self.model_name}_ts.csv"))
                self.save_out_poses_dict_to_tum(str(recon_dir / f"poses_{self.model_name}.txt"))
                pts_global, colors_global = self._compose_global_pointcloud()
                self.save_global_pointcloud(pts_global, colors_global, suffix=suffix)
                self.save_per_frame_local_pointclouds(
                    Path(self.slam_folder)
                    / "recon"
                    / self.recon_save_folder_name
                    / f"pts_local{suffix}"
                )
                self.save_graph(
                    Path(self.slam_folder)
                    / "recon"
                    / self.recon_save_folder_name
                    / f"opt_graph{suffix}"
                )
                break
            
            ref_ts_sub = [self.ref_timestamps[j] for j in indices]

            # The current map update implementation is intentionally sequential
            # to ensure deterministic and reproducible results.
            # A multi-threaded implementation can be used for real-world deployment.
            if self.config.get("use_slam"):
                map_updated = self._refresh_slam_trajectory_for_batch(ref_ts_sub)
                if not map_updated:
                    # (Optional) Periodically invoke a larger submap optimization
                    _run_periodic_submap_scale_optimization()
            self._ensure_fixed_mono_trajectory_scale(ref_ts_sub)

            indices_overlap = indices[-self.overlap_ref_views :]
            
            inference_start_time = time.perf_counter()
            if self.model_name == "da":
                da_out = depthanything_insta.do_processing(
                    self, 
                    ref_ts_sub, 
                    self.in_ref_poses_dict,
                    self.ph_views_per_batch,
                    use_extrinsics=True,
                )
                predictions, ph_view_poses_sub, new_ph_to_ref_dict = da_out
            else:
                raise NotImplementedError
            # The predicted intrinsics and extrinsics are already overwritten in DA3;
            # this implementation is retained for potential use with other models.
            predictions = self._overwrite_prediction_camera_params_from_inputs(
                predictions,
                ph_view_poses_sub,
            )
            for k, v in new_ph_to_ref_dict.items():
                self.ph_to_ref_dict.setdefault(k, v)
            inference_duration = self._add_elapsed_time("model_inference_time", inference_start_time)

            self._update_max_distance_for_batch(ref_ts_sub)
            predictions = self._filter_prediction_confidence(predictions, conf_thresh_percentile=25)

            if (
                not self.is_mono
                and self._is_open_space(predictions)
                and self.kf_distance != 0
            ):
                self.kf_distance = self.config.get("kf_distance_large", 1.0)
                print(f"next batch may be open space: \033[92mTrue\033[0m")
            elif not self.is_mono and self.kf_distance != 0:
                self.kf_distance = self.config.get("kf_distance", 0.3)
                print(f"next batch may be open space: \033[91mFalse\033[0m")

            total_frame_scale_opt_duration = 0.0
            total_feature_match_duration = 0.0
            total_submap_scale_opt_duration = 0.0
            total_pts_fusion_duration = 0.0
            created_submap_keys: List[str] = []
            num_ref_poses_in_batch = len(ref_ts_sub)
            submap_windows = self._get_submap_ref_pose_windows(num_ref_poses_in_batch)
            for submap_ref_start, submap_ref_end in submap_windows:
                view_start = submap_ref_start * self.num_ph_views_per_ref_pose
                view_end = submap_ref_end * self.num_ph_views_per_ref_pose
                ph_view_poses_sub_slice = ph_view_poses_sub[view_start:view_end]
                predictions_slice = self._slice_prediction_batch(predictions, view_start, view_end)
                _, timing_info = self.optimize_submap(
                    ph_view_poses_sub_slice,
                    predictions_slice,
                )
                created_submap_keys.append(timing_info["ts_key"])
                total_frame_scale_opt_duration += timing_info["frame_scale_opt_duration"]
                total_feature_match_duration += timing_info["feature_match_duration"]
                total_submap_scale_opt_duration += timing_info["submap_scale_opt_duration"]
                total_pts_fusion_duration += timing_info["pts_fusion_duration"]

            timing_info = {
                "ts_key": ",".join(created_submap_keys),
                "num_created_submaps": len(created_submap_keys),
                "frame_scale_opt_duration": total_frame_scale_opt_duration,
                "feature_match_duration": total_feature_match_duration,
                "submap_scale_opt_duration": total_submap_scale_opt_duration,
                "pts_fusion_duration": total_pts_fusion_duration,
            }

            print(f"\033[32mTiming Summary [{timing_info['ts_key']}]\033[0m")
            self._print_timing_line("Model Inference Time", inference_duration, indent=0, color="\033[32m")
            self._print_timing_line("Feature Match Time", timing_info["feature_match_duration"], indent=0, color="\033[32m")
            self._print_timing_line("Frame Scale Optimization Time", timing_info["frame_scale_opt_duration"], indent=0, color="\033[32m")
            self._print_timing_line("Submap Scale Optimization Time", timing_info["submap_scale_opt_duration"], indent=0, color="\033[32m")
            self._print_timing_line("Points Fusion Time", timing_info["pts_fusion_duration"], indent=0, color="\033[32m")

            header_timestamp = ph_view_poses_sub[-1][0]
            self._publish_sampled_global_pointcloud(header_timestamp=header_timestamp)
            self._publish_out_ph_poses_path(header_timestamp=header_timestamp)
            self._publish_processed_images(predictions.processed_images, header_timestamp=header_timestamp)

            del predictions
            gc.collect()
            torch.cuda.empty_cache()

            counter += int(timing_info.get("num_created_submaps", 1))


    def optimize_submap(self, est_poses_sub, predictions):
        print("=== Optimize Submap ===")

        def _get_ts_key(est_poses_sub, index):
            this_sec, this_nsec = est_poses_sub[index][0].sec, est_poses_sub[index][0].nsec
            key = f"{this_sec:010d}_{this_nsec:09d}"
            return key

        # Frame scale optimization
        match_dict: Optional[Dict[str, object]] = None
        feature_match_duration = 0.0
        frame_scale_opt_duration = 0.0
        if self.frame_scale_opt:
            feature_match_start_time = time.perf_counter()
            match_dict = feature_matching.extract_feat_and_match_dl(
                predictions,
                max_prev=5,
                device=self.covisibility_graph.matcher_device,
                matcher_name=self.vis_matcher,
                rm_conf0_kpts=False,
                rm_conf0_mths=True,
                max_num_keypoints=self.covisibility_graph.max_num_keypoints,
                patch_match_limit_threshold=0,
                patch_size_divisor=20,
            )
            feature_match_duration = self._add_elapsed_time(
                "feature_match_time",
                feature_match_start_time,
            )
            frame_scale_opt_start_time = time.perf_counter()
            depth_scales = gtsam_optimization.optimize_frame_scales_gtsam(
                predictions,
                match_dict["matches"],
                match_dict["keypoints"],
                iters=30,
                min_matches_for_node_freeze=0,
                min_matches_for_edge_drop=self.min_matches_for_edge_drop,
                robust_delta=0.1,
                reg_weight=0.05,
                normalize_mean=False,
                use_exp_param=True,
            )
            frame_scale_opt_duration = self._add_elapsed_time(
                "frame_scale_opt_time",
                frame_scale_opt_start_time,
            )
            match_dict["depth_scales"] = depth_scales
            print("optimized depth scales:")
            for i in range(0, len(est_poses_sub)):
                ts_key_i = _get_ts_key(est_poses_sub, i)
                print(f"  {ts_key_i}: {depth_scales[i]:.6f}")
            predictions.depth = predictions.depth * depth_scales[:, None, None]

        poses_out = []

        imgs = predictions.processed_images.copy()  # uint8 [N,H,W,3]
        depths = predictions.depth.copy()           # float32 [N,H,W]
        confs = predictions.conf.copy()             # float32 [N,H,W]
        extrinsics = predictions.extrinsics         # [N,3,4]
        intrinsics = predictions.intrinsics         # [N,3,3]
        N, H, W = depths.shape
        fusion_device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        depths_t = torch.as_tensor(depths, dtype=torch.float32, device=fusion_device)
        intrinsics_batch = intrinsics if intrinsics.ndim == 3 else np.stack([intrinsics] * N)
        intrinsics_batch = np.asarray(intrinsics_batch, dtype=np.float32)
        intrinsics_t = torch.as_tensor(intrinsics_batch, dtype=torch.float32, device=fusion_device)
        extrinsics_t = torch.as_tensor(extrinsics, dtype=torch.float32, device=fusion_device)
        pts_world_batch_t = depth_to_world_points_vectorized(
            depths_t,
            intrinsics_t,
            extrinsics_t,
            device=fusion_device,
        )
        pts_world_batch = pts_world_batch_t.cpu().numpy()
        t_view_world_batch_t = torch.eye(4, dtype=torch.float32, device=fusion_device).repeat(N, 1, 1)
        t_view_world_batch_t[:, :3, :4] = extrinsics_t
        intrinsics_params = [
            [
                float(intrinsics_batch[i, 0, 0]),
                float(intrinsics_batch[i, 1, 1]),
                float(intrinsics_batch[i, 0, 2]),
                float(intrinsics_batch[i, 1, 2]),
            ]
            for i in range(N)
        ]

        submap_point_ids: Dict[str, np.ndarray] = {}
        submap_pts_world = np.empty((0, 4), dtype=np.float32)
        submap_colors = np.empty((0, 3), dtype=np.uint8)
        submap_point_count = 0

        def _ensure_submap_capacity(min_capacity: int) -> None:
            nonlocal submap_pts_world, submap_colors, submap_point_count
            current_capacity = submap_pts_world.shape[0]
            if current_capacity >= min_capacity:
                return

            new_capacity = max(
                min_capacity,
                1024 if current_capacity == 0 else current_capacity * 2,
            )
            while new_capacity < min_capacity:
                new_capacity *= 2

            new_pts = np.empty((new_capacity, 4), dtype=np.float32)
            new_colors = np.empty((new_capacity, 3), dtype=np.uint8)
            if submap_point_count > 0:
                new_pts[:submap_point_count] = submap_pts_world[:submap_point_count]
                new_colors[:submap_point_count] = submap_colors[:submap_point_count]
            submap_pts_world = new_pts
            submap_colors = new_colors

        def _append_submap_points(pts_unmatched: np.ndarray, rgb_unmatched: np.ndarray) -> np.ndarray:
            nonlocal submap_pts_world, submap_colors, submap_point_count
            n_new = int(pts_unmatched.shape[0])
            if n_new == 0:
                return np.empty((0,), dtype=np.int64)

            start_idx = submap_point_count
            end_idx = start_idx + n_new
            _ensure_submap_capacity(end_idx)
            submap_pts_world[start_idx:end_idx] = pts_unmatched
            submap_colors[start_idx:end_idx] = rgb_unmatched
            submap_point_count = end_idx
            return np.arange(start_idx, end_idx, dtype=np.int64)

        # Point cloud fusion
        fusion_start_time = time.perf_counter()
        for i in range(N):
            depth_i = depths[i]
            conf_i = confs[i]
            img_i = imgs[i]
            pts_world_i = pts_world_batch[i]
            pts_world_i_flat = pts_world_i.reshape(-1, 3)
            ts_key_i = _get_ts_key(est_poses_sub, i)

            pts_world_ids = np.full((H, W), -1, dtype=np.int64)

            if submap_point_count > 0:
                for project_idx in range(i - 1, max(-1, i - self.overlap_ph_views - 1), -1):
                    if project_idx == i:
                        continue

                    ts_key_old = _get_ts_key(est_poses_sub, project_idx)
                    if ts_key_old not in submap_point_ids:
                        continue

                    pts_world_ids_old = submap_point_ids[ts_key_old]
                    if self.point_cloud_fusion:
                        matching_unique = map_fusion.get_matching_torch(
                            pts_world_1=pts_world_batch_t[i],
                            mask_1=(pts_world_ids == -1),
                            depth_2=depths_t[project_idx],
                            T_view_world_2=t_view_world_batch_t[project_idx],
                            intrinsics_2=intrinsics_params[project_idx],
                            depth_thresh=0.05,
                            unique_mapping=True,
                        )
                    else:
                        matching_unique = np.full((H, W, 2), -1, dtype=np.int32)

                    matching_unique[pts_world_ids != -1] = -1
                    rows = matching_unique[:, :, 0]
                    cols = matching_unique[:, :, 1]
                    fill_mask = (rows >= 0) & (cols >= 0)
                    if np.any(fill_mask):
                        matched_rows = rows[fill_mask].astype(np.int64)
                        matched_cols = cols[fill_mask].astype(np.int64)
                        prev_ids = pts_world_ids_old[matched_rows, matched_cols]
                        existing_ids = pts_world_ids[pts_world_ids != -1]
                        unique_mask = ~np.isin(prev_ids, existing_ids)
                        fill_idx = np.flatnonzero(fill_mask)
                        pts_world_ids.flat[fill_idx[unique_mask]] = prev_ids[unique_mask]

                    if np.all(pts_world_ids != -1):
                        break

                fill_mask = pts_world_ids != -1
                if np.any(fill_mask):
                    matched_ids = pts_world_ids[fill_mask]
                    colors_new = img_i[fill_mask].astype(np.uint8)
                    pts_new = pts_world_i[fill_mask].astype(np.float32)
                    conf_new = conf_i[fill_mask].astype(np.float32)
                    colors_matched = submap_colors[matched_ids].astype(np.uint8, copy=False)
                    pts_matched = submap_pts_world[matched_ids, :3].astype(np.float32, copy=False)
                    conf_matched = submap_pts_world[matched_ids, 3].astype(np.float32, copy=False)

                    fused_colors, fused_pts, fused_conf = map_fusion.fuse_overlaps_torch(
                        colors_1=colors_new,
                        pts_world_1=pts_new,
                        conf_1=conf_new,
                        colors_2=colors_matched,
                        pts_world_2=pts_matched,
                        conf_2=conf_matched,
                    )
                    submap_colors[matched_ids] = fused_colors.astype(np.uint8)
                    submap_pts_world[matched_ids, :3] = fused_pts.astype(np.float32)
                    submap_pts_world[matched_ids, 3] = fused_conf.astype(np.float32)

            vals = pts_world_ids[pts_world_ids != -1]
            if np.unique(vals).size != vals.size:
                raise ValueError("pts_world_ids contains duplicate IDs (excluding -1)")

            unmatched_mask = pts_world_ids == -1
            flat_unmatched_mask = unmatched_mask.ravel()
            if np.any(flat_unmatched_mask):
                pts_unmatched_xyz = pts_world_i_flat[flat_unmatched_mask].astype(np.float32, copy=False)
                conf_unmatched = conf_i.ravel()[flat_unmatched_mask].astype(np.float32, copy=False)
                pts_unmatched = np.empty((pts_unmatched_xyz.shape[0], 4), dtype=np.float32)
                pts_unmatched[:, :3] = pts_unmatched_xyz
                pts_unmatched[:, 3] = conf_unmatched
                rgb_flat = img_i.reshape(-1, 3)
                rgb_unmatched = rgb_flat[flat_unmatched_mask]
                if pts_unmatched.shape[0] > 0:
                    new_ids = _append_submap_points(pts_unmatched, rgb_unmatched)
                    pts_world_ids_flat = pts_world_ids.ravel()
                    pts_world_ids_flat[flat_unmatched_mask] = new_ids
                    pts_world_ids = pts_world_ids_flat.reshape(H, W)

            submap_point_ids[ts_key_i] = pts_world_ids.copy()

            E = np.eye(4, dtype=np.float32)
            E[:3, :4] = extrinsics[i]
            camera_pose = self.transforms.matrix_to_pose(np.linalg.inv(E))
            poses_out.append((est_poses_sub[i][0], camera_pose))
            self.out_ph_poses_dict[_get_ts_key(est_poses_sub, i)] = camera_pose
        pts_fusion_duration = self._add_elapsed_time("fuse_pts_time", fusion_start_time)

        # Validate point-id maps and anchor (for debugging only).
        ordered_keys = [_get_ts_key(est_poses_sub, i) for i in range(N)]
        pts_ids_list = [submap_point_ids[k] for k in ordered_keys]
        if len(pts_ids_list) == 0:
            raise ValueError("submap_point_ids is empty")
        ref_shape = pts_ids_list[0].shape
        if not all(arr.shape == ref_shape for arr in pts_ids_list):
            raise ValueError(f"Shape mismatch in submap_point_ids: {[arr.shape for arr in pts_ids_list]}")
        ts_key_anchor = _get_ts_key(est_poses_sub, len(est_poses_sub)//2)
        if ts_key_anchor in self.submaps:
            raise ValueError(f"Submap '{ts_key_anchor}' already exists.")

        # Convert fused points to anchor local.
        active_submap_pts_world = np.ascontiguousarray(submap_pts_world[:submap_point_count])
        active_submap_colors = np.ascontiguousarray(submap_colors[:submap_point_count])
        anchor_local_xyz = self._transform_world_points_to_anchor_local(
            active_submap_pts_world[:, :3],
            self.out_ph_poses_dict[ts_key_anchor],
        )
        anchor_local_points = np.concatenate(
            [anchor_local_xyz, active_submap_pts_world[:, 3:4].astype(np.float32)],
            axis=1,
        )
        frame_point_ids = {frame_key: submap_point_ids[frame_key].copy() for frame_key in ordered_keys}
        if self.publish_ros2_pointcloud:
            publish_point_mask = self._build_publish_point_mask(
                anchor_local_points,
                self.ros2_pointcloud_downsample_ratio,
            )
            publish_downsample_ratio = self.ros2_pointcloud_downsample_ratio
        else:
            publish_point_mask = np.empty((0,), dtype=bool)
            publish_downsample_ratio = -1.0

        # Add submap and cache frames.
        self.submaps[ts_key_anchor] = SubmapRecord(
            anchor_key=ts_key_anchor,
            frame_keys=ordered_keys,
            local_points=anchor_local_points,
            colors=active_submap_colors,
            frame_point_ids=frame_point_ids,
            scale=1.0,
            unique_point_ids=np.arange(active_submap_pts_world.shape[0], dtype=np.int64),
            publish_point_mask=publish_point_mask,
            publish_downsample_ratio=publish_downsample_ratio,
        )

        matcher_feature_cache = None
        matcher_feature_cache_meta = None
        if (
            match_dict is not None
            and "frame_feature_cache" in match_dict
            and "frame_feature_cache_meta" in match_dict
        ):
            matcher_feature_cache = match_dict["frame_feature_cache"]
            matcher_feature_cache_meta = match_dict["frame_feature_cache_meta"]

        self.covisibility_graph.cache_frame_inputs(
            ordered_keys,
            imgs,
            intrinsics,
            confidences=confs,
            matcher_feature_cache=matcher_feature_cache,
            matcher_feature_cache_meta=matcher_feature_cache_meta,
        )
        self._register_submap_frames(ts_key_anchor, ordered_keys)

        # Submap scale optimization
        submap_scale_opt_duration = 0.0
        if self.submap_scale_opt and len(self.submaps) >= 2:
            submap_scale_opt_start_time = time.perf_counter()
            self._run_submap_scale_optimization(
                latest_n_submaps=3,
                max_points_per_overlap_frame=1000,
                log_prefix="online",
            )
            submap_scale_opt_duration = self._add_elapsed_time(
                "submap_scale_opt_time",
                submap_scale_opt_start_time,
            )

        timing_info = {
            "ts_key": ts_key_anchor,
            "num_created_submaps": 1,
            "frame_scale_opt_duration": frame_scale_opt_duration,
            "feature_match_duration": feature_match_duration,
            "submap_scale_opt_duration": submap_scale_opt_duration,
            "pts_fusion_duration": pts_fusion_duration,
        }
        return poses_out, timing_info


    def save_out_poses_dict_to_csv(self, poses_csv_path: str) -> None:
        with open(poses_csv_path, "w", newline="") as poses_file:
            poses_file.write("# counter, sec, nsec, x, y, z, qx, qy, qz, qw\n")
            for counter, ts_key in enumerate(sorted(self.out_ph_poses_dict.keys())):
                sec_str, nsec_str = ts_key.split("_")
                sec = int(sec_str)
                nsec = int(nsec_str)
                pose = self.out_ph_poses_dict[ts_key]

                line_out = str(counter) + ", " + str(sec) + ", " + str(nsec)
                line_out = line_out + ", " + str(pose.pos[0])
                line_out = line_out + ", " + str(pose.pos[1])
                line_out = line_out + ", " + str(pose.pos[2])
                line_out = line_out + ", " + str(pose.quat[0])
                line_out = line_out + ", " + str(pose.quat[1])
                line_out = line_out + ", " + str(pose.quat[2])
                line_out = line_out + ", " + str(pose.quat[3])
                line_out = line_out + "\n"
                poses_file.write(line_out)
            poses_file.flush()


    def save_out_poses_ts_to_csv(self, timestamps_path: str) -> None:
        with open(timestamps_path, "w", newline="") as timestamps_file:
            for ts_key in sorted(self.out_ph_poses_dict.keys()):
                sec_str, nsec_str = ts_key.split("_")
                ts_line = f"{int(sec_str):010d}_{int(nsec_str):09d}\n"
                timestamps_file.write(ts_line)
            timestamps_file.flush()


    def save_out_poses_dict_to_tum(self, poses_tum_path: str, poses_dict: Optional[Dict[str, MappingPose]] = None) -> None:
        if poses_dict is None:
            poses_dict = self.out_ph_poses_dict

        with open(poses_tum_path, "w", newline="") as poses_file:
            for ts_key in sorted(poses_dict.keys()):
                sec_str, nsec_str = ts_key.split("_")
                pose = poses_dict[ts_key]
                tum_ts = f"{int(sec_str)}.{int(nsec_str):09d}"
                line_out = (
                    f"{tum_ts} "
                    f"{pose.pos[0]} {pose.pos[1]} {pose.pos[2]} "
                    f"{pose.quat[0]} {pose.quat[1]} {pose.quat[2]} {pose.quat[3]}\n"
                )
                poses_file.write(line_out)
            poses_file.flush()


    def _save_graph_frame_inputs(self, frames_dir: Path) -> Dict[str, Dict[str, object]]:
        frames_dir.mkdir(parents=True, exist_ok=True)
        graph = self.covisibility_graph
        frame_keys = sorted(
            set(graph._frame_image_dict.keys())
            | set(graph._frame_intrinsics_dict.keys())
            | set(graph._frame_conf_dict.keys())
        )
        frame_manifest: Dict[str, Dict[str, object]] = {}

        for frame_key in frame_keys:
            frame_dir = frames_dir / frame_key
            frame_info: Dict[str, object] = {}

            image = graph._frame_image_dict.get(frame_key)
            if image is not None:
                image = np.asarray(image, dtype=np.uint8)
                graph_io.save_graph_array(frame_dir / "image.npy", image)
                frame_info["image"] = "image.npy"
                frame_info["image_shape"] = [int(v) for v in image.shape]
                frame_info["image_dtype"] = str(image.dtype)

            intrinsics = graph._frame_intrinsics_dict.get(frame_key)
            if intrinsics is not None:
                intrinsics = np.asarray(intrinsics, dtype=np.float32)
                graph_io.save_graph_array(frame_dir / "intrinsics.npy", intrinsics)
                frame_info["intrinsics"] = "intrinsics.npy"

            confidence = graph._frame_conf_dict.get(frame_key)
            if confidence is not None:
                confidence = np.asarray(confidence, dtype=np.float32)
                graph_io.save_graph_array(frame_dir / "confidence.npy", confidence)
                frame_info["confidence"] = "confidence.npy"
                frame_info["confidence_shape"] = [int(v) for v in confidence.shape]

            frame_manifest[frame_key] = frame_info

        return frame_manifest


    def _save_graph_submaps(self, submaps_dir: Path) -> Dict[str, Dict[str, object]]:
        submaps_dir.mkdir(parents=True, exist_ok=True)
        submap_manifest: Dict[str, Dict[str, object]] = {}
        for submap_key in sorted(self.submaps.keys()):
            submap = self.submaps[submap_key]
            submap_dir = submaps_dir / submap_key
            frame_point_ids_dir = submap_dir / "frame_point_ids"

            graph_io.save_graph_array(
                submap_dir / "local_points.npy",
                submap.local_points.astype(np.float32, copy=False),
            )
            graph_io.save_graph_array(
                submap_dir / "colors.npy",
                submap.colors.astype(np.uint8, copy=False),
            )
            graph_io.save_graph_array(
                submap_dir / "unique_point_ids.npy",
                submap.unique_point_ids.astype(np.int64, copy=False),
            )
            if submap.publish_point_mask.size > 0:
                graph_io.save_graph_array(
                    submap_dir / "publish_point_mask.npy",
                    submap.publish_point_mask.astype(bool, copy=False),
                )

            frame_point_id_files: Dict[str, str] = {}
            for frame_key in sorted(submap.frame_point_ids.keys()):
                filename = f"{frame_key}.npy"
                graph_io.save_graph_array(
                    frame_point_ids_dir / filename,
                    submap.frame_point_ids[frame_key].astype(np.int64, copy=False),
                )
                frame_point_id_files[frame_key] = f"frame_point_ids/{filename}"

            submap_manifest[submap_key] = {
                "anchor_key": submap.anchor_key,
                "frame_keys": list(submap.frame_keys),
                "scale": float(submap.scale),
                "local_points": "local_points.npy",
                "colors": "colors.npy",
                "unique_point_ids": "unique_point_ids.npy",
                "publish_point_mask": (
                    "publish_point_mask.npy"
                    if submap.publish_point_mask.size > 0
                    else None
                ),
                "publish_downsample_ratio": float(submap.publish_downsample_ratio),
                "num_points": int(submap.local_points.shape[0]),
                "frame_point_ids": frame_point_id_files,
            }

        return submap_manifest


    def _save_graph_matches(self, matches_dir: Path) -> Dict[str, Dict[str, object]]:
        matches_dir.mkdir(parents=True, exist_ok=True)
        covisible_frame_pairs, frame_pair_match_dict = self._collect_global_covisibility_scale_links(
            selected_submap_keys=set(self.submaps.keys())
        )
        match_manifest: Dict[str, Dict[str, object]] = {}

        for (frame_key_a, frame_key_b), match_result in sorted(frame_pair_match_dict.items()):
            filename = graph_io.pair_key_to_filename(frame_key_a, frame_key_b, ".npz")
            path = matches_dir / filename
            path.parent.mkdir(parents=True, exist_ok=True)
            matched_points0 = np.asarray(
                match_result.get("matched_points0", np.zeros((0, 2))),
                dtype=np.float32,
            )
            matched_points1 = np.asarray(
                match_result.get("matched_points1", np.zeros((0, 2))),
                dtype=np.float32,
            )
            num_raw_matches = int(match_result.get("num_raw_matches", 0))
            num_inlier_matches = int(
                match_result.get("num_inlier_matches", matched_points0.shape[0])
            )
            np.savez(
                path,
                matched_points0=np.ascontiguousarray(matched_points0),
                matched_points1=np.ascontiguousarray(matched_points1),
                num_raw_matches=np.array(num_raw_matches, dtype=np.int64),
                num_inlier_matches=np.array(num_inlier_matches, dtype=np.int64),
            )
            match_manifest[f"{frame_key_a}-{frame_key_b}"] = {
                "frame_key_a": frame_key_a,
                "frame_key_b": frame_key_b,
                "file": filename,
                "num_raw_matches": num_raw_matches,
                "num_inlier_matches": num_inlier_matches,
            }

        graph_pairs_manifest = {
            f"{submap_key_a}-{submap_key_b}": {
                "submap_key_a": submap_key_a,
                "submap_key_b": submap_key_b,
                "frame_pairs": [[frame_key_a, frame_key_b] for frame_key_a, frame_key_b in frame_pairs],
            }
            for (submap_key_a, submap_key_b), frame_pairs in sorted(covisible_frame_pairs.items())
        }
        return {
            "frame_pair_matches": match_manifest,
            "covisible_frame_pairs_by_submap": graph_pairs_manifest,
        }


    def save_graph(self, output_dir: Optional[Union[str, Path]] = None, overwrite: bool = True) -> Path:
        """
        Save the state needed to rerun global submap scale optimization.

        The artifact is intentionally directory-based: JSON stores topology and
        scalar metadata, while dense images, confidence maps, point clouds, ID
        maps, and pairwise matches stay in numpy files.
        """
        if not self.submaps:
            raise ValueError("No submaps available; cannot save global optimization graph.")

        if output_dir is None:
            output_dir = (
                Path(self.slam_folder)
                / "recon"
                / self.recon_save_folder_name
                / "opt_graph"
            )
        output_dir = Path(output_dir)
        tmp_output_dir = output_dir.with_name(f"{output_dir.name}.tmp")

        if output_dir.exists() and not overwrite:
            raise FileExistsError(f"Graph output directory already exists: {output_dir}")
        if tmp_output_dir.exists():
            shutil.rmtree(tmp_output_dir)
        tmp_output_dir.mkdir(parents=True, exist_ok=True)

        graph = self.covisibility_graph
        submap_manifest = self._save_graph_submaps(tmp_output_dir / "submaps")
        frame_manifest = self._save_graph_frame_inputs(tmp_output_dir / "frames")
        match_manifest = self._save_graph_matches(tmp_output_dir / "matches")

        manifest = {
            "schema_version": 1,
            "description": "ScaRF-SLAM global optimization graph artifact.",
            "slam_folder": str(self.slam_folder),
            "input_bag": str(self.input_bag),
            "prev_slam_folder": str(self.prev_slam_folder) if self.prev_slam_folder is not None else None,
            "model_name": getattr(self, "model_name", None),
            "current_traj_timestamp": self.current_traj_timestamp,
            "num_submaps": len(self.submaps),
            "num_frames": len(frame_manifest),
            "num_frame_pair_matches": len(match_manifest["frame_pair_matches"]),
            "optimizer_inputs": {
                "is_mono": bool(self.is_mono),
                "overlap_ph_views": int(self.overlap_ph_views),
                "submap_scale_opt": bool(self.submap_scale_opt),
                "frame_scale_opt": bool(self.frame_scale_opt),
                "point_cloud_fusion": bool(self.point_cloud_fusion),
                "fixed_mono_trajectory_scale": self.fixed_mono_trajectory_scale,
            },
            "covisibility_config": {
                "enabled": bool(graph.enabled),
                "max_distance": float(graph.max_distance),
                "max_angle_deg": float(graph.max_angle_deg),
                "min_time_separation_sec": float(graph.min_time_separation_sec),
                "max_old_candidates_per_new_frame": int(graph.max_old_candidates_per_new_frame),
                "max_old_edges_per_new_frame": int(graph.max_old_edges_per_new_frame),
                "matcher_name": str(graph.matcher_name),
                "matcher_device": str(graph.matcher_device),
                "max_num_keypoints": int(graph.max_num_keypoints),
                "ransac_reproj_thresh": float(graph.ransac_reproj_thresh),
                "min_inlier_matches": int(graph.min_inlier_matches),
            },
            "poses": {
                "out_ph_poses": graph_io.pose_dict_to_graph_json(self.out_ph_poses_dict),
                "in_ref_poses": graph_io.pose_dict_to_graph_json(self.in_ref_poses_dict),
                "odom_ref_poses": graph_io.pose_dict_to_graph_json(self.odom_ref_poses_dict),
            },
            "ph_to_ref": {
                ph_key: {
                    "ref_key": f"{ref_time.sec:010d}_{ref_time.nsec:09d}",
                    "ph_to_ref_pose": graph_io.pose_to_graph_json(ph_to_ref_pose),
                }
                for ph_key, (ref_time, ph_to_ref_pose) in sorted(self.ph_to_ref_dict.items())
            },
            "graph": {
                "submap_frame_keys": {
                    key: list(values)
                    for key, values in sorted(graph.submap_frame_keys_dict.items())
                },
                "submap_registration_index": {
                    key: int(value)
                    for key, value in sorted(graph._submap_registration_index.items())
                },
                "frame_to_submaps": {
                    key: sorted(values)
                    for key, values in sorted(graph.frame_to_submaps_dict.items())
                },
                "frame_covisibility_graph": graph_io.set_graph_to_graph_json(graph.frame_covisibility_graph),
                "submap_covisibility_graph": graph_io.set_graph_to_graph_json(graph.submap_covisibility_graph),
                "submap_covisibility_frame_pairs": {
                    f"{submap_key_a}-{submap_key_b}": [
                        [frame_key_a, frame_key_b]
                        for frame_key_a, frame_key_b in frame_pairs
                    ]
                    for (submap_key_a, submap_key_b), frame_pairs in sorted(graph.submap_covisibility_frame_pairs.items())
                },
            },
            "frames": frame_manifest,
            "submaps": submap_manifest,
            "matches": match_manifest,
        }

        manifest_path = tmp_output_dir / "manifest.json"
        with open(manifest_path, "w", encoding="utf-8") as f:
            json.dump(manifest, f, indent=2, sort_keys=True)
            f.write("\n")

        if output_dir.exists():
            shutil.rmtree(output_dir)
        tmp_output_dir.replace(output_dir)
        print(f"Saved global optimization graph to: {output_dir}")
        return output_dir


    def save_global_pointcloud(self, pts_global, colors_global, suffix=""):
        o3d = _require_open3d()
        conf = pts_global[:, 3]
        mask = conf != 0.0

        if np.any(mask):
            pcd_global = o3d.geometry.PointCloud()
            pcd_global.points = o3d.utility.Vector3dVector(
                pts_global[mask, :3].astype(np.float64)
            )
            pcd_global.colors = o3d.utility.Vector3dVector(
                colors_global[mask].astype(np.float32) / 255.0
            )
            recon_dir = Path(self.slam_folder) / "recon" / self.recon_save_folder_name
            recon_dir.mkdir(parents=True, exist_ok=True)
            o3d.io.write_point_cloud(
                str(recon_dir / f"pts_global{suffix}.pcd"), pcd_global
            )


    def save_per_frame_local_pointclouds(self, output_dir: Union[str, Path]) -> None:
        o3d = _require_open3d()
        output_dir = Path(output_dir)
        if output_dir.exists():
            shutil.rmtree(output_dir)  # deletes everything inside
        output_dir.mkdir(parents=True, exist_ok=True)

        if not self.submaps:
            raise ValueError("No submaps available for per-frame local point cloud export.")

        poses_tum_dict: Dict[str, MappingPose] = {}
        for submap_key in sorted(self.submaps.keys()):
            submap = self.submaps[submap_key]
            pts_world_all, colors_all = self._submap_to_world_pointcloud(submap)

            for frame_key in submap.frame_keys:
                if frame_key not in self.out_ph_poses_dict:
                    raise KeyError(f"Missing pose for frame '{frame_key}' in self.out_ph_poses_dict.")
                if frame_key not in submap.frame_point_ids:
                    raise KeyError(f"Missing frame_point_ids for frame '{frame_key}' in submap '{submap_key}'.")

                pts_ids = np.asarray(submap.frame_point_ids[frame_key], dtype=np.int64)
                valid_ids = np.unique(pts_ids[pts_ids >= 0])
                if valid_ids.size == 0:
                    continue
                if np.any(valid_ids >= pts_world_all.shape[0]):
                    raise ValueError(
                        f"Submap '{submap_key}' frame '{frame_key}' has out-of-range local point ids."
                    )

                pts_world = np.asarray(pts_world_all[valid_ids, :3], dtype=np.float64)
                colors = colors_all[valid_ids].astype(np.float64) / 255.0
                conf = pts_world_all[valid_ids, 3]
                conf_mask = np.isfinite(conf) & (conf != 0.0)
                if not np.any(conf_mask):
                    continue

                pts_world = pts_world[conf_mask]
                colors = colors[conf_mask]
                pose = self.out_ph_poses_dict[frame_key]
                t_world_cam = self.transforms.pose_to_matrix(pose)
                t_cam_world = np.linalg.inv(t_world_cam)
                pts_world_h = np.concatenate(
                    [pts_world, np.ones((pts_world.shape[0], 1), dtype=np.float64)],
                    axis=1,
                )
                pts_local = (t_cam_world @ pts_world_h.T).T[:, :3]
                finite_mask = np.isfinite(pts_local).all(axis=1)
                if not np.any(finite_mask):
                    continue

                pcd = o3d.geometry.PointCloud()
                pcd.points = o3d.utility.Vector3dVector(pts_local[finite_mask])
                pcd.colors = o3d.utility.Vector3dVector(colors[finite_mask])
                pcd_path = output_dir / f"cloud_{frame_key}.pcd"
                o3d.io.write_point_cloud(str(pcd_path), pcd)
                poses_tum_dict[frame_key] = pose

        self.save_out_poses_dict_to_tum(str(output_dir / f"poses_{self.model_name}.txt"), poses_tum_dict)


    def save_per_submap_local_pointclouds(self, output_dir: Union[str, Path]) -> None:
        o3d = _require_open3d()
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        if not self.submaps:
            raise ValueError("No submaps available for per-submap local point cloud export.")

        poses_tum_dict: Dict[str, MappingPose] = {}
        for submap_key in sorted(self.submaps.keys()):
            submap = self.submaps[submap_key]
            if not submap.frame_keys:
                continue

            last_frame_key = submap.frame_keys[-1]
            if last_frame_key not in self.out_ph_poses_dict:
                raise KeyError(
                    f"Missing pose for submap '{submap_key}' last frame '{last_frame_key}' in self.out_ph_poses_dict."
                )

            pts_world_all, colors_all = self._submap_to_world_pointcloud(submap)
            if pts_world_all.shape[0] == 0:
                continue

            conf = pts_world_all[:, 3]
            valid_mask = np.isfinite(conf) & (conf != 0.0)
            if not np.any(valid_mask):
                continue

            pts_world = np.asarray(pts_world_all[valid_mask, :3], dtype=np.float64)
            colors = colors_all[valid_mask].astype(np.float64) / 255.0

            pose = self.out_ph_poses_dict[last_frame_key]
            t_world_last = self.transforms.pose_to_matrix(pose)
            t_last_world = np.linalg.inv(t_world_last)
            pts_world_h = np.concatenate(
                [pts_world, np.ones((pts_world.shape[0], 1), dtype=np.float64)],
                axis=1,
            )
            pts_local = (t_last_world @ pts_world_h.T).T[:, :3]
            finite_mask = np.isfinite(pts_local).all(axis=1)
            if not np.any(finite_mask):
                continue

            pcd = o3d.geometry.PointCloud()
            pcd.points = o3d.utility.Vector3dVector(pts_local[finite_mask])
            pcd.colors = o3d.utility.Vector3dVector(colors[finite_mask])

            sec_str, nsec_str = last_frame_key.split("_")
            pcd_path = output_dir / f"cloud_{int(sec_str)}_{int(nsec_str)}.pcd"
            o3d.io.write_point_cloud(str(pcd_path), pcd)
            poses_tum_dict[last_frame_key] = pose

        self.save_out_poses_dict_to_tum(str(output_dir / f"poses_{self.model_name}.txt"), poses_tum_dict)


def main(args=None):
    parser = argparse.ArgumentParser(
        description="Run mapping from a SLAM folder and YAML config file."
    )
    parser.add_argument(
        "--slam_folder",
        required=True,
        help="Path to the SLAM session folder.",
    )
    parser.add_argument(
        "--input_bag",
        default=None,
        help="Path to the input ROS 2 bag.",
    )
    parser.add_argument(
        "--image_folder",
        default=None,
        help="Path to a folder containing image_<sec>_<nsec>[.<ext>] files.",
    )
    parser.add_argument(
        "--poses",
        default=None,
        help=(
            "Path to poses in counter,sec,nsec,x,y,z,qx,qy,qz,qw CSV format "
            "or TUM timestamp x y z qx qy qz qw format."
        ),
    )
    parser.add_argument(
        "--config",
        required=True,
        help="Path to the mapping YAML config file.",
    )
    parser.add_argument(
        "--prev_slam_folder",
        default=None,
        help="Optional path to the previous SLAM session folder.",
    )
    parsed_args = parser.parse_args(args)
    using_bag_input = parsed_args.input_bag is not None
    using_file_input = parsed_args.image_folder is not None or parsed_args.poses is not None
    if using_bag_input == using_file_input:
        parser.error("Pass either --input_bag or both --image_folder and --poses.")
    if using_file_input and (parsed_args.image_folder is None or parsed_args.poses is None):
        parser.error("--image_folder and --poses must be provided together.")

    print("Arguments:")
    print("  slam_folder:", parsed_args.slam_folder)
    if parsed_args.input_bag is not None:
        print("  input_bag:", parsed_args.input_bag)
    else:
        print("  image_folder:", parsed_args.image_folder)
        print("  poses:", parsed_args.poses)
    print("  config:", parsed_args.config)
    if parsed_args.prev_slam_folder is not None:
        print("  prev_slam_folder:", parsed_args.prev_slam_folder)
    print(" ")

    app = ScaRFSLAM(
        parsed_args.slam_folder,
        input_bag=parsed_args.input_bag,
        prev_slam_folder=parsed_args.prev_slam_folder,
        image_folder=parsed_args.image_folder,
        poses=parsed_args.poses,
    )
    try:
        app.app_configure(parsed_args.config)
        app.do_processing_outer()
    finally:
        if app.timestamp_sync_tmp_root is not None:
            cleanup_synced_bag_temporary_data(app.timestamp_sync_tmp_root)
        if app.ros2_node is not None:
            app.ros2_node.destroy_node()
        if rclpy is not None and rclpy.ok():
            rclpy.shutdown()


if __name__ == '__main__':
    main()
