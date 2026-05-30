#!/usr/bin/env python3

import argparse
import bisect
import csv
import re
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import open3d as o3d


GREEN = "\033[92m"
RESET = "\033[0m"
PCD_TIMESTAMP_RE = re.compile(r"^(?:cloud_)?(\d+)_(\d+)\.pcd$")
VIS_REFERENCE_VOXEL_SIZE_M = 0.10


@dataclass(frozen=True)
class PoseEntry:
    timestamp_ns: int
    translation: np.ndarray
    rotation_wc: np.ndarray
    pcd_path: Path | None = None


@dataclass(frozen=True)
class PRMetrics:
    threshold_m: float
    gt_points: int
    recon_points: int
    recon_inlier_count: int
    gt_covered_count: int
    precision: float
    recall: float
    f1: float


@dataclass(frozen=True)
class ChamferMetrics:
    threshold_m: float | None
    accuracy_m: float
    completeness_m: float
    chamfer_distance_m: float


def print_metric(name: str, value: str) -> None:
    padding = " " * max(1, 28 - len(name))
    print(f"{GREEN}{name}{padding}{value}{RESET}")


def format_percentage_metric(value: float) -> str:
    return f"{value * 100.0:.6f}%"


def parse_timestamp_to_ns(value: str) -> int:
    timestamp = value.strip()
    if not timestamp:
        raise ValueError("Empty timestamp")
    if timestamp.startswith("-"):
        raise ValueError(f"Negative timestamp is not supported: {value}")

    if "." in timestamp:
        sec_text, nsec_text = timestamp.split(".", 1)
    else:
        sec_text, nsec_text = timestamp, ""

    if sec_text == "":
        sec_text = "0"
    if not sec_text.isdigit():
        raise ValueError(f"Invalid seconds in timestamp: {value}")
    if nsec_text and not nsec_text.isdigit():
        raise ValueError(f"Invalid fractional seconds in timestamp: {value}")

    sec = int(sec_text)
    nsec = int((nsec_text + "0" * 9)[:9]) if nsec_text else 0
    return sec * 1_000_000_000 + nsec


def parse_sec_nsec_to_ns(sec_value: str, nsec_value: str) -> int:
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


def quaternion_xyzw_to_rotation_matrix(qx: float, qy: float, qz: float, qw: float) -> np.ndarray:
    quat = np.array([qx, qy, qz, qw], dtype=np.float64)
    norm = np.linalg.norm(quat)
    if norm == 0.0:
        raise ValueError("Zero-norm quaternion")
    x, y, z, w = quat / norm

    return np.array(
        [
            [1.0 - 2.0 * (y * y + z * z), 2.0 * (x * y - z * w), 2.0 * (x * z + y * w)],
            [2.0 * (x * y + z * w), 1.0 - 2.0 * (x * x + z * z), 2.0 * (y * z - x * w)],
            [2.0 * (x * z - y * w), 2.0 * (y * z + x * w), 1.0 - 2.0 * (x * x + y * y)],
        ],
        dtype=np.float64,
    )


def load_trajectory(path: Path) -> list[PoseEntry]:
    if not path.is_file():
        raise FileNotFoundError(f"Trajectory file does not exist: {path}")

    entries: list[PoseEntry] = []
    with path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            line = line.strip()
            if not line or line.startswith("#"):
                continue

            if "," in line:
                parts = next(csv.reader([line], skipinitialspace=True))
                if parts and parts[0].strip().lower() == "counter":
                    continue
                if len(parts) != 10:
                    raise ValueError(
                        "Invalid CSV trajectory row at line "
                        f"{line_number}; expected counter, sec, nsec, x, y, z, qx, qy, qz, qw: {line}"
                    )

                timestamp_ns = parse_sec_nsec_to_ns(parts[1], parts[2])
                translation = np.array([float(parts[3]), float(parts[4]), float(parts[5])], dtype=np.float64)
                rotation_wc = quaternion_xyzw_to_rotation_matrix(*map(float, parts[6:10]))
            else:
                parts = line.split()
                if len(parts) < 8:
                    raise ValueError(f"Invalid TUM row at line {line_number}: {line}")

                timestamp_ns = parse_timestamp_to_ns(parts[0])
                translation = np.array([float(parts[1]), float(parts[2]), float(parts[3])], dtype=np.float64)
                rotation_wc = quaternion_xyzw_to_rotation_matrix(*map(float, parts[4:8]))
            entries.append(PoseEntry(timestamp_ns=timestamp_ns, translation=translation, rotation_wc=rotation_wc))

    return entries


def index_pcds(pcd_dir: Path) -> dict[int, Path]:
    if not pcd_dir.is_dir():
        raise NotADirectoryError(f"PCD folder does not exist: {pcd_dir}")

    lookup: dict[int, Path] = {}
    for path in sorted(pcd_dir.iterdir()):
        if not path.is_file():
            continue
        match = PCD_TIMESTAMP_RE.match(path.name)
        if match is None:
            continue
        timestamp_ns = int(match.group(1)) * 1_000_000_000 + int(match.group(2))
        lookup[timestamp_ns] = path
    return lookup


def attach_pcd_paths(entries: list[PoseEntry], pcd_dir: Path) -> list[PoseEntry]:
    pcd_lookup = index_pcds(pcd_dir)
    attached: list[PoseEntry] = []
    for entry in entries:
        pcd_path = pcd_lookup.get(entry.timestamp_ns)
        if pcd_path is None:
            continue
        attached.append(
            PoseEntry(
                timestamp_ns=entry.timestamp_ns,
                translation=entry.translation,
                rotation_wc=entry.rotation_wc,
                pcd_path=pcd_path,
            )
        )
    return attached


def load_pcd(path: Path) -> o3d.geometry.PointCloud:
    cloud = o3d.io.read_point_cloud(str(path))
    if cloud.is_empty():
        raise RuntimeError(f"Point cloud is empty or unreadable: {path}")
    return cloud


def find_chunk_indices(entries: list[PoseEntry], start_ns: int, end_ns: int) -> tuple[int, int]:
    timestamps = [entry.timestamp_ns for entry in entries]
    return bisect.bisect_left(timestamps, start_ns), bisect.bisect_left(timestamps, end_ns)


def find_chunk_end_by_distance(
    entries: list[PoseEntry],
    start_ns: int,
    overlap_end_ns: int,
    chunk_distance_m: float,
) -> int:
    timestamps = [entry.timestamp_ns for entry in entries]
    start_idx = bisect.bisect_left(timestamps, start_ns)
    overlap_end_idx = bisect.bisect_left(timestamps, overlap_end_ns)
    if start_idx >= overlap_end_idx:
        return overlap_end_ns

    total_distance_m = 0.0
    prev_translation = entries[start_idx].translation
    for idx in range(start_idx + 1, overlap_end_idx):
        curr_translation = entries[idx].translation
        total_distance_m += float(np.linalg.norm(curr_translation - prev_translation))
        prev_translation = curr_translation
        if total_distance_m >= chunk_distance_m:
            if idx + 1 < overlap_end_idx:
                return entries[idx + 1].timestamp_ns
            return overlap_end_ns
    return overlap_end_ns


def match_chunk_entries(
    gt_entries: list[PoseEntry],
    recon_entries: list[PoseEntry],
    tolerance_ns: int,
) -> list[tuple[PoseEntry, PoseEntry]]:
    matches: list[tuple[PoseEntry, PoseEntry]] = []
    recon_timestamps = [entry.timestamp_ns for entry in recon_entries]
    used_recon_indices: set[int] = set()

    for gt_entry in gt_entries:
        pos = bisect.bisect_left(recon_timestamps, gt_entry.timestamp_ns)
        candidates = []
        if pos < len(recon_entries):
            candidates.append(pos)
        if pos > 0:
            candidates.append(pos - 1)

        best_index = None
        best_dt = None
        for candidate in candidates:
            if candidate in used_recon_indices:
                continue
            dt = abs(recon_entries[candidate].timestamp_ns - gt_entry.timestamp_ns)
            if dt > tolerance_ns:
                continue
            if best_dt is None or dt < best_dt:
                best_dt = dt
                best_index = candidate

        if best_index is None:
            continue

        used_recon_indices.add(best_index)
        matches.append((gt_entry, recon_entries[best_index]))

    return matches


def estimate_se3(source_points: np.ndarray, target_points: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    mu_src = np.mean(source_points, axis=0)
    mu_tgt = np.mean(target_points, axis=0)
    src_centered = source_points - mu_src
    tgt_centered = target_points - mu_tgt

    u, _, vt = np.linalg.svd(src_centered.T @ tgt_centered)
    rotation = vt.T @ u.T
    if np.linalg.det(rotation) < 0.0:
        vt[2, :] *= -1.0
        rotation = vt.T @ u.T

    translation = mu_tgt - rotation @ mu_src
    return rotation, translation


def estimate_sim3(source_points: np.ndarray, target_points: np.ndarray) -> tuple[float, np.ndarray, np.ndarray]:
    if source_points.shape != target_points.shape:
        raise ValueError(
            "Source and target trajectories must have matching shapes for Sim(3) alignment, "
            f"got {source_points.shape} and {target_points.shape}."
        )
    if source_points.shape[0] < 3:
        raise ValueError("Need at least 3 matched poses to estimate Sim(3).")

    mu_src = np.mean(source_points, axis=0)
    mu_tgt = np.mean(target_points, axis=0)
    src_centered = source_points - mu_src
    tgt_centered = target_points - mu_tgt

    covariance = tgt_centered.T @ src_centered / source_points.shape[0]
    u, singular_values, vt = np.linalg.svd(covariance)
    sign = np.ones(3, dtype=np.float64)
    if np.linalg.det(u) * np.linalg.det(vt) < 0.0:
        sign[-1] = -1.0
    rotation = u @ np.diag(sign) @ vt

    src_variance = float(np.mean(np.sum(src_centered * src_centered, axis=1)))
    if src_variance <= 1e-15:
        raise ValueError("Source trajectory chunk variance is too small to estimate Sim(3).")
    scale = float(np.sum(singular_values * sign) / src_variance)
    translation = mu_tgt - scale * (rotation @ mu_src)
    return scale, rotation, translation


def transform_camera_points(points_camera: np.ndarray, translation_wc: np.ndarray, rotation_wc: np.ndarray) -> np.ndarray:
    return points_camera @ rotation_wc.T + translation_wc


def apply_sim3(points: np.ndarray, scale: float, rotation: np.ndarray, translation: np.ndarray) -> np.ndarray:
    return scale * (points @ rotation.T) + translation


def apply_sim3_to_cloud(
    cloud: o3d.geometry.PointCloud,
    scale: float,
    rotation: np.ndarray,
    translation: np.ndarray,
) -> o3d.geometry.PointCloud:
    transformed = o3d.geometry.PointCloud(cloud)
    points = np.asarray(transformed.points, dtype=np.float64)
    transformed.points = o3d.utility.Vector3dVector(apply_sim3(points, scale, rotation, translation))
    return transformed


def compute_ate_errors(
    gt_positions: np.ndarray,
    recon_positions: np.ndarray,
    scale: float,
    rotation: np.ndarray,
    translation: np.ndarray,
) -> np.ndarray:
    aligned_recon_positions = apply_sim3(recon_positions, scale, rotation, translation)
    return np.linalg.norm(aligned_recon_positions - gt_positions, axis=1)


def build_world_cloud_for_entries(
    entries: list[PoseEntry],
    scale: float = 1.0,
    sim3_rotation: np.ndarray | None = None,
    sim3_translation: np.ndarray | None = None,
    random_downsample: float = 1.0,
    rng: np.random.Generator | None = None,
) -> o3d.geometry.PointCloud:
    world_points: list[np.ndarray] = []
    world_colors: list[np.ndarray | None] = []
    has_any_colors = False

    for entry in entries:
        if entry.pcd_path is None:
            raise RuntimeError("Pose entry is missing a point-cloud path.")

        local_cloud = load_pcd(entry.pcd_path)
        points_local = np.asarray(local_cloud.points, dtype=np.float64)
        colors_local = np.asarray(local_cloud.colors, dtype=np.float64) if local_cloud.has_colors() else None

        if random_downsample < 1.0 and points_local.shape[0] > 0:
            if rng is None:
                rng = np.random.default_rng()
            keep_mask = rng.random(points_local.shape[0]) < random_downsample
            if not np.any(keep_mask):
                keep_mask[int(rng.integers(points_local.shape[0]))] = True
            points_local = points_local[keep_mask]
            if colors_local is not None:
                colors_local = colors_local[keep_mask]

        points_world = transform_camera_points(points_local, entry.translation, entry.rotation_wc)
        if sim3_rotation is not None and sim3_translation is not None:
            points_world = apply_sim3(points_world, scale, sim3_rotation, sim3_translation)

        world_points.append(points_world)
        world_colors.append(colors_local)
        has_any_colors = has_any_colors or colors_local is not None

    if not world_points:
        raise RuntimeError("No point clouds were loaded for this chunk.")

    cloud = o3d.geometry.PointCloud()
    merged_points = np.concatenate(world_points, axis=0)
    cloud.points = o3d.utility.Vector3dVector(merged_points)

    if has_any_colors:
        merged_colors = []
        for points, colors in zip(world_points, world_colors):
            if colors is None:
                fallback = np.tile(np.array([[0.1, 0.7, 1.0]], dtype=np.float64), (points.shape[0], 1))
                merged_colors.append(fallback)
            else:
                merged_colors.append(colors)
        cloud.colors = o3d.utility.Vector3dVector(np.concatenate(merged_colors, axis=0))

    return cloud


def voxel_downsample(cloud: o3d.geometry.PointCloud, voxel_size: float) -> o3d.geometry.PointCloud:
    if voxel_size <= 0.0:
        return o3d.geometry.PointCloud(cloud)
    return cloud.voxel_down_sample(voxel_size)


def crop_gt_cloud_near_recon(
    gt_cloud: o3d.geometry.PointCloud,
    recon_cloud: o3d.geometry.PointCloud,
    margin_m: float,
) -> o3d.geometry.PointCloud:
    if margin_m < 0.0:
        raise ValueError(f"GT crop margin must be >= 0, got {margin_m}")
    if len(gt_cloud.points) == 0 or len(recon_cloud.points) == 0:
        return o3d.geometry.PointCloud(gt_cloud)

    recon_points = np.asarray(recon_cloud.points, dtype=np.float64)
    min_bound = recon_points.min(axis=0) - margin_m
    max_bound = recon_points.max(axis=0) + margin_m
    cropped = gt_cloud.crop(o3d.geometry.AxisAlignedBoundingBox(min_bound=min_bound, max_bound=max_bound))
    if len(cropped.points) == 0:
        return o3d.geometry.PointCloud(gt_cloud)
    return cropped


def resolve_gt_crop_margin(args: argparse.Namespace) -> float:
    if args.gt_crop_margin is not None:
        return args.gt_crop_margin
    margin = max(args.threshold, args.voxel_size)
    if args.chamfer_threshold is None:
        return max(margin, 1.0)
    return max(margin, args.chamfer_threshold)


def align_source_to_target_icp(
    source_cloud: o3d.geometry.PointCloud,
    target_cloud: o3d.geometry.PointCloud,
    max_correspondence_distance: float,
) -> tuple[o3d.geometry.PointCloud, o3d.pipelines.registration.RegistrationResult]:
    if max_correspondence_distance <= 0.0:
        raise ValueError(
            "ICP max_correspondence_distance must be > 0, "
            f"got {max_correspondence_distance}"
        )

    registration = o3d.pipelines.registration.registration_icp(
        source=source_cloud,
        target=target_cloud,
        max_correspondence_distance=max_correspondence_distance,
        init=np.eye(4, dtype=np.float64),
        estimation_method=o3d.pipelines.registration.TransformationEstimationPointToPoint(),
    )

    aligned_source = o3d.geometry.PointCloud(source_cloud)
    aligned_source.transform(registration.transformation)
    return aligned_source, registration


def transform_cloud(cloud: o3d.geometry.PointCloud, transformation: np.ndarray) -> o3d.geometry.PointCloud:
    transformed = o3d.geometry.PointCloud(cloud)
    transformed.transform(transformation)
    return transformed


def compute_nn_distances(
    gt_cloud: o3d.geometry.PointCloud,
    recon_cloud: o3d.geometry.PointCloud,
) -> tuple[np.ndarray, np.ndarray]:
    if len(gt_cloud.points) == 0 or len(recon_cloud.points) == 0:
        raise RuntimeError("Both point clouds must contain points.")

    recon_to_gt = np.asarray(recon_cloud.compute_point_cloud_distance(gt_cloud), dtype=np.float64)
    gt_to_recon = np.asarray(gt_cloud.compute_point_cloud_distance(recon_cloud), dtype=np.float64)
    return recon_to_gt, gt_to_recon


def compute_precision_recall_from_distances(
    recon_to_gt_dists: np.ndarray,
    gt_to_recon_dists: np.ndarray,
    threshold_m: float,
) -> PRMetrics:
    if threshold_m <= 0.0:
        raise ValueError(f"threshold_m must be > 0, got {threshold_m}")

    recon_inliers = int(np.count_nonzero(recon_to_gt_dists <= threshold_m))
    gt_covered = int(np.count_nonzero(gt_to_recon_dists <= threshold_m))
    precision = recon_inliers / recon_to_gt_dists.shape[0]
    recall = gt_covered / gt_to_recon_dists.shape[0]
    f1 = 0.0 if precision + recall == 0.0 else 2.0 * precision * recall / (precision + recall)
    return PRMetrics(
        threshold_m=threshold_m,
        gt_points=int(gt_to_recon_dists.shape[0]),
        recon_points=int(recon_to_gt_dists.shape[0]),
        recon_inlier_count=recon_inliers,
        gt_covered_count=gt_covered,
        precision=precision,
        recall=recall,
        f1=f1,
    )


def compute_chamfer_from_distances(
    recon_to_gt_dists: np.ndarray,
    gt_to_recon_dists: np.ndarray,
    threshold_m: float | None,
) -> ChamferMetrics:
    if threshold_m is not None and threshold_m <= 0.0:
        raise ValueError(f"threshold_m must be > 0, got {threshold_m}")

    recon_valid = recon_to_gt_dists
    gt_valid = gt_to_recon_dists
    if threshold_m is not None:
        recon_valid = recon_to_gt_dists[recon_to_gt_dists <= threshold_m]
        gt_valid = gt_to_recon_dists[gt_to_recon_dists <= threshold_m]

    accuracy = float(np.mean(recon_valid)) if recon_valid.size > 0 else float("nan")
    completeness = float(np.mean(gt_valid)) if gt_valid.size > 0 else float("nan")
    return ChamferMetrics(
        threshold_m=threshold_m,
        accuracy_m=accuracy,
        completeness_m=completeness,
        chamfer_distance_m=0.5 * (accuracy + completeness),
    )


def cloud_colors(cloud: o3d.geometry.PointCloud, fallback: np.ndarray) -> np.ndarray:
    point_count = len(cloud.points)
    if cloud.has_colors():
        return np.asarray(cloud.colors, dtype=np.float64).copy()
    return np.tile(fallback.reshape(1, 3), (point_count, 1))


def colors_by_distance(distances: np.ndarray, max_distance_m: float) -> np.ndarray:
    if max_distance_m <= 0.0:
        raise ValueError(f"max_distance_m must be > 0, got {max_distance_m}")
    normalized = np.clip(distances, 0.0, max_distance_m) / max_distance_m
    colors = np.zeros((distances.shape[0], 3), dtype=np.float64)
    colors[:, 0] = normalized
    colors[:, 1] = 1.0 - normalized
    return colors


def green_colors_by_height(cloud: o3d.geometry.PointCloud) -> np.ndarray:
    points = np.asarray(cloud.points, dtype=np.float64)
    if points.shape[0] == 0:
        return np.empty((0, 3), dtype=np.float64)

    heights = points[:, 2]
    height_range = float(np.ptp(heights))
    if height_range <= 1e-12:
        normalized = np.full(heights.shape, 0.5, dtype=np.float64)
    else:
        normalized = (heights - float(np.min(heights))) / height_range

    low = np.array([0.0, 0.25, 0.05], dtype=np.float64)
    high = np.array([0.55, 1.0, 0.25], dtype=np.float64)
    return low + normalized[:, None] * (high - low)


def show_clouds(window_name: str, clouds: list[o3d.geometry.PointCloud]) -> None:
    vis = o3d.visualization.Visualizer()
    vis.create_window(window_name=window_name)
    for cloud in clouds:
        vis.add_geometry(cloud)
    render_option = vis.get_render_option()
    render_option.point_size = 1.0
    vis.run()
    vis.destroy_window()


def visualize_distances(
    gt_cloud: o3d.geometry.PointCloud,
    recon_cloud: o3d.geometry.PointCloud,
    recon_to_gt_dists: np.ndarray,
    gt_to_recon_dists: np.ndarray,
    threshold_m: float,
    max_distance_m: float,
    recall_only: bool,
    include_reference: bool,
) -> None:
    if recall_only:
        gt_vis = o3d.geometry.PointCloud(gt_cloud)
        gt_vis.colors = o3d.utility.Vector3dVector(colors_by_distance(gt_to_recon_dists, max_distance_m))
        clouds = [gt_vis]
        if include_reference:
            recon_vis = o3d.geometry.PointCloud(recon_cloud)
            recon_vis.colors = o3d.utility.Vector3dVector(
                cloud_colors(recon_vis, np.array([0.1, 0.7, 1.0], dtype=np.float64))
            )
            clouds.append(recon_vis)
        show_clouds(f"GT distance to recon (0-{max_distance_m:.3f}m)", clouds)
        return

    recon_vis = o3d.geometry.PointCloud(recon_cloud)
    recon_vis.colors = o3d.utility.Vector3dVector(colors_by_distance(recon_to_gt_dists, max_distance_m))
    clouds = []
    if include_reference:
        gt_vis = voxel_downsample(gt_cloud, VIS_REFERENCE_VOXEL_SIZE_M)
        gt_vis.colors = o3d.utility.Vector3dVector(green_colors_by_height(gt_vis))
        clouds.append(gt_vis)
    clouds.append(recon_vis)
    show_clouds(f"Recon distance to GT (0-{max_distance_m:.3f}m, threshold={threshold_m:.4f}m)", clouds)


def visualize_outliers(
    gt_cloud: o3d.geometry.PointCloud,
    recon_cloud: o3d.geometry.PointCloud,
    recon_to_gt_dists: np.ndarray,
    gt_to_recon_dists: np.ndarray,
    threshold_m: float,
    recall_only: bool,
    include_reference: bool,
) -> None:
    outlier_color = np.array([1.0, 0.0, 0.0], dtype=np.float64)
    if recall_only:
        gt_vis = o3d.geometry.PointCloud(gt_cloud)
        gt_colors = green_colors_by_height(gt_vis)
        gt_colors[gt_to_recon_dists > threshold_m] = outlier_color
        gt_vis.colors = o3d.utility.Vector3dVector(gt_colors)
        clouds = [gt_vis]
        if include_reference:
            recon_vis = o3d.geometry.PointCloud(recon_cloud)
            recon_vis.colors = o3d.utility.Vector3dVector(
                cloud_colors(recon_vis, np.array([0.1, 0.7, 1.0], dtype=np.float64))
            )
            clouds.append(recon_vis)
        show_clouds(f"GT uncovered points (red, threshold={threshold_m:.4f}m)", clouds)
        return

    recon_vis = o3d.geometry.PointCloud(recon_cloud)
    recon_colors = cloud_colors(recon_vis, np.array([0.1, 0.7, 1.0], dtype=np.float64))
    recon_colors[recon_to_gt_dists > threshold_m] = outlier_color
    recon_vis.colors = o3d.utility.Vector3dVector(recon_colors)
    clouds = []
    if include_reference:
        gt_vis = voxel_downsample(gt_cloud, VIS_REFERENCE_VOXEL_SIZE_M)
        gt_vis.colors = o3d.utility.Vector3dVector(green_colors_by_height(gt_vis))
        clouds.append(gt_vis)
    clouds.append(recon_vis)
    show_clouds(f"Recon outliers (red, threshold={threshold_m:.4f}m)", clouds)


def compute_total_translation(entries: list[PoseEntry]) -> float:
    if len(entries) < 2:
        return 0.0
    positions = np.vstack([entry.translation for entry in entries])
    return float(np.linalg.norm(positions[1:] - positions[:-1], axis=1).sum())


def uniformly_subsample_entries(entries: list[PoseEntry], max_frames: int | None) -> list[PoseEntry]:
    if max_frames is None or len(entries) <= max_frames:
        return entries
    if max_frames <= 0:
        raise ValueError("--max-frames must be > 0")

    indices = np.linspace(0, len(entries) - 1, num=max_frames)
    return [entries[int(round(index))] for index in indices]


def format_timestamp_ns(timestamp_ns: int) -> str:
    sec = timestamp_ns // 1_000_000_000
    nsec = timestamp_ns % 1_000_000_000
    return f"{sec}.{nsec:09d}"


def format_metric_list(values: list[float], as_percentage: bool = False) -> str:
    if as_percentage:
        return "[" + ", ".join(format_percentage_metric(value) for value in values) + "]"
    return "[" + ", ".join(f"{value:.6f}" for value in values) + "]"


def print_metric_summary(name: str, values: list[float], as_percentage: bool = False) -> None:
    if not values:
        return
    average = float(np.mean(np.asarray(values, dtype=np.float64)))
    average_text = format_percentage_metric(average) if as_percentage else f"{average:.6f}"
    print_metric(name, f"{format_metric_list(values, as_percentage=as_percentage)}, avg={average_text}")


def report_metrics(
    gt_cloud: o3d.geometry.PointCloud,
    recon_cloud: o3d.geometry.PointCloud,
    args: argparse.Namespace,
) -> tuple[PRMetrics, ChamferMetrics, np.ndarray, np.ndarray]:
    recon_to_gt_dists, gt_to_recon_dists = compute_nn_distances(gt_cloud, recon_cloud)
    metrics = compute_precision_recall_from_distances(
        recon_to_gt_dists=recon_to_gt_dists,
        gt_to_recon_dists=gt_to_recon_dists,
        threshold_m=args.threshold,
    )
    chamfer = compute_chamfer_from_distances(
        recon_to_gt_dists=recon_to_gt_dists,
        gt_to_recon_dists=gt_to_recon_dists,
        threshold_m=args.chamfer_threshold,
    )

    print(f"threshold_m:       {metrics.threshold_m:.6f}")
    if chamfer.threshold_m is not None:
        print(f"chamfer_thr_m:     {chamfer.threshold_m:.6f}")
    print(f"gt_points:         {metrics.gt_points}")
    print(f"recon_points:      {metrics.recon_points}")

    if args.precision:
        print(f"recon_inliers:     {metrics.recon_inlier_count}")
        print_metric("precision:", format_percentage_metric(metrics.precision))
        print_metric("reconstruction error (m):", f"{chamfer.accuracy_m:.6f}")
    elif args.recall:
        print(f"gt_covered:        {metrics.gt_covered_count}")
        print_metric("recall:", format_percentage_metric(metrics.recall))
        print_metric("coverage error (m):", f"{chamfer.completeness_m:.6f}")
    else:
        print(f"recon_inliers:     {metrics.recon_inlier_count}")
        print(f"gt_covered:        {metrics.gt_covered_count}")
        print_metric("precision:", format_percentage_metric(metrics.precision))
        print_metric("recall:", format_percentage_metric(metrics.recall))
        print_metric("f1:", f"{metrics.f1:.6f}")
        print_metric("reconstruction error (m):", f"{chamfer.accuracy_m:.6f}")
        print_metric("coverage error (m):", f"{chamfer.completeness_m:.6f}")
        print_metric("chamfer_m:", f"{chamfer.chamfer_distance_m:.6f}")

    if args.vis:
        visualize_outliers(
            gt_cloud=gt_cloud,
            recon_cloud=recon_cloud,
            recon_to_gt_dists=recon_to_gt_dists,
            gt_to_recon_dists=gt_to_recon_dists,
            threshold_m=args.threshold,
            recall_only=args.recall,
            include_reference=args.vis_all,
        )
    if args.vis_dist:
        max_distance_m = args.chamfer_threshold if args.chamfer_threshold is not None else max(args.threshold, 0.1)
        visualize_distances(
            gt_cloud=gt_cloud,
            recon_cloud=recon_cloud,
            recon_to_gt_dists=recon_to_gt_dists,
            gt_to_recon_dists=gt_to_recon_dists,
            threshold_m=args.threshold,
            max_distance_m=max_distance_m,
            recall_only=args.recall,
            include_reference=args.vis_dist_all,
        )

    return metrics, chamfer, recon_to_gt_dists, gt_to_recon_dists


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Compare point clouds by using chunk-wise evaluation when --recon is a folder, "
            "or one global evaluation when --recon is a world-coordinate PCD file."
        )
    )
    parser.add_argument(
        "--gt",
        required=True,
        help=(
            "Either a folder containing GT per-pose .pcd files named {sec}_{nsec}.pcd or cloud_{sec}_{nsec}.pcd, "
            "or a single GT .pcd already in world coordinates."
        ),
    )
    parser.add_argument(
        "--recon",
        required=True,
        help=(
            "Either a folder containing recon per-pose .pcd files named {sec}_{nsec}.pcd or "
            "cloud_{sec}_{nsec}.pcd, or a single recon .pcd already in world coordinates."
        ),
    )
    parser.add_argument(
        "--gt-traj",
        required=True,
        help="GT trajectory in TUM or CSV counter/sec/nsec format (cam2world).",
    )
    parser.add_argument(
        "--recon-traj",
        required=True,
        help="Recon trajectory in TUM or CSV counter/sec/nsec format (cam2world).",
    )
    parser.add_argument(
        "--max-frames",
        type=int,
        default=None,
        help="Uniformly subsample eligible recon trajectory frames to at most this many frames for evaluation.",
    )
    parser.add_argument(
        "--chunk-m",
        type=float,
        default=10.0,
        help="Target GT travel distance in meters for each chunk.",
    )
    parser.add_argument(
        "--chunk-m-min",
        type=float,
        default=None,
        help="Skip chunks whose actual GT travel distance is smaller than this value. Defaults to --chunk-m.",
    )
    parser.add_argument(
        "--threshold",
        type=float,
        default=0.03,
        help="Distance threshold in meters for precision/recall matching.",
    )
    parser.add_argument(
        "--chamfer-threshold",
        type=float,
        default=None,
        help=(
            "Distance threshold in meters for accuracy/completeness. "
            "Distances above this are ignored; unset means use all distances."
        ),
    )
    metric_group = parser.add_mutually_exclusive_group()
    metric_group.add_argument(
        "--precision",
        action="store_true",
        help="Report only precision metrics and accuracy.",
    )
    metric_group.add_argument(
        "--recall",
        action="store_true",
        help="Report only recall metrics and completeness.",
    )
    parser.add_argument(
        "--voxel-size",
        type=float,
        default=0.02,
        help="Voxel size in meters. Evaluation and ICP clouds are always downsampled in memory with this size.",
    )
    parser.add_argument(
        "--input-downsample",
        type=float,
        default=1.0,
        help="When --recon is a folder, randomly keep this fraction of points from each input recon PCD, in [0, 1].",
    )
    parser.add_argument(
        "--gt-crop-margin",
        type=float,
        default=None,
        help=(
            "When --gt is a global PCD, crop GT to aligned recon bounds expanded by this margin in meters. "
            "Defaults to max(--threshold, --voxel-size, --chamfer-threshold), or at least 1.0 m when "
            "--chamfer-threshold is unset."
        ),
    )
    parser.add_argument(
        "--no-gt-crop",
        action="store_true",
        help="Disable automatic global-GT cropping before metric evaluation.",
    )
    parser.add_argument(
        "--match-tolerance",
        type=float,
        default=0.01,
        help="Maximum allowed timestamp difference in seconds when matching GT and recon poses.",
    )
    parser.add_argument(
        "--align-mode",
        choices=("sim3", "se3"),
        default="sim3",
        help="Trajectory alignment model to use before point-cloud evaluation.",
    )
    parser.add_argument(
        "--vis",
        action="store_true",
        help="Visualize outliers in red while preserving original colors for inliers.",
    )
    parser.add_argument(
        "--vis-dist",
        action="store_true",
        help="Visualize points colored by nearest-neighbor distance.",
    )
    parser.add_argument(
        "--vis-all",
        action="store_true",
        help="Visualize outliers and include the reference cloud. Implies --vis.",
    )
    parser.add_argument(
        "--vis-dist-all",
        action="store_true",
        help="Visualize nearest-neighbor distances and include the reference cloud. Implies --vis-dist.",
    )
    parser.add_argument(
        "--icp",
        action="store_true",
        help="Align the downsampled recon cloud to GT with ICP after trajectory alignment.",
    )
    return parser


def validate_args(args: argparse.Namespace) -> None:
    if args.chunk_m <= 0.0:
        raise ValueError("--chunk-m must be > 0")
    if args.chunk_m_min is None:
        args.chunk_m_min = args.chunk_m
    if args.chunk_m_min < 0.0:
        raise ValueError("--chunk-m-min must be >= 0")
    if args.threshold <= 0.0:
        raise ValueError("--threshold must be > 0")
    if args.chamfer_threshold is not None and args.chamfer_threshold <= 0.0:
        raise ValueError("--chamfer-threshold must be > 0")
    if args.voxel_size < 0.0:
        raise ValueError("--voxel-size must be >= 0")
    if not (0.0 <= args.input_downsample <= 1.0):
        raise ValueError("--input-downsample must be within [0, 1]")
    if args.gt_crop_margin is not None and args.gt_crop_margin < 0.0:
        raise ValueError("--gt-crop-margin must be >= 0")
    if args.match_tolerance < 0.0:
        raise ValueError("--match-tolerance must be >= 0")
    if args.max_frames is not None and args.max_frames <= 0:
        raise ValueError("--max-frames must be > 0")
    if args.vis_all:
        args.vis = True
    if args.vis_dist_all:
        args.vis_dist = True


def main() -> None:
    args = build_parser().parse_args()
    validate_args(args)

    gt_path = Path(args.gt)
    recon_path = Path(args.recon)
    gt_is_global_pcd = gt_path.is_file()
    recon_is_global_pcd = recon_path.is_file()
    if not recon_is_global_pcd and not recon_path.is_dir():
        raise FileNotFoundError(f"--recon must be either a folder of per-pose PCD files or a single PCD file: {recon_path}")
    if gt_is_global_pcd and not (args.precision or recon_is_global_pcd):
        raise AssertionError("--gt must be a folder of per-pose PCD files unless --precision is set or --recon is a PCD file.")

    gt_traj_entries = load_trajectory(Path(args.gt_traj))
    recon_traj_entries = load_trajectory(Path(args.recon_traj))
    if recon_is_global_pcd:
        recon_global_cloud = load_pcd(recon_path)
        recon_entries = recon_traj_entries
        print(f"eval_recon:        {recon_path}")
    else:
        recon_global_cloud = None
        recon_entries = attach_pcd_paths(recon_traj_entries, recon_path)

    if gt_is_global_pcd:
        gt_global_cloud = load_pcd(gt_path)
        gt_pcd_entries = None
        print(f"eval_gt:           {gt_path}")
    else:
        gt_global_cloud = None
        gt_pcd_entries = attach_pcd_paths(gt_traj_entries, gt_path)

    if len(gt_traj_entries) < 3:
        raise RuntimeError("Need at least 3 GT trajectory poses.")
    if len(recon_entries) < 3:
        if recon_is_global_pcd:
            raise RuntimeError("Need at least 3 recon trajectory poses.")
        raise RuntimeError("Need at least 3 recon poses with matching .pcd files.")

    overlap_start_ns = max(gt_traj_entries[0].timestamp_ns, recon_entries[0].timestamp_ns)
    overlap_end_ns = min(gt_traj_entries[-1].timestamp_ns, recon_entries[-1].timestamp_ns)
    if overlap_end_ns <= overlap_start_ns:
        raise RuntimeError("GT and recon inputs do not overlap in time.")

    recon_entries_before_max_frames = len(recon_entries)
    if args.max_frames is not None:
        recon_begin, recon_end = find_chunk_indices(recon_entries, overlap_start_ns, overlap_end_ns)
        recon_entries = uniformly_subsample_entries(recon_entries[recon_begin:recon_end], args.max_frames)
        if len(recon_entries) < 3:
            if recon_is_global_pcd:
                raise RuntimeError("Need at least 3 recon trajectory poses after --max-frames.")
            raise RuntimeError("Need at least 3 recon poses with matching .pcd files after --max-frames.")
        overlap_start_ns = max(overlap_start_ns, recon_entries[0].timestamp_ns)
        overlap_end_ns = min(overlap_end_ns, recon_entries[-1].timestamp_ns + 1)
        if overlap_end_ns <= overlap_start_ns:
            raise RuntimeError("GT and recon inputs do not overlap in time after --max-frames.")

    tolerance_ns = int(round(args.match_tolerance * 1_000_000_000))
    print(f"gt_traj_poses:     {len(gt_traj_entries)}")
    if gt_pcd_entries is not None:
        print(f"gt_pcd_entries:    {len(gt_pcd_entries)}")
    print(f"recon_entries:     {len(recon_entries)}")
    if args.max_frames is not None:
        print(f"max_frames:        {args.max_frames}")
        print(f"recon_entries_all: {recon_entries_before_max_frames}")
    print(f"overlap_start:     {format_timestamp_ns(overlap_start_ns)}")
    print(f"overlap_end:       {format_timestamp_ns(overlap_end_ns)}")
    print(f"eval_mode:         {'global' if recon_is_global_pcd else 'chunk'}")
    if not recon_is_global_pcd:
        print("chunk_mode:        meter")
        print(f"chunk_m_m:         {args.chunk_m:.6f}")
        if args.chunk_m_min > 0.0:
            print(f"chunk_m_min_m:     {args.chunk_m_min:.6f}")
    print(f"match_tol_s:       {args.match_tolerance:.6f}")
    print(f"align_mode:        {args.align_mode}")
    print(f"voxel_size_m:      {args.voxel_size:.6f}")
    if args.input_downsample < 1.0:
        print(f"input_downsample:  {args.input_downsample:.6f}")
    if args.chamfer_threshold is not None:
        print(f"chamfer_thr_m:     {args.chamfer_threshold:.6f}")
    if args.icp:
        print("icp:               enabled")

    crop_global_gt = args.precision and gt_is_global_pcd and not args.no_gt_crop
    gt_crop_margin_m = resolve_gt_crop_margin(args) if crop_global_gt else None
    if gt_crop_margin_m is not None:
        print(f"gt_crop_margin_m:  {gt_crop_margin_m:.6f}")

    chunks_processed = 0
    chunks_skipped = 0
    chunks_skipped_short = 0
    precision_values: list[float] = []
    recall_values: list[float] = []
    f1_values: list[float] = []
    accuracy_values: list[float] = []
    completeness_values: list[float] = []
    chamfer_values: list[float] = []
    ate_rmse_values: list[float] = []
    recon_downsample_rng = np.random.default_rng()

    chunk_index = 0
    chunk_start_ns = overlap_start_ns
    while chunk_start_ns < overlap_end_ns:
        if recon_is_global_pcd:
            chunk_end_ns = overlap_end_ns + 1
        else:
            chunk_end_ns = find_chunk_end_by_distance(
                gt_traj_entries,
                start_ns=chunk_start_ns,
                overlap_end_ns=overlap_end_ns,
                chunk_distance_m=args.chunk_m,
            )

        chunk_display_end_ns = min(chunk_end_ns, overlap_end_ns)
        gt_begin, gt_end = find_chunk_indices(gt_traj_entries, chunk_start_ns, chunk_end_ns)
        recon_begin, recon_end = find_chunk_indices(recon_entries, chunk_start_ns, chunk_end_ns)
        gt_chunk = gt_traj_entries[gt_begin:gt_end]
        recon_chunk = recon_entries[recon_begin:recon_end]
        matched_pairs = match_chunk_entries(gt_chunk, recon_chunk, tolerance_ns)

        if gt_pcd_entries is not None:
            gt_pcd_begin, gt_pcd_end = find_chunk_indices(gt_pcd_entries, chunk_start_ns, chunk_end_ns)
            gt_chunk_pcd = gt_pcd_entries[gt_pcd_begin:gt_pcd_end]
        else:
            gt_chunk_pcd = None

        chunk_length_m = compute_total_translation(gt_chunk)
        matched_length_m = compute_total_translation([gt_entry for gt_entry, _ in matched_pairs])
        chunk_length_s = (chunk_display_end_ns - chunk_start_ns) / 1_000_000_000.0

        print(f"\nchunk[{chunk_index}] {format_timestamp_ns(chunk_start_ns)} -> {format_timestamp_ns(chunk_display_end_ns)}")
        print(f"chunk_len_s:       {chunk_length_s:.6f}")
        print(f"chunk_len_m:       {chunk_length_m:.6f}")
        print(f"matched_len_m:     {matched_length_m:.6f}")
        print(f"gt_chunk_poses:    {len(gt_chunk)}")
        if gt_chunk_pcd is not None:
            print(f"gt_chunk_pcds:     {len(gt_chunk_pcd)}")
        print(f"recon_chunk_poses: {len(recon_chunk)}")
        print(f"matched_poses:     {len(matched_pairs)}")

        if not recon_is_global_pcd and chunk_length_m < args.chunk_m_min:
            print("skip:              chunk shorter than --chunk-m-min")
            chunks_skipped += 1
            chunks_skipped_short += 1
            chunk_index += 1
            chunk_start_ns = chunk_end_ns
            continue

        if len(matched_pairs) < 3:
            print(f"skip:              need at least 3 matched poses for {args.align_mode.upper()}")
            chunks_skipped += 1
            chunk_index += 1
            chunk_start_ns = chunk_end_ns
            continue

        gt_positions = np.vstack([gt_entry.translation for gt_entry, _ in matched_pairs])
        recon_positions = np.vstack([recon_entry.translation for _, recon_entry in matched_pairs])
        if args.align_mode == "sim3":
            sim3_scale, sim3_rotation, sim3_translation = estimate_sim3(recon_positions, gt_positions)
            print(f"sim3_scale:        {sim3_scale:.6f}")
        else:
            sim3_rotation, sim3_translation = estimate_se3(recon_positions, gt_positions)
            sim3_scale = 1.0

        ate_errors = compute_ate_errors(
            gt_positions=gt_positions,
            recon_positions=recon_positions,
            scale=sim3_scale,
            rotation=sim3_rotation,
            translation=sim3_translation,
        )
        ate_rmse = float(np.sqrt(np.mean(ate_errors ** 2)))
        ate_rmse_values.append(ate_rmse)
        print_metric("ate_rmse_m:", f"{ate_rmse:.6f}")
        print_metric("ate_mean_m:", f"{float(np.mean(ate_errors)):.6f}")
        print_metric("ate_median_m:", f"{float(np.median(ate_errors)):.6f}")
        print_metric("ate_max_m:", f"{float(np.max(ate_errors)):.6f}")

        if gt_is_global_pcd:
            gt_cloud = o3d.geometry.PointCloud(gt_global_cloud)
        else:
            if not gt_chunk_pcd:
                print("skip:              no GT point clouds available in chunk")
                chunks_skipped += 1
                chunk_index += 1
                chunk_start_ns = chunk_end_ns
                continue
            gt_cloud = build_world_cloud_for_entries(gt_chunk_pcd)

        if recon_global_cloud is not None:
            recon_cloud = apply_sim3_to_cloud(
                recon_global_cloud,
                scale=sim3_scale,
                rotation=sim3_rotation,
                translation=sim3_translation,
            )
        else:
            recon_cloud = build_world_cloud_for_entries(
                recon_chunk,
                scale=sim3_scale,
                sim3_rotation=sim3_rotation,
                sim3_translation=sim3_translation,
                random_downsample=args.input_downsample,
                rng=recon_downsample_rng,
            )

        if args.icp:
            icp_distance = max(args.threshold, args.voxel_size)
            gt_icp_cloud = o3d.geometry.PointCloud(gt_cloud) if recon_is_global_pcd else voxel_downsample(gt_cloud, args.voxel_size)
            recon_icp_cloud = voxel_downsample(recon_cloud, args.voxel_size)
            if args.recall and not gt_is_global_pcd:
                _, icp_result = align_source_to_target_icp(
                    source_cloud=gt_icp_cloud,
                    target_cloud=recon_icp_cloud,
                    max_correspondence_distance=icp_distance,
                )
                recon_cloud = transform_cloud(recon_cloud, np.linalg.inv(icp_result.transformation))
            else:
                _, icp_result = align_source_to_target_icp(
                    source_cloud=recon_icp_cloud,
                    target_cloud=gt_icp_cloud,
                    max_correspondence_distance=icp_distance,
                )
                recon_cloud = transform_cloud(recon_cloud, icp_result.transformation)
            print(f"icp_max_corr_m:    {icp_distance:.6f}")
            print(f"icp_fitness:       {icp_result.fitness:.6f}")
            print(f"icp_inlier_rmse:   {icp_result.inlier_rmse:.6f}")

        if crop_global_gt:
            gt_points_before_crop = len(gt_cloud.points)
            gt_cloud = crop_gt_cloud_near_recon(gt_cloud, recon_cloud, gt_crop_margin_m)
            print(f"gt_crop_points:    {gt_points_before_crop} -> {len(gt_cloud.points)}")

        if args.precision:
            gt_eval_cloud = o3d.geometry.PointCloud(gt_cloud)
            recon_eval_cloud = voxel_downsample(recon_cloud, args.voxel_size)
        elif args.recall:
            gt_eval_cloud = voxel_downsample(gt_cloud, args.voxel_size)
            recon_eval_cloud = o3d.geometry.PointCloud(recon_cloud)
        else:
            gt_eval_cloud = voxel_downsample(gt_cloud, args.voxel_size)
            recon_eval_cloud = voxel_downsample(recon_cloud, args.voxel_size)
        print(f"eval_gt_points:    {len(gt_cloud.points)} -> {len(gt_eval_cloud.points)}")
        print(f"eval_recon_points: {len(recon_cloud.points)} -> {len(recon_eval_cloud.points)}")

        metrics, chamfer, _, _ = report_metrics(gt_eval_cloud, recon_eval_cloud, args)
        precision_values.append(metrics.precision)
        recall_values.append(metrics.recall)
        f1_values.append(metrics.f1)
        accuracy_values.append(chamfer.accuracy_m)
        completeness_values.append(chamfer.completeness_m)
        chamfer_values.append(chamfer.chamfer_distance_m)

        chunks_processed += 1
        chunk_index += 1
        if recon_is_global_pcd:
            break
        chunk_start_ns = chunk_end_ns

    print(f"\nchunks_processed:  {chunks_processed}")
    print(f"chunks_skipped:    {chunks_skipped}")

    if chunks_processed > 0 and not recon_is_global_pcd:
        print("\nsummary:")
        if args.precision:
            print_metric_summary("precision:", precision_values, as_percentage=True)
            print_metric_summary("reconstruction error (m):", accuracy_values)
        elif args.recall:
            print_metric_summary("recall:", recall_values, as_percentage=True)
            print_metric_summary("coverage error (m):", completeness_values)
        else:
            print_metric_summary("precision:", precision_values, as_percentage=True)
            print_metric_summary("recall:", recall_values, as_percentage=True)
            print_metric_summary("f1:", f1_values)
            print_metric_summary("reconstruction error (m):", accuracy_values)
            print_metric_summary("coverage error (m):", completeness_values)
            print_metric_summary("chamfer_m:", chamfer_values)


if __name__ == "__main__":
    main()
