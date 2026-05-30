import math
import os
from dataclasses import dataclass
from typing import Dict, List, Mapping, Optional, Sequence, Set, Tuple

import cv2
import numpy as np

from scarf_slam.core.pose import MappingPose, MappingTransforms
from scarf_slam.mapping import feature_matching


MAX_UNBOUNDED_DISTANCE_CANDIDATES = 10


@dataclass(frozen=True)
class RegisterSubmapSummary:
    submap_key: str
    num_frames: int
    new_frame_edges: int
    cross_submap_neighbors: List[str]


@dataclass(frozen=True)
class RebuildSummary:
    num_frames: int
    frame_edges: int
    cross_submap_links: int


class FrameSubmapCovisibilityGraph:
    def __init__(
        self,
        transforms: MappingTransforms,
        enabled: bool = True,
        max_distance: float = 0.75,
        max_angle_deg: float = 20.0,
        min_time_separation_sec: float = 10.0,
        max_old_candidates_per_new_frame: int = 5,
        max_old_edges_per_new_frame: int = 5,
        matcher_name: str = "superpoint-lightglue",
        matcher_device: str = "cpu",
        max_num_keypoints: int = 1024,
        ransac_reproj_thresh: float = 3.0,
        min_inlier_matches: int = 20,
        recent_gpu_feature_cache_size: int = 16,
    ) -> None:
        self.transforms = transforms
        self.submap_frame_keys_dict: Dict[str, List[str]] = {}
        self._submap_registration_index: Dict[str, int] = {}
        self.frame_to_submaps_dict: Dict[str, Set[str]] = {}
        self.frame_covisibility_graph: Dict[str, Set[str]] = {}
        self.submap_covisibility_graph: Dict[str, Set[str]] = {}
        self.submap_covisibility_frame_pairs: Dict[Tuple[str, str], List[Tuple[str, str]]] = {}
        self._frame_positions_dict: Dict[str, np.ndarray] = {}
        self._frame_view_dirs_dict: Dict[str, np.ndarray] = {}
        self._frame_spatial_cell_dict: Dict[str, Tuple[int, int, int]] = {}
        self._spatial_cell_to_frame_keys: Dict[Tuple[int, int, int], Set[str]] = {}
        self._spatial_cell_size = float(max_distance)
        self._frame_image_dict: Dict[str, np.ndarray] = {}
        self._frame_intrinsics_dict: Dict[str, np.ndarray] = {}
        self._frame_conf_dict: Dict[str, np.ndarray] = {}
        self._frame_matcher_feature_dict: Dict[str, Dict[str, object]] = {}
        self._frame_matcher_feature_storage_device_dict: Dict[str, str] = {}
        self._frame_match_count_dict: Dict[Tuple[str, str], int] = {}
        self._frame_match_result_dict: Dict[Tuple[str, str], Dict[str, np.ndarray]] = {}
        self.configure(
            enabled=enabled,
            max_distance=max_distance,
            max_angle_deg=max_angle_deg,
            min_time_separation_sec=min_time_separation_sec,
            max_old_candidates_per_new_frame=max_old_candidates_per_new_frame,
            max_old_edges_per_new_frame=max_old_edges_per_new_frame,
            matcher_name=matcher_name,
            matcher_device=matcher_device,
            max_num_keypoints=max_num_keypoints,
            ransac_reproj_thresh=ransac_reproj_thresh,
            min_inlier_matches=min_inlier_matches,
            recent_gpu_feature_cache_size=recent_gpu_feature_cache_size,
        )

    def configure(
        self,
        *,
        enabled: bool,
        max_distance: float,
        max_angle_deg: float,
        min_time_separation_sec: float,
        max_old_candidates_per_new_frame: int,
        max_old_edges_per_new_frame: int,
        matcher_name: str,
        matcher_device: str,
        max_num_keypoints: int,
        ransac_reproj_thresh: float,
        min_inlier_matches: int,
        recent_gpu_feature_cache_size: int,
    ) -> None:
        if max_distance <= 0.0:
            raise ValueError("frame_covisibility_max_distance must be positive.")
        if not (0.0 <= max_angle_deg <= 180.0):
            raise ValueError("frame_covisibility_max_angle_deg must be in [0, 180].")
        if min_time_separation_sec < 0.0:
            raise ValueError("frame_covisibility_min_time_separation_sec must be non-negative.")
        if max_old_candidates_per_new_frame < 0:
            raise ValueError("frame_covisibility_max_old_candidates_per_new_frame must be non-negative.")
        if max_old_edges_per_new_frame < 0:
            raise ValueError("frame_covisibility_max_old_edges_per_new_frame must be non-negative.")
        if max_num_keypoints <= 0:
            raise ValueError("frame_covisibility_max_num_keypoints must be positive.")
        if ransac_reproj_thresh <= 0.0:
            raise ValueError("frame_covisibility_ransac_reproj_thresh must be positive.")
        if min_inlier_matches < 0:
            raise ValueError("frame_covisibility_min_inlier_matches must be non-negative.")
        if recent_gpu_feature_cache_size < 0:
            raise ValueError("recent_gpu_feature_cache_size must be non-negative.")

        prev_matcher_signature = (
            getattr(self, "matcher_name", None),
            getattr(self, "matcher_device", None),
            getattr(self, "max_num_keypoints", None),
        )

        self.enabled = bool(enabled)
        self.max_distance = float(max_distance)
        self.max_angle_deg = float(max_angle_deg)
        self.min_view_dot = math.cos(np.deg2rad(self.max_angle_deg))
        self.min_time_separation_sec = float(min_time_separation_sec)
        self._spatial_cell_size = max(float(max_distance), 1e-6)
        self.max_old_candidates_per_new_frame = int(max_old_candidates_per_new_frame)
        self.max_old_edges_per_new_frame = int(max_old_edges_per_new_frame)
        self.matcher_name = str(matcher_name)
        self.matcher_device = str(matcher_device)
        self.max_num_keypoints = int(max_num_keypoints)
        self.ransac_reproj_thresh = float(ransac_reproj_thresh)
        self.min_inlier_matches = int(min_inlier_matches)
        self.recent_gpu_feature_cache_size = int(recent_gpu_feature_cache_size)

        if prev_matcher_signature != (self.matcher_name, self.matcher_device, self.max_num_keypoints):
            self._frame_matcher_feature_dict.clear()
            self._frame_matcher_feature_storage_device_dict.clear()

        if len(self._frame_positions_dict) > 0:
            self._clear_spatial_index()
            for frame_key in self._frame_positions_dict:
                self._index_frame_spatial(frame_key)
        if len(self._frame_matcher_feature_dict) > 0:
            self._rebalance_matcher_feature_cache_storage()

    def cache_frame_inputs(
        self,
        frame_keys: Sequence[str],
        processed_images: np.ndarray,
        intrinsics: np.ndarray,
        confidences: Optional[np.ndarray] = None,
        matcher_feature_cache: Optional[Sequence[Dict[str, object]]] = None,
        matcher_feature_cache_meta: Optional[Dict[str, object]] = None,
    ) -> None:
        if len(frame_keys) != int(processed_images.shape[0]) or len(frame_keys) != int(intrinsics.shape[0]):
            raise ValueError(
                "Frame covisibility cache inputs must have matching batch dimension: "
                f"n_keys={len(frame_keys)}, images={processed_images.shape}, intrinsics={intrinsics.shape}."
            )
        
        if confidences is not None and len(frame_keys) != int(confidences.shape[0]):
            raise ValueError(
                "Frame covisibility confidence batch dimension must match: "
                f"n_keys={len(frame_keys)}, confidences={confidences.shape}."
            )
        if matcher_feature_cache is not None and len(frame_keys) != len(matcher_feature_cache):
            raise ValueError(
                "Frame covisibility matcher feature cache batch dimension must match: "
                f"n_keys={len(frame_keys)}, feature_cache={len(matcher_feature_cache)}."
            )

        matcher_features_compatible = False
        if matcher_feature_cache is not None and matcher_feature_cache_meta is not None:
            matcher_features_compatible = (
                str(matcher_feature_cache_meta.get("matcher_name")) == self.matcher_name
                and str(matcher_feature_cache_meta.get("matcher_device")) == self.matcher_device
                and int(matcher_feature_cache_meta.get("max_num_keypoints")) == self.max_num_keypoints
            )

        for idx, frame_key in enumerate(frame_keys):
            img = np.asarray(processed_images[idx])
            if img.ndim == 3:
                gray = cv2.cvtColor(img.astype(np.uint8, copy=False), cv2.COLOR_RGB2GRAY)
            elif img.ndim == 2:
                gray = img.astype(np.uint8, copy=False)
            else:
                raise ValueError(f"Unsupported processed image shape for frame '{frame_key}': {img.shape}")
            self._frame_image_dict[frame_key] = np.ascontiguousarray(gray)
            self._frame_intrinsics_dict[frame_key] = np.asarray(intrinsics[idx], dtype=np.float32).copy()
            
            if confidences is not None:
                conf = np.asarray(confidences[idx], dtype=np.float32).copy()
                self._frame_conf_dict[frame_key] = conf
            if matcher_features_compatible:
                self._frame_matcher_feature_dict[frame_key] = dict(matcher_feature_cache[idx])
                self._frame_matcher_feature_storage_device_dict[frame_key] = self.matcher_device
        if matcher_features_compatible:
            self._rebalance_matcher_feature_cache_storage()

    def verify_frame_pair(self, frame_key_a: str, frame_key_b: str) -> bool:
        return self.get_frame_pair_match_count(frame_key_a, frame_key_b) >= self.min_inlier_matches

    def get_frame_pair_match_result(self, frame_key_a: str, frame_key_b: str) -> Dict[str, np.ndarray]:
        pair_key = (frame_key_a, frame_key_b) if frame_key_a <= frame_key_b else (frame_key_b, frame_key_a)
        canonical_frame_key_a, canonical_frame_key_b = pair_key
        cached_match_result = self._frame_match_result_dict.get(pair_key)
        if cached_match_result is None:
            img_a = self._frame_image_dict.get(canonical_frame_key_a)
            img_b = self._frame_image_dict.get(canonical_frame_key_b)
            k_a = self._frame_intrinsics_dict.get(canonical_frame_key_a)
            k_b = self._frame_intrinsics_dict.get(canonical_frame_key_b)
            conf_a = self._frame_conf_dict.get(canonical_frame_key_a)
            conf_b = self._frame_conf_dict.get(canonical_frame_key_b)
            precomputed_feature_a = self._frame_matcher_feature_dict.get(canonical_frame_key_a)
            precomputed_feature_b = self._frame_matcher_feature_dict.get(canonical_frame_key_b)
            if img_a is None or img_b is None or k_a is None or k_b is None:
                empty_result = {
                    "num_raw_matches": 0,
                    "num_inlier_matches": 0,
                    "matched_points0": np.zeros((0, 2), dtype=np.float32),
                    "matched_points1": np.zeros((0, 2), dtype=np.float32),
                }
                self._frame_match_result_dict[pair_key] = empty_result
                self._frame_match_count_dict[pair_key] = 0
                return empty_result

            match_result = feature_matching.verify_frame_pair_match_dl(
                image_0=img_a,
                image_1=img_b,
                intrinsics_0=k_a,
                intrinsics_1=k_b,
                device=self.matcher_device,
                matcher_name=self.matcher_name,
                max_num_keypoints=self.max_num_keypoints,
                ransac_reproj_thresh=self.ransac_reproj_thresh,
                use_inlier_matches=True,
                conf_0=conf_a,
                conf_1=conf_b,
                precomputed_feature_0=precomputed_feature_a,
                precomputed_feature_1=precomputed_feature_b,
            )
            cached_match_result = {
                "num_raw_matches": int(match_result["num_raw_matches"]),
                "num_inlier_matches": int(match_result["num_inlier_matches"]),
                "matched_points0": np.asarray(match_result["matched_points0"], dtype=np.float32),
                "matched_points1": np.asarray(match_result["matched_points1"], dtype=np.float32),
            }
            self._frame_match_result_dict[pair_key] = cached_match_result
            self._frame_match_count_dict[pair_key] = int(cached_match_result["num_inlier_matches"])

        if frame_key_a == canonical_frame_key_a and frame_key_b == canonical_frame_key_b:
            return cached_match_result

        return {
            "num_raw_matches": int(cached_match_result["num_raw_matches"]),
            "num_inlier_matches": int(cached_match_result["num_inlier_matches"]),
            "matched_points0": np.asarray(cached_match_result["matched_points1"], dtype=np.float32),
            "matched_points1": np.asarray(cached_match_result["matched_points0"], dtype=np.float32),
        }

    def get_frame_pair_match_count(self, frame_key_a: str, frame_key_b: str) -> int:
        pair_key = (frame_key_a, frame_key_b) if frame_key_a <= frame_key_b else (frame_key_b, frame_key_a)
        cached_match_count = self._frame_match_count_dict.get(pair_key)
        if cached_match_count is None:
            cached_match_count = int(self.get_frame_pair_match_result(frame_key_a, frame_key_b)["num_inlier_matches"])

        return int(cached_match_count)

    def get_cross_submap_covisible_frame_pairs(
        self,
        selected_submap_keys: Optional[Set[str]] = None,
    ) -> Dict[Tuple[str, str], List[Tuple[str, str]]]:
        cross_pairs: Dict[Tuple[str, str], List[Tuple[str, str]]] = {}
        for (submap_key_a, submap_key_b), frame_pairs in sorted(self.submap_covisibility_frame_pairs.items()):
            if submap_key_a == submap_key_b:
                continue
            if selected_submap_keys is not None:
                if submap_key_a not in selected_submap_keys or submap_key_b not in selected_submap_keys:
                    continue

            unique_frame_pairs: List[Tuple[str, str]] = []
            seen_frame_pairs: Set[Tuple[str, str]] = set()
            for frame_key_a, frame_key_b in frame_pairs:
                if frame_key_a == frame_key_b:
                    continue

                frame_submaps_a = self.frame_to_submaps_dict.get(frame_key_a, set())
                frame_submaps_b = self.frame_to_submaps_dict.get(frame_key_b, set())
                if submap_key_a not in frame_submaps_a:
                    raise AssertionError(
                        f"Frame '{frame_key_a}' is not registered under submap '{submap_key_a}'."
                    )
                if submap_key_b not in frame_submaps_b:
                    raise AssertionError(
                        f"Frame '{frame_key_b}' is not registered under submap '{submap_key_b}'."
                    )

                frame_pair_key = (frame_key_a, frame_key_b)
                if frame_pair_key in seen_frame_pairs:
                    continue
                seen_frame_pairs.add(frame_pair_key)
                unique_frame_pairs.append(frame_pair_key)

            if unique_frame_pairs:
                cross_pairs[(submap_key_a, submap_key_b)] = unique_frame_pairs

        return cross_pairs

    @staticmethod
    def _timestamp_to_seconds(ts_key: str) -> float:
        sec_str, nsec_str = ts_key.split("_")
        return float(int(sec_str)) + float(int(nsec_str)) * 1e-9

    def _timestamp_gap_large_enough(self, frame_key_a: str, frame_key_b: str) -> bool:
        return abs(self._timestamp_to_seconds(frame_key_a) - self._timestamp_to_seconds(frame_key_b)) > self.min_time_separation_sec

    @staticmethod
    def _normalize_vector(vec: np.ndarray) -> np.ndarray:
        arr = np.asarray(vec, dtype=np.float64).reshape(3)
        norm = np.linalg.norm(arr)
        if norm <= 1e-12:
            return np.zeros((3,), dtype=np.float64)
        return arr / norm

    def _pose_view_direction(self, pose: MappingPose) -> np.ndarray:
        pose_matrix = self.transforms.pose_to_matrix(pose)
        forward_world = pose_matrix[:3, :3] @ np.array([0.0, 0.0, 1.0], dtype=np.float64)
        return self._normalize_vector(forward_world)

    def _cache_frame_pose_features(
        self,
        frame_key: str,
        pose_dict: Mapping[str, MappingPose],
    ) -> None:
        if frame_key not in pose_dict:
            raise KeyError(f"Missing pose for frame '{frame_key}' while caching covisibility features.")

        pose = pose_dict[frame_key]
        self._frame_positions_dict[frame_key] = np.asarray(pose.pos, dtype=np.float64).reshape(3)
        self._frame_view_dirs_dict[frame_key] = self._pose_view_direction(pose)
        self._index_frame_spatial(frame_key)

    def _position_to_spatial_cell(self, position: np.ndarray) -> Tuple[int, int, int]:
        pos = np.asarray(position, dtype=np.float64).reshape(3)
        return tuple(np.floor(pos / self._spatial_cell_size).astype(np.int64).tolist())

    def _clear_spatial_index(self) -> None:
        self._frame_spatial_cell_dict.clear()
        self._spatial_cell_to_frame_keys.clear()

    def _set_frame_matcher_feature_storage_device(self, frame_key: str, target_device: str) -> None:
        if frame_key not in self._frame_matcher_feature_dict:
            return
        current_device = self._frame_matcher_feature_storage_device_dict.get(frame_key)
        if current_device == target_device:
            return
        frame_feature = self._frame_matcher_feature_dict[frame_key]
        if target_device == "cpu":
            self._frame_matcher_feature_dict[frame_key] = feature_matching.offload_vismatch_frame_feature_to_cpu(
                frame_feature
            )
        else:
            self._frame_matcher_feature_dict[frame_key] = feature_matching.prepare_vismatch_frame_feature_for_device(
                frame_feature,
                target_device,
            )
        self._frame_matcher_feature_storage_device_dict[frame_key] = target_device

    def _rebalance_matcher_feature_cache_storage(self) -> None:
        if len(self._frame_matcher_feature_dict) == 0:
            return

        if self.recent_gpu_feature_cache_size <= 0:
            gpu_frame_keys: Set[str] = set()
        else:
            sorted_frame_keys = sorted(self._frame_matcher_feature_dict.keys())
            gpu_frame_keys = set(sorted_frame_keys[-self.recent_gpu_feature_cache_size :])

        for frame_key in list(self._frame_matcher_feature_dict.keys()):
            target_device = self.matcher_device if frame_key in gpu_frame_keys else "cpu"
            self._set_frame_matcher_feature_storage_device(frame_key, target_device)

    def _index_frame_spatial(self, frame_key: str) -> None:
        if frame_key not in self._frame_positions_dict:
            raise KeyError(f"Missing cached position for frame '{frame_key}' while indexing spatial candidates.")

        new_cell = self._position_to_spatial_cell(self._frame_positions_dict[frame_key])
        old_cell = self._frame_spatial_cell_dict.get(frame_key)
        if old_cell == new_cell:
            return

        if old_cell is not None:
            old_bucket = self._spatial_cell_to_frame_keys.get(old_cell)
            if old_bucket is not None:
                old_bucket.discard(frame_key)
                if len(old_bucket) == 0:
                    self._spatial_cell_to_frame_keys.pop(old_cell, None)

        self._frame_spatial_cell_dict[frame_key] = new_cell
        self._spatial_cell_to_frame_keys.setdefault(new_cell, set()).add(frame_key)

    def _collect_spatial_neighbor_frame_keys(self, frame_key: str) -> Set[str]:
        if frame_key not in self._frame_positions_dict:
            raise KeyError(f"Missing cached position for frame '{frame_key}' while querying spatial candidates.")

        center_cell = self._frame_spatial_cell_dict.get(frame_key)
        if center_cell is None:
            self._index_frame_spatial(frame_key)
            center_cell = self._frame_spatial_cell_dict[frame_key]

        cell_radius = max(1, int(math.ceil(self.max_distance / self._spatial_cell_size)))
        neighbor_frame_keys: Set[str] = set()
        for dx in range(-cell_radius, cell_radius + 1):
            for dy in range(-cell_radius, cell_radius + 1):
                for dz in range(-cell_radius, cell_radius + 1):
                    neighbor_cell = (
                        center_cell[0] + dx,
                        center_cell[1] + dy,
                        center_cell[2] + dz,
                    )
                    neighbor_frame_keys.update(self._spatial_cell_to_frame_keys.get(neighbor_cell, set()))
        neighbor_frame_keys.discard(frame_key)
        return neighbor_frame_keys

    def _frame_pair_is_covisible(
        self,
        frame_key_a: str,
        frame_key_b: str,
        pose_dict: Mapping[str, MappingPose],
    ) -> bool:
        if frame_key_a == frame_key_b:
            return False
        if not self._timestamp_gap_large_enough(frame_key_a, frame_key_b):
            return False

        if frame_key_a not in self._frame_positions_dict:
            self._cache_frame_pose_features(frame_key_a, pose_dict)
        if frame_key_b not in self._frame_positions_dict:
            self._cache_frame_pose_features(frame_key_b, pose_dict)

        pos_a = self._frame_positions_dict[frame_key_a]
        pos_b = self._frame_positions_dict[frame_key_b]
        if np.linalg.norm(pos_a - pos_b) > self.max_distance:
            return False

        view_a = self._frame_view_dirs_dict[frame_key_a]
        view_b = self._frame_view_dirs_dict[frame_key_b]
        if float(np.dot(view_a, view_b)) < self.min_view_dot:
            return False

        return True

    def _latest_submap_for_frame(self, frame_key: str) -> str:
        submap_keys = self.frame_to_submaps_dict.get(frame_key)
        if not submap_keys:
            raise KeyError(f"Missing frame-to-submap assignment for frame '{frame_key}'.")
        return max(
            submap_keys,
            key=lambda submap_key: (
                self._submap_registration_index.get(submap_key, -1),
                submap_key,
            ),
        )

    def _frame_pair_covisibility_submap_keys(
        self,
        frame_key_a: str,
        frame_key_b: str,
    ) -> Optional[Tuple[str, str]]:
        if frame_key_a == frame_key_b:
            return None

        submap_keys_a = self.frame_to_submaps_dict.get(frame_key_a)
        submap_keys_b = self.frame_to_submaps_dict.get(frame_key_b)
        if submap_keys_a is None or submap_keys_b is None:
            raise KeyError(
                f"Missing frame-to-submap assignment for covisible frames: {frame_key_a}, {frame_key_b}"
            )
        if submap_keys_a & submap_keys_b:
            return None

        submap_key_a = self._latest_submap_for_frame(frame_key_a)
        submap_key_b = self._latest_submap_for_frame(frame_key_b)
        if submap_key_a == submap_key_b:
            return None
        return submap_key_a, submap_key_b

    def _add_frame_covisibility_edge(self, frame_key_a: str, frame_key_b: str) -> bool:
        submap_pair = self._frame_pair_covisibility_submap_keys(frame_key_a, frame_key_b)
        if submap_pair is None:
            return False
        submap_key_a, submap_key_b = submap_pair
        return self._add_frame_covisibility_edge_for_submaps(
            submap_key_a,
            submap_key_b,
            frame_key_a,
            frame_key_b,
        )

    def _add_frame_covisibility_edge_for_submaps(
        self,
        submap_key_a: str,
        submap_key_b: str,
        frame_key_a: str,
        frame_key_b: str,
    ) -> bool:
        if frame_key_a == frame_key_b or submap_key_a == submap_key_b:
            return False
        self.frame_covisibility_graph.setdefault(frame_key_a, set())
        self.frame_covisibility_graph.setdefault(frame_key_b, set())
        if frame_key_b in self.frame_covisibility_graph[frame_key_a]:
            return False
        if not self._add_submap_link(submap_key_a, submap_key_b, frame_key_a, frame_key_b):
            return False

        self.frame_covisibility_graph[frame_key_a].add(frame_key_b)
        self.frame_covisibility_graph[frame_key_b].add(frame_key_a)
        return True

    def _add_submap_link(
        self,
        submap_key_a: str,
        submap_key_b: str,
        frame_key_a: str,
        frame_key_b: str,
    ) -> bool:
        self.submap_covisibility_graph.setdefault(submap_key_a, set())
        self.submap_covisibility_graph.setdefault(submap_key_b, set())
        if submap_key_a == submap_key_b:
            return False

        self.submap_covisibility_graph[submap_key_a].add(submap_key_b)
        self.submap_covisibility_graph[submap_key_b].add(submap_key_a)

        if submap_key_a <= submap_key_b:
            pair_key = (submap_key_a, submap_key_b)
            frame_pair = (frame_key_a, frame_key_b)
        else:
            pair_key = (submap_key_b, submap_key_a)
            frame_pair = (frame_key_b, frame_key_a)
        self.submap_covisibility_frame_pairs.setdefault(pair_key, []).append(frame_pair)
        return True

    def _reset_graph_edges(self) -> None:
        self._frame_positions_dict.clear()
        self._frame_view_dirs_dict.clear()
        self._clear_spatial_index()

        self.frame_covisibility_graph.clear()
        for frame_key in self.frame_to_submaps_dict:
            self.frame_covisibility_graph[frame_key] = set()

        self.submap_covisibility_graph.clear()
        for submap_key in self.submap_frame_keys_dict:
            self.submap_covisibility_graph[submap_key] = set()

        self.submap_covisibility_frame_pairs.clear()

    def refresh_pose_features(self, pose_dict: Mapping[str, MappingPose]) -> RebuildSummary:
        self._frame_positions_dict.clear()
        self._frame_view_dirs_dict.clear()
        self._clear_spatial_index()

        for frame_key in sorted(self.frame_to_submaps_dict.keys()):
            self._cache_frame_pose_features(frame_key, pose_dict)

        frame_edges = sum(len(neighbors) for neighbors in self.frame_covisibility_graph.values()) // 2
        return RebuildSummary(
            num_frames=len(self.frame_to_submaps_dict),
            frame_edges=frame_edges,
            cross_submap_links=len(self.submap_covisibility_frame_pairs),
        )

    def _collect_ranked_old_frame_candidates(
        self,
        frame_key: str,
        old_frame_keys: Sequence[str],
        pose_dict: Mapping[str, MappingPose],
    ) -> List[Tuple[float, str]]:
        candidates: List[Tuple[float, str]] = []
        if self.max_old_candidates_per_new_frame == 0:
            return candidates

        old_frame_key_set = set(old_frame_keys)
        pos_a = np.asarray(pose_dict[frame_key].pos, dtype=np.float64).reshape(3)

        def _distance_to_frame(old_frame_key: str) -> float:
            pos_b = np.asarray(pose_dict[old_frame_key].pos, dtype=np.float64).reshape(3)
            return float(np.linalg.norm(pos_a - pos_b))

        if not math.isfinite(self.max_distance):
            candidate_frame_keys = (
                old_frame_key_set
                if len(old_frame_key_set) > 0
                else set(self._frame_positions_dict.keys())
            )
            candidate_frame_keys.discard(frame_key)
            candidate_frame_keys = sorted(
                candidate_frame_keys,
                key=lambda old_frame_key: (_distance_to_frame(old_frame_key), old_frame_key),
            )[:MAX_UNBOUNDED_DISTANCE_CANDIDATES]
        else:
            spatial_neighbor_frame_keys = self._collect_spatial_neighbor_frame_keys(frame_key)
            if len(old_frame_key_set) > 0:
                candidate_frame_keys = spatial_neighbor_frame_keys & old_frame_key_set
            else:
                candidate_frame_keys = spatial_neighbor_frame_keys

        for old_frame_key in candidate_frame_keys:
            if not self._frame_pair_is_covisible(frame_key, old_frame_key, pose_dict):
                continue
            candidates.append((_distance_to_frame(old_frame_key), old_frame_key))

        if not math.isfinite(self.max_distance):
            candidates.sort(key=lambda item: (item[0], item[1]))
        else:
            frame_time_sec = self._timestamp_to_seconds(frame_key)
            candidates.sort(
                key=lambda item: (
                    -abs(frame_time_sec - self._timestamp_to_seconds(item[1])),
                    item[0],
                    item[1],
                )
            )
        return candidates[: self.max_old_candidates_per_new_frame]

    def register_submap_frames(
        self,
        submap_key: str,
        frame_keys: Sequence[str],
        pose_dict: Mapping[str, MappingPose],
    ) -> RegisterSubmapSummary:
        frame_keys_list = list(frame_keys)
        if len(frame_keys_list) == 0:
            raise ValueError(f"Cannot register empty frame list for submap '{submap_key}'.")

        existing_frame_keys = self.submap_frame_keys_dict.get(submap_key)
        if existing_frame_keys is not None:
            if list(existing_frame_keys) == frame_keys_list:
                return RegisterSubmapSummary(
                    submap_key=submap_key,
                    num_frames=len(frame_keys_list),
                    new_frame_edges=0,
                    cross_submap_neighbors=sorted(self.submap_covisibility_graph.get(submap_key, set())),
                )
            raise ValueError(
                f"Submap '{submap_key}' already registered with different frames: "
                f"existing={existing_frame_keys}, new={frame_keys_list}"
            )

        self.submap_frame_keys_dict[submap_key] = frame_keys_list
        self._submap_registration_index.setdefault(submap_key, len(self._submap_registration_index))
        self.submap_covisibility_graph.setdefault(submap_key, set())

        new_frame_edges = 0
        linked_submaps: Set[str] = set()
        frame_keys_set = set(frame_keys_list)
        for frame_key in frame_keys_list:
            frame_new_edges = 0
            old_frame_keys = sorted(k for k in self.frame_to_submaps_dict.keys() if k not in frame_keys_set)
            prev_submap_keys = self.frame_to_submaps_dict.setdefault(frame_key, set())
            if submap_key in prev_submap_keys:
                continue
            for prev_submap_key in sorted(prev_submap_keys):
                self._add_submap_link(prev_submap_key, submap_key, frame_key, frame_key)
                if prev_submap_key != submap_key:
                    linked_submaps.add(prev_submap_key)
            prev_submap_keys.add(submap_key)
            self.frame_covisibility_graph.setdefault(frame_key, set())
            self._cache_frame_pose_features(frame_key, pose_dict)

            if not self.enabled:
                continue

            ranked_candidates = self._collect_ranked_old_frame_candidates(
                frame_key=frame_key,
                old_frame_keys=old_frame_keys,
                pose_dict=pose_dict,
            )
            for _, old_frame_key in ranked_candidates:
                old_submap_key = self._latest_submap_for_frame(old_frame_key)
                if old_submap_key == submap_key:
                    continue
                if not self.verify_frame_pair(frame_key, old_frame_key):
                    continue
                if not self._add_frame_covisibility_edge_for_submaps(
                    submap_key,
                    old_submap_key,
                    frame_key,
                    old_frame_key,
                ):
                    continue
                frame_new_edges += 1
                new_frame_edges += 1
                other_submap_keys = self.frame_to_submaps_dict.get(old_frame_key, set())
                for other_submap_key in other_submap_keys:
                    if other_submap_key != submap_key:
                        linked_submaps.add(other_submap_key)
                if frame_new_edges >= self.max_old_edges_per_new_frame:
                    break

        linked_submaps.update(self.submap_covisibility_graph.get(submap_key, set()))

        return RegisterSubmapSummary(
            submap_key=submap_key,
            num_frames=len(frame_keys_list),
            new_frame_edges=new_frame_edges,
            cross_submap_neighbors=sorted(linked_submaps),
        )

    def rebuild(
        self,
        pose_dict: Mapping[str, MappingPose],
    ) -> RebuildSummary:
        self._reset_graph_edges()
        num_frames = len(self.frame_to_submaps_dict)
        if not self.enabled or num_frames < 2:
            return RebuildSummary(
                num_frames=num_frames,
                frame_edges=0,
                cross_submap_links=0,
            )

        for frame_key, submap_keys in self.frame_to_submaps_dict.items():
            if len(submap_keys) < 2:
                continue
            sorted_submap_keys = sorted(submap_keys)
            for i in range(len(sorted_submap_keys) - 1):
                for j in range(i + 1, len(sorted_submap_keys)):
                    self._add_submap_link(
                        sorted_submap_keys[i],
                        sorted_submap_keys[j],
                        frame_key,
                        frame_key,
                    )

        self.refresh_pose_features(pose_dict)

        new_frame_edges = 0
        registered_frame_keys: Set[str] = set()
        ordered_submap_keys = sorted(
            self.submap_frame_keys_dict.keys(),
            key=lambda key: (self._submap_registration_index.get(key, -1), key),
        )
        for submap_key in ordered_submap_keys:
            current_frame_keys = list(self.submap_frame_keys_dict[submap_key])
            current_frame_key_set = set(current_frame_keys)
            old_frame_keys = sorted(registered_frame_keys - current_frame_key_set)

            for frame_key in current_frame_keys:
                frame_new_edges = 0
                ranked_candidates = self._collect_ranked_old_frame_candidates(
                    frame_key=frame_key,
                    old_frame_keys=old_frame_keys,
                    pose_dict=pose_dict,
                )
                for _, old_frame_key in ranked_candidates:
                    old_submap_key = self._latest_submap_for_frame(old_frame_key)
                    if old_submap_key == submap_key:
                        continue
                    if not self.verify_frame_pair(frame_key, old_frame_key):
                        continue
                    if not self._add_frame_covisibility_edge_for_submaps(
                        submap_key,
                        old_submap_key,
                        frame_key,
                        old_frame_key,
                    ):
                        continue
                    frame_new_edges += 1
                    new_frame_edges += 1
                    if frame_new_edges >= self.max_old_edges_per_new_frame:
                        break

            registered_frame_keys.update(current_frame_keys)

        return RebuildSummary(
            num_frames=num_frames,
            frame_edges=new_frame_edges,
            cross_submap_links=len(self.submap_covisibility_frame_pairs),
        )

    def visualize_covisible_edge_matches(
        self,
        frame_key_a: str,
        frame_key_b: str,
        match_result: Dict,
        output_dir: str,
        scale_factor: float = 2.0,
    ) -> str:
        """
        Visualize feature matching between two covisible frames with zero-confidence regions highlighted.
        
        Args:
            frame_key_a: First frame timestamp key
            frame_key_b: Second frame timestamp key
            match_result: Dict with 'matched_points0', 'matched_points1', 'num_inlier_matches'
            output_dir: Directory to save visualization
            scale_factor: Scale factor for higher resolution (default 2.0x)
        
        Returns:
            Path to saved visualization image
        """
        def _overlay_zero_conf_red(
            img_rgb: np.ndarray, 
            conf_map: Optional[np.ndarray], 
            alpha: float = 0.35
        ) -> np.ndarray:
            """Overlay zero-confidence regions with red tint."""
            if conf_map is None:
                return img_rgb
            out = img_rgb.copy()
            h, w = out.shape[:2]
            if conf_map.shape[:2] != (h, w):
                conf_map = cv2.resize(conf_map.astype(np.float32), (w, h), interpolation=cv2.INTER_NEAREST)
            zero_mask = (conf_map == 0)
            if not np.any(zero_mask):
                return out
            red = np.zeros_like(out)
            red[..., 0] = 255  # RGB red
            out_f = out.astype(np.float32)
            red_f = red.astype(np.float32)
            out_f[zero_mask] = (1.0 - alpha) * out_f[zero_mask] + alpha * red_f[zero_mask]
            return np.clip(out_f, 0, 255).astype(np.uint8)
        
        img_a = self._frame_image_dict.get(frame_key_a)
        img_b = self._frame_image_dict.get(frame_key_b)
        conf_a = self._frame_conf_dict.get(frame_key_a)
        conf_b = self._frame_conf_dict.get(frame_key_b)
        
        if img_a is None or img_b is None:
            return ""
        
        # Convert grayscale to color for visualization
        if img_a.ndim == 2:
            img_a_color = cv2.cvtColor(img_a, cv2.COLOR_GRAY2BGR)
        else:
            img_a_color = img_a if img_a.shape[2] == 3 else cv2.cvtColor(img_a, cv2.COLOR_GRAY2BGR)
        
        if img_b.ndim == 2:
            img_b_color = cv2.cvtColor(img_b, cv2.COLOR_GRAY2BGR)
        else:
            img_b_color = img_b if img_b.shape[2] == 3 else cv2.cvtColor(img_b, cv2.COLOR_GRAY2BGR)
        
        # Overlay zero-confidence regions with red
        img_a_color = _overlay_zero_conf_red(img_a_color, conf_a, alpha=0.35)
        img_b_color = _overlay_zero_conf_red(img_b_color, conf_b, alpha=0.35)
        
        # Create matched keypoints
        matched_points_0 = match_result.get("matched_points0", np.zeros((0, 2), dtype=np.float32))
        matched_points_1 = match_result.get("matched_points1", np.zeros((0, 2), dtype=np.float32))
        num_matches = int(matched_points_0.shape[0])
        
        h_a, w_a = img_a_color.shape[:2]
        h_b, w_b = img_b_color.shape[:2]
        h_total = int((max(h_a, h_b) + 50) * scale_factor)  # Add space for text at top
        w_total = int((w_a + w_b + 10) * scale_factor)
        
        # Create white background canvas
        canvas = np.ones((h_total, w_total, 3), dtype=np.uint8) * 255
        
        # Scale images and place on canvas
        img_a_scaled = cv2.resize(img_a_color, (int(w_a * scale_factor), int(h_a * scale_factor)))
        img_b_scaled = cv2.resize(img_b_color, (int(w_b * scale_factor), int(h_b * scale_factor)))
        
        y_offset = int(50 * scale_factor)
        canvas[y_offset:y_offset+img_a_scaled.shape[0], 0:img_a_scaled.shape[1]] = img_a_scaled
        canvas[y_offset:y_offset+img_b_scaled.shape[0], int((w_a+10)*scale_factor):int((w_a+10)*scale_factor)+img_b_scaled.shape[1]] = img_b_scaled
        
        if num_matches > 0:
            # Define color palette for match lines (different colors)
            colors = [
                (255, 0, 0),      # Blue
                (0, 255, 0),      # Green
                (0, 0, 255),      # Red
                (255, 255, 0),    # Cyan
                (255, 0, 255),    # Magenta
                (0, 255, 255),    # Yellow
                (128, 0, 255),    # Purple
                (255, 128, 0),    # Orange
            ]
            
            # Draw match lines and circles
            line_thickness = max(1, int(scale_factor * 0.5))  # Scale line thickness
            circle_radius = max(2, int(scale_factor * 1.5))
            circle_thickness = max(1, int(scale_factor * 0.5))
            
            for match_idx in range(num_matches):
                # Get point coordinates
                pt_a = matched_points_0[match_idx]
                pt_b = matched_points_1[match_idx]
                
                # Adjust for offset and scale (images start at y_offset)
                pt_a_canvas = (int(pt_a[0] * scale_factor), int(pt_a[1] * scale_factor + y_offset))
                pt_b_canvas = (int((pt_b[0] + w_a + 10) * scale_factor), int(pt_b[1] * scale_factor + y_offset))
                
                # Select color for this match (rotate through palette)
                color = colors[match_idx % len(colors)]
                
                # Draw line connecting the matches
                cv2.line(canvas, pt_a_canvas, pt_b_canvas, color, line_thickness)
                
                # Draw circles at keypoint locations
                cv2.circle(canvas, pt_a_canvas, circle_radius, color, circle_thickness)
                cv2.circle(canvas, pt_b_canvas, circle_radius, color, circle_thickness)
        else:
            cv2.putText(
                canvas, "No matches found", (int(10 * scale_factor), int(30 * scale_factor)),
                cv2.FONT_HERSHEY_SIMPLEX, scale_factor * 0.7, (0, 0, 0), int(scale_factor * 1.5)
            )
        
        # Add timestamp text at the top (black color)
        cv2.putText(
            canvas, f"Frame A: {frame_key_a}", (int(10 * scale_factor), int(25 * scale_factor)),
            cv2.FONT_HERSHEY_SIMPLEX, scale_factor * 0.5, (0, 0, 0), int(scale_factor * 1.5)
        )
        
        # Add timestamp text for right image (black color)
        w_mid = int(canvas.shape[1] / 2)
        cv2.putText(
            canvas, f"Frame B: {frame_key_b}", (w_mid + int(10 * scale_factor), int(25 * scale_factor)),
            cv2.FONT_HERSHEY_SIMPLEX, scale_factor * 0.5, (0, 0, 0), int(scale_factor * 1.5)
        )
        
        # Add summary information at the bottom (black color)
        summary_text = f"Covisible Match: {num_matches} inlier matches"
        cv2.putText(
            canvas, summary_text, (int(10 * scale_factor), canvas.shape[0] - int(10 * scale_factor)),
            cv2.FONT_HERSHEY_SIMPLEX, scale_factor * 0.6, (0, 0, 0), int(scale_factor * 1.5)
        )
        
        # Save visualization
        os.makedirs(output_dir, exist_ok=True)
        save_path = os.path.join(
            output_dir,
            f"match_{frame_key_a}_vs_{frame_key_b}.jpg"
        )
        cv2.imwrite(save_path, canvas)
        return save_path

    def save_covisible_graphs_visualizations(
        self,
        output_dir: str,
        max_edges_to_visualize: Optional[int] = None,
    ) -> None:
        """
        Save visualization of all covisible edge feature matches.
        
        Args:
            output_dir: Base directory to save visualizations
            max_edges_to_visualize: Limit the number of edges to visualize. None means all.
        """
        viz_dir = os.path.join(output_dir, "covisible_matches")
        os.makedirs(viz_dir, exist_ok=True)
        
        frame_graph = self.frame_covisibility_graph
        if not frame_graph:
            print("No covisible frame edges to visualize")
            return
        
        # Collect all unique edges
        edges_visualized = 0
        edges_set = set()
        
        for frame_key_a in sorted(frame_graph.keys()):
            for frame_key_b in sorted(frame_graph[frame_key_a]):
                # Avoid duplicate edges (a-b is same as b-a)
                edge_key = tuple(sorted([frame_key_a, frame_key_b]))
                if edge_key in edges_set:
                    continue
                edges_set.add(edge_key)
                
                if max_edges_to_visualize is not None and edges_visualized >= max_edges_to_visualize:
                    break
                
                match_result = self.get_frame_pair_match_result(frame_key_a, frame_key_b)
                match_count = int(match_result["num_inlier_matches"])
                if match_count < self.min_inlier_matches:
                    continue

                save_path = self.visualize_covisible_edge_matches(
                    frame_key_a, frame_key_b, match_result, viz_dir
                )
                if save_path:
                    print(f"Saved covisible edge visualization: {save_path}")
                    edges_visualized += 1
            
            if max_edges_to_visualize is not None and edges_visualized >= max_edges_to_visualize:
                break
        
        print(f"Covisible edge visualizations saved to: {viz_dir}")
        print(f"Total edges visualized: {edges_visualized}")

    def plot_trajectory_covisibility(
        self,
        pose_dict: Mapping[str, MappingPose],
        block: bool = True,
        save_path: Optional[str] = None,
    ) -> None:
        display_available = bool(os.environ.get("DISPLAY") or os.environ.get("WAYLAND_DISPLAY"))
        allow_interactive = os.environ.get("REACT_MAPPING_INTERACTIVE_PLOTS", "").lower() in {
            "1",
            "true",
            "yes",
        }
        use_file_backend = save_path is not None or not (block and display_available and allow_interactive)

        if use_file_backend:
            import matplotlib
            matplotlib.use("Agg", force=True)
            if save_path is None:
                save_path = os.path.abspath("trajectory_covisibility.png")
        else:
            for env_name in ("QT_PLUGIN_PATH", "QT_QPA_PLATFORM_PLUGIN_PATH"):
                env_value = os.environ.get(env_name, "")
                if "cv2" in env_value.lower():
                    os.environ.pop(env_name, None)

        import matplotlib.pyplot as plt

        frame_keys_sorted = sorted(self.frame_to_submaps_dict.keys())
        if len(frame_keys_sorted) == 0:
            raise ValueError("No mapping frames registered; cannot plot covisibility graph.")

        positions = []
        for frame_key in frame_keys_sorted:
            if frame_key not in pose_dict:
                raise KeyError(f"Missing pose for frame '{frame_key}' while plotting covisibility graph.")
            positions.append(np.asarray(pose_dict[frame_key].pos, dtype=np.float64).reshape(3))
        positions_arr = np.stack(positions, axis=0)

        fig = plt.figure(figsize=(10, 8))
        ax = fig.add_subplot(111, projection="3d")

        ax.plot(
            positions_arr[:, 0],
            positions_arr[:, 1],
            positions_arr[:, 2],
            color="tab:blue",
            linewidth=1.5,
            label="mapping trajectory",
        )
        ax.scatter(
            positions_arr[:, 0],
            positions_arr[:, 1],
            positions_arr[:, 2],
            color="tab:blue",
            s=10,
            alpha=0.8,
        )

        plotted_edge_label = False
        seen_edges: Set[Tuple[str, str]] = set()
        for frame_key_a, neighbors in self.frame_covisibility_graph.items():
            if frame_key_a not in pose_dict:
                continue
            pos_a = np.asarray(pose_dict[frame_key_a].pos, dtype=np.float64).reshape(3)
            for frame_key_b in neighbors:
                if frame_key_b not in pose_dict:
                    continue
                edge_key = (frame_key_a, frame_key_b) if frame_key_a <= frame_key_b else (frame_key_b, frame_key_a)
                if edge_key in seen_edges:
                    continue
                seen_edges.add(edge_key)
                pos_b = np.asarray(pose_dict[frame_key_b].pos, dtype=np.float64).reshape(3)
                ax.plot(
                    [pos_a[0], pos_b[0]],
                    [pos_a[1], pos_b[1]],
                    [pos_a[2], pos_b[2]],
                    color="tab:orange",
                    linewidth=0.8,
                    alpha=0.45,
                    label="covisibility edge" if not plotted_edge_label else None,
                )
                plotted_edge_label = True

        mins = positions_arr.min(axis=0)
        maxs = positions_arr.max(axis=0)
        center = 0.5 * (mins + maxs)
        radius = max(float(np.max(maxs - mins)) * 0.5, 1e-3)
        ax.set_xlim(center[0] - radius, center[0] + radius)
        ax.set_ylim(center[1] - radius, center[1] + radius)
        ax.set_zlim(center[2] - radius, center[2] + radius)

        ax.set_xlabel("x")
        ax.set_ylabel("y")
        ax.set_zlabel("z")
        ax.set_title(
            f"Mapping Frame Trajectory and Covisibility Edges\n"
            f"frames={len(frame_keys_sorted)}, edges={len(seen_edges)}"
        )
        ax.legend(loc="best")
        ax.grid(True)
        fig.tight_layout()
        if save_path is not None:
            save_path = os.path.abspath(save_path)
            os.makedirs(os.path.dirname(save_path), exist_ok=True)
            fig.savefig(save_path, dpi=200)
            print(f"Saved trajectory covisibility plot: {save_path}")
            plt.close(fig)
            return

        plt.show(block=block)
        if not block:
            plt.pause(0.001)
