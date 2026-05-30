import os
import time
import numpy as np
import cv2
from typing import Dict, List, Tuple, Optional, Any

_VISMATCH_MATCHER_CACHE: Dict[Tuple[str, str, int], Any] = {}


def _frame_to_vismatch_input(img: np.ndarray) -> np.ndarray:
    img_np = img.astype(np.float32)
    if img_np.max() > 1.0:
        img_np = img_np / 255.0
    if img_np.ndim == 2:
        img_np = np.repeat(img_np[None, ...], 3, axis=0)
    elif img_np.ndim == 3 and img_np.shape[-1] == 3:
        img_np = np.transpose(img_np, (2, 0, 1))
    elif img_np.ndim != 3 or img_np.shape[0] != 3:
        raise ValueError(f"Unsupported image shape for vismatch: {img.shape}")
    return np.ascontiguousarray(img_np)


def _to_numpy_safe(x: Any) -> Any:
    try:
        import torch
    except Exception:
        torch = None

    if isinstance(x, dict):
        return {k: _to_numpy_safe(v) for k, v in x.items()}
    if isinstance(x, list):
        return [_to_numpy_safe(v) for v in x]
    if isinstance(x, tuple):
        return tuple(_to_numpy_safe(v) for v in x)
    if torch is not None and isinstance(x, torch.Tensor):
        return x.detach().cpu().numpy()
    return x


def _move_tensor_tree_to_device(x: Any, device: str) -> Any:
    try:
        import torch
    except Exception:
        torch = None

    if isinstance(x, dict):
        return {k: _move_tensor_tree_to_device(v, device) for k, v in x.items()}
    if isinstance(x, list):
        return [_move_tensor_tree_to_device(v, device) for v in x]
    if isinstance(x, tuple):
        return tuple(_move_tensor_tree_to_device(v, device) for v in x)
    if torch is not None and isinstance(x, torch.Tensor):
        return x.to(device=device, non_blocking=True)
    return x


def offload_vismatch_frame_feature_to_cpu(frame_feature: Dict[str, Any]) -> Dict[str, Any]:
    return {
        "feats": _move_tensor_tree_to_device(frame_feature["feats"], "cpu"),
        "keypoint_coords": np.asarray(frame_feature["keypoint_coords"], dtype=np.float32),
    }


def prepare_vismatch_frame_feature_for_device(
    frame_feature: Dict[str, Any],
    device: str,
) -> Dict[str, Any]:
    return {
        "feats": _move_tensor_tree_to_device(frame_feature["feats"], device),
        "keypoint_coords": np.asarray(frame_feature["keypoint_coords"], dtype=np.float32),
    }


def _extract_vismatch_frame_feature(
    matcher: Any,
    image: np.ndarray,
    to_tensor_image: Any,
    torch_module: Any,
) -> Dict[str, Any]:
    matcher_img = _frame_to_vismatch_input(image)
    matcher_img_tensor = to_tensor_image(matcher_img).to(matcher.device)
    with torch_module.inference_mode():
        feats = matcher.extractor.extract(matcher_img_tensor)
    keypoint_coords = _to_numpy_safe(feats["keypoints"])[0].astype(np.float32, copy=False)
    return {
        "feats": feats,
        "keypoint_coords": keypoint_coords,
    }


def _get_vismatch_matcher_cached(
    matcher_name: str,
    device: str,
    max_num_keypoints: int,
):
    try:
        from vismatch import get_matcher
    except Exception as exc:
        raise ImportError(
            "vismatch is unavailable. Ensure the `vismatch` package is importable."
        ) from exc

    cache_key = (str(matcher_name), str(device), int(max_num_keypoints))
    matcher = _VISMATCH_MATCHER_CACHE.get(cache_key)
    if matcher is None:
        matcher = get_matcher(
            matcher_name,
            device=device,
            max_num_keypoints=int(max_num_keypoints),
        )
        matcher.skip_ransac = True
        _VISMATCH_MATCHER_CACHE[cache_key] = matcher
    return matcher


def _limit_matches_per_patch(
    pair_matches: List[Tuple[int, int]],
    keypoints_cur: List[cv2.KeyPoint],
    img_shape: Tuple[int, int],
    patch_divisor: int,
    min_total_matches_for_patch_limit: int,
    pair_scores: Optional[List[float]] = None,
) -> List[Tuple[int, int]]:
    """
    When the match count is high, keep at most one match per square patch in the
    current image. Patch size is defined as image_width // patch_divisor.
    """
    if (
        patch_divisor <= 0
        or min_total_matches_for_patch_limit <= 0
        or len(pair_matches) <= min_total_matches_for_patch_limit
        or len(pair_matches) == 0
    ):
        return pair_matches

    img_h, img_w = img_shape[:2]
    patch_size = max(1, int(img_w) // int(patch_divisor))
    best_by_patch: Dict[Tuple[int, int], Tuple[float, Tuple[int, int]]] = {}

    for match_idx, match in enumerate(pair_matches):
        cur_kp_idx, _ = match
        u_cur, v_cur = keypoints_cur[cur_kp_idx].pt
        patch_x = int(max(0.0, min(float(img_w - 1), float(u_cur))) // patch_size)
        patch_y = int(max(0.0, min(float(img_h - 1), float(v_cur))) // patch_size)
        patch_key = (patch_x, patch_y)
        score = (
            float(pair_scores[match_idx])
            if pair_scores is not None and match_idx < len(pair_scores)
            else float(match_idx)
        )
        prev_best = best_by_patch.get(patch_key)
        if prev_best is None or score < prev_best[0]:
            best_by_patch[patch_key] = (score, match)

    filtered_matches = [match for _, match in sorted(best_by_patch.values(), key=lambda item: item[0])]
    filtered_matches.sort(key=lambda x: (x[0], x[1]))
    return filtered_matches


def _coord_has_nonzero_conf(
    conf_map: Optional[np.ndarray],
    pt: np.ndarray,
    nearby_size: int = 0,
) -> bool:
    if conf_map is None:
        return True

    u_i = int(round(float(pt[0])))
    v_i = int(round(float(pt[1])))
    h, w = conf_map.shape[:2]
    if u_i < 0 or v_i < 0 or u_i >= w or v_i >= h:
        return False

    if nearby_size <= 0:
        return bool(conf_map[v_i, u_i] != 0)

    radius = int(nearby_size)
    u_min = max(0, u_i - radius)
    u_max = min(w, u_i + radius + 1)
    v_min = max(0, v_i - radius)
    v_max = min(h, v_i + radius + 1)
    return bool(np.all(conf_map[v_min:v_max, u_min:u_max] != 0))


def _filter_zero_confidence_matches(
    matched_points_0: np.ndarray,
    matched_points_1: np.ndarray,
    conf_0: Optional[np.ndarray] = None,
    conf_1: Optional[np.ndarray] = None,
    nearby_size: int = 0,
) -> Tuple[np.ndarray, np.ndarray]:
    if len(matched_points_0) == 0 or len(matched_points_1) == 0:
        return matched_points_0, matched_points_1
    if conf_0 is None and conf_1 is None:
        return matched_points_0, matched_points_1

    valid_mask = np.ones(len(matched_points_0), dtype=bool)
    for i, (pt_0, pt_1) in enumerate(zip(matched_points_0, matched_points_1)):
        if not _coord_has_nonzero_conf(conf_0, pt_0, nearby_size=nearby_size):
            valid_mask[i] = False
            continue
        if not _coord_has_nonzero_conf(conf_1, pt_1, nearby_size=nearby_size):
            valid_mask[i] = False

    return matched_points_0[valid_mask], matched_points_1[valid_mask]


def _project_kp_to_prev(
    u: float,
    v: float,
    depth: float,
    K_inv: np.ndarray,
    T_c2w: np.ndarray,
    K_prev: np.ndarray,
    T_w2c_prev: np.ndarray,
) -> Optional[Tuple[float, float]]:
    """
    Project a keypoint (u,v) with depth from current camera to previous image.
    Returns (u_prev, v_prev) or None if invalid.
    """
    if not np.isfinite(depth) or depth <= 0:
        return None

    # cam coords in current frame
    pix = np.array([u, v, 1.0], dtype=np.float32)
    xyz_c = (K_inv @ pix) * depth
    xyz1 = np.array([xyz_c[0], xyz_c[1], xyz_c[2], 1.0], dtype=np.float32)

    # world coords
    xyz_w = T_c2w @ xyz1

    # prev cam coords
    xyz_c_prev = (T_w2c_prev @ xyz_w)[:3]
    if xyz_c_prev[2] <= 0:
        return None

    # project to prev image
    uvw = K_prev @ xyz_c_prev
    u_prev = uvw[0] / uvw[2]
    v_prev = uvw[1] / uvw[2]
    return float(u_prev), float(v_prev)


def _depth_at_point(depth_map: np.ndarray, u: float, v: float) -> Optional[float]:
    u_i = int(round(float(u)))
    v_i = int(round(float(v)))
    h, w = depth_map.shape[:2]
    if u_i < 0 or v_i < 0 or u_i >= w or v_i >= h:
        return None
    return float(depth_map[v_i, u_i])


def _essential_inlier_matches(
    pts0: np.ndarray,
    pts1: np.ndarray,
    k0: np.ndarray,
    k1: np.ndarray,
    threshold_px: float,
) -> Tuple[np.ndarray, np.ndarray]:
    if len(pts0) < 5 or len(pts1) < 5:
        return np.zeros((0, 2), dtype=np.float32), np.zeros((0, 2), dtype=np.float32)

    pts0_norm = cv2.undistortPoints(
        pts0.reshape(-1, 1, 2).astype(np.float64),
        k0.astype(np.float64),
        None,
    ).reshape(-1, 2)
    pts1_norm = cv2.undistortPoints(
        pts1.reshape(-1, 1, 2).astype(np.float64),
        k1.astype(np.float64),
        None,
    ).reshape(-1, 2)
    mean_focal = float(
        0.25
        * (
            float(k0[0, 0])
            + float(k0[1, 1])
            + float(k1[0, 0])
            + float(k1[1, 1])
        )
    )
    threshold_norm = float(threshold_px) / max(mean_focal, 1e-9)
    E, inlier_mask = cv2.findEssentialMat(
        pts0_norm,
        pts1_norm,
        focal=1.0,
        pp=(0.0, 0.0),
        method=cv2.RANSAC,
        prob=0.99,
        threshold=threshold_norm,
    )
    if E is None or inlier_mask is None:
        return np.zeros((0, 2), dtype=np.float32), np.zeros((0, 2), dtype=np.float32)
    inlier_mask = inlier_mask.ravel().astype(bool)
    return pts0[inlier_mask], pts1[inlier_mask]


def extract_feat_and_match_dl(
    predictions: Any,
    max_prev: int = 1,
    device: str = "cuda",
    matcher_name: str = "superpoint-lightglue",
    max_num_keypoints: int = 1024,
    ransac_reproj_thresh: float = 3.0,
    use_inlier_matches: bool = True,
    rm_conf0_kpts: bool = False,
    rm_conf0_mths: bool = False,
    conf0_match_nearby_size: int = 5,
    max_reproj_error: float = 10.0,
    patch_match_limit_threshold: int = 0,
    patch_size_divisor: int = 0,
) -> Dict[str, Any]:
    """
    Extract and match features with a vismatch deep matcher.

    Args:
      predictions: object with processed_images [N,H,W,3].
      max_prev: number of previous frames to match against for each image.
      device: device passed to vismatch.
      matcher_name: vismatch matcher name, e.g. "aliked-lightglue".
      max_num_keypoints: max keypoints passed to vismatch.
      ransac_reproj_thresh: RANSAC reprojection threshold passed to vismatch.
      use_inlier_matches: if True, consume post-RANSAC matches; otherwise use all
        matcher correspondences.
      rm_conf0_kpts: if True, drop extracted keypoints whose confidence is zero
        before converting outputs to the extract_feat_and_match format.
      rm_conf0_mths: if True, drop matched pairs whose matched keypoint in either
        image has confidence zero.
      conf0_match_nearby_size: if greater than zero, also drop matched pairs when
        either matched pixel has a zero-confidence value inside the square
        neighborhood centered at that pixel. The value is used as the pixel radius
        of that square neighborhood.
      max_reproj_error: maximum allowed reprojection error in pixels in either
        direction for a matched pair.
      patch_match_limit_threshold: if total matches for a pair exceeds this
        value, apply patch-based filtering.
      patch_size_divisor: square patch size is image_width // patch_size_divisor.

    Returns:
      dict with:
        "keypoints": list of list[cv2.KeyPoint]
        "matches": dict[(cur_idx, prev_idx)] -> list of (cur_kp_idx, prev_kp_idx)
    """
    try:
        from vismatch.utils import to_tensor_image
        import torch
    except Exception as exc:
        raise ImportError(
            "vismatch is unavailable. Ensure the `vismatch` package is importable."
        ) from exc

    imgs = predictions.processed_images.copy()
    depths = predictions.depth.copy()
    intrinsics = predictions.intrinsics
    extrinsics = predictions.extrinsics
    conf_all = getattr(predictions, "conf", None)
    n = imgs.shape[0]
    if n == 0:
        return {"keypoints": [], "matches": {}}
    if rm_conf0_kpts and rm_conf0_mths:
        raise ValueError("rm_conf0_kpts and rm_conf0_mths cannot both be True.")

    matcher = _get_vismatch_matcher_cached(
        matcher_name=matcher_name,
        device=device,
        max_num_keypoints=int(max_num_keypoints),
    )

    def _coords_to_cv_keypoints(coords: np.ndarray) -> List[cv2.KeyPoint]:
        return [
            cv2.KeyPoint(float(pt[0]), float(pt[1]), 1.0, -1.0, 1.0)
            for pt in coords
        ]

    def _coords_to_index_lookup(coords: np.ndarray) -> Dict[Tuple[float, float], int]:
        return {
            (round(float(pt[0]), 4), round(float(pt[1]), 4)): idx
            for idx, pt in enumerate(coords)
        }

    def _lookup_coord_index(
        pt: np.ndarray,
        lookup: Dict[Tuple[float, float], int],
        coords: np.ndarray,
        atol: float = 1e-3,
    ) -> Optional[int]:
        key = (round(float(pt[0]), 4), round(float(pt[1]), 4))
        idx = lookup.get(key)
        if idx is not None:
            return idx
        if len(coords) == 0:
            return None
        d2 = np.sum((coords - pt[None, :]) ** 2, axis=1)
        best_idx = int(np.argmin(d2))
        if float(d2[best_idx]) <= float(atol) * float(atol):
            return best_idx
        return None

    frame_feature_cache: List[Dict[str, Any]] = []
    matcher_feats: List[Dict[str, Any]] = []
    keypoint_coords: List[np.ndarray] = []
    keypoint_lookups: List[Dict[Tuple[float, float], int]] = []
    keypoints: List[List[cv2.KeyPoint]] = []
    k_inv_list: List[np.ndarray] = []
    t_c2w_list: List[np.ndarray] = []
    t_w2c_list: List[np.ndarray] = []

    extract_start_time = time.perf_counter()
    for i in range(n):
        frame_feature = _extract_vismatch_frame_feature(
            matcher=matcher,
            image=imgs[i],
            to_tensor_image=to_tensor_image,
            torch_module=torch,
        )
        feats = frame_feature["feats"]
        coords = np.asarray(frame_feature["keypoint_coords"], dtype=np.float32)
        if rm_conf0_kpts:
            conf_map = None if conf_all is None else conf_all[i]
            keep_mask = np.array(
                [_coord_has_nonzero_conf(conf_map, pt) for pt in coords],
                dtype=bool,
            )
            coords = coords[keep_mask]
            keep_idx = np.where(keep_mask)[0]
            keep_idx_t = torch.as_tensor(keep_idx, device=feats["keypoints"].device, dtype=torch.long)
            feats["keypoints"] = feats["keypoints"].index_select(1, keep_idx_t)
            feats["descriptors"] = feats["descriptors"].index_select(1, keep_idx_t)
            if "keypoint_scores" in feats:
                feats["keypoint_scores"] = feats["keypoint_scores"].index_select(1, keep_idx_t)
            frame_feature = {
                "feats": feats,
                "keypoint_coords": coords,
            }
        kps_cv = _coords_to_cv_keypoints(coords)

        frame_feature_cache.append(frame_feature)
        matcher_feats.append(feats)
        keypoint_coords.append(coords)
        keypoint_lookups.append(_coords_to_index_lookup(keypoint_coords[-1]))
        keypoints.append(kps_cv)
        k_inv_list.append(np.linalg.inv(intrinsics[i].astype(np.float32)))
        t_w2c = np.eye(4, dtype=np.float32)
        t_w2c[:3, :4] = extrinsics[i].astype(np.float32)
        t_w2c_list.append(t_w2c)
        t_c2w_list.append(np.linalg.inv(t_w2c))
    extract_elapsed = time.perf_counter() - extract_start_time
    print(
        f"[feat-match] extraction: {extract_elapsed:.4f}s"
    )

    matches: Dict[Tuple[int, int], List[Tuple[int, int]]] = {}
    match_pair_count = 0
    ransac_pair_count = 0
    ransac_elapsed_total = 0.0
    match_start_time = time.perf_counter()
    for cur_idx in range(n):
        start_prev = max(0, cur_idx - max_prev)
        for prev_idx in range(start_prev, cur_idx):
            match_pair_count += 1
            if len(keypoints[cur_idx]) == 0 or len(keypoints[prev_idx]) == 0:
                continue

            conf_cur = None if conf_all is None else conf_all[cur_idx]
            depth_cur = depths[cur_idx]
            k_inv_cur = k_inv_list[cur_idx]
            t_c2w_cur = t_c2w_list[cur_idx]
            k_cur = intrinsics[cur_idx].astype(np.float32)
            t_w2c_cur = t_w2c_list[cur_idx]
            k_prev = intrinsics[prev_idx].astype(np.float32)
            t_w2c_prev = t_w2c_list[prev_idx]
            h_prev, w_prev = imgs[prev_idx].shape[:2]

            projected_inside = 0
            for pt_cur in keypoint_coords[cur_idx]:
                if not _coord_has_nonzero_conf(conf_cur, pt_cur):
                    continue
                u_i = int(round(float(pt_cur[0])))
                v_i = int(round(float(pt_cur[1])))
                if (
                    u_i < 0
                    or v_i < 0
                    or u_i >= depth_cur.shape[1]
                    or v_i >= depth_cur.shape[0]
                ):
                    continue
                proj = _project_kp_to_prev(
                    float(pt_cur[0]),
                    float(pt_cur[1]),
                    float(depth_cur[v_i, u_i]),
                    k_inv_cur,
                    t_c2w_cur,
                    k_prev,
                    t_w2c_prev,
                )
                if proj is None:
                    continue
                if 0 <= proj[0] < w_prev and 0 <= proj[1] < h_prev:
                    projected_inside += 1
                    if projected_inside >= 5:
                        break
            if projected_inside < 5:
                continue

            with torch.inference_mode():
                pred = matcher.matcher(
                    {
                        "image0": matcher_feats[cur_idx],
                        "image1": matcher_feats[prev_idx],
                    }
                )
            pred = _to_numpy_safe(pred)
            matched_indices = pred["matches"][0] if len(pred["matches"]) > 0 else np.zeros((0, 2), dtype=np.int64)
            if len(matched_indices) == 0:
                continue
            all_kpts_cur = keypoint_coords[cur_idx]
            all_kpts_prev = keypoint_coords[prev_idx]
            matched_cur = all_kpts_cur[matched_indices[:, 0]]
            matched_prev = all_kpts_prev[matched_indices[:, 1]]
            if rm_conf0_mths:
                matched_cur, matched_prev = _filter_zero_confidence_matches(
                    matched_cur,
                    matched_prev,
                    conf_0=conf_cur,
                    conf_1=None if conf_all is None else conf_all[prev_idx],
                    nearby_size=int(conf0_match_nearby_size),
                )
            if use_inlier_matches:
                ransac_start_time = time.perf_counter()
                matched_cur, matched_prev = _essential_inlier_matches(
                    matched_cur,
                    matched_prev,
                    k_cur,
                    k_prev,
                    threshold_px=float(ransac_reproj_thresh),
                )
                ransac_elapsed_total += time.perf_counter() - ransac_start_time
                ransac_pair_count += 1
            if len(matched_cur) == 0 or len(matched_prev) == 0:
                continue

            pair_matches: List[Tuple[int, int]] = []
            seen_pairs = set()
            depth_prev = depths[prev_idx]
            k_inv_prev = k_inv_list[prev_idx]
            t_c2w_prev = t_c2w_list[prev_idx]
            for pt_cur, pt_prev in zip(matched_cur, matched_prev):
                cur_kp_idx = _lookup_coord_index(
                    pt_cur,
                    keypoint_lookups[cur_idx],
                    keypoint_coords[cur_idx],
                )
                prev_kp_idx = _lookup_coord_index(
                    pt_prev,
                    keypoint_lookups[prev_idx],
                    keypoint_coords[prev_idx],
                )
                if cur_kp_idx is None or prev_kp_idx is None:
                    continue
                pair = (int(cur_kp_idx), int(prev_kp_idx))
                if pair in seen_pairs:
                    continue
                seen_pairs.add(pair)
                pair_matches.append(pair)

            if pair_matches:
                geo_filtered: List[Tuple[int, int]] = []
                geo_scores: List[float] = []
                for cur_kp_idx, prev_kp_idx in pair_matches:
                    u_cur, v_cur = keypoints[cur_idx][cur_kp_idx].pt
                    u_prev, v_prev = keypoints[prev_idx][prev_kp_idx].pt
                    depth_val_cur = _depth_at_point(depth_cur, u_cur, v_cur)
                    depth_val_prev = _depth_at_point(depth_prev, u_prev, v_prev)
                    if depth_val_cur is None or depth_val_prev is None:
                        continue

                    proj_cur_to_prev = _project_kp_to_prev(
                        float(u_cur),
                        float(v_cur),
                        depth_val_cur,
                        k_inv_cur,
                        t_c2w_cur,
                        k_prev,
                        t_w2c_prev,
                    )
                    proj_prev_to_cur = _project_kp_to_prev(
                        float(u_prev),
                        float(v_prev),
                        depth_val_prev,
                        k_inv_prev,
                        t_c2w_prev,
                        k_cur,
                        t_w2c_cur,
                    )
                    if proj_cur_to_prev is None or proj_prev_to_cur is None:
                        continue

                    err_cur_to_prev = float(
                        np.hypot(proj_cur_to_prev[0] - u_prev, proj_cur_to_prev[1] - v_prev)
                    )
                    err_prev_to_cur = float(
                        np.hypot(proj_prev_to_cur[0] - u_cur, proj_prev_to_cur[1] - v_cur)
                    )
                    if err_cur_to_prev <= float(max_reproj_error) and err_prev_to_cur <= float(max_reproj_error):
                        geo_filtered.append((cur_kp_idx, prev_kp_idx))
                        geo_scores.append(err_cur_to_prev + err_prev_to_cur)
                pair_matches = geo_filtered
                pair_matches = _limit_matches_per_patch(
                    pair_matches=pair_matches,
                    keypoints_cur=keypoints[cur_idx],
                    img_shape=imgs[cur_idx].shape,
                    patch_divisor=int(patch_size_divisor),
                    min_total_matches_for_patch_limit=int(patch_match_limit_threshold),
                    pair_scores=geo_scores,
                )

            if pair_matches:
                matches[(cur_idx, prev_idx)] = pair_matches
    match_elapsed = time.perf_counter() - match_start_time
    print(
        f"[feat-match] matching: {match_elapsed:.4f}s"
    )
    if use_inlier_matches:
        print(
            f"[feat-match] RANSAC: {ransac_elapsed_total:.4f}s"
        )

    return {
        "keypoints": keypoints,
        "matches": matches,
        "frame_feature_cache": frame_feature_cache,
        "frame_feature_cache_meta": {
            "matcher_name": str(matcher_name),
            "matcher_device": str(device),
            "max_num_keypoints": int(max_num_keypoints),
        },
    }


def verify_frame_pair_match_dl(
    image_0: np.ndarray,
    image_1: np.ndarray,
    intrinsics_0: np.ndarray,
    intrinsics_1: np.ndarray,
    device: str = "cuda",
    matcher_name: str = "superpoint-lightglue",
    max_num_keypoints: int = 1024,
    ransac_reproj_thresh: float = 3.0,
    use_inlier_matches: bool = True,
    conf_0: Optional[np.ndarray] = None,
    conf_1: Optional[np.ndarray] = None,
    conf0_match_nearby_size: int = 5,
    precomputed_feature_0: Optional[Dict[str, Any]] = None,
    precomputed_feature_1: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    """
    Verify a frame pair using a vismatch deep matcher followed by essential-matrix RANSAC.

    This helper is intentionally image/intrinsics-only. It is meant for pair validation
    after a cheaper pose-based candidate search, such as frame-covisibility verification.

    Args:
      image_0: First image (uint8 grayscale or RGB)
      image_1: Second image (uint8 grayscale or RGB)
      intrinsics_0: Camera intrinsics for first camera [3,3]
      intrinsics_1: Camera intrinsics for second camera [3,3]
      device: Device for matcher ("cuda" or "cpu")
      matcher_name: Name of the deep matcher (e.g., "superpoint-lightglue")
      max_num_keypoints: Maximum number of keypoints to extract
      ransac_reproj_thresh: RANSAC reprojection threshold in pixels
      use_inlier_matches: If True, keep only RANSAC inlier matches
      conf_0: Optional confidence map for image_0 to filter zero-confidence matches
      conf_1: Optional confidence map for image_1 to filter zero-confidence matches
      conf0_match_nearby_size: If greater than zero, also reject matches when
        either matched pixel has a zero-confidence value inside the square
        neighborhood centered at that pixel. The value is used as the pixel radius
        of that square neighborhood.

    Returns:
      dict with:
        "num_raw_matches": int
        "num_inlier_matches": int
        "matched_points0": np.ndarray [M,2]
        "matched_points1": np.ndarray [M,2]
    """
    try:
        from vismatch.utils import to_tensor_image
        import torch
    except Exception as exc:
        raise ImportError(
            "vismatch is unavailable. Ensure the `vismatch` package is importable."
        ) from exc

    matcher = _get_vismatch_matcher_cached(
        matcher_name=matcher_name,
        device=device,
        max_num_keypoints=int(max_num_keypoints),
    )

    if precomputed_feature_0 is not None:
        prepared_feature_0 = prepare_vismatch_frame_feature_for_device(precomputed_feature_0, matcher.device)
        feats_0 = prepared_feature_0["feats"]
        keypoints_0 = np.asarray(prepared_feature_0["keypoint_coords"], dtype=np.float32)
    else:
        frame_feature_0 = _extract_vismatch_frame_feature(
            matcher=matcher,
            image=image_0,
            to_tensor_image=to_tensor_image,
            torch_module=torch,
        )
        feats_0 = frame_feature_0["feats"]
        keypoints_0 = np.asarray(frame_feature_0["keypoint_coords"], dtype=np.float32)

    if precomputed_feature_1 is not None:
        prepared_feature_1 = prepare_vismatch_frame_feature_for_device(precomputed_feature_1, matcher.device)
        feats_1 = prepared_feature_1["feats"]
        keypoints_1 = np.asarray(prepared_feature_1["keypoint_coords"], dtype=np.float32)
    else:
        frame_feature_1 = _extract_vismatch_frame_feature(
            matcher=matcher,
            image=image_1,
            to_tensor_image=to_tensor_image,
            torch_module=torch,
        )
        feats_1 = frame_feature_1["feats"]
        keypoints_1 = np.asarray(frame_feature_1["keypoint_coords"], dtype=np.float32)

    with torch.inference_mode():
        pred = matcher.matcher(
            {
                "image0": feats_0,
                "image1": feats_1,
            }
        )

    pred = _to_numpy_safe(pred)
    matched_indices = pred["matches"][0] if len(pred["matches"]) > 0 else np.zeros((0, 2), dtype=np.int64)

    num_raw_matches = int(len(matched_indices))
    if num_raw_matches == 0:
        return {
            "num_raw_matches": 0,
            "num_inlier_matches": 0,
            "matched_points0": np.zeros((0, 2), dtype=np.float32),
            "matched_points1": np.zeros((0, 2), dtype=np.float32),
        }

    matched_points_0 = keypoints_0[matched_indices[:, 0]]
    matched_points_1 = keypoints_1[matched_indices[:, 1]]

    matched_points_0, matched_points_1 = _filter_zero_confidence_matches(
        matched_points_0,
        matched_points_1,
        conf_0=conf_0,
        conf_1=conf_1,
        nearby_size=int(conf0_match_nearby_size),
    )

    if use_inlier_matches:
        matched_points_0, matched_points_1 = _essential_inlier_matches(
            matched_points_0,
            matched_points_1,
            np.asarray(intrinsics_0, dtype=np.float32),
            np.asarray(intrinsics_1, dtype=np.float32),
            threshold_px=float(ransac_reproj_thresh),
        )

    return {
        "num_raw_matches": num_raw_matches,
        "num_inlier_matches": int(len(matched_points_0)),
        "matched_points0": np.asarray(matched_points_0, dtype=np.float32),
        "matched_points1": np.asarray(matched_points_1, dtype=np.float32),
    }


def visualize_matching(
    predictions: Any,
    match_dict: Dict[str, Any],
    topk: Optional[int] = None,
    prev: Optional[int] = None,
    wait_for_enter: bool = True,
    save_path_lst: Optional[List[str]] = None,
) -> None:
    """
    Visualize previous-frame matches for each current frame.

    Args:
      predictions: object with processed_images.
      match_dict: output dict from extract_feat_and_match.
      topk: number of best matched previous frames to render per current frame.
        Must be None when `prev` is set.
      prev: if set, render exactly the previous `prev` frame slots for each
        current frame in reverse chronological order. Must be None when `topk`
        is set. Missing history is shown as a black placeholder image.
      wait_for_enter: if True, wait for Enter/Space before advancing.
      save_path_lst: if empty/None, show with OpenCV window; otherwise save each
        current-frame visualization to save_path_lst[cur_idx].
    """
    imgs = predictions.processed_images
    conf_all = getattr(predictions, "conf", None)
    keypoints = match_dict["keypoints"]
    matches = match_dict["matches"]
    save_paths = save_path_lst or []
    should_save = bool(save_paths)
    if prev is not None and topk is not None:
        raise ValueError("visualize_matching: `prev` and `topk` cannot both be set.")
    if prev is None and topk is None:
        raise ValueError("visualize_matching: either `prev` or `topk` must be set.")

    def _overlay_zero_conf_red(img_rgb: np.ndarray, conf_map: Optional[np.ndarray], alpha: float = 0.2) -> np.ndarray:
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

    cur_indices = list(range(len(imgs)))
    for cur_idx in cur_indices:
        if prev is not None:
            prev_candidates = [(cur_idx - offset, 0) for offset in range(1, max(0, int(prev)) + 1)]
        else:
            prev_candidates = [
                (prev_idx, len(pair_matches))
                for (m_cur, prev_idx), pair_matches in matches.items()
                if m_cur == cur_idx
            ]
            prev_candidates.sort(key=lambda x: (-x[1], x[0]))
            prev_candidates = prev_candidates[: max(0, int(topk))]
            if not prev_candidates:
                prev_candidates = [(-1, 0) for _ in range(max(0, int(topk)))]
        if not prev_candidates:
            continue

        canvases = []
        conf_cur = conf_all[cur_idx] if conf_all is not None else None
        img_cur = _overlay_zero_conf_red(imgs[cur_idx], conf_cur)
        kps_cur = keypoints[cur_idx]
        cur_bgr = cv2.cvtColor(img_cur, cv2.COLOR_RGB2BGR)

        for prev_idx, _ in prev_candidates:
            if prev_idx < 0:
                img_prev = np.zeros_like(img_cur)
                kps_prev = []
                pair_matches = []
            else:
                conf_prev = conf_all[prev_idx] if conf_all is not None else None
                img_prev = _overlay_zero_conf_red(imgs[prev_idx], conf_prev)
                kps_prev = keypoints[prev_idx]
                pair_matches = matches.get((cur_idx, prev_idx), [])
            dmatches = [
                cv2.DMatch(_queryIdx=q, _trainIdx=p, _distance=0)
                for (q, p) in pair_matches
            ]

            prev_bgr = cv2.cvtColor(img_prev, cv2.COLOR_RGB2BGR)
            canvas = cv2.drawMatches(
                cur_bgr,
                kps_cur,
                prev_bgr,
                kps_prev,
                dmatches,
                None,
                flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS,
            )
            canvases.append(canvas)

        stacked = np.vstack(canvases)
        if should_save:
            if cur_idx >= len(save_paths):
                continue
            save_path = save_paths[cur_idx]
            if not save_path:
                continue
            save_dir = os.path.dirname(save_path)
            if save_dir:
                os.makedirs(save_dir, exist_ok=True)
            cv2.imwrite(save_path, stacked)
        else:
            win_name = f"Top-{topk} matches for {cur_idx}"
            cv2.imshow(win_name, stacked)
            if wait_for_enter:
                while True:
                    key = cv2.waitKey(0)
                    if key in (10, 13, 32):
                        break
            else:
                cv2.waitKey(0)
            cv2.destroyWindow(win_name)
