import contextlib
import os
import time
import numpy as np
import cv2
from typing import Dict, List, Tuple, Optional, Any

_VISMATCH_MATCHER_CACHE: Dict[Tuple[str, str, int], Any] = {}


def _inference_autocast(torch_module: Any, device: Any):
    if os.environ.get("SCARF_MATCHER_AUTOCAST", "0") == "0":
        return contextlib.nullcontext()
    if not str(device).startswith("cuda") or not torch_module.cuda.is_available():
        return contextlib.nullcontext()
    dtype = (
        torch_module.bfloat16
        if torch_module.cuda.is_bf16_supported()
        else torch_module.float16
    )
    return torch_module.autocast(device_type="cuda", dtype=dtype)


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
        if x.dtype in (torch.bfloat16, torch.float16):
            # numpy has no bfloat16; autocast outputs come back as float32
            x = x.float()
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
    _ext = getattr(matcher, "extractor", matcher)   # NN matchers (sift-nn/orb-nn) expose extract() directly
    with torch_module.inference_mode(), _inference_autocast(torch_module, matcher.device):
        feats = _ext.extract(matcher_img_tensor)
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
        trt_dir = os.environ.get("SCARF_MATCHER_TRT")
        if trt_dir and str(matcher_name) == "superpoint-lightglue" and str(device).startswith("cuda"):
            # SuperPoint + LightGlue as TensorRT engines (scarf_slam/backends/superpoint_lightglue_trt.py); same extract()/matcher() contracts
            from scarf_slam.backends.superpoint_lightglue_trt import SuperPointTrt, LightGlueTrt
            matcher.extractor = SuperPointTrt(matcher.extractor, trt_dir)
            matcher.matcher = LightGlueTrt(matcher.matcher, trt_dir, n_kpts=int(max_num_keypoints))
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

    # vectorised _coord_has_nonzero_conf over all matches: one erosion per confidence map instead of a window per point
    valid_mask = np.ones(len(matched_points_0), dtype=bool)
    for pts, cmap in ((matched_points_0, conf_0), (matched_points_1, conf_1)):
        if cmap is None:
            continue
        u_i, v_i, inside = _round_pixel_indices(np.asarray(pts, dtype=np.float32).reshape(-1, 2), cmap.shape)
        win = _window_all_nonzero_map(cmap, int(nearby_size))
        valid_mask &= inside & win[np.clip(v_i, 0, cmap.shape[0] - 1), np.clip(u_i, 0, cmap.shape[1] - 1)]

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


def _project_points_batch(
    uv: np.ndarray,
    depth: np.ndarray,
    K_inv: np.ndarray,
    T_c2w: np.ndarray,
    K_prev: np.ndarray,
    T_w2c_prev: np.ndarray,
) -> Tuple[np.ndarray, np.ndarray]:
    """Vectorised _project_kp_to_prev: uv (M,2), depth (M,) -> projected (M,2) float32 and a validity mask
    (finite positive depth, positive depth in the previous camera). Same float32 arithmetic as the scalar version."""
    uv = np.asarray(uv, dtype=np.float32).reshape(-1, 2)
    depth = np.asarray(depth, dtype=np.float32).reshape(-1)
    valid = np.isfinite(depth) & (depth > 0)
    pix = np.concatenate([uv, np.ones((len(uv), 1), dtype=np.float32)], axis=1)
    xyz_c = (pix @ K_inv.astype(np.float32).T) * depth[:, None]
    xyz1 = np.concatenate([xyz_c, np.ones((len(uv), 1), dtype=np.float32)], axis=1)
    xyz_w = xyz1 @ T_c2w.astype(np.float32).T
    xyz_c_prev = (xyz_w @ T_w2c_prev.astype(np.float32).T)[:, :3]
    valid &= xyz_c_prev[:, 2] > 0
    uvw = xyz_c_prev @ K_prev.astype(np.float32).T
    with np.errstate(divide="ignore", invalid="ignore"):
        proj = uvw[:, :2] / uvw[:, 2:3]
    return proj, valid


def _round_pixel_indices(uv: np.ndarray, shape: Tuple[int, int]) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """int(round(.)) of (u, v) for every point (round-half-to-even like Python's round) and an in-bounds mask."""
    h, w = shape[:2]
    u_i = np.rint(np.asarray(uv[:, 0], dtype=np.float64)).astype(np.int64)
    v_i = np.rint(np.asarray(uv[:, 1], dtype=np.float64)).astype(np.int64)
    inside = (u_i >= 0) & (v_i >= 0) & (u_i < w) & (v_i < h)
    return u_i, v_i, inside


def _window_all_nonzero_map(conf_map: np.ndarray, radius: int) -> np.ndarray:
    """Per-pixel `all(conf[v-r:v+r+1, u-r:u+r+1] != 0)` with the window clipped at the image border (what
    _coord_has_nonzero_conf(nearby_size=r) computes per point), as one erosion of the nonzero mask."""
    nz = np.asarray(conf_map != 0, dtype=np.uint8)
    if radius > 0:
        if os.environ.get("SCARF_WINDOW_TORCH", "0") == "1":
            # the same erosion as a torch min-pool (max_pool2d pads with -inf, so the border behaves like cval=1), identical output,
            # but measured 10x SLOWER than scipy at 504x378 (9.0 vs 0.9 ms, 2026-11-06): kept only as an A/B switch, off by default
            import torch
            t = torch.from_numpy(nz).to(torch.float32)[None, None]; r = int(radius)
            return ((-torch.nn.functional.max_pool2d(-t, 2 * r + 1, stride=1, padding=r))[0, 0] > 0.5).numpy()
        from scipy.ndimage import minimum_filter
        nz = minimum_filter(nz, size=2 * int(radius) + 1, mode="constant", cval=1)
    return nz.astype(bool)


def _depth_at_point(depth_map: np.ndarray, u: float, v: float) -> Optional[float]:
    u_i = int(round(float(u)))
    v_i = int(round(float(v)))
    h, w = depth_map.shape[:2]
    if u_i < 0 or v_i < 0 or u_i >= w or v_i >= h:
        return None
    return float(depth_map[v_i, u_i])


def _essential_inlier_mask(
    pts0: np.ndarray,
    pts1: np.ndarray,
    k0: np.ndarray,
    k1: np.ndarray,
    threshold_px: float,
) -> np.ndarray:
    """Boolean inlier mask of the essential-matrix RANSAC of _essential_inlier_matches (all False when it fails)."""
    n = len(pts0)
    if n < 5 or len(pts1) < 5:
        return np.zeros(n, dtype=bool)
    pts0_norm = cv2.undistortPoints(pts0.reshape(-1, 1, 2).astype(np.float64), k0.astype(np.float64), None).reshape(-1, 2)
    pts1_norm = cv2.undistortPoints(pts1.reshape(-1, 1, 2).astype(np.float64), k1.astype(np.float64), None).reshape(-1, 2)
    mean_focal = float(0.25 * (float(k0[0, 0]) + float(k0[1, 1]) + float(k1[0, 0]) + float(k1[1, 1])))
    threshold_norm = float(threshold_px) / max(mean_focal, 1e-9)
    E, inlier_mask = cv2.findEssentialMat(pts0_norm, pts1_norm, focal=1.0, pp=(0.0, 0.0), method=cv2.RANSAC, prob=0.99, threshold=threshold_norm)
    if E is None or inlier_mask is None:
        return np.zeros(n, dtype=bool)
    return inlier_mask.ravel().astype(bool)


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
    def _pair_ransac(p0, p1, K):
        if len(p0) >= 8:
            _, inl = cv2.findEssentialMat(p0, p1, np.asarray(K, np.float64),
                                          method=cv2.USAC_MAGSAC, prob=0.999, threshold=ransac_reproj_thresh)
            if inl is not None:
                keep = inl.reshape(-1).astype(bool)
                return p0[keep], p1[keep]
        return p0, p1
    if matcher_name.endswith("-nn"):   # one-shot classical matchers
        from vismatch.utils import to_tensor_image as _tti
        import torch as _t
        m = _get_vismatch_matcher_cached(matcher_name=matcher_name, device=device, max_num_keypoints=max_num_keypoints)
        kps = [[] for _ in range(n)]
        matches = {}
        tens = [_tti(_frame_to_vismatch_input(imgs[i])).to(m.device) for i in range(n)]
        for i in range(n):
            for j in range(max(0, i - max_prev), i):
                with _t.inference_mode():
                    out = m(tens[i], tens[j])
                p0 = np.asarray(out["matched_kpts0"], np.float32).reshape(-1, 2)
                p1 = np.asarray(out["matched_kpts1"], np.float32).reshape(-1, 2)
                p0, p1 = _pair_ransac(p0, p1, intrinsics[i][:3, :3] if intrinsics[i].shape[0] > 2 else intrinsics[i])
                base_i, base_j = len(kps[i]), len(kps[j])
                kps[i] += [cv2.KeyPoint(float(x), float(y), 1.0) for x, y in p0]
                kps[j] += [cv2.KeyPoint(float(x), float(y), 1.0) for x, y in p1]
                matches[(i, j)] = [(base_i + k, base_j + k) for k in range(len(p0))]
        return {"keypoints": kps, "matches": matches}
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

    frame_feature_cache: List[Dict[str, Any]] = []
    matcher_feats: List[Dict[str, Any]] = []
    keypoint_coords: List[np.ndarray] = []
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

    # confidence-window maps per frame: radius 0 for the overlap pre-check, conf0_match_nearby_size for the match filter
    nz_maps = None if conf_all is None else [_window_all_nonzero_map(conf_all[i], 0) for i in range(n)]
    win_maps = None if conf_all is None else [_window_all_nonzero_map(conf_all[i], int(conf0_match_nearby_size)) for i in range(n)]
    matches: Dict[Tuple[int, int], List[Tuple[int, int]]] = {}
    match_pair_count = 0
    ransac_pair_count = 0
    ransac_elapsed_total = 0.0
    match_start_time = time.perf_counter()
    # pass 1: overlap pre-check (vectorised): at least 5 keypoints with nonzero confidence project inside the previous image
    eligible: List[Tuple[int, int]] = []
    for cur_idx in range(n):
        start_prev = max(0, cur_idx - max_prev)
        for prev_idx in range(start_prev, cur_idx):
            match_pair_count += 1
            if len(keypoints[cur_idx]) == 0 or len(keypoints[prev_idx]) == 0:
                continue
            depth_cur = depths[cur_idx]
            h_prev, w_prev = imgs[prev_idx].shape[:2]
            kc = keypoint_coords[cur_idx]
            u_i, v_i, inside = _round_pixel_indices(kc, depth_cur.shape)
            if nz_maps is not None:
                inside &= nz_maps[cur_idx][np.clip(v_i, 0, depth_cur.shape[0] - 1), np.clip(u_i, 0, depth_cur.shape[1] - 1)]
            proj, ok = _project_points_batch(kc[inside], depth_cur[v_i[inside], u_i[inside]], k_inv_list[cur_idx], t_c2w_list[cur_idx],
                                             intrinsics[prev_idx].astype(np.float32), t_w2c_list[prev_idx])
            ok &= (proj[:, 0] >= 0) & (proj[:, 0] < w_prev) & (proj[:, 1] >= 0) & (proj[:, 1] < h_prev)
            if int(ok.sum()) >= 5:
                eligible.append((cur_idx, prev_idx))

    # pass 2: the matcher on every eligible pair (one batched TensorRT call per submap when the backend offers it)
    forward_batch = getattr(matcher.matcher, "forward_batch", None)
    with torch.inference_mode(), _inference_autocast(torch, matcher.device):
        if forward_batch is not None and getattr(matcher.matcher, "batch", 0) > 0 and len(eligible) >= 4:
            preds = forward_batch([(matcher_feats[i], matcher_feats[j]) for i, j in eligible])
        else:
            preds = [matcher.matcher({"image0": matcher_feats[i], "image1": matcher_feats[j]}) for i, j in eligible]

    # pass 3: confidence-window filter, RANSAC, bidirectional depth-reprojection gate
    for (cur_idx, prev_idx), pred in zip(eligible, preds):
            depth_cur = depths[cur_idx]
            k_inv_cur = k_inv_list[cur_idx]
            t_c2w_cur = t_c2w_list[cur_idx]
            k_cur = intrinsics[cur_idx].astype(np.float32)
            t_w2c_cur = t_w2c_list[cur_idx]
            k_prev = intrinsics[prev_idx].astype(np.float32)
            t_w2c_prev = t_w2c_list[prev_idx]

            pred = _to_numpy_safe(pred)
            matched_indices = pred["matches"][0] if len(pred["matches"]) > 0 else np.zeros((0, 2), dtype=np.int64)
            if len(matched_indices) == 0:
                continue
            # The matcher returns keypoint indices, so the filters below carry index arrays (the former per-point Python
            # loops mapped coordinates back to indices through a lookup dict; SuperPoint keypoints are unique after NMS,
            # so this is the same set). All steps are vectorised: 15 pairs x ~1k matches per submap.
            all_kpts_cur = keypoint_coords[cur_idx]
            all_kpts_prev = keypoint_coords[prev_idx]
            idx_cur = matched_indices[:, 0].astype(np.int64)
            idx_prev = matched_indices[:, 1].astype(np.int64)
            depth_prev = depths[prev_idx]
            k_inv_prev = k_inv_list[prev_idx]
            t_c2w_prev = t_c2w_list[prev_idx]
            if rm_conf0_mths and conf_all is not None:
                # both matched pixels need a fully nonzero confidence window (radius conf0_match_nearby_size)
                keep = np.ones(len(idx_cur), dtype=bool)
                for kp_all, idx, win in ((all_kpts_cur, idx_cur, win_maps[cur_idx]), (all_kpts_prev, idx_prev, win_maps[prev_idx])):
                    u_i, v_i, inside = _round_pixel_indices(kp_all[idx], win.shape)
                    keep &= inside & win[np.clip(v_i, 0, win.shape[0] - 1), np.clip(u_i, 0, win.shape[1] - 1)]
                idx_cur, idx_prev = idx_cur[keep], idx_prev[keep]
            matched_cur = all_kpts_cur[idx_cur]
            matched_prev = all_kpts_prev[idx_prev]
            if use_inlier_matches:
                ransac_start_time = time.perf_counter()
                keep = _essential_inlier_mask(matched_cur, matched_prev, k_cur, k_prev, threshold_px=float(ransac_reproj_thresh))
                idx_cur, idx_prev = idx_cur[keep], idx_prev[keep]
                ransac_elapsed_total += time.perf_counter() - ransac_start_time
                ransac_pair_count += 1
            if len(idx_cur) == 0:
                continue
            # unique (cur, prev) index pairs, first occurrence kept (the former seen_pairs set)
            _, first = np.unique(idx_cur * (len(all_kpts_prev) + 1) + idx_prev, return_index=True)
            first.sort(); idx_cur, idx_prev = idx_cur[first], idx_prev[first]
            pair_matches: List[Tuple[int, int]] = []

            if len(idx_cur):
                # bidirectional depth-reprojection gate
                uv_c, uv_p = all_kpts_cur[idx_cur], all_kpts_prev[idx_prev]
                uc_i, vc_i, in_c = _round_pixel_indices(uv_c, depth_cur.shape)
                up_i, vp_i, in_p = _round_pixel_indices(uv_p, depth_prev.shape)
                ok = in_c & in_p
                d_c = np.zeros(len(idx_cur), dtype=np.float32); d_p = np.zeros(len(idx_cur), dtype=np.float32)
                d_c[ok] = depth_cur[vc_i[ok], uc_i[ok]]; d_p[ok] = depth_prev[vp_i[ok], up_i[ok]]
                proj_cp, ok_cp = _project_points_batch(uv_c, d_c, k_inv_cur, t_c2w_cur, k_prev, t_w2c_prev)
                proj_pc, ok_pc = _project_points_batch(uv_p, d_p, k_inv_prev, t_c2w_prev, k_cur, t_w2c_cur)
                ok &= ok_cp & ok_pc
                err_cp = np.hypot(proj_cp[:, 0] - uv_p[:, 0], proj_cp[:, 1] - uv_p[:, 1])
                err_pc = np.hypot(proj_pc[:, 0] - uv_c[:, 0], proj_pc[:, 1] - uv_c[:, 1])
                ok &= (err_cp <= float(max_reproj_error)) & (err_pc <= float(max_reproj_error))
                pair_matches = [(int(a), int(b)) for a, b in zip(idx_cur[ok], idx_prev[ok])]
                geo_scores = (err_cp[ok] + err_pc[ok]).astype(float).tolist()
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
    if matcher_name.endswith("-nn"):   # one-shot matchers (sift-nn/orb-nn): no extractor/matcher split
        from vismatch.utils import to_tensor_image as _tti
        import torch as _t
        m = _get_vismatch_matcher_cached(matcher_name=matcher_name, device=device, max_num_keypoints=max_num_keypoints)
        t0 = _tti(_frame_to_vismatch_input(image_0)).to(m.device)
        t1 = _tti(_frame_to_vismatch_input(image_1)).to(m.device)
        with _t.inference_mode():
            out = m(t0, t1)
        p0 = np.asarray(out["matched_kpts0"], np.float32).reshape(-1, 2)
        p1 = np.asarray(out["matched_kpts1"], np.float32).reshape(-1, 2)
        n_raw = len(p0)
        if n_raw >= 8:
            _, inl = cv2.findEssentialMat(p0, p1, intrinsics_0.astype(np.float64),
                                          method=cv2.USAC_MAGSAC, prob=0.999, threshold=ransac_reproj_thresh)
            if inl is not None:
                keep = inl.reshape(-1).astype(bool); p0, p1 = p0[keep], p1[keep]
        return {"num_raw_matches": int(n_raw), "num_inlier_matches": int(len(p0)),
                "matched_points0": p0, "matched_points1": p1}
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

    with torch.inference_mode(), _inference_autocast(torch, matcher.device):
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
