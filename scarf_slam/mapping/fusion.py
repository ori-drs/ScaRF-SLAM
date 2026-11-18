from typing import Dict, List, Optional, Tuple, Union

import numpy as np
import torch
from scipy.ndimage import maximum_filter, minimum_filter


def mask_depth_edge_batch(
    depth: np.ndarray,
    kernel_size: int = 3,
    atol: Optional[float] = None,
    rtol: Optional[float] = None,
    invalid_value: float = 0.0,
) -> np.ndarray:
    """
    Compute a depth edge mask for a batch of depth maps.

    Args:
        depth (np.ndarray): float32 depth maps, shape [N, H, W]
        kernel_size (int): neighborhood size (odd number)
        atol (float): absolute depth difference threshold (meters)
        rtol (float): relative depth difference threshold
        invalid_value (float): invalid depth value (e.g., 0)

    Returns:
        edge (np.ndarray): bool array [N, H, W]
            True  -> edge pixel
            False -> non-edge
    """
    assert depth.ndim == 3, "depth must have shape [N, H, W]"
    assert kernel_size % 2 == 1, "kernel_size must be odd"

    depth = depth.astype(np.float32)

    # valid depth pixels
    valid = np.isfinite(depth) & (depth > invalid_value)

    # compute local max and min independently per batch element
    depth_max = maximum_filter(
        np.where(valid, depth, -np.inf),
        size=(1, kernel_size, kernel_size),
        mode="nearest",
    )
    depth_min = minimum_filter(
        np.where(valid, depth, np.inf),
        size=(1, kernel_size, kernel_size),
        mode="nearest",
    )

    # local depth variation
    diff = depth_max - depth_min

    # edge condition
    edge = np.zeros_like(depth, dtype=bool)

    if atol is not None:
        edge |= diff > atol

    if rtol is not None:
        edge |= (diff / np.maximum(depth, 1e-6)) > rtol

    # treat invalid pixels as edges (common and usually desired)
    edge |= ~valid

    return edge


def fuse_overlaps_torch(
    colors_1: np.ndarray,
    pts_world_1: np.ndarray,
    conf_1: np.ndarray,
    colors_2: np.ndarray,
    pts_world_2: np.ndarray,
    conf_2: np.ndarray,
    eps: float = 1e-8,
    dist_thresh: float = 0.1,
    use_cuda: bool = True,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    PyTorch version of fuse_overlaps (CUDA if available and enabled).

    Args:
        colors_1: uint8 colors, shape (N, 3).
        pts_world_1: float32 points, shape (N, 3).
        conf_1: float32 confidence, shape (N,).
        colors_2: uint8 colors, shape (N, 3).
        pts_world_2: float32 points, shape (N, 3).
        conf_2: float32 confidence, shape (N,).
        eps: Small epsilon to avoid division by zero.
        dist_thresh: Distance threshold for override logic.
        debug: Print dist-mask proportion if True.
        use_cuda: Use CUDA if available.

    Returns:
        (fused_colors, fused_pts_world, fused_conf)
    """
    # Device selection
    device = torch.device("cuda" if (use_cuda and torch.cuda.is_available()) else "cpu")

    # Basic validation (same checks as your numpy function)
    if colors_1.ndim != 2 or colors_1.shape[1] != 3:
        raise ValueError("colors_1 must have shape (N, 3)")
    if colors_2.ndim != 2 or colors_2.shape[1] != 3:
        raise ValueError("colors_2 must have shape (N, 3)")
    if pts_world_1.ndim != 2 or pts_world_1.shape[1] != 3:
        raise ValueError("pts_world_1 must have shape (N, 3)")
    if pts_world_2.ndim != 2 or pts_world_2.shape[1] != 3:
        raise ValueError("pts_world_2 must have shape (N, 3)")
    if conf_1.ndim != 1 or conf_2.ndim != 1:
        raise ValueError("conf_1 and conf_2 must be 1-D arrays of shape (N,)")

    N = colors_1.shape[0]
    if not (colors_2.shape[0] == pts_world_1.shape[0] == pts_world_2.shape[0] == conf_1.shape[0] == conf_2.shape[0] == N):
        raise ValueError("All inputs must have the same leading dimension N")

    # Move to torch on device and to float for computation
    colors_1_t = torch.from_numpy(colors_1.astype(np.float32)).to(device=device)
    colors_2_t = torch.from_numpy(colors_2.astype(np.float32)).to(device=device)

    pts1_t = torch.from_numpy(pts_world_1.astype(np.float32)).to(device=device)
    pts2_t = torch.from_numpy(pts_world_2.astype(np.float32)).to(device=device)

    conf_1_t = torch.from_numpy(conf_1.astype(np.float32)).to(device=device)
    conf_2_t = torch.from_numpy(conf_2.astype(np.float32)).to(device=device)

    # Fuse confidence
    fused_conf_t = conf_1_t + conf_2_t  # (N,)

    # Safe denominator
    safe_denom_t = torch.where(fused_conf_t > eps, fused_conf_t, torch.ones_like(fused_conf_t, device=device))
    zero_mask_t = fused_conf_t <= eps  # (N,)

    # Confidence-weighted fusion (colors + pts)
    fused_colors_f_t = (conf_1_t.unsqueeze(1) * colors_1_t + conf_2_t.unsqueeze(1) * colors_2_t) / safe_denom_t.unsqueeze(1)
    fused_pts_world_t = (conf_1_t.unsqueeze(1) * pts1_t + conf_2_t.unsqueeze(1) * pts2_t) / safe_denom_t.unsqueeze(1)

    if zero_mask_t.any():
        fused_colors_f_t[zero_mask_t] = 0.0
        fused_pts_world_t[zero_mask_t] = 0.0

    # Distance-based override
    # diff_pts = pts1_t - pts2_t
    # dist_t = torch.norm(diff_pts, dim=1)  # (N,)
    # dist_mask_t = (dist_t > dist_thresh) & (conf_1_t > 0) & (conf_2_t > 0)  # (N,)

    # if dist_mask_t.any():
    #     fused_pts_world_t[dist_mask_t] = pts2_t[dist_mask_t]
    #     fused_colors_f_t[dist_mask_t] = colors_2_t[dist_mask_t]
    #     fused_conf_t[dist_mask_t] = conf_2_t[dist_mask_t]

    # Finalize types and move to CPU / numpy
    fused_colors_np = torch.clamp(torch.round(fused_colors_f_t), 0, 255).to(dtype=torch.uint8).cpu().numpy()
    fused_pts_world_np = fused_pts_world_t.cpu().numpy().astype(np.float32)
    fused_conf_np = fused_conf_t.cpu().numpy().astype(np.float32)

    return fused_colors_np, fused_pts_world_np, fused_conf_np


def _as_device_tensor(value, *, device: torch.device, dtype: Optional[torch.dtype] = None) -> torch.Tensor:
    if isinstance(value, torch.Tensor):
        return value.to(device=device, dtype=dtype) if dtype is not None else value.to(device=device)
    return torch.as_tensor(value, dtype=dtype, device=device)


def get_matching_torch(
    pts_world_1: Union[np.ndarray, torch.Tensor],  # float32 [H_1, W_1, 3]
    mask_1: Union[np.ndarray, torch.Tensor],       # bool    [H_1, W_1]
    depth_2: Union[np.ndarray, torch.Tensor],      # float32 [H_2, W_2]
    T_view_world_2: Union[np.ndarray, torch.Tensor],  # float32 [4, 4]  (SLAM: world -> view / cam)
    intrinsics_2: List[float],         # [fx, fy, cx, cy]
    depth_thresh: float = 0.05,
    unique_mapping: bool = False,
    return_tensor: bool = False,
) -> Union[np.ndarray, torch.Tensor]:
    """
    GPU-first PyTorch implementation. Inputs are numpy, output is numpy (or the int64 device tensor with return_tensor).

    Args:
        pts_world_1: float32, shape (H1, W1, 3).
        mask_1: bool mask, shape (H1, W1).
        depth_2: float32 depth, shape (H2, W2).
        T_view_world_2: float32, shape (4, 4), world -> frame.
        intrinsics_2: [fx, fy, cx, cy].
        depth_thresh: Depth consistency threshold.
        unique_mapping: If True, enforce unique target mapping.

    Returns:
        pixel_matching: int32, shape (H1, W1, 2), [v, u] or [-1, -1] if unmatched.
    """

    # Choose device (prefer CUDA if available)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    pts = _as_device_tensor(pts_world_1, device=device, dtype=torch.float32)
    mask = _as_device_tensor(mask_1, device=device, dtype=torch.bool)
    depth2 = _as_device_tensor(depth_2, device=device, dtype=torch.float32)
    T = _as_device_tensor(T_view_world_2, device=device, dtype=torch.float32)
    fx, fy, cx, cy = intrinsics_2

    H1, W1, _ = pts.shape
    H2, W2 = depth2.shape

    # Flatten
    N = H1 * W1
    mask_flat = mask.reshape(-1)
    pts_flat = pts.reshape(N, 3).float()

    # Homogeneous coords
    ones = torch.ones((N, 1), dtype=pts_flat.dtype, device=device)
    pts_world_h = torch.cat([pts_flat, ones], dim=1)  # (N,4)

    # Transform world -> cam2
    pts_cam_h = (T @ pts_world_h.t()).t()  # (N,4)
    x_cam = pts_cam_h[:, 0]
    y_cam = pts_cam_h[:, 1]
    z_cam = pts_cam_h[:, 2]

    # Prepare output (-1 default)
    pixel_matching = -torch.ones((N, 2), dtype=torch.int64, device=device)

    # Valid z>0
    valid_z_mask = z_cam > 0

    # Project to image plane (float)
    u_proj = (fx * (x_cam / z_cam) + cx)
    v_proj = (fy * (y_cam / z_cam) + cy)

    # Round to int pixel indices
    u_int = torch.round(u_proj).to(torch.int64)
    v_int = torch.round(v_proj).to(torch.int64)

    # In-bounds
    in_bounds_mask = (u_int >= 0) & (u_int < W2) & (v_int >= 0) & (v_int < H2)

    # Combined mask
    combined_mask = valid_z_mask & in_bounds_mask & mask_flat

    if combined_mask.any():
        indices = torch.nonzero(combined_mask, as_tuple=False).squeeze(1)  # (M,)
        tgt_v = v_int[indices]  # row
        tgt_u = u_int[indices]  # col
        cam_depths = z_cam[indices]

        # Read depths at projected pixels (device tensors)
        depth_at_tgt = depth2[tgt_v, tgt_u]  # shape (M,)

        depth_valid_mask = depth_at_tgt > 0
        depth_diff = torch.abs(cam_depths - depth_at_tgt)

        match_mask = depth_valid_mask & (depth_diff < depth_thresh)

        if match_mask.any():
            matched_indices = indices[match_mask]           # flattened source indices (K,)
            matched_v = v_int[matched_indices]              # (K,)
            matched_u = u_int[matched_indices]              # (K,)
            matched_depth_diff = depth_diff[match_mask]     # (K,)

            # write matches
            pixel_matching[matched_indices, 0] = matched_v
            pixel_matching[matched_indices, 1] = matched_u

            if unique_mapping:
                # Construct unique target id per matched (0 .. H2*W2-1)
                tgt_flat_idx = matched_v * W2 + matched_u      # (K,) int64
                num_targets = H2 * W2

                # 1) Find best (minimum) depth difference per target using scatter_reduce_ (amin)
                best_diff_per_target = torch.full((num_targets,), float("inf"), device=device, dtype=matched_depth_diff.dtype)
                # scatter_reduce_ requires index and src same shape
                best_diff_per_target.scatter_reduce_(0, tgt_flat_idx, matched_depth_diff, reduce="amin", include_self=True)

                # 2) Create a candidate source index array where only entries that match the best diff are kept,
                #    otherwise set to a large sentinel. Then per-target take amin of these candidate source indices
                #    to deterministically pick a single source index per target (smallest source index in ties).
                # mask of best candidates
                is_best_candidate = matched_depth_diff == best_diff_per_target[tgt_flat_idx]

                # Prepare candidate source indices (int64). Use a large sentinel for non-candidates.
                big = torch.iinfo(torch.int64).max
                matched_src_indices = matched_indices.clone().to(torch.int64)  # (K,)
                cand_src = torch.full_like(matched_src_indices, big, device=device)
                if is_best_candidate.any():
                    cand_src[is_best_candidate] = matched_src_indices[is_best_candidate]

                # For each target, pick the minimal candidate src index (if any) -> deterministic single source per target
                best_src_for_target = torch.full((num_targets,), -1, dtype=torch.int64, device=device)
                # scatter_reduce_ with amin on cand_src gives minimal candidate src index per target (or big if none)
                temp = torch.full((num_targets,), big, dtype=torch.int64, device=device)
                temp.scatter_reduce_(0, tgt_flat_idx, cand_src, reduce="amin", include_self=True)
                # convert big -> -1
                has_candidate = temp != big
                best_src_for_target[has_candidate] = temp[has_candidate]

                # accepted sources are those best_src_for_target >=0
                accepted_src = best_src_for_target[best_src_for_target >= 0]  # list of flattened source indices

                # Reset all matched pixel_matching to -1, then re-enable accepted ones
                pixel_matching[matched_indices, :] = -1
                if accepted_src.numel() > 0:
                    # accepted_src contains flattened source indices; set their v,u
                    v_vals = v_int[accepted_src]
                    u_vals = u_int[accepted_src]
                    pixel_matching[accepted_src, 0] = v_vals
                    pixel_matching[accepted_src, 1] = u_vals

    # reshape and return numpy int32 on CPU (or the device tensor)
    pixel_matching = pixel_matching.reshape(H1, W1, 2)
    if return_tensor:
        return pixel_matching
    return pixel_matching.cpu().numpy().astype(np.int32)


def fuse_overlaps_tensors(colors_1, pts_world_1, conf_1, colors_2, pts_world_2, conf_2, eps: float = 1e-8):
    """fuse_overlaps_torch on device tensors (colors uint8 [N,3], pts float32 [N,3], conf float32 [N]); same arithmetic, no host round trip."""
    c1, c2 = colors_1.to(torch.float32), colors_2.to(torch.float32)
    fused_conf = conf_1 + conf_2
    safe_denom = torch.where(fused_conf > eps, fused_conf, torch.ones_like(fused_conf))
    zero_mask = fused_conf <= eps
    fused_colors = (conf_1.unsqueeze(1) * c1 + conf_2.unsqueeze(1) * c2) / safe_denom.unsqueeze(1)
    fused_pts = (conf_1.unsqueeze(1) * pts_world_1 + conf_2.unsqueeze(1) * pts_world_2) / safe_denom.unsqueeze(1)
    if zero_mask.any():
        fused_colors[zero_mask] = 0.0
        fused_pts[zero_mask] = 0.0
    return torch.clamp(torch.round(fused_colors), 0, 255).to(torch.uint8), fused_pts, fused_conf


def fuse_submap_frames(imgs, confs, pts_world_batch_t, depths_t, t_view_world_batch_t, intrinsics_params, ts_keys, overlap_ph_views,
                       point_cloud_fusion: bool, gpu: bool = True):
    """Sequential per-frame point fusion of one submap (the loop formerly inline in ScaRFSLAM.optimize_submap).
    Frame i's back-projected pixels are matched against the previous `overlap_ph_views` frames' points (depth-consistent
    reprojection with unique targets); matched pixels fuse into the existing point (confidence-weighted), the rest append.
    gpu=True keeps every array on the device (integer bookkeeping identical, float fusion the same ops); gpu=False is the
    original numpy loop. Returns (submap_pts_world [P,4] float32 = xyz+conf, submap_colors [P,3] uint8, {ts_key: point-id map [H,W] int64}).
    """
    N, H, W, _ = imgs.shape
    device = pts_world_batch_t.device
    if not gpu:
        return _fuse_submap_frames_numpy(imgs, confs, pts_world_batch_t, depths_t, t_view_world_batch_t, intrinsics_params, ts_keys, overlap_ph_views, point_cloud_fusion)
    imgs_t = torch.as_tensor(imgs, device=device)
    confs_t = torch.as_tensor(confs, dtype=torch.float32, device=device)
    submap_point_ids = {}
    pts_store = torch.empty((0, 4), dtype=torch.float32, device=device)
    col_store = torch.empty((0, 3), dtype=torch.uint8, device=device)
    count = 0

    def _ensure(min_capacity):
        nonlocal pts_store, col_store
        cap = pts_store.shape[0]
        if cap >= min_capacity:
            return
        new_cap = max(min_capacity, 1024 if cap == 0 else cap * 2)
        while new_cap < min_capacity:
            new_cap *= 2
        new_pts = torch.empty((new_cap, 4), dtype=torch.float32, device=device); new_col = torch.empty((new_cap, 3), dtype=torch.uint8, device=device)
        if count > 0:
            new_pts[:count] = pts_store[:count]; new_col[:count] = col_store[:count]
        pts_store, col_store = new_pts, new_col

    for i in range(N):
        ids = torch.full((H, W), -1, dtype=torch.int64, device=device)
        if count > 0:
            for project_idx in range(i - 1, max(-1, i - overlap_ph_views - 1), -1):
                if project_idx == i or ts_keys[project_idx] not in submap_point_ids:
                    continue
                ids_old = submap_point_ids[ts_keys[project_idx]]
                if point_cloud_fusion:
                    matching = get_matching_torch(pts_world_1=pts_world_batch_t[i], mask_1=(ids == -1), depth_2=depths_t[project_idx],
                                                  T_view_world_2=t_view_world_batch_t[project_idx], intrinsics_2=intrinsics_params[project_idx],
                                                  depth_thresh=0.05, unique_mapping=True, return_tensor=True)
                else:
                    matching = torch.full((H, W, 2), -1, dtype=torch.int64, device=device)
                matching[ids != -1] = -1
                rows, cols = matching[:, :, 0], matching[:, :, 1]
                fill_mask = (rows >= 0) & (cols >= 0)
                if bool(fill_mask.any()):
                    prev_ids = ids_old[rows[fill_mask], cols[fill_mask]]
                    existing_ids = ids[ids != -1]
                    unique_mask = ~torch.isin(prev_ids, existing_ids)
                    fill_idx = torch.nonzero(fill_mask.reshape(-1), as_tuple=False).squeeze(1)
                    ids.view(-1)[fill_idx[unique_mask]] = prev_ids[unique_mask]
                if bool((ids != -1).all()):
                    break
            fill_mask = ids != -1
            if bool(fill_mask.any()):
                matched_ids = ids[fill_mask]
                fused_colors, fused_pts, fused_conf = fuse_overlaps_tensors(
                    imgs_t[i][fill_mask], pts_world_batch_t[i][fill_mask], confs_t[i][fill_mask],
                    col_store[matched_ids], pts_store[matched_ids, :3], pts_store[matched_ids, 3])
                col_store[matched_ids] = fused_colors
                pts_store[matched_ids, :3] = fused_pts
                pts_store[matched_ids, 3] = fused_conf
        vals = ids[ids != -1]
        if torch.unique(vals).numel() != vals.numel():
            raise ValueError("pts_world_ids contains duplicate IDs (excluding -1)")
        flat_unmatched = (ids == -1).reshape(-1)
        n_new = int(flat_unmatched.sum())
        if n_new > 0:
            _ensure(count + n_new)
            pts_store[count:count + n_new, :3] = pts_world_batch_t[i].reshape(-1, 3)[flat_unmatched]
            pts_store[count:count + n_new, 3] = confs_t[i].reshape(-1)[flat_unmatched]
            col_store[count:count + n_new] = imgs_t[i].reshape(-1, 3)[flat_unmatched]
            ids.view(-1)[flat_unmatched] = torch.arange(count, count + n_new, dtype=torch.int64, device=device)
            count += n_new
        submap_point_ids[ts_keys[i]] = ids
    return (pts_store[:count].cpu().numpy(), col_store[:count].cpu().numpy(), {k: v.cpu().numpy() for k, v in submap_point_ids.items()})


def _fuse_submap_frames_numpy(imgs, confs, pts_world_batch_t, depths_t, t_view_world_batch_t, intrinsics_params, ts_keys, overlap_ph_views, point_cloud_fusion):
    """the original numpy fusion loop, kept as the reference implementation (SCARF_FUSION_GPU=0)"""
    N, H, W, _ = imgs.shape
    pts_world_batch = pts_world_batch_t.cpu().numpy()
    submap_point_ids: Dict[str, np.ndarray] = {}
    submap_pts_world = np.empty((0, 4), dtype=np.float32)
    submap_colors = np.empty((0, 3), dtype=np.uint8)
    submap_point_count = 0

    def _ensure_submap_capacity(min_capacity: int) -> None:
        nonlocal submap_pts_world, submap_colors, submap_point_count
        current_capacity = submap_pts_world.shape[0]
        if current_capacity >= min_capacity:
            return
        new_capacity = max(min_capacity, 1024 if current_capacity == 0 else current_capacity * 2)
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

    for i in range(N):
        conf_i = confs[i]
        img_i = imgs[i]
        pts_world_i = pts_world_batch[i]
        pts_world_i_flat = pts_world_i.reshape(-1, 3)
        ts_key_i = ts_keys[i]
        pts_world_ids = np.full((H, W), -1, dtype=np.int64)
        if submap_point_count > 0:
            for project_idx in range(i - 1, max(-1, i - overlap_ph_views - 1), -1):
                if project_idx == i:
                    continue
                ts_key_old = ts_keys[project_idx]
                if ts_key_old not in submap_point_ids:
                    continue
                pts_world_ids_old = submap_point_ids[ts_key_old]
                if point_cloud_fusion:
                    matching_unique = get_matching_torch(
                        pts_world_1=pts_world_batch_t[i], mask_1=(pts_world_ids == -1), depth_2=depths_t[project_idx],
                        T_view_world_2=t_view_world_batch_t[project_idx], intrinsics_2=intrinsics_params[project_idx],
                        depth_thresh=0.05, unique_mapping=True)
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
                fused_colors, fused_pts, fused_conf = fuse_overlaps_torch(
                    colors_1=colors_new, pts_world_1=pts_new, conf_1=conf_new,
                    colors_2=colors_matched, pts_world_2=pts_matched, conf_2=conf_matched)
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
    return (np.ascontiguousarray(submap_pts_world[:submap_point_count]), np.ascontiguousarray(submap_colors[:submap_point_count]), submap_point_ids)
