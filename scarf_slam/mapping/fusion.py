from typing import List, Optional, Tuple, Union

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
) -> np.ndarray:
    """
    GPU-first PyTorch implementation. Inputs are numpy, output is numpy.

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

    # reshape and return numpy int32 on CPU
    pixel_matching = pixel_matching.reshape(H1, W1, 2).cpu().numpy().astype(np.int32)
    return pixel_matching
