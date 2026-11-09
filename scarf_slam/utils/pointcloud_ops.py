from typing import Optional, Tuple

import numpy as np
import torch

from scarf_slam.core.pose import MappingPose, MappingTransforms
from scarf_slam.core.submap import SubmapRecord


def depth_to_world_points_vectorized(depth, intrinsics, extrinsics, device=None):
    """
    Convert a batch of depth maps to world-frame points.

    Args:
        depth: [N, H, W] numpy array or torch tensor.
        intrinsics: [N, 3, 3] numpy array or torch tensor.
        extrinsics: [N, 3, 4] world-to-camera transforms.
        device: optional torch device for computation.

    Returns:
        [N, H, W, 3] points in the same container type as depth.
    """
    input_is_numpy = isinstance(depth, np.ndarray)
    if input_is_numpy:
        depth_tensor = torch.tensor(depth, dtype=torch.float32)
        intrinsics_tensor = torch.tensor(intrinsics, dtype=torch.float32)
        extrinsics_tensor = torch.tensor(extrinsics, dtype=torch.float32)
    else:
        depth_tensor = depth
        intrinsics_tensor = intrinsics
        extrinsics_tensor = extrinsics

    if depth_tensor.ndim != 3:
        raise ValueError(f"depth must have shape [N, H, W], got {tuple(depth_tensor.shape)}")
    if intrinsics_tensor.ndim != 3 or tuple(intrinsics_tensor.shape[-2:]) != (3, 3):
        raise ValueError(
            f"intrinsics must have shape [N, 3, 3], got {tuple(intrinsics_tensor.shape)}"
        )
    if extrinsics_tensor.ndim != 3 or tuple(extrinsics_tensor.shape[-2:]) != (3, 4):
        raise ValueError(
            f"extrinsics must have shape [N, 3, 4], got {tuple(extrinsics_tensor.shape)}"
        )
    if (
        depth_tensor.shape[0] != intrinsics_tensor.shape[0]
        or depth_tensor.shape[0] != extrinsics_tensor.shape[0]
    ):
        raise ValueError(
            "depth, intrinsics, and extrinsics must have the same batch size: "
            f"depth={tuple(depth_tensor.shape)}, intrinsics={tuple(intrinsics_tensor.shape)}, "
            f"extrinsics={tuple(extrinsics_tensor.shape)}"
        )

    if device is not None:
        depth_tensor = depth_tensor.to(device)
        intrinsics_tensor = intrinsics_tensor.to(device)
        extrinsics_tensor = extrinsics_tensor.to(device)

    n, height, width = depth_tensor.shape
    tensor_device = depth_tensor.device

    u = torch.arange(width, device=tensor_device).float().view(1, 1, width, 1).expand(n, height, width, 1)
    v = torch.arange(height, device=tensor_device).float().view(1, height, 1, 1).expand(n, height, width, 1)
    ones = torch.ones((n, height, width, 1), device=tensor_device)
    pixel_coords = torch.cat([u, v, ones], dim=-1)

    intrinsics_inv = torch.inverse(intrinsics_tensor)
    camera_coords = torch.einsum("nij,nhwj->nhwi", intrinsics_inv, pixel_coords)
    camera_coords = camera_coords * depth_tensor.unsqueeze(-1)
    camera_coords_homo = torch.cat([camera_coords, ones], dim=-1)

    extrinsics_4x4 = torch.zeros(n, 4, 4, device=tensor_device)
    extrinsics_4x4[:, :3, :4] = extrinsics_tensor
    extrinsics_4x4[:, 3, 3] = 1.0

    c2w = torch.inverse(extrinsics_4x4)
    world_coords_homo = torch.einsum("nij,nhwj->nhwi", c2w, camera_coords_homo)
    point_cloud_world = world_coords_homo[..., :3]

    if input_is_numpy:
        point_cloud_world = point_cloud_world.cpu().numpy()

    return point_cloud_world


def world_points_to_anchor_local(
    pts_world: np.ndarray,
    anchor_pose: MappingPose,
    transforms: MappingTransforms,
) -> np.ndarray:
    pts_world = np.asarray(pts_world, dtype=np.float32)
    if pts_world.ndim != 2 or pts_world.shape[1] != 3:
        raise ValueError(f"pts_world must have shape (N, 3), got {pts_world.shape}")
    if pts_world.shape[0] == 0:
        return np.empty((0, 3), dtype=np.float32)

    t_world_anchor = transforms.pose_to_matrix(anchor_pose).astype(np.float32)
    r_world_anchor = t_world_anchor[:3, :3]
    t_world_anchor_vec = t_world_anchor[:3, 3]
    pts_local = (pts_world - t_world_anchor_vec) @ r_world_anchor
    return np.ascontiguousarray(pts_local, dtype=np.float32)


def submap_to_world_pointcloud(
    submap: SubmapRecord,
    anchor_pose: MappingPose,
    transforms: MappingTransforms,
    point_mask: Optional[np.ndarray] = None,
) -> Tuple[np.ndarray, np.ndarray]:
    local_points = np.asarray(submap.local_points)
    colors = np.asarray(submap.colors)
    if point_mask is not None:
        point_mask = np.asarray(point_mask, dtype=bool)
        if point_mask.ndim != 1 or point_mask.shape[0] != local_points.shape[0]:
            raise ValueError(
                "point_mask must be a 1D array with one entry per submap point: "
                f"mask={point_mask.shape}, points={local_points.shape[0]}"
            )
        local_points = local_points[point_mask]
        colors = colors[point_mask]
    local_points = np.asarray(local_points, dtype=np.float32)
    colors = np.asarray(colors, dtype=np.uint8)
    if local_points.shape[0] == 0:
        return (
            np.empty((0, 4), dtype=np.float32),
            np.empty((0, 3), dtype=np.uint8),
        )

    t_world_anchor = transforms.pose_to_matrix(anchor_pose).astype(np.float32)
    r_world_anchor = t_world_anchor[:3, :3]
    t_world_anchor_vec = t_world_anchor[:3, 3]
    scaled_local_xyz = float(submap.scale) * local_points[:, :3]
    pts_world = scaled_local_xyz @ r_world_anchor.T + t_world_anchor_vec
    pts_world_with_conf = np.concatenate(
        [pts_world.astype(np.float32, copy=False), local_points[:, 3:4]],
        axis=1,
    )
    return np.ascontiguousarray(pts_world_with_conf), np.ascontiguousarray(colors)


def _voxel_downsample_mean(xyz: np.ndarray, rgb: np.ndarray, voxel_size: float, device: str) -> Tuple[np.ndarray, np.ndarray]:
    pts = torch.from_numpy(np.ascontiguousarray(xyz, dtype=np.float32)).to(device)
    key = torch.floor(pts / voxel_size).to(torch.int64)
    key -= key.min(0).values
    dims = key.max(0).values + 1
    lin = (key[:, 0] * dims[1] + key[:, 1]) * dims[2] + key[:, 2]
    del key
    uniq, inv = torch.unique(lin, return_inverse=True)
    del lin
    n = len(uniq)
    del uniq
    cnt = torch.bincount(inv, minlength=n).unsqueeze(1).to(torch.float32)
    out_xyz = torch.zeros((n, 3), dtype=torch.float32, device=device).index_add_(0, inv, pts).div_(cnt)
    del pts
    cols = torch.from_numpy(np.ascontiguousarray(rgb, dtype=np.float32)).to(device)
    out_rgb = torch.zeros((n, 3), dtype=torch.float32, device=device).index_add_(0, inv, cols).div_(cnt)
    return out_xyz.cpu().numpy(), out_rgb.cpu().numpy()


def write_ply_xyzrgb(path: str, xyz: np.ndarray, rgb: np.ndarray) -> None:
    """Binary little-endian PLY (float32 x y z, uint8 red green blue) written in one numpy write: no size limit, 15 bytes per
    point. rgb is uint8 0-255, or float in [0, 1]."""
    xyz = np.asarray(xyz)
    rgb = np.asarray(rgb)
    if rgb.dtype != np.uint8:
        rgb = np.clip(np.rint(rgb.astype(np.float32) * 255.0), 0, 255).astype(np.uint8)
    rec = np.empty(len(xyz), dtype=[("x", "<f4"), ("y", "<f4"), ("z", "<f4"), ("red", "u1"), ("green", "u1"), ("blue", "u1")])
    rec["x"], rec["y"], rec["z"] = xyz[:, 0], xyz[:, 1], xyz[:, 2]
    rec["red"], rec["green"], rec["blue"] = rgb[:, 0], rgb[:, 1], rgb[:, 2]
    header = (
        "ply\nformat binary_little_endian 1.0\n"
        f"element vertex {len(xyz)}\n"
        "property float x\nproperty float y\nproperty float z\n"
        "property uchar red\nproperty uchar green\nproperty uchar blue\nend_header\n"
    )
    with open(path, "wb") as handle:
        handle.write(header.encode("ascii"))
        rec.tofile(handle)


def voxel_downsample_mean(xyz: np.ndarray, rgb: np.ndarray, voxel_size: float) -> Tuple[np.ndarray, np.ndarray]:
    """Mean position and colour of the points in each occupied voxel (voxel index = floor(p / voxel_size)).
    Runs on the GPU when one is available (~1 s for 1.8e8 points), otherwise, or if the GPU runs out of
    memory, on the CPU (~11 s)."""
    if torch.cuda.is_available():
        try:
            return _voxel_downsample_mean(xyz, rgb, voxel_size, "cuda")
        except torch.cuda.OutOfMemoryError:
            pass
        finally:
            torch.cuda.empty_cache()
    return _voxel_downsample_mean(xyz, rgb, voxel_size, "cpu")
