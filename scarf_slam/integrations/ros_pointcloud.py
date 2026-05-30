import numpy as np


def point_cloud_xyzrgb(points: np.ndarray, colors: np.ndarray, parent_frame: str):
    from sensor_msgs.msg import PointCloud2, PointField
    from std_msgs.msg import Header

    points = np.ascontiguousarray(points[:, :3].astype(np.float32, copy=False))
    colors = np.ascontiguousarray(colors.astype(np.uint8, copy=False))
    if points.ndim != 2 or points.shape[1] != 3:
        raise ValueError(f"points must have shape (N, 3), got {points.shape}")
    if colors.ndim != 2 or colors.shape[1] != 3:
        raise ValueError(f"colors must have shape (N, 3), got {colors.shape}")
    if colors.shape[0] != points.shape[0]:
        raise ValueError(f"Point/color mismatch: points={points.shape[0]}, colors={colors.shape[0]}")

    rgb = (
        (colors[:, 0].astype(np.uint32) << 16)
        | (colors[:, 1].astype(np.uint32) << 8)
        | colors[:, 2].astype(np.uint32)
    )
    cloud = np.empty(
        points.shape[0],
        dtype=np.dtype(
            [
                ("x", np.float32),
                ("y", np.float32),
                ("z", np.float32),
                ("rgb", np.uint32),
            ]
        ),
    )
    cloud["x"] = points[:, 0]
    cloud["y"] = points[:, 1]
    cloud["z"] = points[:, 2]
    cloud["rgb"] = rgb

    fields = [
        PointField(name="x", offset=0, datatype=PointField.FLOAT32, count=1),
        PointField(name="y", offset=4, datatype=PointField.FLOAT32, count=1),
        PointField(name="z", offset=8, datatype=PointField.FLOAT32, count=1),
        PointField(name="rgb", offset=12, datatype=PointField.UINT32, count=1),
    ]
    point_step = cloud.dtype.itemsize
    return PointCloud2(
        header=Header(frame_id=parent_frame),
        height=1,
        width=points.shape[0],
        is_dense=False,
        is_bigendian=False,
        fields=fields,
        point_step=point_step,
        row_step=point_step * points.shape[0],
        data=cloud.tobytes(),
    )
