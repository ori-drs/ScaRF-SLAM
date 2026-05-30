#!/usr/bin/env python3

import argparse
from pathlib import Path

import open3d as o3d


def parse_downsample(value: str) -> float:
    ratio = float(value)
    if ratio < 0.0 or ratio > 1.0:
        raise argparse.ArgumentTypeError("--downsample must be between 0 and 1")
    return ratio


def load_cloud(path: Path, downsample: float) -> o3d.geometry.PointCloud:
    if not path.is_file():
        raise FileNotFoundError(f"PCD file does not exist: {path}")

    cloud = o3d.io.read_point_cloud(str(path))
    if cloud.is_empty():
        raise RuntimeError(f"Point cloud is empty or unreadable: {path}")

    if downsample < 1.0:
        cloud = cloud.random_down_sample(downsample)

    return cloud


def visualize_clouds(
    clouds: list[o3d.geometry.PointCloud],
    point_size: float,
    window_name: str,
) -> None:
    vis = o3d.visualization.Visualizer()
    vis.create_window(window_name=window_name)
    for cloud in clouds:
        vis.add_geometry(cloud)

    render_option = vis.get_render_option()
    render_option.point_size = point_size

    vis.run()
    vis.destroy_window()


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Visualize one or more PCD point clouds.")
    parser.add_argument(
        "--pcd",
        nargs="+",
        required=True,
        help="One or more .pcd files to visualize in the same Open3D window.",
    )
    parser.add_argument(
        "--point-size",
        type=float,
        default=2.0,
        help="Rendered point size. Default: 2.0",
    )
    parser.add_argument(
        "--downsample",
        type=parse_downsample,
        default=1.0,
        help="Random sampling ratio in [0, 1] applied to each input cloud before visualization. Default: 1.0",
    )
    parser.add_argument(
        "--window-name",
        default="PCD Viewer",
        help="Open3D visualization window title. Default: PCD Viewer",
    )
    return parser


def main() -> None:
    args = build_parser().parse_args()
    if args.point_size <= 0.0:
        raise ValueError(f"--point-size must be > 0, got {args.point_size}")

    clouds = []
    for pcd_path in args.pcd:
        path = Path(pcd_path).expanduser()
        cloud = load_cloud(path, args.downsample)
        print(f"{path}: {len(cloud.points)} points")
        clouds.append(cloud)

    visualize_clouds(
        clouds=clouds,
        point_size=args.point_size,
        window_name=args.window_name,
    )


if __name__ == "__main__":
    main()
