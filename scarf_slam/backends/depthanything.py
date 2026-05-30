#!/usr/bin/env python3
# Optional config for better memory efficiency
# -------------------------
# Standard Library Imports
# -------------------------
import os
import pathlib
import sys

# Environment variables
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"

# -------------------------
# Third-Party Imports
# -------------------------
import numpy as np
import torch

# -------------------------
# Local Package Imports
# -------------------------

from scarf_slam.core.timestamp import MappingTimestamp

PACKAGE_DIR = pathlib.Path(__file__).resolve().parents[1]
REPO_ROOT = PACKAGE_DIR.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from scarf_slam.utils import camera_rectification


def _format_mib(num_bytes):
    return f"{num_bytes / (1024 ** 2):.2f} MiB"


def log_cuda_memory(tag):
    if not torch.cuda.is_available():
        print(f"[depthanything3 cuda] {tag}: CUDA not available")
        return

    device = torch.cuda.current_device()
    props = torch.cuda.get_device_properties(device)
    allocated = torch.cuda.memory_allocated(device)
    reserved = torch.cuda.memory_reserved(device)
    max_allocated = torch.cuda.max_memory_allocated(device)
    max_reserved = torch.cuda.max_memory_reserved(device)

    print(
        f"[depthanything3 cuda] {tag}: "
        f"device={props.name}, "
        f"allocated={_format_mib(allocated)}, "
        f"reserved={_format_mib(reserved)}, "
        f"peak_allocated={_format_mib(max_allocated)}, "
        f"peak_reserved={_format_mib(max_reserved)}"
    )


def _load_image_data(app, cam_name, timestamp):
    slam_bag_data = getattr(app, "slam_bag_data", None)
    if slam_bag_data is None:
        raise RuntimeError("DepthAnything backend requires app.slam_bag_data for image loading")
    return slam_bag_data.decode_image(timestamp, cam_name=cam_name)


def do_processing(app, ts_sub, in_poses_dict, ph_views_per_batch, use_extrinsics=True):
    print("\n\n=== DA3 Inference ===")
    print (app.pinhole_camera.intrinsics_mtx)
    num_views = len(ts_sub) * len(ph_views_per_batch[0])
    intrinsics_array = np.stack([app.pinhole_camera.intrinsics_mtx.astype(np.float32)] * num_views)

    new_ph_to_ref_dict = {}
    image_data_lst = []
    extrinsics_lst = []
    ph_view_poses_sub = []
    for index, this_time_str in enumerate(ts_sub):
        this_pose = in_poses_dict[this_time_str]
        
        ph_views_current_pose = ph_views_per_batch[index]
        for index_innr, ph_view in enumerate(ph_views_current_pose):
            cam_name = ph_view.cam
            yaw, pitch, roll = ph_view.yaw, ph_view.pitch, ph_view.roll

            image_data = _load_image_data(app, cam_name, this_time_str)
            if cam_name not in app.fisheye_cameras:
                T_cam_view = np.eye(4)
            elif (
                app.fisheye_cameras[cam_name].camera_model == "pinhole"
                and app.fisheye_cameras[cam_name].distortion_model == "equidistant"
            ):
                image_data, T_cam_view = camera_rectification.rectify_pinhole_equidistant_vectorized(
                    fisheye_img=image_data,
                    fisheye_intrinsics=app.fisheye_cameras[cam_name].intrinsics,
                    distortion_coeffs=app.fisheye_cameras[cam_name].distortion_coeffs,
                    pinhole_intrinsics=app.pinhole_camera.intrinsics,
                    pinhole_resolution=app.pinhole_camera.resolution,
                    yaw=yaw,
                    pitch=pitch
                )
            else:
                raise ValueError("Fisheye camera must use pinhole camera model with equidistant distortion")
            image_data_lst.append(image_data)

            T_world_cam0 = app.transforms.pose_to_matrix(this_pose)
            T_cam0_view = None
            T_world_view = None
            if cam_name == "cam0":
                T_cam0_view = T_cam_view
                T_world_view = T_world_cam0 @ T_cam0_view
            elif cam_name == "cam1":
                if "cam0" not in app.fisheye_cameras or "cam1" not in app.fisheye_cameras:
                    raise KeyError("cam1 processing requires fisheye cam0 and cam1 extrinsics in config")
                T_cam0_imu = app.fisheye_cameras["cam0"].T_cam_imu
                T_cam1_imu = app.fisheye_cameras["cam1"].T_cam_imu
                T_cam0_cam1 = T_cam0_imu @ np.linalg.inv(T_cam1_imu)
                T_cam0_view = T_cam0_cam1 @ T_cam_view
                T_world_view = T_world_cam0 @ T_cam0_view
            else:
                raise ValueError("Invalid camera name, should be either cam0 or cam1")
            
            extrinsics_lst.append(np.linalg.inv(T_world_view))

            sec_str, nsec_str = this_time_str.split("_")
            ref_sec, ref_nsec = int(sec_str), int(nsec_str)
            ph_sec, ph_nsec = int(sec_str), int(nsec_str)+index_innr
            ph_view_poses_sub.append((MappingTimestamp(sec=ph_sec, nsec=ph_nsec), app.transforms.matrix_to_pose(T_world_view)))
            new_ph_to_ref_dict[f"{ph_sec:010d}_{ph_nsec:09d}"] = (MappingTimestamp(sec=ref_sec, nsec=ref_nsec), app.transforms.matrix_to_pose(T_cam0_view))

    # if torch.cuda.is_available():
    #     torch.cuda.synchronize()
    #     torch.cuda.reset_peak_memory_stats()
    #     log_cuda_memory("before inference")

    if use_extrinsics:
        predictions = app.model.inference(
            image=image_data_lst,
            extrinsics=np.stack(extrinsics_lst, axis=0).astype(np.float32),
            intrinsics=intrinsics_array,
            align_to_input_ext_scale=True,
            ref_view_strategy="saddle_balanced",
        )
    else:
        predictions = app.model.inference(
            image=image_data_lst,
            intrinsics=intrinsics_array,
        )

    # if torch.cuda.is_available():
    #     torch.cuda.synchronize()
    #     log_cuda_memory("after inference")

    predictions.extrinsics = predictions.extrinsics[:, :3, :]

    return predictions, ph_view_poses_sub, new_ph_to_ref_dict
