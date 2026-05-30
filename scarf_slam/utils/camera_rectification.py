import numpy as np
import cv2
import math

# -------------------------
# Helpers: rotations & pinhole-equidistant projection
# -------------------------
def _rotx(a):
    ca = math.cos(a); sa = math.sin(a)
    return np.array([[1.0, 0.0, 0.0],
                     [0.0, ca, -sa],
                     [0.0, sa, ca]], dtype=float)

def _roty(a):
    ca = math.cos(a); sa = math.sin(a)
    return np.array([[ca, 0.0, sa],
                     [0.0, 1.0, 0.0],
                     [-sa, 0.0, ca]], dtype=float)

def _rotz(a):
    ca = math.cos(a); sa = math.sin(a)
    return np.array([[ca, -sa, 0.0],
                     [sa, ca, 0.0],
                     [0.0, 0.0, 1.0]], dtype=float)


def rectify_pinhole_equidistant_vectorized(
    fisheye_img,
    fisheye_intrinsics,
    distortion_coeffs,
    pinhole_intrinsics,
    pinhole_resolution,
    yaw=0.0,
    pitch=0.0,
    roll=0.0
):
    if isinstance(pinhole_intrinsics, dict):
        fx_p = float(pinhole_intrinsics["fx"]); fy_p = float(pinhole_intrinsics["fy"])
        cx_p = float(pinhole_intrinsics["cx"]); cy_p = float(pinhole_intrinsics["cy"])
    else:
        fx_p, fy_p, cx_p, cy_p = map(float, pinhole_intrinsics)
    w_p, h_p = int(pinhole_resolution[0]), int(pinhole_resolution[1])

    yaw_r = math.radians(yaw); pitch_r = math.radians(pitch); roll_r = math.radians(roll)
    R = (_rotz(roll_r) @ _rotx(pitch_r) @ _roty(yaw_r)).T  # R_view_to_cam

    if len(fisheye_intrinsics) != 4:
        raise ValueError("fisheye_intrinsics must contain [fx, fy, cx, cy]")
    if len(distortion_coeffs) != 4:
        raise ValueError("distortion_coeffs must contain [k1, k2, k3, k4]")
    fx_f, fy_f, cx_f, cy_f = map(float, fisheye_intrinsics)
    k1, k2, k3, k4 = map(float, distortion_coeffs)

    u_coords = np.arange(w_p, dtype=np.float32)
    v_coords = np.arange(h_p, dtype=np.float32)
    uu, vv = np.meshgrid(u_coords, v_coords)

    x_norm = (uu - cx_p) / fx_p
    y_norm = (vv - cy_p) / fy_p
    ones = np.ones_like(x_norm, dtype=np.float32)
    dirs_view = np.stack((x_norm, y_norm, ones), axis=-1)

    norms = np.linalg.norm(dirs_view, axis=2, keepdims=True)
    safe = norms > 1e-12
    dirs_view[safe[...,0] == False] = np.array([0.0,0.0,1.0], dtype=np.float32)
    dirs_view /= norms

    dir_cam = dirs_view @ R.T

    X = dir_cam[...,0].astype(np.float64)
    Y = dir_cam[...,1].astype(np.float64)
    Z = dir_cam[...,2].astype(np.float64)

    r = np.sqrt(X * X + Y * Y)
    theta = np.arctan2(r, Z)
    theta2 = theta * theta
    theta4 = theta2 * theta2
    theta6 = theta4 * theta2
    theta8 = theta4 * theta4
    theta_d = theta * (1.0 + k1 * theta2 + k2 * theta4 + k3 * theta6 + k4 * theta8)
    scale = np.zeros_like(theta_d)
    valid = Z > 0.0
    nonzero_radius = valid & (r > 1e-12)
    scale[nonzero_radius] = theta_d[nonzero_radius] / r[nonzero_radius]
    scale[(Z > 0.0) & (r <= 1e-12)] = 1.0

    u_f = fx_f * X * scale + cx_f
    v_f = fy_f * Y * scale + cy_f

    fh, fw = fisheye_img.shape[:2]
    inside = valid & (u_f >= 0.0) & (u_f < fw) & (v_f >= 0.0) & (v_f < fh)

    map_x = np.full_like(u_f, -1.0, dtype=np.float32)
    map_y = np.full_like(v_f, -1.0, dtype=np.float32)
    map_x[inside] = u_f[inside].astype(np.float32)
    map_y[inside] = v_f[inside].astype(np.float32)

    remapped = cv2.remap(fisheye_img, map_x, map_y, interpolation=cv2.INTER_LINEAR,
                         borderMode=cv2.BORDER_CONSTANT, borderValue=0)

    T_cam_view = np.eye(4, dtype=np.float64)
    T_cam_view[:3,:3] = R
    return remapped, T_cam_view
