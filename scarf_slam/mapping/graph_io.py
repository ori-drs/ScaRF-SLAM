import json
from pathlib import Path
from typing import Dict, List, Optional, Set, Tuple

import numpy as np

from scarf_slam.core.pose import MappingPose

GRAPH_SCHEMA_VERSION = 1


def pose_to_graph_json(pose: MappingPose) -> Dict[str, List[float]]:
    return {
        "position": [float(v) for v in pose.pos],
        "quaternion_xyzw": [float(v) for v in pose.quat],
    }


def graph_json_to_pose(pose_json: Dict[str, object]) -> MappingPose:
    if not isinstance(pose_json, dict):
        raise ValueError(f"Invalid pose JSON entry: {pose_json!r}")
    position = pose_json.get("position")
    quat = pose_json.get("quaternion_xyzw")
    if position is None or quat is None:
        raise KeyError(f"Pose JSON entry must contain position and quaternion_xyzw: {pose_json!r}")
    return MappingPose(
        [float(v) for v in position],
        [float(v) for v in quat],
    )


def pose_dict_to_graph_json(
    poses_dict: Dict[str, MappingPose],
) -> Dict[str, Dict[str, List[float]]]:
    return {
        ts_key: pose_to_graph_json(poses_dict[ts_key])
        for ts_key in sorted(poses_dict.keys())
    }


def set_graph_to_graph_json(graph: Dict[str, Set[str]]) -> Dict[str, List[str]]:
    return {
        key: sorted(str(v) for v in values)
        for key, values in sorted(graph.items())
    }


def split_graph_pair_key(pair_key: str) -> Tuple[str, str]:
    key_a, sep, key_b = pair_key.partition("-")
    if sep != "-" or not key_a or not key_b:
        raise ValueError(f"Invalid graph pair key '{pair_key}'. Expected '<key_a>-<key_b>'.")
    return key_a, key_b


def pair_key_to_filename(key_a: str, key_b: str, suffix: str) -> str:
    return f"{key_a}-{key_b}{suffix}"


def load_graph_array(
    path: Path,
    *,
    dtype: Optional[np.dtype] = None,
    ndim: Optional[int] = None,
    shape_tail: Optional[Tuple[int, ...]] = None,
) -> np.ndarray:
    if not path.exists():
        raise FileNotFoundError(f"Missing graph array: {path}")
    array = np.load(path, allow_pickle=False)
    if not isinstance(array, np.ndarray):
        raise ValueError(f"{path}: np.load did not return an ndarray.")
    if ndim is not None and array.ndim != ndim:
        raise ValueError(f"{path}: expected ndim={ndim}, got shape {array.shape}.")
    if shape_tail is not None and tuple(array.shape[-len(shape_tail) :]) != tuple(shape_tail):
        raise ValueError(f"{path}: expected trailing shape {shape_tail}, got {array.shape}.")
    if dtype is not None:
        array = array.astype(dtype, copy=False)
    return np.ascontiguousarray(array)


def save_graph_array(path: Path, array: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    np.save(path, np.ascontiguousarray(array))


def load_graph_manifest(graph_dir: Path) -> Dict[str, object]:
    manifest_path = graph_dir / "manifest.json"
    if not manifest_path.exists():
        raise FileNotFoundError(f"Previous graph manifest not found: {manifest_path}")
    with open(manifest_path, "r", encoding="utf-8") as f:
        manifest = json.load(f)
    if not isinstance(manifest, dict):
        raise ValueError(f"Previous graph manifest must be a JSON object: {manifest_path}")
    if "schema_version" not in manifest:
        raise KeyError(f"Previous graph manifest is missing schema_version: {manifest_path}")
    schema_version = int(manifest["schema_version"])
    if schema_version != GRAPH_SCHEMA_VERSION:
        raise ValueError(
            f"Unsupported previous graph schema_version={schema_version}; expected {GRAPH_SCHEMA_VERSION}."
        )
    return manifest


def require_graph_key(mapping: Dict[str, object], key: str, context: str):
    if key not in mapping:
        raise KeyError(f"{context} is missing required key '{key}'.")
    return mapping[key]
