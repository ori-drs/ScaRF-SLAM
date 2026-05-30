import math
import logging
import time
from collections import defaultdict
from typing import TYPE_CHECKING, Any, Dict, List, Optional, Set, Tuple, TypeAlias

import numpy as np
from scarf_slam.core.pose import MappingTransforms

try:
    import gtsam
except ImportError:  # pragma: no cover
    gtsam = None

if TYPE_CHECKING:
    from gtsam import NonlinearFactorGraph as GtsamNonlinearFactorGraph
    from gtsam import Values as GtsamValues
else:
    GtsamValues: TypeAlias = Any
    GtsamNonlinearFactorGraph: TypeAlias = Any

_LOGGER = logging.getLogger(__name__)
_ANSI_YELLOW = "\033[33m"
_ANSI_RESET = "\033[0m"
_FRAME_LOG_PREFIX = "[gtsam-frame]"
_SUBMAP_LOG_PREFIX = "[gtsam-submap]"


def _build_match_observations(
    matches: Dict[Tuple[int, int], List[Tuple[int, int]]],
    keypoints: List[List[Any]],
) -> List[Tuple[int, int, float, float, float, float]]:
    """
    Convert matches (kp indices) + keypoints into per-match pixel observations.
    Returns list of (i, j, u_i, v_i, u_j, v_j).
    """
    obs = []
    for (i, j), pairs in matches.items():
        kps_i = keypoints[i]
        kps_j = keypoints[j]
        for qi, pj in pairs:
            u_i, v_i = kps_i[qi].pt
            u_j, v_j = kps_j[pj].pt
            obs.append((i, j, float(u_i), float(v_i), float(u_j), float(v_j)))
    return obs


def _prepare_geometry(predictions: Any) -> Dict[str, np.ndarray]:
    depth = np.asarray(predictions.depth)
    intrinsics = np.asarray(predictions.intrinsics, dtype=np.float64)
    extrinsics = np.asarray(predictions.extrinsics, dtype=np.float64)  # w2c, [N,3,4]

    n = depth.shape[0]
    k_inv = np.linalg.inv(intrinsics)

    r_w2c = extrinsics[:, :3, :3]
    t_w2c = extrinsics[:, :3, 3]

    r_c2w = np.transpose(r_w2c, (0, 2, 1))
    t_c2w = -np.einsum("nij,nj->ni", r_c2w, t_w2c)

    return {
        "depth": depth,
        "k": intrinsics,
        "k_inv": k_inv,
        "r_w2c": r_w2c,
        "t_w2c": t_w2c,
        "r_c2w": r_c2w,
        "t_c2w": t_c2w,
        "n": np.array([n], dtype=np.int64),
    }


def _unproject_world_linear_terms(
    u: float,
    v: float,
    depth: float,
    k_inv: np.ndarray,
    r_c2w: np.ndarray,
    t_c2w: np.ndarray,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    World point as affine function of depth scale s:
        X_w(s) = c * s + b
    where c,b are returned here.
    """
    pix = np.array([u, v, 1.0], dtype=np.float64)
    ray_cam = (k_inv @ pix) * depth
    c = r_c2w @ ray_cam
    b = t_c2w
    return c, b


def _make_variable_key(use_exp_param: bool, idx: int) -> int:
    return gtsam.symbol("a" if use_exp_param else "s", idx)


def _make_key_vector(*keys: int):
    key_vec = gtsam.KeyVector()
    for key in keys:
        key_vec.append(key)
    return key_vec


def _value_to_scale(value: float, use_exp_param: bool) -> float:
    return float(math.exp(value)) if use_exp_param else float(value)


def _build_connected_components(
    indices: List[int],
    edges: List[Tuple[int, int]],
) -> List[List[int]]:
    adjacency: Dict[int, Set[int]] = {idx: set() for idx in indices}
    for i, j in edges:
        adjacency.setdefault(i, set()).add(j)
        adjacency.setdefault(j, set()).add(i)

    remaining = set(indices)
    components: List[List[int]] = []
    while remaining:
        seed = remaining.pop()
        stack = [seed]
        comp = [seed]
        while stack:
            cur = stack.pop()
            for nxt in adjacency.get(cur, set()):
                if nxt in remaining:
                    remaining.remove(nxt)
                    stack.append(nxt)
                    comp.append(nxt)
        components.append(sorted(comp))
    return components


def _select_most_connected_anchor(
    comp_indices: List[int],
    comp_obs: List[Dict[str, Any]],
    frozen_indices: Set[int],
) -> int:
    """
    Select anchor node using component connectivity:
      1) maximum unique neighbor count (graph degree)
      2) maximum incident factor count
      3) closest to component midpoint
      4) lower index for deterministic tie-breaking
    Frozen nodes are excluded when possible.
    """
    comp_set = set(comp_indices)
    candidate_indices = [idx for idx in comp_indices if idx not in frozen_indices]
    if len(candidate_indices) == 0:
        candidate_indices = list(comp_indices)

    neighbor_sets: Dict[int, Set[int]] = {idx: set() for idx in comp_indices}
    incident_counts: Dict[int, int] = defaultdict(int)

    for m in comp_obs:
        i, j = m["i"], m["j"]
        if i not in comp_set or j not in comp_set or i == j:
            continue
        neighbor_sets[i].add(j)
        neighbor_sets[j].add(i)
        incident_counts[i] += 1
        incident_counts[j] += 1

    midpoint = 0.5 * (float(comp_indices[0]) + float(comp_indices[-1]))

    def _score(idx: int) -> Tuple[int, int, float, int]:
        return (
            len(neighbor_sets[idx]),
            incident_counts.get(idx, 0),
            -abs(float(idx) - midpoint),
            -idx,
        )

    return max(candidate_indices, key=_score)


def optimize_frame_scales_gtsam(
    predictions: Any,
    matches: Dict[Tuple[int, int], List[Tuple[int, int]]],
    keypoints: List[List[Any]],
    iters: int = 10,
    robust_delta: float = 1.0,
    use_exp_param: bool = True,
    reg_weight: float = 0.0,
    anchor_prior_sigma: Optional[float] = 0.05,
    normalize_mean: bool = False,
    print_scales_each_iter: bool = False,
    force_full_iters: bool = False,
    min_matches_for_node_freeze: int = 0,
    min_matches_for_edge_drop: int = 0,
) -> np.ndarray:
    """
    Optimize per-image depth scales with GTSAM using 3D point distance residuals.
    """
    if gtsam is None:
        raise ImportError("gtsam is required for optimize_frame_scales_gtsam. Please install python-gtsam.")

    start_total_time = time.perf_counter()
    geom = _prepare_geometry(predictions)
    depth = geom["depth"]
    k_inv = geom["k_inv"]
    r_c2w = geom["r_c2w"]
    t_c2w = geom["t_c2w"]
    n = int(geom["n"][0])

    obs_raw = _build_match_observations(matches, keypoints)
    if len(obs_raw) == 0:
        msg = "No matches provided for optimization; skip optimization and return all-one scales."
        print(f"{_ANSI_YELLOW}{_FRAME_LOG_PREFIX} {msg}{_ANSI_RESET}", flush=True)
        return np.ones((n,), dtype=np.float32)

    start_observation_time = time.perf_counter()
    obs = []
    h, w = depth.shape[1], depth.shape[2]
    for (i, j, u_i, v_i, u_j, v_j) in obs_raw:
        ui = int(u_i)
        vi = int(v_i)
        uj = int(u_j)
        vj = int(v_j)
        if ui < 0 or ui >= w or vi < 0 or vi >= h:
            continue
        if uj < 0 or uj >= w or vj < 0 or vj >= h:
            continue

        d_i = float(depth[i, vi, ui])
        d_j = float(depth[j, vj, uj])
        if not (np.isfinite(d_i) and np.isfinite(d_j)):
            continue
        if d_i <= 0 or d_j <= 0:
            continue

        c_i, b_i = _unproject_world_linear_terms(u_i, v_i, d_i, k_inv[i], r_c2w[i], t_c2w[i])
        c_j, b_j = _unproject_world_linear_terms(u_j, v_j, d_j, k_inv[j], r_c2w[j], t_c2w[j])

        obs.append(
            {
                "i": i,
                "j": j,
                "c_i": c_i,
                "b_i": b_i,
                "c_j": c_j,
                "b_j": b_j,
            }
        )

    if len(obs) == 0:
        raise ValueError("All matches invalid after depth filtering.")
    end_observation_time = time.perf_counter()

    frozen_indices_global: Set[int] = set()
    if min_matches_for_edge_drop < 0 or min_matches_for_node_freeze < 0:
        raise ValueError("min_matches_for_edge_drop and min_matches_for_node_freeze must be non-negative.")

    raw_edge_counts: Dict[Tuple[int, int], int] = defaultdict(int)
    for m in obs:
        raw_edge_counts[(m["i"], m["j"])] += 1

    # 1) Edge-level dropping rule.
    if min_matches_for_edge_drop > 0:
        drop_edges = {edge for edge, cnt in raw_edge_counts.items() if cnt < min_matches_for_edge_drop}
        obs = [m for m in obs if (m["i"], m["j"]) not in drop_edges]
        if len(obs) == 0:
            msg = "No matches left after dropping edges with count below min_matches_for_edge_drop."
            print(f"{_ANSI_YELLOW}{_FRAME_LOG_PREFIX} {msg}{_ANSI_RESET}", flush=True)
            return np.ones((n,), dtype=np.float32)
        if drop_edges:
            msg = "Dropped %d edges with count < min_matches_for_edge_drop=%d." % (
                len(drop_edges),
                min_matches_for_edge_drop,
            )
            print(f"{_ANSI_YELLOW}{_FRAME_LOG_PREFIX} {msg}{_ANSI_RESET}", flush=True)

    # 2) Node-level freezing rule using total matches to all connected nodes.
    if min_matches_for_node_freeze > 0:
        node_match_totals: Dict[int, int] = defaultdict(int)
        related_nodes: Set[int] = set()
        for (i, j), cnt in raw_edge_counts.items():
            if i == j:
                continue
            node_match_totals[i] += cnt
            node_match_totals[j] += cnt
            related_nodes.add(i)
            related_nodes.add(j)

        frozen_indices_global = {
            idx for idx in related_nodes if node_match_totals.get(idx, 0) < min_matches_for_node_freeze
        }

    base_noise = gtsam.noiseModel.Isotropic.Sigma(3, 1.0)
    robust = gtsam.noiseModel.Robust.Create(
        gtsam.noiseModel.mEstimator.Huber(robust_delta),
        base_noise,
    )

    active_indices = sorted({m["i"] for m in obs} | {m["j"] for m in obs})
    if len(active_indices) == 0:
        raise ValueError("No valid active frame indices after filtering matches.")

    component_edges = [(m["i"], m["j"]) for m in obs]
    components = _build_connected_components(active_indices, component_edges)
    largest_component_size = max((len(comp) for comp in components), default=0)
    if largest_component_size < 3:
        msg = (
            "Largest connected component has %d node(s) (< 3); "
            "skip optimization and return all-one scales."
        ) % largest_component_size
        print(f"{_ANSI_YELLOW}{_FRAME_LOG_PREFIX} {msg}{_ANSI_RESET}", flush=True)
        return np.ones((n,), dtype=np.float32)

    obs_by_component: List[List[Dict[str, Any]]] = []
    for comp in components:
        comp_set = set(comp)
        obs_by_component.append([m for m in obs if m["i"] in comp_set and m["j"] in comp_set])

    if len(components) > 1:
        msg = (
            "Optimizing %d disconnected GTSAM components independently (active_vars=%d, match_factors=%d)."
            % (len(components), len(active_indices), len(obs))
        )
        print(f"{_ANSI_YELLOW}{_FRAME_LOG_PREFIX} {msg}{_ANSI_RESET}", flush=True)
        # _LOGGER.warning(msg)
    if len(frozen_indices_global) > 0:
        msg = (
            "Freezing %d nodes with total node matches < min_matches_for_node_freeze=%d."
            % (len(frozen_indices_global), min_matches_for_node_freeze)
        )
        print(f"{_ANSI_YELLOW}{_FRAME_LOG_PREFIX} {msg}{_ANSI_RESET}", flush=True)
        # _LOGGER.warning(msg)

    scales = np.ones((n,), dtype=np.float64)
    anchor_val = 0.0 if use_exp_param else 1.0

    start_optimize_time = time.perf_counter()
    for comp_idx, (comp_indices, comp_obs) in enumerate(zip(components, obs_by_component)):
        graph = gtsam.NonlinearFactorGraph()
        comp_frozen = set(comp_indices) & frozen_indices_global
        anchor_idx = _select_most_connected_anchor(comp_indices, comp_obs, comp_frozen)
        msg = (
            "Component %d/%d: anchor_idx=%d, nodes=%d, match_factors=%d."
            % (comp_idx + 1, len(components), anchor_idx, len(comp_indices), len(comp_obs))
        )
        print(f"{_ANSI_YELLOW}{_FRAME_LOG_PREFIX} {msg}{_ANSI_RESET}", flush=True)
        anchor_key = _make_variable_key(use_exp_param, anchor_idx)
        if anchor_prior_sigma is not None and anchor_prior_sigma > 0.0:
            graph.add(
                gtsam.PriorFactorDouble(
                    anchor_key,
                    anchor_val,
                    gtsam.noiseModel.Isotropic.Sigma(1, anchor_prior_sigma),
                )
            )

        if len(comp_frozen) > 0:
            freeze_noise = gtsam.noiseModel.Isotropic.Sigma(1, 1e-6)
            for idx in comp_frozen:
                key = _make_variable_key(use_exp_param, idx)
                graph.add(gtsam.PriorFactorDouble(key, anchor_val, freeze_noise))

        if reg_weight > 0.0:
            sigma_reg = 1.0 / math.sqrt(reg_weight)
            reg_noise = gtsam.noiseModel.Isotropic.Sigma(1, sigma_reg)
            for idx in comp_indices:
                key = _make_variable_key(use_exp_param, idx)
                graph.add(gtsam.PriorFactorDouble(key, anchor_val, reg_noise))

        for m in comp_obs:
            i, j = m["i"], m["j"]
            key_i = _make_variable_key(use_exp_param, i)
            key_j = _make_variable_key(use_exp_param, j)

            c_i = m["c_i"]
            b_i = m["b_i"]
            c_j = m["c_j"]
            b_j = m["b_j"]

            def error_func(
                this,
                values,
                jacobians,
                key_i=key_i,
                key_j=key_j,
                c_i=c_i,
                b_i=b_i,
                c_j=c_j,
                b_j=b_j,
            ):
                v_i = values.atDouble(key_i)
                v_j = values.atDouble(key_j)

                s_i = _value_to_scale(v_i, use_exp_param)
                s_j = _value_to_scale(v_j, use_exp_param)

                x_i = c_i * s_i + b_i
                x_j = c_j * s_j + b_j
                err = (x_i - x_j).astype(np.float64)

                if jacobians is not None:
                    dxi_dvi = c_i * (s_i if use_exp_param else 1.0)
                    dxj_dvj = c_j * (s_j if use_exp_param else 1.0)
                    jacobians[0] = dxi_dvi.reshape(3, 1).astype(np.float64)
                    jacobians[1] = (-dxj_dvj).reshape(3, 1).astype(np.float64)

                return err

            factor = gtsam.CustomFactor(robust, _make_key_vector(key_i, key_j), error_func)
            graph.add(factor)

        initial = gtsam.Values()
        init_value = 0.0 if use_exp_param else 1.0
        for idx in comp_indices:
            initial.insert(_make_variable_key(use_exp_param, idx), init_value)

        def _extract_component_scales(values: GtsamValues) -> np.ndarray:
            out = np.ones((n,), dtype=np.float64)
            for idx in comp_indices:
                key = _make_variable_key(use_exp_param, idx)
                out[idx] = _value_to_scale(float(values.atDouble(key)), use_exp_param)
            return out

        def _optimize_component(cur_graph: GtsamNonlinearFactorGraph, run_tag: str = "") -> GtsamValues:
            total_iters = max(1, int(iters))
            if not (print_scales_each_iter or force_full_iters):
                params = gtsam.LevenbergMarquardtParams()
                params.setMaxIterations(total_iters)
                params.setVerbosity("SILENT")
                optimizer = gtsam.LevenbergMarquardtOptimizer(cur_graph, initial, params)
                return optimizer.optimize()

            step_params = gtsam.LevenbergMarquardtParams()
            step_params.setMaxIterations(1)
            step_params.setVerbosity("SILENT")
            values = initial
            for step_idx in range(total_iters):
                optimizer = gtsam.LevenbergMarquardtOptimizer(cur_graph, values, step_params)
                values = optimizer.optimize()
                if print_scales_each_iter:
                    cur_scales = _extract_component_scales(values)
                    active_scales = np.array([cur_scales[idx] for idx in comp_indices], dtype=np.float64)
                    active_scales_str = np.array2string(active_scales, precision=6, suppress_small=True)
                    prefix = f"{run_tag} " if run_tag else ""
                    print(
                        f"{_FRAME_LOG_PREFIX} comp={comp_idx + 1}/{len(components)} {prefix}iter={step_idx + 1}/{total_iters}, "
                        f"active_vars={len(comp_indices)}, scales={active_scales_str}"
                    )
            return values

        try:
            result = _optimize_component(graph, run_tag="")
        except RuntimeError as exc:
            _LOGGER.warning(
                "GTSAM fallback triggered on component %d/%d: active_vars=%d, match_factors=%d, total_factors_before_fallback=%d, error=%s",
                comp_idx + 1,
                len(components),
                len(comp_indices),
                len(comp_obs),
                graph.size(),
                str(exc),
            )
            weak_noise = gtsam.noiseModel.Isotropic.Sigma(1, 1e3)
            for idx in comp_indices:
                key = _make_variable_key(use_exp_param, idx)
                graph.add(gtsam.PriorFactorDouble(key, anchor_val, weak_noise))
            result = _optimize_component(graph, run_tag="fallback")

        for idx in comp_indices:
            key = _make_variable_key(use_exp_param, idx)
            scales[idx] = _value_to_scale(float(result.atDouble(key)), use_exp_param)
    end_optimize_time = time.perf_counter()

    if normalize_mean:
        mean_s = float(scales.mean())
        if mean_s > 1e-12:
            scales = scales / mean_s

    end_total_time = time.perf_counter()
    observation_time = end_observation_time - start_observation_time
    optimize_time = end_optimize_time - start_optimize_time
    total_time = end_total_time - start_total_time
    other_time = max(0.0, total_time - (observation_time + optimize_time))
    print(
        f"{_ANSI_YELLOW}{_FRAME_LOG_PREFIX} Timing: "
        f"build_observations={observation_time:.6f}s, "
        f"optimize={optimize_time:.6f}s, "
        f"other={other_time:.6f}s, "
        f"total={total_time:.6f}s{_ANSI_RESET}",
        flush=True,
    )

    return np.asarray(scales, dtype=np.float32)


def optimize_submap_scales_gtsam(
    submaps: Dict[str, Any],
    out_ph_poses_dict: Dict[str, Any],
    overlap_frames: int,
    iters: int = 30,
    robust_delta: float = 0.10,
    use_exp_param: bool = True,
    reg_weight: float = 0.01,
    normalize_mean: bool = False,
    max_points_per_overlap_frame: int = 1500,
    min_total_matches_per_pair: int = 30,
    random_seed: int = 0,
    latest_n_submaps: Optional[int] = None,
    covisible_frame_pairs: Optional[Dict[Tuple[str, str], List[Tuple[str, str]]]] = None,
    frame_pair_match_dict: Optional[Dict[Tuple[str, str], Dict[str, Any]]] = None,
) -> Dict[str, float]:
    start_total_time = time.perf_counter()
    if gtsam is None:
        raise ImportError("gtsam is required for optimize_submap_scales_gtsam. Please install python-gtsam.")
    if overlap_frames <= 0:
        raise ValueError("overlap_frames must be positive.")
    if max_points_per_overlap_frame <= 0:
        raise ValueError("max_points_per_overlap_frame must be positive.")
    if min_total_matches_per_pair < 0:
        raise ValueError("min_total_matches_per_pair must be non-negative.")
    if latest_n_submaps is not None and latest_n_submaps < 2:
        raise ValueError("latest_n_submaps must be >= 2 when provided.")

    submap_keys = sorted(submaps.keys())
    n_submaps = len(submap_keys)
    if n_submaps < 2:
        return {k: float(getattr(submaps[k], "scale", 1.0)) for k in submap_keys}

    first_submap_idx = 0 if latest_n_submaps is None else max(0, n_submaps - int(latest_n_submaps))
    selected_submap_keys = submap_keys[first_submap_idx:]
    selected_submap_count = len(selected_submap_keys)
    selected_key_to_idx = {k: i for i, k in enumerate(selected_submap_keys)}
    rng = np.random.default_rng(seed=random_seed)
    transforms = MappingTransforms()

    rotations = np.zeros((selected_submap_count, 3, 3), dtype=np.float64)
    translations = np.zeros((selected_submap_count, 3), dtype=np.float64)
    centers_local = np.zeros((selected_submap_count, 3), dtype=np.float64)
    current_scales = np.ones((selected_submap_count,), dtype=np.float64)

    for key, idx in selected_key_to_idx.items():
        submap = submaps[key]
        anchor_key = getattr(submap, "anchor_key", key)
        if anchor_key not in out_ph_poses_dict:
            raise KeyError(f"Missing anchor pose for submap '{key}' via anchor_key='{anchor_key}'.")
        pose_matrix = transforms.pose_to_matrix(out_ph_poses_dict[anchor_key])
        rotations[idx] = pose_matrix[:3, :3]
        translations[idx] = pose_matrix[:3, 3]
        current_scales[idx] = float(getattr(submap, "scale", 1.0))

    def _extract_local_points_from_matched_pixels(
        frame_ids_a: np.ndarray,
        frame_ids_b: np.ndarray,
        matched_points_a: np.ndarray,
        matched_points_b: np.ndarray,
        local_points_a: np.ndarray,
        local_points_b: np.ndarray,
    ) -> Tuple[np.ndarray, np.ndarray]:
        if matched_points_a.shape != matched_points_b.shape:
            raise ValueError(
                f"Matched point arrays must have the same shape, got {matched_points_a.shape} and {matched_points_b.shape}."
            )
        if matched_points_a.ndim != 2 or matched_points_a.shape[1] != 2:
            raise ValueError(f"Matched point arrays must have shape (N,2), got {matched_points_a.shape}.")
        if matched_points_a.shape[0] == 0:
            return (
                np.empty((0, 3), dtype=np.float64),
                np.empty((0, 3), dtype=np.float64),
            )

        rows_a = np.rint(matched_points_a[:, 1]).astype(np.int64)
        cols_a = np.rint(matched_points_a[:, 0]).astype(np.int64)
        rows_b = np.rint(matched_points_b[:, 1]).astype(np.int64)
        cols_b = np.rint(matched_points_b[:, 0]).astype(np.int64)

        in_bounds = (
            (rows_a >= 0)
            & (rows_a < frame_ids_a.shape[0])
            & (cols_a >= 0)
            & (cols_a < frame_ids_a.shape[1])
            & (rows_b >= 0)
            & (rows_b < frame_ids_b.shape[0])
            & (cols_b >= 0)
            & (cols_b < frame_ids_b.shape[1])
        )
        if not np.any(in_bounds):
            return (
                np.empty((0, 3), dtype=np.float64),
                np.empty((0, 3), dtype=np.float64),
            )

        ids_a = frame_ids_a[rows_a[in_bounds], cols_a[in_bounds]].astype(np.int64, copy=False)
        ids_b = frame_ids_b[rows_b[in_bounds], cols_b[in_bounds]].astype(np.int64, copy=False)
        valid_ids = (
            (ids_a >= 0)
            & (ids_a < local_points_a.shape[0])
            & (ids_b >= 0)
            & (ids_b < local_points_b.shape[0])
        )
        if not np.any(valid_ids):
            return (
                np.empty((0, 3), dtype=np.float64),
                np.empty((0, 3), dtype=np.float64),
            )

        ids_a = ids_a[valid_ids]
        ids_b = ids_b[valid_ids]
        conf_a = local_points_a[ids_a, 3]
        conf_b = local_points_b[ids_b, 3]
        valid_conf = np.isfinite(conf_a) & (conf_a > 0.0) & np.isfinite(conf_b) & (conf_b > 0.0)
        if not np.any(valid_conf):
            return (
                np.empty((0, 3), dtype=np.float64),
                np.empty((0, 3), dtype=np.float64),
            )

        ids_a = ids_a[valid_conf]
        ids_b = ids_b[valid_conf]
        # id_pairs = np.stack([ids_a, ids_b], axis=1)
        # _, unique_indices = np.unique(id_pairs, axis=0, return_index=True)
        # unique_indices = np.sort(unique_indices)
        # ids_a = ids_a[unique_indices]
        # ids_b = ids_b[unique_indices]

        p_a = np.asarray(local_points_a[ids_a, :3], dtype=np.float64)
        p_b = np.asarray(local_points_b[ids_b, :3], dtype=np.float64)
        finite = np.all(np.isfinite(p_a), axis=1) & np.all(np.isfinite(p_b), axis=1)
        if not np.any(finite):
            return (
                np.empty((0, 3), dtype=np.float64),
                np.empty((0, 3), dtype=np.float64),
            )

        return p_a[finite], p_b[finite]

    setup_done_time = time.perf_counter()
    start_observation_time = time.perf_counter()
    pair_observations: List[Dict[str, Any]] = []
    for a_idx in range(selected_submap_count - 1):
        b_idx = a_idx + 1
        key_a = selected_submap_keys[a_idx]
        key_b = selected_submap_keys[b_idx]
        submap_a = submaps[key_a]
        submap_b = submaps[key_b]

        frame_keys_a = list(getattr(submap_a, "frame_keys"))
        frame_keys_b = list(getattr(submap_b, "frame_keys"))
        m_use = min(overlap_frames, len(frame_keys_a), len(frame_keys_b))
        if m_use <= 0:
            continue

        local_points_a = np.asarray(getattr(submap_a, "local_points"))
        local_points_b = np.asarray(getattr(submap_b, "local_points"))
        pair_p_i: List[np.ndarray] = []
        pair_p_j: List[np.ndarray] = []
        total_pair_matches = 0

        for t in range(m_use):
            frame_key_a = frame_keys_a[-m_use + t]
            frame_key_b = frame_keys_b[t]
            if frame_key_a != frame_key_b:
                continue
            frame_a = np.asarray(getattr(submap_a, "frame_point_ids")[frame_key_a], dtype=np.int64)
            frame_b = np.asarray(getattr(submap_b, "frame_point_ids")[frame_key_b], dtype=np.int64)
            if frame_a.shape != frame_b.shape:
                raise ValueError(
                    f"Overlap frame shape mismatch between submaps {key_a} and {key_b}: "
                    f"{frame_a.shape} vs {frame_b.shape}."
                )

            valid = (frame_a >= 0) & (frame_b >= 0)
            if not np.any(valid):
                continue

            ids_a_flat = frame_a[valid].astype(np.int64, copy=False)
            ids_b_flat = frame_b[valid].astype(np.int64, copy=False)
            in_range = (
                (ids_a_flat >= 0)
                & (ids_a_flat < local_points_a.shape[0])
                & (ids_b_flat >= 0)
                & (ids_b_flat < local_points_b.shape[0])
            )
            if not np.any(in_range):
                continue

            ids_a_flat = ids_a_flat[in_range]
            ids_b_flat = ids_b_flat[in_range]
            conf_a = local_points_a[ids_a_flat, 3]
            conf_b = local_points_b[ids_b_flat, 3]
            valid_conf = np.isfinite(conf_a) & (conf_a > 0.0) & np.isfinite(conf_b) & (conf_b > 0.0)
            if not np.any(valid_conf):
                continue

            ids_a_flat = ids_a_flat[valid_conf]
            ids_b_flat = ids_b_flat[valid_conf]
            p_a = np.asarray(local_points_a[ids_a_flat, :3], dtype=np.float64)
            p_b = np.asarray(local_points_b[ids_b_flat, :3], dtype=np.float64)
            finite = np.all(np.isfinite(p_a), axis=1) & np.all(np.isfinite(p_b), axis=1)
            if not np.any(finite):
                continue
            p_a = p_a[finite]
            p_b = p_b[finite]
            if p_a.shape[0] == 0:
                continue

            total_pair_matches += int(p_a.shape[0])
            if p_a.shape[0] > max_points_per_overlap_frame:
                sel = rng.choice(p_a.shape[0], size=max_points_per_overlap_frame, replace=False)
                p_a = p_a[sel]
                p_b = p_b[sel]
            pair_p_i.append(p_a)
            pair_p_j.append(p_b)

        if total_pair_matches <= 0 or len(pair_p_i) == 0:
            continue

        pair_observations.append(
            {
                "i": a_idx,
                "j": b_idx,
                "p_i": np.concatenate(pair_p_i, axis=0),
                "p_j": np.concatenate(pair_p_j, axis=0),
                "total_matches": total_pair_matches,
                "key_i": key_a,
                "key_j": key_b,
                "source": "adjacent",
            }
        )

    if covisible_frame_pairs is not None:
        if frame_pair_match_dict is None:
            raise ValueError("frame_pair_match_dict must be provided when covisible_frame_pairs is provided.")

        for (key_a, key_b), frame_pairs in sorted(covisible_frame_pairs.items()):
            if key_a not in selected_key_to_idx or key_b not in selected_key_to_idx:
                continue

            a_idx = selected_key_to_idx[key_a]
            b_idx = selected_key_to_idx[key_b]
            # if abs(a_idx - b_idx) <= 1:
            #     continue

            submap_a = submaps[key_a]
            submap_b = submaps[key_b]
            local_points_a = np.asarray(getattr(submap_a, "local_points"))
            local_points_b = np.asarray(getattr(submap_b, "local_points"))
            pair_p_i: List[np.ndarray] = []
            pair_p_j: List[np.ndarray] = []
            total_pair_matches = 0

            for frame_key_a, frame_key_b in frame_pairs:
                if frame_key_a not in getattr(submap_a, "frame_point_ids"):
                    raise AssertionError(
                        f"Covisibility frame '{frame_key_a}' is not stored in submap '{key_a}'."
                    )
                if frame_key_b not in getattr(submap_b, "frame_point_ids"):
                    raise AssertionError(
                        f"Covisibility frame '{frame_key_b}' is not stored in submap '{key_b}'."
                    )

                match_result = frame_pair_match_dict.get((frame_key_a, frame_key_b))
                if match_result is None:
                    raise KeyError(
                        f"Missing cached match result for covisible frame pair ({frame_key_a}, {frame_key_b})."
                    )

                matched_points_a = np.asarray(match_result["matched_points0"], dtype=np.float32)
                matched_points_b = np.asarray(match_result["matched_points1"], dtype=np.float32)
                p_a, p_b = _extract_local_points_from_matched_pixels(
                    frame_ids_a=np.asarray(getattr(submap_a, "frame_point_ids")[frame_key_a], dtype=np.int64),
                    frame_ids_b=np.asarray(getattr(submap_b, "frame_point_ids")[frame_key_b], dtype=np.int64),
                    matched_points_a=matched_points_a,
                    matched_points_b=matched_points_b,
                    local_points_a=local_points_a,
                    local_points_b=local_points_b,
                )
                if p_a.shape[0] == 0:
                    continue

                total_pair_matches += int(p_a.shape[0])
                if p_a.shape[0] > max_points_per_overlap_frame:
                    sel = rng.choice(p_a.shape[0], size=max_points_per_overlap_frame, replace=False)
                    p_a = p_a[sel]
                    p_b = p_b[sel]
                pair_p_i.append(p_a)
                pair_p_j.append(p_b)

            if total_pair_matches <= 0 or len(pair_p_i) == 0:
                continue

            pair_observations.append(
                {
                    "i": a_idx,
                    "j": b_idx,
                    "p_i": np.concatenate(pair_p_i, axis=0),
                    "p_j": np.concatenate(pair_p_j, axis=0),
                    "total_matches": total_pair_matches,
                    "key_i": key_a,
                    "key_j": key_b,
                    "source": "covisibility",
                }
            )
    end_observation_time = time.perf_counter()

    if len(pair_observations) == 0:
        raise ValueError("No valid adjacent-overlap or covisibility correspondences found between selected submaps.")

    valid_pair_observations: List[Dict[str, Any]] = []
    dropped_pairs: List[Tuple[str, str, int, str]] = []
    for pair_obs in pair_observations:
        total_matches = int(pair_obs["total_matches"])
        if total_matches < min_total_matches_per_pair:
            dropped_pairs.append((pair_obs["key_i"], pair_obs["key_j"], total_matches, str(pair_obs.get("source", "unknown"))))
            continue
        valid_pair_observations.append(pair_obs)

    if dropped_pairs:
        dropped_pairs_str = ", ".join(f"{src}:{ka}-{kb}:{cnt}" for ka, kb, cnt, src in dropped_pairs)
        print(
            f"{_ANSI_YELLOW}{_SUBMAP_LOG_PREFIX} Dropping {len(dropped_pairs)} weak submap link(s) with "
            f"total_matches < min_total_matches_per_pair={min_total_matches_per_pair}: "
            f"{dropped_pairs_str}{_ANSI_RESET}",
            flush=True,
        )

    base_noise = gtsam.noiseModel.Isotropic.Sigma(3, 1.0)
    robust = gtsam.noiseModel.Robust.Create(
        gtsam.noiseModel.mEstimator.Huber(robust_delta),
        base_noise,
    )

    deltas = np.ones((selected_submap_count,), dtype=np.float64)
    anchor_val = 0.0 if use_exp_param else 1.0

    start_graph_build_time = time.perf_counter()
    component_edges = [(int(m["i"]), int(m["j"])) for m in valid_pair_observations]
    all_selected_indices = list(range(selected_submap_count))
    components = _build_connected_components(all_selected_indices, component_edges)
    component_pairs: List[Tuple[List[int], List[Dict[str, Any]]]] = []
    for comp_indices in components:
        comp_set = set(comp_indices)
        comp_obs = [
            m for m in valid_pair_observations if int(m["i"]) in comp_set and int(m["j"]) in comp_set
        ]
        component_pairs.append((comp_indices, comp_obs))
    end_graph_build_time = time.perf_counter()

    start_init_time = time.perf_counter()
    end_init_time = time.perf_counter()

    start_optimize_time = time.perf_counter()
    component_summaries: List[str] = []
    for comp_indices, comp_obs in component_pairs:
        if len(comp_indices) <= 1 or len(comp_obs) == 0:
            component_key = selected_submap_keys[comp_indices[0]]
            component_summaries.append(f"{component_key}:singleton->scale={current_scales[comp_indices[0]]:.6f}")
            continue

        graph = gtsam.NonlinearFactorGraph()
        anchor_idx = comp_indices[len(comp_indices) // 2]
        anchor_sigma = 0.03
        anchor_key = _make_variable_key(use_exp_param, anchor_idx)
        graph.add(gtsam.PriorFactorDouble(anchor_key, anchor_val, gtsam.noiseModel.Isotropic.Sigma(1, anchor_sigma)))

        if reg_weight > 0.0:
            sigma_reg = 1.0 / math.sqrt(reg_weight)
            reg_noise = gtsam.noiseModel.Isotropic.Sigma(1, sigma_reg)
            for idx in comp_indices:
                key = _make_variable_key(use_exp_param, idx)
                graph.add(gtsam.PriorFactorDouble(key, anchor_val, reg_noise))

        for m in comp_obs:
            i = int(m["i"])
            j = int(m["j"])
            key_i = _make_variable_key(use_exp_param, i)
            key_j = _make_variable_key(use_exp_param, j)
            p_i = np.asarray(m["p_i"], dtype=np.float64)
            p_j = np.asarray(m["p_j"], dtype=np.float64)
            c_i_local = centers_local[i]
            c_j_local = centers_local[j]
            base_i = rotations[i] @ c_i_local + translations[i]
            base_j = rotations[j] @ c_j_local + translations[j]

            for k in range(p_i.shape[0]):
                di_world = rotations[i] @ (p_i[k] - c_i_local)
                dj_world = rotations[j] @ (p_j[k] - c_j_local)
                current_scale_i = current_scales[i]
                current_scale_j = current_scales[j]

                def error_func(
                    this,
                    values,
                    jacobians,
                    key_i=key_i,
                    key_j=key_j,
                    base_i=base_i,
                    base_j=base_j,
                    di_world=di_world,
                    dj_world=dj_world,
                    current_scale_i=current_scale_i,
                    current_scale_j=current_scale_j,
                ):
                    v_i = values.atDouble(key_i)
                    v_j = values.atDouble(key_j)
                    delta_i = _value_to_scale(v_i, use_exp_param)
                    delta_j = _value_to_scale(v_j, use_exp_param)
                    s_i = current_scale_i * delta_i
                    s_j = current_scale_j * delta_j
                    err = (base_i + s_i * di_world - (base_j + s_j * dj_world)).astype(np.float64)

                    if jacobians is not None:
                        ddelta_i_dvi = delta_i if use_exp_param else 1.0
                        ddelta_j_dvj = delta_j if use_exp_param else 1.0
                        jacobians[0] = (di_world * current_scale_i * ddelta_i_dvi).reshape(3, 1).astype(np.float64)
                        jacobians[1] = (-dj_world * current_scale_j * ddelta_j_dvj).reshape(3, 1).astype(np.float64)
                    return err

                graph.add(gtsam.CustomFactor(robust, _make_key_vector(key_i, key_j), error_func))

        initial = gtsam.Values()
        init_value = 0.0 if use_exp_param else 1.0
        for idx in comp_indices:
            initial.insert(_make_variable_key(use_exp_param, idx), init_value)

        params = gtsam.LevenbergMarquardtParams()
        params.setMaxIterations(max(1, int(iters)))
        params.setVerbosity("SILENT")
        optimizer = gtsam.LevenbergMarquardtOptimizer(graph, initial, params)
        result = optimizer.optimize()

        comp_abs_scales = []
        for idx in comp_indices:
            key = _make_variable_key(use_exp_param, idx)
            delta_value = _value_to_scale(float(result.atDouble(key)), use_exp_param)
            abs_scale = current_scales[idx] * delta_value
            deltas[idx] = delta_value
            comp_abs_scales.append(abs_scale)

        comp_mean = float(np.mean(comp_abs_scales)) if normalize_mean and len(comp_abs_scales) > 0 else 1.0
        if normalize_mean and comp_mean > 1e-12:
            for idx in comp_indices:
                deltas[idx] = (current_scales[idx] * deltas[idx]) / comp_mean
                current_scales[idx] = 1.0

        component_key_range = f"{selected_submap_keys[comp_indices[0]]}..{selected_submap_keys[comp_indices[-1]]}"
        anchor_submap_key = selected_submap_keys[anchor_idx]
        component_summaries.append(
            f"{component_key_range}:nodes={len(comp_indices)}, pair_links={len(comp_obs)}, "
            f"anchor={anchor_submap_key}, anchor_sigma={anchor_sigma:.6g}, factors={graph.size()}"
        )
    end_optimize_time = time.perf_counter()

    start_post_optimize_time = time.perf_counter()
    scales_dict = {}
    for key, idx in selected_key_to_idx.items():
        scales_dict[key] = float(current_scales[idx] * deltas[idx])
    print(
        f"{_ANSI_YELLOW}{_SUBMAP_LOG_PREFIX} Submap scale graph components={len(component_pairs)}, "
        f"valid_pair_links={len(valid_pair_observations)}, summaries={component_summaries}{_ANSI_RESET}",
        flush=True,
    )
    end_post_optimize_time = time.perf_counter()

    end_total_time = time.perf_counter()
    setup_time = setup_done_time - start_total_time
    observation_time = end_observation_time - start_observation_time
    graph_build_time = end_graph_build_time - start_graph_build_time
    init_time = end_init_time - start_init_time
    optimize_time = end_optimize_time - start_optimize_time
    post_optimize_time = end_post_optimize_time - start_post_optimize_time
    total_time = end_total_time - start_total_time
    accounted_time = (
        setup_time
        + observation_time
        + graph_build_time
        + init_time
        + optimize_time
        + post_optimize_time
    )
    other_time = max(0.0, total_time - accounted_time)
    print(
        f"{_ANSI_YELLOW}{_SUBMAP_LOG_PREFIX} Timing: "
        f"setup={setup_time:.6f}s, "
        f"build_observations={observation_time:.6f}s, "
        f"build_graph={graph_build_time:.6f}s, "
        f"init_optimizer={init_time:.6f}s, "
        f"optimize={optimize_time:.6f}s, "
        f"post_optimize={post_optimize_time:.6f}s, "
        f"other={other_time:.6f}s, "
        f"total={total_time:.6f}s{_ANSI_RESET}",
        flush=True,
    )
    return scales_dict
