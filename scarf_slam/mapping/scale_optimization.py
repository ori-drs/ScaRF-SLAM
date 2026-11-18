import math
import logging
import os
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



def _optimize_frame_scales_numpy_arrays(predictions, matches, keypoints, iters, robust_delta, use_exp_param, reg_weight,
                                        anchor_prior_sigma, normalize_mean, min_matches_for_node_freeze,
                                        min_matches_for_edge_drop, start_total_time) -> np.ndarray:
    """optimize_frame_scales_gtsam(solver="numpy") on arrays instead of one Python dict per match: the same filters,
    edge dropping, node freezing, components, anchor choice, log lines and solver inputs in the same order, and the
    unprojection as a batched matmul, which is bit-identical to the per-match 3x3 @ 3 products (checked on recorded
    CityPark calls). About 10x faster; the per-match Python work was most of the frame-scale cost on the Jetson."""
    geom = _prepare_geometry(predictions)
    depth, k_inv, r_c2w, t_c2w = geom["depth"], geom["k_inv"], geom["r_c2w"], geom["t_c2w"]
    n = int(geom["n"][0])
    ii, jj, ui, vi, uj, vj = [], [], [], [], [], []
    for (i, j), pairs in matches.items():
        if len(pairs) == 0:
            continue
        kps_i, kps_j = keypoints[i], keypoints[j]
        pi = np.array([kps_i[q].pt for q, _ in pairs], dtype=np.float64).reshape(-1, 2)
        pj = np.array([kps_j[p].pt for _, p in pairs], dtype=np.float64).reshape(-1, 2)
        ii.append(np.full(len(pairs), i, np.int64)); jj.append(np.full(len(pairs), j, np.int64))
        ui.append(pi[:, 0]); vi.append(pi[:, 1]); uj.append(pj[:, 0]); vj.append(pj[:, 1])
    if len(ii) == 0:
        msg = "No matches provided for optimization; skip optimization and return all-one scales."
        print(f"{_ANSI_YELLOW}{_FRAME_LOG_PREFIX} {msg}{_ANSI_RESET}", flush=True)
        return np.ones((n,), dtype=np.float32)
    start_observation_time = time.perf_counter()
    I, J = np.concatenate(ii), np.concatenate(jj)
    UI, VI, UJ, VJ = np.concatenate(ui), np.concatenate(vi), np.concatenate(uj), np.concatenate(vj)
    h, w = depth.shape[1], depth.shape[2]
    cui, cvi, cuj, cvj = UI.astype(np.int64), VI.astype(np.int64), UJ.astype(np.int64), VJ.astype(np.int64)   # int(): toward zero
    keep = (cui >= 0) & (cui < w) & (cvi >= 0) & (cvi < h) & (cuj >= 0) & (cuj < w) & (cvj >= 0) & (cvj < h)
    I, J, UI, VI, UJ, VJ = I[keep], J[keep], UI[keep], VI[keep], UJ[keep], VJ[keep]
    d_i = depth[I, cvi[keep], cui[keep]].astype(np.float64)
    d_j = depth[J, cvj[keep], cuj[keep]].astype(np.float64)
    keep = np.isfinite(d_i) & np.isfinite(d_j) & (d_i > 0) & (d_j > 0)
    I, J, UI, VI, UJ, VJ, d_i, d_j = I[keep], J[keep], UI[keep], VI[keep], UJ[keep], VJ[keep], d_i[keep], d_j[keep]
    if I.size == 0:
        raise ValueError("All matches invalid after depth filtering.")

    def unproject(F, U, V, d):
        pix = np.stack([U, V, np.ones_like(U)], axis=1)[:, :, None]
        ray = (k_inv[F] @ pix)[:, :, 0] * d[:, None]
        return (r_c2w[F] @ ray[:, :, None])[:, :, 0], t_c2w[F]

    c_i, b_i = unproject(I, UI, VI, d_i)
    c_j, b_j = unproject(J, UJ, VJ, d_j)
    end_observation_time = time.perf_counter()

    if min_matches_for_edge_drop < 0 or min_matches_for_node_freeze < 0:
        raise ValueError("min_matches_for_edge_drop and min_matches_for_node_freeze must be non-negative.")
    edge_code = I * max(n, 1) + J
    codes, counts = np.unique(edge_code, return_counts=True)
    raw_edge_counts = {(int(c) // max(n, 1), int(c) % max(n, 1)): int(k) for c, k in zip(codes, counts)}
    if min_matches_for_edge_drop > 0:
        drop_codes = codes[counts < min_matches_for_edge_drop]
        if drop_codes.size:
            keep = ~np.isin(edge_code, drop_codes)
            I, J, c_i, b_i, c_j, b_j = I[keep], J[keep], c_i[keep], b_i[keep], c_j[keep], b_j[keep]
            if I.size == 0:
                msg = "No matches left after dropping edges with count below min_matches_for_edge_drop."
                print(f"{_ANSI_YELLOW}{_FRAME_LOG_PREFIX} {msg}{_ANSI_RESET}", flush=True)
                return np.ones((n,), dtype=np.float32)
            msg = "Dropped %d edges with count < min_matches_for_edge_drop=%d." % (int(drop_codes.size), min_matches_for_edge_drop)
            print(f"{_ANSI_YELLOW}{_FRAME_LOG_PREFIX} {msg}{_ANSI_RESET}", flush=True)
    frozen_indices_global: Set[int] = set()
    if min_matches_for_node_freeze > 0:
        node_match_totals: Dict[int, int] = defaultdict(int)
        related_nodes: Set[int] = set()
        for (i, j), cnt in raw_edge_counts.items():
            if i == j:
                continue
            node_match_totals[i] += cnt; node_match_totals[j] += cnt
            related_nodes.add(i); related_nodes.add(j)
        frozen_indices_global = {idx for idx in related_nodes if node_match_totals.get(idx, 0) < min_matches_for_node_freeze}

    active_indices = sorted(set(I.tolist()) | set(J.tolist()))
    uniq_edges = np.unique(np.stack([I, J], 1), axis=0)
    components = _build_connected_components(active_indices, [(int(a), int(b)) for a, b in uniq_edges])
    largest_component_size = max((len(comp) for comp in components), default=0)
    if largest_component_size < 3:
        msg = ("Largest connected component has %d node(s) (< 3); skip optimization and return all-one scales." % largest_component_size)
        print(f"{_ANSI_YELLOW}{_FRAME_LOG_PREFIX} {msg}{_ANSI_RESET}", flush=True)
        return np.ones((n,), dtype=np.float32)
    if len(components) > 1:
        msg = ("Optimizing %d disconnected GTSAM components independently (active_vars=%d, match_factors=%d)."
               % (len(components), len(active_indices), int(I.size)))
        print(f"{_ANSI_YELLOW}{_FRAME_LOG_PREFIX} {msg}{_ANSI_RESET}", flush=True)
    if len(frozen_indices_global) > 0:
        msg = ("Freezing %d nodes with total node matches < min_matches_for_node_freeze=%d." % (len(frozen_indices_global), min_matches_for_node_freeze))
        print(f"{_ANSI_YELLOW}{_FRAME_LOG_PREFIX} {msg}{_ANSI_RESET}", flush=True)

    scales = np.ones((n,), dtype=np.float64)
    anchor_val = 0.0 if use_exp_param else 1.0
    start_optimize_time = time.perf_counter()
    for comp_idx, comp_indices in enumerate(components):
        in_comp = np.isin(I, comp_indices) & np.isin(J, comp_indices)
        cI, cJ = I[in_comp], J[in_comp]
        comp_frozen = set(comp_indices) & frozen_indices_global
        # _select_most_connected_anchor on arrays: degree, incident factor count, distance to the midpoint, index
        candidates = [idx for idx in comp_indices if idx not in comp_frozen] or list(comp_indices)
        off = cI != cJ
        inc = defaultdict(int)
        for k_, c_ in zip(*np.unique(np.concatenate([cI[off], cJ[off]]), return_counts=True)):
            inc[int(k_)] = int(c_)
        pairs_u = np.unique(np.stack([np.concatenate([cI[off], cJ[off]]), np.concatenate([cJ[off], cI[off]])], 1), axis=0)
        deg = defaultdict(int)
        for k_, c_ in zip(*np.unique(pairs_u[:, 0], return_counts=True)) if pairs_u.size else []:
            deg[int(k_)] = int(c_)
        mid = 0.5 * (float(comp_indices[0]) + float(comp_indices[-1]))
        anchor_idx = max(candidates, key=lambda idx: (deg[idx], inc.get(idx, 0), -abs(float(idx) - mid), -idx))
        msg = ("Component %d/%d: anchor_idx=%d, nodes=%d, match_factors=%d."
               % (comp_idx + 1, len(components), anchor_idx, len(comp_indices), int(cI.size)))
        print(f"{_ANSI_YELLOW}{_FRAME_LOG_PREFIX} {msg}{_ANSI_RESET}", flush=True)
        lut = np.full(max(n, int(max(comp_indices)) + 1), -1, np.int64); lut[comp_indices] = np.arange(len(comp_indices))
        local = {idx: n_local for n_local, idx in enumerate(comp_indices)}
        v_opt = _optimize_component_scales_numpy(
            num_vars=len(comp_indices), var_i=lut[cI], var_j=lut[cJ],
            base_diff=b_i[in_comp] - b_j[in_comp], dir_i=c_i[in_comp], dir_j=c_j[in_comp],
            prior_weight=_prior_weights(len(comp_indices), [local[anchor_idx]], anchor_prior_sigma, reg_weight,
                                        frozen_vars=[local[idx] for idx in comp_frozen]),
            prior_val=anchor_val, robust_delta=robust_delta, use_exp_param=use_exp_param, iters=iters)
        for idx in comp_indices:
            scales[idx] = _value_to_scale(float(v_opt[local[idx]]), use_exp_param)
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
    print(f"{_ANSI_YELLOW}{_FRAME_LOG_PREFIX} Timing: build_observations={observation_time:.6f}s, optimize={optimize_time:.6f}s, "
          f"other={other_time:.6f}s, total={total_time:.6f}s{_ANSI_RESET}", flush=True)
    return np.asarray(scales, dtype=np.float32)


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
    solver: str = "gtsam",
) -> np.ndarray:
    """
    Optimize per-image depth scales with GTSAM using 3D point distance residuals.

    solver: "gtsam" (one CustomFactor per match) or "numpy" (_optimize_component_scales_numpy: the same objective and LM
    schedule, vectorized). print_scales_each_iter / force_full_iters always use gtsam.
    """
    if solver not in ("gtsam", "numpy"):
        raise ValueError(f"solver must be 'gtsam' or 'numpy', got {solver!r}.")
    use_numpy = solver == "numpy" and not (print_scales_each_iter or force_full_iters)
    if not use_numpy and gtsam is None:
        raise ImportError("gtsam is required for optimize_frame_scales_gtsam. Please install python-gtsam.")

    start_total_time = time.perf_counter()
    if use_numpy and os.environ.get("SCARF_FRAME_SCALE_LEGACY") != "1":   # the per-match Python path stays available for A/B checks
        return _optimize_frame_scales_numpy_arrays(predictions, matches, keypoints, iters, robust_delta, use_exp_param, reg_weight,
                                                   anchor_prior_sigma, normalize_mean, min_matches_for_node_freeze,
                                                   min_matches_for_edge_drop, start_total_time)
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
        comp_frozen = set(comp_indices) & frozen_indices_global
        anchor_idx = _select_most_connected_anchor(comp_indices, comp_obs, comp_frozen)
        msg = (
            "Component %d/%d: anchor_idx=%d, nodes=%d, match_factors=%d."
            % (comp_idx + 1, len(components), anchor_idx, len(comp_indices), len(comp_obs))
        )
        print(f"{_ANSI_YELLOW}{_FRAME_LOG_PREFIX} {msg}{_ANSI_RESET}", flush=True)
        if use_numpy:
            local = {idx: n_local for n_local, idx in enumerate(comp_indices)}
            v_opt = _optimize_component_scales_numpy(
                num_vars=len(comp_indices),
                var_i=np.array([local[m["i"]] for m in comp_obs], dtype=np.int64),
                var_j=np.array([local[m["j"]] for m in comp_obs], dtype=np.int64),
                base_diff=np.array([np.asarray(m["b_i"], np.float64) - np.asarray(m["b_j"], np.float64) for m in comp_obs]),
                dir_i=np.array([np.asarray(m["c_i"], np.float64) for m in comp_obs]),
                dir_j=np.array([np.asarray(m["c_j"], np.float64) for m in comp_obs]),
                prior_weight=_prior_weights(
                    len(comp_indices), [local[anchor_idx]], anchor_prior_sigma, reg_weight,
                    frozen_vars=[local[idx] for idx in comp_frozen],
                ),
                prior_val=anchor_val,
                robust_delta=robust_delta,
                use_exp_param=use_exp_param,
                iters=iters,
            )
            for idx in comp_indices:
                scales[idx] = _value_to_scale(float(v_opt[local[idx]]), use_exp_param)
            continue
        graph = gtsam.NonlinearFactorGraph()
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


def _optimize_component_scales_numpy(
    num_vars: int,
    var_i: np.ndarray,
    var_j: np.ndarray,
    base_diff: np.ndarray,
    dir_i: np.ndarray,
    dir_j: np.ndarray,
    prior_weight: np.ndarray,
    prior_val: float,
    robust_delta: float,
    use_exp_param: bool,
    iters: int,
) -> np.ndarray:
    """Vectorized equivalent of the per-component gtsam graphs of optimize_submap_scales_gtsam and
    optimize_frame_scales_gtsam.

    Point residuals r = base_diff + g(v_i) * dir_i - g(v_j) * dir_j, with g = exp (use_exp_param) or identity, under a
    Huber(robust_delta) loss on |r| (unit isotropic noise), plus Gaussian priors 0.5 * prior_weight * (v - prior_val)^2
    (prior_weight = sum of 1 / sigma^2 over the PriorFactorDouble terms on that variable: anchor, freeze and
    regularization priors). Optimized with gtsam's Levenberg-Marquardt schedule
    (default LevenbergMarquardtParams: lambda 1e-5 x/÷ 10 up to 1e5, min model fidelity 1e-3, relative and absolute
    error tolerance 1e-5, iterative reweighting of the robust loss at each linearization), so the result matches the
    gtsam solve to its convergence tolerance without one Python callback per point per iteration."""
    k = float(robust_delta)
    prior_weight = np.asarray(prior_weight, dtype=np.float64)
    if _numba is not None and os.environ.get("SCARF_SCALE_NUMBA", "1") != "0":
        return _lm_nb(int(num_vars), np.ascontiguousarray(var_i, dtype=np.int64), np.ascontiguousarray(var_j, dtype=np.int64),
                      np.ascontiguousarray(base_diff, dtype=np.float64), np.ascontiguousarray(dir_i, dtype=np.float64),
                      np.ascontiguousarray(dir_j, dtype=np.float64), np.ascontiguousarray(prior_weight, dtype=np.float64),
                      float(prior_val), k, bool(use_exp_param), int(iters))

    def point_terms(v):
        gi = np.exp(v[var_i]) if use_exp_param else v[var_i]
        gj = np.exp(v[var_j]) if use_exp_param else v[var_j]
        ji = gi[:, None] * dir_i                       # d r / d v_i
        jj = -(gj[:, None] * dir_j)                    # d r / d v_j
        if not use_exp_param:
            ji, jj = dir_i, -dir_j
        r = base_diff + gi[:, None] * dir_i - gj[:, None] * dir_j
        return r, ji, jj

    def error(v):
        r, _, _ = point_terms(v)
        d = np.linalg.norm(r, axis=1)
        loss = np.where(d <= k, 0.5 * d * d, k * d - 0.5 * k * k).sum()
        loss += 0.5 * np.sum(prior_weight * (v - prior_val) ** 2)
        return float(loss)

    def linearize(v):
        r, ji, jj = point_terms(v)
        d = np.linalg.norm(r, axis=1)
        sw = np.sqrt(np.where(d <= k, 1.0, k / np.maximum(d, 1e-300)))[:, None]
        ai, aj, b = ji * sw, jj * sw, -r * sw
        h = np.zeros((num_vars, num_vars))
        h[np.diag_indices(num_vars)] += np.bincount(var_i, (ai * ai).sum(1), num_vars) + np.bincount(var_j, (aj * aj).sum(1), num_vars)
        cross = (ai * aj).sum(1)
        np.add.at(h, (var_i, var_j), cross)
        np.add.at(h, (var_j, var_i), cross)
        g = np.bincount(var_i, (ai * b).sum(1), num_vars) + np.bincount(var_j, (aj * b).sum(1), num_vars)
        bb = float((b * b).sum())
        h[np.diag_indices(num_vars)] += prior_weight
        g += prior_weight * (prior_val - v)
        bb += float(np.sum(prior_weight * (v - prior_val) ** 2))
        return h, g, 0.5 * bb

    v = np.full(num_vars, prior_val, dtype=np.float64)
    current_error = error(v)
    lam, lam_factor, lam_max, min_fidelity, rel_tol, abs_tol = 1e-5, 10.0, 1e5, 1e-3, 1e-5, 1e-5
    iterations = 0
    while True:
        h, g, old_lin_error = linearize(v)
        while True:  # gtsam LevenbergMarquardtOptimizer::tryLambda
            step_ok, stop_search, new_v, new_error = False, False, v, current_error
            try:
                delta = np.linalg.solve(h + lam * np.eye(num_vars), g)
                solved = bool(np.all(np.isfinite(delta)))
            except np.linalg.LinAlgError:
                solved = False
            if solved:
                lin_change = float(g @ delta - 0.5 * delta @ h @ delta)   # old - new linearized error
                if lin_change >= 0.0:
                    new_v = v + delta
                    new_error = error(new_v)
                    cost_change = current_error - new_error
                    if lin_change > np.finfo(float).eps * old_lin_error:
                        step_ok = cost_change / lin_change > min_fidelity
                    else:
                        step_ok = True
                    if abs(cost_change) < rel_tol * current_error:
                        stop_search = True
            if step_ok:
                lam /= lam_factor
                v, prev_error, current_error = new_v, current_error, new_error
                iterations += 1
                break
            if not stop_search:
                lam *= lam_factor
                if lam >= lam_max:
                    prev_error = current_error
                    break
                continue
            prev_error = current_error
            break
        if iterations >= max(1, int(iters)):
            break
        decrease = prev_error - current_error
        if current_error <= 0.0 or decrease <= abs_tol or decrease / prev_error <= rel_tol:
            break
    return v



# numba version of _optimize_component_scales_numpy (the same objective and Levenberg-Marquardt schedule, one fused loop per
# linearization instead of a dozen small numpy calls). It sums in a different order than numpy / BLAS, so results agree to
# floating-point round-off (and can differ by one accepted LM step near a threshold), not bit for bit. On by default when
# numba imports; SCARF_SCALE_NUMBA=0 falls back to numpy.
try:
    import numba as _numba
except Exception:  # pragma: no cover
    _numba = None

if _numba is not None:
    @_numba.njit(cache=True)
    def _lm_error_nb(v, var_i, var_j, base_diff, dir_i, dir_j, prior_weight, prior_val, k, use_exp):
        loss = 0.0
        for m in range(var_i.shape[0]):
            gi = np.exp(v[var_i[m]]) if use_exp else v[var_i[m]]
            gj = np.exp(v[var_j[m]]) if use_exp else v[var_j[m]]
            d2 = 0.0
            for a in range(3):
                r = base_diff[m, a] + gi * dir_i[m, a] - gj * dir_j[m, a]
                d2 += r * r
            d = np.sqrt(d2)
            loss += 0.5 * d * d if d <= k else k * d - 0.5 * k * k
        for q in range(v.shape[0]):
            loss += 0.5 * prior_weight[q] * (v[q] - prior_val) ** 2
        return loss

    @_numba.njit(cache=True)
    def _lm_linearize_nb(v, var_i, var_j, base_diff, dir_i, dir_j, prior_weight, prior_val, k, use_exp):
        nv = v.shape[0]
        h = np.zeros((nv, nv)); g = np.zeros(nv); bb = 0.0
        for m in range(var_i.shape[0]):
            i = var_i[m]; j = var_j[m]
            gi = np.exp(v[i]) if use_exp else v[i]
            gj = np.exp(v[j]) if use_exp else v[j]
            r0 = base_diff[m, 0] + gi * dir_i[m, 0] - gj * dir_j[m, 0]
            r1 = base_diff[m, 1] + gi * dir_i[m, 1] - gj * dir_j[m, 1]
            r2 = base_diff[m, 2] + gi * dir_i[m, 2] - gj * dir_j[m, 2]
            d = np.sqrt(r0 * r0 + r1 * r1 + r2 * r2)
            sw = 1.0 if d <= k else np.sqrt(k / max(d, 1e-300))
            si = gi if use_exp else 1.0
            sj = gj if use_exp else 1.0
            aii = 0.0; ajj = 0.0; aij = 0.0; gbi = 0.0; gbj = 0.0
            for a, ra in ((0, r0), (1, r1), (2, r2)):
                ai = si * dir_i[m, a] * sw; aj = -sj * dir_j[m, a] * sw; b = -ra * sw
                aii += ai * ai; ajj += aj * aj; aij += ai * aj; gbi += ai * b; gbj += aj * b; bb += b * b
            h[i, i] += aii; h[j, j] += ajj; h[i, j] += aij; h[j, i] += aij
            g[i] += gbi; g[j] += gbj
        for q in range(nv):
            h[q, q] += prior_weight[q]
            g[q] += prior_weight[q] * (prior_val - v[q])
            bb += prior_weight[q] * (v[q] - prior_val) ** 2
        return h, g, 0.5 * bb

    @_numba.njit(cache=True)
    def _lm_solve_nb(a, b):
        """Gaussian elimination with partial pivoting; ok = False on a (numerically) singular system."""
        n = b.shape[0]; m = a.copy(); x = b.copy()
        for c in range(n):
            p = c
            for r in range(c + 1, n):
                if abs(m[r, c]) > abs(m[p, c]):
                    p = r
            if not (abs(m[p, c]) > 1e-300):
                return x, False
            if p != c:
                for q in range(n):
                    t = m[c, q]; m[c, q] = m[p, q]; m[p, q] = t
                t = x[c]; x[c] = x[p]; x[p] = t
            for r in range(c + 1, n):
                f = m[r, c] / m[c, c]
                for q in range(c, n):
                    m[r, q] -= f * m[c, q]
                x[r] -= f * x[c]
        for c in range(n - 1, -1, -1):
            acc = x[c]
            for q in range(c + 1, n):
                acc -= m[c, q] * x[q]
            x[c] = acc / m[c, c]
        for c in range(n):
            if not np.isfinite(x[c]):
                return x, False
        return x, True

    @_numba.njit(cache=True)
    def _lm_nb(num_vars, var_i, var_j, base_diff, dir_i, dir_j, prior_weight, prior_val, k, use_exp, iters):
        v = np.full(num_vars, prior_val)
        current_error = _lm_error_nb(v, var_i, var_j, base_diff, dir_i, dir_j, prior_weight, prior_val, k, use_exp)
        lam = 1e-5; lam_factor = 10.0; lam_max = 1e5; min_fidelity = 1e-3; rel_tol = 1e-5; abs_tol = 1e-5
        eps = 2.220446049250313e-16
        iterations = 0; prev_error = current_error
        while True:
            h, g, old_lin_error = _lm_linearize_nb(v, var_i, var_j, base_diff, dir_i, dir_j, prior_weight, prior_val, k, use_exp)
            while True:
                step_ok = False; stop_search = False; new_v = v; new_error = current_error
                hl = h.copy()
                for q in range(num_vars):
                    hl[q, q] += lam
                delta, solved = _lm_solve_nb(hl, g)
                if solved:
                    hd = h @ delta
                    lin_change = 0.0
                    for q in range(num_vars):
                        lin_change += g[q] * delta[q] - 0.5 * delta[q] * hd[q]
                    if lin_change >= 0.0:
                        new_v = v + delta
                        new_error = _lm_error_nb(new_v, var_i, var_j, base_diff, dir_i, dir_j, prior_weight, prior_val, k, use_exp)
                        cost_change = current_error - new_error
                        if lin_change > eps * old_lin_error:
                            step_ok = cost_change / lin_change > min_fidelity
                        else:
                            step_ok = True
                        if abs(cost_change) < rel_tol * current_error:
                            stop_search = True
                if step_ok:
                    lam /= lam_factor
                    v = new_v; prev_error = current_error; current_error = new_error
                    iterations += 1
                    break
                if not stop_search:
                    lam *= lam_factor
                    if lam >= lam_max:
                        prev_error = current_error
                        break
                    continue
                prev_error = current_error
                break
            if iterations >= max(1, iters):
                break
            decrease = prev_error - current_error
            if current_error <= 0.0 or decrease <= abs_tol or decrease / prev_error <= rel_tol:
                break
        return v


def _prior_weights(num_vars: int, anchor_vars, anchor_sigma: Optional[float], reg_weight: float,
                   frozen_vars=(), frozen_sigma: float = 1e-6) -> np.ndarray:
    """Per-variable sum of 1 / sigma^2 of the PriorFactorDouble terms the gtsam graphs add: anchor (if anchor_sigma),
    freeze and regularization (reg_weight = 1 / sigma_reg^2 on every variable) priors."""
    w = np.zeros(num_vars, dtype=np.float64)
    if anchor_sigma is not None and anchor_sigma > 0.0:
        for a in anchor_vars:
            w[a] += 1.0 / anchor_sigma**2
    for f in frozen_vars:
        w[f] += 1.0 / frozen_sigma**2
    if reg_weight > 0.0:
        w += reg_weight
    return w


# Per submap-pair point lookups of optimize_submap_scales_gtsam, cached across calls. The lookups (frame point ids ->
# confidence / finiteness masks -> gathered local points) depend only on arrays a submap never modifies after it is
# created, but the sliding window recomputed them for every pair on every call. An entry keeps references to the exact
# arrays it was computed from and is used only if the call sees the same objects (identity), so a replaced submap or a
# new match result simply misses. The random subsampling still runs on every call in the same order, so results are
# unchanged. SCARF_SUBMAP_PAIR_CACHE=0 disables it.
_SUBMAP_PAIR_CACHE: Dict[Tuple, Tuple[Tuple[Any, ...], Any]] = {}


def _pair_cache_get(key, refs, compute):
    if os.environ.get("SCARF_SUBMAP_PAIR_CACHE", "1") == "0":
        return compute()
    hit = _SUBMAP_PAIR_CACHE.get(key)
    if hit is not None and len(hit[0]) == len(refs) and all(a is b for a, b in zip(hit[0], refs)):
        return hit[1]
    val = compute()
    _SUBMAP_PAIR_CACHE[key] = (tuple(refs), val)
    return val


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
    solver: str = "gtsam",
) -> Dict[str, float]:
    """solver: "gtsam" (one CustomFactor per matched point pair) or "numpy" (_optimize_component_scales_numpy: the same
    objective and LM schedule, vectorized; much faster on long sequences)."""
    start_total_time = time.perf_counter()
    if solver not in ("gtsam", "numpy"):
        raise ValueError(f"solver must be 'gtsam' or 'numpy', got {solver!r}.")
    if solver == "gtsam" and gtsam is None:
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
    _selected = set(selected_submap_keys)
    for _k in [k for k in _SUBMAP_PAIR_CACHE if k[1] not in _selected or k[2] not in _selected]:
        del _SUBMAP_PAIR_CACHE[_k]
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
            ids_obj_a = getattr(submap_a, "frame_point_ids")[frame_key_a]
            ids_obj_b = getattr(submap_b, "frame_point_ids")[frame_key_b]

            def _adjacent_points(ids_obj_a=ids_obj_a, ids_obj_b=ids_obj_b, local_points_a=local_points_a, local_points_b=local_points_b,
                                 key_a=key_a, key_b=key_b):
                frame_a = np.asarray(ids_obj_a, dtype=np.int64)
                frame_b = np.asarray(ids_obj_b, dtype=np.int64)
                if frame_a.shape != frame_b.shape:
                    raise ValueError(
                        f"Overlap frame shape mismatch between submaps {key_a} and {key_b}: "
                        f"{frame_a.shape} vs {frame_b.shape}."
                    )
                valid = (frame_a >= 0) & (frame_b >= 0)
                if not np.any(valid):
                    return None
                ids_a_flat = frame_a[valid].astype(np.int64, copy=False)
                ids_b_flat = frame_b[valid].astype(np.int64, copy=False)
                in_range = (
                    (ids_a_flat >= 0)
                    & (ids_a_flat < local_points_a.shape[0])
                    & (ids_b_flat >= 0)
                    & (ids_b_flat < local_points_b.shape[0])
                )
                if not np.any(in_range):
                    return None
                ids_a_flat = ids_a_flat[in_range]
                ids_b_flat = ids_b_flat[in_range]
                conf_a = local_points_a[ids_a_flat, 3]
                conf_b = local_points_b[ids_b_flat, 3]
                valid_conf = np.isfinite(conf_a) & (conf_a > 0.0) & np.isfinite(conf_b) & (conf_b > 0.0)
                if not np.any(valid_conf):
                    return None
                ids_a_flat = ids_a_flat[valid_conf]
                ids_b_flat = ids_b_flat[valid_conf]
                p_a = np.asarray(local_points_a[ids_a_flat, :3], dtype=np.float64)
                p_b = np.asarray(local_points_b[ids_b_flat, :3], dtype=np.float64)
                finite = np.all(np.isfinite(p_a), axis=1) & np.all(np.isfinite(p_b), axis=1)
                if not np.any(finite):
                    return None
                p_a = p_a[finite]
                p_b = p_b[finite]
                if p_a.shape[0] == 0:
                    return None
                return p_a, p_b

            got = _pair_cache_get(("adjacent", key_a, key_b, frame_key_a),
                                  (local_points_a, local_points_b, ids_obj_a, ids_obj_b), _adjacent_points)
            if got is None:
                continue
            p_a, p_b = got

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

                mp_obj_a, mp_obj_b = match_result["matched_points0"], match_result["matched_points1"]
                fid_obj_a = getattr(submap_a, "frame_point_ids")[frame_key_a]
                fid_obj_b = getattr(submap_b, "frame_point_ids")[frame_key_b]
                p_a, p_b = _pair_cache_get(
                    ("covisibility", key_a, key_b, frame_key_a, frame_key_b),
                    (local_points_a, local_points_b, fid_obj_a, fid_obj_b, mp_obj_a, mp_obj_b),
                    lambda fid_obj_a=fid_obj_a, fid_obj_b=fid_obj_b, mp_obj_a=mp_obj_a, mp_obj_b=mp_obj_b,
                           local_points_a=local_points_a, local_points_b=local_points_b: _extract_local_points_from_matched_pixels(
                        frame_ids_a=np.asarray(fid_obj_a, dtype=np.int64),
                        frame_ids_b=np.asarray(fid_obj_b, dtype=np.int64),
                        matched_points_a=np.asarray(mp_obj_a, dtype=np.float32),
                        matched_points_b=np.asarray(mp_obj_b, dtype=np.float32),
                        local_points_a=local_points_a,
                        local_points_b=local_points_b,
                    ),
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

        anchor_idx = comp_indices[len(comp_indices) // 2]
        anchor_sigma = 0.03
        if solver == "numpy":
            local = {idx: n for n, idx in enumerate(comp_indices)}
            var_i, var_j, base_diff, dir_i, dir_j = [], [], [], [], []
            for m in comp_obs:
                i = int(m["i"])
                j = int(m["j"])
                p_i = np.asarray(m["p_i"], dtype=np.float64)
                p_j = np.asarray(m["p_j"], dtype=np.float64)
                base_i = rotations[i] @ centers_local[i] + translations[i]
                base_j = rotations[j] @ centers_local[j] + translations[j]
                var_i.append(np.full(p_i.shape[0], local[i]))
                var_j.append(np.full(p_j.shape[0], local[j]))
                base_diff.append(np.broadcast_to(base_i - base_j, p_i.shape))
                dir_i.append(current_scales[i] * ((p_i - centers_local[i]) @ rotations[i].T))
                dir_j.append(current_scales[j] * ((p_j - centers_local[j]) @ rotations[j].T))
            v_opt = _optimize_component_scales_numpy(
                num_vars=len(comp_indices),
                var_i=np.concatenate(var_i),
                var_j=np.concatenate(var_j),
                base_diff=np.concatenate(base_diff),
                dir_i=np.concatenate(dir_i),
                dir_j=np.concatenate(dir_j),
                prior_weight=_prior_weights(len(comp_indices), [local[anchor_idx]], anchor_sigma, reg_weight),
                prior_val=anchor_val,
                robust_delta=robust_delta,
                use_exp_param=use_exp_param,
                iters=iters,
            )
            solved_values = {idx: float(v_opt[local[idx]]) for idx in comp_indices}
            num_factors = 1 + (len(comp_indices) if reg_weight > 0.0 else 0) + sum(len(x) for x in var_i)
        else:
            graph = gtsam.NonlinearFactorGraph()
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
            solved_values = {idx: float(result.atDouble(_make_variable_key(use_exp_param, idx))) for idx in comp_indices}
            num_factors = graph.size()

        comp_abs_scales = []
        for idx in comp_indices:
            delta_value = _value_to_scale(solved_values[idx], use_exp_param)
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
            f"anchor={anchor_submap_key}, anchor_sigma={anchor_sigma:.6g}, factors={num_factors}, solver={solver}"
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
