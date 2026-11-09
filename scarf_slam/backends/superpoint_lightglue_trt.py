"""SuperPoint + LightGlue (vismatch's bundled LightGlue, ScaRF's `superpoint-lightglue` matcher) with the two networks as TensorRT engines.
Drop-in for `matcher.extractor` (extract(img) -> feats dict) and `matcher.matcher` (dict -> matches dict) of the vismatch matcher.
- SuperPoint: the dense part (VGG encoder, score head with softmax / depth-to-space / NMS, descriptor head) is one fp32 engine per input
  size (the scores are too small for fp16); border removal, thresholding, top-k and descriptor sampling stay in torch (data-dependent, cheap).
- LightGlue: the 9 self/cross-attention layers with rotary positional encoding are one static engine for N keypoints per image
  (max_num_keypoints); shorter keypoint sets are zero-padded and the padded keys are masked with an additive bias, so the real tokens see
  exactly the unpadded computation. The final projection / assignment / mutual-NN filter stay in torch on the unpadded tokens
  (the double softmax must not see padding). Early stopping / point pruning are off in ScaRF (depth_confidence = width_confidence = -1).
ONNX files are exported on first use (fp32; LightGlue also as an fp16-typed copy for TensorRT >= 11), engines are built / cached per GPU by TrtRunner.
Enable with SCARF_MATCHER_TRT=<dir> (see scarf_slam/mapping/feature_matching.py)."""
import os, sys, time
import torch, torch.nn.functional as F
from torch import nn
from scarf_slam.backends.trt_runner import TrtRunner


def _lightglue_modules():
    from vismatch import THIRD_PARTY_DIR
    from vismatch.utils import add_to_path
    add_to_path(THIRD_PARTY_DIR.joinpath("LightGlue"))
    from lightglue import superpoint as sp_mod, lightglue as lg_mod
    from lightglue.utils import ImagePreprocessor
    return sp_mod, lg_mod, ImagePreprocessor


def _to_fp16_onnx(src: str) -> str:
    """fp16-typed copy for the strongly typed TensorRT >= 11 builder (the listed ops keep fp32: they overflow or lose the index semantics)."""
    import onnx
    from onnxconverter_common import float16
    dst = src.replace(".onnx", "_fp16.onnx")
    if os.path.exists(dst): return dst
    block = list(float16.DEFAULT_OP_BLOCK_LIST) + ["Greater", "GreaterOrEqual", "Less", "LessOrEqual", "Equal", "Where", "And", "Or", "Not",
                                                   "IsNaN", "IsInf", "Clip", "Max", "Min"]
    m16 = float16.convert_float_to_float16(onnx.load(src), keep_io_types=True, disable_shape_infer=True, check_fp16_ready=False, op_block_list=block)
    # the converter leaves explicit Cast(to=FLOAT) nodes (from .float() in the model) untouched while retyping their outputs to fp16:
    # retarget them to fp16 unless they feed a graph output (kept fp32 by keep_io_types)
    outs = {o.name for o in m16.graph.output}
    for n in m16.graph.node:   # (the converter's own casts around blocked ops are named *_input_cast* / *_output_cast* and stay as they are)
        if n.op_type == "Cast" and not (set(n.output) & outs) and "_input_cast" not in n.name and "_output_cast" not in n.name:
            for a in n.attribute:
                if a.name == "to" and a.i == onnx.TensorProto.FLOAT: a.i = onnx.TensorProto.FLOAT16
    onnx.save(m16, dst); return dst


def _export(module: nn.Module, args: tuple, path: str, input_names, output_names, fp16: bool = True):
    if os.path.exists(path): return path
    os.makedirs(os.path.dirname(path), exist_ok=True); t0 = time.time()
    with torch.inference_mode():   # torch.export-based exporter: the TorchScript one trips over the head unflatten/transpose in the attention blocks
        torch.onnx.export(module, args, path, input_names=input_names, output_names=output_names, dynamo=True, optimize=True)
    if fp16: _to_fp16_onnx(path)
    print(f"[lg-trt] exported {os.path.basename(path)} ({'+fp16' if fp16 else 'fp32 only'}) in {time.time() - t0:.0f} s", flush=True)
    return path


class _SuperPointDense(nn.Module):
    """image [1,1,H,W] (gray, 0..1) -> NMS'd score map [1,H,W], raw descriptor map [1,256,H/8,W/8] (SuperPoint.forward up to the thresholding)."""
    def __init__(self, sp, nms_radius: int):
        super().__init__(); self.sp = sp; self.nms_radius = nms_radius; self.simple_nms = _lightglue_modules()[0].simple_nms

    def forward(self, image):
        sp = self.sp; relu = F.relu
        x = relu(sp.conv1a(image)); x = relu(sp.conv1b(x)); x = sp.pool(x)
        x = relu(sp.conv2a(x)); x = relu(sp.conv2b(x)); x = sp.pool(x)
        x = relu(sp.conv3a(x)); x = relu(sp.conv3b(x)); x = sp.pool(x)
        x = relu(sp.conv4a(x)); x = relu(sp.conv4b(x))
        scores = F.softmax(sp.convPb(relu(sp.convPa(x))), 1)[:, :-1]
        b, _, h, w = scores.shape
        scores = scores.permute(0, 2, 3, 1).reshape(b, h, w, 8, 8).permute(0, 1, 3, 2, 4).reshape(b, h * 8, w * 8)
        scores = self.simple_nms(scores, self.nms_radius)
        desc = sp.convDb(relu(sp.convDa(x)))   # unit-normalised outside the engine: the L2 norm over 256 channels overflows fp16 in TensorRT
        return scores, desc


class SuperPointTrt(nn.Module):
    def __init__(self, extractor, onnx_dir: str, workspace_gb: float = 2.0):
        super().__init__()
        self.sp = extractor; self.conf = extractor.conf; self.preprocess_conf = dict(extractor.preprocess_conf); self.onnx_dir = onnx_dir
        self.device = next(extractor.parameters()).device; self.runners = {}; self.workspace_gb = workspace_gb
        sp_mod, _, self.ImagePreprocessor = _lightglue_modules()
        self.sample_descriptors, self.top_k_keypoints = sp_mod.sample_descriptors, sp_mod.top_k_keypoints
        self.t_net = self.t_post = 0.0; self.n = 0

    def _runner(self, h: int, w: int) -> TrtRunner:
        if (h, w) not in self.runners:
            assert h % 8 == 0 and w % 8 == 0, (h, w)
            path = os.path.join(self.onnx_dir, f"SUPERPOINT_dense_{h}x{w}.onnx")
            dense = _SuperPointDense(self.sp, int(self.conf.nms_radius)).eval()
            # fp32 only: the keypoint scores (softmax over 65 cells, values ~1e-5) fall into the fp16 subnormal range and the NMS then
            # picks different maxima (TensorRT 11 runs an fp16-typed network in pure fp16); the net is small, fp32 is a few ms
            _export(dense, (torch.zeros(1, 1, h, w, device=self.device),), path, ["image"], ["scores", "desc_map"], fp16=False)
            self.runners[(h, w)] = TrtRunner(path, workspace_gb=self.workspace_gb)
            print(f"[scarf] SuperPoint TensorRT engine: {os.path.basename(self.runners[(h, w)].engine_path)}", flush=True)
        return self.runners[(h, w)]

    @torch.no_grad()
    def extract(self, img: torch.Tensor, **conf) -> dict:
        """Same contract as lightglue.utils.Extractor.extract (online resize to the long side, keypoints mapped back to the input size)."""
        from kornia.color import rgb_to_grayscale
        if img.dim() == 3: img = img[None]
        assert img.dim() == 4 and img.shape[0] == 1
        shape = img.shape[-2:][::-1]
        img, scales = self.ImagePreprocessor(**{**self.preprocess_conf, **conf})(img)
        if img.shape[1] == 3: img = rgb_to_grayscale(img)
        h, w = img.shape[-2:]
        t0 = time.time(); scores, desc_map = self._runner(h, w)(img.float().contiguous()); self.t_net += time.time() - t0
        t0 = time.time()
        scores = scores[0]; desc_map = F.normalize(desc_map, p=2, dim=1)
        pad = int(self.conf.remove_borders)
        if pad: scores[:pad] = -1; scores[:, :pad] = -1; scores[-pad:] = -1; scores[:, -pad:] = -1
        best = torch.where(scores > self.conf.detection_threshold)
        kp, s = torch.stack(best, dim=-1), scores[best]
        if self.conf.max_num_keypoints is not None: kp, s = self.top_k_keypoints(kp, s, int(self.conf.max_num_keypoints))
        kp = torch.flip(kp, [1]).float()                                          # (h, w) -> (x, y)
        desc = self.sample_descriptors(kp[None], desc_map, 8)[0]                 # [256, K]
        feats = {"keypoints": kp[None], "keypoint_scores": s[None], "descriptors": desc.transpose(-1, -2).contiguous()[None]}
        feats["image_size"] = torch.tensor(shape)[None].to(img).float()
        feats["keypoints"] = (feats["keypoints"] + 0.5) / scales[None] - 0.5
        self.t_post += time.time() - t0; self.n += 1
        return feats


class _LightGlueTrunk(nn.Module):
    """kpts0/1 [1,N,2] (normalized), desc0/1 [1,N,256], bias0/1 [1,1,1,N] (0 = real key, -1e4 = padded key) -> desc0/1 after all layers.
    Same maths as TransformerLayer (SelfBlock / CrossBlock) with explicit attention; only keys are masked, padded query rows are discarded by the caller."""
    def __init__(self, lg):
        super().__init__(); self.lg = lg; self.layers = lg.transformers; self.posenc = lg.posenc; self.input_proj = lg.input_proj
        self.rot = _lightglue_modules()[1].apply_cached_rotary_emb

    @staticmethod
    def _attn(q, k, v, bias):
        sim = torch.matmul(q, k.transpose(-2, -1)) * (q.shape[-1] ** -0.5) + bias
        return torch.matmul(F.softmax(sim, -1), v)

    def _self(self, blk, x, enc, bias):
        qkv = blk.Wqkv(x).unflatten(-1, (blk.num_heads, -1, 3)).transpose(1, 2)
        q, k, v = qkv[..., 0], qkv[..., 1], qkv[..., 2]
        q, k = self.rot(enc, q), self.rot(enc, k)
        msg = blk.out_proj(self._attn(q, k, v, bias).transpose(1, 2).flatten(start_dim=-2))
        return x + blk.ffn(torch.cat([x, msg], -1))

    def _cross(self, blk, x0, x1, bias0, bias1):
        heads = blk.heads
        qk0, qk1, v0, v1 = [t.unflatten(-1, (heads, -1)).transpose(1, 2) for t in (blk.to_qk(x0), blk.to_qk(x1), blk.to_v(x0), blk.to_v(x1))]
        qk0, qk1 = qk0 * blk.scale ** 0.5, qk1 * blk.scale ** 0.5
        sim = torch.matmul(qk0, qk1.transpose(-2, -1))
        m0 = torch.matmul(F.softmax(sim + bias1, -1), v1)
        m1 = torch.matmul(F.softmax(sim.transpose(-2, -1) + bias0, -1), v0)
        m0, m1 = [blk.to_out(t.transpose(1, 2).flatten(start_dim=-2)) for t in (m0, m1)]
        return x0 + blk.ffn(torch.cat([x0, m0], -1)), x1 + blk.ffn(torch.cat([x1, m1], -1))

    def forward(self, kpts0, kpts1, desc0, desc1, bias0, bias1):
        d0, d1 = self.input_proj(desc0), self.input_proj(desc1)
        e0, e1 = self.posenc(kpts0), self.posenc(kpts1)
        for layer in self.layers:
            d0 = self._self(layer.self_attn, d0, e0, bias0); d1 = self._self(layer.self_attn, d1, e1, bias1)
            d0, d1 = self._cross(layer.cross_attn, d0, d1, bias0, bias1)
        return d0, d1


class LightGlueTrt(nn.Module):
    def __init__(self, lightglue, onnx_dir: str, n_kpts: int = 1024, workspace_gb: float = 2.0):
        super().__init__()
        self.lg = lightglue; self.conf = lightglue.conf; self.n = n_kpts; self.device = next(lightglue.parameters()).device
        assert self.conf.depth_confidence < 0 and self.conf.width_confidence < 0, "the TensorRT trunk is static: no early stopping / point pruning"
        _, lg_mod, _ = _lightglue_modules(); self.normalize_keypoints, self.filter_matches = lg_mod.normalize_keypoints, lg_mod.filter_matches
        path = os.path.join(onnx_dir, f"LIGHTGLUE_{self.conf.weights}_trunk_{n_kpts}.onnx")
        d = self.conf.input_dim
        z = lambda *s: torch.zeros(*s, device=self.device)
        _export(_LightGlueTrunk(lightglue).eval(), (z(1, n_kpts, 2), z(1, n_kpts, 2), z(1, n_kpts, d), z(1, n_kpts, d), z(1, 1, 1, n_kpts), z(1, 1, 1, n_kpts)),
                path, ["kpts0", "kpts1", "desc0", "desc1", "bias0", "bias1"], ["out0", "out1"])
        self.trunk = TrtRunner(path, workspace_gb=workspace_gb)
        print(f"[scarf] LightGlue TensorRT engine: {os.path.basename(self.trunk.engine_path)}", flush=True)
        self.t_net = self.t_post = 0.0; self.calls = 0
        # batched trunk (forward_batch): one engine call for the pairs of a submap (6 frames x 5 previous = 15 pairs); built on first use
        # off by default (SCARF_LG_TRT_BATCH=0): on Jetson Thor the trunk is compute-bound and the batched engine is slower per pair
        # (5.9 vs 5.3 ms); on an RTX 5090 it saves ~20 % (1.24 vs 1.53 ms per pair)
        self.onnx_dir = onnx_dir; self.workspace_gb = workspace_gb; self.batch = int(os.environ.get("SCARF_LG_TRT_BATCH", 0)); self.trunk_b = None

    def _pad(self, kpts, desc):
        m = kpts.shape[1]; assert m <= self.n, f"{m} keypoints > engine size {self.n}"
        kp = F.pad(kpts, (0, 0, 0, self.n - m)); de = F.pad(desc, (0, 0, 0, self.n - m))
        bias = torch.zeros(1, 1, 1, self.n, device=kpts.device); bias[..., m:] = -1e4
        return kp, de, bias

    def _empty(self, kpts0, kpts1):
        b, m, _ = kpts0.shape; b, n, _ = kpts1.shape
        return {"matches0": kpts0.new_full((b, m), -1, dtype=torch.long), "matches1": kpts1.new_full((b, n), -1, dtype=torch.long),
                "matching_scores0": kpts0.new_zeros((b, m)), "matching_scores1": kpts1.new_zeros((b, n)), "stop": self.conf.n_layers,
                "matches": kpts0.new_empty((b, 0, 2), dtype=torch.long), "scores": kpts0.new_empty((b, 0)),
                "prune0": kpts0.new_ones((b, m)) * self.conf.n_layers, "prune1": kpts1.new_ones((b, n)) * self.conf.n_layers}

    def _assign(self, o0, o1, m, n):
        """final projection + assignment + mutual-NN filter on the unpadded tokens (torch), -> the LightGlue output dict"""
        scores, _ = self.lg.log_assignment[-1](o0[:, :m], o1[:, :n])
        m0, m1, ms0, ms1 = self.filter_matches(scores, self.conf.filter_threshold)
        valid = m0[0] > -1; i0 = torch.where(valid)[0]; i1 = m0[0][valid]
        return {"matches0": m0, "matches1": m1, "matching_scores0": ms0, "matching_scores1": ms1, "stop": self.conf.n_layers,
                "matches": [torch.stack([i0, i1], -1)], "scores": [ms0[0][valid]],
                "prune0": torch.ones_like(ms0) * self.conf.n_layers, "prune1": torch.ones_like(ms1) * self.conf.n_layers}

    def _get_trunk_b(self):
        if self.trunk_b is None:
            B, N, d = self.batch, self.n, self.conf.input_dim
            path = os.path.join(self.onnx_dir, f"LIGHTGLUE_{self.conf.weights}_trunk_{N}_b{B}.onnx")
            z = lambda *s: torch.zeros(*s, device=self.device)
            _export(_LightGlueTrunk(self.lg).eval(), (z(B, N, 2), z(B, N, 2), z(B, N, d), z(B, N, d), z(B, 1, 1, N), z(B, 1, 1, N)),
                    path, ["kpts0", "kpts1", "desc0", "desc1", "bias0", "bias1"], ["out0", "out1"])
            self.trunk_b = TrtRunner(path, workspace_gb=self.workspace_gb)
            print(f"[scarf] LightGlue TensorRT batched engine: {os.path.basename(self.trunk_b.engine_path)}", flush=True)
        return self.trunk_b

    @torch.inference_mode()
    def forward_batch(self, pairs) -> list:
        """pairs: list of (data0, data1) feature dicts -> list of output dicts (same contract as forward), the trunk run on
        batches of self.batch pairs (empty slots are zero-padded and ignored)."""
        outs = [None] * len(pairs); trunk = self._get_trunk_b(); B = self.batch
        todo = []
        for i, (data0, data1) in enumerate(pairs):
            k0, k1 = data0["keypoints"].float(), data1["keypoints"].float()
            if k0.shape[1] == 0 or k1.shape[1] == 0: outs[i] = self._empty(k0, k1)
            else: todo.append(i)
        for s in range(0, len(todo), B):
            chunk = todo[s:s + B]; K0 = []; K1 = []; D0 = []; D1 = []; B0 = []; B1 = []; mn = []
            for i in chunk:
                data0, data1 = pairs[i]; k0, k1 = data0["keypoints"].float(), data1["keypoints"].float()
                d0, d1 = data0["descriptors"].float().contiguous(), data1["descriptors"].float().contiguous()
                mn.append((k0.shape[1], k1.shape[1]))
                k0 = self.normalize_keypoints(k0, data0.get("image_size")); k1 = self.normalize_keypoints(k1, data1.get("image_size"))
                k0, d0, b0 = self._pad(k0, d0); k1, d1, b1 = self._pad(k1, d1)
                K0.append(k0); K1.append(k1); D0.append(d0); D1.append(d1); B0.append(b0); B1.append(b1)
            pad = B - len(chunk)   # unused slots: zeros, every key masked
            cat = lambda L, shape: torch.cat(L + [torch.zeros(pad, *shape, device=self.device)] if pad else L, 0)
            zb = torch.full((1, 1, 1, self.n), -1e4, device=self.device)
            t0 = time.time()
            o0, o1 = trunk(cat(K0, (self.n, 2)), cat(K1, (self.n, 2)), cat(D0, (self.n, D0[0].shape[-1])), cat(D1, (self.n, D1[0].shape[-1])),
                           torch.cat(B0 + [zb] * pad, 0), torch.cat(B1 + [zb] * pad, 0))
            self.t_net += time.time() - t0; t0 = time.time()
            for j, i in enumerate(chunk):
                m, n = mn[j]; outs[i] = self._assign(o0[j:j + 1], o1[j:j + 1], m, n)
            self.t_post += time.time() - t0; self.calls += len(chunk)
        return outs

    @torch.inference_mode()
    def forward(self, data: dict) -> dict:
        data0, data1 = data["image0"], data["image1"]
        kpts0, kpts1 = data0["keypoints"].float(), data1["keypoints"].float()
        b, m, _ = kpts0.shape; b, n, _ = kpts1.shape; assert b == 1
        desc0, desc1 = data0["descriptors"].float().contiguous(), data1["descriptors"].float().contiguous()
        if m == 0 or n == 0:
            return {"matches0": kpts0.new_full((b, m), -1, dtype=torch.long), "matches1": kpts1.new_full((b, n), -1, dtype=torch.long),
                    "matching_scores0": kpts0.new_zeros((b, m)), "matching_scores1": kpts1.new_zeros((b, n)), "stop": self.conf.n_layers,
                    "matches": kpts0.new_empty((b, 0, 2), dtype=torch.long), "scores": kpts0.new_empty((b, 0)),
                    "prune0": kpts0.new_ones((b, m)) * self.conf.n_layers, "prune1": kpts1.new_ones((b, n)) * self.conf.n_layers}
        k0 = self.normalize_keypoints(kpts0, data0.get("image_size")); k1 = self.normalize_keypoints(kpts1, data1.get("image_size"))
        k0, d0, b0 = self._pad(k0, desc0); k1, d1, b1 = self._pad(k1, desc1)
        t0 = time.time(); o0, o1 = self.trunk(k0, k1, d0, d1, b0, b1); self.t_net += time.time() - t0
        t0 = time.time()
        scores, _ = self.lg.log_assignment[-1](o0[:, :m], o1[:, :n])
        m0, m1, ms0, ms1 = self.filter_matches(scores, self.conf.filter_threshold)
        valid = m0[0] > -1; i0 = torch.where(valid)[0]; i1 = m0[0][valid]
        out = {"matches0": m0, "matches1": m1, "matching_scores0": ms0, "matching_scores1": ms1, "stop": self.conf.n_layers,
               "matches": [torch.stack([i0, i1], -1)], "scores": [ms0[0][valid]],
               "prune0": torch.ones_like(ms0) * self.conf.n_layers, "prune1": torch.ones_like(ms1) * self.conf.n_layers}
        self.t_post += time.time() - t0; self.calls += 1
        return out

    def timing(self, sp: "SuperPointTrt" = None):
        s = f"lightglue-trt per pair: net {1000*self.t_net/max(self.calls,1):.1f} ms, post {1000*self.t_post/max(self.calls,1):.1f} ms ({self.calls} pairs)"
        if sp is not None: s += f"; superpoint-trt per frame: net {1000*sp.t_net/max(sp.n,1):.1f} ms, post {1000*sp.t_post/max(sp.n,1):.1f} ms ({sp.n} frames)"
        return s
