"""DA3NESTED-GIANT-LARGE-1.1 (ScaRF's default depth model) with its two networks as TensorRT engines and the nested combination in torch.
Drop-in for `DepthAnything3.model` (the api keeps its input/output processing): called as
    model(image [1,N,3,H,W], extrinsics_norm [1,N,4,4], intrinsics [1,N,3,3], export_feat_layers, infer_gs, use_ray_pose, ref_view_strategy)
and returns the same addict Dict (depth, depth_conf [1,N,H,W], extrinsics [1,N,3,4], intrinsics [1,N,3,3], is_metric, scale_factor).
Engines: DA3NESTED_anyview_<N>x<H>x<W>.onnx and DA3NESTED_metric_<N>x<H>x<W>.onnx, exported from the checkpoint outside this repo and
built / cached per GPU by TrtRunner on first use. The post-network steps are the model's own methods (they use no weights):
metric branch sky quantile step, metric scaling by the predicted intrinsics, least-squares depth alignment, sky handling."""
import os, sys, time
import torch
from addict import Dict
from depth_anything_3.model.da3 import DepthAnything3Net, NestedDepthAnything3Net
from scarf_slam.backends.trt_runner import TrtRunner


class DA3NestedTrt(torch.nn.Module):
    def __init__(self, onnx_dir: str, views: int = 6, height: int = 378, width: int = 504):
        super().__init__()
        tag = f"{views}x{height}x{width}"
        self.anyview = TrtRunner(os.path.join(onnx_dir, f"DA3NESTED_anyview_{tag}.onnx"), workspace_gb=8)
        self.metric = TrtRunner(os.path.join(onnx_dir, f"DA3NESTED_metric_{tag}.onnx"), workspace_gb=8)
        self.shape = (1, views, 3, height, width); self.t_any = self.t_met = self.t_post = 0.0; self.n = 0
        self.register_buffer("_device_anchor", torch.zeros(1, device="cuda"))   # DepthAnything3._get_model_device() looks for a parameter/buffer
        print(f"[scarf] DA3 nested TensorRT engines: {os.path.basename(self.anyview.engine_path)}, {os.path.basename(self.metric.engine_path)}", flush=True)

    @torch.inference_mode()
    def forward(self, x, extrinsics=None, intrinsics=None, export_feat_layers=None, infer_gs=False, use_ray_pose=False, ref_view_strategy="saddle_balanced"):
        assert tuple(x.shape) == self.shape, f"engine is {self.shape}, got {tuple(x.shape)}"
        assert extrinsics is not None and intrinsics is not None, "the nested TensorRT path needs camera extrinsics/intrinsics (ScaRF always passes the VIO poses)"
        with torch.autocast("cuda", enabled=False):
            t0 = time.time(); d, c, e, k = self.anyview(x.float(), extrinsics.float(), intrinsics.float()); torch.cuda.synchronize(); self.t_any += time.time() - t0
            t0 = time.time(); md, ms = self.metric(x.float()); torch.cuda.synchronize(); self.t_met += time.time() - t0
            t0 = time.time()
            out = Dict(depth=d, depth_conf=c, extrinsics=e, intrinsics=k)
            mout = DepthAnything3Net._process_mono_sky_estimation(None, Dict(depth=md, sky=ms))   # the metric branch's in-model sky step
            out = NestedDepthAnything3Net._apply_metric_scaling(None, out, mout)
            out = NestedDepthAnything3Net._apply_depth_alignment(None, out, mout)
            out = NestedDepthAnything3Net._handle_sky_regions(None, out, mout)
            out.aux = {}
            self.t_post += time.time() - t0; self.n += 1
        return out

    def timing(self):
        n = max(self.n, 1); return f"nested-trt per submap: anyview {1000*self.t_any/n:.0f} ms, metric {1000*self.t_met/n:.0f} ms, post {1000*self.t_post/n:.0f} ms ({self.n} submaps)"
