"""Single-view Depth Anything 3 (DA3METRIC-LARGE / DA3MONO-LARGE) as a TensorRT engine, for the mono path only
(the multi-view nested model of the default config is in da3_nested_trt.py), exposed with the subset of the DA3 api the
depthanything backend uses on its mono path (`inference(image=..., intrinsics=...)` -> Prediction-like object). The ONNX is exported from the
checkpoint outside this repo (as ros2-depth-anything-v3-trt does) and must take input [1, 3, H, W] ImageNet-normalised
RGB and return depth [1, 1, H, W] (metric) and sky [1, 1, H, W] (>= 0.5 = sky, as DA3's OutputProcessor). Preprocessing mirrors DA3's
InputProcessor (longest side -> process_res with INTER_AREA, sizes rounded to the patch multiple), so the engine's fixed H x W must be the
size DA3 would process the camera frames at (800 x 600 -> 504 x 378). Confidence is uniform (mono models carry none; ScaRF's percentile
filter then is a no-op, as for DA3MONO). Poses are never estimated: the backend's mono path attaches the input VIO extrinsics."""
import os, sys, time
from types import SimpleNamespace
import cv2, numpy as np, torch

from scarf_slam.backends.trt_runner import TrtRunner

MEAN = np.array([0.485, 0.456, 0.406], np.float32); STD = np.array([0.229, 0.224, 0.225], np.float32)


class DA3MonoTrt:
    def __init__(self, onnx_or_engine: str, process_res: int = 504, patch: int = 14):
        self.runner = TrtRunner(onnx_or_engine); self.process_res, self.patch = process_res, patch
        _, _, self.H, self.W = self.runner.shapes[self.runner.inputs[0]]
        self.t_pre = self.t_run = 0.0; self.n = 0
        print(f"[scarf] DA3 TensorRT engine {os.path.basename(self.runner.engine_path)} ({self.W}x{self.H})", flush=True)

    def _preprocess(self, img: np.ndarray):   # DA3 InputProcessor: upper_bound_resize to process_res, then floor to the patch multiple
        h, w = img.shape[:2]; s = self.process_res / float(max(h, w))
        nw, nh = max(1, int(round(w * s))), max(1, int(round(h * s)))
        arr = cv2.resize(img, (nw, nh), interpolation=cv2.INTER_CUBIC if s > 1.0 else cv2.INTER_AREA)
        nw, nh = (nw // self.patch) * self.patch, (nh // self.patch) * self.patch
        arr = arr[(arr.shape[0] - nh) // 2:(arr.shape[0] - nh) // 2 + nh, (arr.shape[1] - nw) // 2:(arr.shape[1] - nw) // 2 + nw]
        if (nh, nw) != (self.H, self.W):
            raise ValueError(f"DA3 TensorRT engine is {self.W}x{self.H} but the frame would be processed at {nw}x{nh}; export the ONNX at that size")
        return arr, (arr.astype(np.float32) / 255.0 - MEAN) / STD

    @torch.inference_mode()
    def inference(self, image, intrinsics=None, extrinsics=None, **kw):
        depths, skies, procs, ixts = [], [], [], []
        for i, im in enumerate(image):
            im = np.asarray(im)
            if im.ndim == 2: im = np.repeat(im[..., None], 3, -1)
            im = im[..., :3]; h0, w0 = im.shape[:2]
            t0 = time.time(); proc, x = self._preprocess(im); self.t_pre += time.time() - t0
            t0 = time.time(); d, sky = self.runner(torch.from_numpy(np.ascontiguousarray(x.transpose(2, 0, 1)))[None].cuda()); self.t_run += time.time() - t0
            depths.append(d[0, 0].cpu().numpy()); skies.append((sky[0, 0] >= 0.5).cpu().numpy()); procs.append(proc)
            if intrinsics is not None:
                K = np.asarray(intrinsics[i], np.float32).copy(); K[:1] *= self.W / float(w0); K[1:2] *= self.H / float(h0); ixts.append(K)
        self.n += len(image)
        depth = np.stack(depths).astype(np.float32)
        return SimpleNamespace(depth=depth, is_metric=1, sky=np.stack(skies), conf=np.ones_like(depth), extrinsics=None,
                               intrinsics=np.stack(ixts) if ixts else None, processed_images=np.stack(procs), gaussians=None, aux={}, scale_factor=None)
