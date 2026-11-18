"""Minimal TensorRT runner: one fixed-shape engine, built from the ONNX on first use and cached next to it (engines are GPU-specific;
the cache name carries the device). The engine's input and output buffers are torch CUDA tensors, bound by their data_ptr and run on
torch's current stream, so there is no separate device allocator (pycuda / cuda-python) and no host round trip around a call.
TensorRT 10 builds fp16 through the builder flag; TensorRT >= 11 networks are strongly typed, so an fp16-typed ONNX is used when
present. Written for the Jetson, where onnxruntime has no CUDA provider; used by the TensorRT backends beside it and usable for any
other fixed-shape ONNX model."""
import os, re, time
import numpy as np, torch


class TrtRunner:
    def __init__(self, onnx_path: str, fp16: bool = True, workspace_gb: float = 4.0, cache_dir: str = None, verbose: bool = True):
        import tensorrt as trt
        self.trt = trt; logger = trt.Logger(trt.Logger.WARNING)
        major = int(trt.__version__.split(".")[0])
        if major >= 11 and fp16 and os.path.exists(onnx_path.replace(".onnx", "_fp16.onnx")):
            onnx_path = onnx_path.replace(".onnx", "_fp16.onnx")   # precision follows the ONNX dtypes on TensorRT >= 11
        dev = re.sub(r"[^A-Za-z0-9]+", "_", torch.cuda.get_device_name(0)).strip("_")
        cache_dir = cache_dir or os.path.dirname(os.path.abspath(onnx_path))
        eng = os.path.join(cache_dir, os.path.basename(onnx_path).replace(".onnx", f".{dev}.trt{trt.__version__.split('.')[0]}.{'fp16' if fp16 else 'fp32'}.engine"))
        if not os.path.exists(eng):
            t0 = time.time(); builder = trt.Builder(logger); net = builder.create_network(0); parser = trt.OnnxParser(net, logger)
            if not parser.parse_from_file(onnx_path):   # path-based parse resolves external weight files (models > 2 GB) next to the ONNX
                raise RuntimeError("TensorRT could not parse " + onnx_path + ": " + "; ".join(str(parser.get_error(i)) for i in range(parser.num_errors)))
            cfg = builder.create_builder_config(); cfg.set_memory_pool_limit(trt.MemoryPoolType.WORKSPACE, int(workspace_gb * (1 << 30)))
            if fp16 and hasattr(trt.BuilderFlag, "FP16"): cfg.set_flag(trt.BuilderFlag.FP16)
            if "_fp8" in os.path.basename(onnx_path) and hasattr(trt.BuilderFlag, "FP8"): cfg.set_flag(trt.BuilderFlag.FP8)     # explicitly quantized (Q/DQ) ONNX from ModelOpt
            if "_int8" in os.path.basename(onnx_path) and hasattr(trt.BuilderFlag, "INT8"): cfg.set_flag(trt.BuilderFlag.INT8)
            ser = builder.build_serialized_network(net, cfg)
            if ser is None: raise RuntimeError("TensorRT engine build failed for " + onnx_path)
            os.makedirs(cache_dir, exist_ok=True); open(eng, "wb").write(ser)
            if verbose: print(f"[trt] built {os.path.basename(eng)} in {time.time() - t0:.0f} s", flush=True)
        self.engine = trt.Runtime(logger).deserialize_cuda_engine(open(eng, "rb").read()); self.ctx = self.engine.create_execution_context()
        self.names = [self.engine.get_tensor_name(i) for i in range(self.engine.num_io_tensors)]
        self.inputs = [n for n in self.names if self.engine.get_tensor_mode(n) == trt.TensorIOMode.INPUT]
        self.outputs = [n for n in self.names if n not in self.inputs]
        self.shapes = {n: tuple(self.engine.get_tensor_shape(n)) for n in self.names}
        self.bufs = {n: torch.empty(self.shapes[n], dtype=torch.float32, device="cuda") for n in self.names}
        for n in self.names: self.ctx.set_tensor_address(n, self.bufs[n].data_ptr())
        self.stream = torch.cuda.Stream(); self.engine_path = eng

    def __call__(self, *xs: torch.Tensor) -> list:
        """xs: one float32 CUDA tensor per engine input (same shape). Returns the output tensors (clones, so buffers can be reused)."""
        # the inputs are produced on the caller's stream: make the engine stream wait for it, otherwise the input copies can
        # read tensors whose producing kernels have not finished (seen as run-to-run varying SuperPoint keypoints / DA3 depth)
        self.stream.wait_stream(torch.cuda.current_stream())
        with torch.cuda.stream(self.stream):
            for n, x in zip(self.inputs, xs): self.bufs[n].copy_(x.reshape(self.shapes[n]))
            self.ctx.execute_async_v3(self.stream.cuda_stream)
            outs = [self.bufs[n].clone() for n in self.outputs]
        self.stream.synchronize()
        for o in outs: o.record_stream(torch.cuda.current_stream())
        return outs
