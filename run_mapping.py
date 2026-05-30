#!/usr/bin/env python3
"""Compatibility entrypoint for the ScaRF-SLAM mapping application."""

from scarf_slam.core.camera import FisheyeCamera, PinholeCamera, RotateParam

__all__ = [
    "FisheyeCamera",
    "ScaRFSLAM",
    "PinholeCamera",
    "RotateParam",
    "main",
]


def __getattr__(name: str):
    if name in {"ScaRFSLAM", "main"}:
        from scarf_slam.mapping_app import ScaRFSLAM, main

        return {"ScaRFSLAM": ScaRFSLAM, "main": main}[name]
    raise AttributeError(f"module 'run_mapping' has no attribute {name!r}")


if __name__ == "__main__":
    __getattr__("main")()
