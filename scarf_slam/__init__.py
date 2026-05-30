"""ScaRF-SLAM package entrypoints."""

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
    raise AttributeError(f"module 'scarf_slam' has no attribute {name!r}")
