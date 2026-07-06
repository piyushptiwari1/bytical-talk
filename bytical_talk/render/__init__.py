"""Render path: base per-frame reference + fast batched/fp16/streamed path,
plus the base quality modules (temporal smoothing, feather paste)."""

"""Render path: base per-frame reference + fast batched/fp16/streamed path,
plus content-adaptive quality and the base quality modules.

`render_video` / `render_video_fast` pull in cv2/torch, so they are imported lazily
via __getattr__ — the light modules (adaptive, smoothing) import with no heavy deps.
"""

from .adaptive import QualityPlan, build_quality_plan, segment_frame_spans  # noqa: F401
from .smoothing import BoxSmoother, OneEuroFilter, feather_paste  # noqa: F401

__all__ = [
    "QualityPlan", "build_quality_plan", "segment_frame_spans",
    "BoxSmoother", "OneEuroFilter", "feather_paste",
    "render_video", "render_video_fast",
]


def __getattr__(name):  # lazy: only import cv2/torch-heavy modules on demand
    if name == "render_video":
        from .infer import render_video
        return render_video
    if name == "render_video_fast":
        from .fast import render_video_fast
        return render_video_fast
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
