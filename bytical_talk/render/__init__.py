"""Render path: base per-frame reference + fast batched/fp16/streamed path,
plus the base quality modules (temporal smoothing, feather paste)."""

from .fast import render_video_fast  # noqa: F401
from .infer import render_video  # noqa: F401
from .smoothing import BoxSmoother, OneEuroFilter, feather_paste  # noqa: F401
