"""The brain layer — LLM/embedding-driven intelligence."""

from .autoconfig import RenderConfig, VideoStats, analyze_video, auto_config, recommend
from .director import Director, PerformancePlan, Segment, build_ssml
from .llm import LLMClient, LLMConfig
from .qc import QCMetrics, QCReport, SelfQC, apply_suggestion, measure

__all__ = [
    "LLMClient", "LLMConfig",
    "Director", "PerformancePlan", "Segment", "build_ssml",
    "RenderConfig", "VideoStats", "analyze_video", "recommend", "auto_config",
    "SelfQC", "QCMetrics", "QCReport", "measure", "apply_suggestion",
]
