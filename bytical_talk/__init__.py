"""bytical-talk — a smart-AI talking-head video system.

Two layers:
  * renderer  — SyncTalk_2D (fetched into upstream/, see scripts/fetch_upstream.sh)
                plus weight-safe quality modules in bytical_talk.render / .losses / .audio.
  * brain     — LLM/embedding-driven intelligence in bytical_talk.brain:
                  Director   (script -> emotion/prosody/SSML)
                  AutoConfig (video -> optimal render settings)
                  SelfQC     (evaluate output -> retry with better settings)

The brain is what makes this "actual AI" rather than a fixed pipeline: it
understands the script, adapts to the input video, and critiques its own output.
"""

__version__ = "0.1.0"

from .brain.autoconfig import RenderConfig, VideoStats, auto_config  # noqa: F401
from .brain.director import Director, PerformancePlan  # noqa: F401
from .brain.llm import LLMClient  # noqa: F401
from .brain.qc import SelfQC  # noqa: F401
