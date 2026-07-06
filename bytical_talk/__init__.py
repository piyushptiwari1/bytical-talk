"""bytical-talk — a smart-AI talking-head video system.

Two layers:
  * renderer  — SyncTalk_2D (fetched into upstream/, see scripts/fetch_upstream.sh)
                plus a fast GPU-batched render path in bytical_talk.render.
  * brain     — content-analysis intelligence in bytical_talk.brain:
                  Director   (script -> per-segment importance / emphasis / prosody)
                  AutoConfig (video -> render settings)
                  SelfQC     (evaluate output -> retry with better settings)

Focus:
  A — speed: the render is CPU-bound (crop/paste + encode), not GPU-bound, so the
      fast path batches frames on the GPU, does crop/paste on the GPU, and uses
      hardware (NVENC) encode.
  B — content-adaptive quality: the brain reads the spoken content and decides
      where to spend render quality.
"""

__version__ = "0.1.0"

from .brain.autoconfig import RenderConfig, VideoStats, auto_config  # noqa: F401
from .brain.director import Director, PerformancePlan  # noqa: F401
from .brain.llm import LLMClient  # noqa: F401
from .brain.qc import SelfQC  # noqa: F401
