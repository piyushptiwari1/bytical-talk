"""Content-adaptive quality (B): turn the Director's performance plan into a
per-frame importance profile, so the renderer can spend the expensive quality
operation only where the *content* matters and the cheap fast path elsewhere.

This is the "smart compute" synthesis: same (or lower) total cost, concentrated on
the segments that carry the message. The importance comes from the Director (which
already scored emphasis / intensity / emotion per sentence — no extra model call).

Frame alignment: the Director works on text and does not know per-word audio timing,
so we distribute the N output frames across segments in proportion to their spoken
length (character count, a good proxy for duration at a steady pace) and tag each
frame with its segment's importance. If real word-level timings are available later
(e.g. from the TTS), swap `segment_frame_spans` for those.
"""

from __future__ import annotations

from dataclasses import dataclass

from ..brain.director import PerformancePlan, Segment


@dataclass
class QualityPlan:
    """Per-frame importance in [0,1] plus a boolean 'enhance this frame' mask built
    from a threshold. The renderer reads `enhance` to decide where to spend effort."""

    importance: list[float]
    enhance: list[bool]

    @property
    def enhanced_fraction(self) -> float:
        return sum(self.enhance) / len(self.enhance) if self.enhance else 0.0


def segment_frame_spans(segments: list[Segment], n_frames: int) -> list[tuple[int, int, Segment]]:
    """Assign contiguous frame ranges to segments proportional to text length.
    Returns [(start, end, segment), ...] covering [0, n_frames)."""
    if not segments or n_frames <= 0:
        return []
    weights = [max(1, len(s.text)) for s in segments]
    total = sum(weights)
    spans, start, acc = [], 0, 0.0
    for i, (seg, w) in enumerate(zip(segments, weights)):
        acc += w
        end = n_frames if i == len(segments) - 1 else round(n_frames * acc / total)
        end = max(start, min(n_frames, end))
        spans.append((start, end, seg))
        start = end
    return spans


def build_quality_plan(plan: PerformancePlan, n_frames: int,
                       threshold: float = 0.6, budget: float | None = None) -> QualityPlan:
    """Per-frame importance profile from a Director PerformancePlan.

    Two ways to pick which frames get the expensive quality op:
      * absolute `threshold` — enhance frames with importance >= threshold.
      * relative `budget` (0..1) — enhance only the top `budget` fraction of frames
        by importance. Use this when the Director rates most lines important (common
        on punchy copy): it keeps the compute bounded and still concentrates effort
        on the *most* important moments. `budget` overrides `threshold` when given.
    """
    importance = [0.0] * max(0, n_frames)
    for start, end, seg in segment_frame_spans(plan.segments, n_frames):
        imp = seg.importance
        for f in range(start, end):
            importance[f] = imp
    if budget is not None and importance:
        budget = max(0.0, min(1.0, budget))
        k = int(round(budget * len(importance)))
        if k <= 0:
            enhance = [False] * len(importance)
        elif k >= len(importance):
            enhance = [True] * len(importance)
        else:
            cutoff = sorted(importance, reverse=True)[k - 1]
            enhance, taken = [False] * len(importance), 0
            # take highest-importance frames first, up to the budget
            for idx in sorted(range(len(importance)), key=lambda j: importance[j], reverse=True):
                if taken >= k:
                    break
                enhance[idx] = True
                taken += 1
    else:
        enhance = [imp >= threshold for imp in importance]
    return QualityPlan(importance=importance, enhance=enhance)
