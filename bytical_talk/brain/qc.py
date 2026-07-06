"""Self-QC — the system evaluates its own output and decides whether to retry.

After a render, Self-QC measures objective quality (mouth-region sharpness and,
when a SyncNet checkpoint is available, lip-sync confidence), then asks the LLM to
judge the result against those numbers and propose concrete setting changes for a
retry. This closes a feedback loop: the model looks at what it produced, reasons
about it, and improves the next attempt — instead of blindly emitting one take.

Objective metrics use cv2/numpy (lazy-imported). The judgement degrades to a
deterministic rule if no LLM is configured.
"""

from __future__ import annotations

from dataclasses import dataclass, field

from .autoconfig import RenderConfig
from .llm import LLMClient


@dataclass
class QCMetrics:
    sharpness: float = 0.0          # mean Laplacian variance over the mouth ROI
    sharpness_min: float = 0.0      # worst frame (catches intermittent blur)
    temporal_jitter: float = 0.0    # mean inter-frame diff over the mouth ROI
    sync_confidence: float | None = None  # SyncNet LSE-C style score if available
    n_frames: int = 0


@dataclass
class QCReport:
    metrics: QCMetrics
    passed: bool
    issues: list[str] = field(default_factory=list)
    suggestion: dict = field(default_factory=dict)   # partial RenderConfig overrides
    rationale: str = ""


def measure(video_path: str, roi: tuple[float, float, float, float] = (0.30, 0.55, 0.70, 0.95)) -> QCMetrics:
    """Measure sharpness + temporal jitter over a mouth-region ROI (fractions of
    the frame: x0,y0,x1,y1). Defaults target the lower-center where the mouth is."""
    import cv2
    import numpy as np

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise FileNotFoundError(f"cannot open video: {video_path}")
    sharps, prev = [], None
    jitters = []
    n = 0
    while True:
        ok, f = cap.read()
        if not ok:
            break
        n += 1
        h, w = f.shape[:2]
        x0, y0, x1, y1 = int(roi[0] * w), int(roi[1] * h), int(roi[2] * w), int(roi[3] * h)
        crop = cv2.cvtColor(f[y0:y1, x0:x1], cv2.COLOR_BGR2GRAY)
        sharps.append(float(cv2.Laplacian(crop, cv2.CV_64F).var()))
        if prev is not None and prev.shape == crop.shape:
            jitters.append(float(np.abs(crop.astype(np.float32) - prev).mean()))
        prev = crop.astype(np.float32)
    cap.release()
    if not sharps:
        return QCMetrics()
    return QCMetrics(
        sharpness=float(sum(sharps) / len(sharps)),
        sharpness_min=float(min(sharps)),
        temporal_jitter=float(sum(jitters) / len(jitters)) if jitters else 0.0,
        n_frames=n,
    )


_SYSTEM = (
    "You are a strict QC reviewer for lip-sync presenter videos. You are given "
    "numeric quality metrics and the render settings used. Decide if the result "
    "passes, list issues, and suggest concrete setting changes to improve a retry. "
    "Return STRICT JSON."
)


def _judge_prompt(m: QCMetrics, cfg: RenderConfig) -> list[dict]:
    user = {
        "metrics": {
            "sharpness_mean": round(m.sharpness, 2),
            "sharpness_min": round(m.sharpness_min, 2),
            "temporal_jitter": round(m.temporal_jitter, 3),
            "sync_confidence": m.sync_confidence,
            "n_frames": m.n_frames,
        },
        "current_config": cfg.to_dict(),
        "instructions": (
            "Higher sharpness is better. Lower temporal_jitter is smoother. "
            "sync_confidence (if not null) higher is better; below ~3.0 is weak sync. "
            "Suggest changes only among: smooth(bool), smooth_min_cutoff(0.05-1.0), "
            "feather(0-20), match_train(bool)."
        ),
        "return_schema": {"passed": "bool", "issues": ["..."],
                          "suggestion": {"<config key>": "<value>"},
                          "rationale": "one sentence"},
    }
    import json
    return [{"role": "system", "content": _SYSTEM},
            {"role": "user", "content": json.dumps(user)}]


class SelfQC:
    def __init__(self, llm: LLMClient | None = None,
                 min_sharpness: float = 30.0, min_sync: float = 3.0):
        self.llm = llm or LLMClient()
        self.min_sharpness = min_sharpness
        self.min_sync = min_sync

    def review(self, metrics: QCMetrics, cfg: RenderConfig) -> QCReport:
        if self.llm.available:
            try:
                return self._review_llm(metrics, cfg)
            except Exception:
                pass
        return self._review_rules(metrics, cfg)

    def _review_llm(self, m: QCMetrics, cfg: RenderConfig) -> QCReport:
        data = self.llm.chat_json(_judge_prompt(m, cfg))
        suggestion = data.get("suggestion") or {}
        # keep only known, safe keys
        allowed = {"smooth", "smooth_min_cutoff", "feather", "match_train"}
        suggestion = {k: v for k, v in suggestion.items() if k in allowed}
        return QCReport(
            metrics=m,
            passed=bool(data.get("passed", False)),
            issues=[str(i) for i in (data.get("issues") or [])],
            suggestion=suggestion,
            rationale=str(data.get("rationale", "")),
        )

    def _review_rules(self, m: QCMetrics, cfg: RenderConfig) -> QCReport:
        issues, suggestion = [], {}
        passed = True
        if m.sharpness and m.sharpness < self.min_sharpness:
            passed = False
            issues.append(f"low sharpness {m.sharpness:.1f} < {self.min_sharpness}")
        if m.temporal_jitter and m.temporal_jitter > 4.0 and not cfg.smooth:
            passed = False
            issues.append(f"high jitter {m.temporal_jitter:.2f} with smoothing off")
            suggestion["smooth"] = True
        if m.sync_confidence is not None and m.sync_confidence < self.min_sync:
            passed = False
            issues.append(f"weak sync {m.sync_confidence:.2f} < {self.min_sync}")
        return QCReport(metrics=m, passed=passed, issues=issues,
                        suggestion=suggestion, rationale="rule-based review")


def apply_suggestion(cfg: RenderConfig, suggestion: dict) -> RenderConfig:
    """Return a new RenderConfig with QC suggestions applied."""
    from dataclasses import replace
    valid = {k: v for k, v in suggestion.items() if hasattr(cfg, k)}
    new = replace(cfg, **valid)
    if valid:
        new.notes = list(cfg.notes) + [f"QC retry: {valid}"]
    return new
