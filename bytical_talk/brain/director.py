"""Director — the LLM-driven "performance brain".

Given a raw script, the Director produces a structured *performance plan*: it reads
the meaning of each sentence and decides emotion, intensity, emphasis words, pacing
and pauses, then emits SSML (Amazon Polly-compatible) for TTS plus an emotion
timeline the (emotion-conditioned) renderer can consume.

This is the difference between "read these words" and "perform this script":
delivery is derived from understanding, not from flat text.

Degrades gracefully: with no LLM credentials it falls back to neutral sentence
segmentation so the pipeline still runs.
"""

from __future__ import annotations

import re
from dataclasses import asdict, dataclass, field

from .llm import LLMClient

EMOTIONS = ["neutral", "happy", "excited", "confident", "empathetic",
            "serious", "concerned", "reassuring", "curious", "sad"]

PACES = ["slow", "normal", "fast"]


@dataclass
class Segment:
    text: str
    emotion: str = "neutral"
    intensity: float = 0.5            # 0..1
    emphasis: list[str] = field(default_factory=list)
    pace: str = "normal"
    pause_after_ms: int = 250

    @property
    def importance(self) -> float:
        """Content-importance in 0..1, derived from what the Director already
        produced (no extra LLM call). Drives content-adaptive quality: emphasized,
        emotionally-charged, or non-neutral lines score higher and get more render
        effort. Neutral filler scores low."""
        score = 0.55 * self.intensity
        score += 0.20 if self.emphasis else 0.0
        # non-neutral emotions carry more weight; "neutral" stays low
        score += 0.0 if self.emotion == "neutral" else 0.20
        # emphatic emotions get a touch more
        if self.emotion in ("excited", "confident", "serious", "concerned"):
            score += 0.05
        return max(0.0, min(1.0, score))



@dataclass
class PerformancePlan:
    segments: list[Segment]
    overall_tone: str = "neutral"
    ssml: str = ""

    def emotion_timeline(self) -> list[dict]:
        """A compact per-segment emotion track for the renderer."""
        return [{"text": s.text, "emotion": s.emotion, "intensity": s.intensity,
                 "pace": s.pace} for s in self.segments]

    def to_dict(self) -> dict:
        return {"overall_tone": self.overall_tone, "ssml": self.ssml,
                "segments": [asdict(s) for s in self.segments]}


_SYSTEM = (
    "You are a performance director for a talking-head presenter video. "
    "You read a script and decide how it should be delivered. "
    "Return STRICT JSON only."
)


def _prompt(script: str) -> list[dict]:
    schema = {
        "overall_tone": "one short phrase",
        "segments": [
            {
                "text": "the sentence verbatim",
                "emotion": f"one of {EMOTIONS}",
                "intensity": "0.0-1.0",
                "emphasis": ["words", "to", "stress"],
                "pace": f"one of {PACES}",
                "pause_after_ms": "integer 0-1200",
            }
        ],
    }
    user = (
        "Split the script into natural spoken sentences. For each, choose emotion, "
        "intensity, up to 3 emphasis words, pace and a natural pause after it. "
        "Keep 'text' verbatim (do not rewrite).\n\n"
        f"Allowed emotions: {EMOTIONS}\nAllowed paces: {PACES}\n\n"
        f"Return JSON of shape: {schema}\n\nSCRIPT:\n{script.strip()}"
    )
    return [{"role": "system", "content": _SYSTEM}, {"role": "user", "content": user}]


def _split_sentences(text: str) -> list[str]:
    parts = re.split(r"(?<=[.!?])\s+", text.strip())
    return [p.strip() for p in parts if p.strip()]


def _ssml_escape(s: str) -> str:
    return (s.replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;"))


def _rate_for(pace: str) -> str:
    return {"slow": "90%", "normal": "100%", "fast": "112%"}.get(pace, "100%")


def build_ssml(segments: list[Segment]) -> str:
    """Amazon Polly-compatible SSML from the performance plan."""
    body = []
    for s in segments:
        text = _ssml_escape(s.text)
        for w in sorted(set(s.emphasis), key=len, reverse=True):
            if not w:
                continue
            pat = re.compile(rf"\b({re.escape(_ssml_escape(w))})\b")
            text = pat.sub(r'<emphasis level="strong">\1</emphasis>', text, count=1)
        body.append(f'<prosody rate="{_rate_for(s.pace)}">{text}</prosody>')
        if s.pause_after_ms > 0:
            body.append(f'<break time="{int(s.pause_after_ms)}ms"/>')
    return "<speak>" + " ".join(body) + "</speak>"


class Director:
    def __init__(self, llm: LLMClient | None = None):
        self.llm = llm or LLMClient()

    def direct(self, script: str) -> PerformancePlan:
        if self.llm.available:
            try:
                return self._direct_llm(script)
            except Exception:
                pass
        return self._direct_fallback(script)

    def _direct_llm(self, script: str) -> PerformancePlan:
        data = self.llm.chat_json(_prompt(script))
        raw_segments = data.get("segments") or []
        segments: list[Segment] = []
        for item in raw_segments:
            text = (item.get("text") or "").strip()
            if not text:
                continue
            emotion = item.get("emotion", "neutral")
            emotion = emotion if emotion in EMOTIONS else "neutral"
            pace = item.get("pace", "normal")
            pace = pace if pace in PACES else "normal"
            try:
                intensity = max(0.0, min(1.0, float(item.get("intensity", 0.5))))
            except (TypeError, ValueError):
                intensity = 0.5
            emphasis = [str(w) for w in (item.get("emphasis") or [])][:3]
            try:
                pause = int(item.get("pause_after_ms", 250))
            except (TypeError, ValueError):
                pause = 250
            pause = max(0, min(1200, pause))
            segments.append(Segment(text, emotion, intensity, emphasis, pace, pause))
        if not segments:
            return self._direct_fallback(script)
        plan = PerformancePlan(segments, data.get("overall_tone", "neutral"))
        plan.ssml = build_ssml(segments)
        return plan

    def _direct_fallback(self, script: str) -> PerformancePlan:
        segments = [Segment(text=s) for s in _split_sentences(script)]
        if not segments:
            segments = [Segment(text=script.strip() or " ")]
        plan = PerformancePlan(segments, "neutral")
        plan.ssml = build_ssml(segments)
        return plan
