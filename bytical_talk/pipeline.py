"""Pipeline — orchestrates the brain + renderer into one smart flow.

    script ──Director──▶ performance plan (emotion timeline + SSML)
                              │
                        (your TTS: SSML ─▶ wav)      ← pluggable, provider-specific
                              │
    input video ──AutoConfig──▶ render settings
                              │
                          render(video, wav, settings) ─▶ mp4
                              │
                         SelfQC ─▶ pass? ──no──▶ adjust settings, re-render (bounded)
                              │ yes
                              ▼
                          final mp4 + report

The renderer + QC metrics need a GPU box; the Director/QC-judgement run anywhere
with LLM credentials. TTS is intentionally pluggable (Polly / Azure TTS / etc.).
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from typing import Callable

from .brain.autoconfig import RenderConfig, auto_config
from .brain.director import Director, PerformancePlan
from .brain.qc import SelfQC, apply_suggestion, measure


@dataclass
class GenerationResult:
    output_path: str | None
    plan: PerformancePlan
    config: RenderConfig
    attempts: int = 1
    qc_passed: bool = False
    qc_reports: list = field(default_factory=list)


class Pipeline:
    def __init__(self, director: Director | None = None, qc: SelfQC | None = None,
                 max_retries: int = 1):
        self.director = director or Director()
        self.qc = qc or SelfQC()
        self.max_retries = max_retries

    # -- brain-only steps (run anywhere) ---------------------------------
    def direct(self, script: str) -> PerformancePlan:
        return self.director.direct(script)

    def configure(self, video_path: str) -> tuple:
        return auto_config(video_path)

    # -- full flow (renderer needs a GPU box) ----------------------------
    def generate(
        self,
        script: str,
        checkpoint: str,
        dataset_dir: str,
        out_path: str,
        reference_video: str | None = None,
        tts: Callable[[str, str], str] | None = None,
        audio_path: str | None = None,
        upstream_dir: str = "upstream/synctalk2d",
    ) -> GenerationResult:
        """End-to-end. Provide either `audio_path` (pre-synthesized wav) or a `tts`
        callable `tts(ssml_or_text, out_wav) -> wav_path`."""
        from .render.infer import render_video

        plan = self.direct(script)

        if audio_path is None:
            if tts is None:
                raise ValueError("Provide audio_path or a tts callable.")
            audio_path = tts(plan.ssml, out_path.replace(".mp4", ".wav"))

        # settings: analyze the reference video if given, else safe defaults
        if reference_video:
            _, cfg = self.configure(reference_video)
        else:
            cfg = RenderConfig(match_train=True)

        result = GenerationResult(output_path=None, plan=plan, config=cfg)
        attempt = 0
        while True:
            attempt += 1
            rendered = render_video(
                name=os.path.basename(dataset_dir), audio_path=audio_path,
                checkpoint=checkpoint, dataset_dir=dataset_dir, out_path=out_path,
                config=cfg, upstream_dir=upstream_dir,
            )
            report = self.qc.review(measure(rendered), cfg)
            result.qc_reports.append(report)
            result.output_path = rendered
            result.config = cfg
            result.attempts = attempt
            result.qc_passed = report.passed
            if report.passed or attempt > self.max_retries or not report.suggestion:
                break
            cfg = apply_suggestion(cfg, report.suggestion)  # smarter retry
        return result
