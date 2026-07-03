"""Tests for the brain layer that do NOT require GPU/LLM (offline, deterministic)."""

from bytical_talk.brain.autoconfig import RenderConfig, VideoStats, recommend
from bytical_talk.brain.director import Director, Segment, build_ssml
from bytical_talk.brain.llm import LLMClient, LLMConfig, _loads_lenient
from bytical_talk.brain.qc import QCMetrics, SelfQC, apply_suggestion


def test_ssml_emphasis_and_breaks():
    seg = Segment(text="We protect your family.", emphasis=["protect"],
                  pace="slow", pause_after_ms=400)
    ssml = build_ssml([seg])
    assert ssml.startswith("<speak>") and ssml.endswith("</speak>")
    assert '<emphasis level="strong">protect</emphasis>' in ssml
    assert 'rate="90%"' in ssml
    assert '<break time="400ms"/>' in ssml


def test_ssml_escapes_xml():
    ssml = build_ssml([Segment(text="A & B < C")])
    assert "&amp;" in ssml and "&lt;" in ssml


def test_director_fallback_without_llm():
    # Force no-LLM by using an empty config/provider with no creds.
    d = Director(llm=LLMClient(LLMConfig(provider="openai", openai_key=None)))
    plan = d.direct("Hello there. How are you today? I am well!")
    assert len(plan.segments) == 3
    assert plan.ssml.startswith("<speak>")


def test_loads_lenient_handles_fenced_json():
    assert _loads_lenient('```json\n{"a": 1}\n```') == {"a": 1}
    assert _loads_lenient('prose {"a": 2} more') == {"a": 2}
    assert _loads_lenient("not json") == {}


def test_autoconfig_recommend_rules():
    # high motion, small face, low contrast -> smoothing + wide feather + restore
    cfg = recommend(VideoStats(motion=8.0, face_frac=0.04, contrast=20))
    assert cfg.smooth and cfg.smooth_min_cutoff == 0.15
    assert cfg.feather == 14 and cfg.restore_face
    # stable, large, well-lit -> smoothing off
    cfg2 = recommend(VideoStats(motion=1.0, face_frac=0.3, contrast=60))
    assert not cfg2.smooth and cfg2.feather == 8


def test_qc_rules_flag_low_quality():
    qc = SelfQC(llm=LLMClient(LLMConfig(provider="openai", openai_key=None)))
    report = qc.review(QCMetrics(sharpness=10.0, temporal_jitter=5.0, n_frames=100),
                       RenderConfig(smooth=False))
    assert not report.passed
    assert report.suggestion.get("smooth") is True
    assert report.suggestion.get("restore_face") is True


def test_apply_suggestion_returns_new_config():
    base = RenderConfig(feather=8)
    new = apply_suggestion(base, {"feather": 14, "bogus": 1})
    assert new.feather == 14 and base.feather == 8  # immutably applied
