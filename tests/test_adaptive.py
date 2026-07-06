"""Tests for content-adaptive quality (B) — importance map + frame allocation.
No GPU/LLM needed."""

from bytical_talk.brain.director import PerformancePlan, Segment
from bytical_talk.render.adaptive import build_quality_plan, segment_frame_spans


def test_importance_ordering():
    neutral = Segment(text="Okay.", emotion="neutral", intensity=0.3)
    key = Segment(text="We protect your family.", emotion="confident",
                  intensity=0.9, emphasis=["protect", "family"])
    assert key.importance > neutral.importance
    assert 0.0 <= neutral.importance <= 1.0
    assert 0.0 <= key.importance <= 1.0


def test_neutral_low_emphatic_high():
    assert Segment(text="hi", emotion="neutral", intensity=0.2).importance < 0.4
    assert Segment(text="hi", emotion="excited", intensity=0.9,
                   emphasis=["now"]).importance > 0.8


def test_segment_frame_spans_cover_all_frames():
    segs = [Segment(text="a" * 10), Segment(text="b" * 30)]
    spans = segment_frame_spans(segs, 100)
    # contiguous, cover [0,100), longer text -> more frames
    assert spans[0][0] == 0 and spans[-1][1] == 100
    assert spans[1][1] - spans[1][0] > spans[0][1] - spans[0][0]
    # no gaps
    for (s0, e0, _), (s1, e1, _) in zip(spans, spans[1:]):
        assert e0 == s1


def test_quality_plan_flags_only_important_frames():
    plan = PerformancePlan(segments=[
        Segment(text="Filler words here.", emotion="neutral", intensity=0.2),
        Segment(text="This is the crucial point!", emotion="confident",
                intensity=0.95, emphasis=["crucial"]),
    ])
    qp = build_quality_plan(plan, n_frames=100, threshold=0.6)
    assert len(qp.importance) == 100 and len(qp.enhance) == 100
    # some but not all frames enhanced (only the important segment)
    assert 0.0 < qp.enhanced_fraction < 1.0
    # the last frames (important segment) should be enhanced
    assert qp.enhance[-1] is True
    assert qp.enhance[0] is False


def test_quality_plan_empty_safe():
    qp = build_quality_plan(PerformancePlan(segments=[]), n_frames=0)
    assert qp.importance == [] and qp.enhance == []
