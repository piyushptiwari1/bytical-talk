"""Tests for the relative-budget mode of the content-adaptive quality plan."""

from bytical_talk.brain.director import PerformancePlan, Segment
from bytical_talk.render.adaptive import build_quality_plan


def test_quality_plan_budget_bounds_enhanced_fraction():
    # even when every segment is "important", a relative budget bounds the work
    plan = PerformancePlan(segments=[
        Segment(text="a" * 20, emotion="confident", intensity=0.9, emphasis=["x"]),
        Segment(text="b" * 20, emotion="excited", intensity=0.95, emphasis=["y"]),
    ])
    qp = build_quality_plan(plan, n_frames=100, budget=0.3)
    assert 0.25 <= qp.enhanced_fraction <= 0.35   # ~30%, not 100%


def test_quality_plan_budget_zero_and_full():
    plan = PerformancePlan(segments=[Segment(text="hello there", intensity=0.9)])
    assert build_quality_plan(plan, 50, budget=0.0).enhanced_fraction == 0.0
    assert build_quality_plan(plan, 50, budget=1.0).enhanced_fraction == 1.0
