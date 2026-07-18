"""Tests for the temporal EMA stabilizer + mouth-biased elliptical paste.

These are the inference-side fixes that removed the "fluid mouth" wobble and the
visible/moving paste square (no retraining required).
"""
import numpy as np

from bytical_talk.brain.autoconfig import RenderConfig
from bytical_talk.render import EMAStabilizer, feather_paste_ellipse


def test_ema_stabilizer_blends_toward_previous():
    s = EMAStabilizer(strength=0.5)
    first = np.zeros((8, 8, 3), dtype=np.uint8)
    second = np.full((8, 8, 3), 100, dtype=np.uint8)
    assert s(first).mean() == 0.0            # first frame passes through
    out = s(second)                          # blended 50/50 with the previous
    assert 40 <= out.mean() <= 60


def test_ema_stabilizer_off_is_identity():
    s = EMAStabilizer(strength=0.0)
    patch = np.full((8, 8, 3), 123, dtype=np.uint8)
    assert np.array_equal(s(patch), patch)


def test_ellipse_paste_center_stronger_than_corner():
    dst = np.zeros((40, 40, 3), dtype=np.uint8)
    patch = np.full((40, 40, 3), 200, dtype=np.uint8)
    out = feather_paste_ellipse(dst.copy(), patch, 0, 0)
    center = out[26, 20].mean()              # mouth-biased center (cy=0.66)
    corner = out[0, 0].mean()
    assert center > corner                   # oval: center keeps patch, corner keeps real
    assert corner < 40                       # corners stay (near) the real pixels


def test_ellipse_paste_full_frame_ok():
    dst = np.zeros((20, 20, 3), dtype=np.uint8)
    patch = np.full((20, 20, 3), 255, dtype=np.uint8)
    out = feather_paste_ellipse(dst.copy(), patch, 0, 0)
    assert out.shape == dst.shape
    assert out[13, 10].mean() > out[0, 0].mean()  # mouth-biased center brighter


def test_render_config_defaults_enable_fixes():
    cfg = RenderConfig()
    assert cfg.temporal == 0.0
    assert cfg.poisson is True
