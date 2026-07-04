"""Tests for the C1 FiLM module (no upstream / GPU needed)."""

import torch

from bytical_talk.arch.film import FiLM


def test_film_zero_init_is_identity():
    torch.manual_seed(0)
    film = FiLM(audio_dim=512, feat_ch=64).eval()
    feat = torch.randn(2, 64, 20, 20)
    audio = torch.randn(2, 512)
    out = film(feat, audio)
    # zero-init -> gamma=0, beta=0 -> output == input exactly
    assert torch.allclose(out, feat, atol=0.0)


def test_film_learns_nonidentity_after_perturb():
    film = FiLM(audio_dim=512, feat_ch=64).eval()
    # simulate a trained state: nonzero weights -> modulation changes the feature
    with torch.no_grad():
        film.to_scale.weight.normal_(0, 0.1)
        film.to_shift.bias.fill_(0.5)
    feat = torch.ones(1, 64, 4, 4)
    audio = torch.ones(1, 512)
    out = film(feat, audio)
    assert not torch.allclose(out, feat)          # it now modulates
    assert out.shape == feat.shape                 # shape preserved


def test_film_shapes_broadcast():
    film = FiLM(audio_dim=256, feat_ch=32)
    feat = torch.randn(3, 32, 10, 10)
    audio = torch.randn(3, 256)
    assert film(feat, audio).shape == (3, 32, 10, 10)
