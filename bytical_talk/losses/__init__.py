"""Opt-in research-grade training losses (VGG-fixed, mouth-weighted L1, PatchGAN, LPIPS)."""

from .improved import (  # noqa: F401
    MouthWeightedL1,
    NLayerDiscriminator,
    VGGPerceptualLossFixed,
    d_hinge_loss,
    g_hinge_loss,
    make_lpips,
)
