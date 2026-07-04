"""C1 — FiLM multi-scale audio injection into the U-Net decoder.

Baseline SyncTalk_2D fuses audio ONCE, at the 5x5 bottleneck. Every decoder level
then reconstructs the mouth from that single audio-conditioned feature. C1 also
modulates each decoder level (up1..up4) with the pooled audio embedding via FiLM
(Feature-wise Linear Modulation, Perez et al. 2018):

    feat -> feat * (1 + gamma) + beta      # gamma, beta predicted per-channel from audio

Design choices that make this safe:
  * Keeps the proven pretrained **AVE** audio encoder (HuBERT was tested & lost).
  * FiLM predictors are **zero-initialized**, so at the start of training gamma=0
    (scale 1) and beta=0 -> the model is byte-for-byte the baseline, then it learns
    to use audio at multiple scales.
  * The inference contract Model(6, mode)(img, audio) is unchanged; only new params
    are added, so a C1 checkpoint is loaded/run exactly like a stock one (after
    retraining -- C1 is not weight-compatible with a baseline checkpoint).

`FiLM` is a pure nn.Module (unit-testable). `build_c1_model` lazily imports the
upstream `Model`/building blocks so this file only needs torch (the [render] extra),
not the fetched upstream, to be importable.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

# Decoder-level output channels for the 328 U-Net (ch=[32,64,128,256,512]):
#   up1 -> ch[3]//2 = 128, up2 -> ch[2]//2 = 64, up3 -> ch[1]//2 = 32, up4 -> ch[0] = 32
_DECODER_CH = (128, 64, 32, 32)
_AUDIO_DIM = 512  # AVE/Hubert/Wenet audio_model all output ch[4]=512 at the bottleneck


class FiLM(nn.Module):
    """Per-channel feature-wise linear modulation from an audio vector.

    Zero-initialized -> identity at start (returns feat unchanged).
    """

    def __init__(self, audio_dim: int, feat_ch: int):
        super().__init__()
        self.to_scale = nn.Linear(audio_dim, feat_ch)
        self.to_shift = nn.Linear(audio_dim, feat_ch)
        for lin in (self.to_scale, self.to_shift):
            nn.init.zeros_(lin.weight)
            nn.init.zeros_(lin.bias)

    def forward(self, feat: torch.Tensor, audio_vec: torch.Tensor) -> torch.Tensor:
        gamma = self.to_scale(audio_vec)[:, :, None, None]
        beta = self.to_shift(audio_vec)[:, :, None, None]
        return feat * (1.0 + gamma) + beta


def build_c1_model(n_channels: int = 6, mode: str = "ave"):
    """Return a ModelC1 instance. Lazily imports the upstream Model so importing this
    module does not require the fetched upstream to be on sys.path."""
    from unet_328 import Model  # type: ignore

    class ModelC1(Model):
        def __init__(self, n_channels=6, mode="ave"):
            super().__init__(n_channels, mode)
            self.film1 = FiLM(_AUDIO_DIM, _DECODER_CH[0])
            self.film2 = FiLM(_AUDIO_DIM, _DECODER_CH[1])
            self.film3 = FiLM(_AUDIO_DIM, _DECODER_CH[2])
            self.film4 = FiLM(_AUDIO_DIM, _DECODER_CH[3])

        def forward(self, x, audio_feat):
            x1 = self.inc(x)
            x2 = self.down1(x1)
            x3 = self.down2(x2)
            x4 = self.down3(x3)
            x5 = self.down4(x4)
            x6 = self.down5(x5)
            audio_feat = self.audio_model(audio_feat)
            audio_vec = F.adaptive_avg_pool2d(audio_feat, 1).flatten(1)  # (B, 512)
            x6 = torch.cat([x6, audio_feat], dim=1)
            x6 = self.fuse_conv(x6)
            x = self.film1(self.up1(x6, x5), audio_vec)
            x = self.film2(self.up2(x, x4), audio_vec)
            x = self.film3(self.up3(x, x3), audio_vec)
            x = self.film4(self.up4(x, x2), audio_vec)
            x = self.up5(x, x1)
            return torch.sigmoid(self.outc(x))

    return ModelC1(n_channels, mode)
