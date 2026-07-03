"""Opt-in, research-grade training losses for SyncTalk_2D.

IMPORTANT: these affect *training* only and therefore REQUIRE retraining from
scratch. They do not change the U-Net architecture or the checkpoint format, so a
model trained with them still loads/infers exactly like a stock SyncTalk_2D model.
Wire them in via train_328_improved.py (flags default to the stock behavior).

What is here and why (grounded in current talking-head / lip-sync research):

1. VGGPerceptualLossFixed
   The stock PerceptualLoss feeds cv2-BGR images in [0,1] straight into VGG19
   with no ImageNet normalization and no BGR->RGB swap. VGG was trained on
   RGB, ImageNet-normalized inputs, so its features are being read off-manifold.
   This version fixes the channel order + normalization and optionally uses the
   standard multi-layer perceptual loss (relu1_2 + relu2_2 + relu3_3), which is
   the Johnson et al. (2016) formulation used by virtually every modern
   image-synthesis model.

2. MouthWeightedL1
   Only the mouth/jaw actually moves, but the stock L1 weights every pixel of
   the 320x320 crop equally, so most of the gradient is spent on static skin.
   A center-lower weight map concentrates capacity where lip-sync quality is
   judged. (Region-weighted reconstruction is standard in Wav2Lip/DINet/IP-LAP.)

3. NLayerDiscriminator (PatchGAN) + adversarial loss
   Stock SyncTalk_2D has NO discriminator, which is why fine texture (teeth,
   inner-lip) can look soft. A 70x70 PatchGAN with a hinge/BCE adversarial loss
   is the established, low-cost way to sharpen this class of crop-inpaint model
   (Wav2Lip-GAN, pix2pix, DINet all use exactly this). Adds one optimizer.

4. lpips_loss (optional)
   LPIPS (Zhang et al., 2018) is a better perceptual metric than a single VGG
   layer. Used only if the `lpips` package is installed.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models


# ---------------------------------------------------------------------------
# 1. Fixed VGG perceptual loss (ImageNet-normalized, BGR->RGB, multi-layer)
# ---------------------------------------------------------------------------
class VGGPerceptualLossFixed(nn.Module):
    """VGG19 perceptual loss with correct preprocessing.

    inputs are expected as SyncTalk produces them: NCHW, range [0,1], BGR order.
    """

    # relu1_2=3, relu2_2=8, relu3_3=17 (indices into vgg19().features)
    _LAYERS = {"relu1_2": 3, "relu2_2": 8, "relu3_3": 17}

    def __init__(self, criterion=None, multi_layer: bool = True, bgr_input: bool = True):
        super().__init__()
        try:
            weights = models.VGG19_Weights.IMAGENET1K_V1
            vgg = models.vgg19(weights=weights).features
        except AttributeError:  # very old torchvision
            vgg = models.vgg19(pretrained=True).features
        self.multi_layer = multi_layer
        self.bgr_input = bgr_input
        self.criterion = criterion or nn.MSELoss()
        last = max(self._LAYERS.values()) if multi_layer else self._LAYERS["relu3_3"]
        self.slice = nn.Sequential(*[vgg[i] for i in range(last + 1)]).eval()
        for p in self.slice.parameters():
            p.requires_grad_(False)
        self.register_buffer("mean", torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1))
        self.register_buffer("std", torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1))
        self._taps = sorted(self._LAYERS.values()) if multi_layer else [self._LAYERS["relu3_3"]]

    def _prep(self, x: torch.Tensor) -> torch.Tensor:
        if self.bgr_input:
            x = x[:, [2, 1, 0], :, :]  # BGR -> RGB
        return (x - self.mean) / self.std

    def get_loss(self, fake: torch.Tensor, real: torch.Tensor) -> torch.Tensor:
        f, r = self._prep(fake), self._prep(real.detach())
        loss = fake.new_zeros(())
        taps = set(self._taps)
        for i, layer in enumerate(self.slice):
            f = layer(f)
            r = layer(r)
            if i in taps:
                loss = loss + self.criterion(f, r.detach())
        return loss


# ---------------------------------------------------------------------------
# 2. Mouth-region-weighted L1
# ---------------------------------------------------------------------------
class MouthWeightedL1(nn.Module):
    """Weighted L1 that emphasises the center-lower (mouth/jaw) region.

    `base` weight applies everywhere; the mouth box gets `base + peak`. The map is
    built lazily for the incoming spatial size and cached.
    """

    def __init__(self, base: float = 1.0, peak: float = 3.0,
                 cy: float = 0.62, cx: float = 0.5, ry: float = 0.30, rx: float = 0.34):
        super().__init__()
        self.base, self.peak = base, peak
        self.cy, self.cx, self.ry, self.rx = cy, cx, ry, rx
        self._cache = {}

    def _weights(self, h: int, w: int, device, dtype):
        key = (h, w, device, dtype)
        if key not in self._cache:
            ys = torch.linspace(0, 1, h, device=device, dtype=dtype).view(h, 1)
            xs = torch.linspace(0, 1, w, device=device, dtype=dtype).view(1, w)
            # smooth Gaussian-ish bump centered on the mouth
            d = ((ys - self.cy) / self.ry) ** 2 + ((xs - self.cx) / self.rx) ** 2
            bump = torch.exp(-d)
            wmap = (self.base + self.peak * bump).view(1, 1, h, w)
            self._cache[key] = wmap
        return self._cache[key]

    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        h, w = pred.shape[-2:]
        wmap = self._weights(h, w, pred.device, pred.dtype)
        diff = (pred - target).abs() * wmap
        return diff.sum() / (wmap.sum() * pred.shape[0] * pred.shape[1])


# ---------------------------------------------------------------------------
# 3. PatchGAN discriminator + adversarial loss
# ---------------------------------------------------------------------------
class NLayerDiscriminator(nn.Module):
    """70x70 PatchGAN (pix2pix / Wav2Lip-GAN style). Outputs a logit map."""

    def __init__(self, in_ch: int = 3, ndf: int = 64, n_layers: int = 3):
        super().__init__()
        layers = [nn.Conv2d(in_ch, ndf, 4, 2, 1), nn.LeakyReLU(0.2, True)]
        nf, nf_prev = ndf, ndf
        for n in range(1, n_layers):
            nf_prev, nf = nf, min(ndf * (2 ** n), 512)
            layers += [nn.Conv2d(nf_prev, nf, 4, 2, 1),
                       nn.InstanceNorm2d(nf), nn.LeakyReLU(0.2, True)]
        nf_prev, nf = nf, min(ndf * (2 ** n_layers), 512)
        layers += [nn.Conv2d(nf_prev, nf, 4, 1, 1),
                   nn.InstanceNorm2d(nf), nn.LeakyReLU(0.2, True)]
        layers += [nn.Conv2d(nf, 1, 4, 1, 1)]
        self.model = nn.Sequential(*layers)

    def forward(self, x):
        return self.model(x)


def d_hinge_loss(disc, real, fake):
    """Discriminator hinge loss. `fake` should be detached by the caller."""
    r = disc(real)
    f = disc(fake)
    return F.relu(1.0 - r).mean() + F.relu(1.0 + f).mean()


def g_hinge_loss(disc, fake):
    """Generator adversarial (hinge) loss."""
    return -disc(fake).mean()


# ---------------------------------------------------------------------------
# 4. Optional LPIPS
# ---------------------------------------------------------------------------
def make_lpips(net: str = "vgg"):
    """Return an LPIPS module or None if the package isn't installed.
    Expects inputs in [-1,1] RGB — caller must convert."""
    try:
        import lpips  # type: ignore
    except Exception:
        return None
    return lpips.LPIPS(net=net)
