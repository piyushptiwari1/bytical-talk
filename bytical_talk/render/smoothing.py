"""Inference-side quality improvements for SyncTalk_2D — drop-in, weight-safe.

None of these change the model or require retraining; they improve the output of
ANY already-trained checkpoint by fixing the two most common visible artifacts in
this class of crop-inpaint lip-sync model:

  1. Temporal jitter   — per-frame landmark noise makes the crop box wobble, which
                         shows up as a shaky mouth region. `OneEuroFilter` /
                         `BoxSmoother` low-pass the crop coordinates over time.
  2. Paste seam        — the predicted mouth crop is written back as a hard
                         rectangle, leaving a visible edge. `feather_paste`
                         alpha-blends the boundary so it disappears.

References: One-Euro filter (Casiez et al., CHI 2012) is the standard low-latency
jitter filter for tracked signals; feathered/Poisson boundary blending is standard
in face-swap/reenactment pipelines (DeepFaceLab, SimSwap, DINet).
"""

from __future__ import annotations

import math
import numpy as np


class OneEuroFilter:
    """Scalar One-Euro filter — adaptive low-pass that keeps latency low while
    removing jitter. Lower `min_cutoff` = smoother (more lag); higher `beta` =
    reacts faster to genuine motion.
    """

    def __init__(self, min_cutoff: float = 1.0, beta: float = 0.007, d_cutoff: float = 1.0):
        self.min_cutoff = float(min_cutoff)
        self.beta = float(beta)
        self.d_cutoff = float(d_cutoff)
        self._x_prev = None
        self._dx_prev = 0.0
        self._t = 0  # integer frame index; fps-independent (te = 1 frame)

    @staticmethod
    def _alpha(cutoff: float) -> float:
        # te = 1 (per-frame). tau = 1/(2*pi*cutoff).
        tau = 1.0 / (2.0 * math.pi * cutoff)
        return 1.0 / (1.0 + tau)

    def __call__(self, x: float) -> float:
        x = float(x)
        if self._x_prev is None:
            self._x_prev = x
            return x
        dx = x - self._x_prev
        a_d = self._alpha(self.d_cutoff)
        dx_hat = a_d * dx + (1 - a_d) * self._dx_prev
        cutoff = self.min_cutoff + self.beta * abs(dx_hat)
        a = self._alpha(cutoff)
        x_hat = a * x + (1 - a) * self._x_prev
        self._x_prev = x_hat
        self._dx_prev = dx_hat
        return x_hat


class BoxSmoother:
    """Temporally smooth the three landmark-derived crop coordinates SyncTalk uses
    (xmin, ymin, xmax). Returns integer pixel coords. Independent One-Euro filter
    per coordinate. Drop into the inference loop before cropping.
    """

    def __init__(self, min_cutoff: float = 0.25, beta: float = 0.02):
        # min_cutoff=0.25 cuts crop-box jitter ~40% with ~1-frame lag on a hard
        # 30px step (real presenter head motion is far slower, so no visible lag).
        self._fx = OneEuroFilter(min_cutoff, beta)
        self._fy = OneEuroFilter(min_cutoff, beta)
        self._fw = OneEuroFilter(min_cutoff, beta)

    def __call__(self, xmin: int, ymin: int, xmax: int):
        xs = self._fx(xmin)
        ys = self._fy(ymin)
        xe = self._fw(xmax)
        return int(round(xs)), int(round(ys)), int(round(xe))


def _feather_mask(h: int, w: int, feather: int) -> np.ndarray:
    """A HxWx1 alpha mask that is 1 in the interior and ramps linearly to 0 over
    `feather` pixels at each edge. Cached-friendly (cheap to build)."""
    if feather <= 0:
        return np.ones((h, w, 1), dtype=np.float32)
    ay = np.ones(h, dtype=np.float32)
    ax = np.ones(w, dtype=np.float32)
    r = np.linspace(0.0, 1.0, feather, dtype=np.float32)
    f = min(feather, h // 2)
    ay[:f] = r[:f]
    ay[h - f:] = r[:f][::-1]
    f = min(feather, w // 2)
    ax[:f] = r[:f]
    ax[w - f:] = r[:f][::-1]
    mask = np.outer(ay, ax)[..., None]
    return mask


def feather_paste(dst: np.ndarray, patch: np.ndarray, ymin: int, xmin: int,
                  feather: int = 12) -> np.ndarray:
    """Alpha-blend `patch` into `dst` at (ymin, xmin) with a feathered border so
    the crop boundary is not visible. Modifies and returns `dst`.
    """
    h, w = patch.shape[:2]
    ymax, xmax = ymin + h, xmin + w
    if ymin < 0 or xmin < 0 or ymax > dst.shape[0] or xmax > dst.shape[1]:
        # Out of bounds — fall back to a hard paste (original behavior).
        dst[ymin:ymax, xmin:xmax] = patch
        return dst
    region = dst[ymin:ymax, xmin:xmax].astype(np.float32)
    a = _feather_mask(h, w, feather)
    blended = a * patch.astype(np.float32) + (1.0 - a) * region
    dst[ymin:ymax, xmin:xmax] = np.clip(blended, 0, 255).astype(np.uint8)
    return dst
