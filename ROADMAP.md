# bytical-talk — Roadmap

**Thesis.** SyncTalk_2D is the renderer. The value we add is making it *spend compute
intelligently*, in two directions:

- **A — Speed.** The render is **CPU-bound**, not GPU-bound. Measured on a T4: the
  U-Net uses ~500 MB VRAM and a fraction of the wall time; the wall time is dominated
  by per-frame **crop/paste (CPU)** and **ffmpeg encode**. So the speed work targets
  those, not the GPU.
- **B — Content-adaptive quality.** Read the spoken content and decide *where* to
  spend render quality, instead of treating every frame the same.

Both are validated with **SyncNet LSE-C** (model-independent lip-sync metric,
`scripts/score_syncnet.py`) so nothing ships that hurts sync.

---

## A — Speed (attack the measured bottleneck)

Baseline profile (T4, ~210-frame clip): ~10-11 s wall, of which the GPU forward is a
small slice. Fixed-stride frame-skipping was tested and **rejected**: skipping 50% of
GPU forwards cut wall time only ~13% (confirming the GPU isn't the bottleneck) and
hurt sync.

| Step | What | Attacks |
|---|---|---|
| A1 | **GPU frame-batching** - push N frames per U-Net forward | GPU utilisation / Python overhead |
| A2 | **GPU / vectorized crop + paste** - do the 328 crop, mask, resize, paste on-device | **CPU bottleneck** |
| A3 | **NVENC hardware encode** (h264_nvenc) + stream frames (no temp AVI) | **ffmpeg encode bottleneck** |
| A4 | **ONNX export -> onnxruntime-gpu** | net latency |
| A5 | **TensorRT** engine | net latency |

Acceptance gate for every step: **SyncNet LSE-C unchanged** vs. the reference render.

---

## B — Content-adaptive quality

The brain (bytical_talk.brain) reads the content and produces a per-segment
**importance / emphasis** map (Director) and analyzes the input video (AutoConfig).
Quality is then allocated by content rather than uniformly:

- High-importance / emphasized segments -> higher-effort rendering.
- Ordinary / low-energy segments -> standard (cheaper) rendering.

This is the "quality smart video generation based on the content provided to speak"
idea: the same compute budget, concentrated where it matters.

---

## Kept base (validated, low-risk)

- **Crop-box temporal smoothing** (One-Euro) - removes landmark jitter; auto-enabled by
  AutoConfig only when the input actually moves.
- **Feathered paste-back** - removes the crop seam.
- **Train/inference resize parity** (INTER_AREA) - fixes a real train/test mismatch.

## Metric

scripts/score_syncnet.py - SyncNet LSE-C (confidence, higher = better) + AV offset.
Any change is A/B'd against the reference render on a multi-voice set before shipping.
