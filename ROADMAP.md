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

Profiled on a T4 (~210-frame clip). Naive frame-skipping was tested and **rejected**
(it kept all the per-frame CPU work and hurt sync). The real breakdown, and what we
shipped in `bytical_talk/render/fast.py` (`render_video_fast`), all **SyncNet-neutral**:

| Lever | Effect |
|---|---|
| **A1 GPU frame-batching** (`batch=16`) | net was ~64% of the loop at batch 1; batching fills GPU occupancy |
| **fp16 autocast** | T4 tensor cores ~2x the net: **10.1 -> 5.1 ms/frame** |
| **A3 single-pass streamed encode** | removed the MJPG-temp + re-encode double pass; NVENC optional |

**Result:** end-to-end **12.9s -> 8.2s** on a short clip (~1.4x), **~2x at scale**
(per-frame loop 28.3 -> 17.1 ms), with SyncNet LSE-C unchanged (7.20 -> 7.22).

**A4/A5 ONNX/TensorRT — dropped.** Benchmarked: onnxruntime-CUDA/TensorRT need a
cuDNN9/TRT10 stack the torch-2.2 environment doesn't have (both fell back to CPU),
and **torch fp16 already captures the 2x net speedup** with zero new deps. Revisit
only inside a fresh CUDA 12.4+ container.

Remaining (diminishing): threaded I/O prefetch (read ~21%), GPU crop/paste (~15%).

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
