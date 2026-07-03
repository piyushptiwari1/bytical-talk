# Bytical Avatar — Master Research Roadmap

**Thesis.** Take **SyncTalk_2D** as the base (real-person cloning, fast CPU-bound
inference, per-identity training) and layer in open-source models, functions, and
techniques to evolve it into an advanced, controllable, self-improving talking-video
generator. Everything is additive and behind flags: the validated baseline is never
regressed, each capability is independently A/B-testable, and "conditional" features
(swaps) are strictly optional.

This document is the north star. Concrete, already-shipped work lives in the repo
(`improvements.py`, `losses_improved.py`, `train_328_improved.py`,
`inference_328_improved.py`, `data_utils/get_hubert.py`, `PLAN_C_ARCHITECTURE.md`).

---

## Where we are (baseline, shipped)

| Layer | Status |
|---|---|
| SyncTalk_2D packaged in portable Docker (`bytical/synctalk2d:latest`) | ✅ done |
| 7 safe robustness fixes (sys.executable, subprocess, ckpt-pick, etc.) | ✅ done |
| Inference quality (One-Euro box smoothing, feather paste, train/infer resize match) | ✅ done, A/B'd |
| Training upgrades opt-in (VGG-fixed, mouth-weighted L1, PatchGAN, LPIPS) | ✅ code + CPU-validated |
| **C3 — HuBERT audio encoder** (better voice generalization) | 🔄 in progress (this session) |

---

## Pillar 1 — Core quality & sync (the model itself)

Goal: sharper finishing, better lip-sync, higher clarity, temporally stable.

- **P1.1 Audio encoder → HuBERT/Wav2Vec2** *(C3, in progress)*. Richer phonetic
  features, better generalization to TTS/unseen voices. Repo already wires `--asr hubert`.
- **P1.2 Multi-scale audio injection (FiLM into decoder)** *(C1)*. Audio currently
  fuses only at the 5×5 bottleneck; feed it to `up1..up4` too. Identity-init → safe.
- **P1.3 Audio cross-attention / AdaIN fusion** *(C2)*. Content-adaptive phoneme→mouth
  alignment instead of channel-concat.
- **P1.4 Adversarial finishing**: PatchGAN (shipped opt-in) → optionally a
  **lightweight temporal discriminator** (3-frame) to remove flicker; consider
  **GFPGAN/CodeFormer** as an optional face-restoration post-pass for teeth/skin
  micro-detail (swap-safe, off by default).
- **P1.5 Higher resolution / super-resolution**: keep 328 core, add optional
  **Real-ESRGAN** upscaler on the pasted mouth region only (quality vs. time knob).
- **P1.6 Temporal consistency loss**: penalize frame-to-frame change in static
  regions during training → less shimmer without hurting mouth motion.
- **P1.7 Sync supervision**: retrain/att a HuBERT-compatible **SyncNet** so the sync
  loss works with the new audio encoder (current SyncNet is AVE-dim only).

**Metrics.** LSE-C / LSE-D (SyncNet confidence & distance), FID/LPIPS on mouth crop,
Laplacian sharpness, temporal-jitter (frame-diff on static ROI), wall-time/frame.

---

## Pillar 2 — Speed & scale (time / faster)

- **P2.1 ONNX / TensorRT export** of the U-Net (repo already has an ONNX `__main__`
  in `unet_328.py`) → 2–4× inference on the same GPU.
- **P2.2 Half precision (fp16/bf16)** inference; the model is tiny (~500 MB VRAM) so
  batch multiple frames per forward.
- **P2.3 Decouple CPU crop/paste from GPU**: current bottleneck is CPU (7–8 cores/job).
  Move crop/mask/paste to vectorized/GPU ops or a producer-consumer pipeline.
- **P2.4 Horizontal scale**: SQS + ASG scale-from-0 (already designed in infra) —
  train-once-per-presenter, serve many.

---

## Pillar 3 — Expressiveness (emotions, gestures, sentences)

- **P3.1 Emotion conditioning**: add an emotion embedding (from text sentiment or an
  explicit tag) as an extra conditioning vector alongside audio → expressive mouth/brow.
  Reference: EAT, EMO, PD-FGC.
- **P3.2 Gesture / head-motion**: SyncTalk_2D keeps the *source* body motion. To add
  controllable gestures, integrate a pose/gesture driver (e.g. **EchoMimicV2** pose
  templates already explored, or **DiffGesture/CyberHost**) as an optional pre-stage
  that regenerates the body track before lip-sync.
- **P3.3 Sentence/prosody awareness**: use the ASR/LLM layer to drive pacing, emphasis,
  and pauses in TTS (Polly SSML / other TTS) so lip motion matches meaning, not just
  phonemes.
- **P3.4 Blink/gaze**: optional procedural or learned eye motion for long clips.

---

## Pillar 4 — Self-learning & self-understanding

- **P4.1 Automatic quality scoring**: run LSE-C/LSE-D + face-restoration confidence per
  render; auto-flag low-quality outputs for re-render with different settings.
- **P4.2 Active data curation**: from a presenter's renders, mine the frames where sync
  confidence is lowest and prioritize them in fine-tuning (hard-example mining).
- **P4.3 Continual per-presenter fine-tuning**: start from a shared base checkpoint and
  adapt fast to each presenter (LoRA-style light adaptation) instead of full 2 h train.
- **P4.4 Self-supervised pretraining**: pretrain the U-Net across many identities so a
  new presenter needs minutes, not hours (moves toward few-shot cloning).

---

## Pillar 5 — Conditional swaps (OPTIONAL, off by default)

These are opt-in creative controls, isolated so they never affect the default path.
Each has clear consent/ethics gating (only on content the user owns/has rights to).

- **P5.1 Face swap**: **InsightFace / inswapper** or **SimSwap/FaceFusion** as a
  post/pre-stage. SyncTalk already crops by landmarks — a face-swap module can run on
  the source track before cloning. Gated, watermarked, consent-checked.
- **P5.2 Voice swap / conversion**: **RVC** or **so-vits-svc** to convert a base TTS/
  narration into a target voice, or **XTTS/OpenVoice** for cloning a provided voice
  sample. Feeds the same audio pipeline (HuBERT features downstream stay identical).
- **P5.3 Clothes swap**: virtual try-on (**IDM-VTON / OOTDiffusion**) on the presenter
  frames before lip-sync. Body region only; face untouched.
- **P5.4 Background swap**: matting (**RobustVideoMatting / BiRefNet**) → composite new
  background. Cheap, high-value, low-risk; likely the first swap to ship.

**Guardrails for Pillar 5.** All swaps require explicit user opt-in, operate only on
rights-owned media, are logged, and default OFF. No identity impersonation of real
non-consenting people.

---

## Sequencing (research order)

1. **P1.1 HuBERT (C3)** — in progress; biggest generalization win for TTS-driven video.
2. **P1.2 FiLM (C1)** — best quality/effort architecture change.
3. **P2.1 ONNX/TensorRT + P2.2 fp16** — speed, unblocks everything downstream.
4. **P5.4 Background swap + P1.4 face-restoration post-pass** — high-value, low-risk,
   user-visible polish.
5. **P1.7 HuBERT-SyncNet + P1.6 temporal loss** — lock in sync + stability gains.
6. **P3.1 emotion / P3.2 gesture** — expressiveness.
7. **P4.x self-learning** — few-shot cloning + auto-quality loops.
8. **P5.1–P5.3 remaining swaps** — optional creative controls, gated.

Each item ships as a new file / flag over SyncTalk_2D, CPU-validated then A/B'd on a
real checkpoint (the aarav clone is the standing test fixture), keeping the baseline
always runnable.
