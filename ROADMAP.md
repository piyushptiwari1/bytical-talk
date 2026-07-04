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
| **C3 — HuBERT audio encoder** (better voice generalization) | ✅ tested — **rejected** (see finding below) |

### Finding: HuBERT vs AVE audio encoder (2026-07)

We trained two 100-epoch models on one presenter, identical except the audio
encoder, and scored 10 renders of **unseen** TTS voices (US/GB/IN, M/F) with the
gold-standard **SyncNet LSE-C** metric. Result: the stock **AVE encoder won every
single voice** (mean LSE-C **8.03 vs 4.47**, offset ≈ 0 for both). The hypothesis
that HuBERT would generalize better to unseen voices was **refuted**.

*Why:* AVE (`audio_visual_encoder.pth`) is **pretrained jointly for this lip-sync
task** with sync supervision, so its features are already sync-discriminative. HuBERT
is a generic speech-SSL model whose features the U-Net must learn to map from scratch
in ~100 epochs on ~30 s of data — a losing trade in this low-data, per-identity
regime. HuBERT may still help under large-scale multi-identity pretraining (a
different regime). **AVE stays the default;** the `--asr hubert` path remains as
opt-in for future multi-identity work.

---

## Pillar 1 — Core quality & sync (the model itself)

Goal: sharper finishing, better lip-sync, higher clarity, temporally stable.

- **P1.1 Audio encoder** — ~~HuBERT/Wav2Vec2 (C3)~~ **tested & rejected** (AVE wins,
  see finding above). Keep pretrained **AVE** as default. Revisit HuBERT only with
  multi-identity pretraining.
- **P1.2 Multi-scale audio injection (FiLM into decoder)** *(C1 — now the top lever)*.
  Audio fuses only at the 5×5 bottleneck; feed it to `up1..up4` too. Identity-init →
  safe. Keeps the proven AVE encoder and adds capacity where it's missing.
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
- **P1.7 Sync supervision / eval**: the stock AVE **SyncNet** already works with the
  training sync loss. Separately, we now use external **SyncNet LSE-C/LSE-D**
  (`scripts/score_syncnet.py`) as a model-independent evaluation metric for any A/B.

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

0. ~~**P1.1 HuBERT (C3)**~~ — ✅ done & **rejected** (AVE wins on SyncNet LSE-C).
1. **P1.2 FiLM (C1)** — now the top lever; best quality/effort architecture change,
   keeps the proven AVE encoder.
2. **P2.1 ONNX/TensorRT + P2.2 fp16** — speed, unblocks everything downstream.
3. **P5.4 Background swap + P1.4 face-restoration post-pass** — high-value, low-risk,
   user-visible polish.
4. **P1.6 temporal-consistency loss** — lock in stability (less shimmer).
5. **P3.1 emotion / P3.2 gesture** — expressiveness.
6. **P4.x self-learning** — few-shot cloning + auto-quality loops.
7. **P5.1–P5.3 remaining swaps** — optional creative controls, gated.

Each item ships as a new file / flag over SyncTalk_2D, CPU-validated then A/B'd on a
real checkpoint with **SyncNet LSE-C** (any short clip is a valid test fixture),
keeping the baseline
always runnable.
