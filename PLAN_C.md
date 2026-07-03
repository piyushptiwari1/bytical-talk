# Plan C — Architecture upgrades for SyncTalk_2D (proposed, staged)

These are the higher-ceiling, more invasive changes deliberately left out of the
opt-in training upgrades. Each one **breaks checkpoint compatibility** (adds/moves
parameters) and therefore requires a full retrain (~2 h/presenter on a T4). They do
NOT change the inference contract (still `Model(6, mode)(img_concat, audio_feat)`),
so `inference_328.py` / `inference_328_improved.py` keep working unchanged once a
new checkpoint is trained.

Baseline fact (measured from `unet_328.py`): audio is fused **exactly once**, at the
bottleneck —
```python
x6 = self.down5(x5)                 # (B, 512, 5, 5)
audio_feat = self.audio_model(audio_feat)
x6 = torch.cat([x6, audio_feat], 1) # (B, 1024, 5, 5)
x6 = self.fuse_conv(x6)             # -> (B, 256, 5, 5)
# up1..up5 decode with skip connections; audio never seen again
```
So every decoder level from `up1` onward reconstructs the mouth from a single
audio-conditioned bottleneck. That is the structural limit this plan attacks.

---

## C1 — Multi-scale audio injection (FiLM into decoder levels)

**Why.** The mouth is reconstructed by `up1..up5` at 10²→320² resolution, but audio
only informs the 5×5 bottleneck. Fine mouth/jaw detail at higher decoder resolutions
has no direct audio signal. Injecting audio at each decoder stage is the standard fix
(FiLM: Perez et al. 2018; used by DINet, IP-LAP, RAD-NeRF for exactly this).

**What.** Add a small `FiLM` module that maps the pooled audio embedding to per-channel
`(gamma, beta)` and modulates each `Up` block's output:
```python
class FiLM(nn.Module):
    def __init__(self, audio_dim, feat_ch):
        super().__init__()
        self.to_scale = nn.Linear(audio_dim, feat_ch)
        self.to_shift = nn.Linear(audio_dim, feat_ch)
    def forward(self, feat, a):           # feat: (B,C,H,W), a: (B,audio_dim)
        g = self.to_scale(a)[..., None, None]
        b = self.to_shift(a)[..., None, None]
        return feat * (1 + g) + b
```
- Pool `audio_feat` (after `self.audio_model`) with global avg pool → `(B,512)`.
- Add `self.film1..film4` (one per `up1..up4`), apply after each `Up`.
- `+1` on gamma makes it identity-at-init → training starts == baseline, then learns.

**Files.** `unet_328.py` only (new `FiLM`, 4 members, 4 forward lines). New file
`unet_328_c1.py` to keep the original pristine; `train_328_improved.py` gets a
`--arch c1` switch that imports it.

**Effort.** ~1 h code. **Params.** +~1.3 M (small). **Risk.** Low — identity init.
**Checkpoint.** New (retrain). **Validate.** shape-trace on CPU, then a 30-epoch aarav
retrain + A/B render vs baseline on the same audio.

**Expected win.** Sharper, more phoneme-accurate mouth interior; best single
architecture lever for this model class.

---

## C2 — Audio cross-attention / AdaIN modulation at the bottleneck

**Why.** `torch.cat + conv` fusion treats audio as extra channels; it cannot do
content-adaptive alignment between a phoneme and the mouth region. A lightweight
cross-attention (visual queries, audio keys/values) or AdaIN gives the model an
explicit "which mouth shape for this sound" mechanism (Transformer fusion is now
standard: TalkLip, SadTalker, EMO).

**What (pick one, A/B them).**
- **AdaIN (cheaper):** replace the first `fuse_conv` block with AdaIN where the audio
  embedding predicts the normalization scale/shift of the visual bottleneck. ~0.5 M
  params, very stable.
- **1-layer cross-attention (stronger):** flatten `x6` to 25 visual tokens (5×5),
  attend over audio tokens; single `nn.MultiheadAttention(embed_dim=512, heads=4)`
  + residual + LayerNorm. ~1.5 M params.

**Files.** `unet_328_c2.py` (subclass overriding only the fuse step). `--arch c2`.

**Effort.** ~1.5 h. **Risk.** Medium — attention can be unstable at high LR; use
`lr=5e-4`, warmup 2 epochs, residual+zero-init on the attention output projection so
it starts == baseline. **Checkpoint.** New. **Validate.** same 30-epoch aarav A/B.

**Expected win.** Better sync on fast/complex phonemes; complements C1 (can stack).

---

## C3 — Swap AVE audio encoder for HuBERT / Wav2Vec2 features

**Why.** AVE (Wav2Lip-style mel encoder, `utils.AudioEncoder`) is weaker than modern
self-supervised speech features. HuBERT/Wav2Vec2 embeddings carry richer phonetic
information and are the current default in top talking-head systems (GeneFace,
RAD-NeRF, real3d-portrait). The repo **already supports `--asr hubert`** end-to-end
(model path `AudioConvHubert`, dataset reshape `(32,32,32)`), so this is the
lowest-code-risk of the three — the work is in preprocessing, not the model.

**What.**
1. Preprocessing: add a HuBERT feature extractor to `data_utils/` producing
   `aud_hu.npy` (per-frame HuBERT features aligned to 25 fps), mirroring how
   `test_w2l_audio.py` produces `aud_ave.npy`. Use `transformers`
   `Wav2Vec2/HubertModel` (chinese-hubert-base or facebook/hubert-large-ls960).
2. Inference: add a HuBERT branch to `AudDataset`/feature loading in an
   `inference_328_improved.py` `--asr hubert` path (encode wav → per-frame feats →
   reshape `(32,32,32)`), replacing the AVE `AudioEncoder`.
3. Train with `--asr hubert` (model already builds `AudioConvHubert`).

**Files.** new `data_utils/get_hubert.py`; small feature-load branch in inference;
no `unet_328.py` change. **Extra dep.** `transformers` (already installed) + one
HuBERT weight download (~360 MB, one-time; cache in the image/EBS).

**Effort.** ~2–3 h (feature alignment + fps resample is the fiddly part).
**Risk.** Medium — HuBERT runs at 50 Hz, video at 25 fps → must resample/pool feats
to one vector/frame; off-by-one alignment hurts sync. **Checkpoint.** New (different
audio encoder + mode). **Validate.** compare sync on aarav vs the AVE baseline.

**Expected win.** Better generalization to unseen audio (accents, TTS voices) —
directly relevant since we drive with Polly, not the training speaker.

---

## Recommended sequencing

1. **C3 first** — lowest model risk (repo already wired), biggest generalization win
   for TTS-driven inference, and it's orthogonal to C1/C2.
2. **C1 next** — best quality-per-effort architecture change; identity-init makes it
   safe.
3. **C2 last** — stack on C1 only if A/B shows headroom; most tuning-sensitive.

Each lands as a separate `unet_328_cN.py` / `get_hubert.py` behind a
`train_328_improved.py --arch {baseline,c1,c2}` + `--asr {ave,hubert}` switch, so the
validated baseline is never overwritten and every step is independently A/B-testable
on the aarav checkpoint.

## Validation protocol (all C items)
1. CPU shape-trace: `Model(6, mode)(rand(2,6,320,320), rand(2,32,H,W))` returns
   `(2,3,320,320)`.
2. 30-epoch aarav retrain on the T4 (subset of full 100 for a fast signal).
3. A/B render the **same** Polly audio through baseline `99.pth` vs the new checkpoint
   with `inference_328_improved.py --smooth --feather 12 --match-train`.
4. Compare: LSE-C/LSE-D (SyncNet confidence), sharpness (Laplacian var) on the mouth
   crop, and eyeball 3 frames. Keep only if sync and/or sharpness improve without
   identity drift.
