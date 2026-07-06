# bytical-talk

**A smart-AI talking-head video system.** It pairs the fast [SyncTalk_2D](https://github.com/ZiqiaoPeng/SyncTalk_2D)
lip-sync **renderer** with an LLM/embedding-driven **brain** that understands the
script, adapts to any input video, and critiques its own output — so it *performs* a
script instead of just reading it.

> Research project. The renderer clones a real presenter from a short video; the
> brain is the layer that makes the system behave like "actual AI" rather than a
> fixed pipeline. Built to be extended (see the roadmap).

---

## Why this exists

Vanilla lip-sync models take `(video, audio) → video`. That's a renderer, not
intelligence. `bytical-talk` adds a reasoning layer on top:

| Brain module | Input → Output | What it means |
|---|---|---|
| **Director** | script → emotion/emphasis/pacing + SSML | reads *meaning* and decides delivery |
| **AutoConfig** | any video → render settings | no manual tuning; works on *any* clip |
| **SelfQC** | rendered video → pass/fail + fix | the system reviews itself and retries smarter |

The renderer stays weight-safe and swappable; the brain is provider-agnostic
(defaults to Azure OpenAI, works with any OpenAI-compatible endpoint).

---

## Architecture

```
 script ──Director──▶ performance plan (emotion timeline + SSML)
                           │
                     [your TTS: SSML ─▶ wav]           (pluggable: Polly / Azure TTS / …)
                           │
 input video ─AutoConfig─▶ render settings
                           │
                       render(video, wav, settings) ─▶ mp4     (SyncTalk_2D + improvements)
                           │
                       SelfQC ─▶ pass? ──no──▶ adjust settings, re-render (bounded)
                           │ yes
                           ▼
                       final mp4 + QC report
```

- **`bytical_talk/brain/`** — `llm.py`, `director.py`, `autoconfig.py`, `qc.py`
- **`bytical_talk/render/`** — render path: One-Euro crop smoothing, feather
  paste-back, train/inference resize parity (GPU-batched fast path in progress)
- **`upstream/synctalk2d/`** — the renderer, fetched by `scripts/fetch_upstream.sh`
  (not re-hosted)

---

## Install

```bash
git clone https://github.com/piyushptiwari1/bytical-talk.git
cd bytical-talk
pip install -e .            # brain only (light: openai, numpy, pyyaml)
pip install -e ".[render]"  # + renderer/CV deps (torch, cv2, librosa, …)

cp .env.example .env        # then fill in your keys
bash scripts/fetch_upstream.sh   # only needed for rendering
```

### Configure the brain

Edit `.env` (never committed). Default backend is Azure OpenAI:

```
BYTICAL_LLM_PROVIDER=azure
AZURE_OPENAI_ENDPOINT=https://<resource>.openai.azure.com/
AZURE_OPENAI_API_KEY=<key>
AZURE_DEPLOYMENT_NAME=gpt-4o-mini
AZURE_EMBEDDING_MODEL_NAME=text-embedding-3-small
```

Any OpenAI-compatible endpoint works with `BYTICAL_LLM_PROVIDER=openai`.

---

## Use

```bash
# verify credentials + upstream
bytical-talk env-check

# LLM performance plan (no GPU needed)
bytical-talk direct --script "We protect what matters most. Let's find your plan."

# analyze any video -> recommended render settings (needs [render])
bytical-talk autoconfig --video presenter.mp4

# quality review of a rendered clip
bytical-talk qc --video out.mp4

# full pipeline (needs a trained checkpoint + a wav)
bytical-talk generate --script "..." --checkpoint ckpt.pth \
  --dataset dataset/presenter --audio speech.wav --out out.mp4 --reference presenter.mp4
```

Python:

```python
from bytical_talk import Director, auto_config, SelfQC

plan = Director().direct("Hi, I'm here to help you choose the right cover.")
print(plan.ssml)                 # Polly-ready SSML with emphasis + pauses
print(plan.emotion_timeline())   # per-sentence emotion for the renderer
```

---

## Training a presenter (renderer)

Any short, front-facing talking clip works. Standard SyncTalk_2D flow (AVE audio):

```bash
python upstream/synctalk2d/data_utils/process.py dataset/<name>/<name>.mp4
# then train per the upstream instructions; render with bytical-talk
```

---

## Roadmap

Two focused directions (see `ROADMAP.md`):
- **A — Speed:** the render is CPU-bound, not GPU-bound; GPU frame-batching, GPU
  crop/paste, NVENC encode, ONNX/TensorRT.
- **B — Content-adaptive quality:** the brain reads the spoken content and allocates
  render quality where it matters instead of uniformly.

Every change is gated on **SyncNet LSE-C** (`scripts/score_syncnet.py`) so it can't
regress lip-sync.

---

## Credits & license

- Renderer: [ZiqiaoPeng/SyncTalk_2D](https://github.com/ZiqiaoPeng/SyncTalk_2D)
  (based on Ultralight-Digital-Human and SyncTalk) — fetched, not re-hosted.
- `bytical_talk/` (the brain + render path) is licensed **Apache-2.0**.
