"""HuBERT audio feature extractor for SyncTalk_2D  (Plan C — item C3).

Upstream SyncTalk_2D ships ONLY the AVE (Wav2Lip-mel) extractor; the `hubert`
mode is wired in the model/dataset but has no preprocessing script. This adds it.

Contract enforced by datasetsss_328.py + AudioConvHubert:
  * dataset loads `aud_hu.npy`, windows 16 rows via get_audio_features, then
    `reshape(32, 32, 32)` == 32768 == 16 * 2048.
  * => aud_hu.npy must be (N_frames, 2048), one row per 25 fps video frame.

HuBERT-large outputs 1024-d features at 50 Hz (2 timesteps per 25 fps frame), so
each video-frame row = 2 consecutive HuBERT timesteps concatenated (2*1024=2048).
Because the SAME function produces features for both training and inference, the
train/inference representation is identical (this is the alignment risk the plan
flagged — solved by construction).

Why HuBERT: self-supervised speech features carry far richer phonetic content than
a mel encoder and generalize better to unseen voices (e.g. Polly TTS), which is the
whole point for driving the model with synthesized audio.
"""

from __future__ import annotations

import argparse
import numpy as np
import torch

HUBERT_ID = "facebook/hubert-large-ls960-ft"  # 1024-d hidden, standard in this lineage
_HUBERT_DIM = 1024


def _load_wav(path: str, sr: int = 16000) -> np.ndarray:
    import librosa
    wav, _ = librosa.load(path, sr=sr, mono=True)
    return wav.astype(np.float32)


@torch.no_grad()
def extract_hubert_50hz(wav: np.ndarray, device, model=None, processor=None):
    """Return (T, 1024) HuBERT last-hidden-state at ~50 Hz plus the (cached) model
    and processor so callers can reuse them across many wavs."""
    from transformers import Wav2Vec2FeatureExtractor, HubertModel
    if processor is None:
        processor = Wav2Vec2FeatureExtractor.from_pretrained(HUBERT_ID)
    if model is None:
        model = HubertModel.from_pretrained(HUBERT_ID).to(device).eval()
    input_values = processor(wav, sampling_rate=16000,
                             return_tensors="pt").input_values.to(device)
    hidden = model(input_values).last_hidden_state[0]  # (T, 1024)
    return hidden.detach().cpu().numpy().astype(np.float32), model, processor


def to_perframe_2048(feat_1024: np.ndarray, num_frames: int | None = None) -> np.ndarray:
    """(T,1024) @50Hz -> (T//2, 2048) @25fps. Optionally pad/truncate to num_frames
    so the row count never exceeds the extracted image count (avoids dataset
    IndexError)."""
    assert feat_1024.shape[1] == _HUBERT_DIM, f"expected 1024-d, got {feat_1024.shape}"
    T = feat_1024.shape[0]
    if T % 2 == 1:                      # drop a dangling half-frame
        feat_1024 = feat_1024[:-1]
        T -= 1
    perframe = feat_1024.reshape(T // 2, 2 * _HUBERT_DIM)  # (N, 2048)
    if num_frames is not None:
        M = perframe.shape[0]
        if M >= num_frames:
            perframe = perframe[:num_frames]
        elif M > 0:                     # pad by repeating the last frame
            pad = np.repeat(perframe[-1:], num_frames - M, axis=0)
            perframe = np.concatenate([perframe, pad], axis=0)
    return perframe.astype(np.float32)


def extract_to_npy(wav_path: str, save_path: str | None = None,
                   num_frames: int | None = None, device=None) -> np.ndarray:
    device = device or ("cuda" if torch.cuda.is_available() else "cpu")
    wav = _load_wav(wav_path)
    feat, _, _ = extract_hubert_50hz(wav, device)
    perframe = to_perframe_2048(feat, num_frames)
    if save_path is None:
        save_path = wav_path.replace(".wav", "_hu.npy")
    np.save(save_path, perframe)
    print(f"[hubert] {wav_path} -> {save_path} shape={perframe.shape}")
    return perframe


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description="Extract HuBERT features -> aud_hu.npy")
    ap.add_argument("--wav_path", required=True)
    ap.add_argument("--save_path", default=None)
    ap.add_argument("--num_frames", type=int, default=None,
                    help="clamp/pad row count to the number of extracted frames")
    a = ap.parse_args()
    extract_to_npy(a.wav_path, a.save_path, a.num_frames)
