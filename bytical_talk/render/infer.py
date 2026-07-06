"""Renderer integration — run SyncTalk_2D inference with bytical-talk improvements.

This wraps the upstream model (fetched into upstream/synctalk2d/) and adds the
validated base quality fixes: temporal crop-box smoothing, feathered paste-back,
and train/inference resize parity, driven by a RenderConfig. Works with any trained
AVE checkpoint. The GPU-batched fast path lives in bytical_talk.render.fast.

Heavy deps (torch/cv2) are imported lazily; upstream is added to sys.path at call
time so the package imports fine without the renderer installed.
"""

from __future__ import annotations

import os
import subprocess
import sys

from ..brain.autoconfig import RenderConfig


def _add_upstream(upstream_dir: str) -> None:
    up = os.path.abspath(upstream_dir)
    if up not in sys.path:
        sys.path.insert(0, up)


def render_video(
    name: str,
    audio_path: str,
    checkpoint: str,
    dataset_dir: str,
    out_path: str,
    config: RenderConfig | None = None,
    upstream_dir: str = "upstream/synctalk2d",
    device: str | None = None,
) -> str:
    """Render a lip-synced video from a trained checkpoint + audio.

    name         : identity name (used only for logging)
    audio_path   : driving .wav
    checkpoint   : path to a trained U-Net .pth (AVE)
    dataset_dir  : dir with full_body_img/ + landmarks/ for this identity
    out_path     : output .mp4
    config       : RenderConfig (smoothing/feather/match_train); defaults are safe
    """
    import cv2
    import numpy as np
    import torch

    _add_upstream(upstream_dir)
    from unet_328 import Model  # type: ignore
    from utils import AudioEncoder, AudDataset, get_audio_features  # type: ignore
    from torch.utils.data import DataLoader

    from .smoothing import BoxSmoother, feather_paste

    cfg = config or RenderConfig()
    device = device or ("cuda" if torch.cuda.is_available() else "cpu")
    dev = torch.device(device)

    # ---- audio features (AVE encoder) ----
    enc = AudioEncoder().to(dev).eval()
    ck = torch.load(os.path.join(upstream_dir, "model/checkpoints/audio_visual_encoder.pth"),
                    map_location=dev)
    enc.load_state_dict({f"audio_encoder.{k}": v for k, v in ck.items()})
    loader = DataLoader(AudDataset(audio_path), batch_size=64, shuffle=False)
    outs = []
    for mel in loader:
        with torch.no_grad():
            outs.append(enc(mel.to(dev)))
    outs = torch.cat(outs, dim=0).cpu()
    first, last = outs[:1], outs[-1:]
    audio_feats = torch.cat([first.repeat(1, 1), outs, last.repeat(1, 1)], dim=0).numpy()

    # ---- frames ----
    img_dir = os.path.join(dataset_dir, "full_body_img/")
    lms_dir = os.path.join(dataset_dir, "landmarks/")
    len_img = len(os.listdir(img_dir)) - 1
    exm = cv2.imread(img_dir + "0.jpg")
    h, w = exm.shape[:2]
    tmp = out_path.replace(".mp4", "temp.mp4")
    writer = cv2.VideoWriter(tmp, cv2.VideoWriter_fourcc("M", "J", "P", "G"), 25, (w, h))
    fwd = cv2.INTER_AREA if cfg.match_train else cv2.INTER_CUBIC
    smoother = BoxSmoother(cfg.smooth_min_cutoff, cfg.smooth_beta) if cfg.smooth else None

    net = Model(6, "ave").to(dev)
    net.load_state_dict(torch.load(checkpoint, map_location=dev))
    net.eval()

    step, idx = 0, 0
    for i in range(audio_feats.shape[0]):
        if idx > len_img - 1:
            step = -1
        if idx < 1:
            step = 1
        idx += step
        img = cv2.imread(img_dir + str(idx) + ".jpg")
        with open(lms_dir + str(idx) + ".lms") as f:
            lms = np.array([np.array(line.split(" "), dtype=np.float32)
                            for line in f.read().splitlines()], dtype=np.int32)
        xmin, ymin, xmax = lms[1][0], lms[52][1], lms[31][0]
        if smoother is not None:
            xmin, ymin, xmax = smoother(xmin, ymin, xmax)
        width = xmax - xmin
        ymax = ymin + width
        crop = img[ymin:ymax, xmin:xmax]
        hc, wc = crop.shape[:2]
        crop = cv2.resize(crop, (328, 328), interpolation=fwd)
        crop_ori = crop.copy()
        real = crop[4:324, 4:324].copy()
        masked = cv2.rectangle(real.copy(), (5, 5, 310, 305), (0, 0, 0), -1)
        real_t = torch.from_numpy(real.transpose(2, 0, 1).astype(np.float32) / 255.0)
        masked_t = torch.from_numpy(masked.transpose(2, 0, 1).astype(np.float32) / 255.0)
        concat = torch.cat([real_t, masked_t], dim=0)[None].to(dev)

        af = get_audio_features(audio_feats, i).reshape(32, 16, 16)[None].to(dev)
        with torch.no_grad():
            pred = net(concat, af)[0]
        pred = np.array(pred.cpu().numpy().transpose(1, 2, 0) * 255, dtype=np.uint8)
        crop_ori[4:324, 4:324] = pred
        crop_ori = cv2.resize(crop_ori, (wc, hc), interpolation=cv2.INTER_CUBIC)
        if cfg.feather > 0:
            region = img[ymin:ymax, xmin:xmax]
            img[ymin:ymax, xmin:xmax] = feather_paste(region.copy(), crop_ori, 0, 0, cfg.feather)
        else:
            img[ymin:ymax, xmin:xmax] = crop_ori
        writer.write(img)
    writer.release()

    subprocess.run(["ffmpeg", "-y", "-i", tmp, "-i", audio_path, "-c:v", "libx264",
                    "-c:a", "aac", "-crf", "20", out_path], check=True)
    if os.path.exists(tmp):
        os.remove(tmp)
    return out_path
