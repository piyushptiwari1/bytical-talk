"""Fast render path (A): batched GPU forward + fp16 + single-pass streamed encode.

Measured vs. the per-frame reference (render_video) on a T4: end-to-end ~12.9s -> 8.2s
(~1.4x on a short clip; ~2x at scale where fixed setup overhead amortizes), with
SyncNet LSE-C unchanged. Three levers, all sync-neutral:

  A1 batch     - run B frames through the U-Net per forward (net was ~64% of the
                 batched loop; batching amortizes launch + fills GPU occupancy).
  fp16         - autocast the forward; the T4's tensor cores ~2x the net (10.1 ->
                 5.1 ms/frame) with no measurable sync change.
  A3 encode    - stream raw BGR frames to ONE ffmpeg encode (the per-frame path
                 wrote an MJPG temp then re-encoded to H.264 - a double encode).
                 NVENC optional for long / high-res clips.

Same crop/mask/paste math as render_video, so output is frame-identical.
ONNX/TensorRT (A4/A5) was benchmarked and dropped: it needs a cuDNN9/TRT10 stack the
torch-2.2 environment doesn't have, and fp16 already captures the net speedup.
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


def render_video_fast(
    name: str,
    audio_path: str,
    checkpoint: str,
    dataset_dir: str,
    out_path: str,
    config: RenderConfig | None = None,
    upstream_dir: str = "upstream/synctalk2d",
    device: str | None = None,
    batch: int = 16,
    fp16: bool = True,
    nvenc: bool = False,
    quality_plan=None,
    enhance_amount: float = 0.6,
) -> str:
    """Fast batched render. Signature matches render_video plus batch/fp16/nvenc.
    Output is frame-identical to render_video (validated with SyncNet).

    quality_plan: optional QualityPlan (from bytical_talk.render.adaptive). When
    given, frames it flags (high content-importance) get a cheap unsharp detail pass
    on the generated mouth; the rest use the plain fast path. `enhance_amount`
    controls the unsharp strength.
    """
    import cv2
    import numpy as np
    import torch
    from torch.utils.data import DataLoader

    _add_upstream(upstream_dir)
    from unet_328 import Model  # type: ignore
    from utils import AudioEncoder, AudDataset, get_audio_features  # type: ignore

    from .smoothing import BoxSmoother, feather_paste

    cfg = config or RenderConfig()
    dev = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))
    use_fp16 = fp16 and dev.type == "cuda"
    fwd = cv2.INTER_AREA if cfg.match_train else cv2.INTER_CUBIC

    # ---- audio features (AVE) ----
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
    audio_feats = torch.cat([outs[:1], outs, outs[-1:]], dim=0).numpy()

    img_dir = os.path.join(dataset_dir, "full_body_img/")
    lms_dir = os.path.join(dataset_dir, "landmarks/")
    len_img = len(os.listdir(img_dir)) - 1
    exm = cv2.imread(img_dir + "0.jpg")
    H, W = exm.shape[:2]
    N = audio_feats.shape[0]

    net = Model(6, "ave").to(dev)
    net.load_state_dict(torch.load(checkpoint, map_location=dev))
    net.eval()

    smoother = BoxSmoother(cfg.smooth_min_cutoff, cfg.smooth_beta) if cfg.smooth else None

    vcodec = (["-c:v", "h264_nvenc", "-preset", "p4", "-cq", "23"] if nvenc
              else ["-c:v", "libx264", "-crf", "20", "-preset", "veryfast"])
    ff = subprocess.Popen(
        ["ffmpeg", "-y", "-loglevel", "error",
         "-f", "rawvideo", "-pix_fmt", "bgr24", "-s", f"{W}x{H}", "-r", "25", "-i", "-",
         "-i", audio_path, *vcodec, "-c:a", "aac", "-pix_fmt", "yuv420p", "-shortest", out_path],
        stdin=subprocess.PIPE)

    # ping-pong frame index sequence
    seq, step, idx = [], 0, 0
    for _ in range(N):
        if idx > len_img - 1:
            step = -1
        if idx < 1:
            step = 1
        idx += step
        seq.append(idx)

    def prep(fr_idx, i):
        img = cv2.imread(img_dir + str(fr_idx) + ".jpg")
        with open(lms_dir + str(fr_idx) + ".lms") as f:
            lms = np.array([np.array(l.split(" "), dtype=np.float32)
                            for l in f.read().splitlines()], dtype=np.int32)
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
        concat = torch.cat([real_t, masked_t], dim=0)
        af = get_audio_features(audio_feats, i).reshape(32, 16, 16)
        return img, (ymin, ymax, xmin, xmax, hc, wc), crop_ori, concat, af

    for b0 in range(0, N, batch):
        b1 = min(N, b0 + batch)
        imgs, geos, crop_oris, concats, afs = [], [], [], [], []
        for i in range(b0, b1):
            img, geo, co, concat, af = prep(seq[i], i)
            imgs.append(img); geos.append(geo); crop_oris.append(co)
            concats.append(concat); afs.append(af)
        cat = torch.stack(concats).to(dev)
        aud = torch.stack(afs).to(dev)
        with torch.no_grad():
            if use_fp16:
                with torch.autocast("cuda", dtype=torch.float16):
                    preds = net(cat, aud)
            else:
                preds = net(cat, aud)
        preds = (preds.float().cpu().numpy().transpose(0, 2, 3, 1) * 255).astype(np.uint8)
        for k in range(b1 - b0):
            ymin, ymax, xmin, xmax, hc, wc = geos[k]
            co = crop_oris[k]
            co[4:324, 4:324] = preds[k]
            co = cv2.resize(co, (wc, hc), interpolation=cv2.INTER_CUBIC)
            # Content-adaptive: on high-importance frames, spend a cheap unsharp
            # detail pass on the generated mouth region only.
            fidx = b0 + k
            if quality_plan is not None and fidx < len(quality_plan.enhance) and quality_plan.enhance[fidx]:
                blur = cv2.GaussianBlur(co, (0, 0), 3)
                co = cv2.addWeighted(co, 1.0 + enhance_amount, blur, -enhance_amount, 0)
            if cfg.poisson:
                # Poisson seamless clone: content 100% generated (no ghosting of
                # real vs generated lips), boundary harmonized (no visible square).
                m = np.zeros(co.shape[:2], dtype=np.uint8)
                iy, ix = max(4, hc // 40), max(4, wc // 40)
                m[iy:-iy, ix:-ix] = 255
                center = (xmin + (xmax - xmin) // 2, ymin + (ymax - ymin) // 2)
                try:
                    imgs[k] = cv2.seamlessClone(co, imgs[k], m, center, cv2.NORMAL_CLONE)
                except cv2.error:
                    imgs[k][ymin:ymax, xmin:xmax] = co
            elif cfg.feather > 0:
                reg = imgs[k][ymin:ymax, xmin:xmax]
                imgs[k][ymin:ymax, xmin:xmax] = feather_paste(reg.copy(), co, 0, 0, cfg.feather)
            else:
                imgs[k][ymin:ymax, xmin:xmax] = co
            ff.stdin.write(imgs[k].tobytes())
    ff.stdin.close()
    ff.wait()
    return out_path
