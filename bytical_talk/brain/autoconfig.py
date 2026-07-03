"""Auto-config — analyze any input video and choose optimal render settings.

The renderer has knobs (temporal smoothing strength, feather width, resize mode,
whether a face-restoration pass is worthwhile). Manually tuning them per video does
not scale. Auto-config measures the input and picks them automatically, which is
what lets the system accept "any video" rather than a hand-tuned fixture.

Core measurements are deterministic CV (robust, cheap). An optional LLM pass can
narrate *why* a config was chosen (useful for logs / UX), but never overrides the
measured decision.

Heavy deps (cv2/numpy) are imported lazily so this module imports fine anywhere.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field


@dataclass
class VideoStats:
    width: int = 0
    height: int = 0
    fps: float = 0.0
    n_frames: int = 0
    face_frac: float = 0.0        # face bbox area / frame area (0 if undetected)
    motion: float = 0.0           # mean abs inter-frame diff in the face region (0..255)
    brightness: float = 0.0       # mean luma 0..255
    contrast: float = 0.0         # std luma


@dataclass
class RenderConfig:
    smooth: bool = False
    smooth_min_cutoff: float = 0.25
    smooth_beta: float = 0.02
    feather: int = 8
    match_train: bool = True
    restore_face: bool = False    # hook for the optional GFPGAN/CodeFormer pass
    notes: list[str] = field(default_factory=list)

    def to_dict(self) -> dict:
        return asdict(self)


def analyze_video(path: str, max_frames: int = 120) -> VideoStats:
    import cv2
    import numpy as np

    cap = cv2.VideoCapture(path)
    if not cap.isOpened():
        raise FileNotFoundError(f"cannot open video: {path}")
    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = float(cap.get(cv2.CAP_PROP_FPS) or 0.0)
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)

    cascade = cv2.CascadeClassifier(
        cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
    )

    prev_face_gray = None
    face_fracs, motions, brights, contrasts = [], [], [], []
    read = 0
    while read < max_frames:
        ok, frame = cap.read()
        if not ok:
            break
        read += 1
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        brights.append(float(gray.mean()))
        contrasts.append(float(gray.std()))
        faces = cascade.detectMultiScale(gray, 1.2, 5, minSize=(60, 60))
        if len(faces):
            fx, fy, fw, fh = max(faces, key=lambda b: b[2] * b[3])
            face_fracs.append((fw * fh) / float(w * h))
            fg = cv2.resize(gray[fy:fy + fh, fx:fx + fw], (128, 128))
            if prev_face_gray is not None:
                motions.append(float(np.abs(fg.astype(np.float32) - prev_face_gray).mean()))
            prev_face_gray = fg.astype(np.float32)
    cap.release()

    def _mean(x):
        return float(sum(x) / len(x)) if x else 0.0

    return VideoStats(
        width=w, height=h, fps=fps, n_frames=total,
        face_frac=_mean(face_fracs), motion=_mean(motions),
        brightness=_mean(brights), contrast=_mean(contrasts),
    )


def recommend(stats: VideoStats) -> RenderConfig:
    """Map measured stats to render settings using simple, explainable rules."""
    cfg = RenderConfig()
    cfg.match_train = True  # always: fixes the train/inference resize mismatch
    cfg.notes.append("match_train on (INTER_AREA parity with training)")

    # Motion → smoothing strength. More face motion/jitter => smoother box.
    if stats.motion >= 6.0:
        cfg.smooth = True
        cfg.smooth_min_cutoff = 0.15
        cfg.notes.append(f"high motion ({stats.motion:.1f}) -> strong smoothing")
    elif stats.motion >= 2.5:
        cfg.smooth = True
        cfg.smooth_min_cutoff = 0.25
        cfg.notes.append(f"moderate motion ({stats.motion:.1f}) -> smoothing on")
    else:
        cfg.smooth = False
        cfg.notes.append(f"low motion ({stats.motion:.1f}) -> smoothing off (already stable)")

    # Face size → feather + super-res hint. Small face => wider feather + restore.
    if stats.face_frac and stats.face_frac < 0.06:
        cfg.feather = 14
        cfg.restore_face = True
        cfg.notes.append(f"small face ({stats.face_frac:.3f}) -> wide feather + restore hint")
    elif stats.face_frac and stats.face_frac < 0.15:
        cfg.feather = 10
        cfg.notes.append(f"medium face ({stats.face_frac:.3f}) -> feather 10")
    else:
        cfg.feather = 8
        cfg.notes.append("large/near face -> feather 8")

    # Lighting → restoration hint (low contrast tends to produce mushy mouths).
    if stats.contrast and stats.contrast < 35:
        cfg.restore_face = True
        cfg.notes.append(f"low contrast ({stats.contrast:.0f}) -> restore hint")

    return cfg


def auto_config(path: str) -> tuple[VideoStats, RenderConfig]:
    stats = analyze_video(path)
    return stats, recommend(stats)
