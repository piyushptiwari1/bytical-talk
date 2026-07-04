#!/usr/bin/env python3
"""Score one video's lip-sync with SyncNet (LSE-C confidence + AV offset).

Two stages, mirroring the standard syncnet_python evaluation used by Wav2Lip:
  1. run_pipeline.py  -> s3fd face detect/track/crop to 224x224 face .avi
  2. SyncNetInstance.evaluate -> AV offset, confidence (LSE-C)

Prints one line per detected face track:
    SYNC <ref> crop=<file> LSE-C=<conf> offset=<frames>

LSE-C (confidence) higher = better sync; offset should be near 0. This metric is
model-independent (runs on generated pixels + audio), so it fairly compares the
AVE and HuBERT variants. Place this file inside the syncnet_python repo dir.
"""
import argparse
import glob
import os
import subprocess
import sys

import numpy as np
from SyncNetInstance import SyncNetInstance


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--videofile", required=True)
    ap.add_argument("--reference", required=True)
    ap.add_argument("--data_dir", default="/tmp/syncwork")
    ap.add_argument("--initial_model", default="data/syncnet_v2.model")
    a = ap.parse_args()

    # stage 1: face detect / track / crop
    subprocess.run(
        [sys.executable, "run_pipeline.py", "--videofile", a.videofile,
         "--reference", a.reference, "--data_dir", a.data_dir, "--overwrite"],
        check=True, stdout=subprocess.DEVNULL,
    )

    # stage 2: SyncNet evaluate
    class Opt:
        pass

    opt = Opt()
    opt.tmp_dir = os.path.join(a.data_dir, "pytmp")
    opt.reference = a.reference
    opt.batch_size = 20
    opt.vshift = 15

    s = SyncNetInstance()
    s.loadParameters(a.initial_model)

    crops = sorted(glob.glob(os.path.join(a.data_dir, "pycrop", a.reference, "0*.avi")))
    if not crops:
        print(f"SYNC {a.reference} NO_FACE_TRACK")
        return

    for c in crops:
        offset, conf, dist = s.evaluate(opt, videofile=c)
        print(f"SYNC {a.reference} crop={os.path.basename(c)} "
              f"LSE-C={float(conf):.4f} offset={int(offset)}", flush=True)


if __name__ == "__main__":
    main()
