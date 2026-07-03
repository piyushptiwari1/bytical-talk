#!/usr/bin/env bash
# Fetch the SyncTalk_2D renderer + its bundled weights into upstream/synctalk2d/.
# We do NOT re-host upstream code (it has no clear license); we fetch it on setup.
set -euo pipefail

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
DEST="$ROOT/upstream/synctalk2d"
REPO="https://github.com/ZiqiaoPeng/SyncTalk_2D.git"

echo "[fetch] cloning SyncTalk_2D (with LFS weights) into $DEST"
rm -rf "$DEST"
git clone --depth 1 "$REPO" "$DEST"

echo "[fetch] pulling LFS weights (audio encoder, PFLD landmarks, scrfd)"
( cd "$DEST" && git lfs pull || echo "[fetch] git lfs not available; fetch weights manually" )

echo "[fetch] done. Renderer at: $DEST"
