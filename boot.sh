#!/usr/bin/env bash
# boot.sh — rebuild this pod's container layer from scratch.
#
# Only /workspace survives a pod restart; everything installed into the image
# layer (apt packages, site-packages, Claude Code) does not. Run this once after
# every fresh pod start to put the box back the way it was.
#
# Idempotent: safe to re-run at any time. Re-running reinstalls packages,
# restarts uvicorn, and leaves exactly one server on :8888.
#
#   bash /workspace/boot.sh
#
set -euo pipefail

REPO=/workspace/rio-phase1
PORT=8888
HF_HOME_DIR=/workspace/.cache/huggingface

log() { printf '\n\033[1;36m==> %s\033[0m\n' "$*"; }

# ---------------------------------------------------------------------------
# 1. HF_HOME — keep model weights on the persistent volume
# ---------------------------------------------------------------------------
# Without this, transformers caches into ~/.cache inside the container layer and
# re-downloads all 16GB of Qwen3-VL-8B on every pod start.
log "HF_HOME -> $HF_HOME_DIR"
export HF_HOME="$HF_HOME_DIR"
mkdir -p "$HF_HOME_DIR"

# Persist for future interactive shells. grep-guarded so re-runs don't stack
# duplicate lines into .bashrc.
if ! grep -q "export HF_HOME=$HF_HOME_DIR" ~/.bashrc 2>/dev/null; then
    echo "export HF_HOME=$HF_HOME_DIR" >> ~/.bashrc
    echo "   appended to ~/.bashrc"
else
    echo "   already in ~/.bashrc"
fi

# Claude Code installs to ~/.local/bin, which isn't on the default PATH.
if ! grep -q 'export PATH="$HOME/.local/bin:$PATH"' ~/.bashrc 2>/dev/null; then
    echo 'export PATH="$HOME/.local/bin:$PATH"' >> ~/.bashrc
fi
export PATH="$HOME/.local/bin:$PATH"

# ---------------------------------------------------------------------------
# 2. System packages
# ---------------------------------------------------------------------------
# ffmpeg: ElevenLabs audio muxing. nano: editing on the box. git: push/pull.
log "apt packages (ffmpeg nano git)"
export DEBIAN_FRONTEND=noninteractive
apt-get update -qq
apt-get install -y -qq --no-install-recommends ffmpeg nano git

# ---------------------------------------------------------------------------
# 3. Python dependencies
# ---------------------------------------------------------------------------
log "pip install -r requirements.txt"
pip install --no-cache-dir -r "$REPO/requirements.txt"

# ---------------------------------------------------------------------------
# 4. Torch cu128 — MUST come after requirements.txt
# ---------------------------------------------------------------------------
# requirements.txt pins torch==2.4.1 (plain PyPI, CPU/default-CUDA wheels). This
# pod is an L40S on driver 570 / CUDA 12.8, which that build does not target.
# Step 3 will happily drag torch back down to 2.4.1, so this force-reinstall runs
# afterwards and wins. --force-reinstall (not plain install) because pip
# considers 2.11.0 already-satisfied and would no-op on a re-run of this script
# after a partial/mixed install.
log "torch cu128 force-reinstall"
pip install --no-cache-dir --force-reinstall \
    torch==2.11.0+cu128 \
    torchvision==0.26.0+cu128 \
    torchaudio==2.11.0+cu128 \
    --index-url https://download.pytorch.org/whl/cu128

# The torch wheels are ~3GB; the cache is on the container layer but the disk
# pressure is real during install.
log "pip cache purge"
pip cache purge || true

# ---------------------------------------------------------------------------
# 5. GPU sanity
# ---------------------------------------------------------------------------
log "GPU sanity"
python - <<'PY'
import torch
print(f"  torch        : {torch.__version__}")
print(f"  cuda build   : {torch.version.cuda}")
print(f"  is_available : {torch.cuda.is_available()}")
if torch.cuda.is_available():
    for i in range(torch.cuda.device_count()):
        p = torch.cuda.get_device_properties(i)
        print(f"  device {i}     : {p.name} ({p.total_memory / 1024**3:.1f} GB)")
    # Actually touch the GPU — is_available() can be True while the runtime is
    # broken by a driver/wheel mismatch.
    x = torch.randn(1000, 1000, device="cuda")
    torch.cuda.synchronize()
    print(f"  matmul check : ok ({float((x @ x).sum()):.1f})")
else:
    raise SystemExit("  !! CUDA not available — check driver / wheel match")
PY

# ---------------------------------------------------------------------------
# 5b. UFLDv2 lane weights
# ---------------------------------------------------------------------------
# 825 MB, gitignored, fetched from the upstream model zoo and checksum-verified.
# Deliberately NOT fatal: without it headway falls back to the static trapezoid
# corridor and logs corridor_source=static, which is the pre-UFLDv2 behaviour.
# A pod that comes up with no lane model should still come up.
log "UFLDv2 lane weights"
python -m tools.fetch_lane_weights || log "  !! lane weights unavailable - headway will use the static corridor"

# ---------------------------------------------------------------------------
# 5c. RF-DETR detector (Apache-2.0) — the headway candidate source
# ---------------------------------------------------------------------------
# --no-deps on purpose: rfdetr's dependency list is a training stack, and
# `supervision` pulls opencv-python 5.x which shadows the contrib-headless
# build and removes CSRT/MOSSE. See the note in requirements.txt.
log "RF-DETR (--no-deps: its dep tree breaks the pinned cv2/transformers)"
pip install --no-cache-dir --no-deps rfdetr==1.5.0 supervision==0.29.1 pycocotools peft
# Both of the next two run `python -m` / import from the repo, so they need the
# repo as cwd. The script is documented as `bash /workspace/boot.sh`, which
# leaves cwd wherever the caller happened to be; the cd further down (step 7)
# was too late to help them.
cd "$REPO"

log "RF-DETR weights"
python -m tools.fetch_detector_weights || log "  !! detector weights unavailable - headway will have no candidates"

# Prove the two things that have actually broken here before, while the log is
# still being read, rather than discovering them mid-drive as an UNKNOWN band:
#   * the sideways rfdetr import (it was pinned to a hardcoded python3.12 path
#     and silently died when the pod was rebuilt on 3.11)
#   * CSRT, which disappears if anything drags opencv-python 5.x in on top of
#     the pinned contrib-headless build
log "detector + tracker smoke check"
python - <<'PY' || log "  !! SMOKE CHECK FAILED - headway will have no candidates"
import cv2
from headway import detect
detect._rfdetr_models()
print(f"   rfdetr importable from {detect._pkg_dir()}")
assert any(hasattr(cv2, n) for n in ("TrackerCSRT_create", "TrackerCSRT")) or hasattr(cv2, "legacy"), \
    f"cv2 {cv2.__version__} has no CSRT - check for a stray opencv-python install"
print(f"   cv2 {cv2.__version__} has the tracking API")
PY

# ---------------------------------------------------------------------------
# 6. Free port 8888
# ---------------------------------------------------------------------------
# RunPod starts JupyterLab on 8888, which is the port the proxy exposes and the
# port RIO needs. Jupyter has to go.
log "killing jupyter"
pkill -f jupyter || echo "   no jupyter running"

# Also clear any uvicorn from a previous run of this script, so a re-run doesn't
# leave two servers fighting over the port.
if pgrep -f "uvicorn app:app" >/dev/null; then
    echo "   stopping existing uvicorn"
    pkill -f "uvicorn app:app" || true
    sleep 2
fi

# ---------------------------------------------------------------------------
# 7. Launch RIO
# ---------------------------------------------------------------------------
log "launching uvicorn on :$PORT"
cd "$REPO"
nohup uvicorn app:app --host 0.0.0.0 --port "$PORT" > "$REPO/uvicorn.log" 2>&1 &
echo "   pid $!"

# ---------------------------------------------------------------------------
# 8. Health check
# ---------------------------------------------------------------------------
# uvicorn reports ready before the model is warm (vision warms on a daemon
# thread), so /health answers in a few seconds. 60s is generous headroom.
log "health check"
for i in $(seq 1 60); do
    if curl -sf "http://127.0.0.1:$PORT/health" > /dev/null 2>&1; then
        echo "   up after ${i}s: $(curl -s http://127.0.0.1:$PORT/health)"
        break
    fi
    if [ "$i" -eq 60 ]; then
        echo "   !! no response after 60s — tail of uvicorn.log:"
        tail -20 "$REPO/uvicorn.log"
        exit 1
    fi
    sleep 1
done

# Model warm runs in the background and takes ~40s more. Not fatal, just noted.
echo "   (Qwen3-VL warm continues in background — watch: tail -f $REPO/uvicorn.log)"

# ---------------------------------------------------------------------------
# 9. Claude Code
# ---------------------------------------------------------------------------
log "Claude Code"
if command -v claude > /dev/null 2>&1; then
    echo "   already installed: $(claude --version)"
else
    curl -fsSL https://claude.ai/install.sh | bash
    echo "   installed: $("$HOME/.local/bin/claude" --version)"
fi

log "boot complete — RIO on :$PORT"
