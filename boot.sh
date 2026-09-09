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
# JUST BOUNCING THE SERVER? Do not reach for pkill -- it kills the shell that
# runs it, which is explained at length above rio_pids() below. Use:
#
#   bash /workspace/boot.sh restart     stop, start, wait for /health
#   bash /workspace/boot.sh stop
#   bash /workspace/boot.sh start
#   bash /workspace/boot.sh status      pids, listener, /health
#
# Those touch nothing but the server: no apt, no pip, no weights, no preflight.
#
# RUN IT SO IT CANNOT BE HUNG UP ON. The torch step downloads ~3 GB and the
# Qwen weights are 16 GB; if the terminal drops during either, SIGHUP kills this
# script somewhere in the middle and leaves a PARTIALLY provisioned pod. Use
# tmux, or:
#
#   nohup bash /workspace/boot.sh &
#
# That is not a hypothetical failure. On 2026-08-02 this pod came up with steps
# 1-4 complete and step 5c (RF-DETR) never run: the detector was missing for the
# whole day, headway had no candidate source, and nobody noticed because a
# missing detector is deliberately non-fatal. Reconstructing that took package
# mtimes and dpkg timestamps, because the only record of what this script did
# was scrollback in a terminal that had gone. Hence BOOT_LOG below.
#
# THERE ARE TWO COPIES OF THIS FILE, AND THEY DRIFT.
# This one, in the repo, is the source. /workspace/boot.sh is a COPY, and it is
# the path the header above, tools/preflight.py's --fix line and everyone's
# muscle memory all point at -- because only /workspace survives a pod restart.
#
# On 2026-08-26 that copy turned out to be from 2026-07-29: 160 lines against
# this file's 273, with no RF-DETR step, no detector-weight fetch, no boot log
# and no preflight. It is, precisely, the version whose missing step 5c caused
# the incident described in the paragraph above -- so following the documented
# command would have re-provisioned the pod back into that state, and the only
# symptom would have been a scene graph that stayed empty.
#
# So: after editing this file, copy it. Every time.
#
#   cp /workspace/rio-phase1/boot.sh /workspace/boot.sh
#
# Or run the repo copy directly, which is unambiguous and always current:
#
#   nohup bash /workspace/rio-phase1/boot.sh &
#
set -euo pipefail

REPO=/workspace/rio-phase1
PORT=8888
HF_HOME_DIR=/workspace/.cache/huggingface
ENV_FILE=/workspace/env.sh

# On the PERSISTENT volume, deliberately. A boot log inside the container layer
# is a boot log that disappears with the thing it was describing.
BOOT_LOG=/workspace/boot.log
exec > >(tee -a "$BOOT_LOG") 2>&1
printf '\n===== boot.sh %s (pid %s) =====\n' "$(date -Is)" "$$"

log() { printf '\n\033[1;36m==> %s\033[0m\n' "$*"; }

# ---------------------------------------------------------------------------
# The server: find it, stop it, start it -- and the subcommands that do only
# that, without re-provisioning the pod.
# ---------------------------------------------------------------------------
#
#   bash boot.sh restart     stop the server, start it again, wait for /health
#   bash boot.sh stop        stop it
#   bash boot.sh start       start it (stops one first if it is already up)
#   bash boot.sh status      what is running, and what /health says
#   bash boot.sh             the full provision, as always
#
# WHY THIS EXISTS: `pkill -f "uvicorn app:app"` KILLS THE SHELL THAT RUNS IT.
#
# -f matches the pattern against the whole command line of every process, and
# the shell running the pkill has that pattern on its own command line -- any
# `bash -c '...'`, which is what every agent harness and every ssh one-liner
# uses. So the shell matches itself, kills itself with SIGTERM (exit 144, which
# is 128+15), and dies BEFORE it gets to the line that starts the replacement.
# The pod is then left with nothing on :8888 and a caller that thinks it just
# restarted the server.
#
# It is not hypothetical and it is not rare: it happened twice in one session on
# 2026-09-09, and the second time was after knowing about the first. Checked on
# this pod while writing this, `pgrep -f 'uvicorn app:app'` returned two pids --
# the server, and the shell doing the asking.
#
# The fix is to stop matching on the command line alone. A shell is never the
# server: keep only the candidates whose /proc/<pid>/comm is the interpreter
# uvicorn actually runs as, and never the current shell.

# Every RUNNING uvicorn for this app, one pid per line, and nothing that merely
# mentions it.
rio_pids() {
    local pid comm
    for pid in $(pgrep -f 'uvicorn app:app' 2>/dev/null || true); do
        [ "$pid" = "$$" ] && continue
        comm=$(cat "/proc/$pid/comm" 2>/dev/null || true)
        case "$comm" in
            uvicorn|python|python3|python3.*) echo "$pid" ;;
        esac
    done
}

# The pids on one line, trimmed, or empty. Used for reporting only.
rio_pidline() { rio_pids | tr '\n' ' ' | sed 's/ *$//'; }

rio_stop() {
    local pids i
    pids=$(rio_pidline)
    if [ -z "$pids" ]; then
        echo "   no uvicorn running"
        return 0
    fi
    echo "   stopping uvicorn: $pids"
    # shellcheck disable=SC2086
    kill $pids 2>/dev/null || true
    # It has a lifespan handler and an open port; give it time to let go of
    # both rather than racing the next bind.
    for i in $(seq 1 20); do
        [ -z "$(rio_pids)" ] && break
        sleep 0.5
    done
    pids=$(rio_pidline)
    if [ -n "$pids" ]; then
        echo "   still up after 10s — SIGKILL: $pids"
        # shellcheck disable=SC2086
        kill -9 $pids 2>/dev/null || true
        sleep 1
    fi
}

rio_start() {
    cd "$REPO"
    # The log of the run you are restarting BECAUSE OF is the one thing you
    # need after a crash, and `>` erases it. One generation back is enough.
    [ -f "$REPO/uvicorn.log" ] && mv -f "$REPO/uvicorn.log" "$REPO/uvicorn.log.prev"
    # setsid + nohup + </dev/null: the server has to outlive the shell that
    # started it. A plain background job belongs to the caller's session, so an
    # agent's `bash -c` wrapper exiting takes the server with it -- the other
    # half of the failure this whole block is about.
    #
    # HF_HOME is passed EXPLICITLY rather than left to inheritance. This is the
    # line people copy out of the log and re-run by hand, and a uvicorn started
    # without HF_HOME re-downloads 16 GB of Qwen3-VL into a container layer that
    # is about to disappear.
    setsid env HF_HOME="$HF_HOME_DIR" nohup \
        uvicorn app:app --host 0.0.0.0 --port "$PORT" \
        > "$REPO/uvicorn.log" 2>&1 < /dev/null &
    sleep 1
    local pids
    pids=$(rio_pidline)
    if [ -z "$pids" ]; then
        echo "   !! uvicorn did not come up — tail of uvicorn.log:"
        tail -20 "$REPO/uvicorn.log"
        return 1
    fi
    echo "   started: $pids (HF_HOME=$HF_HOME_DIR)"
}

# uvicorn answers /health well before the model is warm (vision warms on a
# daemon thread), so this waits for the SERVER, not for readiness.
rio_wait_healthy() {
    local i
    for i in $(seq 1 60); do
        if curl -sf "http://127.0.0.1:$PORT/health" > /dev/null 2>&1; then
            echo "   up after ${i}s: $(curl -s "http://127.0.0.1:$PORT/health")"
            return 0
        fi
        if [ "$i" -eq 60 ]; then
            echo "   !! no response after 60s — tail of uvicorn.log:"
            tail -20 "$REPO/uvicorn.log"
            return 1
        fi
        sleep 1
    done
}

rio_status() {
    local pids
    pids=$(rio_pidline)
    if [ -z "$pids" ]; then
        echo "   uvicorn: not running"
    else
        echo "   uvicorn: $pids"
    fi
    echo "   :$PORT   $(ss -lntp 2>/dev/null | grep ":$PORT" || echo 'nothing listening')"
    echo "   health:  $(curl -s -m 5 "http://127.0.0.1:$PORT/health" 2>/dev/null || echo 'no answer')"
}

# Dispatch before the provisioning traps below are installed: none of these
# subcommands provisions anything, so none of them should abort saying the pod
# is half-built.
case "${1-}" in
    restart) log "restarting uvicorn on :$PORT"; rio_stop; rio_start
             log "health check"; rio_wait_healthy; exit $? ;;
    stop)    log "stopping uvicorn"; rio_stop; exit 0 ;;
    start)   log "starting uvicorn on :$PORT"; rio_stop; rio_start
             log "health check"; rio_wait_healthy; exit $? ;;
    status)  log "RIO on :$PORT"; rio_status; exit 0 ;;
    "")      ;;
    *)       echo "usage: bash boot.sh [restart|stop|start|status]" >&2; exit 2 ;;
esac

# `set -e` makes this script abort on the first failure, which is right -- but a
# silent abort is how a half-provisioned pod happens. Say where it stopped, in
# the persistent log, and say what to run to find out what is missing.
trap 'rc=$?; printf "\n!! boot.sh ABORTED at line %s (exit %s): %s\n" \
      "$LINENO" "$rc" "$BASH_COMMAND"; \
      printf "   The pod is PARTIALLY provisioned. What is missing:\n"; \
      printf "     cd %s && python -m tools.preflight\n" "$REPO"; \
      printf "   Full log: %s\n" "$BOOT_LOG"; exit $rc' ERR
trap 'printf "\n!! boot.sh was INTERRUPTED (signal). The pod is PARTIALLY\n"; \
      printf "   provisioned -- run: cd %s && python -m tools.preflight\n" "$REPO"; \
      exit 130' HUP INT TERM

# ---------------------------------------------------------------------------
# 1. HF_HOME — keep model weights on the persistent volume
# ---------------------------------------------------------------------------
# Without this, transformers caches into ~/.cache inside the container layer and
# re-downloads all 16GB of Qwen3-VL-8B on every pod start.
log "HF_HOME -> $HF_HOME_DIR"
export HF_HOME="$HF_HOME_DIR"
mkdir -p "$HF_HOME_DIR"

# The environment itself goes on the PERSISTENT volume, for the same reason the
# boot log does. ~/.bashrc is in the container layer: every line ever appended
# to it is gone on the next pod start, so "it is in .bashrc" was only ever true
# until the rebuild -- and preflight, run from any shell that had not been
# through boot.sh, reported HF_HOME unset.
#
# So the value lives HERE, in a file on /workspace, and ~/.bashrc gets one line
# that sources it. That line still has to be re-added after every rebuild (it is
# in the container layer too), but what it points at survives, and anything that
# needs the environment without a login shell -- a cron entry, an agent, a
# hand-restarted uvicorn -- can just `. /workspace/env.sh`.
log "env -> $ENV_FILE"
cat > "$ENV_FILE" <<EOF
# Generated by boot.sh step 1. On the persistent volume on purpose: ~/.bashrc
# does not survive a pod rebuild and this does. Source it from anything that
# needs RIO's environment without going through a login shell:
#
#   . $ENV_FILE
#
export HF_HOME=$HF_HOME_DIR
# Claude Code installs to ~/.local/bin, which isn't on the default PATH.
export PATH="\$HOME/.local/bin:\$PATH"
EOF
echo "   wrote $ENV_FILE"

# grep-guarded so re-runs don't stack duplicate lines into .bashrc.
if ! grep -qF ". $ENV_FILE" ~/.bashrc 2>/dev/null; then
    printf '\n# RIO environment (persistent volume -- see %s)\n' "$ENV_FILE" >> ~/.bashrc
    printf '[ -f %s ] && . %s\n' "$ENV_FILE" "$ENV_FILE" >> ~/.bashrc
    echo "   sourced from ~/.bashrc"
else
    echo "   already sourced from ~/.bashrc"
fi

# shellcheck source=/dev/null
. "$ENV_FILE"

# ---------------------------------------------------------------------------
# 2. System packages
# ---------------------------------------------------------------------------
# ffmpeg: ElevenLabs audio muxing. nano: editing on the box. git: push/pull.
#
# nodejs: two of this project's test suites are JavaScript, because two of the
# things worth testing are -- the arbiter and the route tracker both run in the
# browser, and both are written as pure modules precisely so `node` can drive
# them without a page. Without it, `node tools/nav_selftest.js` and
# `node tools/realtime_selftest.js` are 240-odd checks nobody can run, and the
# pod comes up able to verify only half of itself. Ubuntu's node is 12, which
# is ancient and entirely sufficient: those files are deliberately ES5/ES6 with
# no build step.
log "apt packages (ffmpeg nano git nodejs)"
export DEBIAN_FRONTEND=noninteractive
apt-get update -qq
apt-get install -y -qq --no-install-recommends ffmpeg nano git nodejs

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
#
# scipy is listed EXPLICITLY and must stay. --no-deps means we take on the job
# of supplying whatever the modules we actually import need, and scipy is the
# one thing in that set nothing else installs for us:
# `rfdetr.models.lwdetr` -> `rfdetr.models.matcher` -> `scipy.optimize`, at
# IMPORT time, so without it the sideways import in headway/detect.py dies with
# `No module named 'scipy'` and the pod comes up with no detector at all.
#
# Checked rather than guessed: of everything the import pulls in, scipy is the
# only package whose sole non-extra requirers are rfdetr and supervision --
# i.e. the two things installed here with --no-deps. pydantic, PyYAML, tqdm,
# regex and defusedxml all arrive properly as dependencies of fastapi,
# transformers, accelerate or huggingface_hub, so they are NOT listed here.
#
# This is not hypothetical. RF-DETR is the candidate source for the headway
# lead AND for the visual conversation's scene graph (docs/visual_qa.md), so a
# missing scipy is a pod where the gap warnings never fire and RIO cannot see
# anything to talk about. The smoke check below is what catches it.
log "RF-DETR (--no-deps: its dep tree breaks the pinned cv2/transformers)"
pip install --no-cache-dir --no-deps rfdetr==1.5.0 supervision==0.29.1 pycocotools peft
pip install --no-cache-dir --no-deps scipy
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
python - <<'PY' || log "  !! SMOKE CHECK FAILED - no headway candidates AND an empty scene graph"
import cv2
from headway import detect
detect._rfdetr_models()          # also proves scipy is present: matcher imports it
print(f"   rfdetr importable from {detect._pkg_dir()}")
assert any(hasattr(cv2, n) for n in ("TrackerCSRT_create", "TrackerCSRT")) or hasattr(cv2, "legacy"), \
    f"cv2 {cv2.__version__} has no CSRT - check for a stray opencv-python install"
print(f"   cv2 {cv2.__version__} has the tracking API")
PY

# ---------------------------------------------------------------------------
# 5d. Playwright + Chromium — the two selftests that need a real browser
# ---------------------------------------------------------------------------
# tools/output_bus_selftest.py and tools/mobile_layout_selftest.py drive an
# actual browser: the shared audio bus and its unlock, and the mobile layout
# swept over its whole scroll range. Neither can be faked in node, because what
# they assert is what a browser does with the page rather than what the code
# says it should.
#
# NOT FATAL, and nothing about the server needs it -- this is a test dependency
# and RIO drives without it. What a pod without it loses is the ability to run
# those two suites, which is worth knowing IN THE LOG rather than discovering
# as an ImportError months later.
#
# AND THAT IS NOT HYPOTHETICAL. On 2026-09-09 both of these had been quietly
# unrunnable here for as long as anyone had been asking for "the full suite":
# playwright was in neither requirements.txt nor this script, so both exited 2
# with an install hint. A skipped suite reads almost exactly like a passing one
# at a glance, which is the whole reason this step is here and logs when it
# fails.
#
# The browser lands in ~/.cache/ms-playwright, which is the container layer and
# does NOT survive a pod restart -- the same as apt packages, pip packages and
# torch above, and the reason all of them are reinstalled here on every boot.
# It is ~115 MB, against torch's ~3 GB, so it is not worth the persistent
# volume and a PLAYWRIGHT_BROWSERS_PATH to go with it.
#
# Three commands, because the browser is three things: the python package, the
# apt half it needs to run (fonts, libnss3, xvfb and the rest), and the browser
# binary itself. Chained on && so that a failure anywhere short-circuits to the
# one message -- there is no useful half-installed state to carry forward, and
# the message says what is lost rather than which of the three fell over. The
# log line above it is what says how far it got.
log "playwright + chromium (browser selftests)"
pip install --no-cache-dir playwright \
    && python -m playwright install-deps chromium \
    && python -m playwright install chromium \
    || log "  !! playwright unavailable - output_bus_selftest and mobile_layout_selftest cannot run"

# ---------------------------------------------------------------------------
# 6. Free port 8888
# ---------------------------------------------------------------------------
# RunPod starts JupyterLab on 8888, which is the port the proxy exposes and the
# port RIO needs. Jupyter has to go.
log "killing jupyter"
pkill -f jupyter || echo "   no jupyter running"

# Also clear any uvicorn from a previous run of this script, so a re-run doesn't
# leave two servers fighting over the port. Through rio_stop, which is careful
# about which processes it is allowed to kill -- see the block near the top.
echo "   clearing any existing uvicorn"
rio_stop

# ---------------------------------------------------------------------------
# 7. Launch RIO
# ---------------------------------------------------------------------------
# One implementation, shared with `bash boot.sh restart` -- see rio_start near
# the top for why it is setsid'd and why HF_HOME is stated rather than
# inherited.
log "launching uvicorn on :$PORT"
rio_start

# ---------------------------------------------------------------------------
# 8. Health check
# ---------------------------------------------------------------------------
# uvicorn reports ready before the model is warm (vision warms on a daemon
# thread), so /health answers in a few seconds. 60s is generous headroom.
log "health check"
rio_wait_healthy

# Model warm runs in the background and takes ~40s more. Not fatal, just noted.
echo "   (Qwen3-VL warm continues in background — watch: tail -f $REPO/uvicorn.log)"

# ---------------------------------------------------------------------------
# 8b. Preflight — did any of the above silently not happen?
# ---------------------------------------------------------------------------
# The steps above are individually loud and collectively easy to lose: several
# are deliberately non-fatal (a pod that comes up degraded beats a pod that does
# not come up), and this script is long enough that an abort in the middle looks
# like a successful run to anyone who only sees the end.
#
# So the last thing it does is ask, from scratch, what is actually present.
# Non-fatal on purpose -- the server is already up by this point and taking it
# down over a missing lane model would be the wrong trade -- but it prints what
# is missing and what breaks because of it, into a log that survives the pod.
log "preflight"
python -m tools.preflight || {
    echo "   !! this pod is INCOMPLETE — see the list above"
    echo "   !! for the repair commands: python -m tools.preflight --fix"
}

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
