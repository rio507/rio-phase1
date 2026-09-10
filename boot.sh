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

# ---------------------------------------------------------------------------
# The teacher panel's two services (docs/teacher_panel.md)
# ---------------------------------------------------------------------------
#
#   bash boot.sh teachers          start both, wait for /health
#   bash boot.sh teachers-stop     stop both
#   bash boot.sh teachers-build    (re)build the two isolated environments
#
# THEY ARE NOT RIO AND THEY DO NOT LIVE IN RIO'S INTERPRETER. Alpamayo 1.5
# wants Python 3.12, torch 2.8 and transformers 4.57.1; Cosmos-Reason2 wants
# torch 2.9 and transformers 4.57.3; RIO runs 3.11 with torch 2.11. Those three
# sets cannot coexist, which is why there are three environments and why the
# panel talks to its models over a loopback socket instead of importing them.
#
# WHERE THINGS LIVE, AND WHY THERE:
#   /workspace/teachers/src     the two pinned upstream checkouts. Persistent
#                               volume: they are ~50 MB and re-cloning them on
#                               every pod start is a network dependency at boot
#                               for no benefit.
#   /opt/teachers/venvs         the environments. CONTAINER LAYER, deliberately,
#                               and rebuilt by this script -- exactly like the
#                               pip packages in step 3 and the torch wheels in
#                               step 4. They are ~14 GB each and they are
#                               derived from a lockfile; the volume's quota is
#                               better spent on weights.
#   $HF_HOME                    the weights, on the volume, like Qwen3-VL's.
#   /workspace/teachers/fp8     the FP8 checkpoints, on the volume, because
#                               regenerating one takes 20 minutes and the L40S
#                               pod loads them on every boot.
#
# NON-FATAL, everywhere. A pod with no teacher services drives exactly as it
# did before the panel existed: /health reports them degraded, the Teachers
# card says which one is not answering, and nothing else changes.
TEACHERS_ROOT=/workspace/teachers
TEACHERS_VENVS=/opt/teachers/venvs
TEACHERS_LOGS=/opt/teachers/logs
ALPAMAYO_REPO=https://github.com/NVlabs/alpamayo1.5.git
ALPAMAYO_SHA=36aeb4c5938cbc2eb2aed33b22434773da4ab639
COSMOS_REPO=https://github.com/nvidia-cosmos/cosmos-reason2.git
COSMOS_SHA=a3b4a1db4065fe13c4b1f4d2fb8605bad647f4b9

# The HF token for the gated nvidia/Cosmos-Reason2-8B repo. On the volume,
# OUTSIDE the git worktree, and sourced rather than baked in -- see the header
# of that file. Never echoed.
[ -f "$TEACHERS_ROOT/secrets.env" ] && . "$TEACHERS_ROOT/secrets.env"

teacher_pids() {
    local pid comm
    for pid in $(pgrep -f 'teachers.service.(alpamayo|cosmos)_service' 2>/dev/null || true); do
        [ "$pid" = "$$" ] && continue
        comm=$(cat "/proc/$pid/comm" 2>/dev/null || true)
        case "$comm" in
            python|python3|python3.*) echo "$pid" ;;
        esac
    done
}

teachers_stop() {
    local pids
    pids=$(teacher_pids | tr '\n' ' ' | sed 's/ *$//')
    if [ -z "$pids" ]; then
        echo "   no teacher services running"
        return 0
    fi
    echo "   stopping teachers: $pids"
    # shellcheck disable=SC2086
    kill $pids 2>/dev/null || true
    for i in $(seq 1 20); do
        [ -z "$(teacher_pids)" ] && break
        sleep 0.5
    done
    pids=$(teacher_pids | tr '\n' ' ' | sed 's/ *$//')
    if [ -n "$pids" ]; then
        echo "   still up after 10s — SIGKILL: $pids"
        # shellcheck disable=SC2086
        kill -9 $pids 2>/dev/null || true
    fi
}

teachers_clone() {
    mkdir -p "$TEACHERS_ROOT/src" "$TEACHERS_LOGS"
    local name url sha dir
    for spec in "alpamayo1.5|$ALPAMAYO_REPO|$ALPAMAYO_SHA" \
                "cosmos-reason2|$COSMOS_REPO|$COSMOS_SHA"; do
        name=${spec%%|*}; url=${spec#*|}; sha=${url#*|}; url=${url%%|*}
        dir="$TEACHERS_ROOT/src/$name"
        if [ ! -d "$dir/.git" ]; then
            echo "   cloning $name"
            git clone --quiet "$url" "$dir" || { echo "   !! clone failed"; return 1; }
        fi
        # PINNED, and re-pinned on every run. A teacher whose inference code
        # moved is a corpus whose rows are no longer comparable with the ones
        # before it, and the only way to notice is to state the sha.
        (cd "$dir" && git fetch --quiet origin "$sha" 2>/dev/null || true
         git -C "$dir" checkout --quiet "$sha" 2>/dev/null) \
            || echo "   !! could not pin $name to $sha"
        echo "   $name @ $(git -C "$dir" rev-parse --short HEAD)"
    done
}

teachers_build() {
    log "teacher environments (isolated, container layer)"
    export PATH="$HOME/.local/bin:$PATH"
    if ! command -v uv > /dev/null 2>&1; then
        echo "   installing uv"
        curl -LsSf https://astral.sh/uv/install.sh | sh > /dev/null 2>&1
        export PATH="$HOME/.local/bin:$PATH"
    fi
    teachers_clone || return 1
    mkdir -p "$TEACHERS_VENVS" "$TEACHERS_LOGS"
    # Copy rather than hardlink: the uv cache is on the container layer and the
    # checkouts are on a network volume, and hardlinks do not cross that.
    export UV_LINK_MODE=copy UV_PYTHON_INSTALL_DIR=/opt/teachers/pythons

    if [ ! -x "$TEACHERS_VENVS/alpamayo/bin/python" ]; then
        echo "   building alpamayo env (python 3.12, torch 2.8)"
        # --no-install-package flash-attn: flash-attn compiles from source
        # against nvcc and takes half an hour. The upstream README documents
        # SDPA as the supported fallback and the service asks for it by name.
        ( cd "$TEACHERS_ROOT/src/alpamayo1.5" \
          && uv venv --python 3.12 "$TEACHERS_VENVS/alpamayo" \
          && VIRTUAL_ENV="$TEACHERS_VENVS/alpamayo" uv sync --active \
               --no-install-package flash-attn ) \
            > "$TEACHERS_LOGS/venv_alpamayo.log" 2>&1 \
            || echo "   !! alpamayo env failed — see $TEACHERS_LOGS/venv_alpamayo.log"
    fi
    echo "   alpamayo: $("$TEACHERS_VENVS/alpamayo/bin/python" -c \
        'import torch,transformers;print("torch",torch.__version__,"tf",transformers.__version__)' \
        2>/dev/null || echo 'NOT BUILT')"

    if [ ! -x "$TEACHERS_VENVS/cosmos/bin/python" ]; then
        echo "   building cosmos env (python 3.12, torch 2.9)"
        # No vendor package: Cosmos-Reason2 is a Qwen3-VL architecture and
        # loads through plain transformers. The cosmos-reason2 checkout is here
        # for its quantization recipe and its prompts, not for inference.
        ( uv venv --python 3.12 "$TEACHERS_VENVS/cosmos" \
          && uv pip install -q --python "$TEACHERS_VENVS/cosmos/bin/python" \
               --extra-index-url https://download.pytorch.org/whl/cu128 \
               --index-strategy unsafe-best-match \
               "torch==2.9.0+cu128" "torchvision==0.24.0+cu128" \
               "transformers==4.57.3" "accelerate==1.12.0" "pillow==12.0.0" \
               "numpy<3" safetensors compressed-tensors ) \
            > "$TEACHERS_LOGS/venv_cosmos.log" 2>&1 \
            || echo "   !! cosmos env failed — see $TEACHERS_LOGS/venv_cosmos.log"
    fi
    echo "   cosmos:   $("$TEACHERS_VENVS/cosmos/bin/python" -c \
        'import torch,transformers;print("torch",torch.__version__,"tf",transformers.__version__)' \
        2>/dev/null || echo 'NOT BUILT')"
}

# Which weights each service loads. FP8 when a checkpoint is there, BF16
# otherwise -- so the L40S pod picks up the quantized build automatically and
# this box, where BF16 fits, keeps using it unless somebody built one.
teacher_weights() {
    local model=$1 fp8dir
    case "$model" in
        alpamayo) fp8dir="$TEACHERS_ROOT/fp8/alpamayo_fp8" ;;
        cosmos)   fp8dir="$TEACHERS_ROOT/fp8/model_fp8" ;;
    esac
    if [ -n "${TEACHERS_PRECISION-}" ] && [ "$TEACHERS_PRECISION" = "bf16" ]; then
        echo "bf16|"
    elif [ -f "$fp8dir/config.json" ]; then
        echo "fp8|$fp8dir"
    else
        echo "bf16|"
    fi
}

teachers_start() {
    log "teacher services"
    mkdir -p "$TEACHERS_LOGS"
    teachers_stop
    local spec prec weights py port mod
    for model in alpamayo cosmos; do
        case "$model" in
            alpamayo) py="$TEACHERS_VENVS/alpamayo/bin/python"; port=8801
                      mod=teachers.service.alpamayo_service ;;
            cosmos)   py="$TEACHERS_VENVS/cosmos/bin/python";   port=8802
                      mod=teachers.service.cosmos_service ;;
        esac
        if [ ! -x "$py" ]; then
            echo "   $model: no environment — run: bash boot.sh teachers-build"
            continue
        fi
        spec=$(teacher_weights "$model"); prec=${spec%%|*}; weights=${spec#*|}
        echo "   starting $model on :$port at $prec ${weights:+($weights)}"
        # setsid + nohup, for exactly the reason rio_start is: the service has
        # to outlive the shell that started it.
        # shellcheck disable=SC2086
        setsid env HF_HOME="$HF_HOME_DIR" nohup "$py" -m "$mod" \
            --port "$port" --precision "$prec" \
            ${weights:+--model "$weights"} \
            > "$TEACHERS_LOGS/$model.log" 2>&1 < /dev/null &
    done
    echo "   loading (~40-90 s per model; watch: tail -f $TEACHERS_LOGS/*.log)"
    local i loaded
    for i in $(seq 1 90); do
        loaded=0
        for port in 8801 8802; do
            curl -sf "http://127.0.0.1:$port/health" 2>/dev/null \
                | grep -q '"loaded": *true' && loaded=$((loaded + 1))
        done
        [ "$loaded" -eq 2 ] && break
        sleep 3
    done
    teachers_status
}

teachers_status() {
    local port name body
    for spec in "8801|alpamayo1.5" "8802|cosmos-reason2"; do
        port=${spec%%|*}; name=${spec#*|}
        printf '   %-16s ' "$name"
        # Captured rather than piped. Under `set -o pipefail` a curl against a
        # closed port fails the WHOLE pipeline, so a `| python | || echo`
        # printed the fallback on top of python's own message -- two lines
        # saying the same thing, one of them from a command that succeeded.
        body=$(curl -s -m 5 "http://127.0.0.1:$port/health" 2>/dev/null || true)
        if [ -z "$body" ]; then
            echo "no answer on :$port"
            continue
        fi
        printf '%s' "$body" | python3 -c 'import json,sys
try:
    h = json.load(sys.stdin)
except Exception:
    print("unreadable /health"); raise SystemExit
print(("loaded  " if h.get("loaded") else "NOT LOADED  ")
      + str(h.get("model_id") or "") + " @ " + str(h.get("precision") or "")
      + "  vram=" + str(h.get("vram_reserved_mb")) + " MB"
      + ("  " + str(h.get("load_error"))[:120] if h.get("load_error") else ""))'
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
    status)  log "RIO on :$PORT"; rio_status
             log "teachers"; teachers_status; exit 0 ;;
    teachers)       teachers_start; exit $? ;;
    teachers-stop)  log "stopping teacher services"; teachers_stop; exit 0 ;;
    teachers-build) teachers_build; exit $? ;;
    "")      ;;
    *)       echo "usage: bash boot.sh [restart|stop|start|status|teachers|teachers-stop|teachers-build]" >&2; exit 2 ;;
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
# 5e. The teacher panel's two environments (docs/teacher_panel.md)
# ---------------------------------------------------------------------------
# Two AV foundation models in shadow, each in its own Python. Deliberately NOT
# fatal and deliberately late: a pod that comes up without them drives exactly
# as it did before the panel existed, and the two `uv sync` runs are ~28 GB of
# wheels that must not stand between a fresh pod and a working server.
#
# The services themselves are started AFTER uvicorn (step 8c), so a pod whose
# weights are missing still gets a dashboard.
log "teacher environments"
teachers_build || log "  !! teacher environments unavailable - the panel's two columns will be empty"

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
# 8c. The teacher services
# ---------------------------------------------------------------------------
# After the server, because they are the shadow and it is the drive. FP8 if a
# quantized checkpoint is on the volume, BF16 otherwise -- see teacher_weights.
log "teacher services"
teachers_start || log "  !! teacher services did not start - the panel's columns will be empty"

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
