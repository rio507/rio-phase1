#!/usr/bin/env bash
# boot.sh — rebuild this pod's container layer from scratch.
#
# Only /workspace survives a pod restart, and everything installed into the
# image layer does not. What that used to cost is written out under "WHAT
# SURVIVES A POD REBUILD" below; the short version is that the python
# environment, node and Claude Code have MOVED ONTO THE VOLUME, and what is
# still in the image layer -- apt packages, the teacher environments, the
# playwright browser -- is rebuilt by this script. Run it once after every
# fresh pod start to put the box back the way it was.
#
# Idempotent: safe to re-run at any time. Re-running reinstalls packages,
# restarts uvicorn, and leaves exactly one server on :8888.
#
#   bash /workspace/boot.sh
#
# A COMPLETELY FRESH POD IS ONE COMMAND AND ABOUT FIVE MINUTES:
#
#   bash /workspace/rio-phase1/boot.sh
#
# That is everything -- and on a pod whose volume came back, almost all of it
# is already there. What a fresh container actually pays for, measured on this
# pod on 2026-09-18:
#
#   python environment: requirements.txt, torch cu128,
#     rfdetr, scipy, playwright wheels                        210 s
#   apt (ffmpeg nano git tmux) and the browser's apt half      ~30 s
#   lane + detector weights, and the detector smoke check       24 s
#   uvicorn up and answering /health                            16 s
#   preflight                                                   31 s
#   ------------------------------------------------------------------
#   about five and a half minutes, and Qwen3-VL finishes warming
#   ~40 s after that
#
# The same run on a pod whose container is merely being re-provisioned --
# nothing to download, everything already where it was left -- was measured at
# 96 s end to end on 2026-09-18.
#
# ...and what it pays NOTHING for, because the volume kept it: the repo, the
# 16 GB of Qwen weights, the lane and detector weights, node, npm, Claude Code,
# chromium, and the teacher environments. The single biggest change is what is
# no longer here at all: the two teacher services are out of the live stack
# (see step 5e), so a fresh pod no longer builds 14 GB of environments or waits
# for two 8-to-10B models to load into 38 GB of VRAM.
#
# Every step times itself and the run ends with a table of where the time went,
# in /workspace/boot.log -- so the next fresh pod reports its own number rather
# than trusting this comment.
#
# JUST THE ENVIRONMENT, without the provisioning:
#
#   bash /workspace/rio-phase1/boot.sh bootstrap
#
# The python environment, node, claude, tmux, the PATH file, and :8888 taken
# back from whatever the template left on it. It is what a pod needs before it
# can run ANY of this repository, including the rest of this script, and the
# full run does it first as step 0.
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
# 8888 is the port RunPod's proxy exposes, so it is not really a choice. It is
# overridable only so that the port-ownership paths below -- who holds it, and
# taking it back -- can be exercised against a decoy without touching the real
# server. Same idiom as RIO_HTTPS_PORT further down.
PORT=${RIO_PORT:-8888}
HF_HOME_DIR=/workspace/.cache/huggingface
ENV_FILE=/workspace/env.sh

# The things that RUN this repo, all three on the persistent volume. See
# "WHAT SURVIVES A POD REBUILD" below for why they are not simply installed
# into the image like everything else.
VENV="$REPO/.venv"
VENV_PY="$VENV/bin/python"
VENV_PIP="$VENV/bin/pip"
VENV_UVICORN="$VENV/bin/uvicorn"
NODE_DIR=/workspace/node
NODE_VERSION=${RIO_NODE_VERSION:-22.11.0}

# CHROMIUM ON THE VOLUME. playwright's default is ~/.cache/ms-playwright, which
# is the container layer, so the browser went with every rebuild and the two
# suites that need it -- the shared audio bus and its unlock, and the mobile
# layout -- were unrunnable until someone re-ran the install. It is 130 MB
# against the volume's spare terabytes, and it is the difference between a
# fresh pod that can verify itself and one that cannot.
#
# Named here and exported into $ENV_FILE, and config.py defaults the same value
# for any python process that never saw a shell -- because a browser suite that
# cannot find its browser exits 2 with an install hint, which reads like a pass
# at a glance.
PLAYWRIGHT_DIR=/workspace/.cache/ms-playwright
export PLAYWRIGHT_BROWSERS_PATH="$PLAYWRIGHT_DIR"

# $PY and $PIP are what every step below invokes, and they are the VENV's --
# never a bare `python`, which means whatever the caller's PATH happens to say
# and, on a fresh pod, means an interpreter with none of this project's
# packages in it. The fallback to a bare python3 exists for ONE case: `status`
# and `bootstrap` on a pod that has no venv yet, which must still run.
rio_resolve_python() {
    if [ -x "$VENV_PY" ]; then
        PY="$VENV_PY"; PIP="$VENV_PIP"
    else
        PY=python3; PIP=pip
    fi
}
rio_resolve_python

# On the PERSISTENT volume, deliberately. A boot log inside the container layer
# is a boot log that disappears with the thing it was describing.
BOOT_LOG=/workspace/boot.log
exec > >(tee -a "$BOOT_LOG") 2>&1
printf '\n===== boot.sh %s (pid %s) =====\n' "$(date -Is)" "$$"

log() { printf '\n\033[1;36m==> %s\033[0m\n' "$*"; }

# ---------------------------------------------------------------------------
# WHERE THE TIME WENT — measured on the pod, printed at the end
# ---------------------------------------------------------------------------
# "How long does a fresh pod take?" was, until now, answerable only as somebody
# remembering. It is the question that decides whether a rebuild is a coffee or
# an evening, and it changes every time a step is added or removed -- most
# recently by 28 GB of teacher environments leaving. So each step says how long
# it took, in a log that survives the pod, and the summary at the end is the
# answer with no arithmetic.
#
# `step` replaces `log` for anything in the provisioning body: same heading,
# plus a stopwatch.
STEP_TIMES=()
STEP_NAME=""
STEP_T0=0
BOOT_T0=$(date +%s)

_step_close() {
    local now
    now=$(date +%s)
    [ -n "$STEP_NAME" ] && STEP_TIMES+=("$((now - STEP_T0))|$STEP_NAME")
    STEP_NAME=""
}

step() {
    _step_close
    STEP_NAME="$*"
    STEP_T0=$(date +%s)
    log "$*"
}

step_summary() {
    _step_close
    local row secs name total
    total=$(( $(date +%s) - BOOT_T0 ))
    log "where the time went"
    for row in ${STEP_TIMES[@]+"${STEP_TIMES[@]}"}; do
        secs=${row%%|*}; name=${row#*|}
        # Only the steps worth a line. A list where twelve rows say 0 s buries
        # the two that say 300.
        [ "$secs" -ge 2 ] && printf "   %5s s  %s\n" "$secs" "$name"
    done
    printf "   %5s s  TOTAL  (%s min)\n" "$total" "$(( (total + 30) / 60 ))"
    echo "   Steps under 2 s are not listed. Times are this pod, this network,"
    echo "   and this volume's warm caches -- which is the point: a fresh pod"
    echo "   reading this log is reading its own number, not an estimate."
}

# ---------------------------------------------------------------------------
# WHAT SURVIVES A POD REBUILD, AND WHAT DOES NOT
# ---------------------------------------------------------------------------
# /workspace is a network volume and it is the ONLY thing that comes back.
# /usr, /opt, /root and every site-packages directory pip has ever written into
# the image are the container layer, and they are gone the next time this pod
# is recreated.
#
# That distinction cost 45 minutes on 2026-09-17. The volume came back with the
# repository and the 16 GB HF cache intact, so the pod LOOKED provisioned --
# and nothing that RUNS the code was there. pip had installed into the image,
# so there was no Python environment at all; node, npm and Claude Code were
# gone; tmux was gone, so the long-running half of this script had nothing safe
# to run inside; and RunPod's own jupyter-lab held :8888, so the first uvicorn
# died on "[Errno 98] address already in use" -- a message that reads like a
# leftover RIO and was not one.
#
# The fix is not to reinstall faster. It is to put the things that RUN the code
# on the volume beside the repository that needs them:
#
#   $REPO/.venv        the Python environment. Every path in this script names
#                      .venv/bin/python and .venv/bin/uvicorn explicitly rather
#                      than `python`, which means whatever the caller's PATH
#                      says and on a fresh pod means an empty interpreter.
#   /workspace/node    node, npm, and Claude Code as an npm global (npm's
#                      prefix resolves to $NODE_DIR because node lives there).
#   /workspace/env.sh  the PATH that ties those two to any shell, restored by
#                      one line, because ~/.bashrc cannot be.
#
# Everything else is still deliberately container-local and still reinstalled
# on every boot: apt packages, the two ~14 GB teacher environments, the
# playwright browser. They are derived, they are large, and the volume's quota
# is better spent on weights. `bash boot.sh status` prints the whole split,
# marked VOLUME or CONTAINER, with what is actually present right now.

# Volume or container, decided by comparing the device the path is really on
# against /workspace's -- not by a table in this file that can go stale. For a
# path that does not exist yet, the nearest ancestor that does is the answer.
fs_kind() {
    local probe=$1
    while [ ! -e "$probe" ] && [ "$probe" != "/" ] && [ "$probe" != "." ]; do
        probe=$(dirname "$probe")
    done
    if [ "$(stat -c %d "$probe" 2>/dev/null || echo x)" \
       = "$(stat -c %d /workspace 2>/dev/null || echo y)" ]; then
        echo VOLUME
    else
        echo CONTAINER
    fi
}

persistence_report() {
    echo "   Only /workspace survives a pod rebuild. VOLUME rows come back on their"
    echo "   own; CONTAINER rows are re-made by \`bash boot.sh\`."
    echo
    echo "   There are three CONTAINER rows left, and all three are apt or a line"
    echo "   in a file apt owns. Everything that is expensive to fetch or slow to"
    echo "   build -- the python environment, node, Claude Code, chromium, the"
    echo "   weights, and the teacher environments -- is on the volume now."
    echo
    printf '   %-9s  %-7s  %-31s  %s\n' WHERE STATE PATH WHAT
    local spec path what state
    for spec in \
        "$REPO|the repository, and the drive logs under runs/" \
        "$VENV|PYTHON ENV: .venv/bin/python, .venv/bin/uvicorn" \
        "$NODE_DIR|node + npm" \
        "$NODE_DIR/bin/claude|Claude Code (npm global, prefix $NODE_DIR)" \
        "$ENV_FILE|PATH + HF_HOME. Restore with: . $ENV_FILE" \
        "$HF_HOME_DIR|model weights (Qwen3-VL, ~16 GB)" \
        "$REPO/weights|UFLDv2 lane + RF-DETR detector weights" \
        "$REPO/cert|TLS certificate (a phone needs https)" \
        "$BOOT_LOG|every run of this script, oldest first" \
        "$TEACHERS_ROOT|ON DEMAND: teacher checkouts, fp8, uv cache" \
        "$TEACHERS_VENVS|ON DEMAND: teacher envs, built by boot.sh teachers" \
        "$PLAYWRIGHT_DIR|chromium, for the two browser selftests" \
        "$HOME/.bashrc|the one line that sources $ENV_FILE" \
        "/usr/bin/tmux|tmux -- run the slow half of boot.sh inside it" \
        "/usr/bin/ffmpeg|apt: ffmpeg (audio muxing), nano, git" \
        ; do
        path=${spec%%|*}; what=${spec#*|}
        if [ -e "$path" ]; then state=present; else state=MISSING; fi
        # ~/.bashrc exists on every pod; what this report is asking about it is
        # whether it still has the line, which a rebuild is exactly what takes
        # away.
        if [ "$path" = "$HOME/.bashrc" ]; then
            grep -qF ". $ENV_FILE" "$path" 2>/dev/null && state=present || state=MISSING
        fi
        printf '   %-9s  %-7s  %-31s  %s\n' "$(fs_kind "$path")" "$state" "$path" "$what"
    done
    echo
    echo "   A CONTAINER row reading MISSING is normal on a fresh pod: that is the"
    echo "   half this script rebuilds. A VOLUME row reading MISSING is not -- it"
    echo "   means the volume did not come back the way it was left."
}

# ---------------------------------------------------------------------------
# bootstrap — make a fresh pod able to run this repository, in one command
# ---------------------------------------------------------------------------
#
#   bash boot.sh bootstrap
#
# Idempotent and safe to re-run: everything here checks for what it is about to
# make and says "present" instead of remaking it. Nothing in it needs the
# network for anything already on the volume, so on a pod that has been
# bootstrapped before it takes seconds.
#
# It does the CHEAP half. It deliberately does NOT do torch cu128, the Qwen
# weights, the lane and detector weights, playwright, or the teacher
# environments -- those are tens of gigabytes and they are the rest of this
# script. `bash boot.sh` runs bootstrap first and then all of it.

# The environment, on the volume, written by both `bootstrap` and step 1 so the
# two can never disagree about what PATH should be.
write_env_file() {
    cat > "$ENV_FILE" <<EOF
# Generated by boot.sh. ON THE PERSISTENT VOLUME ON PURPOSE: ~/.bashrc lives in
# the container layer, so every line ever appended to it is gone on the next
# pod start -- "it is in .bashrc" was only ever true until the rebuild. This
# file survives, and one line restores the whole environment in any shell, with
# no login and no .bashrc:
#
#   . $ENV_FILE
#
export HF_HOME=$HF_HOME_DIR
# Chromium for the browser selftests, on the volume rather than in ~/.cache,
# so it survives a pod rebuild like everything else that is expensive to fetch.
export PLAYWRIGHT_BROWSERS_PATH=$PLAYWRIGHT_DIR

# The two directories holding the things that RUN this repo, both on the
# volume: the venv's bin (python, pip, uvicorn) and node's (node, npm, npx,
# claude). Ahead of /usr/bin deliberately -- the image's python is not the one
# with this project's packages, and on a fresh pod it has none of them at all.
#
# Guarded rather than prepended blindly: this file is sourced from ~/.bashrc,
# by agents, and by hand, several times a session, and an unguarded prepend
# leaves PATH holding four copies of the same directory by lunchtime.
for _rio_dir in "\$HOME/.local/bin" $NODE_DIR/bin $VENV/bin; do
    case ":\$PATH:" in
        *":\$_rio_dir:"*) ;;
        *) PATH="\$_rio_dir:\$PATH" ;;
    esac
done
unset _rio_dir
export PATH
EOF
    echo "   wrote $ENV_FILE  (restore any shell with: . $ENV_FILE)"

    # grep-guarded so re-runs don't stack duplicate lines into .bashrc. This
    # line is container-local and has to be re-added after every rebuild; what
    # it points at is not, which is the whole arrangement.
    if ! grep -qF ". $ENV_FILE" ~/.bashrc 2>/dev/null; then
        printf '\n# RIO environment (persistent volume -- see %s)\n' "$ENV_FILE" >> ~/.bashrc
        printf '[ -f %s ] && . %s\n' "$ENV_FILE" "$ENV_FILE" >> ~/.bashrc
        echo "   sourced from ~/.bashrc (container-local, re-added every rebuild)"
    else
        echo "   already sourced from ~/.bashrc"
    fi
}

bootstrap_venv() {
    log "python environment (VOLUME) -> $VENV"
    if [ ! -f "$REPO/requirements.txt" ]; then
        echo "   !! no $REPO/requirements.txt — is the volume mounted?"
        return 1
    fi

    # A venv whose interpreter cannot run is the failure mode this arrangement
    # can still have: `python3 -m venv` records the image's interpreter, and
    # while the binary is copied in (--copies below) its shared libraries are
    # not. If the pod comes back on an image with a different python, the venv
    # on the volume is so much dead weight -- so prove it runs before trusting
    # it, and rebuild it from requirements.txt if it does not.
    if [ -e "$VENV" ] && ! "$VENV_PY" -c 'import sys' > /dev/null 2>&1; then
        echo "   !! $VENV_PY does not run — this venv was built against an"
        echo "      interpreter this pod no longer has. Removing and rebuilding it."
        rm -rf "$VENV"
    fi

    if [ ! -x "$VENV_PY" ]; then
        # --copies rather than symlinks: a symlink into /usr/bin is a pointer
        # into the container layer, and the point of this directory is to not
        # be one.
        echo "   creating: python3 -m venv --copies $VENV"
        python3 -m venv --copies "$VENV" || {
            echo "   !! could not create the venv (is python3-venv installed?)"
            return 1
        }
        "$VENV_PY" -m pip install --quiet --upgrade pip setuptools wheel \
            || echo "   !! could not upgrade pip in the new venv — continuing"
    fi
    rio_resolve_python
    echo "   python:  $("$VENV_PY" -V 2>&1) at $VENV_PY"

    # Stamped with the hash of requirements.txt, so re-running bootstrap on a
    # provisioned pod is a no-op rather than a five-minute dependency resolve.
    # The import check is there because a stamp can outlive the packages it
    # describes -- a half-finished install, or a venv someone pruned.
    local stamp want have
    stamp="$VENV/.requirements.sha256"
    want=$(sha256sum "$REPO/requirements.txt" | cut -d' ' -f1)
    have=$(cat "$stamp" 2>/dev/null || true)
    if [ "$want" = "$have" ] && "$VENV_PY" -c 'import fastapi, uvicorn' > /dev/null 2>&1; then
        echo "   packages: requirements.txt unchanged since the last install"
    else
        echo "   pip install -r requirements.txt  (the slow part, a few minutes)"
        "$PIP" install --no-cache-dir -r "$REPO/requirements.txt" || {
            echo "   !! pip install failed — the server will not start"
            return 1
        }
        echo "$want" > "$stamp"
    fi
    echo "   uvicorn: $VENV_UVICORN"

    # Which torch, stated rather than assumed. requirements.txt pins 2.4.1 from
    # PyPI; step 4 of the full run is what replaces it with the cu128 build
    # this pod's driver actually needs. A bootstrap-only pod can therefore be
    # sitting on the wrong one, and that is worth one line here rather than a
    # confusing CUDA error later.
    "$VENV_PY" - <<'PY' 2>/dev/null || echo "   torch:   not installed — run the full: bash boot.sh"
import torch
print("   torch:   %s (cuda build %s, available %s)"
      % (torch.__version__, torch.version.cuda, torch.cuda.is_available()))
PY
}

bootstrap_node() {
    log "node (VOLUME) -> $NODE_DIR"
    if [ -x "$NODE_DIR/bin/node" ] && "$NODE_DIR/bin/node" -v > /dev/null 2>&1; then
        echo "   present: node $("$NODE_DIR/bin/node" -v), npm $("$NODE_DIR/bin/npm" -v 2>/dev/null || echo '?')"
        return 0
    fi
    # NOT apt's nodejs, which is what this script used to install: that is node
    # 12, in the container layer, gone on the next rebuild and reinstalled on
    # every boot. An official tarball unpacked onto the volume is both current
    # and permanent, and it brings npm -- which apt's nodejs does not.
    local arch url tmp
    case "$(uname -m)" in
        x86_64)  arch=x64 ;;
        aarch64) arch=arm64 ;;
        *)       echo "   !! no node build for $(uname -m) — the two JS selftests cannot run"
                 return 0 ;;
    esac
    url="https://nodejs.org/dist/v$NODE_VERSION/node-v$NODE_VERSION-linux-$arch.tar.xz"
    tmp=$(mktemp -d)
    echo "   downloading node $NODE_VERSION ($arch)"
    if ! curl -fsSL "$url" -o "$tmp/node.tar.xz"; then
        echo "   !! download failed: $url"
        echo "   !! node, npm and claude will be unavailable on this pod"
        rm -rf "$tmp"
        return 0
    fi
    mkdir -p "$NODE_DIR"
    # --strip-components=1: the tarball's one top-level directory is dropped so
    # that bin/ lands directly in $NODE_DIR. That is what makes npm's global
    # prefix resolve to $NODE_DIR, which is what puts `claude` on the volume
    # instead of in ~/.local/bin.
    tar -xJf "$tmp/node.tar.xz" -C "$NODE_DIR" --strip-components=1 || {
        echo "   !! could not unpack node"
        rm -rf "$tmp"
        return 0
    }
    rm -rf "$tmp"
    echo "   installed: node $("$NODE_DIR/bin/node" -v), npm $("$NODE_DIR/bin/npm" -v 2>/dev/null || echo '?')"
}

bootstrap_claude() {
    log "Claude Code (VOLUME) -> $NODE_DIR/bin/claude"
    if [ ! -x "$NODE_DIR/bin/npm" ]; then
        echo "   no npm on the volume — skipped"
        return 0
    fi
    if [ -x "$NODE_DIR/bin/claude" ]; then
        echo "   present: $(PATH="$NODE_DIR/bin:$PATH" claude --version 2>/dev/null || echo 'installed')"
        return 0
    fi
    # The npm global, NOT claude.ai/install.sh -- that installer writes to
    # ~/.local/bin, which is the container layer, which is how this pod lost it
    # in the first place.
    echo "   npm install -g @anthropic-ai/claude-code"
    PATH="$NODE_DIR/bin:$PATH" "$NODE_DIR/bin/npm" install -g --silent @anthropic-ai/claude-code \
        || { echo "   !! npm install failed — claude will not be available"; return 0; }
    echo "   installed: $(PATH="$NODE_DIR/bin:$PATH" claude --version 2>/dev/null || echo 'installed')"
}

bootstrap_tmux() {
    log "tmux (CONTAINER — apt, so this runs again on every rebuild)"
    if command -v tmux > /dev/null 2>&1; then
        echo "   present: $(tmux -V)"
        return 0
    fi
    # It is here because of the header's first warning: the slow half of this
    # script downloads ~19 GB, and a dropped terminal SIGHUPs it into a
    # half-provisioned pod. tmux is what makes that not happen, so a pod
    # without it cannot safely run the thing that would install it.
    export DEBIAN_FRONTEND=noninteractive
    apt-get update -qq && apt-get install -y -qq --no-install-recommends tmux \
        || { echo "   !! apt could not install tmux — use: nohup bash boot.sh &"; return 0; }
    echo "   installed: $(tmux -V)"
}

bootstrap() {
    log "bootstrap — putting this pod back together"
    echo "   volume    (survives a rebuild): $VENV, $NODE_DIR, $ENV_FILE"
    echo "   container (rebuilt every time): apt packages, teacher venvs, chromium"

    log "env -> $ENV_FILE"
    write_env_file
    # shellcheck source=/dev/null
    . "$ENV_FILE"

    bootstrap_venv || {
        log "bootstrap FAILED at the python environment"
        echo "   Nothing in this repo can run without it. The rest was skipped."
        return 1
    }
    bootstrap_node
    bootstrap_claude
    bootstrap_tmux

    log "port $PORT"
    free_port || true

    log "what this pod has now"
    persistence_report

    log "bootstrap complete"
    echo "   restore this environment in any shell:  . $ENV_FILE"
    echo "   start the server:                       bash boot.sh start"
    echo "   see what is running:                    bash boot.sh status"
    echo
    echo "   NOT done here, because it is the expensive half: torch cu128, the"
    echo "   Qwen3-VL weights, the lane and detector weights, playwright, and the"
    echo "   teacher environments. For those, inside tmux:  bash boot.sh"
}

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
# server, so the question to ask is not "does this string appear" but "is this
# process actually uvicorn running app:app" -- which is a question about argv.
#
# AND IT IS NOT A QUESTION ABOUT /proc/<pid>/comm, which is what this function
# used to ask. On 2026-09-17 `status` printed "uvicorn: not running" while RIO
# was serving every request on :8888. comm was `pt_main_thread`: torch RENAMES
# THE MAIN THREAD the moment app.py imports it, so the one process that really
# was the server is exactly the one a comm filter of python/python3/uvicorn
# throws away. comm is 15 bytes any library is free to overwrite, and this one
# does, on the only process we care about.
#
# argv cannot be overwritten like that, and it separates us from the shell for
# free: RIO's argv holds `uvicorn` and the literal word `app:app` as SEPARATE
# arguments, while a shell that merely mentions them carries the whole command
# as ONE argv word -- that is what `bash -c` is. So an exact-word test over
# /proc/<pid>/cmdline answers both questions at once.

# True when this pid IS uvicorn serving app:app -- not a shell talking about it.
rio_is_ours() {
    local pid=$1 word saw_uvicorn=0 saw_app=0
    [ -r "/proc/$pid/cmdline" ] || return 1
    while IFS= read -r -d '' word; do
        case "$word" in
            uvicorn|*/uvicorn) saw_uvicorn=1 ;;
            app:app)           saw_app=1 ;;
        esac
    done < "/proc/$pid/cmdline"
    [ "$saw_uvicorn" -eq 1 ] && [ "$saw_app" -eq 1 ]
}

# True when any of the given words is a WHOLE argv element of this pid. Same
# test as rio_is_ours, for the processes named by a module path instead: the
# tls proxy and the two teacher services are started as `python -m <module>`,
# so the module is one argv word, and a shell that merely says the name carries
# it inside a much longer one.
proc_argv_has() {
    local pid=$1 word want
    shift
    [ -r "/proc/$pid/cmdline" ] || return 1
    while IFS= read -r -d '' word; do
        for want in "$@"; do
            [ "$word" = "$want" ] && return 0
        done
    done < "/proc/$pid/cmdline"
    return 1
}

# Every RUNNING uvicorn for this app, one pid per line, and nothing that merely
# mentions it.
rio_pids() {
    local pid
    for pid in $(pgrep -f 'uvicorn app:app' 2>/dev/null || true); do
        [ "$pid" = "$$" ] && continue
        # An `if`, not `rio_is_ours && echo`: a for loop returns the status of
        # its last iteration, so a final candidate that is NOT ours would make
        # this function itself "fail" -- and then `pids=$(rio_pidline)` fails
        # with it under `set -e`. Which is to say: a shell that merely mentions
        # uvicorn would abort the script that was carefully ignoring it.
        if rio_is_ours "$pid"; then
            echo "$pid"
        fi
    done
    return 0
}

# The pids on one line, trimmed, or empty. Used for reporting only.
rio_pidline() { rio_pids | tr '\n' ' ' | sed 's/ *$//'; }

# A pid described the way a person needs it in a report: its command line,
# trimmed. comm is appended in brackets because it is what `ps` and `ss` will
# show them, and on this box that is the misleading `pt_main_thread`.
proc_desc() {
    local pid=$1 comm cmd
    comm=$(cat "/proc/$pid/comm" 2>/dev/null || echo '?')
    # `|| cmd=` because a pid can exit between the listing and this line, and
    # a failed redirect inside a command substitution is a failed assignment.
    cmd=$(tr '\0' ' ' < "/proc/$pid/cmdline" 2>/dev/null | cut -c1-120) || cmd=""
    [ -n "$cmd" ] || cmd='(no command line)'
    printf '%s [%s]' "$cmd" "$comm"
}

# Every pid LISTENING on a port, whoever it belongs to. The question `status`
# has to answer first: something answering on :8888 is not evidence that it is
# us, and for 45 minutes on 2026-09-17 it was jupyter-lab.
# The `|| true` is load-bearing. Nothing listening is the NORMAL answer here,
# and it is also a FAILING one under `set -o pipefail`: grep matches nothing
# and returns 1, so `pids=$(port_pids ...)` inherits a non-zero status and
# `set -e` takes the whole script down at that assignment. On a fresh pod --
# where :8888 being free is the good case -- that was every run of bootstrap.
port_pids() {
    ss -lntpH "sport = :$1" 2>/dev/null | grep -o 'pid=[0-9]*' | cut -d= -f2 | sort -u || true
}

# Take :$PORT back from whatever is squatting on it, and never from ourselves.
#
# RunPod's template starts jupyter-lab on 8888 -- the port the pod's proxy
# exposes, and so the only port RIO can be reached on. A fresh pod therefore
# meets "[Errno 98] address already in use" before it ever serves a page, and
# that message never says who is holding it.
#
# Targeted by pid from the listener itself, never by `pkill -f jupyter` (which
# this script used to do): -f matches every process's whole command line, and
# an agent's `bash -c '... jupyter ...'` wrapper matches ITSELF. That is the
# same foot-gun documented at length above rio_pids, in the other direction.
free_port() {
    local pid pids desc alien i
    alien=""
    pids=$(port_pids "$PORT")
    if [ -z "$pids" ]; then
        echo "   :$PORT is free"
        return 0
    fi
    for pid in $pids; do
        desc=$(proc_desc "$pid")
        if rio_is_ours "$pid"; then
            echo "   :$PORT held by RIO itself (pid $pid) — left alone"
        else
            echo "   :$PORT held by a process that is NOT RIO — taking the port back"
            echo "      pid $pid: $desc"
            alien="$alien $pid"
        fi
    done
    [ -n "$alien" ] || return 0
    # shellcheck disable=SC2086
    kill $alien 2>/dev/null || true
    for i in $(seq 1 20); do
        [ -z "$(port_pids "$PORT")" ] && break
        sleep 0.5
    done
    for pid in $alien; do
        if kill -0 "$pid" 2>/dev/null; then
            echo "      pid $pid ignored SIGTERM after 10s — SIGKILL"
            kill -9 "$pid" 2>/dev/null || true
        fi
    done
    sleep 0.5
    pids=$(port_pids "$PORT" | tr '\n' ' ' | sed 's/ *$//')
    if [ -n "$pids" ]; then
        echo "   !! :$PORT is STILL held (pids $pids) — uvicorn will fail to bind"
        return 1
    fi
    echo "   :$PORT freed"
}

# "<http status><TAB><body>" for one GET, or "000<TAB>" when nothing answered.
# Deliberately without -L: a redirect is an answer about some other page.
http_probe() {
    local out
    out=$(curl -s -m 5 -w '\n%{http_code}' "$1" 2>/dev/null) || { printf '000\t'; return 0; }
    printf '%s\t%s' "${out##*$'\n'}" "${out%$'\n'*}"
}

# OUR /health body, or nothing -- and the whole point is the "our".
#
# `curl -sf .../health` was the other half of the 2026-09-17 status bug. With
# jupyter-lab on :8888, /health answered 302 to its login page: -f only fails
# from 400 up, a redirect body is empty, and the caller printed a blank line
# under the word "health" and reported the pod up. A health check that another
# process can pass is not a health check.
#
# So: 200 or nothing, and the body must carry RIO's own service marker.
rio_health_body() {
    local probe code body
    probe=$(http_probe "http://127.0.0.1:$PORT/health")
    code=${probe%%$'\t'*}
    body=${probe#*$'\t'}
    [ "$code" = "200" ] || return 1
    case "$body" in
        *'"service":"rio-phase1"'*|*'"service": "rio-phase1"'*) printf '%s' "$body" ;;
        *) return 1 ;;
    esac
}

# ---------------------------------------------------------------------------
# TLS, so a phone can be asked for the camera, the microphone and a position
# ---------------------------------------------------------------------------
# Over plain http on a LAN address a browser refuses all three BEFORE drawing a
# prompt — not as a permission the driver denied, but as a capability never
# offered. `localhost` is exempt; a phone is never on localhost. That is what
# the 2026-09-16 drive hit, and it looked like a bad search rather than a
# missing prompt.
#
# TLS terminates in a small proxy (tools/tls_proxy.py) rather than in uvicorn
# because this process loads Qwen3-VL: turning the listener into an HTTPS one
# would mean a second 16 GB server, or no plain-HTTP dashboard for the fifteen
# selftests and every curl in this repository that speak it. So the edge
# speaks TLS and the loopback stays plaintext, which is how it is done in
# production anyway.
#
# It is started ONLY when a certificate exists. No certificate is not a fault:
# the dashboard works perfectly on http://localhost for desk work, and a
# machine that never serves a phone never needs one.
HTTPS_PORT=${RIO_HTTPS_PORT:-8443}
CERT_FILE="$REPO/cert/rio-cert.pem"

# AND THE SAME TEST HERE, because this function is how the lesson above got
# re-learned. On 2026-09-18 `boot.sh restart` killed the shell that ran it: a
# bare `pgrep -f tools.tls_proxy` handed tls_stop the pid of the caller's own
# `bash -c`, whose command line happened to contain that string, and tls_stop
# killed it. The server restarted perfectly; the terminal asking for it died
# mid-sentence. Every function in this file that kills by pattern has to ask
# what a process IS, not what it mentions.
tls_pids() {
    local pid
    for pid in $(pgrep -f 'tools.tls_proxy' 2>/dev/null || true); do
        [ "$pid" = "$$" ] && continue
        if proc_argv_has "$pid" tools.tls_proxy; then
            echo "$pid"
        fi
    done
    return 0
}

tls_stop() {
    local pids
    pids=$(tls_pids)
    [ -z "$pids" ] && return 0
    echo "   stopping tls proxy: $(echo "$pids" | tr '\n' ' ')"
    # shellcheck disable=SC2086
    kill $pids 2>/dev/null || true
    sleep 0.5
}

tls_start() {
    if [ ! -f "$CERT_FILE" ]; then
        echo "   no certificate — https not started (http://localhost:$PORT still works)"
        echo "   for a phone: python -m tools.make_cert --add <the address you will type>"
        return 0
    fi
    tls_stop
    setsid nohup "$PY" -m tools.tls_proxy --listen "$HTTPS_PORT" --to "$PORT" \
        > "$REPO/tls.log" 2>&1 < /dev/null &
    sleep 1
    if [ -z "$(tls_pids)" ]; then
        echo "   !! tls proxy did not come up — tail of tls.log:"
        tail -n 8 "$REPO/tls.log" 2>/dev/null | sed 's/^/      /'
        return 1
    fi
    echo "   https on :$HTTPS_PORT -> :$PORT  (pid $(tls_pids | tr '\n' ' '))"
    echo "   phone: https://<this machine's LAN address>:$HTTPS_PORT/"
    return 0
}

tls_status() {
    local pids
    pids=$(tls_pids)
    if [ -z "$pids" ]; then
        echo "   tls proxy: not running"
    else
        echo "   tls proxy: $(echo "$pids" | tr '\n' ' ')  (:$HTTPS_PORT)"
    fi
    if [ -f "$CERT_FILE" ]; then
        "$PY" -m tools.tls_proxy --check 2>&1 | sed 's/^/   /'
    else
        echo "   certificate: none (python -m tools.make_cert)"
    fi
}

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
    # The interpreter is named, not looked up. A bare `uvicorn` is whatever the
    # caller's PATH says, and on a pod that has not been bootstrapped it is
    # either nothing at all or the image's python with none of this project's
    # packages -- which is a server that dies on `import app` rather than one
    # that never starts, and reads like a code bug.
    if [ ! -x "$VENV_UVICORN" ]; then
        echo "   !! no uvicorn at $VENV_UVICORN"
        echo "   !! this pod has no python environment — run: bash boot.sh bootstrap"
        return 1
    fi
    # Whoever is on the port right now is about to become "[Errno 98] address
    # already in use" in a log nobody is tailing yet. Name them here instead.
    local holder
    for holder in $(port_pids "$PORT"); do
        rio_is_ours "$holder" && continue
        echo "   !! :$PORT is held by something that is not RIO, so uvicorn cannot bind:"
        echo "   !!    pid $holder: $(proc_desc "$holder")"
        echo "   !! free it with: bash boot.sh bootstrap"
        return 1
    done
    # The log of the run you are restarting BECAUSE OF is the one thing you
    # need after a crash, and `>` erases it.
    #
    # ONE GENERATION WAS NOT ENOUGH. On 2026-09-10 a live-conversation failure
    # was reported hours after it happened, and by then the server log covering
    # it had been rotated away by the restarts in between -- so the question
    # "did the browser ever ask for a session?" could not be answered from the
    # server at all, only inferred from the drive's own JSONL. Three
    # generations is a few megabytes and covers a debugging session's worth of
    # restarts.
    for g in 3 2 1; do
        [ -f "$REPO/uvicorn.log.$g" ] && mv -f "$REPO/uvicorn.log.$g" \
            "$REPO/uvicorn.log.$((g + 1))"
    done
    [ -f "$REPO/uvicorn.log.prev" ] && cp -f "$REPO/uvicorn.log.prev" "$REPO/uvicorn.log.1"
    [ -f "$REPO/uvicorn.log" ] && mv -f "$REPO/uvicorn.log" "$REPO/uvicorn.log.prev"
    rm -f "$REPO/uvicorn.log.5"
    # setsid + nohup + </dev/null: the server has to outlive the shell that
    # started it. A plain background job belongs to the caller's session, so an
    # agent's `bash -c` wrapper exiting takes the server with it -- the other
    # half of the failure this whole block is about.
    #
    # HF_HOME is passed EXPLICITLY rather than left to inheritance. This is the
    # line people copy out of the log and re-run by hand, and a uvicorn started
    # without HF_HOME re-downloads 16 GB of Qwen3-VL into a container layer that
    # is about to disappear.
    #
    # PATH is stated for the same reason: what this server shells out to --
    # ffmpeg, node, and its own python -- must be the volume's copies, not
    # whatever the shell that happened to start it had.
    setsid env HF_HOME="$HF_HOME_DIR" PATH="$VENV/bin:$NODE_DIR/bin:$PATH" nohup \
        "$VENV_UVICORN" app:app --host 0.0.0.0 --port "$PORT" \
        > "$REPO/uvicorn.log" 2>&1 < /dev/null &
    sleep 1
    local pids
    pids=$(rio_pidline)
    if [ -z "$pids" ]; then
        echo "   !! uvicorn did not come up — tail of uvicorn.log:"
        tail -20 "$REPO/uvicorn.log"
        return 1
    fi
    echo "   started: $pids"
    echo "   python:  $VENV_UVICORN (HF_HOME=$HF_HOME_DIR)"
    # Stated on every start, because it decides 38 GB of VRAM and it is decided
    # HERE, at startup, from the environment this shell happens to carry.
    if [ "${RIO_TEACHERS_ENABLED-0}" = "0" ]; then
        echo "   teachers: off — the live pipeline only (bash boot.sh teachers)"
    else
        echo "   teachers: ON (RIO_TEACHERS_ENABLED=$RIO_TEACHERS_ENABLED) — the"
        echo "             panel will poll :8801/:8802 and write corpus rows"
    fi
}

# uvicorn answers /health well before the model is warm (vision warms on a
# daemon thread), so this waits for the SERVER, not for readiness.
rio_wait_healthy() {
    local i body holder
    for i in $(seq 1 60); do
        # rio_health_body, not `curl -sf`: it waits for OUR health answer, and
        # something else on this port answering 302 is not this server coming
        # up -- see the note above that function.
        if body=$(rio_health_body); then
            echo "   up after ${i}s: $body"
            return 0
        fi
        sleep 1
    done
    echo "   !! no RIO /health on :$PORT after 60s"
    for holder in $(port_pids "$PORT"); do
        if rio_is_ours "$holder"; then
            echo "   !! :$PORT is ours (pid $holder) but it is not answering /health yet"
        else
            echo "   !! :$PORT is held by something else — pid $holder: $(proc_desc "$holder")"
        fi
    done
    echo "   !! tail of uvicorn.log:"
    tail -20 "$REPO/uvicorn.log"
    return 1
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
#   /workspace/teachers/venvs   the environments, ON THE VOLUME since
#                               2026-09-18 -- 13.9 GB together, hardlinked out
#                               of the uv cache beside them, so a pod that was
#                               rebuilt can serve teachers without building
#                               anything. See teachers/paths.py for why that
#                               reasoning inverted.
#   /workspace/teachers/pythons uv's standalone interpreters, with them: a venv
#                               on the volume whose python is in the image is
#                               dead weight the moment the pod comes back.
#   $HF_HOME                    the weights, on the volume, like Qwen3-VL's.
#   /workspace/teachers/fp8     the FP8 checkpoints, on the volume, because
#                               regenerating one takes 20 minutes and the L40S
#                               pod loads them on every boot.
#
# NON-FATAL, everywhere. A pod with no teacher services drives exactly as it
# did before the panel existed: /health reports them degraded, the Teachers
# card says which one is not answering, and nothing else changes.
TEACHERS_ROOT=/workspace/teachers
# ON THE VOLUME, and teachers/paths.py is where that decision is written down
# and why. The short version: boot.sh no longer builds these, so a rebuild is
# somebody waiting; they are 13.9 GB against the 75 GB of uv cache that existed
# to make rebuilding them fast; and on the same filesystem as that cache, uv
# hardlinks them out of it rather than copying.
TEACHERS_VENVS="$TEACHERS_ROOT/venvs"
TEACHERS_LOGS="$TEACHERS_ROOT/logs"
# uv's DOWNLOAD AND UNPACK CACHE, ON THE VOLUME. Not a tidiness preference:
# the container layer is 60 GB and the two teacher environments are ~28 GB of
# it, so there is no room left on it for uv to also keep an unpacked copy of
# every wheel AND build the ephemeral environment a `uv run --script` needs.
#
# On 2026-09-10 that filled the overlay completely, part-way through resolving
# the FP8 quantizer's environment, and left the box with no writable temp space
# at all -- which is a much worse failure than a slow download, because nothing
# that could have diagnosed it could run either.
#
# On the volume it also survives a pod rebuild, so `teachers-build` after a
# restart is a copy rather than a 28 GB download.
export UV_CACHE_DIR="$TEACHERS_ROOT/uv-cache"
# UV_LINK_MODE is deliberately NOT forced to "copy" any more: that was for a
# cache on the volume and venvs in the container layer, which hardlinks cannot
# cross. Both live on the volume now, so uv's default hardlinking applies and
# the venvs cost almost nothing beyond the cache that was already there.
#
# THAT CACHE IS 75 GB, and with the venvs beside it it is no longer buying
# much: it exists to make a rebuild fast, and there is now nothing to rebuild
# on a fresh pod. It is safe to delete and it will refill itself the next time
# `teachers-build` genuinely has to resolve something:
#
#   rm -rf /workspace/teachers/uv-cache
#
# Left in place rather than deleted by this script, because 75 GB is not a
# thing a boot script should decide to throw away on somebody's behalf.
ALPAMAYO_REPO=https://github.com/NVlabs/alpamayo1.5.git
ALPAMAYO_SHA=36aeb4c5938cbc2eb2aed33b22434773da4ab639
COSMOS_REPO=https://github.com/nvidia-cosmos/cosmos-reason2.git
COSMOS_SHA=a3b4a1db4065fe13c4b1f4d2fb8605bad647f4b9

# The HF token for the gated nvidia/Cosmos-Reason2-8B repo. On the volume,
# OUTSIDE the git worktree, and sourced rather than baked in -- see the header
# of that file. Never echoed.
[ -f "$TEACHERS_ROOT/secrets.env" ] && . "$TEACHERS_ROOT/secrets.env"

# By argv, not by comm, for exactly the reason rio_is_ours is: these two
# processes load torch, torch renames the main thread to pt_main_thread, and a
# comm filter of python/python3 therefore stops matching a teacher service the
# moment it has finished loading -- i.e. from the point it starts being worth
# stopping. `teachers-stop` would have reported "no teacher services running"
# with both of them holding 30 GB of VRAM.
#
# A module path is one argv word (`python -m teachers.service.cosmos_service`),
# so an exact-word test separates the service from any shell mentioning it.
teacher_pids() {
    local pid
    for pid in $(pgrep -f 'teachers.service.(alpamayo|cosmos)_service' 2>/dev/null || true); do
        [ "$pid" = "$$" ] && continue
        if proc_argv_has "$pid" teachers.service.alpamayo_service \
                                teachers.service.cosmos_service; then
            echo "$pid"
        fi
    done
    return 0
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

# A teacher environment is BUILT when it imports what the service imports --
# not when a bin/python exists there.
#
# The difference is not academic: moving these onto the volume landed
# TEACHERS_VENVS on top of an abandoned attempt from 2026-09-10, whose
# bin/python ran perfectly and whose site-packages were empty. The existence
# test said "built", the build was skipped, and `boot.sh teachers` would then
# have started a service that dies on `import alpamayo1_5` -- which is a
# failure that reads as a crashed model rather than a missing install. This is
# the same lesson as bootstrap_venv's "prove it runs before trusting it", one
# layer up: prove it imports.
teacher_env_ok() {
    local py="$TEACHERS_VENVS/$1/bin/python" mods
    [ -x "$py" ] || return 1
    case "$1" in
        alpamayo) mods="torch, transformers, alpamayo1_5" ;;
        cosmos)   mods="torch, transformers" ;;
        *)        return 1 ;;
    esac
    "$py" -c "import $mods" > /dev/null 2>&1
}

teachers_build() {
    log "teacher environments (isolated, on the volume, built on demand)"
    export PATH="$HOME/.local/bin:$PATH"
    if ! command -v uv > /dev/null 2>&1; then
        echo "   installing uv"
        curl -LsSf https://astral.sh/uv/install.sh | sh > /dev/null 2>&1
        export PATH="$HOME/.local/bin:$PATH"
    fi
    teachers_clone || return 1
    mkdir -p "$TEACHERS_VENVS" "$TEACHERS_LOGS"
    # --clear on both `uv venv` calls below: uv refuses to write into an
    # existing environment, and this code only runs when teacher_env_ok has
    # already said the one that is there does not import what it must. An
    # abandoned venv from 2026-09-10 sat exactly there and turned the rebuild
    # into "error: A virtual environment already exists".
    export UV_PYTHON_INSTALL_DIR="$TEACHERS_ROOT/pythons"
    mkdir -p "$UV_CACHE_DIR"
    echo "   uv cache: $UV_CACHE_DIR ($(du -sh "$UV_CACHE_DIR" 2>/dev/null | cut -f1))"

    if ! teacher_env_ok alpamayo; then
        echo "   building alpamayo env (python 3.12, torch 2.8)"
        # --no-install-package flash-attn: flash-attn compiles from source
        # against nvcc and takes half an hour. The upstream README documents
        # SDPA as the supported fallback and the service asks for it by name.
        # compressed-tensors is added AFTER the sync, with torch and
        # transformers named explicitly so the resolver cannot move them. It is
        # what LOADS an FP8 checkpoint -- without it the service starts fine at
        # BF16 and fails with "compressed_tensors is not installed" the moment
        # a quantized checkpoint appears on the volume, which is exactly when
        # nobody is expecting a new failure.
        #
        # Explicit pins because this is the step that destroyed this venv once:
        # `uv pip install llmcompressor` unpinned resolved to a version wanting
        # transformers 5.x and took torch to cu130 with it. The quantizer lives
        # in its own PEP-723 environment now for that reason; only the small
        # runtime half belongs here.
        ( cd "$TEACHERS_ROOT/src/alpamayo1.5" \
          && uv venv --clear --python 3.12 "$TEACHERS_VENVS/alpamayo" \
          && VIRTUAL_ENV="$TEACHERS_VENVS/alpamayo" uv sync --active \
               --no-install-package flash-attn \
          && uv pip install -q --python "$TEACHERS_VENVS/alpamayo/bin/python" \
               "compressed-tensors==0.13.0" "torch==2.8.0" "transformers==4.57.1" ) \
            > "$TEACHERS_LOGS/venv_alpamayo.log" 2>&1 \
            || echo "   !! alpamayo env failed — see $TEACHERS_LOGS/venv_alpamayo.log"
    fi
    echo "   alpamayo: $("$TEACHERS_VENVS/alpamayo/bin/python" -c \
        'import torch,transformers;print("torch",torch.__version__,"tf",transformers.__version__)' \
        2>/dev/null || echo 'NOT BUILT')"

    if ! teacher_env_ok cosmos; then
        echo "   building cosmos env (python 3.12, torch 2.9)"
        # No vendor package: Cosmos-Reason2 is a Qwen3-VL architecture and
        # loads through plain transformers. The cosmos-reason2 checkout is here
        # for its quantization recipe and its prompts, not for inference.
        ( uv venv --clear --python 3.12 "$TEACHERS_VENVS/cosmos" \
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

# Which weights each service loads.
#
# BF16 BY DEFAULT, EVEN WHERE AN FP8 CHECKPOINT EXISTS -- and that is the
# opposite of what this function did when it was written, because the
# measurement came out the opposite way to the expectation.
#
# The FP8 checkpoints are real and their weights really are a third smaller.
# But they are loaded here through plain transformers + compressed-tensors,
# which DEQUANTIZES on the fly: latency roughly doubled for Cosmos and went up
# sevenfold for Alpamayo, and the peak during inference went UP, not down. See
# docs/teacher_panel.md §10b for the table. NVIDIA's recipe is aimed at vLLM,
# where the FP8 kernels are fused; in this serving path it is a pessimisation.
#
# So FP8 is opt-in, by TEACHERS_PRECISION=fp8, and the checkpoints stay on the
# volume because the moment either teacher is served through vLLM they become
# the right thing to load. Choosing it silently would have made the everyday
# pod slower in exchange for a memory saving it does not actually get.
teacher_weights() {
    local model=$1 fp8dir
    case "$model" in
        alpamayo) fp8dir="$TEACHERS_ROOT/fp8/alpamayo_fp8" ;;
        cosmos)   fp8dir="$TEACHERS_ROOT/fp8/model_fp8" ;;
    esac
    if [ "${TEACHERS_PRECISION-bf16}" = "fp8" ] && [ -f "$fp8dir/config.json" ]; then
        echo "fp8|$fp8dir"
    else
        echo "bf16|"
    fi
}

# THE ONE COMMAND THAT BRINGS THE PANEL BACK.
#
# Builds the environments first if they are not there -- that is the step the
# full boot used to do and no longer does, so this has to, or `boot.sh
# teachers` on a fresh pod would just print "no environment" twice.
teachers_start() {
    log "teacher services (shadow — on demand)"
    mkdir -p "$TEACHERS_LOGS"
    if ! teacher_env_ok alpamayo || ! teacher_env_ok cosmos; then
        echo "   environments are missing — building them first (~28 GB, from"
        echo "   $UV_CACHE_DIR if it is warm, which makes it a copy)"
        teachers_build || return 1
    fi
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
    # AND THE HALF THAT IS NOT A PROCESS. The server decides at STARTUP whether
    # it reads the panel at all (config.TEACHERS_ENABLED, off by default), so
    # two healthy services and an unchanged uvicorn is a corpus that never gets
    # written and a card that never appears. Say so, with the command.
    log "the server still has to be told"
    echo "   These two are serving, but this server was started without them."
    echo "   To collect a corpus:"
    echo
    echo "     RIO_TEACHERS_ENABLED=1 bash boot.sh restart"
    echo
    echo "   That restart is what puts the Teachers card back on the dashboard"
    echo "   and starts writing corpus rows. Plain 'bash boot.sh restart' turns"
    echo "   the panel off again and leaves these services running -- stop them"
    echo "   with 'bash boot.sh teachers-stop' to get the 38 GB back."
}

teachers_status() {
    local port name body answering=0
    # NOTHING RUNNING IS THE NORMAL STATE. Two lines of "no answer on :8801"
    # read like a fault, and after 2026-09-18 they are not one: the panel is
    # off unless a collection session asked for it. Only say something is wrong
    # when the server was told to expect them.
    for port in 8801 8802; do
        curl -sf -m 2 "http://127.0.0.1:$port/health" > /dev/null 2>&1 \
            && answering=$((answering + 1))
    done
    if [ "$answering" -eq 0 ]; then
        echo "   not running — and that is the default: the live stack does not"
        echo "   use them. 38 GB of VRAM, a corpus for post-training, on demand:"
        echo "      bash boot.sh teachers   (then: RIO_TEACHERS_ENABLED=1 bash boot.sh restart)"
        if [ "${RIO_TEACHERS_ENABLED-0}" != "0" ]; then
            echo "   !! but THIS shell has RIO_TEACHERS_ENABLED=$RIO_TEACHERS_ENABLED —"
            echo "   !! a server restarted from here would poll two services that"
            echo "   !! are not there and write no corpus rows."
        fi
        return 0
    fi
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
        printf '%s' "$body" | "$PY" -c 'import json,sys
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

# What is actually running, in three answers that do not borrow from each
# other: is one of OUR processes alive, is the port OURS, and does OUR server
# say it is healthy.
#
# Both halves of this used to lie, on the same evening. It printed "uvicorn:
# not running" while RIO served every request (comm said pt_main_thread -- see
# rio_is_ours), and on the pod before that it printed a healthy-looking line
# that was jupyter-lab's 302 to its own login page. The second is the worse
# failure: a status that reports someone else's answer as ours is worse than
# one that reports nothing, because it ends the investigation.
rio_status() {
    local pids lp body probe code lpids ours_on_port=0
    pids=$(rio_pidline)
    if [ -z "$pids" ]; then
        echo "   uvicorn: no RIO process — nothing on this box has 'uvicorn app:app' as its argv"
    else
        echo "   uvicorn: $pids"
        for lp in $pids; do
            echo "            pid $lp: $(proc_desc "$lp")"
        done
    fi

    lpids=$(port_pids "$PORT")
    if [ -z "$lpids" ]; then
        echo "   :$PORT   nothing is listening"
    else
        for lp in $lpids; do
            if rio_is_ours "$lp"; then
                ours_on_port=1
                echo "   :$PORT   OURS — pid $lp: $(proc_desc "$lp")"
            else
                echo "   :$PORT   NOT OURS — pid $lp: $(proc_desc "$lp")"
                echo "            RIO cannot bind while that holds the port."
                echo "            Take it back with: bash boot.sh bootstrap"
            fi
        done
    fi

    if [ -z "$lpids" ]; then
        echo "   health:  not asked — nothing is listening on :$PORT"
        return 0
    fi
    probe=$(http_probe "http://127.0.0.1:$PORT/health")
    code=${probe%%$'\t'*}
    if [ "$ours_on_port" -ne 1 ]; then
        echo "   health:  NOT ASKED OF RIO — the listener on :$PORT is not ours."
        echo "            It answered /health with HTTP $code. That is ITS answer;"
        echo "            it is not evidence of anything about this server."
        return 0
    fi
    if body=$(rio_health_body); then
        echo "   health:  $body"
    else
        echo "   health:  OURS, BUT NOT HEALTHY — /health returned HTTP $code"
        echo "            (still starting, or it failed during startup:"
        echo "             tail -40 $REPO/uvicorn.log)"
    fi
}

# Everything below runs `python -m tools.something` sooner or later, and those
# imports are relative to the repository. The script is documented as
# `bash /workspace/boot.sh`, which leaves cwd wherever the caller was standing.
cd "$REPO"

# Dispatch before the provisioning traps below are installed: none of these
# subcommands provisions the pod in the expensive sense, so none of them should
# abort saying the pod is half-built. `bootstrap` is here for that reason too:
# it reports its own failures, in the terms of the thing that failed.
case "${1-}" in
    bootstrap) bootstrap; exit $? ;;
    storage|persistence) log "volume vs container"; persistence_report; exit 0 ;;
    restart) log "restarting uvicorn on :$PORT"; rio_stop; rio_start
             log "health check"; rio_wait_healthy; rc=$?
             log "https"; tls_start; exit $rc ;;
    stop)    log "stopping uvicorn"; rio_stop; tls_stop; exit 0 ;;
    start)   log "starting uvicorn on :$PORT"; rio_stop; rio_start
             log "health check"; rio_wait_healthy; rc=$?
             log "https"; tls_start; exit $rc ;;
    status)  log "RIO on :$PORT"; rio_status
             log "https"; tls_status
             log "teachers (shadow — not part of a drive)"; teachers_status
             log "volume vs container — what a fresh pod would still have"
             persistence_report; exit 0 ;;
    https)   log "starting tls proxy on :$HTTPS_PORT"; tls_start; exit $? ;;
    https-stop) log "stopping tls proxy"; tls_stop; exit 0 ;;
    cert)    shift; "$PY" -m tools.make_cert "$@"; exit $? ;;
    teachers)       teachers_start; exit $? ;;
    teachers-stop)  log "stopping teacher services"; teachers_stop; exit 0 ;;
    teachers-build) teachers_build; exit $? ;;
    "")      ;;
    *)       echo "usage: bash boot.sh [bootstrap|storage|restart|stop|start|status|cert|https|https-stop|teachers|teachers-stop|teachers-build]" >&2
             echo "       (no argument: the full provision, bootstrap included)" >&2; exit 2 ;;
esac

# `set -e` makes this script abort on the first failure, which is right -- but a
# silent abort is how a half-provisioned pod happens. Say where it stopped, in
# the persistent log, and say what to run to find out what is missing.
trap 'rc=$?; printf "\n!! boot.sh ABORTED at line %s (exit %s): %s\n" \
      "$LINENO" "$rc" "$BASH_COMMAND"; \
      printf "   The pod is PARTIALLY provisioned. What is missing:\n"; \
      printf "     cd %s && %s -m tools.preflight\n" "$REPO" "$PY"; \
      printf "   Full log: %s\n" "$BOOT_LOG"; exit $rc' ERR
trap 'printf "\n!! boot.sh was INTERRUPTED (signal). The pod is PARTIALLY\n"; \
      printf "   provisioned -- run: cd %s && %s -m tools.preflight\n" "$REPO" "$PY"; \
      exit 130' HUP INT TERM

# ---------------------------------------------------------------------------
# 0. bootstrap — the environment this script itself runs in
# ---------------------------------------------------------------------------
# FIRST, and not optional, because every step below this one invokes $PY and
# $PIP: a `pip install` is only worth doing once there is somewhere for it to
# land that will still be there tomorrow. This is the same `bash boot.sh
# bootstrap` a fresh pod is told to run on its own, it is idempotent, and on a
# pod that has already had it it costs seconds.
step "bootstrap (step 0)"
bootstrap || {
    echo
    echo "!! bootstrap failed. Nothing below this line can work without it, so"
    echo "!! this run stops here rather than installing into a pod that has"
    echo "!! nowhere to put it. Full log: $BOOT_LOG"
    exit 1
}

# ---------------------------------------------------------------------------
# 1. HF_HOME — keep model weights on the persistent volume
# ---------------------------------------------------------------------------
# Without this, transformers caches into ~/.cache inside the container layer and
# re-downloads all 16GB of Qwen3-VL-8B on every pod start. Step 0 has already
# written it into $ENV_FILE and sourced it; this states it again for this
# script's own children, which is the same reason rio_start states it.
step "HF_HOME -> $HF_HOME_DIR"
export HF_HOME="$HF_HOME_DIR"
mkdir -p "$HF_HOME_DIR"

# The environment itself -- PATH and HF_HOME -- is written to $ENV_FILE on the
# volume by write_env_file, which step 0 has already called. It used to be
# inlined here; it moved so that `bootstrap` and a full run cannot disagree
# about what PATH is supposed to be. See that function for why ~/.bashrc is not
# the place for it.

# ---------------------------------------------------------------------------
# 2. System packages
# ---------------------------------------------------------------------------
# ffmpeg: ElevenLabs audio muxing. nano: editing on the box. git: push/pull.
#
# NODE IS NOT HERE ANY MORE, and neither is tmux: both are step 0's, and node
# is on the volume. Two of this project's test suites are JavaScript, because
# two of the things worth testing are -- the arbiter and the route tracker both
# run in the browser, and both are written as pure modules precisely so `node`
# can drive them without a page, which makes `node tools/nav_selftest.js` and
# `node tools/realtime_selftest.js` 240-odd checks that need an interpreter.
# apt's nodejs is version 12 in the container layer, reinstalled on every boot;
# $NODE_DIR is a current one that survives the rebuild, and it brings npm,
# which apt's nodejs does not and which is how Claude Code gets here.
step "apt packages (ffmpeg nano git)"
export DEBIAN_FRONTEND=noninteractive
apt-get update -qq
apt-get install -y -qq --no-install-recommends ffmpeg nano git

# ---------------------------------------------------------------------------
# 3. Python dependencies — installed in step 0, and here is why they moved
# ---------------------------------------------------------------------------
# This step used to be `pip install -r requirements.txt` against whatever pip
# was on PATH, which was the image's: the packages went into the container
# layer, and on 2026-09-17 this pod came back with the repository and all 16 GB
# of weights intact and NO PYTHON ENVIRONMENT AT ALL to run them with.
#
# They now go into $VENV on the volume, in step 0, early -- because everything
# below this line runs $PY. bootstrap_venv is stamped with the hash of
# requirements.txt, so a re-run is a no-op and a changed requirements.txt is a
# real install.
step "python dependencies"
echo "   $("$VENV_PY" -V 2>&1) at $VENV_PY"
echo "   requirements.txt installed there by step 0"

# ---------------------------------------------------------------------------
# 4. Torch cu128 — MUST come after requirements.txt
# ---------------------------------------------------------------------------
# requirements.txt pins torch==2.4.1 (plain PyPI, CPU/default-CUDA wheels). This
# pod is an L40S on driver 570 / CUDA 12.8, which that build does not target.
# Step 3 will happily drag torch back down to 2.4.1, so this force-reinstall runs
# afterwards and wins. --force-reinstall (not plain install) because pip
# considers 2.11.0 already-satisfied and would no-op on a re-run of this script
# after a partial/mixed install.
# ...AND IT IS SKIPPED WHEN IT IS ALREADY DONE. The force-reinstall exists
# because pip's metadata can say 2.11.0 over a half-installed tree, so the
# question is asked of the RUNTIME instead: the right version, and a CUDA
# context that actually initialises. That is a stronger test than pip's, and it
# takes ~3 GB of download off every re-run of this script while leaving a cold
# pod paying exactly what it paid before.
step "torch cu128"
if "$VENV_PY" - <<'PY'
import sys
try:
    import torch
except Exception:
    sys.exit(1)
sys.exit(0 if (torch.__version__ == "2.11.0+cu128"
               and torch.cuda.is_available()) else 1)
PY
then
    echo "   already torch 2.11.0+cu128 with a working CUDA runtime — skipped"
else
echo "   installing torch 2.11.0+cu128 (~3 GB)"
"$PIP" install --no-cache-dir --force-reinstall \
    torch==2.11.0+cu128 \
    torchvision==0.26.0+cu128 \
    torchaudio==2.11.0+cu128 \
    --index-url https://download.pytorch.org/whl/cu128
fi

# The torch wheels are ~3GB; the cache is on the container layer but the disk
# pressure is real during install.
step "pip cache purge"
"$PIP" cache purge || true

# ---------------------------------------------------------------------------
# 5. GPU sanity
# ---------------------------------------------------------------------------
step "GPU sanity"
"$PY" - <<'PY'
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
step "UFLDv2 lane weights"
"$PY" -m tools.fetch_lane_weights || log "  !! lane weights unavailable - headway will use the static corridor"

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
step "RF-DETR (--no-deps: its dep tree breaks the pinned cv2/transformers)"
"$PIP" install --no-cache-dir --no-deps rfdetr==1.5.0 supervision==0.29.1 pycocotools peft
"$PIP" install --no-cache-dir --no-deps scipy
# Both of the next two run `python -m` / import from the repo, so they need the
# repo as cwd. The script is documented as `bash /workspace/boot.sh`, which
# leaves cwd wherever the caller happened to be; the cd further down (step 7)
# was too late to help them.
cd "$REPO"

step "RF-DETR weights"
"$PY" -m tools.fetch_detector_weights || log "  !! detector weights unavailable - headway will have no candidates"

# Prove the two things that have actually broken here before, while the log is
# still being read, rather than discovering them mid-drive as an UNKNOWN band:
#   * the sideways rfdetr import (it was pinned to a hardcoded python3.12 path
#     and silently died when the pod was rebuilt on 3.11)
#   * CSRT, which disappears if anything drags opencv-python 5.x in on top of
#     the pinned contrib-headless build
step "detector + tracker smoke check"
"$PY" - <<'PY' || log "  !! SMOKE CHECK FAILED - no headway candidates AND an empty scene graph"
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
# THE BROWSER IS ON THE VOLUME NOW ($PLAYWRIGHT_DIR, exported at the top of
# this file and into $ENV_FILE). It used to land in ~/.cache/ms-playwright --
# the container layer -- so every rebuild took it, and the two suites that need
# it went quiet again until somebody noticed. 130 MB is nothing against what
# this volume already holds, and `python -m playwright install chromium` on a
# pod that still has it is a no-op that prints one line.
#
# The apt half (fonts, libnss3, xvfb and the rest) is NOT on the volume and
# cannot be: those are system packages, so install-deps runs on every boot like
# the rest of apt.
#
# Three commands, because the browser is three things: the python package, the
# apt half it needs to run (fonts, libnss3, xvfb and the rest), and the browser
# binary itself. Chained on && so that a failure anywhere short-circuits to the
# one message -- there is no useful half-installed state to carry forward, and
# the message says what is lost rather than which of the three fell over. The
# log line above it is what says how far it got.
step "playwright + chromium (browser selftests)"
"$PIP" install --no-cache-dir playwright \
    && "$PY" -m playwright install-deps chromium \
    && "$PY" -m playwright install chromium \
    || log "  !! playwright unavailable - output_bus_selftest and mobile_layout_selftest cannot run"

# ---------------------------------------------------------------------------
# 5e. The teacher panel is NOT BUILT HERE ANY MORE (docs/teacher_panel.md)
# ---------------------------------------------------------------------------
# It used to be: two `uv sync` runs, ~28 GB of wheels, on every fresh pod. Then
# step 8c started both services, and they sat there holding 38 GB of VRAM
# (alpamayo 21408 MB, cosmos 16760 MB) for the whole life of the pod.
#
# For a drive that buys NOTHING. The panel is shadow by construction: nothing
# either model says can reach the arbiter, the speech path, look() or the
# observer cache -- tools/teacher_firewall_selftest.py asserts it from the AST.
# What they produce is a corpus for post-training, and a corpus is collected in
# sessions, deliberately, not continuously. Paying for it on every pod and
# during every drive meant paying the entire cost of the capability for none of
# its value -- and it set the floor on what GPU this car needs, which is the
# thing that made it worth changing.
#
# So they are ON DEMAND now, and one command brings them back:
#
#   bash boot.sh teachers                     builds if needed, then serves
#   RIO_TEACHERS_ENABLED=1 bash boot.sh restart    the server then reads them
#
# Nothing was deleted. The package, the schema, the corpus, the association and
# the replay path are untouched, `teachers-build` still builds, and every
# teacher selftest sets config.TEACHERS_ENABLED itself, so they all still run.

# ---------------------------------------------------------------------------
# 6. Free port 8888
# ---------------------------------------------------------------------------
# RunPod starts JupyterLab on 8888, which is the port the proxy exposes and so
# the only port RIO can be reached on. Jupyter has to go -- and it goes BY PID,
# taken from the listener itself, rather than by `pkill -f jupyter`, which is
# what this step used to be: -f matches the pattern against every process's
# whole command line, the shell running the pkill included. See free_port, and
# the longer version of that story above rio_pids.
#
# Step 0 already did this. It is repeated because a full run takes the better
# part of an hour, and anything at all may have taken the port back meanwhile.
step "freeing :$PORT"
free_port

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
step "launching uvicorn on :$PORT"
rio_start

# ---------------------------------------------------------------------------
# 8. Health check
# ---------------------------------------------------------------------------
# uvicorn reports ready before the model is warm (vision warms on a daemon
# thread), so /health answers in a few seconds. 60s is generous headroom.
step "health check"
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
step "preflight"
"$PY" -m tools.preflight || {
    echo "   !! this pod is INCOMPLETE — see the list above"
    echo "   !! for the repair commands: python -m tools.preflight --fix"
}

# ---------------------------------------------------------------------------
# 8c. The teacher services are NOT STARTED HERE ANY MORE
# ---------------------------------------------------------------------------
# See step 5e. A pod that comes up without them is COMPLETE, not degraded:
# /health says nothing about them, preflight counts nothing against them, and
# the dashboard has no Teachers card. `bash boot.sh teachers` is the one
# command that brings the whole thing back for a collection session.

# ---------------------------------------------------------------------------
# 9. Claude Code — installed by step 0, onto the volume
# ---------------------------------------------------------------------------
# This used to be `curl -fsSL https://claude.ai/install.sh | bash`, which puts
# the binary in ~/.local/bin: the container layer. So it went with the rebuild,
# and the fresh pod had no agent on it to ask about the fresh pod. It is an npm
# global under $NODE_DIR now, which is why bootstrap_node comes first.
#
# Called again rather than assumed: this is a long script, and step 0 ran
# before ~30 GB of downloads that can fail in ways that leave the box tidy.
bootstrap_claude

# ---------------------------------------------------------------------------
# 10. What this pod keeps, and what it will lose
# ---------------------------------------------------------------------------
# Last, because it is the part worth reading when the pod comes back and this
# log is the only account of what was here. Every VOLUME row is something the
# next pod starts with; every CONTAINER row is something this script had to
# make, and will have to make again.
step "volume vs container"
persistence_report

step_summary

log "boot complete — RIO on :$PORT"
echo "   restore this environment in any shell:  . $ENV_FILE"
