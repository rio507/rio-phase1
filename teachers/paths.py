"""Where things go — stated in code, not left to whoever's shell it is.

WHY THIS FILE EXISTS
--------------------
Twice in one build the container layer filled completely and took the box with
it: no writable temp space, so nothing that could have diagnosed the problem
could run either, and it needed a terminal outside the harness to clear.

Both times the cause was the same. `uv` keeps its download cache AND the
ephemeral environment a PEP-723 `uv run --script` resolves into under
UV_CACHE_DIR, which defaults to ~/.cache/uv on the container layer. That layer
is 60 GB and already holds ~28 GB of teacher venvs; one torch unpack is ~10 GB.

The first fix set UV_CACHE_DIR in boot.sh and /workspace/env.sh. That is right
and not enough: it only applies to a process whose shell sourced one of them,
and `tools/teacher_input_selftest.py` shells out to `uv run` inheriting
os.environ. Run from a plain shell -- an agent's `bash -c`, a fresh ssh, a cron
entry -- the variable is simply absent and uv goes back to the container layer.
So the second time it filled, it filled from a test.

An environment variable that has to be exported by the caller is a convention.
This is the value, in code, applied by the thing that needs it.
"""
import os

# The persistent volume. Everything below hangs off it.
TEACHERS_ROOT = "/workspace/teachers"

# uv's download cache AND its ephemeral script environments. On the volume,
# which has room, rather than the container layer, which does not.
UV_CACHE_DIR = os.path.join(TEACHERS_ROOT, "uv-cache")

# The weights, shared with RIO's own Qwen3-VL.
HF_HOME = "/workspace/.cache/huggingface"

# The FP8 checkpoints.
FP8_DIR = os.path.join(TEACHERS_ROOT, "fp8")

# The pinned upstream checkouts.
SRC_DIR = os.path.join(TEACHERS_ROOT, "src")

# The two isolated environments. ON THE VOLUME since 2026-09-18, and the
# reasoning inverted rather than drifted.
#
# They were in the container layer because boot.sh rebuilt them on every pod
# and the volume's quota was thought better spent on weights. Two facts changed
# that. First, boot.sh does NOT build them any more -- the panel left the live
# stack, so the build only happens when `bash boot.sh teachers` asks for it, and
# a rebuild is then something a person is waiting on rather than something that
# happened overnight. Second, the measurement: the two venvs are 13.9 GB
# together, while the uv cache kept on the volume to make rebuilding them fast
# is 75 GB. Spending 75 GB of volume to avoid spending 14 GB of volume is
# backwards, and it bought a ten-minute wait rather than avoiding one.
#
# On the volume they are simply there after a rebuild, `boot.sh teachers` is a
# start rather than a build, and -- because the cache is now on the SAME
# filesystem -- uv can hardlink them out of it instead of copying, so those
# 13.9 GB largely share extents with the cache that was already paid for.
#
# The interpreters go with them (UV_PYTHON_INSTALL_DIR below): a venv on the
# volume whose python lives in the image is the dangling-interpreter failure
# that boot.sh's bootstrap_venv documents for RIO's own environment, and the
# whole point of moving these is that they survive.
VENVS = {
    "alpamayo": os.path.join(TEACHERS_ROOT, "venvs/alpamayo/bin/python"),
    "cosmos": os.path.join(TEACHERS_ROOT, "venvs/cosmos/bin/python"),
}
LOG_DIR = os.path.join(TEACHERS_ROOT, "logs")

# uv's standalone python builds, on the volume for the same reason.
PYTHON_INSTALL_DIR = os.path.join(TEACHERS_ROOT, "pythons")

# The token for the gated Cosmos repo, outside the git worktree.
SECRETS = os.path.join(TEACHERS_ROOT, "secrets.env")


def subprocess_env(extra: dict = None) -> dict:
    """os.environ, with the paths a child process must not have to be told.

    Use this for EVERY subprocess in this project that might invoke uv, python
    with transformers, or anything that downloads a model. Setting the values
    rather than defaulting them: a caller that has already exported a different
    UV_CACHE_DIR meant it, but a caller that exported nothing must not fall
    back to the container layer.
    """
    env = dict(os.environ)
    env["UV_CACHE_DIR"] = env.get("UV_CACHE_DIR") or UV_CACHE_DIR
    # UV_LINK_MODE is no longer forced to "copy". That was here because the
    # cache was on the volume and the venvs were in the container layer, and a
    # hardlink does not cross filesystems. Both are on the volume now, so uv's
    # default -- hardlink, falling back to copy on its own if the filesystem
    # refuses -- is both faster and very nearly free in space.
    env["UV_PYTHON_INSTALL_DIR"] = (env.get("UV_PYTHON_INSTALL_DIR")
                                    or PYTHON_INSTALL_DIR)
    env["HF_HOME"] = env.get("HF_HOME") or HF_HOME
    # ~/.local/bin is where uv installs itself and is not on a non-login PATH.
    env["PATH"] = os.path.expanduser("~/.local/bin") + ":" + env.get("PATH", "")
    if extra:
        env.update(extra)
    return env


def load_secrets(env: dict = None) -> dict:
    """Read /workspace/teachers/secrets.env into an env dict. Never logs it.

    A plain `KEY=value` parse rather than sourcing a shell: this is read by
    python tools that have no shell, and the file is one variable and a
    comment block.
    """
    env = env if env is not None else {}
    try:
        with open(SECRETS) as f:
            for line in f:
                line = line.strip()
                if not line or line.startswith("#"):
                    continue
                if line.startswith("export "):
                    line = line[len("export "):]
                if "=" not in line:
                    continue
                k, v = line.split("=", 1)
                env.setdefault(k.strip(), v.strip().strip('"').strip("'"))
    except OSError:
        pass
    return env
