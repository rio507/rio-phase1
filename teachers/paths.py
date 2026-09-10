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

# The two isolated environments. CONTAINER LAYER on purpose -- they are derived
# from a lockfile and boot.sh rebuilds them; it is the volume's quota that is
# scarce, and weights are the better use of it.
VENVS = {
    "alpamayo": "/opt/teachers/venvs/alpamayo/bin/python",
    "cosmos": "/opt/teachers/venvs/cosmos/bin/python",
}
LOG_DIR = "/opt/teachers/logs"

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
    # Copy rather than hardlink: the cache and the venvs are on different
    # filesystems and hardlinks do not cross that.
    env["UV_LINK_MODE"] = env.get("UV_LINK_MODE") or "copy"
    env["UV_PYTHON_INSTALL_DIR"] = (env.get("UV_PYTHON_INSTALL_DIR")
                                    or "/opt/teachers/pythons")
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
