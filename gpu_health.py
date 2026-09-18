"""Did the card run out of memory? — counted, so /health can stop being blind.

WHY THIS EXISTS
---------------
On 2026-09-18 the live pipeline was run inside a deliberately undersized card
(tools/vram_budget.py --squeeze 18) to find out how it fails when VRAM is
short. It does not fail loudly. The server started, /health said "ok" with an
empty degraded list, nothing crashed, and every request was answered -- while
the log, and only the log, said this:

    [headway.live] lane detection off: OutOfMemoryError: CUDA out of memory
    [observer] ...: RuntimeError: CUBLAS_STATUS_ALLOC_FAILED
    [perceive] lane detection unavailable: CUDA out of memory

34 frame results came back out of 558 sent, and server time per frame went
from 17 ms to 2767 ms at p50. The car was driving with no lane model, no
observer sentences and no enrichment, and there was no way to know from
outside the process.

That silence is not an accident and it is not a bug: every one of those sites
catches deliberately, because a drive should survive a missing lane model
rather than stop. The mistake was that surviving and being fine looked
identical. This module is the difference.

WHAT IT IS, AND WHAT IT IS NOT
------------------------------
It is a counter with a classifier. A site that already swallows an exception
hands it here on the way past; if it is the card refusing to allocate, it is
counted against that component with a timestamp. Nothing here raises, nothing
retries, and nothing changes what any caller does -- adding it cannot change
how a drive behaves, which is the only way it was safe to put on every one of
these paths.

It deliberately does NOT try to be a GPU monitor. It does not poll nvidia-smi,
it does not import torch, and it holds no opinion about how much memory is
free. It answers one question: has this process been refused memory by the
card, when, and what did that cost. What the pipeline actually needs is
measured by tools/vram_budget.py, which is where the numbers in the health
entry below come from.
"""
import threading
import time

# THE SHAPES AN OUT-OF-MEMORY TAKES, and there are more than one, which is half
# the reason this is centralised. torch raises OutOfMemoryError for its own
# allocator; a cuBLAS workspace that cannot be allocated arrives as a
# RuntimeError naming CUBLAS_STATUS_ALLOC_FAILED and is the same underlying
# condition wearing a different name; cuDNN has its own. Matching on the text
# as well as the type because the type is not always torch's.
_SIGNS = (
    "out of memory",
    "outofmemoryerror",
    "cublas_status_alloc_failed",
    "cudnn_status_alloc_failed",
    "cuda error: out of memory",
    "cuda_error_out_of_memory",
    "failed to allocate",
)

# What each component stops doing when it is refused memory. The point of the
# health entry is not "an exception happened" -- it is what the car has quietly
# stopped being able to do, which is different for each of these.
_COST = {
    "lanes": "no lane model: headway falls back to the static trapezoid "
             "corridor and corridor_source stays 'static'",
    "observer": "no observer sentences: RIO has nothing to say about the road "
                "and /talk answers from a stale cache or not at all",
    "enrich": "tracked objects keep their class and lose their description, so "
              "'the black one' cannot be resolved",
    "perceive": "the Camera panel's geometry stops updating",
    "visual_qa": "look() cannot answer a question about what is out of the "
                 "window",
    "detector": "no detector candidates: headway reports UNKNOWN for the whole "
                "drive and the scene graph is empty",
}

# How recent is "still happening". A card that was short during model load and
# has been fine for an hour is a different story from one refusing allocations
# right now, and the entry says which.
RECENT_S = 120.0

_lock = threading.Lock()
_state = {
    "events": 0,
    "first_at": None,
    "last_at": None,
    "last": "",
    "components": {},       # name -> {"events": n, "last_at": t, "last": str}
}


def looks_like_oom(exc) -> bool:
    """Is this exception the card refusing to allocate?"""
    if exc is None:
        return False
    name = type(exc).__name__.lower()
    text = f"{name}: {exc}".lower()
    return any(s in text for s in _SIGNS)


def note(component: str, exc) -> bool:
    """Record `exc` against `component` if it is an out-of-memory. -> was it one.

    Called from sites that are ALREADY catching and already continuing. It
    returns a bool rather than re-raising precisely so that a caller cannot
    accidentally make this module change the outcome of a drive.
    """
    if not looks_like_oom(exc):
        return False
    now = time.time()
    msg = f"{type(exc).__name__}: {exc}"[:240]
    with _lock:
        _state["events"] += 1
        _state["first_at"] = _state["first_at"] or now
        _state["last_at"] = now
        _state["last"] = msg
        c = _state["components"].setdefault(
            component, {"events": 0, "last_at": None, "last": ""})
        c["events"] += 1
        c["last_at"] = now
        c["last"] = msg
    return True


def status() -> dict:
    """Everything known, for a diagnostic endpoint or a test. Copies."""
    now = time.time()
    with _lock:
        out = {
            "events": _state["events"],
            "first_at": _state["first_at"],
            "last_at": _state["last_at"],
            "last": _state["last"],
            "recent": bool(_state["last_at"]
                           and (now - _state["last_at"]) <= RECENT_S),
            "components": {k: dict(v) for k, v in _state["components"].items()},
        }
    if out["last_at"]:
        out["ago_s"] = round(now - out["last_at"], 1)
    return out


def degraded_entry():
    """The /health row, or None if this card has never refused an allocation.

    ONE ROW, NOT ONE PER SITE. Everything below is a symptom of a single fact
    -- this pipeline does not fit on this card -- and a health list with four
    rows saying that in four vocabularies is a list nobody reads to the end.
    """
    st = status()
    if not st["events"]:
        return None
    names = sorted(st["components"], key=lambda k: -st["components"][k]["events"])
    costs = [f"{n} — {_COST.get(n, 'that path is failing')}" for n in names]
    # Built outside the f-string below: this file has to import on 3.11, where
    # a nested same-quote inside an f-string is a syntax error.
    counts = ", ".join(f"{n}x{st['components'][n]['events']}" for n in names)
    when = (f"most recently {st['ago_s']:.0f} s ago"
            if st.get("ago_s") is not None else "")
    return {
        "component": "gpu_memory",
        "detail": (
            f"the GPU has refused {st['events']} allocation(s) ({counts}), "
            f"{when}. This card is too small for this pipeline, and the "
            f"failures are caught so the drive continues — degraded, not "
            f"stopped: " + "; ".join(costs) + f". Last: {st['last']}"),
        # The measurement, not a guess -- and the tool that produced it, so the
        # next card can be checked before it is rented rather than after.
        "fix": ("python -m tools.vram_budget   # measured peak 19049 MiB under "
                "a full drive-like load, so 24 GB is the floor and 18 GB is "
                "this"),
    }


def reset() -> None:
    """Forget everything. For tests, and for nothing else."""
    with _lock:
        _state.update({"events": 0, "first_at": None, "last_at": None,
                       "last": "", "components": {}})
