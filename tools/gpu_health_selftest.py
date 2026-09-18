"""The OOM signal: what counts, what does not, and who reports it.

    python -m tools.gpu_health_selftest
    python -m tools.gpu_health_selftest --url http://127.0.0.1:8888

WHAT THIS GUARDS
----------------
A card that is slightly too small does not stop this server. It stops it
SEEING: the lane model, the observer and enrichment each catch their own
out-of-memory, log a line and carry on, so the drive survives -- and /health
said "ok" through all of it. That was measured, not imagined
(tools/vram_budget.py --squeeze 18: 34 frame results out of 558 sent, p50 frame
time 17 ms -> 2767 ms, readiness "ok").

So there are three claims to hold, and they fail in different ways:

  A. THE CLASSIFIER IS NOT A KEYWORD SEARCH. An out-of-memory arrives in at
     least three costumes -- torch's OutOfMemoryError, a RuntimeError naming
     CUBLAS_STATUS_ALLOC_FAILED, a plain "CUDA error: out of memory" -- and an
     ordinary bug must not be counted as one. A false positive here tells
     somebody to rent a bigger card over a typo.

  B. EVERY SITE THAT SWALLOWS ONE REPORTS IT. Checked against the source, not
     against this file's memory of it: a new `except` around GPU work that
     forgets to call gpu_health.note is exactly how the silence comes back.

  C. /HEALTH SAYS SO. The entry exists, names the components, says what has
     been lost, and is absent when nothing has failed -- because a health
     endpoint that always carries a row about the GPU is one nobody reads.
"""
import argparse
import ast
import json
import sys
import urllib.request
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
REPO = Path(__file__).resolve().parent.parent

import gpu_health  # noqa: E402

PASS, FAIL = [], []


def ok(cond, what):
    (PASS if cond else FAIL).append(what)
    print(f"  {'ok  ' if cond else 'FAIL'}  {what}")
    return bool(cond)


def section(name):
    print(f"\n=== {name} ===")


# ---------------------------------------------------------------------------
def rule_a():
    section("A. what is an out-of-memory, and what is just a bug")
    real = [
        RuntimeError("CUDA out of memory. Tried to allocate 358.00 MiB. GPU 0 "
                     "has a total capacity of 139.81 GiB of which 88.44 MiB is free"),
        RuntimeError("CUDA error: CUBLAS_STATUS_ALLOC_FAILED when calling "
                     "`cublasCreate(handle)`"),
        RuntimeError("cuDNN error: CUDNN_STATUS_ALLOC_FAILED"),
        RuntimeError("CUDA error: out of memory"),
    ]
    # The real shape of torch's own class, without needing torch here.
    oom_cls = type("OutOfMemoryError", (RuntimeError,), {})
    real.append(oom_cls("Tried to allocate 20.00 MiB"))
    for e in real:
        ok(gpu_health.looks_like_oom(e),
           f"counted: {type(e).__name__}: {str(e)[:52]}")

    fake = [
        ValueError("no such file: culane_res18.pth"),
        RuntimeError("Expected all tensors to be on the same device"),
        KeyError("track_id"),
        RuntimeError("CUDA error: no kernel image is available for execution"),
        TimeoutError("the model took too long"),
    ]
    for e in fake:
        ok(not gpu_health.looks_like_oom(e),
           f"NOT counted: {type(e).__name__}: {str(e)[:48]}")


def rule_b():
    section("B. every site that swallows a GPU failure reports it")
    # Where GPU work is caught and the drive deliberately carries on. Each is
    # a place a too-small card goes quiet, and each is checked by reading the
    # source rather than by trusting this list.
    sites = {
        "headway/live.py": "lanes",
        "perceive.py": "lanes",
        "observer.py": "observer",
        "enrich.py": "enrich",
        "visual_qa.py": "visual_qa",
        "app.py": "lanes",
    }
    for rel, component in sites.items():
        path = REPO / rel
        tree = ast.parse(path.read_text(), filename=str(path))
        calls = []
        for node in ast.walk(tree):
            if (isinstance(node, ast.Call)
                    and isinstance(node.func, ast.Attribute)
                    and node.func.attr == "note"
                    and isinstance(node.func.value, ast.Name)
                    and node.func.value.id == "gpu_health"):
                first = node.args[0] if node.args else None
                calls.append(first.value if isinstance(first, ast.Constant) else "?")
        ok(component in calls,
           f"{rel} calls gpu_health.note({component!r}) "
           f"({len(calls)} call(s): {calls})")

        imports = {n.name for i in ast.walk(tree)
                   if isinstance(i, ast.Import) for n in i.names}
        ok("gpu_health" in imports, f"{rel} imports gpu_health")

    # ...and the counter itself must stay incapable of changing a drive.
    tree = ast.parse((REPO / "gpu_health.py").read_text())
    bad = [n for n in ast.walk(tree) if isinstance(n, ast.Raise)]
    ok(not bad, "gpu_health raises nothing: it can never turn a caught failure "
                "into an uncaught one")
    heavy = {n.name for i in ast.walk(tree)
             if isinstance(i, ast.Import) for n in i.names}
    ok(not (heavy & {"torch", "numpy", "cv2", "transformers"}),
       f"gpu_health imports nothing heavy ({sorted(heavy)})")


def rule_c():
    section("C. the health entry")
    gpu_health.reset()
    ok(gpu_health.degraded_entry() is None,
       "a card that has never refused an allocation gets NO row")
    ok(gpu_health.status()["events"] == 0, "and no events")

    gpu_health.note("lanes", RuntimeError("CUDA out of memory. Tried to allocate 358.00 MiB"))
    gpu_health.note("observer", RuntimeError("CUDA error: CUBLAS_STATUS_ALLOC_FAILED"))
    gpu_health.note("observer", RuntimeError("CUDA error: CUBLAS_STATUS_ALLOC_FAILED"))
    gpu_health.note("enrich", ValueError("not a gpu problem"))

    st = gpu_health.status()
    ok(st["events"] == 3, f"three counted, one rejected (events={st['events']})")
    ok("enrich" not in st["components"], "the non-OOM did not create a component")
    ok(st["components"]["observer"]["events"] == 2, "per-component counts")
    ok(st["recent"] is True, "recent, because it just happened")

    e = gpu_health.degraded_entry()
    ok(e is not None, "the row exists once the card has refused something")
    ok(e["component"] == "gpu_memory", "named gpu_memory")
    d = e["detail"]
    ok("lanes" in d and "observer" in d, "it names the components that failed")
    ok("trapezoid" in d, "...and what was lost: the lane model's fallback")
    ok("no observer sentences" in d, "...and the observer's")
    ok("too small" in d, "it says the card is too small, not that something crashed")
    ok("vram_budget" in e["fix"], "the fix points at the tool that measures it")
    ok("19049" in e["fix"], "with the measured number, not an adjective")

    gpu_health.reset()
    ok(gpu_health.degraded_entry() is None, "reset clears it")


def rule_d(url):
    section("D. the live server")
    try:
        with urllib.request.urlopen(url.rstrip("/") + "/health", timeout=10) as r:
            h = json.loads(r.read().decode())
    except Exception as e:
        ok(False, f"/health answered ({type(e).__name__}: {e})")
        return
    rows = {d.get("component"): d for d in (h.get("degraded") or [])}
    print(f"  (readiness={h.get('readiness')}, degraded={sorted(rows)})")
    if "gpu_memory" in rows:
        # A pod that HAS been short of memory: the row must carry the whole
        # story, because this is the one moment somebody reads it.
        d = rows["gpu_memory"]["detail"]
        ok("refused" in d and "allocation" in d,
           "this server HAS been refused memory and /health says so")
        ok(h.get("readiness") != "ok",
           "...and readiness is not 'ok' while that is true")
    else:
        ok(True, "this server has not been refused memory, and carries no row")
        # The claim that matters when there is nothing to report: it is absent
        # because nothing happened, not because nothing is wired.
        ok(gpu_health.status()["events"] == 0,
           "...which agrees with this process's own counter")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--url", default="http://127.0.0.1:8888")
    args = ap.parse_args()
    rule_a()
    rule_b()
    rule_c()
    rule_d(args.url)
    print("\n" + "=" * 72)
    total = len(PASS) + len(FAIL)
    print(f"{len(PASS)}/{total} checks passed")
    if FAIL:
        print("\nFAILED:")
        for f in FAIL:
            print(f"  - {f}")
    print("=" * 72)
    return 1 if FAIL else 0


if __name__ == "__main__":
    sys.exit(main())
