"""VRAM and latency, per teacher, per precision — the table the L40S needs.

    python -m tools.teacher_bench --model alpamayo --precision bf16
    python -m tools.teacher_bench --all --report /tmp/teachers.md

WHY THIS TOOL RATHER THAN A NUMBER IN A README
----------------------------------------------
The pod this was built on is an H200 with 141 GB. The pod it has to live on is
an L40S with 48 GB, which already holds Qwen3-VL-8B, Depth-Anything, UFLDv2 and
RF-DETR. Whether the teacher panel can exist there at all is a question about
four numbers -- VRAM and latency, for each of two models, at each of two
precisions -- and every one of them has to be MEASURED on the same frames,
through the same service, at the same generation settings, or the comparison
is worthless.

So this starts a service, warms it, feeds it the same synthetic keyframe N
times, and reports:

  vram_reserved_mb    what the process is HOLDING, not what it allocated at
                      peak. The allocator keeps freed blocks, and it is the
                      held figure that decides whether a fifth model fits.
  latency             per inference, p50 and p90, over N runs after a warm-up
                      run that is thrown away -- the first call through a
                      transformers generate() compiles kernels and is not a
                      number anyone will ever see again.

THE FIRST RUN IS ALWAYS DISCARDED, and the count defaults to 5 rather than 1,
because a single sample of a sampled generation is not a latency.
"""
import argparse
import base64
import json
import os
import subprocess
import sys
import time
import urllib.error
import urllib.request

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import config                                                     # noqa: E402
from teachers import paths                                       # noqa: E402

VENVS = paths.VENVS
MODULES = {
    "alpamayo": "teachers.service.alpamayo_service",
    "cosmos": "teachers.service.cosmos_service",
}
PORTS = {"alpamayo": 8801, "cosmos": 8802}
# Where the FP8 checkpoints are written by teachers/service/quantize.py.
FP8_PATHS = {
    "alpamayo": os.path.join(paths.FP8_DIR, "alpamayo_fp8"),
    "cosmos": os.path.join(paths.FP8_DIR, "model_fp8"),
}
REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# The pod this has to live on, and what is already on it. Stated here so the
# table's verdict is arithmetic anyone can check rather than a judgement.
L40S_GB = 48
RIO_GB = 18


def make_keyframe(n=4, w=640, h=480):
    """One synthetic keyframe, identical for every run of every model.

    A road, lane lines and a vehicle that grows across the four frames -- so
    there is something to describe, something to call the critical actor, and
    real motion between the frames rather than four copies of one picture.
    """
    import cv2
    import numpy as np

    frames = []
    for i in range(n):
        a = np.zeros((h, w, 3), np.uint8)
        a[: h // 2] = (170, 155, 140)
        a[h // 2:] = (62, 62, 68)
        cv2.line(a, (w // 2, h // 2 + 10), (30, h), (235, 235, 235), 7)
        cv2.line(a, (w // 2, h // 2 + 10), (w - 30, h), (235, 235, 235), 7)
        s = 46 + i * 5
        cx, cy = w // 2 + 14, h // 2 + 86
        cv2.rectangle(a, (cx - s, cy - s // 2), (cx + s, cy + s // 2),
                      (48, 48, 172), -1)
        cv2.rectangle(a, (cx - s, cy - s // 2), (cx + s, cy + s // 2),
                      (20, 20, 90), 3)
        frames.append(cv2.imencode(".jpg", a)[1].tobytes())

    # 1.6 s of history at 13 m/s, straight: the shape teachers/egomotion.py
    # produces, written out literally so this tool needs no session.
    xyz, yaw = [], []
    for i in range(16):
        xyz.append([round(-(15 - i) * 1.3, 4), 0.0, 0.0])
        yaw.append(0.0)
    return {
        "kf_id": "bench-00000",
        "t0": time.time(),
        "frames": [base64.b64encode(b).decode() for b in frames],
        "frame_offsets_s": [-0.3, -0.2, -0.1, 0.0],
        "ego_history_xyz": xyz,
        "ego_history_yaw": yaw,
        "prompts": dict(config.TEACHER_PROMPTS),
        "physical_prompt": config.TEACHER_COSMOS_PHYSICAL_PROMPT,
        "image": {"w": w, "h": h},
    }


def wait_health(url, timeout_s=900):
    """Block until the service says loaded, or give up. -> the health dict."""
    deadline = time.time() + timeout_s
    last = {}
    while time.time() < deadline:
        try:
            with urllib.request.urlopen(url + "/health", timeout=10) as r:
                last = json.loads(r.read().decode())
            if last.get("loaded"):
                return last
            if last.get("load_error"):
                return last
        except Exception:
            pass
        time.sleep(3)
    return last or {"ok": False, "error": "timeout waiting for /health"}


def start_service(model, precision, port, weights=None, extra=None):
    """Launch one service in its own venv. -> (Popen, url)."""
    py = VENVS[model]
    if not os.path.exists(py):
        raise SystemExit(f"{py} is missing — run: bash boot.sh teachers")
    cmd = [py, "-m", MODULES[model], "--port", str(port),
           "--precision", precision]
    if weights:
        cmd += ["--model", weights]
    cmd += list(extra or [])
    # Paths from code, never from the caller's shell — see teachers/paths.py.
    env = paths.load_secrets(paths.subprocess_env())
    os.makedirs(paths.LOG_DIR, exist_ok=True)
    log = open(os.path.join(paths.LOG_DIR,
                            f"bench_{model}_{precision}.log"), "w")
    print("+ " + " ".join(cmd), flush=True)
    proc = subprocess.Popen(cmd, cwd=REPO, env=env, stdout=log, stderr=log,
                            start_new_session=True)
    return proc, f"http://127.0.0.1:{port}"


def bench_one(model, precision, runs, port=None, weights=None, keep=False):
    port = port or (PORTS[model] + 20)
    payload = make_keyframe()
    proc, url = start_service(model, precision, port, weights)
    out = {"model": model, "precision": precision, "runs": [],
           "weights": weights or "hub default"}
    try:
        t0 = time.time()
        health = wait_health(url)
        out["load_s"] = round(time.time() - t0, 1)
        out["health"] = health
        if not health.get("loaded"):
            out["error"] = health.get("load_error") or health.get("error") \
                or "did not load"
            return out
        out["weights_mb"] = health.get("weights_mb")
        out["vram_after_load_mb"] = health.get("vram_now_mb")
        out["model_id"] = health.get("model_id")
        out["revision"] = health.get("revision")

        for i in range(runs + 1):
            body = json.dumps(payload).encode()
            req = urllib.request.Request(
                url + "/infer", data=body,
                headers={"Content-Type": "application/json"}, method="POST")
            t = time.time()
            try:
                with urllib.request.urlopen(req, timeout=900) as r:
                    res = json.loads(r.read().decode())
            except Exception as e:
                out.setdefault("errors", []).append(f"{type(e).__name__}: {e}")
                continue
            wall = round((time.time() - t) * 1000.0, 1)
            rec = {"i": i, "ok": res.get("ok"), "wall_ms": wall,
                   "latency_ms": res.get("latency_ms"),
                   "timings_ms": res.get("timings_ms"),
                   "vram_reserved_mb": (res.get("gpu") or {}).get("vram_reserved_mb"),
                   "vram_now_mb": (res.get("gpu") or {}).get("vram_now_mb"),
                   "vram_peak_mb": (res.get("gpu") or {}).get("vram_peak_mb"),
                   "error": res.get("error")}
            if i == 0:
                # THROWN AWAY. The first generate() through transformers
                # compiles kernels and allocates the KV cache; it is a number
                # nobody will see twice and averaging it in would flatter FP8
                # or punish it depending on the order the runs happened to go.
                rec["warmup"] = True
                out["warmup"] = rec
                out["sample"] = {k: res.get(k) for k in
                                 ("scene", "critical_actor", "attention",
                                  "reasoning")}
            else:
                out["runs"].append(rec)
            print(f"    [{model}/{precision}] run {i}: {wall} ms wall, "
                  f"{rec['latency_ms']} ms model, "
                  f"{rec['vram_reserved_mb']} MB", flush=True)
    finally:
        if not keep:
            proc.terminate()
            try:
                proc.wait(timeout=30)
            except subprocess.TimeoutExpired:
                proc.kill()
    return summarise(out)


def summarise(out):
    lat = [r["latency_ms"] for r in out["runs"]
           if r.get("ok") and r.get("latency_ms")]
    vram = [r["vram_reserved_mb"] for r in out["runs"] if r.get("vram_reserved_mb")]
    if lat:
        s = sorted(lat)
        out["latency_p50_ms"] = s[len(s) // 2]
        out["latency_p90_ms"] = s[min(len(s) - 1, int(0.9 * (len(s) - 1) + 0.5))]
        out["latency_min_ms"] = s[0]
        out["latency_max_ms"] = s[-1]
    if vram:
        out["vram_reserved_mb"] = max(vram)
    return out


def render(results):
    """The table, in Markdown, because that is what a report needs."""
    lines = [
        "| model | precision | weights | held between keyframes | "
        "peak in inference | latency p50 | p90 | load |",
        "| --- | --- | --- | --- | --- | --- | --- | --- |",
    ]
    for r in results:
        if r.get("error"):
            lines.append(f"| {r['model']} | {r['precision']} | — | — | — | — | "
                         f"FAILED: {str(r['error'])[:60]} |")
            continue
        lines.append(
            f"| {r['model']} | {r['precision']} | "
            f"{r.get('weights_mb', '—')} MB | "
            f"{r.get('vram_after_load_mb', '—')} MB | "
            f"{r.get('vram_reserved_mb', '—')} MB | "
            f"{r.get('latency_p50_ms', '—')} ms | "
            f"{r.get('latency_p90_ms', '—')} ms | "
            f"{r.get('load_s', '—')} s |")
    total, weights = {}, {}
    for r in results:
        p = r["precision"]
        if r.get("vram_reserved_mb"):
            total[p] = total.get(p, 0) + r["vram_reserved_mb"]
        if r.get("weights_mb"):
            weights[p] = weights.get(p, 0) + r["weights_mb"]
    for p in sorted(total):
        lines.append(f"| **both** | **{p}** | **{round(weights.get(p, 0))} MB** "
                     f"| | **{round(total[p])} MB** | | | |")

    # THE QUESTION THIS TOOL EXISTS TO ANSWER, answered rather than left as an
    # exercise. A budget is a peak, not a weight: a model that holds 10 GB and
    # balloons to 20 GB during inference needs 20 GB of card.
    lines += ["", f"**Does it fit on an L40S?** {L40S_GB} GB, of which RIO's "
                  f"live stack (Qwen3-VL-8B, Depth-Anything, UFLDv2, RF-DETR) "
                  f"holds about {RIO_GB} GB — so roughly "
                  f"{L40S_GB - RIO_GB} GB is available.", ""]
    budget_mb = (L40S_GB - RIO_GB) * 1024
    for p in sorted(total):
        fits = total[p] <= budget_mb
        lines.append(f"- both at **{p}**: {round(total[p])} MB peak — "
                     f"{'fits' if fits else 'DOES NOT FIT'}")
    for r in results:
        v = r.get("vram_reserved_mb")
        if not v:
            continue
        lines.append(f"- {r['model']} alone at **{r['precision']}**: "
                     f"{round(v)} MB peak — "
                     f"{'fits' if v <= budget_mb else 'DOES NOT FIT'}")
    return "\n".join(lines)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", choices=("alpamayo", "cosmos"))
    ap.add_argument("--precision", choices=("bf16", "fp8"), default="bf16")
    ap.add_argument("--all", action="store_true",
                    help="both models at both precisions, one at a time")
    ap.add_argument("--runs", type=int, default=5)
    ap.add_argument("--weights", default=None,
                    help="local checkpoint path (an FP8 build)")
    ap.add_argument("--report", default=None, help="write a Markdown table here")
    ap.add_argument("--json", default=None, help="write the raw results here")
    args = ap.parse_args()

    os.makedirs(paths.LOG_DIR, exist_ok=True)
    jobs = []
    if args.all:
        # ONE AT A TIME, deliberately: two 20 GB models loading at once on a
        # 48 GB card is the thing this table exists to find out about, not a
        # thing to do while finding it out.
        for m in ("alpamayo", "cosmos"):
            for p in ("bf16", "fp8"):
                jobs.append((m, p, FP8_PATHS[m] if p == "fp8" else None))
    else:
        if not args.model:
            ap.error("--model or --all")
        jobs.append((args.model, args.precision,
                     args.weights or (FP8_PATHS[args.model]
                                      if args.precision == "fp8" else None)))

    results = []
    for model, precision, weights in jobs:
        if weights and not os.path.exists(weights):
            print(f"!! {model}/{precision}: {weights} does not exist — build it "
                  f"with:\n   {VENVS[model]} -m teachers.service.quantize "
                  f"--model {model}", flush=True)
            results.append({"model": model, "precision": precision,
                            "error": f"no checkpoint at {weights}"})
            continue
        results.append(bench_one(model, precision, args.runs, weights=weights))

    table = render(results)
    print("\n" + table)
    if args.report:
        with open(args.report, "w") as f:
            f.write(table + "\n")
        print(f"\nwrote {args.report}")
    if args.json:
        with open(args.json, "w") as f:
            json.dump(results, f, indent=2)
        print(f"wrote {args.json}")
    return 0 if all(not r.get("error") for r in results) else 1


if __name__ == "__main__":
    sys.exit(main())
