#!/usr/bin/env python3
"""accel_verify.py — does the compiled detector see the same road?

    python3 tools/accel_verify.py                 # verify, then benchmark
    python3 tools/accel_verify.py --frames 400
    python3 tools/accel_verify.py --profile-all   # which models are worth it

WHY THIS EXISTS. headway/detect.py can run its model through
torch.compile(mode="reduce-overhead"), which is CUDA graphs plus Inductor
codegen, and that is worth 3.1x on this hardware. It is also a DIFFERENT
IMPLEMENTATION of the same arithmetic: different kernels, different reduction
orders, fp16 throughout. The compiled model does not return the same bits.

    pred_logits   max |delta| 4.21
    pred_boxes    max |delta| 1.01

Those numbers look alarming and are mostly meaningless -- pred_logits is 300
queries x 91 classes and almost all of it is junk queries at large negative
logits, where a delta of 4 between -12 and -16 changes nothing. But "mostly
meaningless" is not a thing to take on trust in a system that decides when to
warn a driver, so this measures the only output that matters: THE DETECTIONS
THE REST OF THE SYSTEM SEES, after the confidence gate, the size floor, the
ego-structure gate and duplicate suppression.

WHAT IT RUNS ON, and the honest caveat. The first real drive's own frames are
not recoverable: the session log keeps boxes, not pictures (docs/navigation_v1
§31 is the same principle, and framebuf is RAM-only). So the corpus is every
real-road clip in the repo, at the frame sizes the drive actually used --
1280x720 as the clips are shot, and 640x480 and 480x640, which are the
landscape and portrait shapes the phone sent on 2026-09-08. Model input is
384x384 for all of them; what the frame size changes is the postprocess, which
scales boxes back into frame pixels, so a box delta means something different
at each one and all three are reported.

THE VERDICT IS THE POINT. This prints PASS or FAIL against the tolerance in
config.py, and headway/detect.py runs the same comparison at warm-up before it
will use the compiled model at all. A tolerance that is not enforced at runtime
is a paragraph, not a guardrail.
"""
import argparse
import json
import os
import statistics
import sys
import time
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO))

import cv2  # noqa: E402
import numpy as np  # noqa: E402
import torch  # noqa: E402

import config  # noqa: E402
from headway import detect as D  # noqa: E402

# Every real road this checkout has. Deliberately not synthetic: a compiled
# kernel that agrees on a grey rectangle proves nothing.
CLIPS = [
    REPO / "runs" / "road_clip.mp4",
    REPO / "runs" / "night_clip.webm",
    Path("/workspace/ufldv2/example.mp4"),
]

# The shapes the first real drive actually sent, plus the clips' own. See the
# module docstring.
SIZES = [None, (640, 480), (480, 640)]


def frames(limit):
    """Real road frames, spread across every clip and every frame size."""
    out = []
    per_clip = max(1, limit // (len([c for c in CLIPS if c.exists()]) or 1))
    for clip in CLIPS:
        if not clip.exists():
            continue
        cap = cv2.VideoCapture(str(clip))
        n = int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) or 300
        step = max(1, n // max(1, per_clip // len(SIZES)))
        i = 0
        taken = 0
        while taken < per_clip:
            cap.set(cv2.CAP_PROP_POS_FRAMES, i)
            ok, f = cap.read()
            if not ok:
                break
            for size in SIZES:
                g = f if size is None else cv2.resize(f, size,
                                                      interpolation=cv2.INTER_AREA)
                out.append((f"{clip.name}@{i}", g.shape[1], g.shape[0], g))
                taken += 1
            i += step
            if i >= n:
                break
        cap.release()
    return out[:limit] if limit else out


def detections(frame):
    """Everything a consumer of detect() receives, after every gate.

    The `confirmed` flag rides along because it is the size floor -- whether a
    range may be claimed for a box at all -- and a compiled model that moved a
    box across it would be changing behaviour with every other number looking
    fine.
    """
    r = D.detect(frame)
    return [(d[0], tuple(float(v) for v in d[1]), float(d[2]),
             bool((d[3] or {}).get("confirmed", True)) if len(d) > 3 else True)
            for d in r["detections"]]


def compare(a, b, frame_w):
    """One frame's eager detections against the compiled ones.

    Box deltas as a FRACTION OF FRAME WIDTH: the model emits normalised
    coordinates and the postprocess multiplies by the frame size, so pixels are
    the wrong unit and a large frame would be charged for arithmetic it did not
    do. See config.HEADWAY_ACCEL_MAX_BOX_DELTA_FRAC.
    """
    out = {"n_eager": len(a), "n_accel": len(b), "same_count": len(a) == len(b),
           "label_mismatch": 0, "confirm_flip": 0, "tiebreak": 0,
           "max_box_frac": 0.0, "max_box_px": 0.0, "max_of_box": 0.0,
           "max_score": 0.0}
    if len(a) != len(b):
        return out
    for (la, ba, sa, ca), (lb, bb, sb, cb) in zip(a, b):
        if la != lb:
            # A tie-break between two competing readings of one object, at a
            # margin the eager model does not hold stable either. See
            # headway/detect._arbitrary_tie for the measurement.
            if D._arbitrary_tie(la, lb, sa, sb):
                out["tiebreak"] += 1
            else:
                out["label_mismatch"] += 1
        if ca != cb:
            out["confirm_flip"] += 1
        px = max(abs(p - q) for p, q in zip(ba, bb))
        side = max(1e-6, min(ba[2] - ba[0], ba[3] - ba[1]))
        out["max_box_px"] = max(out["max_box_px"], px)
        out["max_box_frac"] = max(out["max_box_frac"], px / max(frame_w, 1.0))
        out["max_of_box"] = max(out["max_of_box"], px / side)
        out["max_score"] = max(out["max_score"], abs(sa - sb))
    return out


def pct(v, q):
    if not v:
        return None
    v = sorted(v)
    return v[min(len(v) - 1, int(round((len(v) - 1) * q)))]


def kernel_floor(fn, x, n=60):
    """The contention-robust cost of one forward. See headway/live_selftest.py."""
    with torch.inference_mode():
        for _ in range(15):
            fn(x)
        torch.cuda.synchronize()
        v = []
        for _ in range(n):
            e0 = torch.cuda.Event(enable_timing=True)
            e1 = torch.cuda.Event(enable_timing=True)
            e0.record()
            fn(x)
            e1.record()
            torch.cuda.synchronize()
            v.append(e0.elapsed_time(e1))
    return pct(v, 0.10), pct(v, 0.50)


def launch_bound_ratio(tag, prep, call, n=20):
    """elapsed / GPU-work for one model. Above ~2 it is waiting to be fed.

    THIS IS THE TEST THAT DECIDES WHETHER A MODEL IS WORTH COMPILING AT ALL.
    CUDA graphs remove launch overhead and nothing else, so a model whose GPU
    is already saturated has nothing to gain from them -- and would be paying
    a numeric tolerance for it.
    """
    from torch.profiler import profile, ProfilerActivity
    with torch.inference_mode():
        x = prep()
        for _ in range(10):
            call(x)
        torch.cuda.synchronize()
        e0 = torch.cuda.Event(enable_timing=True)
        e1 = torch.cuda.Event(enable_timing=True)
        e0.record()
        for _ in range(n):
            call(x)
        e1.record()
        torch.cuda.synchronize()
        elapsed = e0.elapsed_time(e1) / n
        with profile(activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA]) as prof:
            for _ in range(n):
                call(x)
            torch.cuda.synchronize()
    ka = prof.key_averages()
    gpu = sum(e.self_device_time_total for e in ka) / 1000.0 / n
    cpu = sum(e.self_cpu_time_total for e in ka) / 1000.0 / n
    calls = sum(e.count for e in ka if e.key.startswith("aten::")) / n
    ratio = elapsed / max(gpu, 1e-9)
    print(f"  {tag:<8} elapsed {elapsed:6.2f} ms   GPU work {gpu:5.2f} ms   "
          f"CPU {cpu:6.2f} ms   {calls:5.0f} aten calls   "
          f"elapsed/GPU {ratio:4.1f}x   {'LAUNCH BOUND' if ratio > 2 else 'gpu bound'}")
    return ratio


def profile_all():
    """Which of the three vision models is worth compiling. Measured, not assumed."""
    import cv2 as _cv2
    print("\n=== launch-bound ratio: which models would CUDA graphs help ===")
    frame = None
    for c in CLIPS:
        if c.exists():
            cap = _cv2.VideoCapture(str(c))
            cap.set(_cv2.CAP_PROP_POS_FRAMES, 100)
            ok, frame = cap.read()
            cap.release()
            if ok:
                break
    if frame is None:
        print("  no clip to profile on")
        return
    D._ensure_loaded()
    with torch.inference_mode():
        launch_bound_ratio("detect", lambda: D._preprocess(frame),
                           lambda x: D._model(x))
    try:
        from headway import depth as DE
        DE._ensure_loaded()
        with torch.inference_mode():
            t = torch.from_numpy(frame).to("cuda")
            t = t[:, :, [2, 1, 0]].permute(2, 0, 1).contiguous()
            pv = DE._processor(images=t, return_tensors="pt",
                               device="cuda")["pixel_values"].half()
            launch_bound_ratio("depth", lambda: pv,
                               lambda x: DE._model(pixel_values=x))
    except Exception as e:
        print(f"  depth: {type(e).__name__}: {e}")
    try:
        from headway import lanes as LN
        LN._ensure_loaded()
        with torch.inference_mode():
            lx = LN._preprocess(frame)
            launch_bound_ratio("lanes", lambda: lx, lambda x: LN._net(x))
    except Exception as e:
        print(f"  lanes: {type(e).__name__}: {e}")
    print("  Only a LAUNCH BOUND model is worth a compile: CUDA graphs remove "
          "launch overhead\n  and nothing else, and the compile is paid for "
          "with a numeric tolerance.")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--frames", type=int, default=180)
    ap.add_argument("--profile-all", action="store_true",
                    help="which of detect/depth/lanes is launch bound, and by "
                         "how much — the measurement that decided the scope")
    ap.add_argument("--json", default=None)
    args = ap.parse_args()

    if args.profile_all:
        if not torch.cuda.is_available():
            print("no CUDA")
            return 0
        profile_all()
        return 0

    if not torch.cuda.is_available():
        print("no CUDA — acceleration is not applicable here")
        return 0
    D._ensure_loaded()
    print(f"GPU {torch.cuda.get_device_name(0)}  torch {torch.__version__}")
    print(f"model input {D._resolution}x{D._resolution} {D._dtype}")

    corpus = frames(args.frames)
    print(f"corpus: {len(corpus)} real-road frames from "
          f"{len({c.split('@')[0] for c, _, _, _ in corpus})} clips at "
          f"{sorted({(w, h) for _, w, h, _ in corpus})}")

    report = {"n_frames": len(corpus)}

    # ---- eager, first and complete -------------------------------------
    # All of it before anything is compiled: interleaving the two would have
    # the CUDA graph pool live alongside eager allocations for no reason, and
    # the reference has to be the reference.
    D.set_accel(False)
    t0 = time.time()
    eager = [detections(f) for _, _, _, f in corpus]
    print(f"eager pass: {time.time() - t0:.1f}s, "
          f"{sum(len(d) for d in eager)} detections")

    x = D._preprocess(corpus[0][3])
    e_floor, e_p50 = kernel_floor(lambda t: D._model(t), x)

    # ---- and now the compiled one --------------------------------------
    t0 = time.time()
    ok = D.set_accel(True)
    if not ok:
        print("\nFAIL — the model would not compile at all. "
              f"detect.accel_status(): {D.accel_status()}")
        return 1
    print(f"compiled in {time.time() - t0:.1f}s")
    accel = [detections(f) for _, _, _, f in corpus]

    a_floor, a_p50 = kernel_floor(lambda t: D._accel_model(t), x)

    # ---- the comparison -------------------------------------------------
    rows = [compare(a, b, c[1]) for a, b, c in zip(eager, accel, corpus)]
    diff_count = [i for i, r in enumerate(rows) if not r["same_count"]]
    labels_off = [i for i, r in enumerate(rows) if r["label_mismatch"]]
    flips = [i for i, r in enumerate(rows) if r["confirm_flip"]]
    n_ties = sum(r["tiebreak"] for r in rows)
    boxes = [r["max_box_frac"] for r in rows if r["same_count"]]
    boxes_px = [r["max_box_px"] for r in rows if r["same_count"]]
    of_box = [r["max_of_box"] for r in rows if r["same_count"]]
    scores = [r["max_score"] for r in rows if r["same_count"]]

    print("\n=== detections: eager vs compiled ===")
    print(f"  frames compared            {len(rows)}")
    print(f"  detections (eager/accel)   {sum(len(d) for d in eager)} / "
          f"{sum(len(d) for d in accel)}")
    print(f"  frames with a DIFFERENT number of detections   {len(diff_count)}")
    print(f"  frames with a label mismatch                   {len(labels_off)}")
    print(f"  frames where `confirmed` flipped               {len(flips)}")
    print(f"  exclusive-label tie-breaks (allowed, counted)  {n_ties}"
          + ("   — eager itself flips these on a one-grey-level change"
             if n_ties else ""))
    if boxes:
        print(f"  box delta /frameW  p50 {pct(boxes,.5):.6f}  "
              f"p95 {pct(boxes,.95):.6f}  max {max(boxes):.6f}")
        print(f"  box delta px       p50 {pct(boxes_px,.5):.4f}  "
              f"p95 {pct(boxes_px,.95):.4f}  max {max(boxes_px):.4f}  "
              f"(the unit the tolerance is NOT in — see config)")
        print(f"  box delta /box     p50 {pct(of_box,.5):.6f}  "
              f"p95 {pct(of_box,.95):.6f}  max {max(of_box):.6f}  "
              f"(of the box's smaller side — the behaviour bound)")
        print(f"  score delta        p50 {pct(scores,.5):.6f}  "
              f"p95 {pct(scores,.95):.6f}  max {max(scores):.6f}")
        # Per frame size, because the whole point of the fractional unit is
        # that the pixel number is not comparable across them.
        import collections as _c
        by = _c.defaultdict(list)
        for r, c in zip(rows, corpus):
            if r["same_count"]:
                by[(c[1], c[2])].append((r["max_box_px"], r["max_box_frac"]))
        for (w, h), v in sorted(by.items()):
            print(f"    {w:4d}x{h:<4d} n={len(v):3d}  max {max(x[0] for x in v):.4f} px"
                  f"  = {max(x[1] for x in v):.6f} of frame width")
    for i in diff_count[:6]:
        print(f"    differing frame {corpus[i][0]} {corpus[i][1]}x{corpus[i][2]}: "
              f"eager {rows[i]['n_eager']} vs accel {rows[i]['n_accel']}")

    report.update({
        "frames_diff_count": len(diff_count),
        "frames_label_mismatch": len(labels_off),
        "frames_confirm_flip": len(flips),
        "exclusive_label_tiebreaks": n_ties,
        "box_delta_frac_max": round(max(boxes), 7) if boxes else None,
        "box_delta_px_max": round(max(boxes_px), 5) if boxes_px else None,
        "box_delta_of_box_max": round(max(of_box), 5) if of_box else None,
        "score_delta_max": round(max(scores), 6) if scores else None,
        "eager_floor_ms": round(e_floor, 2), "accel_floor_ms": round(a_floor, 2),
        "speedup": round(e_floor / a_floor, 2) if a_floor else None,
    })

    print("\n=== cost ===")
    print(f"  eager     floor {e_floor:.2f}  p50 {e_p50:.2f} ms")
    print(f"  compiled  floor {a_floor:.2f}  p50 {a_p50:.2f} ms")
    print(f"  speedup   {e_floor / a_floor:.2f}x")

    # ---- the verdict ----------------------------------------------------
    tol_box = config.HEADWAY_ACCEL_MAX_BOX_DELTA_FRAC
    tol_score = config.HEADWAY_ACCEL_MAX_SCORE_DELTA
    fails = []
    if diff_count:
        fails.append(f"{len(diff_count)} frames returned a different NUMBER of "
                     f"detections — no tolerance covers that")
    if labels_off:
        fails.append(f"{len(labels_off)} frames disagreed about a label")
    if flips:
        fails.append(f"{len(flips)} frames flipped the `confirmed` flag — that "
                     f"is the size floor, and it has no tolerance either")
    if boxes and max(boxes) > tol_box:
        fails.append(f"box delta {max(boxes):.6f} of frame width over the "
                     f"{tol_box} tolerance")
    if of_box and max(of_box) > config.HEADWAY_ACCEL_MAX_BOX_DELTA_OF_BOX:
        fails.append(f"box delta {max(of_box):.4f} of its own smaller side over "
                     f"the {config.HEADWAY_ACCEL_MAX_BOX_DELTA_OF_BOX} tolerance")
    if scores and max(scores) > tol_score:
        fails.append(f"score delta {max(scores):.6f} over the {tol_score} tolerance")

    print("\n=== verdict ===")
    if fails:
        for f in fails:
            print(f"  FAIL  {f}")
        print("  The compiled detector does not see the same road. "
              "config.HEADWAY_DETECT_ACCEL must not default on.")
    else:
        print(f"  PASS  same detections, same labels (bar {n_ties} exclusive "
              f"tie-break{'' if n_ties == 1 else 's'}) and the same "
              f"`confirmed` on all {len(rows)} frames. Boxes within "
              f"{max(boxes) if boxes else 0:.6f} of frame width "
              f"(tolerance {tol_box}) and {max(of_box) if of_box else 0:.4f} of "
              f"their own smaller side (tolerance "
              f"{config.HEADWAY_ACCEL_MAX_BOX_DELTA_OF_BOX}), scores within "
              f"{max(scores) if scores else 0:.6f} (tolerance {tol_score})")

    if args.json:
        Path(args.json).write_text(json.dumps(report, indent=2))
        print(f"\nwrote {args.json}")
    return 1 if fails else 0


if __name__ == "__main__":
    sys.exit(main())
