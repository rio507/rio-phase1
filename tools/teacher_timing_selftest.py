"""Does the fast loop move when two teachers are running? — measured, not argued.

    python -m tools.teacher_timing_selftest              hook cost + real pipeline
    python -m tools.teacher_timing_selftest --gpu-load   ...with the real services

THE PROMISE BEING CHECKED
-------------------------
The headway loop is the only thing on this pod with a deadline. It runs at
8-15 fps, it is what a following-distance warning rests on, and every number in
it -- d_dot, tau, the Kalman filter's dt -- is computed against the interval
between frames. A pipeline that slows down does not produce late warnings; it
produces WRONG ones, because the velocity estimate is a difference over a dt
that has quietly changed.

So "the teacher panel is shadow" has to mean something stronger than "it does
not call the arbiter". It has to mean the loop cannot feel it. This suite is
where that claim is a number.

THREE PLACES IT COULD COST SOMETHING, AND ALL THREE ARE MEASURED
----------------------------------------------------------------
  1. THE HOOK. panel.on_frame() runs on the frame path. On most frames it is a
     few float comparisons; on a keyframe it also snapshots four object
     references and hands a job to another thread. Phase A measures both, over
     thousands of frames, and reports the tail rather than the mean -- a p50 of
     nothing with a p99 of 40 ms would still be a frame missed every few
     seconds.

  2. THE THREADS. Two client workers, a corpus writer and the GIL. Phase B runs
     the REAL headway pipeline -- RF-DETR, Depth-Anything, UFLDv2, the tracker,
     the filter -- over the same frames twice: once with the panel off, once
     with it on and both teachers saturated so the queue is always full and
     jobs are always being evicted. It compares the distributions.

  3. THE GPU. Two more models on the same card is the cost that cannot be
     reasoned away, and it is the one that matters on an L40S. Phase C
     (--gpu-load) repeats Phase B with the real services loaded and inferring,
     which is the only honest version of this measurement.

WHAT COUNTS AS UNCHANGED
------------------------
Not "identical" -- this is a GPU with a scheduler and a p99 that moves by a
millisecond between two runs of the same code. The gates are stated as
constants below, they are generous where the noise is and tight where it is
not, and a failure prints both distributions so the number can be argued with.
"""
import argparse
import os
import statistics
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import config                                                     # noqa: E402

config.TEACHERS_ENABLED = True

import framebuf                                                   # noqa: E402
from teachers import egomotion                                    # noqa: E402
from teachers import panel                                        # noqa: E402

PASS, FAIL = [], []

# --- the gates -------------------------------------------------------------
# The hook, on an ordinary frame. It is comparisons and a deque append; a
# millisecond is already two orders of magnitude more than it should need, and
# it is 0.4% of the 250 ms frame budget.
HOOK_P50_MS = 1.0
HOOK_P99_MS = 5.0
HOOK_MAX_MS = 25.0

# The pipeline, with the panel on against the panel off. Stated as a fraction
# rather than a millisecond count because the absolute number depends entirely
# on which GPU this is running on.
PIPE_P50_GROWTH = 0.06      # 6% at the median
PIPE_P95_GROWTH = 0.15      # 15% in the tail, where the scheduler lives
PIPE_ABS_MS = 12.0          # ...and never more than this many ms, either way


def ok(cond, what):
    (PASS if cond else FAIL).append(what)
    print(("  ok    " if cond else "  FAIL  ") + what)


def section(name):
    print(f"\n=== {name} ===")


def pct(v, q):
    if not v:
        return 0.0
    s = sorted(v)
    return s[min(len(s) - 1, int(q * (len(s) - 1) + 0.5))]


def describe(name, v):
    return (f"{name}: n={len(v)} p50={pct(v, .5):.2f} p90={pct(v, .9):.2f} "
            f"p95={pct(v, .95):.2f} p99={pct(v, .99):.2f} "
            f"max={max(v) if v else 0:.2f} ms")


def make_frames(n=24, w=640, h=480):
    """A short synthetic clip: a road, and a box that grows as it closes."""
    import cv2
    import numpy as np

    out = []
    for i in range(n):
        a = np.zeros((h, w, 3), np.uint8)
        a[: h // 2] = (150, 140, 130)
        a[h // 2:] = (60, 60, 65)
        cv2.line(a, (w // 2, h // 2), (40, h), (230, 230, 230), 6)
        cv2.line(a, (w // 2, h // 2), (w - 40, h), (230, 230, 230), 6)
        s = 40 + i * 3
        cx, cy = w // 2, h // 2 + 70
        cv2.rectangle(a, (cx - s, cy - s // 2), (cx + s, cy + s // 2),
                      (40, 40, 160), -1)
        out.append(cv2.imencode(".jpg", a)[1].tobytes())
    return out


class SlowTeacher:
    """A fake service that always takes longer than the keyframe interval.

    That is the worst case for the panel and therefore the right case to
    measure: the queue is permanently full, every keyframe evicts one, and both
    client threads are always inside a socket read.
    """

    def __init__(self, port, delay=3.0):
        import json
        import threading
        from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

        class H(BaseHTTPRequestHandler):
            protocol_version = "HTTP/1.1"

            def _send(self, code, obj):
                body = json.dumps(obj).encode()
                self.send_response(code)
                self.send_header("Content-Type", "application/json")
                self.send_header("Content-Length", str(len(body)))
                self.end_headers()
                self.wfile.write(body)

            def do_GET(self):
                self._send(200, {"ok": True, "loaded": True,
                                 "model_id": "fake", "precision": "bf16"})

            def do_POST(self):
                n = int(self.headers.get("Content-Length") or 0)
                self.rfile.read(n)
                time.sleep(delay)
                self._send(200, {"ok": True, "scene": "x", "reasoning": "y",
                                 "critical_actor": "the car ahead",
                                 "attention": "z", "latency_ms": delay * 1000,
                                 "precision": "bf16", "raw": {}})

            def log_message(self, *a):
                pass

        self.httpd = ThreadingHTTPServer(("127.0.0.1", port), H)
        self.t = threading.Thread(target=self.httpd.serve_forever, daemon=True)
        self.t.start()

    def stop(self):
        self.httpd.shutdown()
        self.httpd.server_close()


# ---------------------------------------------------------------------------
def phase_a():
    section("A. what the hook costs on the frame path")
    a = SlowTeacher(18911)
    c = SlowTeacher(18912)
    config.TEACHER_ALPAMAYO_URL = "http://127.0.0.1:18911"
    config.TEACHER_COSMOS_URL = "http://127.0.0.1:18912"
    panel.reset_all()
    panel.stop()
    # THE WORST CASE, BUILT ON PURPOSE. The shipped floor is 2 s and the
    # teachers here take 3 s each, so at the real cadence the queue never fills
    # and the measurement would be of a quiet panel. Wound right down, every
    # frame is a candidate keyframe, the depth-2 queue is permanently full, and
    # every keyframe evicts one -- which is the state the hook has to be cheap
    # in, because it is the state a busy junction produces.
    floor, gap = config.TEACHER_KEYFRAME_FLOOR_S, config.TEACHER_KEYFRAME_MIN_GAP_S
    config.TEACHER_KEYFRAME_FLOOR_S = 0.05
    config.TEACHER_KEYFRAME_MIN_GAP_S = 0.02
    panel.start()

    key = "timing"
    framebuf.drop_ring(key)
    ring = framebuf.get_ring(key)
    jpeg = make_frames(1)[0]
    result = {"ok": True, "t": 0.0, "band": "NORMAL", "distance_m": 30.0,
              "ttc_s": None, "tau_s": 2.4, "trend": "steady", "lead_id": 1,
              "speed": {"v_ms": 14.0, "source": "obd", "degraded": False},
              "scene_objects": [{"id": 1, "label": "car",
                                 "box": [280, 200, 380, 280], "range_m": 30.0,
                                 "member": True, "is_lead": True}],
              "image": {"w": 640, "h": 480}}

    ordinary, keyframes = [], []
    n = 1200
    for i in range(n):
        r = dict(result)
        r["t"] = i * 0.08
        ring.push(jpeg, r, origin=f"{key}:camera")
        t0 = time.perf_counter()
        raised = panel.on_frame(key, r, ring)
        dt = (time.perf_counter() - t0) * 1000.0
        (keyframes if raised else ordinary).append(dt)
        # 12.5 fps, compressed: the loop under test runs at 8-15 and the point
        # is to cross the 2 s floor many times, not to run in real time.
        time.sleep(0.004)

    print("       " + describe("ordinary frames", ordinary))
    print("       " + describe("keyframe frames", keyframes))
    ok(len(keyframes) >= 20,
       f"many keyframes were raised against two 3-second teachers "
       f"({len(keyframes)} of {n} frames)")
    ok(pct(ordinary, .5) < HOOK_P50_MS,
       f"an ordinary frame pays {pct(ordinary, .5):.3f} ms at p50 "
       f"(gate {HOOK_P50_MS})")
    ok(pct(ordinary, .99) < HOOK_P99_MS,
       f"...and {pct(ordinary, .99):.3f} ms at p99 (gate {HOOK_P99_MS})")
    allv = ordinary + keyframes
    ok(max(allv) < HOOK_MAX_MS,
       f"the worst frame of {len(allv)} pays {max(allv):.2f} ms "
       f"(gate {HOOK_MAX_MS}) — including the ones that built and sent a "
       f"keyframe while both teachers were mid-inference")

    st = panel.status()["services"]
    ok(any(s["evicted"] for s in st.values()),
       f"and the queue really was saturated "
       f"({[s['evicted'] for s in st.values()]} evictions) — this is the "
       f"worst case, not a quiet one")

    panel.stop()
    a.stop()
    c.stop()
    config.TEACHER_KEYFRAME_FLOOR_S, config.TEACHER_KEYFRAME_MIN_GAP_S = floor, gap
    return ordinary, keyframes


def pipeline_run(frames, with_panel, key):
    """The real headway pipeline over `frames`. -> list of total_ms."""
    from headway import live as headway_live

    headway_live.reset_session(key)
    # use_qwen=True is what turns the DETECTOR on (headway/live.py reads it as
    # `use_vision`). With it off the pipeline skips RF-DETR entirely and the
    # thing being timed is not the thing that runs on a drive.
    session = headway_live.get_session(key, use_qwen=True)
    framebuf.drop_ring(key)
    ring = framebuf.get_ring(key)
    out = []
    for i, jpeg in enumerate(frames):
        t0 = time.perf_counter()
        result = session.process(jpeg, 14.0, 0.0, None)
        ring.push(jpeg, result, origin=f"{key}:camera")
        if with_panel:
            panel.on_frame(key, result, ring)
        out.append((time.perf_counter() - t0) * 1000.0)
    headway_live.reset_session(key)
    return out


def phase_b(reps):
    section("B. the real pipeline, with the panel off and with it saturated")
    try:
        from headway import detect
    except Exception as e:
        ok(False, f"headway is not importable: {e}")
        return
    if not detect.available():
        ok(False, "RF-DETR is not loaded — cannot measure the real pipeline "
                  "(python -m tools.preflight --fix)")
        return

    frames = make_frames(24)
    print(f"       warming ({len(frames)} frames)...", flush=True)
    panel.stop()
    panel.reset_all()
    pipeline_run(frames, False, "warm")

    a = SlowTeacher(18913)
    c = SlowTeacher(18914)
    config.TEACHER_ALPAMAYO_URL = "http://127.0.0.1:18913"
    config.TEACHER_COSMOS_URL = "http://127.0.0.1:18914"
    panel.reset_all()
    panel.start()
    for i in range(40):
        egomotion.note_speed("pipe-on", 14.0, "obd",
                             at=time.time() - 3.0 + i * 0.1)

    # INTERLEAVED, NOT IN TWO BLOCKS. This GPU is shared with a live uvicorn
    # and a card that clocks down when it is idle, so a run measured entirely
    # before another run is measured against a different machine. Alternating
    # the two conditions puts the same drift into both.
    off, on = [], []
    for i in range(reps):
        off += pipeline_run(frames, False, "pipe-off")
        on += pipeline_run(frames, True, "pipe-on")
    kf = panel.status()["sessions"].get("pipe-on", {}).get("keyframes", 0)
    panel.stop()
    a.stop()
    c.stop()

    print("       " + describe("panel OFF", off))
    print("       " + describe("panel ON ", on))
    ok(kf >= 1, f"the panel really was working during the ON run "
                f"({kf} keyframes raised)")

    for label, q, gate in (("p50", .5, PIPE_P50_GROWTH),
                           ("p95", .95, PIPE_P95_GROWTH)):
        a_, b_ = pct(off, q), pct(on, q)
        grew = (b_ - a_) / a_ if a_ else 0.0
        ok(grew <= gate or abs(b_ - a_) <= PIPE_ABS_MS,
           f"{label}: {a_:.2f} -> {b_:.2f} ms ({grew * 100:+.1f}%, gate "
           f"{gate * 100:.0f}% or {PIPE_ABS_MS} ms)")

    med_off, med_on = statistics.median(off), statistics.median(on)
    ok(abs(med_on - med_off) < PIPE_ABS_MS,
       f"and the median moves by {med_on - med_off:+.2f} ms against a 250 ms "
       f"frame budget")


def phase_c(reps, url_a, url_c):
    section("C. ...with the REAL teachers on the same GPU")
    import json
    import urllib.request

    for name, url in (("alpamayo", url_a), ("cosmos", url_c)):
        try:
            with urllib.request.urlopen(url.rstrip("/") + "/health",
                                        timeout=5) as r:
                h = json.loads(r.read().decode())
            ok(h.get("loaded"),
               f"{name} is loaded ({h.get('vram_reserved_mb')} MB reserved)")
        except Exception as e:
            ok(False, f"{name} is not answering on {url}: {e}")
            return

    frames = make_frames(24)
    panel.stop()
    panel.reset_all()
    pipeline_run(frames, False, "warm")
    config.TEACHER_ALPAMAYO_URL = url_a
    config.TEACHER_COSMOS_URL = url_c
    panel.reset_all()
    panel.start()
    for i in range(40):
        egomotion.note_speed("gpu-on", 14.0, "obd",
                             at=time.time() - 3.0 + i * 0.1)
    # Interleaved for the same reason as phase B -- and here it matters more,
    # because the thing being measured IS the other process's use of the card.
    off, on = [], []
    for i in range(reps):
        off += pipeline_run(frames, False, "gpu-off")
        on += pipeline_run(frames, True, "gpu-on")
    kf = panel.status()["sessions"].get("gpu-on", {}).get("keyframes", 0)
    panel.stop()

    print("       " + describe("teachers idle    ", off))
    print("       " + describe("teachers inferring", on))
    ok(kf >= 1, f"keyframes really went to the real services ({kf})")
    for label, q, gate in (("p50", .5, PIPE_P50_GROWTH),
                           ("p95", .95, PIPE_P95_GROWTH)):
        a_, b_ = pct(off, q), pct(on, q)
        grew = (b_ - a_) / a_ if a_ else 0.0
        ok(grew <= gate or abs(b_ - a_) <= PIPE_ABS_MS,
           f"{label} under real GPU contention: {a_:.2f} -> {b_:.2f} ms "
           f"({grew * 100:+.1f}%)")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--reps", type=int, default=6,
                    help="passes over the 24-frame clip per condition")
    ap.add_argument("--gpu-load", action="store_true",
                    help="also measure against the real teacher services")
    ap.add_argument("--alpamayo", default="http://127.0.0.1:8801")
    ap.add_argument("--cosmos", default="http://127.0.0.1:8802")
    ap.add_argument("--skip-pipeline", action="store_true")
    args = ap.parse_args()

    try:
        phase_a()
        if not args.skip_pipeline:
            phase_b(args.reps)
        if args.gpu_load:
            phase_c(args.reps, args.alpamayo, args.cosmos)
    finally:
        panel.stop()

    print("\n" + "=" * 72)
    total = len(PASS) + len(FAIL)
    print(f"{len(PASS)}/{total} checks passed")
    if FAIL:
        print("\nFAILED — the fast loop moved:")
        for f in FAIL:
            print(f"  - {f}")
    print("=" * 72)
    return 1 if FAIL else 0


if __name__ == "__main__":
    sys.exit(main())
