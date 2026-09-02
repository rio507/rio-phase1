"""overlay_lag.py — is the box drawn on the frame it was computed from?

    python -m tools.overlay_lag CLIP.mp4
    python -m tools.overlay_lag CLIP.mp4 --http --at 41.5 --out runs/align

A detection is computed from the frame at time T and cannot exist before
T + latency. Drawn the moment it arrives, it therefore lands on a frame the
road has already moved past: the detector was on time and the OVERLAY was late.
From the passenger seat those two look identical, which is why this measures
the second one specifically.

WHAT IT MEASURES
----------------
The real latency distribution, by pushing real frames of a real clip through
the real pipeline (in-process by default, --http through the running server so
the network is in the number too). Everything else follows from that
distribution and two scheduling policies:

  BEFORE  analyse the visible element, draw on arrival.
          display gap = latency

  AFTER   analyse a hidden element held HEADWAY_REPLAY_LEAD_S ahead, draw when
          the picture reaches the source frame.
          display gap = max(0, latency - lead)

That subtraction is the entire fix, and it is exact rather than modelled: the
result for source time T is in hand at capture + latency, the viewer reaches T
at capture + lead, and the box is drawn at the later of the two.

THE CHECK
---------
In replay the drawn-box-vs-source-frame gap must be within one frame at 24 fps
(config.HEADWAY_ALIGN_TOLERANCE_S). Anything under that is smaller than the
interval between frames and cannot be seen; anything over it is a frame the
buffer failed to cover, and the count of those is what says whether
HEADWAY_REPLAY_LEAD_S is big enough for this machine and this link.

WHY A STATIC CHECK TOO
----------------------
A simulation can only prove the arithmetic. It cannot prove the dashboard
implements the policy, and a harness that passes while the client draws on
arrival would be worse than no harness. So the last section reads
static/index.html and asserts the mechanism is actually there: the scout
element, the timestamp queue, and an age measured against the video clock
rather than against arrival.

LIVE MODE IS NOT SIMULATED HERE
-------------------------------
Reality cannot be buffered, so live mode's lag IS its latency, and the answer
there is extrapolation rather than scheduling. The browser measures it --
RIO.overlay.lagStats().live after a drive -- because it depends on the phone,
the camera and the link, none of which exist on this pod. Inventing a number
for it here would be worse than reporting that it is measured elsewhere.
"""
import argparse
import json
import os
import statistics
import sys
import time

import cv2
import numpy as np

sys.path.insert(0, "/workspace/rio-phase1")

import config                                      # noqa: E402
from headway import live as live_mod               # noqa: E402
from tools.detect_quality import draw_objects, analyse_frame, lead_of  # noqa: E402
from headway import anchor as anchor_mod           # noqa: E402
from headway import depth as depth_mod             # noqa: E402
from headway import detect as detect_mod           # noqa: E402
from headway import lanes as lanes_mod             # noqa: E402
from headway import plausibility as plaus_mod      # noqa: E402

LEAD_S = config.HEADWAY_REPLAY_LEAD_S
# Frames discarded before timing starts. See the note in measure_latency.
WARMUP_FRAMES = 3
TOL_S = config.HEADWAY_ALIGN_TOLERANCE_S

_results = []


def check(name, ok, detail=""):
    _results.append((bool(ok), name, detail))
    print(f"  [{'PASS' if ok else 'FAIL'}] {name}" + (f"  -- {detail}" if detail else ""))
    return bool(ok)


def pct(values, q):
    if not values:
        return None
    v = sorted(values)
    return v[min(len(v) - 1, int(q * (len(v) - 1) + 0.5))]


# ---------------------------------------------------------------------------
# 1. the real latency distribution
# ---------------------------------------------------------------------------
def measure_latency(video, fps, limit, over_http):
    """Push real frames through the real pipeline. -> [seconds per frame]

    In-process measures decode + lanes + depth + detect + membership + filter.
    --http adds JSON, multipart and the socket, which is what the browser
    actually waits for -- on this pod that is a loopback and on a real drive it
    is the internet, so the HTTP number is a floor for what a remote dashboard
    sees, never a ceiling.
    """
    cap = cv2.VideoCapture(video)
    if not cap.isOpened():
        raise SystemExit(f"cannot open {video}")
    src_fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    stride = max(1, int(round(src_fps / float(fps))))

    session = None
    if not over_http:
        # Warm FIRST, and outside the measurement. The lazy model loads land on
        # whichever frame happens to be first -- 12 s of them, measured -- and
        # a browser never sees that: the server warms at startup and
        # /headway_frame refuses frames until it has. Timing a cold load here
        # would put a number in this report that no viewer will ever
        # experience, which is worse than not measuring at all.
        for name, fn in (("lanes", lanes_mod.warm), ("depth", depth_mod.warm),
                         ("detector", detect_mod.warm)):
            try:
                fn()
            except Exception as e:
                print(f"  ({name} warm failed: {type(e).__name__}: {e})")
        live_mod.reset_session("lagprobe")
        session = live_mod.get_session("lagprobe", use_qwen=True)
    else:
        import requests

    lat, idx, n = [], 0, 0
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        if idx % stride:
            idx += 1
            continue
        ok2, buf = cv2.imencode(".jpg", frame, [cv2.IMWRITE_JPEG_QUALITY, 80])
        t_src = idx / src_fps
        t0 = time.perf_counter()
        if over_http:
            requests.post("http://127.0.0.1:8888/headway_frame",
                          files={"image": ("frame.jpg", buf.tobytes(), "image/jpeg")},
                          data={"v_host": "20", "v_host_age_s": "0",
                                "frame_t": str(t_src)}, timeout=60)
        else:
            session.process(buf.tobytes(), 20.0, 0.0, frame_t=t_src)
        dt = time.perf_counter() - t0
        n += 1
        idx += 1
        # The first few frames still carry one-off costs a steady run does not
        # pay -- CUDA kernel autotuning, the first JPEG decode of this size,
        # the session's geometry build. Skipped explicitly and counted, rather
        # than quietly folded into a percentile.
        if n > WARMUP_FRAMES:
            lat.append(dt)
        if limit and n >= limit + WARMUP_FRAMES:
            break
    cap.release()
    if not over_http:
        live_mod.reset_session("lagprobe")
    return lat


# ---------------------------------------------------------------------------
# 2. the two scheduling policies
# ---------------------------------------------------------------------------
def gaps_before(latencies):
    """Draw on arrival: the box is as late as the result was."""
    return [float(x) for x in latencies]


def gaps_after(latencies, lead_s):
    """Draw when the picture reaches the source frame.

    The result is in hand at capture + latency; the viewer reaches the source
    frame at capture + lead. Whichever is later is when the box appears, and
    the gap is measured from the source frame.
    """
    return [max(0.0, float(x) - float(lead_s)) for x in latencies]


def report(latencies, lead_s):
    b, a = gaps_before(latencies), gaps_after(latencies, lead_s)
    ms = lambda v: None if v is None else round(v * 1000, 1)     # noqa: E731
    return {
        "n_frames": len(latencies),
        "lead_s": lead_s,
        "tolerance_ms": round(TOL_S * 1000, 1),
        "latency_ms": {"p50": ms(pct(latencies, 0.5)),
                       "p95": ms(pct(latencies, 0.95)),
                       "max": ms(max(latencies) if latencies else None)},
        "before_ms": {"p50": ms(pct(b, 0.5)), "p95": ms(pct(b, 0.95)),
                      "max": ms(max(b) if b else None),
                      "over_tolerance": sum(1 for x in b if x > TOL_S)},
        "after_ms": {"p50": ms(pct(a, 0.5)), "p95": ms(pct(a, 0.95)),
                     "max": ms(max(a) if a else None),
                     "over_tolerance": sum(1 for x in a if x > TOL_S)},
    }


# ---------------------------------------------------------------------------
# 3. the visual: a box on its own frame vs a box on a later one
# ---------------------------------------------------------------------------
def find_fast_segment(video, fps, search_from=0.0, search_to=None):
    """The timestamp with the fastest box motion. -> (t, px_per_s)

    Alignment error is invisible on a static scene and obvious on a fast one,
    so the comparison frame is chosen by measurement rather than by eye: track
    the largest detection frame to frame and take the peak speed.
    """
    cap = cv2.VideoCapture(video)
    src_fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    stride = max(1, int(round(src_fps / float(fps))))
    prev, best = None, (None, 0.0)
    idx = 0
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        if idx % stride:
            idx += 1
            continue
        t_s = idx / src_fps
        if t_s < search_from or (search_to is not None and t_s > search_to):
            idx += 1
            continue
        dets = detect_mod.detect(frame)["detections"]
        if dets:
            box = dets[0][1]
            cx, cy = (box[0] + box[2]) / 2.0, (box[1] + box[3]) / 2.0
            if prev is not None and (t_s - prev[0]) > 0:
                dt = t_s - prev[0]
                sp = ((cx - prev[1]) ** 2 + (cy - prev[2]) ** 2) ** 0.5 / dt
                if sp > best[1] and sp < 4000:      # 4000 px/s = a new object, not motion
                    best = (t_s, sp)
            prev = (t_s, cx, cy)
        else:
            prev = None
        idx += 1
    cap.release()
    return best


def frame_at(cap, t_s, src_fps):
    cap.set(cv2.CAP_PROP_POS_FRAMES, max(0, int(round(t_s * src_fps))))
    ok, frame = cap.read()
    return frame if ok else None


def render_alignment(video, t_display, latency_s, out_dir):
    """Two panels of the SAME displayed frame, boxed two ways.

    LEFT   the boxes that were in hand when that frame was displayed -- i.e.
           computed from the frame `latency_s` earlier, which is what drawing
           on arrival puts on the screen.
    RIGHT  the boxes computed from the displayed frame itself, which is what
           the presentation buffer makes possible.

    Same pixels, same detector, same drawing code. The only difference is which
    frame the boxes came from, so the gap between the box and the car IS the
    rendering lag, in pixels, at this speed.
    """
    cap = cv2.VideoCapture(video)
    src_fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    shown = frame_at(cap, t_display, src_fps)
    stale = frame_at(cap, max(0.0, t_display - latency_s), src_fps)
    cap.release()
    if shown is None or stale is None:
        return None

    h, w = shown.shape[:2]
    base = anchor_mod.EgoCorridor(w, h)
    f_px = plaus_mod.focal_px(w)
    try:
        lane_result = lanes_mod.detect_lanes(shown)
    except Exception:
        lane_result = None
    corridor, _info = anchor_mod.build_corridor(base, lane_result)

    def objs_of(frame):
        dmap = None
        try:
            if depth_mod.frame_trust(frame)[0]:
                dmap = depth_mod.depth_map(frame)
        except Exception:
            dmap = None
        dets = detect_mod.detect(frame)["detections"]
        return analyse_frame(frame, corridor, f_px, dets, dmap, "after")

    late_objs = objs_of(stale)         # computed from the older frame
    live_objs = objs_of(shown)         # computed from the frame on screen

    # Which object moved most between the two frames -- the one where the error
    # is largest, and the one the inset zooms into. A misalignment of a few tens
    # of pixels is real, is what the eye reads as "the tracker is behind", and
    # is invisible at 1280 px wide unless it is enlarged.
    moved, shift = None, None
    for b in live_objs:
        a = _nearest(late_objs, b)
        if a is None:
            continue
        d = ((_cx(a) - _cx(b)) ** 2 + (_cy(a) - _cy(b)) ** 2) ** 0.5
        if shift is None or d > shift:
            shift, moved = d, (a, b)

    lag_ms = latency_s * 1000
    left = draw_objects(shown.copy(), late_objs, lead_of(late_objs), corridor,
                        f"BEFORE  drawn on arrival: boxes from t={t_display - latency_s:.3f}s "
                        f"on the frame at t={t_display:.3f}s  ({lag_ms:.0f} ms late)",
                        "ALIGNMENT  off by one round trip")
    right = draw_objects(shown.copy(), live_objs, lead_of(live_objs), corridor,
                         f"AFTER   buffered {LEAD_S * 1000:.0f} ms: boxes from "
                         f"t={t_display:.3f}s on the frame at t={t_display:.3f}s",
                         "ALIGNMENT  same frame")
    if moved is not None:
        box = moved[1]["box"]
        _inset(left, shown, box, f"{shift:.0f} px late")
        _inset(right, shown, box, "on the car")
    sep = np.full((left.shape[0], 4, 3), 60, np.uint8)
    img = np.hstack([left, sep, right])
    os.makedirs(out_dir, exist_ok=True)
    name = f"align_{t_display:07.3f}s_{lag_ms:.0f}ms.jpg".replace(".", "_", 1)
    path = os.path.join(out_dir, name)
    cv2.imwrite(path, img, [cv2.IMWRITE_JPEG_QUALITY, 92])
    return {"path": path, "t": t_display, "latency_ms": round(lag_ms, 1),
            "box_shift_px": None if shift is None else round(shift, 1)}


def _cx(o):
    return (o["box"][0] + o["box"][2]) / 2.0


def _cy(o):
    return (o["box"][1] + o["box"][3]) / 2.0


def _same_thing(a, b):
    """Two labels for one object.

    The class flips between car and truck on the same vehicle from frame to
    frame -- observed on this clip at t=4.43s, where the vehicle matched at
    4.68s as a car was a truck 250 ms earlier. detect.EXCLUSIVE_LABELS already
    encodes that those four are competing readings of one thing; matching has to
    honour it, or a class flip reads as "the object vanished" and no shift is
    reported for exactly the frames being compared.
    """
    if a == b:
        return True
    return a in detect_mod.EXCLUSIVE_LABELS and b in detect_mod.EXCLUSIVE_LABELS


def _nearest(objs, target):
    """The same object, one frame apart: nearest centre of the same class.

    The radius is generous -- 2.5 box widths -- because at a quarter of a second
    a car crossing the frame moves further than its own width, and a radius that
    refused that would report "no match" for exactly the fast objects this is
    measuring. It is still bounded, so a car that has left the frame is not
    matched to a different car across the road and reported as a huge shift.
    """
    best, best_d = None, None
    w = max(150.0, 2.5 * (target["box"][2] - target["box"][0]))
    for o in objs:
        if not _same_thing(o["label"], target["label"]):
            continue
        d = ((_cx(o) - _cx(target)) ** 2 + (_cy(o) - _cy(target)) ** 2) ** 0.5
        if d <= w and (best_d is None or d < best_d):
            best, best_d = o, d
    return best


def _inset(panel, source, box, caption, zoom=3.0):
    """Paste a magnified crop of `box` into the panel's bottom-left corner.

    The crop is taken from the panel itself, so it carries whatever boxes were
    drawn on it: side by side, the two insets are the whole argument.
    """
    h, w = panel.shape[:2]
    cx, cy = (box[0] + box[2]) / 2.0, (box[1] + box[3]) / 2.0
    half_w = max(60.0, (box[2] - box[0]) * 0.9)
    half_h = max(45.0, (box[3] - box[1]) * 0.9)
    x1 = int(max(0, min(w - 2 * half_w, cx - half_w)))
    y1 = int(max(0, min(h - 2 * half_h, cy - half_h)))
    x2 = int(min(w, x1 + 2 * half_w))
    y2 = int(min(h, y1 + 2 * half_h))
    crop = panel[y1:y2, x1:x2]
    if crop.size == 0:
        return
    big = cv2.resize(crop, None, fx=zoom, fy=zoom, interpolation=cv2.INTER_NEAREST)
    bh, bw = big.shape[:2]
    # Bottom-left, which on a forward-facing road frame is tarmac.
    py, px = h - bh - 12, 12
    if py < 60 or bw > w - 24:
        scale = min((w - 24) / bw, (h - 72) / bh)
        big = cv2.resize(big, None, fx=scale, fy=scale)
        bh, bw = big.shape[:2]
        py, px = h - bh - 12, 12
    panel[py:py + bh, px:px + bw] = big
    cv2.rectangle(panel, (px, py), (px + bw, py + bh), (200, 200, 200), 1)
    cv2.putText(panel, caption, (px + 6, py + 20), cv2.FONT_HERSHEY_SIMPLEX,
                0.6, (0, 0, 0), 3, cv2.LINE_AA)
    cv2.putText(panel, caption, (px + 6, py + 20), cv2.FONT_HERSHEY_SIMPLEX,
                0.6, (240, 240, 240), 1, cv2.LINE_AA)


# ---------------------------------------------------------------------------
# 4. does the client actually do this?
# ---------------------------------------------------------------------------
def check_client():
    """Read the dashboard and assert the mechanism is present.

    Static, deliberately. The simulation above proves the arithmetic of the
    policy; only this proves the browser implements it, and a green harness
    over a client that still draws on arrival is exactly the failure mode a
    harness is supposed to prevent.
    """
    print("\nclient (static/index.html)")
    html = open("/workspace/rio-phase1/static/index.html").read()
    check("a hidden analysis element exists", 'id="scout"' in html,
          "the clip is analysed from this, not from the visible element")
    check("the presentation buffer comes from config",
          "__HEADWAY_REPLAY_LEAD_S__" in html,
          "injected at '/' like the Maps key, so config.py is the only source")
    check("the scout is held ahead of the picture",
          "preview.currentTime + REPLAY_LEAD_S" in html)
    check("results are queued by source timestamp",
          "hwQueue" in html and "a.srcT - b.srcT" in html,
          "a result waits for its frame instead of being drawn on arrival")
    check("a result is drawn when the picture reaches its frame",
          "function dueEntry" in html and "el.currentTime + DUE_TOL_S" in html)
    check("age is measured against the video clock",
          "el.currentTime - entry.srcT" in html,
          "not against the moment the response landed")
    check("live mode measures from the capture instant",
          "performance.now() - entry.capturedAt" in html,
          "extrapolation driven by the real gap, not a fixed offset")
    check("extrapolation is still capped",
          "MAX_EXTRAP_S" in html and "Math.min(ageS, MAX_EXTRAP_S)" in html)
    check("the client measures its own draw lag",
          "function recordLag" in html and "lagStats" in html)


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("video")
    ap.add_argument("--fps", type=float, default=24.0,
                    help="replay analysis rate (default 24)")
    ap.add_argument("--limit", type=int, default=120,
                    help="frames to measure latency over")
    ap.add_argument("--http", action="store_true",
                    help="measure through the running server, not in-process")
    ap.add_argument("--lead", type=float, default=LEAD_S,
                    help=f"presentation buffer in seconds "
                         f"(default config.HEADWAY_REPLAY_LEAD_S={LEAD_S})")
    ap.add_argument("--at", type=float, action="append", default=[],
                    help="render an alignment comparison at this timestamp; "
                         "repeatable. Omitted, the fastest-moving moment in the "
                         "clip is found and used.")
    ap.add_argument("--lag-ms", type=float, action="append", default=[],
                    help="stage the BEFORE panel at this latency (ms) as well "
                         "as at the measured p95. Repeatable — use it to show "
                         "what a remote browser sees, whose round trip is the "
                         "internet rather than this pod's loopback.")
    ap.add_argument("--out", default="runs/alignment")
    ap.add_argument("--no-render", action="store_true")
    args = ap.parse_args()

    print("=" * 74)
    print("overlay alignment — is the box drawn on the frame it came from?")
    print("=" * 74)
    print(f"clip      : {args.video}")
    print(f"buffer    : {args.lead * 1000:.0f} ms "
          f"(config.HEADWAY_REPLAY_LEAD_S)")
    print(f"tolerance : {TOL_S * 1000:.1f} ms (one frame at 24 fps)")
    print(f"path      : {'HTTP (server + socket)' if args.http else 'in-process'}")

    lat = measure_latency(args.video, args.fps, args.limit, args.http)
    rep = report(lat, args.lead)
    rep["clip"] = args.video
    rep["measured_over"] = "http" if args.http else "in_process"

    print(f"\nlatency over {rep['n_frames']} frames: "
          f"p50 {rep['latency_ms']['p50']} ms, p95 {rep['latency_ms']['p95']} ms, "
          f"max {rep['latency_ms']['max']} ms")
    print("\ndrawn-box vs source-frame gap")
    print(f"{'':10s} {'p50':>10s} {'p95':>10s} {'max':>10s} {'over tol':>10s}")
    for name in ("before", "after"):
        r = rep[f"{name}_ms"]
        print(f"{name:10s} {str(r['p50']):>10s} {str(r['p95']):>10s} "
              f"{str(r['max']):>10s} {str(r['over_tolerance']):>10s}")

    print("\nchecks")
    check(f"replay p95 gap is within one frame at 24 fps",
          (rep["after_ms"]["p95"] or 0) <= TOL_S * 1000,
          f"p95 {rep['after_ms']['p95']} ms vs {TOL_S * 1000:.1f} ms "
          f"(before: {rep['before_ms']['p95']} ms)")
    check("every frame is within tolerance",
          rep["after_ms"]["over_tolerance"] == 0,
          f"{rep['after_ms']['over_tolerance']} of {rep['n_frames']} frames "
          f"took longer than the {args.lead * 1000:.0f} ms buffer "
          f"(before: {rep['before_ms']['over_tolerance']})")
    check("the buffer covers the measured p95 latency",
          (rep["latency_ms"]["p95"] or 0) <= args.lead * 1000,
          f"p95 {rep['latency_ms']['p95']} ms vs a {args.lead * 1000:.0f} ms buffer")

    check_client()

    if not args.no_render:
        print("\nalignment comparison")
        times = list(args.at)
        if not times:
            t_fast, speed = find_fast_segment(args.video, args.fps)
            if t_fast is not None:
                print(f"  fastest box motion at t={t_fast:.2f}s ({speed:.0f} px/s)")
                times = [t_fast]
        rendered = []
        # The measured p95 first -- the lag a viewer complains about, since the
        # median frame is barely late and the tail is what is visible. Then any
        # latency asked for on the command line: this pod measures its own
        # loopback, and a dashboard reached over the internet waits longer than
        # that for every single frame.
        lags = [(rep["latency_ms"]["p95"] or 0) / 1000.0]
        lags += [x / 1000.0 for x in args.lag_ms]
        for t in times:
            for lag in lags:
                out = render_alignment(args.video, t, lag, args.out)
                if out:
                    rendered.append(out)
                    print(f"  {out['path']}  (boxes {out['latency_ms']} ms stale "
                          f"= {out['box_shift_px']} px of box shift)")
        rep["rendered"] = rendered

    os.makedirs(args.out, exist_ok=True)
    with open(os.path.join(args.out, "alignment.json"), "w") as fh:
        json.dump(rep, fh, indent=2)

    passed = sum(1 for ok, _, _ in _results if ok)
    print("\n" + "=" * 74)
    print(f"{passed}/{len(_results)} checks passed")
    if passed != len(_results):
        print("\nFAILED:")
        for ok, name, detail in _results:
            if not ok:
                print(f"  - {name}  {detail}")
    print("=" * 74)
    return 0 if passed == len(_results) else 1


if __name__ == "__main__":
    raise SystemExit(main())
