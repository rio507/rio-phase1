"""eye_video_gates.py — the acceptance gates for a grounded video eye.

    python -m tools.eye_video_gates --clip /workspace/ufldv2/example.mp4
    python -m tools.eye_video_gates --gate changes          # one gate only

WHAT THIS RUNS, AND WHY IT RUNS IT IN THIS PROCESS
---------------------------------------------------
The whole pipeline: the detector, depth, the lane net, the corridor, the
headway filter and the eye, on the same card at the same time, fed by a clip
at 4 fps. Not over HTTP against the live server, for one reason that decides
it: the grounded and ungrounded arms have to read THE SAME FRAMES. Over HTTP
the ring is trimmed by wall clock between two requests, so the second arm would
read a different stretch of road and the comparison the spec actually asks for
-- "readings side by side verbatim" -- would be a comparison of two roads.

In here the window is built once and handed to both arms.

THE CLIP RULE
-------------
Nothing from runs/. Every clip in there is an annotated render with RIO's own
overlay burned into the pixels -- runs/road_clip.mp4 and runs/probe/road_40s.mp4
both carry a "car 18.7m" label in the top right -- and a model that reads a
distance off our own overlay and reports it back would pass the grounding check
for the worst possible reason. Default is /workspace/ufldv2/example.mp4, which
is raw road footage and is what tools/vision_ab.py already uses.

THE GATES
---------
  changes        different stretches of road produce different readings, and a
                 static or black stretch produces a reading that says so. A
                 reading that does not move with the road fails regardless of
                 anything else, so this gate runs first and its failure is
                 fatal to the run.
  grounded       the same footage, with and without the measured block, six
                 situations minimum, both readings kept verbatim.
  corroboration  for every road user a reading names, is there a track or a
                 headway state behind it. Counts fabrications, and counts what
                 the headway loop saw that the reading did not.
"""
import argparse
import hashlib
import json
import os
import sys
import time
from pathlib import Path

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from dotenv import load_dotenv                                   # noqa: E402

load_dotenv(str(Path(__file__).resolve().parent.parent / ".env"))

import config                                                    # noqa: E402
import eyeread                                                   # noqa: E402
import eyewindow                                                 # noqa: E402
import framebuf                                                  # noqa: E402
import grounding                                                 # noqa: E402
from headway import live as headway_live                         # noqa: E402

DEFAULT_CLIP = os.environ.get("RIO_BENCH_CLIP", "/workspace/ufldv2/example.mp4")
FPS = 4.0
WINDOW_S = 6.0

# SIX SITUATIONS, spread across the clip so they are six different stretches of
# road rather than six samples of one. The last two are built rather than
# filmed, because the honest answers to them are knowable in advance and that
# is what makes them gates rather than observations:
#
#   black    a dead camera. The honest reading says the view is unusable.
#   frozen   a live transport showing a still picture -- the fault a rolling
#            window can detect and a single frame structurally cannot.
SITUATIONS = [
    {"id": "open_road", "at_s": 8.0,
     "why": "four lanes, one vehicle well ahead and to the right"},
    {"id": "traffic_right", "at_s": 14.0,
     "why": "a sedan overtaking on the right, close enough to range"},
    {"id": "curve", "at_s": 22.0,
     "why": "the road bending, lane geometry changing across the window"},
    {"id": "mid_clip", "at_s": 31.0,
     "why": "a different stretch again, for the picture-changes gate"},
    {"id": "late_clip", "at_s": 44.0,
     "why": "the last distinct stretch the clip offers"},
    {"id": "black", "synth": "black",
     "why": "a dead camera. 'I cannot see' is the correct reading."},
    {"id": "frozen", "at_s": 20.0, "synth": "freeze",
     "why": "one frame held for six seconds: a stalled transport."},
    # A DIFFERENT ROAD ENTIRELY, and the only other real footage on this pod:
    # NVIDIA's own sample, a suburban street of parked cars. Two and a half
    # seconds is shorter than the window, which is itself worth having in the
    # set -- a live drive that has only just started has a short ring too.
    {"id": "suburban", "clip": "/workspace/teachers/src/cosmos-reason2/assets/sample.mp4",
     "at_s": 2.5, "seconds": 2.5,
     "why": "a suburban street, parked cars either side. A different road."},
    # THE LEAD THIS FOOTAGE DOES NOT HAVE.
    #
    # Measured, not assumed: the whole 50 s of /workspace/ufldv2/example.mp4
    # produces a gap on ZERO of 210 frames, because the ego lane is empty for
    # the length of the clip. So the most valuable half of the grounding block
    # -- gap, time to contact, band, the plausibility verdict -- is never
    # exercised by any situation above, and a report that did not say so would
    # be claiming a test it had not run.
    #
    # This situation runs real frames and real tracks, and INJECTS a lead into
    # the measured state. The numbers are fabricated and are marked as such
    # everywhere they appear. It answers one question and no other: given a
    # gap and a TTC, does the reading use them. It is not evidence about the
    # detector, the filter or the road.
    {"id": "synthetic_lead", "at_s": 14.0, "inject_lead": True,
     "why": "SYNTHETIC lead injected into the measured state — the clip has "
            "no in-lane vehicle anywhere, so this is the only way to see "
            "whether a gap and a TTC change the reading."},
]

# The fabricated headway state for the synthetic_lead situation. Close enough
# to matter and not close enough to be a warning: a gap that would put the
# band in CRITICAL would tell us only that the model can read the word
# CRITICAL.
INJECTED_LEAD = {
    "gap_m": 18.4, "gap_first_m": 27.9, "ttc_s": 4.6, "closing_ms": -2.1,
    "band": "GETTING_UNSAFE", "plausibility": "accepted", "lead_id": None,
}


def jpeg_of(bgr, q=90):
    ok, buf = cv2.imencode(".jpg", bgr, [int(cv2.IMWRITE_JPEG_QUALITY), q])
    return buf.tobytes() if ok else None


def clip_frames(path, at_s, seconds, fps):
    """Consecutive frames ending at `at_s`. -> [bgr, ...]."""
    cap = cv2.VideoCapture(str(path))
    if not cap.isOpened():
        raise SystemExit(f"cannot open clip: {path}")
    try:
        src_fps = float(cap.get(cv2.CAP_PROP_FPS) or 0) or 30.0
        step = max(1, int(round(src_fps / fps)))
        first = max(0, int(round((at_s - seconds) * src_fps)))
        last = int(round(at_s * src_fps))
        cap.set(cv2.CAP_PROP_POS_FRAMES, first)
        out, idx, want = [], first, first
        while idx <= last:
            ok, bgr = cap.read()
            if not ok:
                break
            if idx >= want:
                out.append(bgr)
                want = idx + step
            idx += 1
        return out
    finally:
        cap.release()


def synth_frames(kind, base, n):
    if kind == "black":
        h, w = (base[0].shape[:2] if base else (720, 1280))
        return [np.zeros((h, w, 3), dtype=np.uint8) for _ in range(n)]
    if kind == "freeze":
        one = base[len(base) // 2]
        return [one.copy() for _ in range(n)]
    return base


def feed(key, frames, v_host=25.0):
    """Push frames through the real loop and into the ring. -> [results].

    Paced to the rate the window will claim. Not for realism's sake: the ring
    trims by WALL CLOCK, so frames pushed faster than 4 fps would produce a
    six-second window holding twelve seconds of road, and every timestamp the
    model was given would be wrong by the difference.
    """
    # use_qwen is the DETECTOR switch despite its name (headway.live reads it
    # as self.use_vision). Passing False here is how the first run of this
    # harness produced six readings grounded in zero tracks, which looked like
    # a grounding failure and was a harness bug. app.py passes
    # config.VISION_ENABLED; so does this.
    session = headway_live.get_session(key, use_qwen=config.VISION_ENABLED)
    ring = framebuf.get_ring(key)
    results = []
    period = 1.0 / FPS
    for i, bgr in enumerate(frames):
        t0 = time.time()
        jpeg = jpeg_of(bgr)
        if jpeg is None:
            continue
        try:
            res = session.process(jpeg, v_host, 0.0, i / FPS)
        except Exception as e:
            print(f"  [feed] frame {i} failed: {type(e).__name__}: {e}")
            continue
        if res and res.get("ok") is not False:
            ring.push(jpeg, res, origin=f"{key}:gates")
            results.append(res)
        slack = period - (time.time() - t0)
        if slack > 0:
            time.sleep(slack)
    return results


def read_both(key, budget=None, seconds=None, inject=None):
    """One window, both arms. -> (grounded_rec, ungrounded_rec, window)."""
    w = eyewindow.from_ring(key, seconds or WINDOW_S, FPS)
    if w is None:
        return None, None, None
    state = grounding.window_state(w)
    if inject:
        # Overwrites the measured headway block wholesale. The tracks and the
        # ego speed stay real; only the lead is invented, and only here.
        state["headway"] = dict(state["headway"], **inject)
    g = eyeread.read_window(w, state, grounded=True, max_new_tokens=budget)
    # The SAME window object, so the ungrounded arm is the same pixels, the
    # same timestamps and the same prompt minus one block. Anything else is a
    # comparison of two different things wearing the same label.
    u = eyeread.read_window(w, {}, grounded=False, max_new_tokens=budget)
    return g, u, w


def norm(text):
    return " ".join((text or "").lower().split())


def run(args):
    clip = args.clip
    out_dir = Path(args.out or "runs/eye_video_gates")
    out_dir.mkdir(parents=True, exist_ok=True)
    key = "gates"
    records = []

    situations = [s for s in SITUATIONS
                  if not args.only or s["id"] in args.only.split(",")]

    for sit in situations:
        print(f"\n=== {sit['id']}: {sit['why']}")
        framebuf.drop_ring(key)
        secs = float(sit.get("seconds") or WINDOW_S)
        n = int(secs * FPS)
        base = (clip_frames(sit.get("clip") or clip, sit.get("at_s", 20.0),
                            secs, FPS)
                if sit.get("at_s") is not None else [])
        frames = synth_frames(sit["synth"], base, n) if sit.get("synth") else base
        if len(frames) < eyewindow.MIN_FRAMES:
            print(f"  skipped: only {len(frames)} frames")
            continue
        t_feed = time.time()
        results = feed(key, frames)
        feed_ms = (time.time() - t_feed) * 1000.0
        print(f"  fed {len(results)}/{len(frames)} frames in {feed_ms:.0f} ms")

        g, u, w = read_both(key, args.budget, secs,
                            inject=(INJECTED_LEAD if sit.get("inject_lead") else None))
        if g is None:
            print("  no window built")
            continue
        rec = {"situation": sit, "grounded": g, "ungrounded": u,
               "window": w.to_meta(),
               "detector": {
                   "n_tracks": len((g.get("state") or {}).get("tracks") or []),
                   "gap_m": ((g.get("state") or {}).get("headway") or {}).get("gap_m"),
                   "band": ((g.get("state") or {}).get("headway") or {}).get("band"),
               }}
        records.append(rec)
        print(f"  window {w.to_meta()['n_frames']} frames / "
              f"{w.to_meta()['span_s']} s, grid {g['window'].get('grid_thw')}, "
              f"blank={w.to_meta()['blank']} static={w.to_meta()['static']}")
        print(f"  grounded   {g['timing'].get('gen_ms')} ms  "
              f"{len(g.get('answer') or '')} chars  "
              f"sourced={len(g.get('sourced') or [])} "
              f"invented={len(g.get('invented') or [])} "
              f"verdict={(g.get('corroboration') or {}).get('verdict')}")
        print(f"  ungrounded {u['timing'].get('gen_ms')} ms  "
              f"{len(u.get('answer') or '')} chars  "
              f"invented={len(u.get('invented') or [])}")

    (out_dir / "records.json").write_text(json.dumps(records, indent=2, default=str))
    report(records, out_dir)
    return records


def report(records, out_dir):
    print("\n" + "=" * 72)
    print("GATE 1  picture-changes")
    print("=" * 72)
    seen = {}
    fail = []
    for r in records:
        sid = r["situation"]["id"]
        ans = norm(r["grounded"].get("answer"))
        h = hashlib.sha256(ans.encode()).hexdigest()[:12]
        dup = [k for k, v in seen.items() if v == h]
        seen[sid] = h
        mark = "SAME AS " + dup[0] if dup else "distinct"
        print(f"  {sid:16} {h}  {mark}")
        if dup:
            fail.append((sid, dup[0]))
    for r in records:
        sid = r["situation"]["id"]
        if sid not in ("black", "frozen"):
            continue
        ans = (r["grounded"].get("answer") or "").lower()
        says = any(p in ans for p in (
            "cannot", "can't", "no view", "unusable", "black", "dark",
            "nothing visible", "not visible", "obscured", "blank", "no image",
            "static", "stationary", "not moving", "unchanged", "frozen",
            "no motion", "does not change", "no change"))
        print(f"  {sid:16} says-so={says}")
        if not says:
            fail.append((sid, "did not say the view was unusable/static"))
    print(f"  -> {'FAIL' if fail else 'PASS'}  {fail if fail else ''}")

    print("\n" + "=" * 72)
    print("GATE 2  grounded vs ungrounded, verbatim")
    print("=" * 72)
    for r in records:
        sid = r["situation"]["id"]
        g, u = r["grounded"], r["ungrounded"]
        print(f"\n--- {sid} :: {r['situation']['why']}")
        st = (g.get("state") or {})
        hw = st.get("headway") or {}
        print(f"    measured: {len(st.get('tracks') or [])} tracks, "
              f"gap={hw.get('gap_m')} m, ttc={hw.get('ttc_s')} s, "
              f"band={hw.get('band')}")
        print(f"    GROUNDED   ({g['timing'].get('gen_ms')} ms): "
              f"{g.get('answer') or '(refused: %s)' % g.get('refused')}")
        print(f"    UNGROUNDED ({u['timing'].get('gen_ms')} ms): "
              f"{u.get('answer') or '(refused: %s)' % u.get('refused')}")

    print("\n" + "=" * 72)
    print("GATE 3  corroboration")
    print("=" * 72)
    tot = {"fab_g": 0, "fab_u": 0, "miss_g": 0, "miss_u": 0,
           "inv_g": 0, "inv_u": 0, "src_g": 0, "cite_g": 0}
    for r in records:
        sid = r["situation"]["id"]
        g, u = r["grounded"], r["ungrounded"]
        cg = g.get("corroboration") or {}
        # The ungrounded arm is scored against the SAME measured state it was
        # not given. That is the only way the two counts are comparable: what
        # is being asked is whether the reading matches the road, not whether
        # it matches its own prompt.
        cu = eyeread.corroborate(u.get("answer") or "", g.get("state") or {})
        tot["fab_g"] += len(cg.get("fabricated") or [])
        tot["fab_u"] += len(cu.get("fabricated") or [])
        tot["miss_g"] += len(cg.get("missed") or [])
        tot["miss_u"] += len(cu.get("missed") or [])
        tot["inv_g"] += len(g.get("invented") or [])
        tot["inv_u"] += len(u.get("invented") or [])
        tot["src_g"] += len(g.get("sourced") or [])
        tot["cite_g"] += len(cg.get("cited") or [])
        print(f"  {sid:16} grounded: fab={len(cg.get('fabricated') or [])} "
              f"miss={len(cg.get('missed') or [])} "
              f"cited={cg.get('cited')} invented={len(g.get('invented') or [])} "
              f"sourced={len(g.get('sourced') or [])} | "
              f"ungrounded: fab={len(cu.get('fabricated') or [])} "
              f"miss={len(cu.get('missed') or [])} "
              f"invented={len(u.get('invented') or [])}")
    print(f"\n  TOTALS  fabrications  grounded {tot['fab_g']}  "
          f"ungrounded {tot['fab_u']}")
    print(f"          missed tracks  grounded {tot['miss_g']}  "
          f"ungrounded {tot['miss_u']}")
    print(f"          invented nums  grounded {tot['inv_g']}  "
          f"ungrounded {tot['inv_u']}")
    print(f"          sourced nums   grounded {tot['src_g']}")
    print(f"          track cites    grounded {tot['cite_g']}")

    print("\n" + "=" * 72)
    print("LATENCY")
    print("=" * 72)
    for arm in ("grounded", "ungrounded"):
        ms = [r[arm]["timing"].get("gen_ms") for r in records
              if r[arm].get("timing", {}).get("gen_ms")]
        pt = [r[arm]["timing"].get("prompt_tokens") for r in records
              if r[arm].get("timing", {}).get("prompt_tokens")]
        nt = [r[arm]["timing"].get("new_tokens") for r in records
              if r[arm].get("timing", {}).get("new_tokens")]
        if ms:
            ms = sorted(ms)
            print(f"  {arm:11} generate p50 {ms[len(ms)//2]:.0f} ms  "
                  f"max {ms[-1]:.0f} ms   prompt tokens "
                  f"{int(np.mean(pt)) if pt else 0}  new tokens "
                  f"{int(np.mean(nt)) if nt else 0}")
    print("\nflags:", json.dumps({k: v for k, v in eyeread.flags().items()
                                  if k != "rate"}, indent=2))
    print(f"\nrecords written to {out_dir}/records.json")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--clip", default=DEFAULT_CLIP)
    ap.add_argument("--out", default=None)
    ap.add_argument("--only", default=None, help="comma-separated situation ids")
    ap.add_argument("--budget", type=int, default=None,
                    help="max_new_tokens override")
    args = ap.parse_args()
    if "/runs/" in str(args.clip) or str(args.clip).startswith("runs/"):
        raise SystemExit("refusing a clip from runs/: RIO's own overlay is "
                         "burned into those pixels. See this module's docstring.")
    run(args)


if __name__ == "__main__":
    main()
