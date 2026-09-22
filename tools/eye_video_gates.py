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


def read_arms(key, arms, budget=None, seconds=None, inject=None):
    """One window, every arm, grounded and not. -> ({arm: {g, u}}, window).

    ONE WINDOW OBJECT FOR ALL SIX READINGS. The arms are a comparison of
    QUESTIONS, so every other variable has to be nailed down: the same
    pixels, the same timestamps, the same measured block. Rebuilding the
    window per arm would compare three different stretches of road and
    attribute the difference to the prompt.
    """
    w = eyewindow.from_ring(key, seconds or WINDOW_S, FPS)
    if w is None:
        return None, None
    state = grounding.window_state(w)
    if inject:
        # Overwrites the measured headway block wholesale. The tracks and the
        # ego speed stay real; only the lead is invented, and only here.
        state["headway"] = dict(state["headway"], **inject)
    out = {}
    for arm in arms:
        g = eyeread.read_window(w, state, grounded=True, arm=arm,
                                max_new_tokens=budget)
        # The same window minus one block, so a difference is the grounding
        # and not the road.
        u = eyeread.read_window(w, {}, grounded=False, arm=arm,
                                max_new_tokens=budget)
        out[arm] = {"grounded": g, "ungrounded": u}
    return out, w


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
    arms = [a.strip().upper() for a in (args.arms or "A,B,C,D,E").split(",") if a.strip()]
    print(f"arms: " + ", ".join(f"{a} = {eyeread.ARM_NAMES.get(a, a)}" for a in arms))

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

        by_arm, w = read_arms(key, arms, args.budget, secs,
                              inject=(INJECTED_LEAD if sit.get("inject_lead") else None))
        if by_arm is None:
            print("  no window built")
            continue
        first = by_arm[arms[0]]["grounded"]
        st = first.get("state") or {}
        rec = {"situation": sit, "arms": by_arm, "window": w.to_meta(),
               "detector": {
                   "n_tracks": len(st.get("tracks") or []),
                   "gap_m": (st.get("headway") or {}).get("gap_m"),
                   "band": (st.get("headway") or {}).get("band"),
                   "loop_spoke": (st.get("loop") or {}).get("spoke"),
               }}
        records.append(rec)
        m = w.to_meta()
        print(f"  window {m['n_frames']} frames / {m['span_s']} s, "
              f"grid {first['window'].get('grid_thw')}, "
              f"blank={m['blank']} static={m['static']}, "
              f"loop_spoke={(st.get('loop') or {}).get('spoke')}")
        for arm in arms:
            for kind in ("grounded", "ungrounded"):
                r = by_arm[arm][kind]
                j = r.get("judgement") or {}
                print(f"  {arm} {kind:10} {r['timing'].get('gen_ms'):>7.0f} ms  "
                      f"{len(r.get('answer') or ''):>4} ch  "
                      f"risk={j.get('risk') or r.get('judgement_problem') or '-':<9} "
                      f"speak={str(j.get('should_speak')):<5} "
                      f"out={(r.get('shadow') or {}).get('outcome') or '-':<12} "
                      f"ctrl={len(r.get('control_decision') or [])} "
                      f"inv={len(r.get('invented') or [])} "
                      f"{('REFUSED:' + str(r.get('refused'))) if r.get('refused') else ''}")

    (out_dir / "records.json").write_text(json.dumps(records, indent=2, default=str))
    report(records, out_dir)
    return records


def report(records, out_dir):
    arms = sorted({a for r in records for a in r["arms"]})

    print("\n" + "=" * 72)
    print("GATE 1  picture-changes, per arm")
    print("=" * 72)
    fails = []
    for arm in arms:
        print(f"\n  --- arm {arm}: {eyeread.ARM_NAMES.get(arm, arm)}")
        seen = {}
        for r in records:
            sid = r["situation"]["id"]
            ans = norm(r["arms"][arm]["grounded"].get("answer"))
            h = hashlib.sha256(ans.encode()).hexdigest()[:12]
            dup = [k for k, v in seen.items() if v == h]
            seen[sid] = h
            print(f"    {sid:16} {h}  {'SAME AS ' + dup[0] if dup else 'distinct'}")
            if dup:
                fails.append((arm, sid, "identical to " + dup[0]))
        # THE BLANK FRAME DECIDES IT. A prompt that tells the model it is a
        # vehicle on a road is exactly the instruction that could make it
        # describe a road when there is none, so this is checked per arm and
        # a failure here outweighs anything the arm does on real footage.
        for sid in ("black", "frozen"):
            row = next((r for r in records if r["situation"]["id"] == sid), None)
            if not row:
                continue
            rec = row["arms"][arm]["grounded"]
            ans = (rec.get("answer") or "").lower()
            says = bool(eyeread._SAYS_UNUSABLE.search(ans)) or any(
                p in ans for p in ("static", "stationary", "not moving",
                                   "unchanged", "frozen", "no motion",
                                   "does not change", "no change"))
            invents = bool(eyeread._ROAD_WORDS.search(ans)) and not says
            print(f"    {sid:16} says-so={says}  invents-a-road={invents}")
            if not says:
                fails.append((arm, sid, "did not say the view was unusable/static"))
    print(f"\n  -> {'FAIL' if fails else 'PASS'}")
    for f in fails:
        print(f"     arm {f[0]}  {f[1]}: {f[2]}")

    print("\n" + "=" * 72)
    print("GATE 2  grounded vs ungrounded, verbatim, per arm")
    print("=" * 72)
    for r in records:
        sid = r["situation"]["id"]
        st = r["arms"][arms[0]]["grounded"].get("state") or {}
        hw = st.get("headway") or {}
        print(f"\n--- {sid} :: {r['situation']['why']}")
        print(f"    measured: {len(st.get('tracks') or [])} tracks, "
              f"gap={hw.get('gap_m')} m, ttc={hw.get('ttc_s')} s, "
              f"band={hw.get('band')}, loop_spoke="
              f"{(st.get('loop') or {}).get('spoke')}")
        for arm in arms:
            for kind in ("grounded", "ungrounded"):
                rec = r["arms"][arm][kind]
                j = rec.get("judgement") or {}
                body = rec.get("answer") or ""
                if rec.get("refused"):
                    body = f"[REFUSED: {rec['refused']}] " + body[:200]
                print(f"    {arm} {kind.upper():10} "
                      f"({rec['timing'].get('gen_ms'):.0f} ms) "
                      f"[risk={j.get('risk') or rec.get('judgement_problem')} "
                      f"speak={j.get('should_speak')}]: {body}")

    print("\n" + "=" * 72)
    print("GATE 3  corroboration and the four shadow counts, per arm")
    print("=" * 72)
    for arm in arms:
        t = {k: 0 for k in ("fab", "contra", "inv", "src", "cite", "bad",
                            "ctrl", "risk_unver", "urg_contra", "jmiss",
                            "jmal", "refused", "agree_speak", "false_alarm",
                            "miss", "agree_quiet", "refused_loop_spoke",
                            "borrow", "example_echo", "elevated", "fp_viol")}
        tu = {"fab": 0, "contra": 0, "inv": 0, "ctrl": 0}
        for r in records:
            g = r["arms"][arm]["grounded"]
            u = r["arms"][arm]["ungrounded"]
            c = g.get("corroboration") or {}
            jf = g.get("judge_faults") or {}
            t["fab"] += len(c.get("fabricated") or [])
            t["contra"] += len(c.get("contradicts") or [])
            t["inv"] += len(g.get("invented") or [])
            t["src"] += len(g.get("sourced") or [])
            t["cite"] += len(c.get("cited") or [])
            t["bad"] += len(c.get("bad_cites") or [])
            t["ctrl"] += len(g.get("control_decision") or [])
            t["borrow"] += len(g.get("borrowed") or [])
            t["example_echo"] += len(g.get("example_echo") or [])
            t["elevated"] += 1 if (g.get("rubric") or {}).get("elevated") else 0
            t["fp_viol"] += (g.get("rubric") or {}).get("n_violations") or 0
            t["risk_unver"] += 1 if jf.get("risk_unverified") else 0
            t["urg_contra"] += 1 if jf.get("urgent_contradicted") else 0
            t["jmiss"] += 1 if g.get("judgement_problem") == "missing" else 0
            t["jmal"] += 1 if g.get("judgement_problem") == "malformed" else 0
            if g.get("refused"):
                t["refused"] += 1
            out = (g.get("shadow") or {}).get("outcome")
            if out in t:
                t[out] += 1
            if out == "refused" and (g.get("shadow") or {}).get("loop_spoke"):
                t["refused_loop_spoke"] += 1
            # The ungrounded arm is scored against the SAME measured state it
            # was not given: the question is whether it matches the road, not
            # whether it matches its own prompt.
            cu = eyeread.corroborate(u.get("answer") or "",
                                     g.get("state") or {},
                                     u.get("judgement"))
            tu["fab"] += len(cu.get("fabricated") or [])
            tu["contra"] += len(cu.get("contradicts") or [])
            tu["inv"] += len(u.get("invented") or [])
            tu["ctrl"] += len(u.get("control_decision") or [])
            tu["borrow"] = tu.get("borrow", 0) + len(u.get("borrowed") or [])
        n = len(records)
        print(f"\n  arm {arm}  ({eyeread.ARM_NAMES.get(arm, arm)}), "
              f"{n} grounded readings")
        print(f"    fabrications      {t['fab']:>3}   (ungrounded {tu['fab']})")
        print(f"    contradictions    {t['contra']:>3}   (ungrounded {tu['contra']})")
        print(f"    invented numbers  {t['inv']:>3}   (ungrounded {tu['inv']})")
        print(f"    control decisions {t['ctrl']:>3}   (ungrounded {tu['ctrl']})")
        print(f"    sourced numbers   {t['src']:>3}")
        print(f"    track citations   {t['cite']:>3}   bad cites {t['bad']}")
        print(f"    judgement missing {t['jmiss']:>3}   malformed {t['jmal']}")
        print(f"    risk unverified   {t['risk_unver']:>3}   "
              f"urgent contradicted {t['urg_contra']}")
        print(f"    readings refused  {t['refused']:>3}")
        print(f"    PROMPT BORROWING  {t['borrow']:>3}   "
              f"(ungrounded {tu.get('borrow', 0)})   "
              f"worked-example echo {t['example_echo']}")
        print(f"    RUBRIC   elevated {t['elevated']:>3}   "
              f"false-positive-control violations {t['fp_viol']}")
        print(f"    SHADOW  agreement(speak) {t['agree_speak']}  "
              f"FALSE ALARM {t['false_alarm']}  miss {t['miss']}  "
              f"agreement(quiet) {t['agree_quiet']}  "
              f"[refused while loop spoke {t['refused_loop_spoke']}]")

    print("\n" + "=" * 72)
    print("LATENCY, per arm (grounded)")
    print("=" * 72)
    for arm in arms:
        ms = sorted(r["arms"][arm]["grounded"]["timing"].get("gen_ms")
                    for r in records
                    if r["arms"][arm]["grounded"]["timing"].get("gen_ms"))
        pt = [r["arms"][arm]["grounded"]["timing"].get("prompt_tokens")
              for r in records
              if r["arms"][arm]["grounded"]["timing"].get("prompt_tokens")]
        if ms:
            print(f"  {arm}  p50 {ms[len(ms)//2]:.0f} ms   min {ms[0]:.0f}   "
                  f"max {ms[-1]:.0f}   prompt {int(np.mean(pt)) if pt else 0} tokens")

    print("\nflags (all arms pooled):",
          json.dumps({k: v for k, v in eyeread.flags().items()
                      if k not in ("rate",)}, indent=2))
    print("\nflags per arm:",
          json.dumps({a: {k: v for k, v in d.items() if k != "rate"}
                      for a, d in eyeread.arm_flags().items()}, indent=2))
    print(f"\nrecords written to {out_dir}/records.json")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--clip", default=DEFAULT_CLIP)
    ap.add_argument("--out", default=None)
    ap.add_argument("--only", default=None, help="comma-separated situation ids")
    ap.add_argument("--budget", type=int, default=None,
                    help="max_new_tokens override")
    ap.add_argument("--arms", default="A,B,C,D,E",
                    help="which prompt arms to run (default all five)")
    args = ap.parse_args()
    if "/runs/" in str(args.clip) or str(args.clip).startswith("runs/"):
        raise SystemExit("refusing a clip from runs/: RIO's own overlay is "
                         "burned into those pixels. See this module's docstring.")
    run(args)


if __name__ == "__main__":
    main()
