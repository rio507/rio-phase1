"""Replay a clip past both teachers, and print the readings side by side.

    python -m tools.teacher_replay --clip runs/road_clip.mp4 --keyframes 10
    python -m tools.teacher_replay --clip /workspace/ufldv2/example.mp4 \
        --keyframes 10 --report /tmp/readings.md

WHAT THIS IS FOR
----------------
The panel is a comparison, and a comparison has to be READ to be worth
anything. This runs the real headway pipeline over a clip -- RF-DETR,
Depth-Anything, UFLDv2, the tracker, the filter -- exactly as a drive does,
lets the panel raise keyframes off it exactly as a drive does, and prints what
came back from both models against RIO's own deterministic state at the same
instant.

It is the tool the acceptance report is written from, and it is the tool to
reach for when a reading looks wrong: it produces a corpus under a session id
you choose, so the row can be opened afterwards with everything in it.

THE EGO HISTORY ON A CLIP
-------------------------
A video has no GPS and no IMU. The speed is supplied on the command line and
the yaw rate is zero, so the ego history is a straight line at a constant
speed -- which is a real degradation, is marked `yaw_rate: "none"` in every
row, and is the honest state for a desk replay. `--speed` exists so that at
least the magnitude is right: a model told the car is doing 2 m/s reasons
about a very different situation from one told 20.

IT IS NOT A DRIVE
-----------------
Frames go in as fast as the pipeline can take them rather than at the clip's
own rate, so the keyframe cadence in wall-clock terms is not a drive's. What is
faithful is everything the models see: the same four-frame window at the same
0.1 s spacing, the same t0, the same prompt set, the same association.
"""
import argparse
import json
import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import config                                                     # noqa: E402

config.TEACHERS_ENABLED = True

import framebuf                                                   # noqa: E402
from teachers import corpus as corpus_mod                         # noqa: E402
from teachers import egomotion                                    # noqa: E402
from teachers import panel                                        # noqa: E402


def _looks_annotated(path):
    """Cheap tell that a clip is a rendered output rather than footage. -> str

    Two things a camera does not produce and a renderer does: long perfectly
    horizontal or vertical runs of one saturated colour (box edges), and a
    solid letterboxed bar of near-constant colour at the very top (a HUD
    strip). Neither is conclusive on its own, which is why this warns rather
    than refuses.
    """
    try:
        import cv2
        import numpy as np
    except Exception:
        return ""
    cap = cv2.VideoCapture(path)
    cap.set(cv2.CAP_PROP_POS_FRAMES, 5)
    ok_, frame = cap.read()
    cap.release()
    if not ok_ or frame is None:
        return ""
    h, w = frame.shape[:2]
    hits = []
    # A HUD strip: the top rows almost uniform and much darker than the scene.
    top = frame[: max(2, h // 60)]
    if top.std() < 18 and top.mean() < 60 and frame.mean() > 70:
        hits.append("a dark uniform bar across the top")
    # Box edges: rows/columns with a long run of a single very saturated hue.
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    sat = hsv[:, :, 1] > 150
    val = hsv[:, :, 2] > 120
    mask = (sat & val).astype(np.uint8)
    runs = 0
    for axis, length in ((1, w), (0, h)):
        line = mask.sum(axis=axis)
        runs += int((line > length * 0.12).sum())
    if runs > (h + w) * 0.02:
        hits.append("long saturated straight edges, like drawn boxes")
    return "; ".join(hits)


def wrap(text, width=52, limit=None):
    """Hard-wrap for the side-by-side columns. -> list of lines."""
    t = " ".join(str(text or "").split())
    if limit and len(t) > limit:
        t = t[: limit - 1].rstrip() + "…"
    if not t:
        return ["—"]
    out, line = [], ""
    for word in t.split(" "):
        if len(line) + len(word) + 1 > width:
            out.append(line)
            line = word
        else:
            line = (line + " " + word).strip()
    if line:
        out.append(line)
    return out or ["—"]


def side_by_side(row, width=52):
    """One corpus row -> the block of text a person reads."""
    a = row["readings"]["alpamayo1.5"]
    c = row["readings"]["cosmos-reason2"]
    aa = row["associations"]["alpamayo1.5"]
    ca = row["associations"]["cosmos-reason2"]
    st = row["state"]

    def gap(v, unit=" m"):
        return "--" if v is None else f"{v:.1f}{unit}"

    head = (f"KEYFRAME {row['seq']:>3}  ({row['trigger']})   "
            f"t0 {row['t0_session'] if row['t0_session'] is not None else '--'}   "
            f"band {st['band']}   gap {gap(st['gap_m'])}   "
            f"TTC {gap(st['ttc_s'], ' s')}   "
            f"v {gap(st['speed_ms'], ' m/s')} ({st['speed_source']})")
    lines = ["", "=" * (width * 2 + 5), head, "-" * (width * 2 + 5)]
    if row.get("spoken"):
        lines.append(f"RIO said: “{row['spoken']['text']}”")
        lines.append("-" * (width * 2 + 5))

    def two(label, left, right, limit=None):
        lines.append(f"{label}")
        lv = wrap(left, width, limit)
        rv = wrap(right, width, limit)
        for i in range(max(len(lv), len(rv))):
            l = lv[i] if i < len(lv) else ""
            r = rv[i] if i < len(rv) else ""
            lines.append(f"  {l:<{width}} | {r}")

    lines.append(f"  {'ALPAMAYO 1.5':<{width}} | COSMOS-REASON2")
    lines.append(f"  {'-' * width} | {'-' * width}")
    def mark(rec, key):
        """A one-word note when this answer looks recited or looped."""
        f = (rec.get("flags") or {}).get(key)
        if not f:
            return ""
        if f.get("loop"):
            return f"   [LOOP x{f['loop']}]"
        if f.get("markers"):
            return "   [RECITED?]"
        return f"   [repeats x{f.get('repeated')}]"

    two("SCENE" + mark(a, "scene") + mark(c, "scene"), a["scene"], c["scene"])
    two("CHAIN-OF-CAUSATION / PHYSICAL REASONING"
        + mark(a, "reasoning") + mark(c, "reasoning"),
        a["reasoning"], c["reasoning"], limit=520)
    two("ATTENTION (NEXT 2 s)" + mark(a, "attention") + mark(c, "attention"),
        a["attention"], c["attention"])
    two("CRITICAL ACTOR" + mark(a, "critical_actor") + mark(c, "critical_actor"),
        a["critical_actor"], c["critical_actor"])

    def actor(x):
        if x.get("matched"):
            return (f"track #{x['track_id']} {x['label']}"
                    + (f" @ {x['range_m']} m" if x.get("range_m") else "")
                    + f"  [{x['reason']}]")
        return f"unmatched — {x.get('reason')}"

    # THE ROW ONLY ONE COLUMN CAN FILL, side by side like everything else --
    # Alpamayo's derived verdict on its own path against Cosmos's account of
    # the other road users. Neither model can produce the other's.
    dec = a.get("decision") or {}
    ph = c.get("physics") or {}
    dec_txt = (f"{dec.get('text')}   "
               f"[v {dec.get('v_start_ms')}->{dec.get('v_end_ms')} m/s, "
               f"a {dec.get('accel_ms2')} m/s2, lat {dec.get('lateral_end_m')} m]"
               if dec.get("text") else "no predicted path")
    imp = ph.get("implausible")
    ph_txt = ((ph.get("actors") or "") + "   [Plausibility: "
              + ("IMPLAUSIBLE" if imp is True else
                 "plausible" if imp is False else "no verdict")
              + (f" — {ph.get('plausibility')}" if ph.get("plausibility") else "")
              + "]") if ph else "—"
    two("DRIVING DECISION (derived)  |  PHYSICS (per actor)",
        dec_txt, ph_txt, limit=560)
    two("ASSOCIATED TRACK", actor(aa), actor(ca))
    two("LATENCY / PRECISION / FRESHNESS",
        f"{a['latency_ms']} ms · {a['precision']} · {a['freshness_s']} s",
        f"{c['latency_ms']} ms · {c['precision']} · {c['freshness_s']} s")
    traj = a.get("trajectory")
    if traj and traj.get("xyz"):
        end = traj["xyz"][-1]
        lines.append(f"PREDICTED PATH (display only): {len(traj['xyz'])} pts, "
                     f"{traj['horizon_s']} s, ends {end[0]:.1f} m ahead / "
                     f"{end[1]:+.1f} m lateral")
    agree = (aa.get("matched") and ca.get("matched")
             and aa.get("track_id") == ca.get("track_id"))
    lines.append(f"AGREE ON CRITICAL ACTOR: {'yes' if agree else 'no'}")
    return "\n".join(lines)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--clip", default="runs/road_clip.mp4")
    ap.add_argument("--keyframes", type=int, default=10)
    ap.add_argument("--speed", type=float, default=13.0,
                    help="host speed in m/s to give the ego history")
    ap.add_argument("--session", default=None,
                    help="session id for the corpus (default: replay-<clip>)")
    ap.add_argument("--stride", type=int, default=1,
                    help="use every Nth frame of the clip")
    ap.add_argument("--loops", type=int, default=200,
                    help="MAXIMUM passes over the clip. The run stops as soon "
                         "as --keyframes have been raised, so this is only a "
                         "guard against a clip that never triggers one")
    ap.add_argument("--max-seconds", type=float, default=900.0,
                    help="wall-clock cap on the feeding phase")
    ap.add_argument("--floor", type=float, default=None,
                    help="override config.TEACHER_KEYFRAME_FLOOR_S for this "
                         "run. USE IT FOR AN ACCEPTANCE RUN: at the shipped "
                         "2 s floor the faster teacher runs several keyframes "
                         "for every one the slower one gets to, so most rows "
                         "carry a single column. Pacing the floor to the "
                         "SLOWER teacher's latency makes every keyframe a "
                         "comparison, which is what a side-by-side reading is "
                         "for. It is not what a drive does, and the report "
                         "should say which it is looking at.")
    ap.add_argument("--alpamayo", default=None)
    ap.add_argument("--cosmos", default=None)
    ap.add_argument("--report", default=None)
    ap.add_argument("--json", default=None)
    args = ap.parse_args()

    import cv2

    if args.alpamayo:
        config.TEACHER_ALPAMAYO_URL = args.alpamayo
    if args.cosmos:
        config.TEACHER_COSMOS_URL = args.cosmos

    key = args.session or ("replay-" + os.path.basename(args.clip).split(".")[0])
    from headway import live as headway_live

    if args.floor is not None:
        config.TEACHER_KEYFRAME_FLOOR_S = float(args.floor)
        config.TEACHER_KEYFRAME_MIN_GAP_S = min(
            config.TEACHER_KEYFRAME_MIN_GAP_S, float(args.floor))

    # IS THIS CLIP ALREADY ANNOTATED?
    #
    # runs/road_clip.mp4 turned out to be a RENDERED OUTPUT of an earlier
    # headway run -- bounding boxes and "car 26.4m" labels burned into the
    # pixels -- and the first ten acceptance readings were taken on it. Both
    # teachers were shown a picture with the answers written on it, which makes
    # every reading from that run worthless as a measure of perception: a model
    # that says "a car 26 m ahead" may simply be reading the caption.
    #
    # Nothing downstream can detect that, because the frames are correct and
    # identical to the detector's by the time they reach the ring -- the
    # contamination is in the SOURCE. So it is checked here, where the source
    # is chosen, and it is a loud warning rather than a refusal: replaying an
    # annotated clip on purpose is a reasonable thing to want to do.
    warn = _looks_annotated(args.clip)
    print(f"clip     : {args.clip}")
    if warn:
        print(f"  !! WARNING: this clip looks ALREADY ANNOTATED ({warn}).")
        print(f"  !! The teachers will be shown a picture with the answers "
              f"written on it, and readings from it measure nothing.")
    print(f"floor    : {config.TEACHER_KEYFRAME_FLOOR_S} s"
          + ("  (paced for comparisons — not a drive's cadence)"
             if args.floor is not None else "  (as shipped)"))
    print(f"session  : {key}")
    print(f"teachers : {config.TEACHER_ALPAMAYO_URL}  "
          f"{config.TEACHER_COSMOS_URL}")
    panel.reset_all()
    panel.stop()
    panel.start()
    health = panel.refresh_health()
    for name, h in health.items():
        print(f"  {name:<16} loaded={h.get('loaded')} "
              f"{h.get('model_id')} @ {h.get('precision')} "
              f"vram={h.get('vram_reserved_mb')} MB")
    if not all(h.get("loaded") for h in health.values()):
        print("\n!! at least one service is not loaded — start them with:\n"
              "   bash boot.sh teachers\n")

    headway_live.reset_session(key)
    session = headway_live.get_session(key, use_qwen=True)
    framebuf.drop_ring(key)
    ring = framebuf.get_ring(key)
    corpus_mod.close(key)

    # KEYFRAMES ARE RAISED ON A WALL CLOCK, NOT PER FRAME. The floor is one
    # every ~2 s, and a five-second clip processed at ~20 ms a frame is over in
    # about two and a half seconds -- so "six passes over the clip" is not a
    # number of keyframes, it is an accident of how fast the pipeline is. The
    # clip loops until the TARGET is met, capped by wall clock and by a
    # generous pass count so a clip that somehow triggers nothing still ends.
    seen = set()
    t_start = time.time()
    stop = False
    for loop in range(args.loops):
        if time.time() - t_start > args.max_seconds:
            print(f"  !! stopping after {args.max_seconds:.0f}s with "
                  f"{len(seen)} keyframes", flush=True)
            break
        cap = cv2.VideoCapture(args.clip)
        if not cap.isOpened():
            raise SystemExit(f"cannot open {args.clip}")
        idx = 0
        while not stop:
            ok_, frame = cap.read()
            if not ok_:
                break
            idx += 1
            if args.stride > 1 and idx % args.stride:
                continue
            jpeg = cv2.imencode(".jpg", frame,
                                [int(cv2.IMWRITE_JPEG_QUALITY), 72])[1].tobytes()
            result = session.process(jpeg, args.speed, 0.0, None)
            ring.push(jpeg, result, origin=f"{key}:clip")
            egomotion.note_speed(key, args.speed, "manual", at=time.time())
            panel.on_frame(key, result, ring)
            n = panel.status()["sessions"].get(key, {}).get("keyframes", 0)
            if n > len(seen):
                seen.add(n)
                print(f"  keyframe {n}/{args.keyframes} raised at "
                      f"{time.time() - t_start:5.1f}s (band {result.get('band')})",
                      flush=True)
            if n >= args.keyframes:
                stop = True
            if time.time() - t_start > args.max_seconds:
                stop = True
        cap.release()
        if stop:
            break

    print("\nwaiting for the last readings to come back...", flush=True)
    # Bounded by PROGRESS, not just by a deadline: once no new row has landed
    # for a while, the teachers have answered everything they are going to and
    # sitting out the rest of the timeout tells nobody anything.
    deadline = time.time() + 600
    last_n, last_change = -1, time.time()
    while time.time() < deadline:
        rows = corpus_mod.read_rows(key)
        if len(rows) >= args.keyframes:
            break
        if len(rows) != last_n:
            last_n, last_change = len(rows), time.time()
        elif time.time() - last_change > 120:
            print(f"  !! no new row for 120 s; stopping with {len(rows)}",
                  flush=True)
            break
        time.sleep(2)
    panel.stop()
    corpus_mod.close(key)

    rows = corpus_mod.read_rows(key)
    print(f"\n{len(rows)} complete rows in "
          f"{corpus_mod.session_dir(key) / 'keyframes.jsonl'}")

    blocks = [side_by_side(r) for r in rows[: args.keyframes]]
    text = "\n".join(blocks)
    print(text)

    tally = panel.state(key)["tally"] if False else None
    agree = sum(1 for r in rows
                if r["associations"]["alpamayo1.5"].get("matched")
                and r["associations"]["cosmos-reason2"].get("matched")
                and r["associations"]["alpamayo1.5"]["track_id"]
                == r["associations"]["cosmos-reason2"]["track_id"])
    both = sum(1 for r in rows
               if r["readings"]["alpamayo1.5"].get("ok")
               and r["readings"]["cosmos-reason2"].get("ok"))
    matched = sum(1 for r in rows for m in ("alpamayo1.5", "cosmos-reason2")
                  if r["associations"][m].get("matched"))
    named = sum(1 for r in rows for m in ("alpamayo1.5", "cosmos-reason2")
                if (r["readings"][m].get("critical_actor") or "").strip())
    summary = (f"\n{'=' * 109}\n"
               f"{len(rows)} keyframes · both answered on {both} · "
               f"agreed on the critical actor {agree}/{both} · "
               f"matched to a track {matched}/{named}\n{'=' * 109}")
    print(summary)

    if args.report:
        with open(args.report, "w") as f:
            f.write("```\n" + text + summary + "\n```\n")
        print(f"wrote {args.report}")
    if args.json:
        with open(args.json, "w") as f:
            json.dump(rows[: args.keyframes], f, indent=2)
        print(f"wrote {args.json}")
    return 0 if rows else 1


if __name__ == "__main__":
    sys.exit(main())
