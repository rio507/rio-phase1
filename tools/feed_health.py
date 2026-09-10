"""feed_health.py — was the car looking, and how old was what it saw?

    python -m tools.feed_health training_data/<session>.jsonl

Two questions about one drive, and the drive of 2026-09-09 could answer
neither from anything in its own log:

  WAS THE FEED ALIVE?     Frames arrive as `headway` events with a frame_idx,
                          so a gap in the indices next to a gap in the wall
                          clock is a stall, and its length is how long the car
                          was not looking. Session 738fbb82 has one 442 s
                          long, starting 40 s into the drive.

  HOW OLD WAS THE PICTURE the detector ran on? `frame_age_ms` is capture to
                          detection-complete in one clock. It is the number a
                          gap warning's truthfulness rests on, and the number
                          look() now cites.

...and one about the observer, because "the observer went quiet" and "the
observer is describing a road the car left minutes ago" are different faults
with the same symptom from the passenger seat:

  WAS THE DESCRIPTION     visual_qa events carry the ring's state and the
  CURRENT?                chosen frame's age. A ring with `frames: 0` is an
                          honest refusal; a chosen frame several seconds old
                          is an answer about history stated as now.

Everything here is read out of the session JSONL. Nothing is inferred, and a
number that is not in the log is printed as a dash rather than as a guess.
"""
import argparse
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


def read(path):
    rows = []
    with open(path, encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            try:
                rows.append(json.loads(line))
            except Exception:
                continue
    return rows


def pct(values, p):
    if not values:
        return None
    v = sorted(values)
    return v[min(len(v) - 1, int(round((len(v) - 1) * p)))]


def fmt(v, unit="", nd=0):
    if v is None:
        return "     —"
    return f"{v:,.{nd}f}{unit}"


def report(path, stall_s=3.0):
    rows = read(path)
    if not rows:
        print(f"{path}: nothing to read")
        return 2
    t0 = rows[0]["t"]
    t_end = rows[-1]["t"]
    hw = [r for r in rows if r.get("kind") == "headway"]
    vq = [r for r in rows if r.get("kind") == "visual_qa"]
    marks = [r for r in rows if r.get("kind") == "mark"]

    print(f"\n{os.path.basename(path)}")
    print(f"  {t_end - t0:.0f} s of session, {len(hw)} frame results, "
          f"{len(vq)} visual answers")

    # --- was the feed alive? -------------------------------------------------
    print("\n  THE FEED")
    if not hw:
        print("    no frame results at all")
    else:
        gaps = []
        prev = None
        for r in hw:
            if prev is not None and (r["t"] - prev["t"]) > stall_s:
                gaps.append((prev["t"] - t0, r["t"] - prev["t"],
                             prev["payload"].get("frame_idx"),
                             r["payload"].get("frame_idx")))
            prev = r
        covered = sum(g[1] for g in gaps)
        span = hw[-1]["t"] - hw[0]["t"]
        print(f"    first result at {hw[0]['t'] - t0:6.1f} s, "
              f"last at {hw[-1]['t'] - t0:6.1f} s")
        print(f"    stalls over {stall_s:.0f} s: {len(gaps)}, "
              f"{covered:.0f} s total "
              f"({100.0 * covered / span if span else 0:.0f}% of the span)")
        for at, length, a, b in gaps:
            print(f"      at {at:6.1f} s  for {length:6.1f} s   "
                  f"(frame {a} -> {b})")
        # ...and the transport's own account of itself, where the page left one.
        for tag in ("FRAMES_STALLED", "FRAMES_STATE", "FRAMES_WS_LOST",
                    "FRAMES_TICK_OVERRUN", "FRAMES_TRACK_ENDED",
                    "FRAMES_WS_UNANSWERED", "FRAMES_VISIBLE"):
            hits = [m for m in marks if m["payload"].get("tag") == tag]
            if hits:
                print(f"    {tag}: {len(hits)}")
                for m in hits[:6]:
                    print(f"      {m['t'] - t0:6.1f} s  {m['payload'].get('note','')[:110]}")
        if not any(m["payload"].get("tag", "").startswith("FRAMES_") and
                   m["payload"].get("tag") != "FRAMES_WS_READY" for m in marks):
            print("    the page left no transport events at all — which on a "
                  "drive with a stall in it is itself the finding")

    # --- how old was the picture? -------------------------------------------
    ages = [r["payload"].get("frame_age_ms") for r in hw
            if isinstance(r["payload"].get("frame_age_ms"), (int, float))]
    print("\n  FRAME AGE AT DETECTION  (capture -> detection complete)")
    if not ages:
        print("    no frame carried a capture timestamp")
    else:
        print(f"    n={len(ages)}   p50 {fmt(pct(ages, .5), ' ms')}   "
              f"p90 {fmt(pct(ages, .9), ' ms')}   "
              f"p99 {fmt(pct(ages, .99), ' ms')}   "
              f"max {fmt(max(ages), ' ms')}")
        over = [a for a in ages if a > 1000]
        print(f"    over 1 s: {len(over)} of {len(ages)} "
              f"({100.0 * len(over) / len(ages):.1f}%)")

    # --- and was the description current? -----------------------------------
    print("\n  WHAT look() ANSWERED FROM")
    if not vq:
        print("    nothing was asked")
    for r in vq:
        p = r["payload"]
        ring = p.get("ring") or {}
        sel = p.get("frame_selection") or {}
        age = sel.get("chosen_age_s")
        unavailable = p.get("visual_unavailable")
        print(f"    {r['t'] - t0:6.1f} s  {str(p.get('question'))[:44]!r:46} "
              f"ring={ring.get('frames', '?'):>3} "
              f"oldest={fmt(ring.get('oldest_age_s'), ' s', 1):>8} "
              + (f"chose a frame {age} s old" if age is not None
                 else (f"REFUSED: {unavailable}" if unavailable
                       else "no frame chosen")))
    chosen = [(p["payload"].get("frame_selection") or {}).get("chosen_age_s")
              for p in vq]
    chosen = [c for c in chosen if isinstance(c, (int, float))]
    if chosen:
        print(f"    chosen frame age: p50 {pct(chosen, .5)} s, "
              f"max {max(chosen)} s")
    refused = sum(1 for r in vq if r["payload"].get("visual_unavailable"))
    print(f"    answers refused for want of a frame: {refused} of {len(vq)}")
    return 0


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("sessions", nargs="+")
    ap.add_argument("--stall-s", type=float, default=3.0,
                    help="a gap longer than this between frame results")
    args = ap.parse_args()
    rc = 0
    for path in args.sessions:
        rc = report(path, args.stall_s) or rc
    return rc


if __name__ == "__main__":
    sys.exit(main())
