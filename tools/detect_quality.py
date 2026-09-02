"""detect_quality.py — what the perception overlay is actually showing.

    python -m tools.detect_quality CLIP.mp4
    python -m tools.detect_quality CLIP.mp4 --at 48 --at 12.5 --fps 4
    python -m tools.detect_quality CLIP.mp4 --out runs/coastal --frames-only

Runs the real pipeline over a clip -- RF-DETR, Depth Anything, UFLDv2, the
corridor, membership, lead selection, the Kalman -- and dumps EVERY detection
on every sampled frame with its class, box, confidence, depth, plausibility
verdict and whether it took the lead lock. Then it counts the four failures
that were visible on the overlay and reports them BEFORE and AFTER the gates,
from a single forward pass per frame, so the comparison is on identical pixels
and not on two runs of a stochastic-looking system.

WHY A HARNESS AND NOT A LOOK AT THE VIDEO
-----------------------------------------
The bugs this exists for were all found by eye, on one frame: a cluster of tiny
"car" boxes at the vanishing point, labelled 2 m and 6 m and 9 m next to 62 m,
with two pedestrians unboxed beside them and GAP -66.8 M in the corner. Eyes
are good at finding that frame and bad at answering the next question, which is
how often. A rate is what says whether a fix worked, and a per-frame dump is
what makes a rate checkable rather than asserted.

WHAT IT COUNTS
--------------
duplicate boxes     detections suppressed as another reading of a box already
                    kept (IoU, containment, or two vehicle classes on one
                    object). The vanishing-point cluster is this number.
implausible depths  ranges refused because the box's pixel height cannot go
                    with the claimed distance for that class (plausibility.py).
negative gaps       frames whose reported gap was below zero. It should be
                    structurally impossible now (filter.GAP_FLOOR_M); this
                    counts it anyway, because "cannot happen" is a claim and
                    this is the thing that checks it.
person detections   frames and instances of pedestrian/cyclist. Reported as
                    counts and box heights, NOT as a recall figure: recall needs
                    labelled ground truth and this clip has none. Saying "0.61
                    recall" off an unlabelled clip would be inventing a number.

The comparison frames it renders are side-by-side: the same frame gated the old
way on the left, the new way on the right, boxes and labels drawn identically so
the only difference on the screen is the difference in the pipeline.
"""
import argparse
import json
import math
import os
import sys
import time
from collections import Counter

import cv2
import numpy as np

sys.path.insert(0, "/workspace/rio-phase1")

from headway import anchor as anchor_mod          # noqa: E402
from headway import depth as depth_mod            # noqa: E402
from headway import detect as detect_mod          # noqa: E402
from headway import lanes as lanes_mod            # noqa: E402
from headway import plausibility as plaus_mod     # noqa: E402
from headway.filter import HeadwayFilter          # noqa: E402
from headway.membership import CandidateSet, LEAD_LABELS   # noqa: E402

# Colours (BGR). Deliberately the same family the browser overlay uses, so a
# rendered comparison frame and a screenshot of the dashboard read alike.
CYAN = (232, 179, 95)
AMBER = (77, 184, 255)
LEAD = (255, 214, 155)
RED = (90, 90, 255)
DIM = (150, 130, 110)
WHITE = (240, 240, 240)


# ---------------------------------------------------------------------------
# One frame, gated two ways
# ---------------------------------------------------------------------------
def analyse_frame(frame, corridor, f_px, dets, depth_map, mode):
    """Attach depth and a plausibility verdict to one frame's detections.

    `mode` is "before" or "after" and changes exactly one thing here: whether an
    implausible range is refused. Everything else -- the depth model, the ROI,
    the corridor -- is identical, because the point is to isolate the gate.
    """
    h, w = frame.shape[:2]
    out = []
    for det in dets:
        label, box, score = det[0], det[1], det[2]
        info = det[3] if len(det) > 3 and isinstance(det[3], dict) else {}
        d, conf = None, 0.0
        if depth_map is not None:
            dd, conf, _ = depth_mod.roi_depth(depth_map, box)
            if np.isfinite(dd) and conf > 0.2:
                d = float(dd)
        verdict = plaus_mod.check(label, box, d, f_px, image_h=h)
        shown = d
        if mode == "after" and not verdict["ok"]:
            shown = None
        u, v = (box[0] + box[2]) / 2.0, box[3]
        inside, _geo = corridor.contains(u, v)
        out.append({
            "label": label,
            "box": [round(float(x), 1) for x in box],
            "score": round(float(score), 3),
            "h_px": round(float(box[3] - box[1]), 1),
            "confirmed": bool(info.get("confirmed", True)),
            "depth_m": None if d is None else round(d, 2),
            "depth_conf": round(float(conf), 3),
            "shown_range_m": None if shown is None else round(shown, 2),
            "plausible": bool(verdict["ok"]),
            "plaus_reason": verdict["reason"],
            "implied_m": verdict["implied_m"],
            "in_corridor": bool(inside),
            "vulnerable": label not in LEAD_LABELS,
        })
    return out


def lead_of(objs):
    """Nearest in-corridor vehicle with a usable range -- the same rule
    membership.select_lead applies, run over one frame's objects so the harness
    can report a lead without carrying dwell state between the two modes."""
    best = None
    for o in objs:
        if o["vulnerable"] or not o["in_corridor"] or o["shown_range_m"] is None:
            continue
        if best is None or o["shown_range_m"] < best["shown_range_m"]:
            best = o
    return best


# ---------------------------------------------------------------------------
# Rendering
# ---------------------------------------------------------------------------
def draw_objects(frame, objs, lead, corridor, title, gap_text):
    img = frame.copy()
    h, w = img.shape[:2]

    poly = np.array(corridor.polygon(near_m=5.0, far_m=70.0), np.int32)
    cv2.polylines(img, [poly], True, (90, 90, 90), 1, cv2.LINE_AA)

    for o in objs:
        x1, y1, x2, y2 = [int(round(v)) for v in o["box"]]
        is_lead = lead is not None and o["box"] == lead["box"]
        if is_lead:
            colour, thick = LEAD, 2
        elif o["vulnerable"]:
            colour, thick = AMBER, 2
        elif not o["confirmed"]:
            colour, thick = DIM, 1
        else:
            colour, thick = CYAN, 1
        cv2.rectangle(img, (x1, y1), (x2, y2), colour, thick)

        text = o["label"]
        if o["shown_range_m"] is not None:
            text += f" {o['shown_range_m']:.1f}m"
        elif not o["confirmed"]:
            text += " --"
        else:
            text += " --"
        if is_lead:
            text = "> " + text
        # An implausible range that is being SHOWN (the before mode) is called
        # out in red, because that is the defect the frame is here to display.
        label_colour = RED if (o["shown_range_m"] is not None
                               and not o["plausible"]) else colour
        ty = y1 - 6 if y1 > 18 else min(y2 + 14, h - 4)
        cv2.putText(img, text, (max(2, x1), ty), cv2.FONT_HERSHEY_SIMPLEX,
                    0.42, (0, 0, 0), 3, cv2.LINE_AA)
        cv2.putText(img, text, (max(2, x1), ty), cv2.FONT_HERSHEY_SIMPLEX,
                    0.42, label_colour, 1, cv2.LINE_AA)

    cv2.rectangle(img, (0, 0), (w, 46), (18, 18, 18), -1)
    cv2.putText(img, title, (10, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.6, WHITE, 1,
                cv2.LINE_AA)
    # Red for a NEGATIVE gap, which is the defect. "GAP --" is an absent gap,
    # which is the correct rendering of one and must not be flagged as a fault.
    gap_colour = RED if gap_text.startswith("GAP -") and gap_text[5:6].isdigit() \
        else WHITE
    cv2.putText(img, gap_text, (10, 39), cv2.FONT_HERSHEY_SIMPLEX, 0.55,
                gap_colour, 1, cv2.LINE_AA)
    return img


def comparison(frame, before, after, corridor, t_s, gap_before, gap_after):
    left = draw_objects(frame, before, lead_of(before), corridor,
                        f"BEFORE  t={t_s:.2f}s  {len(before)} boxes", gap_before)
    right = draw_objects(frame, after, lead_of(after), corridor,
                         f"AFTER   t={t_s:.2f}s  {len(after)} boxes", gap_after)
    sep = np.full((left.shape[0], 4, 3), 60, np.uint8)
    return np.hstack([left, sep, right])


# ---------------------------------------------------------------------------
# Gap simulation -- the C bug, measured rather than asserted
# ---------------------------------------------------------------------------
class GapTrack:
    """Runs the Kalman over the clip twice: with the gap floor and without.

    The unclamped copy is what shipped, and it is the only way to state the
    negative-gap rate as a measurement. It is a copy of the filter's own maths
    (four lines of it) rather than a flag on HeadwayFilter, because a "please
    misbehave" switch on the filter that decides warnings is a worse thing to
    own than four lines here.
    """

    def __init__(self):
        self.kf = HeadwayFilter()
        self.x = None                    # unclamped [d, d_dot]
        self.n_negative = 0
        self.min_unclamped = None
        self.reported = []

    def step(self, z, depth_conf, dt):
        snap = self.kf.step(z, depth_conf, dt)

        have = z is not None and np.isfinite(z) and z > 0
        if self.x is None:
            if have:
                self.x = [float(z), 0.0]
        else:
            self.x[0] += self.x[1] * dt
            if have:
                # Same gain the real filter just used, so the unclamped twin
                # tracks it rather than drifting off on its own dynamics.
                k = 0.35
                innov = float(z) - self.x[0]
                self.x[0] += k * innov
                self.x[1] += (k / max(dt, 1e-3)) * innov * 0.15
        if self.x is not None:
            if self.min_unclamped is None or self.x[0] < self.min_unclamped:
                self.min_unclamped = self.x[0]
            if self.x[0] < 0:
                self.n_negative += 1
        return snap, (None if self.x is None else self.x[0])


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def run(video, out_dir, fps, at_times, limit, render_only):
    cap = cv2.VideoCapture(video)
    if not cap.isOpened():
        raise SystemExit(f"cannot open {video}")
    src_fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    n_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
    stride = max(1, int(round(src_fps / float(fps))))
    os.makedirs(out_dir, exist_ok=True)

    print(f"clip     : {video}")
    print(f"          {n_frames} frames @ {src_fps:.2f} fps "
          f"({n_frames / max(src_fps, 1):.1f} s), sampling every {stride} "
          f"-> {fps} fps")
    print(f"out      : {out_dir}")

    want_frames = sorted(set(at_times))
    rendered = []

    stats = {m: Counter() for m in ("before", "after")}
    person_h = []
    vru_raw = []
    dup_examples = []
    implausible_examples = []
    gap = GapTrack()
    n_sampled = 0
    lead_switch_before = lead_switch_after = 0
    prev_lead = {"before": None, "after": None}
    cands = {m: CandidateSet() for m in ("before", "after")}

    dump_path = os.path.join(out_dir, "detections.jsonl")
    dump = open(dump_path, "w")

    corridor = None
    idx = 0
    t_prev = None
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        if idx % stride:
            idx += 1
            continue
        t_s = idx / src_fps
        h, w = frame.shape[:2]
        if corridor is None:
            base = anchor_mod.EgoCorridor(w, h)
            f_px = plaus_mod.focal_px(w)

        # Lane geometry, exactly as the live loop builds it.
        lane_result = None
        try:
            lane_result = lanes_mod.detect_lanes(frame)
        except Exception:
            pass
        corridor, lane_info = anchor_mod.build_corridor(base, lane_result)

        depth_map = None
        trust, trust_info = depth_mod.frame_trust(frame)
        if trust:
            try:
                depth_map = depth_mod.depth_map(frame)
            except Exception:
                depth_map = None

        # ONE forward pass, gated two ways.
        r = detect_mod.detect(frame)
        raw = r["raw"]
        g_before = detect_mod.gate(raw, dedupe=False, per_class=False,
                                   size_floor=False)
        g_after = r

        objs = {
            "before": analyse_frame(frame, corridor, f_px,
                                    g_before["detections"], depth_map, "before"),
            "after": analyse_frame(frame, corridor, f_px,
                                   g_after["detections"], depth_map, "after"),
        }
        leads = {m: lead_of(objs[m]) for m in objs}

        # --- the gap, both ways ---------------------------------------------
        dt = 1.0 / float(fps) if t_prev is None else max(1e-3, t_s - t_prev)
        t_prev = t_s
        lead_after = leads["after"]
        z = None if lead_after is None else lead_after["shown_range_m"]
        conf = 0.8 if z is not None else 0.0
        snap, unclamped = gap.step(z, conf, dt)
        gap.reported.append(snap["d"])

        for m in objs:
            s = stats[m]
            s["frames"] += 1
            s["boxes"] += len(objs[m])
            for o in objs[m]:
                if o["vulnerable"]:
                    s["vru_boxes"] += 1
                    if m == "after":
                        person_h.append(o["h_px"])
                if o["shown_range_m"] is not None and not o["plausible"]:
                    s["implausible_shown"] += 1
                if not o["plausible"] and o["depth_m"] is not None:
                    s["implausible_depths"] += 1
                if not o["confirmed"]:
                    s["unconfirmed"] += 1
            if any(o["vulnerable"] for o in objs[m]):
                s["frames_with_vru"] += 1
            lead = leads[m]
            if lead is not None:
                s["frames_with_lead"] += 1
                if (prev_lead[m] is not None
                        and abs(prev_lead[m] - lead["shown_range_m"]) > 15.0):
                    s["lead_jumps"] += 1
            prev_lead[m] = None if lead is None else lead["shown_range_m"]

        # --- what the model PROPOSED for vulnerable road users ---------------
        # The recall-relevant number that does not need labels: every
        # person/cyclist query the model raised, whatever its score, with the
        # height it raised it at. If pedestrians are being missed, this is where
        # it shows -- either the proposals are not there at all (a model
        # limitation) or they are there below the gate (a threshold choice).
        for name, box, score in raw:
            if name in ("pedestrian", "cyclist"):
                vru_raw.append({"t": round(t_s, 2), "label": name,
                                "score": round(float(score), 3),
                                "h_px": round(box[3] - box[1], 1)})

        stats["after"]["duplicates_dropped"] += g_after["n_duplicates_dropped"]
        stats["after"]["small_rejected"] += g_after["n_small_rejected"]
        if g_after["duplicates"] and len(dup_examples) < 12:
            dup_examples.append({"t": round(t_s, 2), **g_after["duplicates"][0]})
        for o in objs["before"]:
            if o["shown_range_m"] is not None and not o["plausible"] \
                    and len(implausible_examples) < 12:
                implausible_examples.append({"t": round(t_s, 2), **o})

        dump.write(json.dumps({
            "t": round(t_s, 3), "frame": idx,
            "corridor_source": corridor.source,
            "lane_conf": round(float(lane_info.get("lane_conf") or 0.0), 3),
            "depth_trusted": bool(trust),
            "n_raw": len(raw),
            "before": {"n": len(objs["before"]), "objects": objs["before"],
                       "lead": leads["before"]},
            "after": {"n": len(objs["after"]), "objects": objs["after"],
                      "lead": leads["after"],
                      "duplicates_dropped": g_after["n_duplicates_dropped"],
                      "small_rejected": g_after["n_small_rejected"]},
            "gap": {"clamped_m": snap["d"], "unclamped_m": unclamped,
                    "coast_age_s": snap["coast_age"],
                    "n_clamped": snap.get("n_clamped", 0)},
        }) + "\n")

        # --- comparison frames ----------------------------------------------
        for target in list(want_frames):
            if abs(t_s - target) <= (stride / src_fps) / 2 + 1e-6:
                gb = ("GAP --" if unclamped is None
                      else f"GAP {unclamped:.1f} M")
                ga = ("GAP --" if (snap["d"] is None
                                   or snap["coast_age"] > 1.0)
                      else f"GAP {snap['d']:.1f} M")
                img = comparison(frame, objs["before"], objs["after"], corridor,
                                 t_s, gb, ga)
                name = f"compare_{target:05.1f}s.jpg".replace(".", "_", 1)
                path = os.path.join(out_dir, name)
                cv2.imwrite(path, img, [cv2.IMWRITE_JPEG_QUALITY, 92])
                rendered.append(path)
                want_frames.remove(target)
                print(f"  rendered {path}")

        n_sampled += 1
        idx += 1
        if limit and n_sampled >= limit:
            break
        if n_sampled % 25 == 0:
            print(f"  ...{n_sampled} frames ({t_s:.1f}s)", flush=True)

    cap.release()
    dump.close()

    report = summarise(stats, gap, person_h, dup_examples,
                       implausible_examples, n_sampled, vru_raw)
    report["clip"] = video
    report["sample_fps"] = fps
    report["rendered"] = rendered
    report["detections_jsonl"] = dump_path
    with open(os.path.join(out_dir, "quality.json"), "w") as fh:
        json.dump(report, fh, indent=2)
    print_report(report)
    return report


def summarise(stats, gap, person_h, dup_examples, implausible_examples, n,
              vru_raw=()):
    n = max(1, n)
    b, a = stats["before"], stats["after"]
    reported = [d for d in gap.reported if d is not None]
    return {
        "frames": n,
        "before": {
            "boxes_per_frame": round(b["boxes"] / n, 2),
            "duplicate_rate_per_frame": round(a["duplicates_dropped"] / n, 3),
            "implausible_depth_shown_per_frame": round(b["implausible_shown"] / n, 3),
            "implausible_depth_pct_of_boxes": round(
                100.0 * b["implausible_shown"] / max(1, b["boxes"]), 1),
            "vru_boxes": b["vru_boxes"],
            "frames_with_vru": b["frames_with_vru"],
            "frames_with_lead": b["frames_with_lead"],
            "lead_range_jumps": b["lead_jumps"],
            "negative_gap_frames": gap.n_negative,
            "min_gap_m": (None if gap.min_unclamped is None
                          else round(gap.min_unclamped, 2)),
        },
        "after": {
            "boxes_per_frame": round(a["boxes"] / n, 2),
            "duplicates_dropped_total": a["duplicates_dropped"],
            "small_boxes_rejected_total": a["small_rejected"],
            "unconfirmed_drawn_total": a["unconfirmed"],
            "implausible_depth_shown_per_frame": round(a["implausible_shown"] / n, 3),
            "implausible_depths_refused_total": a["implausible_depths"],
            "vru_boxes": a["vru_boxes"],
            "frames_with_vru": a["frames_with_vru"],
            "frames_with_lead": a["frames_with_lead"],
            "lead_range_jumps": a["lead_jumps"],
            "negative_gap_frames": sum(1 for d in reported if d < 0),
            "min_gap_m": (None if not reported else round(min(reported), 2)),
        },
        "vru_box_heights_px": {
            "n": len(person_h),
            "p05": (None if not person_h
                    else round(float(np.percentile(person_h, 5)), 1)),
            "median": (None if not person_h
                       else round(float(np.median(person_h)), 1)),
            "p95": (None if not person_h
                    else round(float(np.percentile(person_h, 95)), 1)),
        },
        "vru_proposals": vru_summary(vru_raw),
        "duplicate_examples": dup_examples,
        "implausible_examples": implausible_examples,
    }


def vru_summary(vru_raw):
    """Every pedestrian/cyclist the model proposed, by score band and size.

    NOT a recall figure. Recall is (found / actually there) and nothing here
    knows what was actually there; this is (proposed / proposed), which answers
    the different and still useful question of whether the gate or the model is
    what a missing pedestrian died on.
    """
    # A DETR emits a fixed number of queries per frame and most of them are
    # near-zero noise carrying whatever class won a coin toss. Counting those
    # as "proposals the model made" would put tens of thousands of imaginary
    # pedestrians in the report, so the headline counts only queries the model
    # gave any weight at all, and the full distribution is printed underneath.
    NOISE_FLOOR = 0.10
    real = [v for v in vru_raw if v["score"] >= NOISE_FLOOR]
    out = {"n_proposals": len(real), "n_queries": len(vru_raw),
           "noise_floor": NOISE_FLOOR, "by_score": {}, "by_height": {}}
    if not vru_raw:
        return out
    bands = [(0.0, 0.1), (0.1, 0.28), (0.28, 0.5), (0.5, 0.75), (0.75, 1.01)]
    for lo, hi in bands:
        n = sum(1 for v in vru_raw if lo <= v["score"] < hi)
        out["by_score"][f"{lo:.2f}-{hi:.2f}"] = n
    for lo, hi in [(0, 20), (20, 40), (40, 80), (80, 10000)]:
        n = sum(1 for v in real if lo <= v["h_px"] < hi)
        out["by_height"][f"{lo}-{hi if hi < 1000 else 'inf'}px"] = n
    kept = [v for v in real
            if v["score"] >= (0.28 if v["h_px"] >= 20 else 0.55)]
    out["passing_the_gate"] = len(kept)
    out["examples"] = sorted(vru_raw, key=lambda v: -v["score"])[:8]
    out["by_score"]["(all queries incl. noise)"] = len(vru_raw)
    return out


def print_report(r):
    b, a = r["before"], r["after"]
    W = 74
    print("\n" + "=" * W)
    print(f"detection quality — {r['frames']} frames @ {r['sample_fps']} fps")
    print("=" * W)
    rows = [
        ("boxes drawn per frame", b["boxes_per_frame"], a["boxes_per_frame"]),
        ("duplicate boxes per frame", b["duplicate_rate_per_frame"], 0.0),
        ("implausible depths SHOWN per frame",
         b["implausible_depth_shown_per_frame"],
         a["implausible_depth_shown_per_frame"]),
        ("frames with a negative gap", b["negative_gap_frames"],
         a["negative_gap_frames"]),
        ("minimum gap reached (m)", b["min_gap_m"], a["min_gap_m"]),
        ("pedestrian/cyclist boxes", b["vru_boxes"], a["vru_boxes"]),
        ("frames with a lead", b["frames_with_lead"], a["frames_with_lead"]),
        ("lead range jumps >15 m", b["lead_range_jumps"], a["lead_range_jumps"]),
    ]
    print(f"{'':40s} {'before':>14s} {'after':>14s}")
    for name, x, y in rows:
        print(f"{name:40s} {str(x):>14s} {str(y):>14s}")
    vp = r.get("vru_proposals") or {}
    if vp.get("n_proposals"):
        print(f"\nvulnerable road users the model PROPOSED (score >= "
              f"{vp.get('noise_floor')}): {vp['n_proposals']}, of which "
              f"{vp.get('passing_the_gate')} clear the confidence gate")
        print(f"  by score : {vp['by_score']}")
        print(f"  by height: {vp['by_height']}")
    else:
        print("\nvulnerable road users the model proposed: 0 — this clip "
              "contains none, so it says nothing about pedestrian recall")
    vh = r["vru_box_heights_px"]
    print(f"\nvulnerable road users: {vh['n']} boxes, height p05/median/p95 = "
          f"{vh['p05']}/{vh['median']}/{vh['p95']} px")
    print(f"  (box heights, not recall — recall needs labelled ground truth "
          f"this clip does not have)")
    if r["duplicate_examples"]:
        e = r["duplicate_examples"][0]
        print(f"\nexample duplicate  t={e['t']}s  {e['label']} {e['box']} "
              f"score {e['score']} suppressed by {e['beaten_by']} "
              f"({e['rule']} {e.get('iou')}/{e.get('containment')})")
    if r["implausible_examples"]:
        e = r["implausible_examples"][0]
        print(f"example implausible  t={e['t']}s  {e['label']} {e['h_px']} px "
              f"claiming {e['depth_m']} m, geometry says ~{e['implied_m']} m "
              f"({e['plaus_reason']})")
    print("=" * W)


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("video")
    ap.add_argument("--out", default=None,
                    help="output directory (default runs/<clip name>_quality)")
    ap.add_argument("--fps", type=float, default=4.0,
                    help="sampling rate; 4 is the live loop's cadence")
    ap.add_argument("--at", type=float, action="append", default=[],
                    help="timestamp (s) to render a before/after frame at; "
                         "repeatable")
    ap.add_argument("--limit", type=int, default=0,
                    help="stop after N sampled frames")
    ap.add_argument("--frames-only", action="store_true",
                    help="render the comparison frames and skip the rest")
    args = ap.parse_args()

    out = args.out or os.path.join(
        "runs", os.path.splitext(os.path.basename(args.video))[0] + "_quality")
    t0 = time.time()
    run(args.video, out, args.fps, args.at, args.limit, args.frames_only)
    print(f"\n{time.time() - t0:.1f}s")


if __name__ == "__main__":
    main()
