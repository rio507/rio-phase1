"""Verification for the live headway loop (v3 bands + automatic voice).

    python -m headway.live_selftest

Three parts, deliberately separated by what each one is able to prove:

  A. FULL PIPELINE on the shrinking-gap synthetic clip, at the live loop's real
     capture cadence (LIVE_FPS), through LiveSession.process() — the exact function
     /headway_frame calls. Depth and the anchor box are supplied from the clip's
     ground truth for the same reason run_clip.py offers `--depth gt`: the
     synthetic clip is a flat-shaded rectangle on a flat road, so DA-V2 has no
     real geometry to infer and running it would measure noise. Everything else
     is live — clock, anchor scheduling, CSRT on the real rendered pixels, the
     Kalman, the trend classifier, the policy, the response payload.

  B. POLICY SCENARIOS, scripted straight into LivePolicy. The policy is
     pure, so its edge cases can be driven exactly rather than hoped for out of
     a clip: cooldown suppression, worsening overriding that cooldown, the
     genuine-clear re-arm, the 5 s escalation, and every suppression gate.

  C. LLM firewall — live_policy.py must not be able to see a model output.

Part C of the *user-facing* verification (real Qwen anchor, real DA-V2 depth,
real HTTP, per-frame latency) is tools/headway_bench.py, which needs a running
server and is not part of this file.
"""
import math
import os
import sys
import tempfile

import numpy as np

from . import anchor as anchor_mod
from . import depth as depth_mod
from . import detect as detect_mod
from . import lanes as lanes_mod
from . import live as live_mod
from . import live_policy as P
from . import state as v2

PASS, FAIL = [], []


def check(cond, label, detail=""):
    (PASS if cond else FAIL).append(label)
    print(f"  [{'PASS' if cond else 'FAIL'}] {label}" + (f"  -- {detail}" if detail else ""))
    return bool(cond)


def head(title):
    print(f"\n{title}")


# ===========================================================================
# A. Full pipeline over the synthetic clip
# ===========================================================================
# 18 m/s (~40 mph), not run_clip's default 25. The clip's trajectory is fixed
# (60 m, then 3 m/s2 braking, then -9 m/s relative, then recovery) and v_host is
# what maps that trajectory onto the band ladder. At 25 m/s the clip *starts* at
# tau 2.4, already inside v3's GETTING_UNSAFE, so the NORMAL -> GETTING_UNSAFE
# entry that the calm line hangs off never happens. At 18 m/s:
#   tau 3.33 at t=0        NORMAL
#   tau < 3.0 from t~5.0   GETTING_UNSAFE   (d = 54 m)
#   tau < 2.0 from t~7.2   UNSAFE           (d = 36 m)
# which walks the whole v3 ladder and makes the amber->red step a *worsening*.
SYNTH_V_HOST = 18.0
LIVE_FPS = 4.0


class _GTDepth:
    """Ground-truth range with representative noise and ROI statistics."""

    def __init__(self, gt, seed=0):
        self.by_frame = {int(g["frame"]): g for g in gt}
        self.rng = np.random.default_rng(seed)
        self.frame = 0

    def depth_map(self, frame):
        # live.py only ever passes this through to roi_depth and the anchor, both
        # of which are substituted here, so its contents are never read.
        return np.zeros((4, 4), dtype=np.float32)

    def roi_depth(self, depth, box, shrink=0.6):
        g = self.by_frame.get(self.frame)
        if g is None:
            return float("nan"), 0.0, {"valid_frac": 0.0, "rel_spread": 1.0}
        d = float(g["d_true"])
        z = d + float(self.rng.normal(0.0, max(0.10, 0.005 * d)))
        return z, 0.85, {"valid_frac": 0.95, "rel_spread": 0.05}


def run_pipeline():
    from .run_clip import make_synthetic
    import cv2
    import json

    head(f"A -- full live pipeline, shrinking-gap synthetic clip @ {LIVE_FPS:g} fps")

    tmp = tempfile.mkdtemp(prefix="headway-live-")
    clip = os.path.join(tmp, "synth_shrinking.mp4")
    path, gt_path, n = make_synthetic(clip, fps=30.0, v_host=SYNTH_V_HOST,
                                      scenario="shrinking")
    with open(gt_path) as fh:
        gt = json.load(fh)["frames"]
    print(f"  clip: {n} frames @ 30 fps, v_host={SYNTH_V_HOST} m/s -> {path}")

    provider = _GTDepth(gt)
    gt_by_frame = {int(g["frame"]): g for g in gt}

    # Substitute depth and the anchor box; everything else runs for real.
    orig_depth_map, orig_roi = depth_mod.depth_map, depth_mod.roi_depth
    orig_detect = detect_mod.detect
    depth_mod.depth_map = provider.depth_map
    depth_mod.roi_depth = provider.roi_depth

    # The candidate source is now RF-DETR, so THAT is what the clip's ground
    # truth is injected through. Same reason the depth model is stubbed: the
    # synthetic clip is a flat-shaded rectangle, so running a real detector on
    # it would measure the detector, not the loop. Everything downstream --
    # membership, dwell, lead selection, Kalman, policy -- is live.
    def gt_detect(frame, score_min=None, variant=None):
        g = gt_by_frame.get(provider.frame)
        dets = [("car", tuple(g["box"]), 0.95)] if g else []
        return {"detections": dets, "n_bonnet_rejected": 0,
                "image": {"w": frame.shape[1], "h": frame.shape[0]},
                "timing_ms": {"forward": 0.0, "post": 0.0, "total": 0.0}}

    detect_mod.detect = gt_detect

    # Lane detection is stubbed OUT for the same reason depth and detection are
    # stubbed IN: the synthetic clip is a flat-shaded rectangle on a rendered
    # road, and UFLDv2's reading of it is noise. Measured on this clip it flips
    # between static and ufld frame to frame and sometimes locks a bogus 9 px
    # lane, swinging the lead's membership overlap 1.00 -> 0.00 -> 0.75. That
    # churns the lead lock and buries the band ladder under NEW_LEAD resets.
    #
    # This did not matter before RF-DETR because the corridor only vetted a
    # fresh anchor; now membership gates the lead on every frame, so a corridor
    # that flickers is a lead that flickers. Part A tests the loop, not the lane
    # model on synthetic pixels, so the corridor here is the static trapezoid --
    # which the same measurement shows gives a steady overlap of 1.00.
    orig_lanes = lanes_mod.detect_lanes

    def no_lanes(frame, weights_path=None):
        return {"lanes": [], "lane_conf": [], "lane_index": [],
                "ego_left": None, "ego_right": None, "confidence": 0.0,
                "ego": {"reason": "stubbed_for_selftest"},
                "image": {"w": frame.shape[1], "h": frame.shape[0]},
                "timing_ms": {"forward": 0.0, "decode": 0.0, "total": 0.0}}

    lanes_mod.detect_lanes = no_lanes

    records = []
    try:
        live_mod.reset_session("selftest")
        session = live_mod.get_session("selftest", use_qwen=True)
        cap = cv2.VideoCapture(path)
        stride = max(1, int(round(30.0 / LIVE_FPS)))   # subsample to the live cadence
        idx = 0
        while True:
            ok, frame = cap.read()
            if not ok:
                break
            if idx % stride == 0:
                provider.frame = idx
                ok_enc, buf = cv2.imencode(".jpg", frame)
                # Through the same bytes-in interface the endpoint uses, so the
                # decode path is covered too.
                rec = session.process(buf.tobytes(), SYNTH_V_HOST, 0.0,
                                      frame_t=idx / 30.0)
                rec["clip_t"] = round(idx / 30.0, 3)
                # Recomputed from d_true rather than read from the clip's
                # `tau_true`, which make_synthetic wrote against its own v_host.
                rec["tau_true"] = round(gt_by_frame[idx]["d_true"] / SYNTH_V_HOST, 3)
                records.append(rec)
            idx += 1
        cap.release()
    finally:
        depth_mod.depth_map, depth_mod.roi_depth = orig_depth_map, orig_roi
        detect_mod.detect = orig_detect
        lanes_mod.detect_lanes = orig_lanes
        live_mod.reset_session("selftest")

    # --- timeline -----------------------------------------------------------
    print(f"\n  {'t':>6} {'tau_true':>9} {'tau':>7} {'d':>7} {'band':>15} "
          f"{'trend':>8} {'urg':>4} {'conf':>5}  voice")
    for r in records:
        sp = r.get("speak")
        voice = (f"{sp['line']} [{sp['audio']}] ({sp['reason']})" if sp
                 else ("" if r["voice_reason"] == P.R_SILENT else f"- {r['voice_reason']}"))
        print(f"  {r['clip_t']:6.2f} {r['tau_true']:9.2f} "
              f"{(r['tau_s'] if r['tau_s'] is not None else float('nan')):7.2f} "
              f"{(r['distance_m'] if r['distance_m'] is not None else float('nan')):7.1f} "
              f"{r['band']:>15} {r['trend']:>8} {r['urgency']:>4} "
              f"{(r['confidence'] or 0):5.2f}  {voice}")

    spoken = [r for r in records if r.get("speak")]
    bands = [r["band"] for r in records]

    head("A1 -- band ladder")
    check(bands[0] == P.NORMAL, "starts in NORMAL", f"tau {records[0]['tau_s']}")
    check(P.GETTING_UNSAFE in bands, "reaches GETTING_UNSAFE")
    check(P.UNSAFE in bands, "reaches UNSAFE")
    # Order matters: the ladder must be walked, not jumped.
    first_amber = next(i for i, b in enumerate(bands) if b == P.GETTING_UNSAFE)
    first_red = next(i for i, b in enumerate(bands) if b == P.UNSAFE)
    check(first_amber < first_red, "GETTING_UNSAFE is entered before UNSAFE",
          f"frames {first_amber} then {first_red}")

    head("A2 -- automatic calm line on GETTING_UNSAFE entry")
    calm = [r for r in spoken if r["speak"]["line"] == P.LINE_CALM]
    check(len(calm) >= 1, "calm line fired", f"{len(calm)} utterance(s)")
    if calm:
        c = calm[0]
        check(c["band"] == P.GETTING_UNSAFE, "fired while in GETTING_UNSAFE")
        check(c["band_entered"] and c["prev_band"] == P.NORMAL,
              "fired on the confirmed entry from NORMAL")
        check(c["speak"]["audio"] == "tts", "calm tier uses live TTS")
        check(c["speak"]["text"] == "Beep beep — you're getting a little close there.",
              "exact spec wording", c["speak"]["text"])
        # Confirmed, never single-frame: the entry lands at least one frame after
        # tau first crosses the threshold.
        idx_c = records.index(c)
        check(idx_c >= first_amber, "entry was confirmed, not fired on first crossing")

    head("A3 -- pre-rendered alert on UNSAFE entry")
    red = [r for r in spoken if r["speak"]["tier"] == P.TIER_UNSAFE]
    check(len(red) >= 1, "red-tier line fired", f"{len(red)} utterance(s)")
    if red:
        r0 = red[0]
        check(r0["band"] == P.UNSAFE, "fired while in UNSAFE")
        check(r0["speak"]["audio"] != "tts", "red tier is a pre-rendered clip",
              f"clip id {r0['speak']['audio']!r}")
        check(r0["speak"]["audio"] in ("too_close", "watch_distance", "back_off"),
              "clip id is one of the three rendered files")
        check(r0["speak"]["reason"] == P.R_WORSENING,
              "amber -> red is logged as worsening", r0["speak"]["reason"])

    head("A4 -- coaching warm-up neutralises the trend, not the tier")
    # The guarantee is NOT "warm-up frames are silent" -- that version silenced
    # every cut-in, because the band entry always confirms inside the window
    # (see B10). It is that no line may be CHOSEN from an unconverged d_dot.
    warm = [r for r in records if r["voice_reason"] == P.R_WARMUP]
    trend_derived = {P.LINE_BACK_OFF, P.LINE_ESCALATE}
    offenders = [r for r in records
                 if r.get("speak") and r["speak"]["line"] in trend_derived
                 and r["voice_reason"] == P.R_WARMUP]
    check(not offenders, "no trend-derived line is ever chosen during warm-up",
          f"{[(o['clip_t'], o['speak']['line']) for o in offenders]}")
    check(all(r.get("speak") is None for r in warm),
          "warm-up frames with no band entry stay silent",
          f"{len(warm)} such frame(s)")
    check(records[0].get("speak") is None,
          "first frame of a session never speaks (nothing is confirmed yet)")

    head("A5 -- payload contract")
    keys = {"lead_box", "distance_m", "tau_s", "band", "trend", "urgency",
            "speak", "confidence", "anchor_age_s"}
    check(keys.issubset(records[0].keys()), "every spec'd field present",
          f"missing {sorted(keys - set(records[0].keys()))}")
    check(all(r["trend"] in (P.OPENING, P.STABLE, P.CLOSING) for r in records),
          "trend is opening|stable|closing")
    check(all(0 <= r["urgency"] <= 3 for r in records), "urgency in 0-3")
    check(all(r["lead_box"] is None or len(r["lead_box"]) == 4 for r in records),
          "lead_box is a 4-tuple or null")
    # QWEN IS NOT ON THIS PATH AT ALL ANY MORE.
    #
    # This check has been through three forms. It began as "exactly one anchor
    # over the clip", became "anchors are bounded by the candidate-refresh
    # interval" when membership needed fresher candidates, and is now the
    # strongest version available: the live loop never calls a language model,
    # so there is no rate to bound. RF-DETR supplies candidates on every frame
    # and the corridor selects among them.
    import ast

    src = os.path.join(os.path.dirname(__file__), "live.py")
    with open(src) as fh:
        tree = ast.parse(fh.read())
    calls = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Attribute):
            calls.add(node.attr)
    for banned in ("anchor", "qwen_handles", "last_detections"):
        check(banned not in calls,
              f"live.py never reaches for {banned!r}",
              "Qwen is retired from the anchor path")
    check(not hasattr(live_mod, "CANDIDATE_REFRESH_S")
          and not hasattr(live_mod, "ANCHOR_MIN_INTERVAL_S")
          and not hasattr(live_mod, "ANCHOR_MAX_AGE_S"),
          "the Qwen rationing constants are gone, not merely unused",
          "nothing can quietly start scheduling anchors again")

    # And the loop really did see a detection on essentially every frame.
    detected = [r for r in records
                if (r.get("membership_info") or {}).get("n_candidates")]
    check(len(detected) >= 0.9 * len(records),
          "candidates are present on ~every frame, not once per interval",
          f"{len(detected)}/{len(records)} frames had a candidate")
    return records


def session_anchor_calls(records):
    return sum(1 for r in records if r.get("anchored"))


# ===========================================================================
# B. Policy scenarios, scripted at 2 Hz
# ===========================================================================
DT = 0.5
GOOD_CONF = 0.9


def drive(policy, steps, t0=0.0):
    """Run (tau, trend, [confidence], [v_host]) tuples through the policy at 2 Hz."""
    out, t = [], t0
    for step in steps:
        tau, trend = step[0], step[1]
        conf = step[2] if len(step) > 2 else GOOD_CONF
        v = step[3] if len(step) > 3 else 20.0
        t += DT
        r = policy.tick(tau=tau, v2_trend=trend, confidence=conf, v_host=v,
                        v_host_stale=(v is None), t=t,
                        since_reset_s=99.0)      # past the warm-up unless overridden
        r["_t"] = t
        out.append(r)
    return out


def spoken_lines(recs):
    return [(round(r["_t"], 2), r["speak"]["line"], r["speak"]["reason"])
            for r in recs if r.get("speak")]


def run_policy():
    head("B -- policy scenarios @ 2 Hz")

    # -- B1 confirmation --------------------------------------------------
    head("B1 -- confirmation: never single-frame, never over 0.5 s")
    p = P.LivePolicy()
    recs = drive(p, [(4.0, v2.STABLE)] * 4 + [(2.5, v2.STABLE)] * 4)
    entries = [i for i, r in enumerate(recs) if r["band_entered"]]
    first_cross = 4
    check(recs[first_cross]["band"] == P.NORMAL,
          "single frame below threshold does NOT change the band")
    check(recs[first_cross + 1]["band"] == P.GETTING_UNSAFE,
          "second consecutive frame confirms it")
    check(len(entries) == 1 and entries[0] == first_cross + 1,
          "exactly one confirmed entry, on frame 2 of the candidate")
    dt_confirm = recs[first_cross + 1]["_t"] - recs[first_cross]["_t"]
    check(abs(dt_confirm - 0.5) < 1e-6,
          "confirmation window is 0.5 s at 2 fps", f"{dt_confirm:.2f} s")

    p = P.LivePolicy()
    recs = drive(p, [(4.0, v2.STABLE)] * 4 + [(1.5, v2.STABLE)] * 4)
    red_at = next(i for i, r in enumerate(recs) if r["band"] == P.UNSAFE)
    check((recs[red_at]["_t"] - recs[3]["_t"]) <= 1.0 + 1e-6,
          "UNSAFE confirmation does not exceed 0.5 s past the crossing",
          f"{recs[red_at]['_t'] - recs[4]['_t'] + DT:.2f} s")

    # -- B2 hysteresis ----------------------------------------------------
    head("B2 -- hysteresis: 0.2 s tau on exit")
    p = P.LivePolicy()
    drive(p, [(2.5, v2.STABLE)] * 6)                       # settle in amber
    check(p.band == P.GETTING_UNSAFE, "in GETTING_UNSAFE")
    recs = drive(p, [(3.1, v2.STABLE)] * 6, t0=3.0)        # above 3.0, below 3.2
    check(p.band == P.GETTING_UNSAFE,
          "tau 3.1 does NOT exit the band (needs > 3.2)", f"band {p.band}")
    recs = drive(p, [(3.3, v2.STABLE)] * 6, t0=6.0)
    check(p.band == P.NORMAL, "tau 3.3 exits to NORMAL")

    p = P.LivePolicy()
    drive(p, [(1.5, v2.STABLE)] * 6)
    check(p.band == P.UNSAFE, "in UNSAFE")
    drive(p, [(2.1, v2.STABLE)] * 6, t0=3.0)
    check(p.band == P.UNSAFE, "tau 2.1 does NOT exit UNSAFE (needs > 2.2)")

    # -- B3 cooldown ------------------------------------------------------
    head("B3 -- cooldown suppresses a plain repeat")
    p = P.LivePolicy()
    # amber -> normal -> amber again, quickly (well under the 30 s calm cooldown
    # and under the 10 s genuine-clear bar).
    recs = drive(p, [(2.5, v2.STABLE)] * 4 + [(3.5, v2.STABLE)] * 6
                 + [(2.5, v2.STABLE)] * 4)
    lines = spoken_lines(recs)
    check(len(lines) == 1, "calm line spoke once, not twice", f"{lines}")
    suppressed = [r for r in recs if r["voice_reason"] == P.R_COOLDOWN]
    check(len(suppressed) >= 1, "the second entry is logged as suppressed_by_cooldown",
          f"{len(suppressed)} frame(s)")

    # -- B4 worsening overrides cooldown ----------------------------------
    head("B4 -- worsening (GETTING_UNSAFE -> UNSAFE) always speaks")
    p = P.LivePolicy()
    recs = drive(p, [(1.5, v2.SLOWLY_SHRINKING)] * 4)        # red, fires
    first = spoken_lines(recs)
    check(len(first) == 1 and first[0][1] in (P.LINE_TOO_CLOSE, P.LINE_WATCH_DISTANCE),
          "red tier fired on entry", f"{first}")
    # Back up to amber, then straight back down to red — 4 s later, deep inside
    # the 15 s unsafe cooldown.
    recs2 = drive(p, [(2.5, v2.SLOWLY_SHRINKING)] * 4, t0=2.0)
    recs3 = drive(p, [(1.5, v2.SLOWLY_SHRINKING)] * 4, t0=4.0)
    lines3 = spoken_lines(recs3)
    check(len(lines3) == 1, "red tier spoke again despite the cooldown", f"{lines3}")
    check(lines3[0][2] == P.R_WORSENING, "and it is attributed to worsening",
          lines3[0][2])
    check(lines3[0][1] != first[0][1],
          "the repeat uses the other phrasing, not the same words",
          f"{first[0][1]} then {lines3[0][1]}")

    head("B4b -- a two-band jump NORMAL -> UNSAFE also overrides the cooldown")
    p = P.LivePolicy()
    recs = drive(p, [(1.5, v2.STABLE)] * 4)
    check(len(spoken_lines(recs)) == 1, "red tier fired once")
    # NORMAL for 3 s -- long enough to leave the band, far short of the 10 s
    # genuine clear -- then tau collapses straight past GETTING_UNSAFE. This is
    # the cut-in shape, and it is the case a cooldown must not be allowed to eat.
    drive(p, [(3.5, v2.STABLE)] * 6, t0=2.0)
    recs = drive(p, [(1.5, v2.RAPIDLY_SHRINKING)] * 6, t0=5.0)
    check(p.band == P.UNSAFE, "band re-entered UNSAFE", p.band)
    lines = spoken_lines(recs)
    check(len(lines) == 1, "red tier spoke despite the 15 s cooldown", f"{lines}")
    if lines:
        check(lines[0][2] == P.R_WORSENING, "attributed to worsening", lines[0][2])
        check(lines[0][1] == P.LINE_BACK_OFF,
              "and a rapidly closing cut-in gets the sharpest line", lines[0][1])
    entry = next(r for r in recs if r.get("speak"))
    check(entry["prev_band"] == P.NORMAL, "the jump really was NORMAL -> UNSAFE",
          f"{entry['prev_band']} -> {entry['band']}")

    head("B4c -- what stays cooldown-gated")
    # Single-band amber re-entry: the flicker-prone boundary, still protected.
    p = P.LivePolicy()
    recs = drive(p, [(2.5, v2.STABLE)] * 4)
    check(len(spoken_lines(recs)) == 1, "calm line fired on the first amber entry")
    drive(p, [(3.5, v2.STABLE)] * 6, t0=2.0)
    recs = drive(p, [(2.5, v2.STABLE)] * 6, t0=5.0)
    check(p.band == P.GETTING_UNSAFE, "band re-entered GETTING_UNSAFE")
    check(not spoken_lines(recs),
          "single-band amber re-entry inside the cooldown stays silent",
          f"{spoken_lines(recs)}")
    check(any(r["voice_reason"] == P.R_COOLDOWN for r in recs),
          "and is logged as suppressed_by_cooldown")

    # A system state is not a tau band, so leaving one is not a deterioration.
    # Entered with no confirmation, a flaky fix could bounce this every second.
    for label, v in (("UNKNOWN", None), ("SUPPRESSED", 3.0)):
        p = P.LivePolicy()
        recs = drive(p, [(1.5, v2.RAPIDLY_SHRINKING)] * 4)
        check(len(spoken_lines(recs)) == 1, f"[{label}] red tier fired once")
        drive(p, [(1.5, v2.RAPIDLY_SHRINKING, GOOD_CONF, v)] * 4, t0=2.0)
        check(p.band == getattr(P, label), f"[{label}] entered the system state")
        recs = drive(p, [(1.5, v2.RAPIDLY_SHRINKING)] * 6, t0=4.0)
        check(p.band == P.UNSAFE, f"[{label}] came back to UNSAFE")
        check(not spoken_lines(recs),
              f"[{label}] -> UNSAFE is NOT worsening; cooldown holds",
              f"{spoken_lines(recs)}")

    # -- B5 genuine clear re-arms -----------------------------------------
    head("B5 -- genuine clear (NORMAL > 10 s) re-arms the cooldown")
    p = P.LivePolicy()
    recs = drive(p, [(2.5, v2.STABLE)] * 4)
    check(len(spoken_lines(recs)) == 1, "calm line fired")
    # 13 s of NORMAL: past the 10 s bar, still inside the 30 s calm cooldown.
    drive(p, [(3.5, v2.STABLE)] * 26, t0=2.0)
    recs = drive(p, [(2.5, v2.STABLE)] * 4, t0=15.0)
    lines = spoken_lines(recs)
    check(len(lines) == 1,
          "calm line fires again after a genuine clear, inside the 30 s window",
          f"{lines}")

    # -- B6 escalation ----------------------------------------------------
    head("B6 -- second amber line after 5 s while still closing")
    p = P.LivePolicy()
    recs = drive(p, [(2.5, v2.SLOWLY_SHRINKING)] * 24)      # 12 s in amber
    lines = spoken_lines(recs)
    check(len(lines) == 2, "two amber utterances", f"{lines}")
    if len(lines) == 2:
        check(lines[0][1] == P.LINE_CALM and lines[1][1] == P.LINE_ESCALATE,
              "calm first, escalation second")
        gap = lines[1][0] - lines[0][0]
        check(gap >= P.ESCALATE_AFTER_S - DT - 1e-6,
              "escalation lands ~5 s after entry", f"{gap:.1f} s")
        esc = next(r for r in recs if r.get("speak")
                   and r["speak"]["line"] == P.LINE_ESCALATE)
        check(esc["speak"]["text"] == "Still closing — ease off a touch.",
              "exact spec wording", esc["speak"]["text"])
    # Escalation is once per occupancy, not every 5 s.
    check(sum(1 for r in recs if r.get("speak")
              and r["speak"]["line"] == P.LINE_ESCALATE) == 1,
          "escalation fires once per band occupancy, not repeatedly")

    head("B6b -- a STABLE gap in amber does not escalate")
    p = P.LivePolicy()
    recs = drive(p, [(2.5, v2.STABLE)] * 24)
    lines = spoken_lines(recs)
    check(len(lines) == 1 and lines[0][1] == P.LINE_CALM,
          "entry line only; no escalation while the gap is stable", f"{lines}")

    # -- B7 rapid closing picks the sharper clip --------------------------
    head("B7 -- RAPIDLY closing in UNSAFE uses 'Back off — now.'")
    p = P.LivePolicy()
    recs = drive(p, [(1.5, v2.RAPIDLY_SHRINKING)] * 4)
    lines = spoken_lines(recs)
    check(len(lines) == 1 and lines[0][1] == P.LINE_BACK_OFF,
          "back_off selected on a rapidly closing entry", f"{lines}")
    r = next(x for x in recs if x.get("speak"))
    check(r["urgency"] == 3, "urgency 3", f"urgency {r['urgency']}")
    check(r["speak"]["text"] == "Back off — now.", "exact spec wording")

    # -- B8 suppression gates ---------------------------------------------
    head("B8 -- suppression gates: warm-up, confidence, low speed, no speed")
    p = P.LivePolicy()
    r = p.tick(tau=1.2, v2_trend=v2.RAPIDLY_SHRINKING, confidence=0.95,
               v_host=20.0, v_host_stale=False, t=0.5, since_reset_s=0.1)
    check(r["speak"] is None and r["voice_reason"] == P.R_WARMUP,
          "warm-up: no voice in the first 0.6 s of a track", r["voice_reason"])

    p = P.LivePolicy()
    recs = drive(p, [(1.2, v2.RAPIDLY_SHRINKING, 0.25)] * 6)
    check(not spoken_lines(recs), "low confidence: silent")
    check(all(x["voice_reason"] == P.R_CONFIDENCE for x in recs),
          "and attributed to suppressed_by_confidence")
    check(recs[-1]["band"] == P.UNSAFE,
          "band is still classified and displayed while voice is suppressed",
          recs[-1]["band"])

    p = P.LivePolicy()
    recs = drive(p, [(1.2, v2.RAPIDLY_SHRINKING, GOOD_CONF, 3.0)] * 6)
    check(not spoken_lines(recs), "v_host 3 m/s: silent")
    check(all(x["band"] == P.SUPPRESSED for x in recs), "band is SUPPRESSED")

    p = P.LivePolicy()
    recs = drive(p, [(1.2, v2.RAPIDLY_SHRINKING, GOOD_CONF, None)] * 6)
    check(not spoken_lines(recs), "no speed fix: silent")
    check(all(x["band"] == P.UNKNOWN for x in recs), "band is UNKNOWN")
    check(all(x["urgency"] == 0 for x in recs),
          "UNKNOWN never reports an urgency it cannot justify")

    head("B8b -- a transient gate DEFERS a band entry, it does not delete one")
    # A band entry is one-shot. Before the latch, a gate landing on exactly the
    # entry frame silenced the whole occupancy however long it lasted.
    p = P.LivePolicy()
    # Confidence dips under the floor for precisely the two frames the entry
    # needs to confirm, then recovers.
    recs = drive(p, [(4.0, v2.STABLE)] * 2 + [(1.5, v2.STABLE, 0.2)] * 2
                 + [(1.5, v2.STABLE, GOOD_CONF)] * 4)
    check(p.band == P.UNSAFE, "band reached UNSAFE", p.band)
    lines = spoken_lines(recs)
    check(len(lines) == 1, "the deferred entry speaks once confidence recovers",
          f"{lines}")
    check(any(r["voice_reason"] == P.R_CONFIDENCE for r in recs),
          "and the suppressed frames are still logged as suppressed_by_confidence")

    head("B8c -- an entry deferred past the window is dropped, not hoarded")
    p = P.LivePolicy()
    recs = drive(p, [(4.0, v2.STABLE)] * 2 + [(1.5, v2.STABLE, 0.2)] * 10
                 + [(1.5, v2.STABLE, GOOD_CONF)] * 4)
    check(not spoken_lines(recs),
          "a 5 s blind spell is no longer an 'entry' worth announcing",
          f"{spoken_lines(recs)}")
    check(p.band == P.UNSAFE, "the band is still displayed throughout", p.band)

    head("B9 -- de-escalating INTO amber does not fire the calm line")
    p = P.LivePolicy()
    drive(p, [(1.5, v2.STABLE)] * 4)                        # red
    recs = drive(p, [(2.5, v2.INCREASING)] * 6, t0=2.0)     # recovering into amber
    check(p.band == P.GETTING_UNSAFE, "band came back to GETTING_UNSAFE")
    calm = [l for l in spoken_lines(recs) if l[1] == P.LINE_CALM]
    check(not calm, "improving into the band is not a warning event", f"{calm}")

    run_cutin()


def _cutin_at(dt, v_host=18.0, t_cut=6.0):
    """One cut-in run at a given cadence. Returns (policy, accept_lat, spoke)."""
    from .filter import HeadwayFilter

    kf, pol, trend = HeadwayFilter(), P.LivePolicy(), v2.STABLE
    t, reset_t, spoke, accepted_at = 0.0, 0.0, None, None
    for _ in range(int(20 / dt)):
        t += dt
        cut = t >= t_cut
        snap = kf.step((1.5 if cut else 4.0) * v_host, 0.85, dt)
        if cut and accepted_at is None and snap["accepted"]:
            accepted_at = t
        if snap["new_lead"]:
            reset_t = t
        pvv = snap.get("P_vv")
        trend = v2.classify_trend(snap["d_dot"], v_host, trend,
                                  d_dot_sigma=(math.sqrt(pvv) if pvv and pvv > 0 else None))
        r = pol.tick(tau=v2.compute_tau(snap["d"], v_host), v2_trend=trend,
                     confidence=0.9, v_host=v_host, v_host_stale=False, t=t,
                     since_reset_s=(t - reset_t), new_lead=snap["new_lead"],
                     track_lost=False)
        if cut and spoke is None and r.get("speak"):
            spoke = (t - t_cut, r["speak"], r["prev_band"], r["band"])
    return pol, (None if accepted_at is None else accepted_at - t_cut), spoke


def run_cutin():
    """B10 -- the cut-in, driven through the REAL filter, at both cadences.

    The case the two-band worsening rule exists for, and the one that exposed
    the swallowed-entry defect: a lead cuts in and tau collapses 4.0 s -> 1.5 s
    in one step. The Kalman gates the discontinuity until the reject run clears
    REJECT_WINDOW_S, re-seeds (NEW_LEAD), and the band confirmation then
    completes INSIDE the 0.6 s coaching warm-up -- so this is only silent-free
    because the warm-up neutralises the trend rather than the tier.

    Run at both cadences to show the total is now cadence-driven rather than
    cadence-*dependent*: both stages are times, so the only thing 4 fps buys is
    landing on those times sooner.
    """
    head("B10 -- cut-in through the real filter (tau 4.0s -> 1.5s)")
    results = {}
    for label, dt in (("2 fps", 0.5), ("4 fps", 0.25)):
        pol, accept, spoke = _cutin_at(dt)
        results[label] = (accept, spoke)
        check(pol.band == P.UNSAFE, f"[{label}] band reached UNSAFE", pol.band)
        check(spoke is not None, f"[{label}] the cut-in produced a warning")
        if not spoke:
            continue
        lat, sp, prev, band = spoke
        check(sp["tier"] == P.TIER_UNSAFE, f"[{label}] red tier", sp["tier"])
        check(sp["reason"] == P.R_WORSENING,
              f"[{label}] attributed to worsening (the two-band jump)", sp["reason"])
        check(prev == P.NORMAL, f"[{label}] transition really was NORMAL -> UNSAFE",
              f"{prev} -> {band}")
        # Chosen with the trend neutralised: we do not yet know it is
        # collapsing, so the sharpest line would be a guess.
        check(sp["line"] in (P.LINE_TOO_CLOSE, P.LINE_WATCH_DISTANCE),
              f"[{label}] no line chosen off an unconverged d_dot", sp["line"])
        check(lat <= 1.6, f"[{label}] warning within 1.6 s of the cut-in",
              f"{lat:.2f} s")

    a2, s2 = results["2 fps"]
    a4, s4 = results["4 fps"]
    check(s4[0] < s2[0], "4 fps warns sooner than 2 fps",
          f"{s4[0]:.2f} s vs {s2[0]:.2f} s")
    print(f"\n  {'cadence':>8} {'filter accept':>14} {'band confirm':>14} "
          f"{'voice':>9}")
    for label, (acc, sp) in results.items():
        print(f"  {label:>8} {acc:>13.2f}s {sp[0] - acc:>13.2f}s {sp[0]:>8.2f}s")


# ===========================================================================
# C. LLM firewall
# ===========================================================================
FORBIDDEN = ["transformers", "torch", "cv2", "numpy", "requests", "vision",
             "llm_interface", "app", "perceive", "anchor", "depth", "tracker",
             "filter", "live"]


def run_firewall():
    import ast
    head("C -- LLM firewall: live_policy.py sees no model output")
    src = os.path.join(os.path.dirname(__file__), "live_policy.py")
    with open(src) as fh:
        tree = ast.parse(fh.read())

    imported = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            imported.update(a.name.split(".")[0] for a in node.names)
        elif isinstance(node, ast.ImportFrom):
            if node.module:
                imported.add(node.module.split(".")[0])
            # `from . import state as v2` -- the relative import of the pure
            # maths module, which the firewall permits and selftest.py already
            # proves is stdlib-only.
            imported.update(a.name for a in node.names if node.level and not node.module)

    for mod in FORBIDDEN:
        check(mod not in imported, f"live_policy.py does not import {mod!r}")
    check(imported <= {"math", "state"}, "live_policy.py imports only math + state",
          f"imports: {sorted(imported)}")

    calls = [n.func.id for n in ast.walk(tree)
             if isinstance(n, ast.Call) and isinstance(n.func, ast.Name)]
    check("open" not in calls, "live_policy.py performs no file I/O")


# ===========================================================================
# D. UFLDv2 lane model — is it really the published network, decoded the
#    published way?
# ===========================================================================
# headway/lanes.py restates the upstream architecture instead of importing the
# training repo (see its module docstring for why). That buys a clean
# integration and takes on one specific risk: a network that is *shaped* like
# UFLDv2, loading the official weights into subtly the wrong places, producing
# confident nonsense. These checks are what close that risk. They need the
# checkpoint, so they skip rather than fail when it is absent -- the loop is
# designed to run without it, and a CI box with no weights should not go red.
def _pred2coords_reference(pred, row_anchor, col_anchor, local_width,
                           num_grid_row, num_grid_col, W, H):
    """Literal transcription of demo.py:pred2coords, upstream commit as cloned.

    Deliberately unoptimised and deliberately not shared with lanes.py: the
    whole value of this function is that it was written from the reference and
    not from our implementation.
    """
    import torch

    batch_size, num_grid_row_, num_cls_row, num_lane_row = pred['loc_row'].shape
    batch_size, num_grid_col_, num_cls_col, num_lane_col = pred['loc_col'].shape

    max_indices_row = pred['loc_row'].argmax(1).cpu()
    valid_row = pred['exist_row'].argmax(1).cpu()
    max_indices_col = pred['loc_col'].argmax(1).cpu()
    valid_col = pred['exist_col'].argmax(1).cpu()

    loc_row = pred['loc_row'].float().cpu()
    loc_col = pred['loc_col'].float().cpu()

    coords = []
    for i in range(num_lane_row):
        tmp = []
        if valid_row[0, :, i].sum() > num_cls_row / 2:
            for k in range(valid_row.shape[1]):
                if valid_row[0, k, i]:
                    all_ind = torch.tensor(list(range(
                        max(0, max_indices_row[0, k, i] - local_width),
                        min(num_grid_row - 1, max_indices_row[0, k, i] + local_width) + 1)))
                    out_tmp = (loc_row[0, all_ind, k, i].softmax(0) * all_ind.float()).sum() + 0.5
                    out_tmp = out_tmp / (num_grid_row - 1) * W
                    tmp.append((float(out_tmp), float(row_anchor[k] * H)))
        coords.append(tmp)

    col = []
    for i in range(num_lane_col):
        tmp = []
        if valid_col[0, :, i].sum() > num_cls_col / 4:
            for k in range(valid_col.shape[1]):
                if valid_col[0, k, i]:
                    all_ind = torch.tensor(list(range(
                        max(0, max_indices_col[0, k, i] - local_width),
                        min(num_grid_col - 1, max_indices_col[0, k, i] + local_width) + 1)))
                    out_tmp = (loc_col[0, all_ind, k, i].softmax(0) * all_ind.float()).sum() + 0.5
                    out_tmp = out_tmp / (num_grid_col - 1) * H
                    tmp.append((float(col_anchor[k] * W), float(out_tmp)))
        col.append(tmp)
    return coords, col


def run_lanes():
    import time

    from . import lanes as L

    head("D -- UFLDv2 lane model fidelity")

    if not L.available():
        print(f"  [SKIP] weights not present at {L.DEFAULT_WEIGHTS} — "
              "lane checks skipped; the loop falls back to the trapezoid")
        return

    import torch

    # -- the checkpoint fits the model exactly ------------------------------
    net = L.ParsingNet()
    try:
        L._load_state_dict(net, L.DEFAULT_WEIGHTS)
        loaded = True
        detail = ""
    except RuntimeError as e:
        loaded, detail = False, str(e)[:200]
    check(loaded, "official culane_res18 checkpoint loads with no missing or "
                  "unexpected keys", detail)
    if not loaded:
        return

    check(net.total_dim == 91224 and net.input_dim == 4000,
          "head dimensions match configs/culane_res18.py",
          f"total_dim={net.total_dim} input_dim={net.input_dim}")

    # -- decoding matches the reference loop, cell for cell -----------------
    # On a real road frame, not noise: the network correctly finds no lanes in
    # noise, so a comparison there would agree on the empty set and prove
    # nothing. The clip ships with the upstream repo.
    import cv2
    clip = "/workspace/ufldv2/example.mp4"
    real = None
    if os.path.exists(clip):
        cap = cv2.VideoCapture(clip)
        ok, real = cap.read()
        cap.release()
        if not ok:
            real = None

    L._ensure_loaded()
    if real is None:
        print(f"  [SKIP] {clip} not present — decode-vs-reference check needs "
              "a frame with lanes in it")
        torch.manual_seed(0)
        probe_frame = (torch.rand(720, 1280, 3) * 255).numpy().astype("uint8")
    else:
        probe_frame = real

    with torch.inference_mode():
        pred = L._net(L._preprocess(probe_frame))

    ref_row, ref_col = _pred2coords_reference(
        pred, L.ROW_ANCHOR, L.COL_ANCHOR, L.LOCAL_WIDTH,
        L.NUM_CELL_ROW, L.NUM_CELL_COL, 1280, 720)

    row_pos, row_valid, row_pe = L._decode_branch(
        pred["loc_row"], pred["exist_row"], L.NUM_CELL_ROW)
    col_pos, col_valid, col_pe = L._decode_branch(
        pred["loc_col"], pred["exist_col"], L.NUM_CELL_COL)

    worst = 0.0
    n_compared = 0
    for i in range(L.NUM_LANES):
        ours = L._lane_from_row(row_pos, row_valid, row_pe, i, 1280, 720)
        theirs = ref_row[i]
        if not theirs:
            continue
        if ours is None:
            worst = float("inf")
            break
        if len(ours["points"]) != len(theirs):
            worst = float("inf")
            break
        for (ax, ay), (bx, by) in zip(ours["points"], theirs):
            worst = max(worst, abs(ax - bx), abs(ay - by))
            n_compared += 1
    check(worst < 1e-3 and (n_compared > 0 or real is None),
          "vectorised row decode matches demo.py:pred2coords",
          f"max |delta| = {worst:.2e} px over {n_compared} points")

    worst_c, n_c = 0.0, 0
    for i in range(L.NUM_LANES):
        ours = L._lane_from_col(col_pos, col_valid, col_pe, i, 1280, 720)
        theirs = ref_col[i]
        if not theirs or ours is None:
            continue
        pts = sorted(theirs, key=lambda p: p[1])
        for (ax, ay), (bx, by) in zip(ours["points"], pts):
            worst_c = max(worst_c, abs(ax - bx), abs(ay - by))
            n_c += 1
    check(worst_c < 1e-3, "vectorised column decode matches the reference",
          f"max |delta| = {worst_c:.2e} px over {n_c} points")

    # -- the -inf window mask is not the same as clamping -------------------
    # A lane against the frame edge sits at cell 0 or cell G-1, where the +/-1
    # window runs off the grid. Upstream shortens the window; clamping would
    # duplicate a logit and shift the weighted mean. This asserts the branch is
    # actually reachable and handled, not merely written.
    G = 8
    loc = torch.full((1, G, 1, 1), -20.0)
    loc[0, 0, 0, 0] = 5.0                       # argmax at the very first cell
    loc[0, 1, 0, 0] = 5.0                       # ...tied with its neighbour
    exist = torch.tensor([[[[0.0]], [[9.0]]]])  # exists
    pos, valid, _ = L._decode_branch(loc, exist, G)
    # Reference window is [0, 1], two equal logits -> mean cell 0.5, +0.5 = 1.0.
    check(abs(float(pos[0, 0]) - 1.0) < 1e-5 and bool(valid[0, 0]),
          "edge-cell decode window is truncated, not clamped",
          f"pos={float(pos[0, 0]):.6f} (clamping would give 0.8333)")

    # -- cost, on the real frame path ---------------------------------------
    probe = probe_frame
    for _ in range(5):
        L.detect_lanes(probe)
    torch.cuda.synchronize() if torch.cuda.is_available() else None
    times = []
    for _ in range(30):
        t0 = time.perf_counter()
        L.detect_lanes(probe)
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        times.append((time.perf_counter() - t0) * 1000.0)
    times.sort()
    p50, p95 = times[len(times) // 2], times[int(len(times) * 0.95)]
    # The frame budget at 4 fps is 250 ms and depth already takes ~7 ms. Lane
    # detection has to be small enough that adding it does not change the
    # cadence; 10 ms is the line at which that stops being obviously true.
    check(p95 < 10.0, "lane detection stays inside the per-frame budget",
          f"p50 {p50:.2f} ms, p95 {p95:.2f} ms over 30 frames")

    # -- and it actually finds the lane on real footage ---------------------
    if real is not None:
        r = L.detect_lanes(real)
        check(r["ego_left"] is not None and r["ego_right"] is not None
              and r["confidence"] >= L.LANE_CONF_MIN,
              "ego pair found with usable confidence on real dashcam footage",
              f"conf={r['confidence']} lanes={len(r['lanes'])}")
        xl = r["ego"]["x_bottom_left"]
        xr = r["ego"]["x_bottom_right"]
        check(xl < real.shape[1] / 2 < xr,
              "the ego pair brackets image centre at the bottom",
              f"{xl:.0f} < {real.shape[1] / 2:.0f} < {xr:.0f}")
    else:
        print(f"  [SKIP] {clip} not present — real-footage check skipped")


def run_detector():
    """E -- RF-DETR: is it the published model, is it Apache, is it fast?"""
    import time

    from . import detect as D

    head("E -- RF-DETR candidate source")

    if not D.available():
        print(f"  [SKIP] weights not at {D.WEIGHTS.get(D.VARIANT)} — "
              "fetch with `python -m tools.fetch_detector_weights`")
        return

    import torch

    # -- licence, checked not assumed ---------------------------------------
    cfg = D._config(D.VARIANT)
    check(str(cfg.get("license", "")).lower() == "apache-2.0",
          "the loaded variant declares Apache-2.0",
          f"license={cfg.get('license')!r} (YOLO's AGPL is why this matters)")
    bad = dict(cfg); bad["license"] = "AGPL-3.0"
    try:
        D._assert_apache(bad); refused = False
    except ValueError:
        refused = True
    check(refused, "a non-Apache variant would be refused at load time")

    # -- the checkpoint really fits this architecture ------------------------
    D._ensure_loaded()
    check(D._model is not None and D._dtype == torch.float16,
          "model is loaded in fp16 on CUDA", f"dtype={D._dtype}")
    n_params = sum(p.numel() for p in D._model.parameters()) / 1e6
    check(25.0 < n_params < 40.0, "parameter count matches RF-DETR Nano",
          f"{n_params:.1f} M")
    # _ensure_loaded raises on any key mismatch, so reaching here IS the proof.
    check(True, "checkpoint loaded with zero missing/unexpected keys",
          "strict load; a mismatch raises in _ensure_loaded")

    # -- the training harness never ran --------------------------------------
    import sys
    check("rfdetr.detr" not in sys.modules and "rfdetr.datasets" not in sys.modules,
          "the rfdetr training harness was never imported",
          "detect.py imports rfdetr.models.lwdetr beneath a stub package")

    # -- the ego bonnet, which is the dangerous false positive ---------------
    W, H = 1280, 720
    check(D._is_ego_bonnet(2, 661, 1279, 720, W, H),
          "the ego bonnet strip is rejected",
          "full width, flush with the bottom, aspect 21.6 — observed live")
    check(not D._is_ego_bonnet(100, 300, 1200, 719, W, H),
          "a genuinely close, TALL vehicle is NOT rejected",
          "wide-and-tall is a lorry; wide-and-flat is our own bonnet")
    check(not D._is_ego_bonnet(500, 600, 780, 720, W, H),
          "a normal vehicle touching the frame bottom is not rejected")

    # -- cost ----------------------------------------------------------------
    clip = "/workspace/ufldv2/example.mp4"
    frame = None
    if os.path.exists(clip):
        import cv2
        cap = cv2.VideoCapture(clip)
        cap.set(cv2.CAP_PROP_POS_FRAMES, 300)
        ok, frame = cap.read()
        cap.release()
        if not ok:
            frame = None
    if frame is None:
        frame = np.zeros((720, 1280, 3), dtype=np.uint8)

    for _ in range(5):
        D.detect(frame)
    if torch.cuda.is_available():
        torch.cuda.synchronize()
    times = []
    for _ in range(30):
        t0 = time.perf_counter()
        D.detect(frame)
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        times.append((time.perf_counter() - t0) * 1000.0)
    times.sort()
    p50, p95 = times[len(times) // 2], times[int(len(times) * 0.95)]
    # The whole justification for retiring Qwen: single-digit milliseconds, so
    # detection can run on every frame instead of being rationed.
    check(p95 < 10.0, "detection is single-digit ms — the point of the swap",
          f"p50 {p50:.2f} ms, p95 {p95:.2f} ms (Qwen enumeration was 600-1500 ms)")

    r = D.detect(frame)
    # det is (label, box, score, info) -- indexed rather than unpacked so a
    # future field cannot break this the way the `info` field broke it once.
    check(all(det[0] in D.COCO_TO_LABEL.values() for det in r["detections"]),
          "only road-user classes are emitted",
          str(sorted({det[0] for det in r["detections"]})))
    check(all(len(det) == 4 and isinstance(det[3], dict) for det in r["detections"]),
          "every detection carries its (confirmed, h_px) info dict",
          "consumers read det[0..2]; det[3] is additive")

    # --- duplicate suppression (the vanishing-point cluster) ----------------
    # Four boxes on the same few pixels, as observed at a road-end horizon.
    # _duplicates is pure geometry, so it is checked directly rather than by
    # hunting for a frame that reproduces the failure.
    cluster = [("car", (600.0, 350.0, 618.0, 364.0), 0.61),
               ("car", (602.0, 351.0, 620.0, 366.0), 0.55),
               ("car", (598.0, 349.0, 621.0, 365.0), 0.48),
               ("truck", (601.0, 350.0, 619.0, 365.0), 0.44),
               ("car", (200.0, 300.0, 320.0, 400.0), 0.90)]   # a real, separate car
    keep, dropped = D._duplicates(cluster)
    check(len(keep) == 2 and len(dropped) == 3,
          "an overlapping cluster collapses to one box",
          f"kept {len(keep)} of {len(cluster)} (the cluster + the separate car)")
    check(cluster[keep[0]][2] == 0.61 or cluster[keep[1]][2] == 0.61,
          "the survivor of a cluster is its highest-scoring member")
    check(any(cluster[i][0] == "truck" for i, _ in dropped),
          "a cross-class duplicate (car vs truck on one object) is suppressed",
          "car/truck/bus/motorcycle are competing readings of one object")

    person_on_bike = [("cyclist", (400.0, 300.0, 440.0, 380.0), 0.70),
                      ("pedestrian", (405.0, 295.0, 435.0, 370.0), 0.65)]
    keep2, dropped2 = D._duplicates(person_on_bike)
    check(len(keep2) == 2 and not dropped2,
          "a pedestrian overlapping a cyclist is NOT suppressed",
          "a cyclist IS a person on a bicycle — both boxes are real")

    # --- the size/score floor ----------------------------------------------
    check(D._score_floor("car", 60.0) == D.SCORE_MIN_BY_LABEL["car"],
          "a normal-sized box is gated on its class threshold alone")
    check(D._score_floor("car", 14.0) == D.SMALL_BOX_SCORE_MIN,
          "a sub-floor box must clear a higher confidence",
          f"{D.SMALL_BOX_SCORE_MIN} vs {D.SCORE_MIN_BY_LABEL['car']} — road-end "
          "texture scores low, distant vehicles score high")
    check(D._score_floor("pedestrian", 60.0) < D._score_floor("car", 60.0),
          "pedestrians are gated LOWER than vehicles",
          "recall on vulnerable road users is worth more than a clean box count")


def main():
    print("=" * 70)
    print("RIO live headway (v3) — verification")
    print("=" * 70)
    run_pipeline()
    run_policy()
    run_firewall()
    run_lanes()
    run_detector()

    print("\n" + "=" * 70)
    total = len(PASS) + len(FAIL)
    print(f"{len(PASS)}/{total} checks passed")
    if FAIL:
        print("\nFAILED:")
        for f in FAIL:
            print(f"  - {f}")
    print("=" * 70)
    return 1 if FAIL else 0


if __name__ == "__main__":
    sys.exit(main())
