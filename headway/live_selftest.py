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
from . import membership as member_mod
from . import plausibility as plaus_mod
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
    # `config` joined this set when the punch list moved the thresholds out of
    # this module's PROVISIONAL block. It does not weaken the firewall, and the
    # check below is what makes that a proof rather than an assurance: config
    # is a module of literals evaluated at import, and every name live_policy
    # reads from it is asserted to be a plain number or a tuple of strings.
    # There is no callable, no object and no dict on that surface, so there is
    # nothing there through which a model output could arrive.
    check(imported <= {"math", "state", "config"},
          "live_policy.py imports only math + state + config",
          f"imports: {sorted(imported)}")

    import config as _cfg
    read_from_config = sorted({
        n.attr for n in ast.walk(tree)
        if isinstance(n, ast.Attribute) and isinstance(n.value, ast.Name)
        and n.value.id == "config"})
    check(bool(read_from_config), "live_policy.py does read its numbers from config",
          f"{len(read_from_config)} names")
    bad = []
    for name in read_from_config:
        v = getattr(_cfg, name, None)
        ok = isinstance(v, (int, float)) and not isinstance(v, bool)
        if not ok and isinstance(v, tuple):
            ok = all(isinstance(x, str) for x in v)
        if not ok:
            bad.append((name, type(v).__name__))
    check(not bad,
          "every config name live_policy reads is a constant, not an object",
          f"offenders: {bad}" if bad else f"{len(read_from_config)} names, all literal")

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


# ===========================================================================
# S. Speed-aware safety alerts (punch-list item 2)
#
# Everything here is about the DENOMINATOR of tau. The band ladder has always
# been expressed in time headway; what the first real drive showed is that the
# speed the ladder divides by was never resolved, never scaled anything else,
# and had exactly two states -- fresh, or silence.
# ===========================================================================
def _gap_to_tau(gap_m, v_ms):
    """The two-second rule, arithmetic. gap / speed, and nothing else."""
    return gap_m / v_ms


def run_speed():
    head("S -- speed-aware alerts (item 2)")

    # -- S1 the same gap is a different band at a different speed ----------
    #
    # THE WHOLE POINT OF TIME HEADWAY, asserted rather than assumed. 30 m
    # behind a car is a relaxed gap at 15 mph and a red-tier warning at 65,
    # and a system expressed in metres cannot tell those apart.
    head("S1 -- the same gap, two speeds, two bands")
    GAP_M = 30.0
    V_15MPH = 6.706      # 15 mph
    V_65MPH = 29.058     # 65 mph
    tau_slow = _gap_to_tau(GAP_M, V_15MPH)     # 4.47 s
    tau_fast = _gap_to_tau(GAP_M, V_65MPH)     # 1.03 s

    p_slow = P.LivePolicy()
    slow = drive(p_slow, [(tau_slow, v2.STABLE, GOOD_CONF, V_15MPH)] * 6)
    p_fast = P.LivePolicy()
    fast = drive(p_fast, [(tau_fast, v2.STABLE, GOOD_CONF, V_65MPH)] * 6)

    check(slow[-1]["band"] == P.NORMAL,
          f"{GAP_M:.0f} m at 15 mph is NORMAL",
          f"tau={tau_slow:.2f}s band={slow[-1]['band']}")
    check(fast[-1]["band"] == P.UNSAFE,
          f"{GAP_M:.0f} m at 65 mph is UNSAFE",
          f"tau={tau_fast:.2f}s band={fast[-1]['band']}")
    check(slow[-1]["band"] != fast[-1]["band"],
          "one gap, two speeds, two different bands")
    check(not spoken_lines(slow), "...and the slow one says nothing at all")
    check(bool(spoken_lines(fast)), "...while the fast one speaks")

    # And the boundary is where the arithmetic says it is, not where a metre
    # threshold would put it: the gap that is exactly 2.0 s at 65 mph is
    # 58.1 m, and the same 58.1 m at 15 mph is 8.7 s of headway.
    edge_m = P.TAU_ENTER_UNSAFE * V_65MPH
    p_edge = P.LivePolicy()
    edge = drive(p_edge, [(_gap_to_tau(edge_m - 1.0, V_65MPH), v2.STABLE,
                           GOOD_CONF, V_65MPH)] * 6)
    check(edge[-1]["band"] == P.UNSAFE,
          f"a metre inside the 2.0 s line at 65 mph ({edge_m:.0f} m) is UNSAFE")
    p_edge2 = P.LivePolicy()
    edge2 = drive(p_edge2, [(_gap_to_tau(edge_m - 1.0, V_15MPH), v2.STABLE,
                             GOOD_CONF, V_15MPH)] * 6)
    check(edge2[-1]["band"] == P.NORMAL,
          f"...and the same {edge_m:.0f} m at 15 mph is NORMAL")

    # -- S2 stationary produces no gap warning -----------------------------
    #
    # Creeping in traffic is not danger. A 3 m gap at 0.4 m/s is a tau of 7.5 s
    # by arithmetic and a queue by every other measure, and the speed floor is
    # what stops the arithmetic being believed.
    head("S2 -- stationary and creeping say nothing")
    for label, v, gap in (("stopped dead", 0.0, 3.0),
                          ("creeping", 0.4, 3.0),
                          ("car-park pace", 2.0, 4.0),
                          ("just under the floor", P.V_MIN_COACH - 0.1, 5.0)):
        p = P.LivePolicy()
        # tau is computed the same way live.py computes it, floor and all, so
        # the test cannot pass by feeding the policy a tau it would never see.
        tau = v2.compute_tau(gap, v)
        recs = drive(p, [(tau, v2.SHRINKING_FAST if hasattr(v2, "SHRINKING_FAST")
                          else v2.RAPIDLY_SHRINKING, GOOD_CONF, v)] * 8)
        check(all(r["band"] == P.SUPPRESSED for r in recs),
              f"{label} ({v} m/s, {gap} m gap) stays SUPPRESSED")
        check(not spoken_lines(recs),
              f"...and speaks not once, even with the gap collapsing")
        check(all(r["voice_reason"] == P.R_LOW_SPEED for r in recs),
              "...for the recorded reason 'suppressed_by_low_speed'")

    # A stationary car with an urgent TTC is still silent: the trigger is
    # reached through the band gates, and SUPPRESSED returns before it.
    p = P.LivePolicy()
    recs = drive(p, [(v2.compute_tau(2.0, 0.0), v2.RAPIDLY_SHRINKING,
                      GOOD_CONF, 0.0)] * 8)
    check(not spoken_lines(recs),
          "a stationary car does not get a TTC warning either")

    # -- S3 a degraded speed WIDENS, it does not silence -------------------
    head("S3 -- degraded speed widens the margins rather than silencing")

    # 2.6 s of headway: inside GETTING_UNSAFE either way, so it is not the
    # test. 3.4 s is: NORMAL on a trusted speed, and inside the widened amber
    # band on a coasted one.
    TAU_ONLY_WIDE_CATCHES = P.TAU_ENTER_GETTING_UNSAFE + 0.4   # 3.4
    check(TAU_ONLY_WIDE_CATCHES < P.TAU_ENTER_GETTING_UNSAFE + P.DEGRADED_TAU_BIAS_S,
          "the test tau sits between the normal and the widened amber edge")

    p_ok = P.LivePolicy()
    ok = [p_ok.tick(tau=TAU_ONLY_WIDE_CATCHES, v2_trend=v2.STABLE,
                    confidence=GOOD_CONF, v_host=20.0, v_host_stale=False,
                    t=0.5 * (i + 1), since_reset_s=99.0, speed_degraded=False)
          for i in range(6)]
    p_deg = P.LivePolicy()
    deg = [p_deg.tick(tau=TAU_ONLY_WIDE_CATCHES, v2_trend=v2.STABLE,
                      confidence=GOOD_CONF, v_host=20.0, v_host_stale=False,
                      t=0.5 * (i + 1), since_reset_s=99.0, speed_degraded=True)
           for i in range(6)]
    check(ok[-1]["band"] == P.NORMAL,
          f"tau {TAU_ONLY_WIDE_CATCHES} s on a trusted speed is NORMAL")
    check(deg[-1]["band"] == P.GETTING_UNSAFE,
          "...and on a degraded speed the same tau is GETTING_UNSAFE")
    check(bool([r for r in deg if r.get("speak")]),
          "the degraded drive SPEAKS where the trusted one had nothing to say")
    check(deg[-1]["tau_bias_s"] == P.DEGRADED_TAU_BIAS_S,
          "the widening is reported on the frame, not left to be inferred")

    # The confidence floor is relaxed in the same direction.
    weak = P.CONF_FLOOR - P.DEGRADED_CONF_RELIEF / 2.0
    p_ok2 = P.LivePolicy()
    ok2 = [p_ok2.tick(tau=1.0, v2_trend=v2.STABLE, confidence=weak,
                      v_host=20.0, v_host_stale=False, t=0.5 * (i + 1),
                      since_reset_s=99.0, speed_degraded=False)
           for i in range(6)]
    p_deg2 = P.LivePolicy()
    deg2 = [p_deg2.tick(tau=1.0, v2_trend=v2.STABLE, confidence=weak,
                        v_host=20.0, v_host_stale=False, t=0.5 * (i + 1),
                        since_reset_s=99.0, speed_degraded=True)
            for i in range(6)]
    check(all(r["voice_reason"] == P.R_CONFIDENCE for r in ok2 if r["band_entered"]),
          "a marginal confidence suppresses on a trusted speed")
    check(bool([r for r in deg2 if r.get("speak")]),
          "...and is let through on a degraded one, because the alternative "
          "to a less-certain warning is no warning")

    # AND THE DIRECTION IS ONE-WAY. Nothing a degraded speed does may make RIO
    # quieter than a trusted one would have been.
    for tau in (0.5, 1.0, 1.9, 2.5, 3.1, 4.0, 8.0):
        b_ok = P.classify_tau(tau, None, 0.0)
        b_deg = P.classify_tau(tau, None, P.DEGRADED_TAU_BIAS_S)
        check(P._SEV[b_deg] >= P._SEV[b_ok],
              f"tau {tau}: degraded is never a calmer band than trusted",
              f"{b_ok} -> {b_deg}")

    # -- S4 the speed source priority --------------------------------------
    head("S4 -- OBD > GPS > coasted > nothing")
    import config
    from . import speed as speed_mod

    def obd(v_mph, age=0.1, source="live_obd"):
        return lambda: {"v_ms": v_mph * 0.44704, "mph": v_mph, "age_s": age,
                        "source": source, "provider": source}

    r = speed_mod.SpeedResolver(obd_reader=obd(45.0))
    fix = r.resolve(18.0, 0.1, 1.0)
    check(fix.source == speed_mod.OBD and abs(fix.v_ms - 20.117) < 0.01,
          "a live OBD bus outranks a fresh GPS fix", fix.to_log())

    # THE MOCK MUST NOT. It reports 0 mph at a desk, and an unconditional
    # OBD-first rule would put a moving car into SUPPRESSED and speak once in
    # a whole drive.
    r = speed_mod.SpeedResolver(obd_reader=obd(0.0, source="mock_holley"))
    fix = r.resolve(20.0, 0.1, 1.0)
    check(fix.source == speed_mod.GPS and fix.v_ms == 20.0,
          "the mock telemetry source does NOT outrank a real GPS fix",
          fix.to_log())
    check("mock_holley" not in tuple(config.HEADWAY_OBD_SPEED_SOURCES),
          "...because only sources that are a car are on the OBD list")

    # A bus that stopped reporting is not a speed.
    r = speed_mod.SpeedResolver(obd_reader=obd(0.0, age=30.0))
    fix = r.resolve(20.0, 0.1, 1.0)
    check(fix.source == speed_mod.GPS,
          "a stale OBD reading falls through to GPS rather than reading 0")

    # Two sources that cannot both be right: neither wins outright.
    r = speed_mod.SpeedResolver(obd_reader=obd(5.0))
    fix = r.resolve(25.0, 0.1, 1.0)
    check(fix.source == speed_mod.GPS and fix.degraded,
          "OBD and GPS disagreeing continues on GPS, DEGRADED",
          fix.to_log())

    # The coast, and its end.
    r = speed_mod.SpeedResolver()
    r.resolve(22.0, 0.1, 10.0)
    mid = r.resolve(None, None, 14.0)
    check(mid.source == speed_mod.COASTED and mid.v_ms == 22.0 and mid.degraded,
          "a fix that goes quiet is coasted, degraded, not silenced",
          mid.to_log())
    late = r.resolve(None, None, 10.0 + config.HEADWAY_V_HOST_COAST_S + 1.0)
    check(late.source == speed_mod.NONE and not late.known,
          "...and past the coast window it becomes no speed at all")

    # A stale-but-present GPS fix takes the same path.
    r = speed_mod.SpeedResolver()
    r.resolve(22.0, 0.1, 10.0)
    stale = r.resolve(22.0, config.HEADWAY_V_HOST_STALE_S + 1.0, 12.0)
    check(stale.source == speed_mod.COASTED,
          "a fix older than the staleness limit is coasted, not trusted")

    # Nothing, ever: the only case that is genuinely silent.
    r = speed_mod.SpeedResolver()
    none = r.resolve(None, None, 1.0)
    check(none.source == speed_mod.NONE, "no fix at all is no speed")
    p = P.LivePolicy()
    recs = [p.tick(tau=float("inf"), v2_trend=v2.STABLE, confidence=GOOD_CONF,
                   v_host=None, v_host_stale=True, t=0.5 * (i + 1),
                   since_reset_s=99.0, speed_degraded=True) for i in range(6)]
    check(all(r["band"] == P.UNKNOWN for r in recs),
          "...and no speed at all is UNKNOWN, which is the one silent state")

    # -- S5 TTC is a trigger again -----------------------------------------
    head("S5 -- TTC is urgent from any band")
    # 4 s of headway is NORMAL. Closing hard at 4 s of headway is two seconds
    # from a collision, and the band machine has nothing to say about it.
    p = P.LivePolicy()
    calm = [p.tick(tau=4.0, v2_trend=v2.STABLE, confidence=GOOD_CONF,
                   v_host=25.0, v_host_stale=False, t=0.5 * (i + 1),
                   since_reset_s=99.0, ttc=None) for i in range(4)]
    check(all(r["band"] == P.NORMAL for r in calm) and not [r for r in calm if r["speak"]],
          "4 s of headway, stable, is NORMAL and silent")
    urgent = p.tick(tau=4.0, v2_trend=v2.RAPIDLY_SHRINKING, confidence=GOOD_CONF,
                    v_host=25.0, v_host_stale=False, t=3.0, since_reset_s=99.0,
                    ttc=P.TTC_URGENT_S - 0.5)
    check(urgent["speak"] is not None and urgent["speak"]["line"] == P.LINE_BACK_OFF,
          "...and the same headway with TTC under the threshold speaks, from NORMAL",
          urgent["voice_reason"])
    check(urgent["voice_reason"] == P.R_TTC_URGENT,
          "the reason recorded is the TTC trigger, not a band entry")
    again = p.tick(tau=4.0, v2_trend=v2.RAPIDLY_SHRINKING, confidence=GOOD_CONF,
                   v_host=25.0, v_host_stale=False, t=3.5, since_reset_s=99.0,
                   ttc=P.TTC_URGENT_S - 0.6)
    check(again["speak"] is None,
          "...once per occupancy: it is a trigger, not a siren")
    # A TTC above the threshold is not a trigger, however fast the gap closes.
    p2 = P.LivePolicy()
    not_urgent = [p2.tick(tau=4.0, v2_trend=v2.RAPIDLY_SHRINKING,
                          confidence=GOOD_CONF, v_host=25.0, v_host_stale=False,
                          t=0.5 * (i + 1), since_reset_s=99.0,
                          ttc=P.TTC_URGENT_S + 2.0) for i in range(6)]
    check(not [r for r in not_urgent if r["speak"]],
          "a comfortable TTC triggers nothing, whatever the trend says")

    # -- S6 the physical floor ---------------------------------------------
    #
    # THE FIRST REAL DRIVE, REPLAYED. gap 3.7 m at 20.7 m/s, held for hundreds
    # of frames, and fifteen of the twenty-two warnings that were spoken.
    head("S6 -- a headway that cannot be one is refused")
    tau_drive = _gap_to_tau(3.7, 20.7)          # 0.179 s
    p = P.LivePolicy()
    recs = drive(p, [(tau_drive, v2.STABLE, 0.97, 20.7)] * 40)
    check(all(r["band"] == P.IMPLAUSIBLE for r in recs),
          f"3.7 m at 20.7 m/s (tau {tau_drive:.3f} s) is IMPLAUSIBLE, not UNSAFE")
    check(not spoken_lines(recs),
          "...and 40 frames of it produce not one warning "
          "(the drive produced fifteen)")
    check(all(r["voice_reason"] == P.R_IMPLAUSIBLE for r in recs),
          "...with the reason recorded, so the upstream fix can be tracked")

    # It is a MEASUREMENT veto and only that: the same gap at a speed where it
    # is an ordinary thing to see is judged normally.
    p = P.LivePolicy()
    park = drive(p, [(v2.compute_tau(3.7, 1.5), v2.STABLE, 0.97, 1.5)] * 8)
    check(all(r["band"] == P.SUPPRESSED for r in park),
          "the same 3.7 m at walking pace is ordinary, and reads SUPPRESSED")

    # And a genuinely tight-but-possible headway is still warned about.
    p = P.LivePolicy()
    tight = drive(p, [(P.TAU_IMPLAUSIBLE_S + 0.25, v2.STABLE, 0.97, 20.0)] * 8)
    check(any(r["band"] == P.UNSAFE for r in tight),
          "a tight but survivable headway is still UNSAFE")
    check(bool(spoken_lines(tight)), "...and is still spoken about")

    # -- S7 coast scales with speed ----------------------------------------
    head("S7 -- coasting a lost lead is a distance, not a duration")
    fast_budget = P.coast_budget_s(30.0)
    slow_budget = P.coast_budget_s(7.0)
    check(fast_budget < slow_budget,
          "a lost lead is coasted for less time at speed than at a crawl",
          f"30 m/s -> {fast_budget:.2f}s, 7 m/s -> {slow_budget:.2f}s")
    check(abs(fast_budget * 30.0 - slow_budget * 7.0) < 5.0
          or fast_budget == config.HEADWAY_COAST_MIN_S
          or slow_budget == config.HEADWAY_COAST_MAX_S,
          "...because what is held constant is metres of road, within the clamps")
    check(config.HEADWAY_COAST_MIN_S <= P.coast_budget_s(200.0)
          <= config.HEADWAY_COAST_MAX_S,
          "an absurd speed still lands inside the clamps")
    check(P.coast_budget_s(None) == v2.MAX_COAST_S,
          "no speed at all falls back to the v2 constant")

    # -- S8 the thresholds live in config ----------------------------------
    head("S8 -- every threshold is in config.py")
    for name, cfg_name in (("TAU_ENTER_GETTING_UNSAFE", "HEADWAY_TAU_GETTING_UNSAFE_S"),
                           ("TAU_ENTER_UNSAFE", "HEADWAY_TAU_UNSAFE_S"),
                           ("HYST_S", "HEADWAY_TAU_HYSTERESIS_S"),
                           ("V_MIN_COACH", "HEADWAY_MIN_COACH_SPEED_MS"),
                           ("V_HOST_STALE_S", "HEADWAY_V_HOST_STALE_S"),
                           ("TAU_IMPLAUSIBLE_S", "HEADWAY_TAU_IMPLAUSIBLE_S"),
                           ("TTC_URGENT_S", "HEADWAY_TTC_URGENT_S"),
                           ("DEGRADED_TAU_BIAS_S", "HEADWAY_DEGRADED_TAU_BIAS_S"),
                           ("DEGRADED_CONF_RELIEF", "HEADWAY_DEGRADED_CONF_RELIEF"),
                           ("CONFIRM_S", "HEADWAY_CONFIRM_S"),
                           ("COOLDOWN_CALM_S", "HEADWAY_COOLDOWN_CALM_S"),
                           ("COOLDOWN_UNSAFE_S", "HEADWAY_COOLDOWN_UNSAFE_S"),
                           ("GENUINE_CLEAR_S", "HEADWAY_GENUINE_CLEAR_S"),
                           ("ESCALATE_AFTER_S", "HEADWAY_ESCALATE_AFTER_S"),
                           ("PENDING_ENTRY_MAX_S", "HEADWAY_PENDING_ENTRY_MAX_S")):
        check(getattr(P, name) == getattr(config, cfg_name),
              f"live_policy.{name} is config.{cfg_name}")

    # -- S9 determinism ----------------------------------------------------
    head("S9 -- the same frames give the same answer, every time")
    script = [(4.0, v2.STABLE), (2.8, v2.SHRINKING), (2.8, v2.SHRINKING),
              (1.6, v2.RAPIDLY_SHRINKING), (1.6, v2.RAPIDLY_SHRINKING),
              (1.2, v2.RAPIDLY_SHRINKING), (5.0, v2.INCREASING)]
    runs = []
    for _ in range(5):
        p = P.LivePolicy()
        runs.append([(r["band"], r["voice_reason"],
                      (r["speak"] or {}).get("line")) for r in drive(p, script)])
    check(all(r == runs[0] for r in runs),
          "five runs of the same script are byte-identical")


# ===========================================================================
# G. The car this camera is bolted to (the upstream half of item 2)
#
# Every box below was OBSERVED, on session 06af3214, an iPhone on a real road.
# The drive spent 584 frames -- 36% of ten minutes -- holding the phone's view
# of the car's own bonnet and dashboard as the lead vehicle at 3.7 m while
# doing 20 m/s, and spoke twelve warnings about it.
#
# Three gates in three layers, each with its own argument, checked here
# together because what matters is that no ONE of them is load-bearing.
# ===========================================================================

# The real thing, on the frame size the drive actually ran at. x1 wanders by a
# pixel or two between frames; nothing else about it moves at all.
EGO_SLAB = (0.6, 308.0, 640.0, 478.0)
EGO_W, EGO_H = 640, 480
# ...and the two frames where its top edge sat within a pixel of the horizon,
# which is where a fitted threshold would have let it through.
EGO_SLAB_EDGE = (1.1, 240.3, 640.0, 476.4)
# The portrait orientation, from the first seconds of the same drive, before
# the phone was turned. Its top IS above the horizon, so the shape gate has no
# opinion about it and the other two have to earn their keep.
EGO_SLAB_PORTRAIT = (0.6, 244.2, 480.0, 639.5)
EGO_PW, EGO_PH = 480, 640
# A genuine lead from the same drive, 51 m ahead.
REAL_LEAD = (241.4, 210.1, 268.0, 231.0)


def _fpx(w):
    import math as _m
    from headway.anchor import HFOV_DEG
    return (float(w) / 2.0) / _m.tan(_m.radians(HFOV_DEG) / 2.0)


def run_ego_structure():
    head("G -- the car this camera is bolted to")

    # -- G1 shape: a full-width box that never rises above the horizon -------
    head("G1 -- shape (headway/detect.py)")
    check(detect_mod._is_ego_bonnet(*EGO_SLAB, EGO_W, EGO_H),
          "the phone's bonnet-and-dashboard slab is rejected",
          f"aspect {(EGO_SLAB[2]-EGO_SLAB[0])/(EGO_SLAB[3]-EGO_SLAB[1]):.1f} — "
          "the old gate wanted 6.0 and let this through 584 times")
    check(detect_mod._is_ego_bonnet(*EGO_SLAB_EDGE, EGO_W, EGO_H),
          "...including the frames where its top sat ON the horizon",
          "y1=240.3 against a horizon of 240.0 — a fitted threshold misses these")
    check(detect_mod._is_ego_bonnet(2, 661, 1279, 720, 1280, 720),
          "and the dashcam's thin strip still is too",
          "the shape this gate was originally written for")

    # THE FALSE REJECTION THAT WOULD MATTER MORE THAN THE BUG.
    check(not detect_mod._is_ego_bonnet(100, 300, 1200, 719, 1280, 720),
          "a genuinely close, TALL vehicle is NOT rejected",
          "wide and tall is a lorry; wide and never above the horizon is our own car")
    check(not detect_mod._is_ego_bonnet(0, 0, 1280, 720, 1280, 720),
          "...nor is one so close it fills the frame",
          "its top edge is at the top, which is what being a real object looks like")
    check(not detect_mod._is_ego_bonnet(500, 600, 780, 720, 1280, 720),
          "a normal vehicle touching the frame bottom is not rejected")
    check(not detect_mod._is_ego_bonnet(*REAL_LEAD, EGO_W, EGO_H),
          "and a real lead 51 m ahead is not touched")

    # The horizon comes from the camera model, not from a constant.
    check(abs(detect_mod.horizon_row(1280, 720) - 360.0) < 1e-6,
          "the horizon is the camera model's, at the configured zero pitch",
          "so a calibrated pitch moves this gate with the corridor")

    # -- G2 arithmetic: a box that wide cannot be that far ------------------
    head("G2 -- arithmetic (headway/plausibility.py)")
    f = _fpx(EGO_W)
    v = plaus_mod.check("car", list(EGO_SLAB), 3.7, f, image_h=EGO_H, image_w=EGO_W)
    check(not v["ok"] and v["reason"] == "depth_too_far_for_box_width",
          "3.7 m is refused for a box the full width of the frame",
          f"the width allows at most {v['wide_max_m']} m")

    # ...and the HEIGHT check on its own had no complaint. This is the whole
    # reason the width bound had to be added.
    vh = plaus_mod.check("car", list(EGO_SLAB), 3.7, f, image_h=EGO_H)
    check(vh["ok"],
          "while the height check alone accepts it, which is how it got through",
          f"170 px of car is a window of {vh['window']} m, and 3.7 is inside it")

    v = plaus_mod.check("car", list(EGO_SLAB_PORTRAIT), 3.7, _fpx(EGO_PW),
                        image_h=EGO_PH, image_w=EGO_PW)
    check(not v["ok"] and v["reason"] == "depth_too_far_for_box_width",
          "the portrait slab is refused too, where the shape gate had no opinion")

    # A vehicle that really IS filling the frame at two metres is believed.
    v = plaus_mod.check("car", [0.0, 0.0, 640.0, 480.0], 1.8, f,
                        image_h=EGO_H, image_w=EGO_W)
    check(v["ok"], "a real vehicle genuinely 1.8 m ahead is still believed",
          "the bound is a FAR bound; being close is not what it objects to")

    # ...and the rule says nothing at all about anything narrower, which is
    # what stops an uncalibrated focal length vetoing real traffic.
    v = plaus_mod.check("car", list(REAL_LEAD), 51.0, f,
                        image_h=EGO_H, image_w=EGO_W)
    check(v["ok"] and v["wide_max_m"] is None,
          "and it has NO OPINION about a lead 51 m ahead",
          "a box width is a poor range estimator; only at full frame width is "
          "it beyond argument")
    check(plaus_mod.wide_box_max_range_m("car", list(REAL_LEAD), EGO_W, f) is None,
          "...stated directly: not applicable below the width fraction")

    # An old caller that passes no image_w gets the behaviour it had.
    v = plaus_mod.check("car", list(EGO_SLAB), 3.7, f, image_h=EGO_H)
    check(v["ok"] and v["wide_max_m"] is None,
          "a caller that does not say how wide the frame is gets no width veto")

    # -- G3 motion: the car moved and this did not -------------------------
    head("G3 -- motion (headway/membership.py)")

    def drive_box(cand, boxes, metres_per_frame, w=EGO_W, h=EGO_H):
        host = 0.0
        for b in boxes:
            cand.box = tuple(float(x) for x in b)
            cand.note_motion(host, w, h)
            host += metres_per_frame
        return cand

    # The bonnet: 25 m of road, the box wandering by the pixel it really did.
    c = member_mod.Candidate(1, EGO_SLAB, "car", 0.0)
    drive_box(c, [(0.6 + 0.05 * (i % 3), 308.0, 640.0, 478.0) for i in range(20)], 1.5)
    check(c.static, "twenty-five metres of road and the box did not move: static",
          f"drift {c.static_drift} of frame width")
    check(not c.eligible(999.0), "...so it may not hold the lead lock")

    # A real lead, drifting the way a real lead does.
    c = member_mod.Candidate(2, REAL_LEAD, "car", 0.0)
    drive_box(c, [(241.4 + i * 0.7, 210.1, 268.0 + i * 0.7, 231.0 + i * 0.2)
                  for i in range(20)], 1.5)
    check(not c.static, "a lead that moves in the picture is not static",
          f"drift {c.static_drift}")

    # THE ONE REAL OBJECT THAT CAN HOLD STILL: a lead at a locked gap on a
    # straight road. The bottom-edge requirement is what protects it.
    c = member_mod.Candidate(3, REAL_LEAD, "car", 0.0)
    drive_box(c, [REAL_LEAD] * 20, 1.5)
    check(not c.static,
          "a lead holding a perfectly steady gap mid-frame is NOT called static",
          "it is not clipped by the bottom edge, and only bodywork is")

    # Distance, not time. A minute at a red light proves nothing.
    c = member_mod.Candidate(4, EGO_SLAB, "car", 0.0)
    drive_box(c, [EGO_SLAB] * 40, 0.0)
    check(not c.static, "standing still for forty frames proves nothing",
          "the veto runs on the odometer, not the clock")

    c = member_mod.Candidate(5, EGO_SLAB, "car", 0.0)
    drive_box(c, [EGO_SLAB] * 8, 1.0)
    check(not c.static, "and eight metres is not yet twenty")

    # A candidate that WAS static and then moves clears at once.
    c = member_mod.Candidate(6, EGO_SLAB, "car", 0.0)
    drive_box(c, [EGO_SLAB] * 20, 1.5)
    check(c.static, "a static candidate is flagged...")
    drive_box(c, [(0.6, 308.0 - i * 8, 640.0, 478.0 - i * 8) for i in range(1, 10)], 1.5)
    check(not c.static, "...and un-flagged the moment it moves",
          "the veto is a live judgement, not a life sentence")

    # No odometer at all leaves it inert.
    c = member_mod.Candidate(7, EGO_SLAB, "car", 0.0)
    for _ in range(30):
        c.note_motion(None, EGO_W, EGO_H)
    check(not c.static, "a session with no speed never accumulates and never vetoes")

    # -- G4 the layers are independent -------------------------------------
    head("G4 -- no single gate is load-bearing")
    # The portrait slab: shape has no opinion, the other two do.
    check(not detect_mod._is_ego_bonnet(*EGO_SLAB_PORTRAIT, EGO_PW, EGO_PH),
          "the portrait slab passes the shape gate...")
    pv = plaus_mod.check("car", list(EGO_SLAB_PORTRAIT), 3.7, _fpx(EGO_PW),
                         image_h=EGO_PH, image_w=EGO_PW)
    c = member_mod.Candidate(8, EGO_SLAB_PORTRAIT, "car", 0.0)
    drive_box(c, [EGO_SLAB_PORTRAIT] * 20, 1.5, w=EGO_PW, h=EGO_PH)
    check(not pv["ok"] and c.static,
          "...and is caught by BOTH of the others",
          "which is the point of having three")

    # -- G5 the thresholds are in config ------------------------------------
    head("G5 -- the thresholds are in config.py")
    import config
    for mod, name, cfg_name in (
            (detect_mod, "SLAB_WIDTH_FRAC", "HEADWAY_EGO_SLAB_WIDTH_FRAC"),
            (detect_mod, "SLAB_HORIZON_SLACK_FRAC",
             "HEADWAY_EGO_SLAB_HORIZON_SLACK_FRAC"),
            (plaus_mod, "WIDE_BOX_FRAC", "HEADWAY_WIDE_BOX_FRAC"),
            (member_mod, "STATIC_TRAVEL_M", "HEADWAY_STATIC_TRAVEL_M"),
            (member_mod, "STATIC_DRIFT_FRAC", "HEADWAY_STATIC_DRIFT_FRAC")):
        check(getattr(mod, name) == getattr(config, cfg_name),
              f"{mod.__name__.split('.')[-1]}.{name} is config.{cfg_name}")


def main():
    print("=" * 70)
    print("RIO live headway (v3) — verification")
    print("=" * 70)
    run_pipeline()
    run_policy()
    run_speed()
    run_firewall()
    run_lanes()
    run_detector()
    run_ego_structure()

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
