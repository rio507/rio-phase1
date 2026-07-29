"""Verification for the live headway loop (v3 bands + automatic voice).

    python -m headway.live_selftest

Three parts, deliberately separated by what each one is able to prove:

  A. FULL PIPELINE on the shrinking-gap synthetic clip, at the live loop's real
     2 fps cadence, through LiveSession.process() — the exact function
     /headway_frame calls. Depth and the anchor box are supplied from the clip's
     ground truth for the same reason run_clip.py offers `--depth gt`: the
     synthetic clip is a flat-shaded rectangle on a flat road, so DA-V2 has no
     real geometry to infer and running it would measure noise. Everything else
     is live — clock, anchor scheduling, CSRT on the real rendered pixels, the
     Kalman, the trend classifier, the policy, the response payload.

  B. POLICY SCENARIOS, scripted at 2 Hz straight into LivePolicy. The policy is
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
LIVE_FPS = 2.0


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

    head("A -- full live pipeline, shrinking-gap synthetic clip @ 2 fps")

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
    orig_anchor = anchor_mod.LeadAnchor.anchor
    depth_mod.depth_map = provider.depth_map
    depth_mod.roi_depth = provider.roi_depth

    def gt_anchor(self, frame, frame_idx, depth=None):
        g = gt_by_frame.get(provider.frame)
        self.last_anchor_frame = frame_idx
        return (tuple(g["box"]) if g else None), {"reason": "gt"}

    anchor_mod.LeadAnchor.anchor = gt_anchor

    records = []
    try:
        live_mod.reset_session("selftest")
        session = live_mod.get_session("selftest", use_qwen=True)
        cap = cv2.VideoCapture(path)
        stride = int(round(30.0 / LIVE_FPS))       # every 15th frame -> 2 fps
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
        anchor_mod.LeadAnchor.anchor = orig_anchor
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

    head("A4 -- no voice during the coaching warm-up")
    warm = [r for r in records if r["voice_reason"] == P.R_WARMUP]
    check(all(r.get("speak") is None for r in warm),
          "every warm-up frame is silent", f"{len(warm)} warm-up frame(s)")
    check(records[0].get("speak") is None, "first frame of a session never speaks")

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
    # No Qwen per frame: one anchor at session start, none after, on a 16 s clip.
    check(session_anchor_calls(records) == 1,
          "exactly one anchor over the clip -- no Qwen per frame",
          f"{session_anchor_calls(records)} anchored frame(s)")
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

    head("B4b -- a NON-worsening red repeat is still suppressed")
    p = P.LivePolicy()
    recs = drive(p, [(1.5, v2.STABLE)] * 4)
    check(len(spoken_lines(recs)) == 1, "red tier fired once")
    # NORMAL for 3 s -- long enough to leave the band, far short of the 10 s
    # genuine clear -- then straight back into UNSAFE without passing through a
    # confirmed GETTING_UNSAFE. Nothing "got worse" relative to the warning the
    # driver already had, so the 15 s cooldown must hold.
    drive(p, [(3.5, v2.STABLE)] * 6, t0=2.0)
    recs = drive(p, [(1.5, v2.STABLE)] * 6, t0=5.0)
    check(p.band == P.UNSAFE, "band re-entered UNSAFE", p.band)
    check(not spoken_lines(recs),
          "red re-entry from NORMAL inside the cooldown stays silent",
          f"{spoken_lines(recs)}")
    check(any(r["voice_reason"] == P.R_COOLDOWN for r in recs),
          "and is logged as suppressed_by_cooldown")

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

    head("B9 -- de-escalating INTO amber does not fire the calm line")
    p = P.LivePolicy()
    drive(p, [(1.5, v2.STABLE)] * 4)                        # red
    recs = drive(p, [(2.5, v2.INCREASING)] * 6, t0=2.0)     # recovering into amber
    check(p.band == P.GETTING_UNSAFE, "band came back to GETTING_UNSAFE")
    calm = [l for l in spoken_lines(recs) if l[1] == P.LINE_CALM]
    check(not calm, "improving into the band is not a warning event", f"{calm}")


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


def main():
    print("=" * 70)
    print("RIO live headway (v3) — verification")
    print("=" * 70)
    run_pipeline()
    run_policy()
    run_firewall()

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
