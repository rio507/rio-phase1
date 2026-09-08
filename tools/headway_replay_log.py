#!/usr/bin/env python3
"""headway_replay_log.py — run a real drive's frames back through the policy.

    python3 tools/headway_replay_log.py training_data/<session>.jsonl

Every headway frame of a drive is logged with the measurements the policy
judged on: tau, the trend, the confidence, the host speed, TTC, whether the
lead was new, whether the track was lost. That is the whole input surface of
LivePolicy.tick, so a drive can be replayed through a changed policy exactly,
frame for frame, and the difference read off against what the driver actually
heard.

WHAT IS RECONSTRUCTED AND WHAT IS NOT. Everything above is read straight from
the log. `since_reset_s` is not logged, so it is reconstructed from the two
conditions that set it in headway/live.py -- a new lead or a lead switch, both
of which ARE logged. Nothing else is inferred.

The "before" run rebuilds the policy as it was on the drive by neutralising
what this checkout added: the physical floor on tau, and the TTC parameter.

AND THE RECONSTRUCTION IS CHECKED RATHER THAN ASSERTED. Every replayed frame's
voice_reason is compared with the reason the drive itself recorded, and the
agreement is printed. On session 06af3214 it is 1949 of 1949 -- the replay
reproduces every band, every suppression and all twenty-two warnings, in order,
which is what makes the difference in the second run attributable to the change
rather than to the harness.
"""
import argparse
import json
import math
import sys
from collections import Counter
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO))

from headway import detect as detect_mod        # noqa: E402
from headway import live_policy as P            # noqa: E402
from headway import membership as member_mod    # noqa: E402
from headway import plausibility as plaus_mod   # noqa: E402
from headway import state as v2                 # noqa: E402


# ---------------------------------------------------------------------------
# THE GATES UPSTREAM OF THE POLICY.
#
# The policy replay below judges the tau it is handed. That is the right thing
# for a policy change and the wrong thing for this one: the ego-structure gates
# stop a box BECOMING the lead, so the frames they catch have no tau at all.
#
# The frame's pixels are gone -- a drive log keeps boxes, not pictures -- so
# what is replayed here is the gates' judgement of the LOGGED LEAD BOX, which
# is exactly the box each gate is given in the live loop. The shape and width
# gates are then bit-for-bit the live decision. The motion gate is not quite:
# it needs a per-candidate history and the log carries one box per frame, so it
# is rebuilt per lead id from the logged boxes and the logged speed. That
# reconstruction can only ever make the gate FIRE LESS than it does live, since
# live it sees every candidate and not only the one that won.
# ---------------------------------------------------------------------------
def _frame_dims(box):
    """The frame this box came from. The log does not carry image size.

    Inferred from the box, which is safe here because every box in question is
    clamped to the frame edge: a lead box reaching x=640 is on a 640-wide
    frame. This drive ran portrait for its first seconds and landscape after.
    """
    return (640.0, 480.0) if round(box[2]) == 640 else (480.0, 640.0)


def _focal(width_px):
    from headway.anchor import HFOV_DEG
    return (float(width_px) / 2.0) / math.tan(math.radians(HFOV_DEG) / 2.0)


def static_frames(frames):
    """Which (lead_id, t) the motion veto would have refused."""
    runs = {}
    for f in frames:
        if f.get("lead_box") and f.get("lead_id") is not None:
            runs.setdefault(f["lead_id"], []).append(f)
    out = set()
    for lid, seq in runs.items():
        cand = member_mod.Candidate(lid, seq[0]["lead_box"], "car", seq[0]["t"])
        host = 0.0
        for i, f in enumerate(seq):
            if i:
                dt = min(max(_num(f.get("t"), 0.0) - _num(seq[i - 1].get("t"), 0.0), 0.0), 2.0)
                host += _num(f.get("v_host"), 0.0) * dt
            w, h = _frame_dims(f["lead_box"])
            cand.box = tuple(float(v) for v in f["lead_box"])
            cand.note_motion(host, w, h)
            if cand.static:
                out.add((lid, round(_num(f.get("t"), 0.0), 4)))
    return out


def gates_reject(frame, static_set):
    """Which gate, if any, would have stopped this box being the lead."""
    box = frame.get("lead_box")
    if not box:
        return None
    w, h = _frame_dims(box)
    x1, y1, x2, y2 = box
    if detect_mod._is_ego_bonnet(x1, y1, x2, y2, w, h):
        return "shape"
    depth = _num(frame.get("distance_m"))
    if depth is not None:
        v = plaus_mod.check(frame.get("lead_label") or "car", box, depth,
                            _focal(w), image_h=h, image_w=w)
        if v["reason"] == "depth_too_far_for_box_width":
            return "width"
    if (frame.get("lead_id"), round(_num(frame.get("t"), 0.0), 4)) in static_set:
        return "motion"
    return None


def load(path):
    frames = []
    for line in open(path):
        try:
            ev = json.loads(line)
        except Exception:
            continue
        if ev.get("kind") == "headway":
            frames.append(ev["payload"])
    return frames


def _num(x, default=None):
    if x is None:
        return default
    try:
        f = float(x)
    except (TypeError, ValueError):
        return default
    return f if math.isfinite(f) else default


def replay(frames, mode, static_set=None):
    """mode: 'before' (the drive's policy) or 'after' (this checkout's).

    'before' is reconstructed by neutralising the two things this checkout
    added -- the physical floor on tau, and the TTC parameter -- rather than by
    checking out the old file. The `agrees` count returned alongside is what
    makes that reconstruction checkable: it is the number of frames whose
    replayed voice_reason matches the reason the drive itself recorded.
    """
    floor = P.TAU_IMPLAUSIBLE_S
    if mode == "before":
        # The floor did not exist. Zero disables it without touching config,
        # and it is put back in the `finally` below.
        P.TAU_IMPLAUSIBLE_S = 0.0
    try:
        return _replay(frames, mode, static_set)
    finally:
        P.TAU_IMPLAUSIBLE_S = floor


def _replay(frames, mode, static_set=None):
    pol = P.LivePolicy()
    last_reset_t = None
    spoken, reasons, bands = [], Counter(), Counter()
    agrees = disagrees = 0
    divergence = []
    gated = Counter()
    # Is the gap this frame reports inherited from a lead the gates refused?
    ghost = False

    for f in frames:
        t = _num(f.get("t"), 0.0)
        new_lead = bool(f.get("new_lead"))
        # The filter is re-seeded on a new lead OR a lead switch, and both are
        # logged. reset_t is what the warm-up is measured from -- see
        # headway/live.py, where exactly these two conditions set it.
        if new_lead or f.get("lead_switch"):
            last_reset_t = t
        since_reset = 99.0 if last_reset_t is None else (t - last_reset_t)

        tau = _num(f.get("tau_s"), float("inf"))
        if f.get("tau_s") is None:
            tau = float("inf")
        v = _num(f.get("v_host"))
        stale = bool(f.get("v_host_stale"))
        ttc = _num(f.get("ttc_s"))
        conf = _num(f.get("confidence"), 0.0)
        trend = f.get("trend_detail") or v2.STABLE

        kw = dict(tau=tau, v2_trend=trend, confidence=conf, v_host=v,
                  v_host_stale=stale, t=t, since_reset_s=since_reset,
                  new_lead=new_lead, track_lost=bool(f.get("track_lost")))

        if mode == "after" and static_set is not None:
            why = gates_reject(f, static_set)
            if why:
                # THE LEAD NEVER EXISTS ON THIS FRAME. Not a suppressed
                # warning about a close car -- no car, no range, no tau. The
                # same state the loop is already in whenever nothing eligible
                # is in the corridor.
                gated[why] += 1
                ghost = True
                kw["tau"] = float("inf")
                kw["track_lost"] = True
            elif f.get("lead_box"):
                # A real box: whatever came before, we have a lead again.
                ghost = False
            elif ghost:
                # AND THE GHOST OF ONE. A box-less frame still carrying a
                # distance is the Kalman coasting the lead it had; if that lead
                # was never admitted, there is nothing to coast.
                #
                # This is a MODEL of the filter rather than a replay of it --
                # filter state is not in the log -- and it is the one place
                # this tool infers rather than reads. It is stated here because
                # it accounts for 428 of the frames in the table below, and a
                # reader is entitled to know which ones were reasoned about.
                gated["ghost_of_a_gated_lead"] += 1
                kw["tau"] = float("inf")
                kw["track_lost"] = True

        if mode == "after":
            # The drive's own frames carried no speed-source resolution -- the
            # resolver did not exist -- so the replay is run with the speed
            # taken at face value, DEGRADED off. That makes this a strictly
            # conservative comparison: the widening can only ever add warnings,
            # and none of the ones counted below come from it.
            kw["ttc"] = ttc
            kw["speed_degraded"] = False
        else:
            # The policy as it was: TTC computed, logged, and not passed in.
            kw["ttc"] = None

        r = pol.tick(**kw)
        bands[r["band"]] += 1
        reasons[r["voice_reason"]] += 1
        if mode == "before":
            if r["voice_reason"] == f.get("voice_reason"):
                agrees += 1
            else:
                disagrees += 1
                if len(divergence) < 8:
                    divergence.append((round(t, 1), f.get("voice_reason"),
                                       r["voice_reason"]))
        if r["speak"]:
            spoken.append({
                "t": round(t, 1), "line": r["speak"]["line"],
                "reason": r["voice_reason"],
                "tau": None if not math.isfinite(tau) else round(tau, 2),
                "gap_m": f.get("distance_m"), "v": v, "ttc": ttc,
            })
    return {"spoken": spoken, "reasons": reasons, "bands": bands,
            "agrees": agrees, "disagrees": disagrees, "divergence": divergence,
            "gated": gated}


def actually_spoke(frames):
    return [{"t": round(_num(f.get("t"), 0), 1), "line": f["speak"]["line"],
             "reason": f.get("voice_reason"),
             "tau": f.get("tau_s"), "gap_m": f.get("distance_m"),
             "v": f.get("v_host"), "ttc": f.get("ttc_s")}
            for f in frames if f.get("speak")]


def classify(w):
    """Is this warning about a headway that could physically have been one?

    The only judgement this tool makes, and it is arithmetic: a gap under
    TAU_IMPLAUSIBLE_S of travel at the speed of the moment is shorter than
    human reaction time, and a car that held one for hundreds of frames did
    not hold one.
    """
    tau, v = _num(w.get("tau")), _num(w.get("v"))
    if tau is None or v is None:
        return "unknown"
    if v < P.V_MIN_COACH:
        return "low_speed"
    return "implausible" if tau < P.TAU_IMPLAUSIBLE_S else "plausible"


def show(title, warnings):
    c = Counter(classify(w) for w in warnings)
    print(f"\n{title}: {len(warnings)} warnings "
          f"({c['plausible']} on a plausible headway, "
          f"{c['implausible']} on one that could not have been)")
    for w in warnings:
        print(f"   t={w['t']:7.1f}  {w['line']:<15} tau={w['tau']}  "
              f"gap={w['gap_m']}  v={w['v']}  ttc={w['ttc']}  [{classify(w)}]")
    return c


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("session")
    args = ap.parse_args()

    frames = load(args.session)
    if not frames:
        raise SystemExit(f"no headway frames in {args.session}")
    print(f"{Path(args.session).name}: {len(frames)} headway frames, "
          f"{_num(frames[-1].get('t'), 0) - _num(frames[0].get('t'), 0):.0f} s")

    heard = actually_spoke(frames)
    c_heard = show("WHAT THE DRIVER ACTUALLY HEARD", heard)

    before = replay(frames, "before")
    c_before = show("REPLAY of the policy as it was", before["spoken"])
    n = before["agrees"] + before["disagrees"]
    print(f"   fidelity: {before['agrees']}/{n} frames replay to the same "
          f"voice_reason the drive recorded ({100.0 * before['agrees'] / n:.1f}%)")
    for d in before["divergence"]:
        print(f"     t={d[0]}: log said {d[1]!r}, replay says {d[2]!r}")

    statics = static_frames(frames)
    after = replay(frames, "after", static_set=statics)
    c_after = show("REPLAY through this checkout", after["spoken"])
    if after["gated"]:
        total = sum(after["gated"].values())
        print(f"   {total} of {len(frames)} frames had their lead removed before "
              f"the policy ever saw it:")
        for k, n in after["gated"].most_common():
            print(f"     {n:5d}  {k}")

    print("\n=== bands ===")
    keys = sorted(set(before["bands"]) | set(after["bands"]))
    print(f"  {'band':<16} {'before':>8} {'after':>8}")
    for k in keys:
        print(f"  {k:<16} {before['bands'][k]:>8} {after['bands'][k]:>8}")

    print("\n=== warnings on a headway that could not have been one ===")
    print(f"  heard on the drive : {c_heard['implausible']} of {len(heard)}")
    print(f"  after              : {c_after['implausible']} of {len(after['spoken'])}")

    print("\n=== suppression reasons, after ===")
    for k, n in after["reasons"].most_common():
        print(f"  {n:6d}  {k}")


if __name__ == "__main__":
    main()
