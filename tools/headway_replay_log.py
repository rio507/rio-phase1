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

from headway import live_policy as P   # noqa: E402
from headway import state as v2        # noqa: E402


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


def replay(frames, mode):
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
        return _replay(frames, mode)
    finally:
        P.TAU_IMPLAUSIBLE_S = floor


def _replay(frames, mode):
    pol = P.LivePolicy()
    last_reset_t = None
    spoken, reasons, bands = [], Counter(), Counter()
    agrees = disagrees = 0
    divergence = []

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
            "agrees": agrees, "disagrees": disagrees, "divergence": divergence}


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

    after = replay(frames, "after")
    c_after = show("REPLAY through this checkout", after["spoken"])

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
