"""Spec-compliance checks for the deterministic core.

    python -m headway.selftest

The synthetic clip exercises the happy path (a clean escalation ladder and back
down). These cover the branches it never reaches -- cut-in, occlusion, DEGRADED
suppression, confirmation asymmetry, hysteresis, threshold modulation -- each
asserted against the specific clause of docs/headway_design.md it implements.

No GPU, no models, no video: pure state/filter logic, so it runs in ~1 s and can
gate a commit.
"""
import math

from .filter import HeadwayFilter, MAX_CONSEC_REJECTS
from .state import (COMFORTABLE, MONITOR, CLOSE, WARNING, URGENT, LOST, DEGRADED,
                    INCREASING, STABLE, SLOWLY_SHRINKING, RAPIDLY_SHRINKING,
                    Context, Measurement, Thresholds, WarningStateMachine,
                    classify_trend, compute_confidence, compute_tau, compute_ttc,
                    dead_band, modulate)

_results = []


def check(name, condition, detail=""):
    _results.append((bool(condition), name, detail))
    print(f"  [{'PASS' if condition else 'FAIL'}] {name}" + (f"  -- {detail}" if detail else ""))


def _measure(**kw):
    base = dict(d=50.0, d_dot=0.0, confidence=0.9, depth_conf=0.9,
                track_quality=0.9, trend=STABLE, tau=2.0, ttc=float("inf"))
    base.update(kw)
    ctx = base.pop("ctx", Context(v_host=25.0))
    return Measurement(ctx=ctx, **base)


def test_maths():
    print("\n§3 -- tau / TTC / trend")
    check("tau = d / v_host", abs(compute_tau(50.0, 25.0) - 2.0) < 1e-9)
    check("tau guards v_host -> max(v,0.5)", math.isinf(compute_tau(50.0, 0.0)),
          "below 2 m/s headway is suppressed as meaningless")
    check("tau suppressed under 2 m/s", math.isinf(compute_tau(50.0, 1.9)))
    check("TTC infinite when not closing", math.isinf(compute_ttc(50.0, +2.0)))
    check("TTC infinite below 0.2 m/s closing", math.isinf(compute_ttc(50.0, -0.1)))
    check("TTC = d / -d_dot", abs(compute_ttc(50.0, -10.0) - 5.0) < 1e-9)

    check("trend INCREASING above +0.3", classify_trend(+0.5, 10.0) == INCREASING)
    check("trend STABLE inside dead-band", classify_trend(0.0, 10.0) == STABLE)
    check("trend SLOWLY_SHRINKING", classify_trend(-0.8, 10.0) == SLOWLY_SHRINKING)
    check("trend RAPIDLY_SHRINKING below -1.5", classify_trend(-2.0, 10.0) == RAPIDLY_SHRINKING)
    check("dead-band widens to +-0.5 at highway speed",
          abs(dead_band(30.0) - 0.5) < 1e-9 and abs(dead_band(5.0) - 0.3) < 1e-9)

    # §3 asks for hysteresis so classes don't flicker on a boundary.
    at_edge = -0.30
    stays = classify_trend(at_edge, 10.0, current=STABLE)
    flips = classify_trend(-0.45, 10.0, current=STABLE)
    check("hysteresis holds class exactly on the boundary", stays == STABLE)
    check("hysteresis still yields past the boundary", flips == SLOWLY_SHRINKING)


def test_modulation():
    print("\n§6 -- threshold modulation")
    base = Thresholds()
    dry = modulate(base, Context(v_host=20.0))
    check("no modulation in clear conditions", dry.tau_close == 1.5)

    rain = modulate(base, Context(v_host=20.0, rain=True))
    check("rain scales thresholds by 1.25", abs(rain.tau_comfort - 3.75) < 1e-9,
          f"tau_comfort {base.tau_comfort} -> {rain.tau_comfort}")
    night = modulate(base, Context(v_host=20.0, night=True))
    check("night scales thresholds by 1.25", abs(night.ttc_urgent - 3.125) < 1e-9)

    fast = modulate(base, Context(v_host=30.0))
    check("v_host > 25 overrides tau_close to 1.8", abs(fast.tau_close - 1.8) < 1e-9)
    # Spec order matters: the override lands after the scale, so it is absolute.
    both = modulate(base, Context(v_host=30.0, rain=True))
    check("override applies after scaling (absolute, not scaled)",
          abs(both.tau_close - 1.8) < 1e-9,
          "rain alone would give 1.875; the override tightens it back to 1.8")


def test_confidence():
    print("\n§4 -- confidence")
    ctx = Context(v_host=25.0, v_host_source="CAN")
    hi = compute_confidence(0.9, 0.9, 0.0, 0.0, ctx)
    check("good inputs -> high confidence", hi > 0.7, f"{hi:.3f}")
    check("bad track suppresses", compute_confidence(0.9, 0.05, 0.0, 0.0, ctx) < 0.4,
          "product, not average -- one bad input must suppress")
    check("stale anchor penalised",
          compute_confidence(0.9, 0.9, 12.0, 0.0, ctx) < hi)
    check("coasting decays confidence to 0 by 1.0 s",
          compute_confidence(0.9, 0.9, 0.0, 1.0, ctx) == 0.0)
    check("GPS trusted less than CAN",
          compute_confidence(0.9, 0.9, 0.0, 0.0, Context(v_host_source="GPS")) <
          compute_confidence(0.9, 0.9, 0.0, 0.0, Context(v_host_source="CAN")))


def test_filter_cutin():
    print("\n§4/§5 -- innovation gate and cut-in escape hatch")
    f = HeadwayFilter()
    for _ in range(30):
        f.step(50.0, 0.9, 1 / 15.0)
    check("filter converges on a steady gap", abs(f.d - 50.0) < 1.0, f"d={f.d:.2f}")

    # A car cuts in at 20 m: a discontinuity, not noise.
    s1 = f.step(20.0, 0.9, 1 / 15.0)
    check("first outlier is gated, not absorbed", not s1["accepted"], s1["reason"])
    check("state barely moves after one outlier", abs(f.d - 50.0) < 2.0, f"d={f.d:.2f}")
    s2 = f.step(20.0, 0.9, 1 / 15.0)
    check("second outlier still gated", not s2["accepted"])
    s3 = f.step(20.0, 0.9, 1 / 15.0)
    check(f"{MAX_CONSEC_REJECTS} consecutive rejects -> reset", s3["reason"] == "reset_new_lead")
    check("NEW_LEAD flagged on reset", s3["new_lead"])
    check("filter re-seeds at the new range", abs(f.d - 20.0) < 0.01, f"d={f.d:.2f}")
    check("relative speed reset to unknown (0)", abs(f.d_dot) < 1e-9)


def test_filter_coast():
    print("\n§5 -- occlusion coasting")
    f = HeadwayFilter()
    for _ in range(30):
        f.step(50.0, 0.9, 1 / 15.0)
    for _ in range(5):
        s = f.step(None, 0.0, 1 / 15.0)
    check("coasts on prediction when measurement is missing", s["d"] is not None)
    check("coast_age accumulates", s["coast_age"] > 0.3, f"{s['coast_age']:.2f}s")


def test_state_machine():
    print("\n§6 -- state machine and §4 confirmation asymmetry")

    sm = WarningStateMachine(rate_hz=15.0)
    m = _measure(tau=2.5, trend=STABLE)
    for i in range(6):
        sm.tick(m, t=i / 15.0)
    check("MONITOR needs 6 frames", sm.state == MONITOR)

    # Escalation: 3 frames.
    sm = WarningStateMachine(rate_hz=15.0)
    sm.state = CLOSE
    m = _measure(tau=1.2, trend=SLOWLY_SHRINKING)
    states = [sm.tick(m, t=i / 15.0)["state"] for i in range(4)]
    check("WARNING escalates in exactly 3 frames",
          states[1] == CLOSE and states[2] == WARNING, f"{states}")

    # De-escalation: 12 frames, even though tick() asks for 6.
    sm = WarningStateMachine(rate_hz=15.0)
    sm.state = WARNING
    calm = _measure(tau=2.5, trend=INCREASING)
    seq = [sm.tick(calm, t=i / 15.0)["state"] for i in range(14)]
    first_drop = next((i for i, s in enumerate(seq) if s != WARNING), None)
    check("de-escalation takes 12 frames, not 6", first_drop == 11, f"dropped at index {first_drop}")

    # URGENT bypass, §4 preconditions met.
    sm = WarningStateMachine(rate_hz=15.0)
    sm.state = WARNING
    urgent = _measure(tau=0.8, ttc=1.5, trend=RAPIDLY_SHRINKING,
                      depth_conf=0.9, track_quality=0.9)
    seq = [sm.tick(urgent, t=i / 15.0)["state"] for i in range(3)]
    check("URGENT confirms in 2 frames under high confidence", seq[1] == URGENT, f"{seq}")

    # §4: never single-frame, even at extreme TTC.
    sm = WarningStateMachine(rate_hz=15.0)
    sm.state = WARNING
    check("URGENT never fires on one frame",
          sm.tick(urgent, t=0.0)["state"] != URGENT)

    # Weak inputs -> the bypass is withdrawn, 3 frames instead of 2.
    sm = WarningStateMachine(rate_hz=15.0)
    sm.state = WARNING
    weak = _measure(tau=0.8, ttc=1.5, trend=RAPIDLY_SHRINKING,
                    depth_conf=0.45, track_quality=0.45, confidence=0.55)
    seq = [sm.tick(weak, t=i / 15.0)["state"] for i in range(4)]
    check("URGENT bypass withdrawn when confidence is weak", seq[1] != URGENT, f"{seq}")


def test_suppression():
    print("\n§4/§5 -- DEGRADED and LOST suppression")

    sm = WarningStateMachine(rate_hz=15.0)
    sm.state = COMFORTABLE
    bad = _measure(tau=0.5, ttc=1.0, trend=RAPIDLY_SHRINKING, confidence=0.2)
    check("low confidence -> DEGRADED immediately", sm.tick(bad, t=0.0)["state"] == DEGRADED)
    check("DEGRADED wins over an URGENT-looking measurement", sm.state == DEGRADED,
          "low confidence must SUPPRESS, never cautious-fire")

    sm = WarningStateMachine(rate_hz=15.0)
    stale = _measure(v_host_stale=True)
    check("stale host speed -> DEGRADED", sm.tick(stale, t=0.0)["state"] == DEGRADED)

    sm = WarningStateMachine(rate_hz=15.0)
    lost = _measure(track_lost=True, coast_age=1.5)
    check("track lost past 1.0 s coast -> LOST", sm.tick(lost, t=0.0)["state"] == LOST)

    sm = WarningStateMachine(rate_hz=15.0)
    coasting = _measure(track_lost=True, coast_age=0.5, tau=2.5)
    check("still coasting under 1.0 s is not yet LOST",
          sm.tick(coasting, t=0.0)["state"] != LOST)


def test_new_lead_hold():
    print("\n§5 -- warnings re-confirm after a NEW_LEAD reset")
    sm = WarningStateMachine(rate_hz=15.0)
    sm.state = COMFORTABLE
    m = _measure(tau=0.8, ttc=1.5, trend=RAPIDLY_SHRINKING, new_lead=True)
    r = sm.tick(m, t=0.0)
    check("URGENT is held on the reset frame", r["candidate"] != URGENT, r["reason"])
    check("hold window opened", r["new_lead_hold_s"] > 0.0, f"{r['new_lead_hold_s']}s")


def test_audio_policy():
    print("\n§6/§0 -- transition audio policy")
    sm = WarningStateMachine(rate_hz=15.0)
    sm.state = MONITOR
    m = _measure(tau=1.8, trend=STABLE)
    for i in range(8):
        sm.tick(m, t=i / 15.0)
    tr = [t for t in sm.transitions if t["to"] == CLOSE]
    check("CLOSE uses live TTS", tr and tr[0]["action"]["kind"] == "rio_speak")

    sm = WarningStateMachine(rate_hz=15.0)
    sm.state = WARNING
    urgent = _measure(tau=0.5, ttc=1.2, trend=RAPIDLY_SHRINKING)
    for i in range(3):
        sm.tick(urgent, t=i / 15.0)
    tr = [t for t in sm.transitions if t["to"] == URGENT]
    check("URGENT uses pre-rendered local audio",
          tr and tr[0]["action"]["kind"] == "play_local" and tr[0]["action"]["interrupts"],
          "TTS round trip would be a crash narration, not a warning")
    check("Stage 0 runs in shadow mode", tr and tr[0]["shadow_mode"])
    check("transition carries a full snapshot",
          tr and {"d", "tau", "ttc", "trend", "confidence"} <= set(tr[0]["snapshot"]))


def main():
    print("=" * 68)
    print("headway deterministic-core self-test (docs/headway_design.md)")
    print("=" * 68)
    test_maths()
    test_modulation()
    test_confidence()
    test_filter_cutin()
    test_filter_coast()
    test_state_machine()
    test_suppression()
    test_new_lead_hold()
    test_audio_policy()

    passed = sum(1 for ok, _, _ in _results if ok)
    total = len(_results)
    print("\n" + "=" * 68)
    print(f"{passed}/{total} checks passed")
    if passed != total:
        print("\nFAILURES:")
        for ok, name, detail in _results:
            if not ok:
                print(f"  - {name} {detail}")
    print("=" * 68)
    return 0 if passed == total else 1


if __name__ == "__main__":
    raise SystemExit(main())
