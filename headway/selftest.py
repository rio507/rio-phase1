"""Spec-compliance checks for the v2 deterministic core.

    python -m headway.selftest

Each check names the clause of docs/warning_logic_v2.md it enforces. The
synthetic clips exercise whole trajectories; this covers the branches and
boundaries they don't reach, plus every cell of the §4 voice-policy table.

No GPU, no models, no video -- pure state/filter logic, ~1 s, so it can gate a
commit.
"""
import math

from .filter import (HeadwayFilter, MIN_CONSEC_REJECTS, REJECT_WINDOW_S,
                     REJECT_WINDOW_TOL_S)
from . import state as S
from .state import (COMFORTABLE, NORMAL, GETTING_UNSAFE, UNSAFE, CRITICAL,
                    LOST, DEGRADED, SUPPRESSED_LOW_SPEED, URGENT,
                    INCREASING, STABLE, SLOWLY_SHRINKING, RAPIDLY_SHRINKING,
                    Context, Measurement, WarningStateMachine,
                    classify_tau, classify_trend, compute_confidence,
                    compute_tau, compute_ttc, trend_bands)

_results = []
DT = 1.0 / 12.0          # v2 targets a 12 Hz loop


def check(name, condition, detail=""):
    _results.append((bool(condition), name, detail))
    print(f"  [{'PASS' if condition else 'FAIL'}] {name}" + (f"  -- {detail}" if detail else ""))


def measure(**kw):
    base = dict(d=50.0, d_dot=0.0, tau=3.0, ttc=float("inf"), trend=STABLE,
                confidence=0.9, depth_conf=0.9, track_quality=0.9)
    base.update(kw)
    ctx = base.pop("ctx", None) or Context(v_host=25.0)
    return Measurement(ctx=ctx, **base)


def run(sm, m, n, t0=0.0, dt=DT):
    """Tick n times with the same measurement; return the tick dicts."""
    return [sm.tick(m, t=t0 + i * dt, dt=dt) for i in range(n)]


def settled(band=None, **kw):
    """A machine past COACH_WARMUP_S, optionally parked in a band."""
    sm = WarningStateMachine(rate_hz=12.0, **kw)
    sm.tick(measure(new_lead=True), t=0.0, dt=DT)      # init, starts warmup
    run(sm, measure(), int(S.COACH_WARMUP_S / DT) + 3, t0=DT)
    if band is not None:
        sm.tau_band = band
    sm.voice_log.clear()
    return sm


def voice_clips(sm):
    return [v["action"].get("clip") or v["action"].get("kind") for v in sm.voice_log]


# ---------------------------------------------------------------------------
def test_bands():
    print("\n§1 -- five tau bands")
    for tau, want in ((3.0, COMFORTABLE), (2.5, COMFORTABLE), (2.4, NORMAL),
                      (2.0, NORMAL), (1.9, GETTING_UNSAFE), (1.5, GETTING_UNSAFE),
                      (1.4, UNSAFE), (1.0, UNSAFE), (0.9, CRITICAL), (0.2, CRITICAL)):
        got = classify_tau(tau, None)
        check(f"tau={tau} -> {want}", got == want, f"got {got}")
    check("infinite tau -> COMFORTABLE", classify_tau(float("inf"), None) == COMFORTABLE)


def test_hysteresis():
    print("\n§2 -- 0.2 s exit hysteresis")
    # enter GETTING_UNSAFE at 2.0, exit back to NORMAL only above 2.2
    check("holds GETTING_UNSAFE at tau=2.1",
          classify_tau(2.1, GETTING_UNSAFE) == GETTING_UNSAFE)
    check("exits GETTING_UNSAFE at tau=2.25",
          classify_tau(2.25, GETTING_UNSAFE) == NORMAL)
    check("holds UNSAFE at tau=1.6", classify_tau(1.6, UNSAFE) == UNSAFE)
    check("exits UNSAFE at tau=1.75", classify_tau(1.75, UNSAFE) == GETTING_UNSAFE)
    check("holds CRITICAL at tau=1.1", classify_tau(1.1, CRITICAL) == CRITICAL)
    check("exits CRITICAL at tau=1.25", classify_tau(1.25, CRITICAL) == UNSAFE)
    check("entry is NOT hysteretic (2.0 from NORMAL enters immediately)",
          classify_tau(1.99, NORMAL) == GETTING_UNSAFE)


def test_maths():
    print("\n§3/§9 -- tau, TTC, trend")
    check("tau = d / v_host", abs(compute_tau(50.0, 25.0) - 2.0) < 1e-9)
    check("tau guards div-by-zero", math.isfinite(compute_tau(50.0, 0.0)))
    check("TTC infinite when not closing", math.isinf(compute_ttc(50.0, +2.0)))
    check("TTC = d / -d_dot", abs(compute_ttc(50.0, -10.0) - 5.0) < 1e-9)

    check("trend INCREASING above +0.3", classify_trend(+0.5, 20.0) == INCREASING)
    check("trend STABLE in dead-band", classify_trend(0.1, 20.0) == STABLE)
    check("trend SLOWLY_SHRINKING", classify_trend(-0.8, 20.0) == SLOWLY_SHRINKING)
    check("trend RAPIDLY_SHRINKING below -1.5", classify_trend(-2.0, 20.0) == RAPIDLY_SHRINKING)

    dead_lo, rapid_lo = trend_bands(20.0)
    dead_hi, rapid_hi = trend_bands(30.0)
    check("trend bands x1.5 above 25 m/s",
          abs(dead_hi - dead_lo * 1.5) < 1e-9 and abs(rapid_hi - rapid_lo * 1.5) < 1e-9,
          f"dead {dead_lo}->{dead_hi}, rapid {rapid_lo}->{rapid_hi}")
    check("-2.0 m/s is only SLOWLY at highway speed",
          classify_trend(-2.0, 30.0) == SLOWLY_SHRINKING, "rapid bar is -2.25 there")

    # Estimate-significance gate (not in v2; see TREND_SIGNIFICANCE_SIGMA).
    check("unconverged d_dot reads STABLE",
          classify_trend(-0.53, 25.0, None, d_dot_sigma=1.56) == STABLE,
          "0.34 sigma -- filter does not yet know the closing speed")
    check("converged d_dot is believed",
          classify_trend(-0.53, 25.0, None, d_dot_sigma=0.10) == SLOWLY_SHRINKING)


def test_confidence():
    print("\n§8 -- confidence as a weighted sum")
    check("all-perfect inputs -> 1.0",
          abs(compute_confidence(1.0, 0.0, 1.0, 0.0, 0.0) - 1.0) < 1e-9)
    w = compute_confidence(1.0, 0.0, 0.0, 0.0, 0.0)
    check("dropping track costs exactly its 0.25 weight", abs(w - 0.75) < 1e-9, f"{w}")
    w2 = compute_confidence(0.0, 1.0, 1.0, 0.0, 0.0)
    check("depth terms carry 0.60 combined", abs(w2 - 0.40) < 1e-9, f"{w2}")
    check("sum, not product: one zero input does not zero the score",
          compute_confidence(1.0, 0.0, 0.0, 12.0, 0.0) > 0.0,
          "v1 multiplied these; v2 §8 specifies a weighted sum")
    check("coasting decays to zero by 1.0 s (headway_design §5)",
          compute_confidence(1.0, 0.0, 1.0, 0.0, 1.0) == 0.0)
    check("stale anchor costs its 0.15 weight",
          compute_confidence(1.0, 0.0, 1.0, 99.0, 0.0) < 1.0)

    print("\n§8 -- confidence tiers 0.4 / 0.7")
    sm = settled(band=GETTING_UNSAFE)
    r = sm.tick(measure(tau=1.8, confidence=0.35), t=10.0, dt=DT)
    check("< 0.4 -> DEGRADED", r["state"] == DEGRADED)
    sm = settled(band=GETTING_UNSAFE)
    r = sm.tick(measure(tau=1.8, confidence=0.55), t=10.0, dt=DT)
    check("0.4-0.7 -> coaching allowed", r["state"] != DEGRADED)
    check("0.4-0.7 reported as 'reduced'", r["confidence_tier"] == "reduced")
    sm = settled(band=GETTING_UNSAFE)
    r = sm.tick(measure(tau=1.8, confidence=0.9), t=10.0, dt=DT)
    check(">= 0.7 -> 'full'", r["confidence_tier"] == "full")


def test_confirmation():
    print("\n§3 -- temporal confirmation")
    sm = settled(band=NORMAL)
    seq = [r["tau_band"] for r in run(sm, measure(tau=1.8), 5, t0=10.0)]
    check("escalate one band takes 3 frames",
          seq[1] == NORMAL and seq[2] == GETTING_UNSAFE, f"{seq}")

    sm = settled(band=GETTING_UNSAFE)
    seq = [r["tau_band"] for r in run(sm, measure(tau=3.0), 14, t0=10.0)]
    first = next((i for i, b in enumerate(seq) if b != GETTING_UNSAFE), None)
    check("de-escalate takes 12 frames", first == 11, f"dropped at index {first}")

    print("\n§3 -- URGENT confirmation")
    sm = settled(band=NORMAL)
    urgent = measure(tau=2.2, ttc=1.5, trend=RAPIDLY_SHRINKING,
                     confidence=0.9, depth_conf=0.9, track_quality=0.9)
    seq = [r["urgent"] for r in run(sm, urgent, 3, t0=10.0)]
    check("URGENT confirms in 2 frames at high confidence", seq[1] is True, f"{seq}")
    check("URGENT never fires on one frame", seq[0] is False)

    sm = settled(band=NORMAL)
    weak = measure(tau=2.2, ttc=1.5, trend=RAPIDLY_SHRINKING,
                   confidence=0.55, depth_conf=0.5, track_quality=0.5)
    seq = [r["urgent"] for r in run(sm, weak, 4, t0=10.0)]
    check("bypass disabled below 0.7 -> 3 frames", seq[1] is False and seq[2] is True,
          f"{seq}")

    sm = settled(band=NORMAL)
    mid = measure(tau=2.2, ttc=2.3, trend=RAPIDLY_SHRINKING,
                  confidence=0.9, depth_conf=0.9, track_quality=0.9)
    seq = [r["urgent"] for r in run(sm, mid, 4, t0=10.0)]
    check("TTC in [2.0, 2.5) -> 3 frames (retained §4 condition)",
          seq[1] is False and seq[2] is True, f"{seq}")

    print("\n§3 -- recovery from a system state takes 6 frames")
    sm = settled(band=NORMAL)
    sm.tick(measure(tau=2.2, confidence=0.2), t=10.0, dt=DT)
    check("entered DEGRADED immediately", sm.state == DEGRADED)
    seq = [r["state"] for r in run(sm, measure(tau=2.2), 8, t0=11.0)]
    first = next((i for i, s in enumerate(seq) if s != DEGRADED), None)
    check("leaves DEGRADED after 6 frames", first == 5, f"index {first}: {seq[:8]}")


def test_urgent_orthogonal():
    print("\n§0/§1 -- URGENT is orthogonal to the tau bands")
    # The case v2 §0 Challenge 1 names: comfortable tau, lethal TTC.
    sm = settled(band=COMFORTABLE)
    m = measure(tau=2.6, ttc=1.8, trend=RAPIDLY_SHRINKING)
    seq = run(sm, m, 4, t0=10.0)
    check("URGENT fires from COMFORTABLE when TTC is low",
          any(r["urgent"] for r in seq),
          "tau 2.6 s with TTC 1.8 s -- the bands alone would miss this entirely")
    check("tau band stays COMFORTABLE underneath", seq[-1]["tau_band"] == COMFORTABLE)
    check("display state becomes URGENT", seq[-1]["state"] == URGENT)

    sm = settled(band=NORMAL)
    calm = measure(tau=2.2, ttc=1.5, trend=SLOWLY_SHRINKING)
    check("URGENT requires RAPIDLY_SHRINKING, not just low TTC",
          not any(r["urgent"] for r in run(sm, calm, 5, t0=10.0)))


def test_low_speed():
    print("\n§7 -- low-speed handling")
    sm = settled(band=CRITICAL)
    ctx = Context(v_host=3.0)
    r = sm.tick(measure(tau=0.8, trend=STABLE, ctx=ctx), t=10.0, dt=DT)
    check("v_host < 5 m/s -> SUPPRESSED_LOW_SPEED", r["state"] == SUPPRESSED_LOW_SPEED)
    check("no tau coaching while suppressed", not r["voice_fired"])

    sm = settled(band=COMFORTABLE)
    ctx = Context(v_host=3.0)
    m = measure(tau=0.8, ttc=1.5, trend=RAPIDLY_SHRINKING, ctx=ctx)
    seq = run(sm, m, 4, t0=10.0)
    check("TTC urgent path stays ARMED below 5 m/s",
          any(r["urgent"] for r in seq), "v2 §7: only tau coaching is suppressed")

    sm = settled(band=COMFORTABLE)
    ctx = Context(v_host=6.0)
    r = run(sm, measure(tau=2.3, ctx=ctx), 5, t0=10.0)[-1]
    # tau 2.3 / 1.2 = 1.92 -> GETTING_UNSAFE, where unscaled it would be NORMAL.
    check("5-8 m/s relaxes thresholds by 1.2", r["tau_band"] == GETTING_UNSAFE,
          f"tau 2.3 scaled to {r['tau_scaled']} -> {r['tau_band']}")
    check("low_speed_scale reported", abs(r["low_speed_scale"] - 1.2) < 1e-9)

    sm = settled(band=NORMAL)
    r = sm.tick(measure(tau=2.2, v_host_stale=True), t=10.0, dt=DT)
    check("stale host speed -> DEGRADED", r["state"] == DEGRADED)


def test_system_states():
    print("\n§1 -- LOST / DEGRADED")
    sm = settled(band=NORMAL)
    r = sm.tick(measure(track_lost=True, coast_age=1.5), t=10.0, dt=DT)
    check("track lost past 1.0 s coast -> LOST", r["state"] == LOST)
    sm = settled(band=NORMAL)
    r = sm.tick(measure(tau=2.2, track_lost=True, coast_age=0.5), t=10.0, dt=DT)
    check("still coasting under 1.0 s is not LOST", r["state"] != LOST)

    sm = settled(band=COMFORTABLE)
    m = measure(tau=0.5, ttc=1.0, trend=RAPIDLY_SHRINKING, confidence=0.2)
    seq = run(sm, m, 5, t0=10.0)
    check("DEGRADED suppresses the urgent path entirely",
          all(r["state"] == DEGRADED and not r["urgent"] for r in seq),
          "v2 §8: low confidence suppresses, never warns cautiously")
    check("DEGRADED fires no voice", not any(r["voice_fired"] for r in seq))


def test_voice_policy():
    print("\n§4 -- voice policy table")

    # Row: tau > 2.0 -- silent in every trend column.
    for trend in (INCREASING, STABLE, SLOWLY_SHRINKING, RAPIDLY_SHRINKING):
        sm = settled(band=NORMAL)
        run(sm, measure(tau=2.2, trend=trend), 20, t0=10.0)
        check(f"NORMAL + {trend} -> silent", not voice_clips(sm), f"{voice_clips(sm)}")

    # Row: GETTING_UNSAFE
    sm = settled(band=GETTING_UNSAFE)
    run(sm, measure(tau=1.8, trend=STABLE), 20, t0=10.0)
    check("GETTING_UNSAFE + STABLE -> silent", not voice_clips(sm))

    sm = settled(band=GETTING_UNSAFE)
    run(sm, measure(tau=1.8, trend=SLOWLY_SHRINKING), 20, t0=10.0)
    check("GETTING_UNSAFE + SLOWLY -> calm line",
          any(v["action"]["line"] == S.LINE_CALM for v in sm.voice_log),
          f"{[v['action'].get('line') for v in sm.voice_log]}")
    check("calm line is live TTS, not a pre-rendered clip",
          sm.voice_log and sm.voice_log[0]["action"]["kind"] == "rio_speak")

    sm = settled(band=GETTING_UNSAFE)
    run(sm, measure(tau=1.8, trend=RAPIDLY_SHRINKING, ttc=6.0), 20, t0=10.0)
    check("GETTING_UNSAFE + RAPIDLY -> stronger line",
          "closing_fast.wav" in voice_clips(sm), f"{voice_clips(sm)}")

    # Row: UNSAFE
    sm = settled(band=UNSAFE)
    run(sm, measure(tau=1.2, trend=STABLE), 20, t0=10.0)
    check("UNSAFE + STABLE -> silent (logs instead)", not voice_clips(sm))

    sm = settled(band=UNSAFE)
    run(sm, measure(tau=1.2, trend=SLOWLY_SHRINKING), 20, t0=10.0)
    check("UNSAFE + SLOWLY -> stronger line",
          "too_close.wav" in voice_clips(sm), f"{voice_clips(sm)}")

    sm = settled(band=UNSAFE)
    run(sm, measure(tau=1.2, trend=RAPIDLY_SHRINKING, ttc=5.0), 20, t0=10.0)
    check("UNSAFE + RAPIDLY, TTC >= 4 -> stronger line only",
          "too_close.wav" in voice_clips(sm) and "brake.wav" not in voice_clips(sm),
          f"{voice_clips(sm)}")

    sm = settled(band=UNSAFE)
    run(sm, measure(tau=1.2, trend=RAPIDLY_SHRINKING, ttc=3.5), 20, t0=10.0)
    check("UNSAFE + RAPIDLY, TTC < 4 -> urgent clip (TTC_UNSAFE_ASSIST)",
          "brake.wav" in voice_clips(sm), f"{voice_clips(sm)}")

    # Row: CRITICAL
    sm = settled(band=CRITICAL)
    ticks = run(sm, measure(tau=0.9, trend=STABLE), 20, t0=10.0)
    check("CRITICAL + STABLE -> silent", not voice_clips(sm), f"{voice_clips(sm)}")
    check("CRITICAL + STABLE logs persistent_tailgate",
          any(r["persistent_tailgate"] for r in ticks))
    check("CRITICAL arms the pre-cached urgent buffer",
          ticks[-1]["urgent_buffer_armed"])

    sm = settled(band=CRITICAL)
    run(sm, measure(tau=0.9, trend=SLOWLY_SHRINKING), 20, t0=10.0)
    check("CRITICAL + SLOWLY -> urgent-adjacent (never quieter than UNSAFE)",
          voice_clips(sm), f"{voice_clips(sm)}")

    sm = settled(band=CRITICAL)
    run(sm, measure(tau=0.9, trend=RAPIDLY_SHRINKING, ttc=6.0), 20, t0=10.0)
    check("CRITICAL + RAPIDLY -> brake, even with TTC above the urgent bar",
          "brake.wav" in voice_clips(sm), f"{voice_clips(sm)}")

    print("\n§4 note 1 -- stable tailgating never becomes a warning")
    sm = settled(band=CRITICAL)
    ticks = run(sm, measure(tau=0.9, trend=STABLE), 200, t0=10.0)
    check("200 frames of stable tau=0.9 produce zero voice", not sm.voice_log,
          "this is the case v2 §0 Challenge 1 exists for")
    check("but every frame is logged",
          sum(1 for r in ticks if r["persistent_tailgate"]) == 200)


def test_cooldowns():
    print("\n§6 -- voice cooldowns (voice only)")
    # Strong line: 15 s, and re-entry required.
    sm = settled(band=UNSAFE)
    m = measure(tau=1.2, trend=SLOWLY_SHRINKING)
    run(sm, m, 30, t0=10.0)
    n_first = len(sm.voice_log)
    run(sm, m, 30, t0=12.0)
    check("stronger line does not repeat inside 15 s", len(sm.voice_log) == n_first,
          f"{len(sm.voice_log)} events")

    # Re-arm requires leaving the band, not merely waiting out the cooldown.
    sm = settled(band=UNSAFE)
    run(sm, m, 20, t0=10.0)
    n = len(sm.voice_log)
    run(sm, m, 20, t0=100.0)          # cooldown long expired, never left the band
    check("re-entry required even after the cooldown expires",
          len(sm.voice_log) == n, f"{len(sm.voice_log)} events")

    sm = settled(band=UNSAFE)
    run(sm, m, 20, t0=10.0)
    n = len(sm.voice_log)
    run(sm, measure(tau=3.0), 20, t0=40.0)            # leave the band
    sm.tau_band = UNSAFE                              # and come back
    run(sm, m, 20, t0=100.0)
    check("fires again after leaving and re-entering", len(sm.voice_log) > n,
          f"{n} -> {len(sm.voice_log)}")

    print("\n§6 -- URGENT has no cooldown, only a 2.0 s min gap")
    sm = settled(band=CRITICAL)
    urgent = measure(tau=0.8, ttc=1.5, trend=RAPIDLY_SHRINKING)
    run(sm, urgent, 12, t0=10.0)
    first = len(sm.voice_log)
    check("urgent fires", first >= 1)
    run(sm, urgent, 6, t0=11.0)       # 1 s later -- inside the gap
    check("urgent does not repeat inside 2.0 s", len(sm.voice_log) == first)
    run(sm, urgent, 6, t0=12.5)       # 2.5 s later -- gap elapsed
    check("urgent repeats after 2.0 s", len(sm.voice_log) > first,
          f"{first} -> {len(sm.voice_log)}")


def test_cutin_and_warmup():
    print("\n§5 -- cut-in reset, and the coaching warm-up")
    f = HeadwayFilter()
    for _ in range(30):
        f.step(50.0, 0.9, DT)
    s1 = f.step(20.0, 0.9, DT)
    check("first outlier gated, not absorbed", not s1["accepted"], s1["reason"])

    # The reject run is a TIME window (REJECT_WINDOW_S) with a hard floor of
    # MIN_CONSEC_REJECTS frames, so the frame COUNT varies with cadence while
    # the wall time does not. A flat frame count meant 0.25 s at this 12 Hz
    # harness and 1.5 s at the live loop's 2 fps -- two different behaviours
    # from one constant.
    for label, dt in (("12 Hz", 1.0 / 12.0), ("4 fps", 0.25), ("2 fps", 0.5)):
        g = HeadwayFilter()
        for _ in range(30):
            g.step(50.0, 0.9, dt)
        n, snap = 0, None
        while n < 50:
            n += 1
            snap = g.step(20.0, 0.9, dt)
            if snap["reason"] == "reset_new_lead":
                break
        span = n * dt
        check(f"[{label}] resets on a run, never one frame", n >= MIN_CONSEC_REJECTS,
              f"n={n}")
        check(f"[{label}] run lasts ~{REJECT_WINDOW_S}s regardless of cadence",
              abs(span - REJECT_WINDOW_S) <= REJECT_WINDOW_TOL_S + dt,
              f"n={n} frames, span={span:.3f}s")
        check(f"[{label}] NEW_LEAD flagged", bool(snap and snap["new_lead"]))
        check(f"[{label}] re-seeds at the new range", abs(g.d - 20.0) < 0.01,
              f"d={g.d:.2f}")

    # The floor is load-bearing, not decorative: at 2 fps ONE rejected frame
    # already spans 0.5 s and would clear the time term on its own.
    h = HeadwayFilter()
    for _ in range(30):
        h.step(50.0, 0.9, 0.5)
    check("a single outlier never resets the filter, even at 2 fps",
          h.step(20.0, 0.9, 0.5)["reason"] == "gated")

    sm = settled(band=UNSAFE)
    m = measure(tau=1.2, trend=SLOWLY_SHRINKING, new_lead=True)
    r = sm.tick(m, t=10.0, dt=DT)
    check("voice suppressed on the reset frame", not r["voice_fired"], str(r["voice"]))
    check("hold window opened", r["new_lead_hold_frames"] > 0)

    # Warm-up: coaching stays silent for COACH_WARMUP_S after a reset.
    sm = WarningStateMachine(rate_hz=12.0)
    sm.tick(measure(tau=1.2, trend=SLOWLY_SHRINKING, new_lead=True), t=0.0, dt=DT)
    sm.tau_band = UNSAFE
    early = run(sm, measure(tau=1.2, trend=SLOWLY_SHRINKING), 5, t0=DT)
    check("no coaching inside the warm-up window",
          not any(r["voice_fired"] for r in early),
          f"{S.COACH_WARMUP_S}s after init the velocity estimate is still settling")
    late = run(sm, measure(tau=1.2, trend=SLOWLY_SHRINKING), 12,
               t0=S.COACH_WARMUP_S + 1.0)
    check("coaching resumes after the warm-up", any(r["voice_fired"] for r in late))

    # ... but URGENT is deliberately not warm-up gated.
    sm = WarningStateMachine(rate_hz=12.0)
    sm.tick(measure(new_lead=True), t=0.0, dt=DT)
    seq = run(sm, measure(tau=0.8, ttc=1.5, trend=RAPIDLY_SHRINKING), 8, t0=DT)
    check("URGENT is NOT blocked by the coaching warm-up",
          any(r["urgent"] for r in seq),
          "an emergency during warm-up is still an emergency")


def test_purity():
    print("\n§9 -- LLM firewall")
    import re
    src = open(__file__.replace("selftest.py", "state.py")).read()

    # Match real import statements only. A bare substring scan false-positives:
    # "vision" is inside "provisional", which this module says a lot.
    imported = set()
    for line in src.splitlines():
        mm = re.match(r"\s*(?:from\s+([\w.]+)\s+import|import\s+([\w.]+))", line)
        if mm:
            imported.add((mm.group(1) or mm.group(2)).split(".")[0])

    for forbidden in ("transformers", "torch", "cv2", "requests", "numpy",
                      "vision", "llm_interface", "app", "anchor", "depth", "tracker"):
        check(f"state.py does not import {forbidden!r}", forbidden not in imported)
    check("state.py imports only the stdlib",
          imported <= {"math", "dataclasses"}, f"imports: {sorted(imported)}")
    check("state.py performs no file I/O", "open(" not in src)


# ===========================================================================
# Lane geometry (UFLDv2): the corridor-source switch and the drift detector.
# Both are pure functions of a lane result, so they are tested here with
# synthetic lane polylines rather than a GPU and a video. The model itself is
# checked in live_selftest.py, which is allowed to need weights.
# ===========================================================================
def _lane(x_top, x_bottom, h=720, n=20):
    """A straight lane boundary from x_top at 0.42h down to x_bottom at h."""
    return [(x_top + (x_bottom - x_top) * i / (n - 1),
             0.42 * h + (h - 0.42 * h) * i / (n - 1)) for i in range(n)]


def _lane_result(conf=0.9, w=1280, h=720, left=(560, 300), right=(720, 980)):
    """A well-formed detect_lanes() payload with an ego pair.

    `confidence` comes back out of select_ego_pair rather than being echoed
    from `conf`, so the geometry factor is part of what these tests exercise --
    an implausibly narrow pair has to be able to score itself down to zero.
    """
    from . import lanes as L
    lanes_ = [_lane(*left, h=h), _lane(*right, h=h)]
    ego, pair_conf = L.select_ego_pair(
        [{"points": lanes_[0], "confidence": conf, "lane_index": 1},
         {"points": lanes_[1], "confidence": conf, "lane_index": 2}], w, h)
    return {"lanes": lanes_, "ego_left": ego["ego_left"],
            "ego_right": ego["ego_right"], "confidence": pair_conf, "ego": ego,
            "image": {"w": w, "h": h}}


def test_corridor_source():
    from . import anchor as A
    from . import lanes as L

    base = A.EgoCorridor(1280, 720)
    check("static trapezoid self-identifies", base.source == "static")

    # -- the switch ---------------------------------------------------------
    c, i = A.build_corridor(base, _lane_result(conf=0.90))
    check("high-confidence lanes -> UFLD corridor",
          i["corridor_source"] == "ufld" and c.source == "ufld", str(i))
    check("UFLD corridor is a LaneCorridor", isinstance(c, A.LaneCorridor))

    c, i = A.build_corridor(base, _lane_result(conf=0.20))
    check("low-confidence lanes -> static fallback",
          i["corridor_source"] == "static" and c is base
          and i["fallback_reason"] == "low_confidence", str(i))

    c, i = A.build_corridor(base, None)
    check("no lane result -> static fallback",
          c is base and i["fallback_reason"] == "no_lane_result", str(i))

    lr = _lane_result(conf=0.9)
    lr["ego_left"] = None
    c, i = A.build_corridor(base, lr)
    check("half an ego pair -> static fallback, never half a corridor",
          c is base and i["corridor_source"] == "static", str(i))

    # The threshold is a boundary, so pin both sides of it.
    eps = 1e-6
    c_at, _ = A.build_corridor(base, _lane_result(conf=L.LANE_CONF_MIN))
    c_below, _ = A.build_corridor(base, _lane_result(conf=L.LANE_CONF_MIN - 0.02))
    check("threshold is inclusive at LANE_CONF_MIN, exclusive below",
          c_at.source == "ufld" and c_below.source == "static")

    # -- both lanes on one side of centre is not an ego pair ----------------
    c, i = A.build_corridor(base, _lane_result(conf=0.9, left=(200, 100),
                                               right=(300, 250)))
    check("lanes that do not bracket centre -> static fallback",
          c is base and i["fallback_reason"] == "no_bracketing_pair", str(i))

    # -- an implausibly narrow pair scores zero and therefore falls back ----
    c, i = A.build_corridor(base, _lane_result(conf=0.9, left=(630, 620),
                                               right=(650, 660)))
    check("implausible lane width -> static fallback",
          c is base, str(i))

    # -- containment --------------------------------------------------------
    corr, _ = A.build_corridor(base, _lane_result(conf=0.9))
    # At row 700 the synthetic lane runs from x=286 to x=966.
    inside, geo = corr.contains(640.0, 700.0)
    check("lead between the lines is inside the lane corridor",
          inside and geo["corridor_source"] == "ufld", str(geo))
    outside, geo = corr.contains(1150.0, 700.0)
    check("lead beyond the right line is outside",
          not outside and geo["reason"] == "outside_lane", str(geo))

    # The margin is real and finite: just past the paint is still in, far past
    # it is not.
    xl, xr = corr.bounds_at_row(700.0)
    margin = A.LANE_MARGIN_FRAC * (xr - xl)
    just_in, _ = corr.contains(xr + margin * 0.5, 700.0)
    just_out, _ = corr.contains(xr + margin * 1.5, 700.0)
    check("corridor margin admits a shade past the line but not a lane's worth",
          just_in and not just_out)

    # -- the range gate survives the switch ---------------------------------
    _, geo = corr.contains(640.0, 700.0)
    near = corr.contains(640.0, 719.0)[1]
    check("UFLD corridor still reports a forward range",
          "forward_m" in geo and geo["forward_m"] > 0, str(geo))
    above, geo = corr.contains(640.0, 10.0)
    check("above the horizon is still rejected under UFLD",
          not above, str(geo))

    # -- rows the paint does not reach fall back per-pixel, not per-frame ---
    short = _lane_result(conf=0.9)
    short["ego_left"] = short["ego_left"][:4]     # only the far end survives
    short["ego_right"] = short["ego_right"][:4]
    c2, _ = A.build_corridor(base, short)
    _, geo = c2.contains(640.0, 715.0)
    check("row below the detected paint falls back to the trapezoid",
          geo["corridor_source"] == "ufld_row_fallback", str(geo))

    # -- polygon ------------------------------------------------------------
    poly = corr.polygon()
    check("lane corridor draws a closed-able polygon", len(poly) >= 6)
    check("polygon runs near->far down one side and back",
          poly[0][1] > poly[len(poly) // 2 - 1][1], str(poly[:2]))


def test_lane_drift():
    from .lanes import LaneDriftMonitor as M
    W, H = 1280, 720

    # Offset sign convention: image centre right of the lane centre means the
    # car is toward the RIGHT boundary.
    left_of_centre = {"ego": {"x_bottom_left": 300.0, "x_bottom_right": 800.0}}
    check("offset is +ve when the car sits right of the lane centre",
          M.offset(left_of_centre, W, H) > 0)
    centred = {"ego": {"x_bottom_left": 340.0, "x_bottom_right": 940.0}}
    check("offset is ~0 dead centre", abs(M.offset(centred, W, H)) < 1e-6)
    check("no ego pair -> no offset", M.offset({"ego": {}}, W, H) is None)

    def at(offset_ratio):
        """A lane result placing the car at a given fraction toward a boundary."""
        half = 300.0
        centre = W * 0.5 - offset_ratio * half
        return {"ego": {"x_bottom_left": centre - half,
                        "x_bottom_right": centre + half}}

    # -- must hold, not merely touch ---------------------------------------
    m = M()
    r = m.update(at(0.9), W, H, 0.0, 0.9)
    check("a single frame past the threshold does not fire", not r["drift"])
    r = m.update(at(0.9), W, H, 0.5, 0.9)
    check("half the hold time does not fire", not r["drift"] and r["reason"] == "holding")
    r = m.update(at(0.9), W, H, 1.05, 0.9)
    check("held past DRIFT_HOLD_S fires once", r["drift"] and r["side"] == "right",
          str(r))
    r = m.update(at(0.9), W, H, 2.0, 0.9)
    check("a held drift does not fire again while held",
          not r["drift"] and r["reason"] == "already_reported")

    # -- re-arm requires actually recentring --------------------------------
    m.update(at(0.6), W, H, 3.0, 0.9)                # inside rearm ratio? no (0.6 > 0.5)
    r = m.update(at(0.9), W, H, 5.0, 0.9)
    check("drifting back out without recentring does not re-fire", not r["drift"])
    m.update(at(0.1), W, H, 6.0, 0.9)                # recentred -> re-armed
    m.update(at(0.9), W, H, 7.0, 0.9)
    r = m.update(at(0.9), W, H, 8.1, 0.9)
    check("a second excursion after recentring fires", r["drift"], str(r))
    check("events are counted", m.n_events == 2)

    # -- staying inside the lane never fires --------------------------------
    m2 = M()
    fired = [m2.update(at(0.5), W, H, i * 0.25, 0.9)["drift"] for i in range(40)]
    check("10 s at 50% offset never fires", not any(fired))

    # -- low lane confidence cannot sustain an excursion --------------------
    m3 = M()
    m3.update(at(0.9), W, H, 0.0, 0.9)
    r = m3.update(at(0.9), W, H, 0.5, 0.2)           # paint lost mid-excursion
    check("low lane confidence breaks the hold", not r["drift"]
          and r["reason"] == "no_lane_confidence", str(r))
    r = m3.update(at(0.9), W, H, 1.1, 0.9)
    check("the timer restarts after a confidence gap, not resumes",
          not r["drift"], str(r))

    # -- crossing from one side to the other restarts the timer -------------
    m4 = M()
    m4.update(at(0.9), W, H, 0.0, 0.9)
    r = m4.update(at(-0.9), W, H, 0.6, 0.9)
    check("swapping sides restarts the hold", not r["drift"], str(r))
    r = m4.update(at(-0.9), W, H, 1.7, 0.9)
    check("the new side then fires on its own hold",
          r["drift"] and r["side"] == "left", str(r))

    # -- reset clears everything -------------------------------------------
    m4.reset()
    r = m4.update(at(-0.9), W, H, 2.0, 0.9)
    check("reset drops an excursion in progress", not r["drift"])

    # -- camera mount bias --------------------------------------------------
    biased = M(center_bias=-0.30)
    r = biased.update(at(-0.30), W, H, 0.0, 0.9)
    check("center_bias zeroes out a fixed mounting offset",
          abs(r["offset"]) < 1e-6, str(r))

    # -- incoherent thresholds are refused, not silently obeyed --------------
    # Found the hard way: with rearm_ratio (0.50) above a lowered ratio (0.35),
    # every frame of a single excursion re-armed and re-fired -- one drift
    # produced 16 records in a 10 s clip. This is the exact edit someone makes
    # when tuning DRIFT_RATIO down after a real drive.
    from .lanes import DRIFT_RATIO, DRIFT_REARM_RATIO
    check("shipped defaults are coherent", DRIFT_REARM_RATIO < DRIFT_RATIO,
          f"rearm {DRIFT_REARM_RATIO} < ratio {DRIFT_RATIO}")
    for bad in (dict(ratio=0.35), dict(ratio=0.5, rearm_ratio=0.5),
                dict(ratio=0.5, rearm_ratio=-0.1), dict(hold_s=0.0)):
        try:
            M(**bad)
            ok = False
        except ValueError:
            ok = True
        check(f"a monitor built with {bad} is refused", ok)

    # ...and a coherently-lowered threshold still fires exactly once.
    m5 = M(ratio=0.35, rearm_ratio=0.20)
    fires = [m5.update(at(0.42), W, H, i * 0.25, 0.9)["drift"] for i in range(24)]
    check("a lowered-but-coherent threshold reports one excursion once",
          sum(fires) == 1, f"{sum(fires)} events in {len(fires)} frames")


def test_anchor_enumerates_corridor_selects():
    """Qwen enumerates; the corridor alone decides which vehicle is the lead."""
    from . import anchor as A

    # -- the prompt must not ask the model to choose -----------------------
    # It IS allowed to scope what gets listed (our lane plus the ones either
    # side is all any downstream rule reads, and enumerating the whole horizon
    # cost seconds of decode). What it must never do is name the lead.
    p = A.ANCHOR_PROMPT.lower()
    check("prompt asks for a list, not a pick",
          "list the vehicles" in p and "at most" in p)
    check("prompt covers the adjacent lanes merge detection needs",
          "immediately left or right" in p)
    check("prompt explicitly declines to have the model pick the lead",
          "do not decide which vehicle i am following" in p)
    for banned in ("the single vehicle directly ahead",
                   "the lane i am driving in",
                   "ignore vehicles in other lanes"):
        check(f"prompt no longer says {banned!r}", banned not in p,
              "that phrasing hands the lead selection back to Qwen")

    # -- parsing the enumerated reply --------------------------------------
    W, H = 1280, 720
    reply = ('{"o": [["car", 100, 300, 200, 400], ["truck", 480, 250, 530, 330],'
             ' ["motorcycle", 700, 320, 740, 400]]}')
    got = A._parse_vehicles(reply, W, H)
    check("enumerated reply yields every vehicle with its label",
          len(got) == 3 and [g[0] for g in got] == ["car", "truck", "motorcycle"],
          str(got))
    # 0-1000 normalised, per this checkpoint: x=480/1000*1280 = 614.4
    check("coordinates go through _rescale",
          abs(got[1][1][0] - 614.4) < 0.1, str(got[1][1]))

    check("fenced JSON is accepted",
          len(A._parse_vehicles('```json\n{"o": [["car",1,2,300,400]]}\n```', W, H)) == 1)
    check("empty list parses as no vehicles",
          A._parse_vehicles('{"o": []}', W, H) == [])
    check("the old single-box reply still parses (one unlabelled candidate)",
          [lab for lab, _ in A._parse_vehicles(
              '{"bbox_2d": [100, 300, 200, 400]}', W, H)] == [None])
    check("verbose per-object form is accepted",
          len(A._parse_vehicles(
              '{"o": [{"label": "bus", "bbox_2d": [10,20,300,400]}]}', W, H)) == 1)

    # -- truncation, and the unlabelled-entry trap it used to spring ---------
    # Verbatim from the live server before repair existed: the token cap cut
    # the list mid-array, JSON parse failed, and the bare-coordinate fallback
    # returned six UNLABELLED boxes -- one of which was then selected as the
    # lead. An unknown object with a headway computed to it.
    truncated = ('{"o": [["car", 785, 562, 953, 703], [153, 588, 188, 628], '
                 '[323, 593, 341, 612], [346, 593, 357, 607], [515, 592, 527, '
                 '607], [480, 592, 490, 603], [5')
    got = A._parse_vehicles(truncated, W, H)
    check("a truncated reply is repaired instead of falling back to regex",
          [lab for lab, _ in got] == ["car"],
          f"got {[lab for lab, _ in got]}")
    check("the repaired box keeps its real coordinates",
          abs(got[0][1][0] - 1004.8) < 0.5, str(got[0][1]))
    # The budget is deliberately BELOW a full reply. Decode runs ~52 tokens/s,
    # so 192 tokens measured 3.4 s of frame time on a busy motorway; the reply
    # is closest-first and repair keeps what arrived, so the budget buys
    # latency at the cost of the far tail, which changes no decision.
    check("token budget is bounded by the frame budget, not by reply length",
          A.ANCHOR_MAX_NEW_TOKENS <= 96, f"{A.ANCHOR_MAX_NEW_TOKENS} tokens")
    check("the anchor uses that budget",
          A.LeadAnchor(W, H, use_qwen=False).max_new_tokens
          == A.ANCHOR_MAX_NEW_TOKENS)
    check("truncation is therefore expected, and survivable",
          A.repair_truncated_json('{"o": [["car",1,2,300,400], [5') is not None,
          "a capped budget is only safe because repair exists")

    mixed = '{"o": [["car", 100, 300, 200, 400], [300, 300, 400, 400]]}'
    check("an unlabelled entry beside labelled ones is dropped, not guessed",
          [lab for lab, _ in A._parse_vehicles(mixed, W, H)] == ["car"],
          "the label-repeat bug makes its class unknown")
    check("an all-unlabelled reply is still usable (legacy shape)",
          len(A._parse_vehicles('{"o": [[100,300,200,400],[300,300,400,400]]}',
                                W, H)) == 2)

    # -- selection is the corridor's, not the model's ----------------------
    class _Anchor(A.LeadAnchor):
        """LeadAnchor with the Qwen call replaced by a canned reply."""
        def __init__(self, reply, **kw):
            super().__init__(W, H, **kw)
            self._reply = reply
        def _query_qwen(self, frame):
            return self._reply

    base = A.EgoCorridor(W, H)
    lane, _ = A.build_corridor(base, _lane_result(conf=0.9))

    # Coordinates below are 0-1000 NORMALISED, which is what this checkpoint
    # emits -- writing them as pixels is the mistake _rescale exists to absorb,
    # and a test that wrote them as pixels would be testing a reply the model
    # never sends. Helper keeps the intent readable in pixel terms.
    def veh(label, u_px, v_px, w_px=80, h_px=110):
        """A box whose bottom-centre is (u_px, v_px), expressed normalised."""
        x1, x2 = (u_px - w_px / 2) / W * 1000, (u_px + w_px / 2) / W * 1000
        y1, y2 = (v_px - h_px) / H * 1000, v_px / H * 1000
        return f'["{label}", {x1:.0f}, {y1:.0f}, {x2:.0f}, {y2:.0f}]'

    # At row 700 the synthetic ego lane runs x=312..967 (+112 px margin), so
    # u=1150 is well outside and u=640 is dead centre. Qwen lists the
    # adjacent-lane car FIRST (closest/largest, as the prompt asks) and the
    # true in-lane lead second.
    reply = '{"o": [' + veh("car", 1150, 700) + ', ' + veh("car", 640, 700) + ']}'
    box, info = _Anchor(reply, corridor=lane).anchor(None, 0)
    check("the adjacent-lane vehicle Qwen listed first is rejected",
          info["reason"] == "ok" and info["n_rejected"] == 1
          and info["n_in_corridor"] == 1, str(info))
    check("the in-lane vehicle is selected even though it was listed second",
          box is not None and abs((box[0] + box[2]) / 2 - 640) < 2
          and abs(box[3] - 700) < 2, str(box))
    check("the selected candidate's label is reported", info.get("label") == "car")

    # Nearest in-corridor wins among several in-lane candidates. Lower in the
    # image (larger v) is nearer.
    reply2 = ('{"o": [' + veh("car", 640, 612) + ', '
              + veh("truck", 640, 700) + ']}')
    box, info = _Anchor(reply2, corridor=lane).anchor(None, 0)
    check("among in-lane candidates the NEAREST is chosen",
          info["n_in_corridor"] == 2 and abs(box[3] - 700) < 2
          and info["label"] == "truck", f"{info} {box}")

    # Non-vehicles never become a lead, however Qwen labels them.
    box, info = _Anchor('{"o": [' + veh("pedestrian", 640, 700) + ']}',
                        corridor=lane).anchor(None, 0)
    check("a pedestrian in the corridor is never the lead",
          box is None and info["reason"] == "no_vehicle_labels", str(info))

    # Nothing in the lane -> no anchor, and the rejections are logged.
    box, info = _Anchor('{"o": [' + veh("car", 1150, 700) + ']}',
                        corridor=lane).anchor(None, 0)
    check("no in-lane vehicle means no lead, with reasons recorded",
          box is None and info["reason"] == "all_boxes_outside_corridor"
          and len(info["rejected"]) == 1, str(info))

    # The point of the whole change: the SAME reply resolves to a different
    # lead depending only on the corridor. At this row the lane admits
    # x=200..1079 and the trapezoid a far wider x=26..1254, so a car at u=1150
    # is out of lane but inside the trapezoid -- and being nearer, it is what
    # the trapezoid would hand the tracker.
    both = '{"o": [' + veh("car", 1150, 700) + ', ' + veh("car", 640, 640) + ']}'
    box_lane, info_lane = _Anchor(both, corridor=lane).anchor(None, 0)
    box_trap, info_trap = _Anchor(both, corridor=base).anchor(None, 0)
    check("geometry, not the model, decides which vehicle is the lead",
          info_lane["n_in_corridor"] == 1 and info_trap["n_in_corridor"] == 2
          and abs((box_lane[0] + box_lane[2]) / 2 - 640) < 2
          and abs((box_trap[0] + box_trap[2]) / 2 - 1150) < 2,
          f"lane picked u={(box_lane[0] + box_lane[2]) / 2:.0f}, "
          f"trapezoid picked u={(box_trap[0] + box_trap[2]) / 2:.0f}")
    box, info = _Anchor('{"o": []}', corridor=lane).anchor(None, 0)
    check("an empty list is not an anchor", box is None
          and info["reason"] == "no_box_parsed")

    # The corridor that did the selecting is recorded on every anchor.
    _, info = _Anchor(reply, corridor=lane).anchor(None, 0)
    check("anchor records which corridor selected the lead",
          info["corridor_source"] == "ufld", str(info.get("corridor_source")))
    _, info = _Anchor(reply, corridor=base).anchor(None, 0)
    check("...and says so when it was the trapezoid",
          info["corridor_source"] == "static")


def test_lead_corridor_check():
    """Only a lane corridor may invalidate a lead the tracker is holding."""
    from . import anchor as A
    from .live import lead_corridor_check, LEAD_OUT_OF_CORRIDOR_FRAMES

    base = A.EgoCorridor(1280, 720)
    lane, _ = A.build_corridor(base, _lane_result(conf=0.9))
    in_box = (600.0, 500.0, 680.0, 700.0)      # bottom-centre 640,700: in lane
    out_box = (1100.0, 500.0, 1200.0, 700.0)   # bottom-centre 1150,700: not

    n, geo = lead_corridor_check(lane, in_box, 5)
    check("a lead back inside the lane clears the miss count", n == 0, str(geo))
    n, _ = lead_corridor_check(lane, out_box, 0)
    check("a lead outside the lane starts counting", n == 1)
    n, _ = lead_corridor_check(lane, out_box, 2)
    check("consecutive misses accumulate", n == 3)
    check("three misses is what triggers a re-anchor",
          LEAD_OUT_OF_CORRIDOR_FRAMES == 3)

    # The whole point: the trapezoid does not get a vote on an existing track.
    n, geo = lead_corridor_check(base, out_box, 9)
    check("the static trapezoid never invalidates an established lead",
          n == 0, f"count={n} geo={geo}")
    n, _ = lead_corridor_check(base, in_box, 9)
    check("...whether the trapezoid agrees or not", n == 0)

    n, geo = lead_corridor_check(lane, None, 4)
    check("no lead means no miss count and no geometry",
          n == 0 and geo is None)


def _box(u, v, w=80, h=110):
    """Box with bottom-centre at (u, v)."""
    return (u - w / 2.0, v - h, u + w / 2.0, v)


def test_membership_overlap():
    """Membership is an overlap fraction of the bottom edge, not a centre point."""
    from . import anchor as A
    from . import membership as M

    base = A.EgoCorridor(1280, 720)
    lane, _ = A.build_corridor(base, _lane_result(conf=0.9))
    # Lane at row 700 runs x = 312.6 .. 967.4 (654.8 px wide), no margin.
    xl, xr = lane.bounds_at_row(700.0)
    check("lane bounds at a row come back without the corridor margin",
          abs(xl - 312.6) < 1 and abs(xr - 967.4) < 1, f"{xl:.1f}..{xr:.1f}")

    frac, info = M.bottom_edge_overlap(_box(640, 700, w=100), lane)
    check("a centred vehicle is fully in lane", abs(frac - 1.0) < 1e-6, str(info))

    # THE WIDE-TRUCK CASE. A 400 px truck straddling the right line: its
    # bottom-CENTRE at 967 is right on the boundary, but 50% of its body is in
    # our lane. The centre-point test this replaces would call it a coin flip.
    truck = _box(967, 700, w=400)
    frac, _ = M.bottom_edge_overlap(truck, lane)
    check("a wide truck straddling the line reads ~50% in lane",
          abs(frac - 0.5) < 0.02, f"{frac:.3f}")
    inside_centre, _ = lane.contains(967.0, 700.0)
    check("...and the old centre-point test is the one that is ambiguous here",
          inside_centre, "centre sits exactly on the boundary")

    # THE MOTORCYCLE CASE. Narrow, hugging the inside of the right line: fully
    # in lane by overlap.
    moto = _box(940, 700, w=40)
    frac, _ = M.bottom_edge_overlap(moto, lane)
    check("a motorcycle hugging the lane edge is fully in lane",
          abs(frac - 1.0) < 1e-6, f"{frac:.3f}")

    # Wholly outside.
    frac, _ = M.bottom_edge_overlap(_box(1150, 700, w=100), lane)
    check("an adjacent-lane vehicle has zero overlap", frac == 0.0)

    # Rows the paint does not reach are UNKNOWN, not "outside".
    frac, info = M.bottom_edge_overlap(_box(640, 20, w=80), lane)
    check("a row with no lane is reported as unknown, not as zero overlap",
          info["reason"] == "no_lane_at_row", str(info))

    # The trapezoid answers the same question, so the fallback path is the
    # same code with worse geometry rather than a different rule.
    frac_static, _ = M.bottom_edge_overlap(_box(640, 700, w=100), base)
    check("the static trapezoid supports the same overlap test",
          frac_static > 0.9, f"{frac_static:.3f}")


def test_membership_hysteresis_and_dwell():
    from . import membership as M

    c = M.Candidate(1, _box(640, 700), "car", 0.0)
    check("a candidate starts outside membership", not c.member)

    c.update_overlap(0.30, "ok", 0.0)
    check("30% does not enter membership (needs 40%)", not c.member)
    c.update_overlap(0.45, "ok", 0.25)
    check("45% enters membership", c.member and c.member_since == 0.25)

    # Hysteresis: between the two thresholds, membership is KEPT.
    c.update_overlap(0.30, "ok", 0.5)
    check("30% keeps membership once held (hysteresis band)", c.member)
    check("...and does not restart the dwell clock", c.member_since == 0.25)
    c.update_overlap(0.20, "ok", 0.75)
    check("below 25% leaves membership", not c.member and c.member_since is None)

    # Dwell.
    d = M.Candidate(2, _box(640, 700), "car", 0.0)
    d.update_overlap(0.9, "ok", 0.0)
    check("membership alone is not eligibility", not d.eligible(0.0))
    d.update_overlap(0.9, "ok", 0.25)
    check("0.25 s of membership is not enough", not d.eligible(0.25))
    d.update_overlap(0.9, "ok", 0.5)
    check("0.5 s of continuous membership is eligible", d.eligible(0.5))
    check("MEMBER_HOLD_S is the ~0.5 s the spec asked for",
          abs(M.MEMBER_HOLD_S - 0.5) < 1e-9)

    # A break resets the clock -- this is what stops oncoming traffic swinging
    # through the polygon on a bend from taking the lock.
    d.update_overlap(0.1, "ok", 0.75)
    d.update_overlap(0.9, "ok", 1.0)
    check("membership lost and regained restarts the dwell",
          not d.eligible(1.0) and d.member_since == 1.0)

    # An unknown row neither grants nor revokes membership.
    e = M.Candidate(3, _box(640, 700), "car", 0.0)
    e.update_overlap(0.9, "ok", 0.0)
    e.update_overlap(0.0, "no_lane_at_row", 0.25)
    check("an unreadable row does not revoke membership", e.member)
    check("...and does not enter the trend history", len(e.history) == 1)


def test_merge_promotion():
    from . import membership as M

    def rising(c, t0, start, step, n, rng=25.0, dt=0.25):
        out = []
        for i in range(n):
            c.depths.append(rng)
            c.update_overlap(start + step * i, "ok", t0 + i * dt)
            out.append(c.check_merge(t0 + i * dt, True))
        return [e for e in out if e]

    # A car easing in: overlap climbing ~0.4/s, 25 m ahead.
    c = M.Candidate(1, _box(1000, 700), "car", 0.0)
    events = rising(c, 0.0, 0.05, 0.10, 6)
    check("a steadily merging vehicle is promoted", len(events) == 1, str(events))
    ev = events[0]
    check("the promotion window is at least MERGE_WINDOW_S",
          ev["window_s"] >= M.MERGE_WINDOW_S - 1e-9, str(ev))
    check("the promotion records the slope that justified it",
          ev["slope_per_s"] >= M.MERGE_MIN_SLOPE, str(ev))
    check("promotion fires once, not once per frame",
          not rising(c, 2.0, 0.6, 0.10, 4))

    # The whole point of promotion: eligible while still BELOW the membership
    # threshold, so the lock is available before the merge completes. Overlap
    # climbs 0.05 -> 0.33, never reaching MEMBER_ENTER_FRAC.
    early = M.Candidate(8, _box(1000, 700), "car", 0.0)
    ev = rising(early, 0.0, 0.05, 0.07, 5)
    check("a merging vehicle is promoted before it qualifies as a member",
          len(ev) == 1 and not early.member and early.overlap < M.MEMBER_ENTER_FRAC,
          f"overlap {early.overlap:.2f}, member={early.member}")
    check("...and that promotion alone makes it lead-eligible",
          early.eligible(1.0) and early.member_since is None)

    # Steady adjacent traffic: overlap flat. Must never promote.
    flat = M.Candidate(2, _box(1000, 700), "car", 0.0)
    check("a vehicle holding its lane is never promoted",
          not rising(flat, 0.0, 0.30, 0.0, 8))
    # Drifting away.
    away = M.Candidate(3, _box(1000, 700), "car", 0.0)
    check("a vehicle drifting away is never promoted",
          not rising(away, 0.0, 0.35, -0.05, 8))

    # Too far to matter.
    far = M.Candidate(4, _box(1000, 700), "car", 0.0)
    check("a merge beyond MERGE_MAX_RANGE_M is not promoted",
          not rising(far, 0.0, 0.05, 0.10, 6, rng=M.MERGE_MAX_RANGE_M + 5))

    # Too short a window: three frames at 4 fps is 0.5 s, under the 0.75 s bar.
    quick = M.Candidate(5, _box(1000, 700), "car", 0.0)
    check("a rise seen for under MERGE_WINDOW_S is not yet a merge",
          not rising(quick, 0.0, 0.05, 0.15, 3))

    # Not actually touching our lane yet.
    edge = M.Candidate(6, _box(1000, 700), "car", 0.0)
    check("a rise that has not reached MERGE_MIN_OVERLAP is not promoted",
          not rising(edge, 0.0, 0.0, 0.02, 8))

    # And the trapezoid may not declare one.
    trap = M.Candidate(7, _box(1000, 700), "car", 0.0)
    fired = []
    for i in range(6):
        trap.depths.append(25.0)
        trap.update_overlap(0.05 + 0.10 * i, "ok", i * 0.25)
        fired.append(trap.check_merge(i * 0.25, False))
    check("no merge may be declared off the static trapezoid",
          not any(fired))


def test_lead_selection():
    """Nearest eligible candidate wins; a cut-in is a NEW_LEAD."""
    from . import membership as M

    cs = M.CandidateSet(tracker_factory=None)
    cs._factory = lambda: None      # no frames in this test; boxes set directly

    def add(cid, u, rng, overlap=0.9, t=0.0):
        c = M.Candidate(cid, _box(u, 700), "car", t)
        cs.candidates[cid] = c
        for k in range(3):
            c.depths.append(rng)
            c.update_overlap(overlap, "ok", t + k * 0.25)
        return c

    far = add(1, 640, 40.0)
    lead, info = cs.select_lead(0.5)
    check("the only eligible candidate takes the lock",
          lead is far and info["lead_changed"], str(info))

    # A car cuts in closer.
    near = add(2, 640, 18.0, t=0.5)
    lead, info = cs.select_lead(1.0)
    check("a nearer in-lane vehicle becomes the new lead",
          lead is near and info["lead_changed"]
          and info["prev_lead_id"] == 1, str(info))
    lead, info = cs.select_lead(1.25)
    check("...and holding it is not reported as another change",
          lead is near and not info["lead_changed"])

    # An ineligible (too-brief) candidate must not steal the lock however close.
    closest = M.Candidate(3, _box(640, 700), "car", 1.25)
    cs.candidates[3] = closest
    closest.depths.append(5.0)
    closest.update_overlap(0.95, "ok", 1.25)
    lead, info = cs.select_lead(1.5)
    check("a candidate without its dwell cannot take the lock, however close",
          lead is near, str(info))

    # ...but it does once its own dwell has elapsed. That IS the cut-in rule.
    lead, info = cs.select_lead(1.8)
    check("once the cut-in has served its dwell it takes the lock",
          lead is closest and info["lead_changed"], str(info))

    # A candidate with no depth cannot be ranked, so it cannot be the lead --
    # however close it would be. Fresh set, so no other candidate competes.
    cs3 = M.CandidateSet()
    cs3._factory = lambda: None
    blind = M.Candidate(1, _box(640, 700), "car", 0.0)
    cs3.candidates[1] = blind
    for k in range(3):
        blind.update_overlap(0.95, "ok", k * 0.25)      # no depths appended
    lead, info = cs3.select_lead(1.0)
    check("a candidate with no range is not selectable",
          lead is None and info["reason"] == "no_eligible_candidate", str(info))

    # Non-vehicles never enter the set at all.
    cs2 = M.CandidateSet()
    cs2._factory = lambda: None
    cs2.observe([("pedestrian", _box(640, 700)), ("car", _box(640, 700))], 0.0)
    check("only vehicle labels become candidates",
          len(cs2.candidates) == 1
          and next(iter(cs2.candidates.values())).label == "car")

    # A candidate no enumeration has re-confirmed is retired, even though its
    # tracker is still cheerfully reporting success. MOSSE never admits defeat,
    # so a liveness rule based on the tracker retires nothing -- measured at 21
    # live candidates on a clip whose prompt asks for four.
    cs4 = M.CandidateSet()
    cs4._factory = lambda: None
    cs4.observe([("car", _box(640, 700))], 0.0)
    for k in range(1, 40):                      # 10 s of "successful" tracking
        t = k * 0.25
        for c in cs4.candidates.values():
            c.last_seen = t                     # what track() would set
        cs4._evict(t)
    check("a candidate no enumeration re-confirms is evicted",
          not cs4.candidates,
          "MOSSE reporting success is not evidence the vehicle is there")
    check("...and re-detection keeps it alive",
          _still_alive_after_redetection(M))


def _still_alive_after_redetection(M):
    cs = M.CandidateSet()
    cs._factory = lambda: None
    for k in range(0, 40):
        t = k * 0.25
        if k % 8 == 0:                          # an enumeration every 2 s
            cs.observe([("car", _box(640, 700))], t)
        else:
            for c in cs.candidates.values():
                c.last_seen = t
            cs._evict(t)
    return len(cs.candidates) == 1


def test_membership_is_deterministic():
    """No model, no clock, no randomness in the membership decision."""
    import ast
    from pathlib import Path

    path = Path(__file__).with_name("membership.py")
    tree = ast.parse(path.read_text())
    imported = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            imported.update(a.name.split(".")[0] for a in node.names)
        elif isinstance(node, ast.ImportFrom):
            if node.module:
                imported.add(node.module.split(".")[0])
            imported.update(a.name for a in node.names
                            if node.level and not node.module)
    # cv2 only for the MOSSE factory, anchor only for LEAD_LABELS. No model,
    # no clock, no randomness -- membership must be a pure function of boxes,
    # the lane polygon and the timestamps it is handed.
    check("membership.py imports only maths, collections, cv2 and anchor",
          imported <= {"math", "collections", "cv2", "anchor"},
          f"imports: {sorted(imported)}")
    for forbidden in ("time", "random", "torch", "transformers", "numpy"):
        check(f"membership.py does not import {forbidden!r}",
              forbidden not in imported, "membership is deterministic geometry")

    src = _code_only(path).lower()
    for banned in ("qwen", "generate", "prompt", "llm"):
        check(f"membership.py code never references {banned!r}", banned not in src)

    # Same inputs, same outputs, twice.
    from . import membership as M
    from . import anchor as A
    lane, _ = A.build_corridor(A.EgoCorridor(1280, 720), _lane_result(conf=0.9))
    runs = []
    for _ in range(2):
        c = M.Candidate(1, _box(900, 700, w=200), "car", 0.0)
        seq = []
        for i in range(8):
            c.depths.append(20.0)
            frac, _ = M.bottom_edge_overlap(
                _box(900 - i * 20, 700, w=200), lane)
            c.update_overlap(frac, "ok", i * 0.25)
            seq.append((round(frac, 6), c.member, bool(c.check_merge(i * 0.25, True))))
        runs.append(seq)
    check("identical inputs give identical membership decisions",
          runs[0] == runs[1])


def test_lane_confidence_contribution():
    """Lane confidence may add to §8 confidence; it may never subtract."""
    args = (0.8, 0.3, 0.7, 2.0)      # valid, var, track, anchor_age
    baseline = compute_confidence(*args)
    check("default args reproduce the pre-lane confidence exactly",
          compute_confidence(*args, lane_conf=None) == baseline)
    check("a static-corridor frame scores exactly the baseline",
          compute_confidence(*args, lane_conf=0.95,
                             corridor_source="static") == baseline)
    boosted = compute_confidence(*args, lane_conf=0.95, corridor_source="ufld")
    check("a UFLD-corridor frame scores above the baseline", boosted > baseline)
    check("the lane bonus is capped at CONF_LANE_BONUS",
          boosted - baseline <= S.CONF_LANE_BONUS + 1e-9,
          f"{boosted:.4f} vs {baseline:.4f}")
    check("zero lane confidence is worth nothing, not something",
          compute_confidence(*args, lane_conf=0.0,
                             corridor_source="ufld") == baseline)
    # The bonus must not be able to drag a bad frame over the speaking floor.
    bad = (0.15, 0.9, 0.2, 30.0)
    check("the lane bonus cannot lift a DEGRADED frame over CONF_FLOOR",
          compute_confidence(*bad) < S.CONF_FLOOR
          and compute_confidence(*bad, lane_conf=1.0,
                                 corridor_source="ufld") < S.CONF_FLOOR)


def _code_only(path):
    """A module's source with every comment and string literal removed.

    The point is to check what the module *does*, not what it says about
    itself. lanes.py explains at length that it must never speak -- a naive
    grep for "voice" would fail on the very comment that forbids it, and the
    obvious fix (delete the prose) would trade a real explanation for a green
    check. So the prose is tokenised away and the code alone is inspected.
    """
    import io
    import tokenize
    from pathlib import Path

    out = []
    src = Path(path).read_text()
    for tok in tokenize.generate_tokens(io.StringIO(src).readline):
        if tok.type in (tokenize.COMMENT, tokenize.STRING):
            continue
        out.append(tok.string)
    return " ".join(out)


def test_lane_advisory_only():
    """Lane departure logs. It must have no path to the voice channel."""
    from pathlib import Path
    code = _code_only(Path(__file__).with_name("lanes.py")).lower()

    check("lanes.py code imports no policy or voice module",
          "live_policy" not in code and "voice" not in code
          and "line_text" not in code and "line_audio" not in code)
    for banned in ("speak", "urgency", "audio", "clip", "sessions"):
        check(f"lanes.py code never references {banned!r}",
              banned not in code,
              "drift logging must not acquire a voice by accident")

    # The prose that documents the limit must still be there. A future edit
    # that strips the explanation is exactly when someone is most likely to
    # wire this into the voice channel by mistake.
    src = Path(__file__).with_name("lanes.py").read_text()
    check("lanes.py still documents that drift is advisory-only",
          "ADVISORY LOGGING ONLY" in src and "never speaks" in src.lower())

    # And the policy must not be able to see a drift result even if one is
    # handed to it: LivePolicy.tick takes named arguments and none of them are
    # lane-shaped.
    import inspect
    from .live_policy import LivePolicy
    params = set(inspect.signature(LivePolicy.tick).parameters)
    check("LivePolicy.tick accepts no lane/drift argument",
          not any("lane" in p or "drift" in p for p in params), str(sorted(params)))

    # The drift record must never carry a spoken line into the session log.
    import json as _json
    from . import lanes as L
    m = L.LaneDriftMonitor()
    fired = None
    for i in range(12):
        r = m.update({"ego": {"x_bottom_left": 200.0, "x_bottom_right": 700.0}},
                     1280, 720, i * 0.25, 0.9)
        if r["drift"]:
            fired = r
            break
    blob = _json.dumps(fired or {}).lower()
    check("a fired drift event contains no voice field",
          fired is not None and not any(k in blob for k in
                                        ("speak", "voice", "line", "clip")),
          str(fired))

def main():
    print("=" * 70)
    print("headway warning-logic v2 self-test (docs/warning_logic_v2.md)")
    print("=" * 70)
    test_bands()
    test_hysteresis()
    test_maths()
    test_confidence()
    test_confirmation()
    test_urgent_orthogonal()
    test_low_speed()
    test_system_states()
    test_voice_policy()
    test_cooldowns()
    test_cutin_and_warmup()
    test_purity()
    test_corridor_source()
    test_anchor_enumerates_corridor_selects()
    test_lead_corridor_check()
    test_membership_overlap()
    test_membership_hysteresis_and_dwell()
    test_merge_promotion()
    test_lead_selection()
    test_membership_is_deterministic()
    test_lane_drift()
    test_lane_confidence_contribution()
    test_lane_advisory_only()

    passed = sum(1 for ok, _, _ in _results if ok)
    total = len(_results)
    print("\n" + "=" * 70)
    print(f"{passed}/{total} checks passed")
    if passed != total:
        print("\nFAILURES:")
        for ok, name, detail in _results:
            if not ok:
                print(f"  - {name}  {detail}")
    print("=" * 70)
    return 0 if passed == total else 1


if __name__ == "__main__":
    raise SystemExit(main())
