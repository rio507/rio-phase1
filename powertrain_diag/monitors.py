"""monitors.py — the nine engine monitors, and the pure logic that runs them.

Instances of the framework in diag/, not a second one. The status/last_result
contract, the enabling gate, the dataclasses and `run` all come from
diag/monitors.py; what is here is nine evaluation functions that know what a
coolant temperature is.

WHERE THE LIMITS COME FROM
--------------------------
The coolant ceiling, the charging floor and the fuel-trim band are read from
config.TELEMETRY_BANDS — the same numbers the panel judges its rows against.
Not copied: read. A monitor with its own copy would disagree with the row above
it the first time somebody tuned one of them, and an amber coolant row with RIO
saying nothing about it is exactly the failure the one-threshold-in-one-place
convention exists to prevent.

The numbers that ARE in config.POWERTRAIN_* are the ones the panel has no
opinion about: how long a condition must hold, how fast a rise counts, how far
from its own baseline is far.

§5.5, WHICH IS THE ONE RULE THAT CANNOT BEND
--------------------------------------------
    A learned baseline may raise sensitivity. It may never relax a fixed limit.

Two monitors here use this vehicle's learned history — the contextual coolant
monitor and the fuel-trim monitor. Neither can suppress anything. They only ever
ADD a finding that a fixed band would have missed; the hard-limit and charging
monitors run independently and would fire on the same data whatever the baseline
said. powertrain_diag/selftest.py drives a case where the baseline is absurd and
asserts the hard limit still fires.

WARM-UP IS NOT A FAULT
----------------------
Half of these monitors are INHIBITED on a cold engine, and it is not a
technicality. A coolant temperature climbing at 40°F/min is a healthy engine
ninety seconds after a cold start and a serious problem twenty minutes into a
drive. A monitor that could not tell those apart would fire on every single
journey, which is the fastest way to make a driver ignore it.

LLM FIREWALL
------------
Imports config, the stdlib and diag. Nothing here can read a model output.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

import config

from diag.monitors import (DATA_UNAVAILABLE, FAILED_PENDING, INHIBITED,
                           NO_VERDICT, NOT_READY, NOT_SUPPORTED, PASSED, READY,
                           RUNNING, EnablingConditions, HealingCriteria,
                           MonitorDefinition, Outcome)
from diag.monitors import MonitorInput as _BaseInput
from diag.monitors import Sample as _BaseSample
from diag.monitors import confidence as _confidence
from diag.monitors import definitions_view as _definitions_view
from diag.monitors import run

from . import codes as C


# ---------------------------------------------------------------------------
# What an engine monitor is given
# ---------------------------------------------------------------------------

@dataclass
class Sample(_BaseSample):
    """One instant of the whole engine.

    One sample per tick carrying every channel, rather than one sample per
    channel: these monitors are almost all cross-signal, and a coolant reading
    without the road speed that goes with it cannot answer any of the questions
    below. The tire domain is the other way round for the same reason — four
    corners fail independently and an engine does not.
    """
    values: Dict[str, Optional[float]] = field(default_factory=dict)
    statuses: Dict[str, str] = field(default_factory=dict)
    engine_running: bool = False

    def has_primary(self) -> bool:
        """Did anything at all report this tick?"""
        return any(v is not None for v in self.values.values())

    def get(self, channel: str) -> Optional[float]:
        return self.values.get(channel)


@dataclass
class MonitorInput(_BaseInput):
    """Everything an engine monitor may look at."""
    # This vehicle's own learned history, from insights.py's daily baselines.
    # Read by the engine and passed in, so the monitors stay pure.
    baselines: Dict[str, float] = field(default_factory=dict)
    baseline_days: Dict[str, int] = field(default_factory=dict)
    # Recorded cranking events, oldest first: [{"at", "min_v", "session"}].
    start_events: List[dict] = field(default_factory=list)
    # What the DTC layer currently knows. A view, never a decision.
    dtc: Dict[str, object] = field(default_factory=dict)
    # What the transport currently looks like.
    link: Dict[str, object] = field(default_factory=dict)


# ---------------------------------------------------------------------------
# Helpers. Pure, and small enough to reason about.
# ---------------------------------------------------------------------------

# Channels whose absence means the engine cannot be judged at all, as distinct
# from a channel this vehicle simply does not expose.
_CORE = ("rpm", "coolant_temp", "battery_voltage")


def _band(channel: str, edge: str) -> Optional[float]:
    """A limit, read from where the panel reads it. Never copied."""
    return config.TELEMETRY_BANDS.get(channel, {}).get(edge)


def usable(inp: MonitorInput) -> List[Sample]:
    """Every sample a monitor may look at. NOT bounded by sample_max_age_s.

    That bound belongs to the GATE, where it answers "is the latest reading
    fresh enough to judge at all", and it is thirty seconds because an ECU that
    has stopped answering has stopped answering now.

    A monitor's own HISTORY is a different question and several of them need
    minutes of it: the fuel-trim monitor holds for two, the frozen-channel check
    needs ninety seconds of a value not moving, and the rate of rise fits across
    two. Applying the freshness bound here silently truncated all three to the
    last thirty seconds, so their hold windows could never be satisfied and they
    passed forever — the worst possible failure for a monitor, because it looks
    exactly like a healthy engine.

    Windowing is each monitor's own business, via series(window_s=).
    """
    return [s for s in inp.samples
            if s.valid and s.connected and s.at >= inp.epoch_started_at]


def series(inp: MonitorInput, channel: str,
           window_s: float = None) -> List[Tuple[float, float]]:
    """(at, value) for one channel, oldest first, valid samples only."""
    out = []
    for s in usable(inp):
        v = s.values.get(channel)
        if v is None:
            continue
        if window_s is not None and (inp.now - s.at) > window_s:
            continue
        out.append((s.at, float(v)))
    return out


def latest(inp: MonitorInput, channel: str) -> Optional[float]:
    for s in reversed(usable(inp)):
        v = s.values.get(channel)
        if v is not None:
            return float(v)
    return None


def slope_per_min(points: List[Tuple[float, float]]) -> Optional[float]:
    """Least-squares change per minute. None when there is nothing to fit.

    A fit rather than first-to-last, for the reason telemetry.py's trend arrow
    is a fit: one unusual reading at either end of the window would otherwise
    invent a rise or hide one.
    """
    n = len(points)
    if n < 3:
        return None
    mx = sum(p[0] for p in points) / n
    my = sum(p[1] for p in points) / n
    denom = sum((p[0] - mx) ** 2 for p in points)
    if denom <= 1e-9:
        return None
    slope = sum((p[0] - mx) * (p[1] - my) for p in points) / denom
    return slope * 60.0


def sustained_beyond(points: List[Tuple[float, float]], limit: float,
                     above: bool, hold_s: float) -> Optional[float]:
    """How long the channel has been continuously past `limit`, or None.

    Continuously: one reading back inside the limit resets it. A condition that
    flickers either side of a threshold has not been sustained, and treating it
    as though it had is how a monitor turns sensor noise into a fault.
    """
    if not points:
        return None
    start = None
    for at, v in points:
        beyond = (v >= limit) if above else (v <= limit)
        if beyond:
            if start is None:
                start = at
        else:
            start = None
    if start is None:
        return None
    held = points[-1][0] - start
    return held if held >= hold_s else None


def _warm(inp: MonitorInput) -> Optional[bool]:
    coolant = latest(inp, "coolant_temp")
    if coolant is None:
        return None
    return coolant >= config.POWERTRAIN_WARM_COOLANT_F


def _running(inp: MonitorInput) -> bool:
    u = usable(inp)
    return bool(u and u[-1].engine_running)


def _quality(inp: MonitorInput) -> str:
    u = usable(inp)
    if not u:
        return "none"
    recent = u[-8:]
    bad = sum(1 for s in recent if not s.valid)
    if bad == 0:
        return "good"
    return "degraded" if bad < len(recent) / 2 else "poor"


# ---------------------------------------------------------------------------
# The monitors
# ---------------------------------------------------------------------------

def _eval_new_dtc(inp: MonitorInput) -> Outcome:
    """The vehicle's own codes, surfaced as a finding.

    This monitor makes no diagnostic judgement of its own and must not: the ECU
    has already decided, its verdict is the authority, and the Flagged Error
    Codes section is where the detail belongs. What this exists for is to put a
    reported code into the same issue list, with the same lifecycle and the same
    communication ledger, as everything else RIO knows about the car — so that
    "is anything wrong?" has one answer rather than two.
    """
    dtc = inp.dtc or {}
    if not dtc.get("scanned"):
        return Outcome(NOT_READY, reason="no diagnostic scan has completed yet")
    if not dtc.get("responding", True):
        return Outcome(DATA_UNAVAILABLE,
                       reason="the ECU is not answering diagnostic requests")

    codes = list(dtc.get("codes") or [])
    detail = {"dtc_codes": sorted(codes),
              "dtc_added": sorted(dtc.get("added") or []),
              "mil_commanded_on": bool(dtc.get("mil")),
              "dtc_count": dtc.get("count")}

    if not codes:
        return Outcome(READY, PASSED, confidence=0.9,
                       reason="the vehicle is reporting no codes", detail=detail)

    # Severity follows the WORST reported code, translated into the health
    # ladder by the DTC catalogue — see vehicle/dtc/catalog.HEALTH_SEVERITY on
    # why that is a translation rather than a second ladder.
    severity = dtc.get("worst_health_severity") or C.WARNING
    return Outcome(READY, FAILED_PENDING,
                   confidence=0.95,
                   reason=(f"the vehicle is reporting {len(codes)} diagnostic "
                           f"trouble code{'' if len(codes) == 1 else 's'}: "
                           + ", ".join(sorted(codes))),
                   detail=detail, severity=severity)


def _eval_coolant_limit(inp: MonitorInput) -> Outcome:
    """The fixed ceiling. Nothing learned can move it.

    Reads config.TELEMETRY_BANDS["coolant_temp"]["crit_high"], which is the
    number the panel's own row is judged against. This monitor exists alongside
    that row rather than instead of it: the row says "it is hot right now" and
    this says "it has been hot for long enough to be the engine rather than the
    sensor", and the second is what earns a freeze frame.
    """
    limit = _band("coolant_temp", "crit_high")
    if limit is None:
        return Outcome(NOT_SUPPORTED, reason="no coolant ceiling is configured")
    if not _running(inp):
        return Outcome(INHIBITED, reason="the engine is not running")

    points = series(inp, "coolant_temp")
    if len(points) < 2:
        return Outcome(NOT_READY,
                       reason=f"{len(points)} of 2 coolant readings so far")

    held = sustained_beyond(points, limit, above=True,
                            hold_s=config.POWERTRAIN_COOLANT_LIMIT_HOLD_S)
    peak = max(v for _, v in points)
    current = points[-1][1]
    detail = {"coolant_temp": round(current, 1), "limit_f": limit,
              "peak_coolant_temp": round(peak, 1),
              "held_s": None if held is None else round(held, 1),
              "required_hold_s": config.POWERTRAIN_COOLANT_LIMIT_HOLD_S,
              "sample_count": len(points),
              "first_sample_at": points[0][0], "last_sample_at": points[-1][0],
              "engine_running": True, "rpm": latest(inp, "rpm"),
              "vehicle_speed": latest(inp, "vehicle_speed"),
              "engine_load": latest(inp, "engine_load"),
              "oil_temp": latest(inp, "oil_temp"),
              "intake_air_temp": latest(inp, "intake_air_temp"),
              "data_quality": _quality(inp)}

    if held is not None:
        return Outcome(READY, FAILED_PENDING,
                       confidence=_confidence(len(points), peak - limit, 8.0),
                       reason=f"coolant at {current:.0f}°F, above the "
                              f"{limit:.0f}°F limit for {held:.0f}s",
                       detail=detail, severity=C.CRITICAL)
    return Outcome(READY, PASSED, confidence=0.85,
                   reason=f"coolant at {current:.0f}°F, below the "
                          f"{limit:.0f}°F limit", detail=detail)


def _eval_coolant_rise(inp: MonitorInput) -> Outcome:
    """How fast, rather than how high.

    Warm-up is excluded and that exclusion is the monitor. A cold engine climbs
    at 40°F a minute and is perfectly healthy; a warm one climbing at seven has
    lost something. Without the gate this would fire on every journey, which is
    the fastest way to make a driver stop reading the panel.
    """
    if not _running(inp):
        return Outcome(INHIBITED, reason="the engine is not running")
    warm = _warm(inp)
    if warm is None:
        return Outcome(DATA_UNAVAILABLE, reason="no coolant reading")
    if not warm:
        return Outcome(INHIBITED,
                       reason="the engine is still warming up, where a fast "
                              "rise is normal")

    points = series(inp, "coolant_temp",
                    window_s=config.POWERTRAIN_COOLANT_RISE_WINDOW_S)
    if len(points) < config.POWERTRAIN_COOLANT_RISE_MIN_SAMPLES:
        return Outcome(NOT_READY,
                       reason=f"{len(points)} of "
                              f"{config.POWERTRAIN_COOLANT_RISE_MIN_SAMPLES} "
                              f"readings in the window",
                       detail={"sample_count": len(points)})

    rate = slope_per_min(points)
    if rate is None:
        return Outcome(NOT_READY, reason="the readings do not span enough time")

    limit = config.POWERTRAIN_COOLANT_RISE_F_PER_MIN
    detail = {"coolant_rate_f_per_min": round(rate, 2), "limit_f_per_min": limit,
              "coolant_temp": round(points[-1][1], 1),
              "window_s": round(points[-1][0] - points[0][0], 1),
              "sample_count": len(points),
              "first_sample_at": points[0][0], "last_sample_at": points[-1][0],
              "vehicle_speed": latest(inp, "vehicle_speed"),
              "engine_load": latest(inp, "engine_load"),
              "rpm": latest(inp, "rpm"), "engine_running": True,
              "data_quality": _quality(inp)}

    if rate >= limit:
        return Outcome(READY, FAILED_PENDING,
                       confidence=_confidence(len(points), rate - limit, 5.0),
                       reason=f"coolant climbing at {rate:.1f}°F per minute on a "
                              f"warm engine, against a {limit:.1f} limit",
                       detail=detail, severity=C.WARNING)
    return Outcome(READY, PASSED, confidence=0.8,
                   reason=f"coolant rate {rate:+.1f}°F per minute, inside the "
                          f"limit", detail=detail)


def _eval_coolant_contextual(inp: MonitorInput) -> Outcome:
    """This car, against what this car normally does.

    The only monitor here that can find something while every fixed band still
    passes, and the only one whose evidence is history rather than the last two
    minutes. §5.5 applies with full force: it can RAISE a finding the bands
    would have missed and it cannot relax anything — the hard-limit monitor
    above runs on the same data and does not consult a baseline at all.
    """
    if not _running(inp):
        return Outcome(INHIBITED, reason="the engine is not running")
    warm = _warm(inp)
    if not warm:
        return Outcome(INHIBITED,
                       reason="the engine is not up to temperature yet")

    speed = latest(inp, "vehicle_speed")
    if speed is None:
        return Outcome(DATA_UNAVAILABLE, reason="no road speed")
    if speed < 25.0:
        # The baseline this compares against is the cruise-conditioned one, and
        # comparing a crawl against it would find a fault in traffic.
        return Outcome(INHIBITED,
                       reason="below cruising speed, where the conditioned "
                              "baseline does not apply")

    baseline = inp.baselines.get("coolant_temp@cruise")
    days = inp.baseline_days.get("coolant_temp@cruise", 0)
    if baseline is None or days < config.POWERTRAIN_COOLANT_CONTEXT_MIN_DAYS:
        return Outcome(NOT_READY,
                       reason=f"only {days} days of comparable history for this "
                              f"vehicle",
                       detail={"baseline_days": days})

    points = series(inp, "coolant_temp",
                    window_s=config.POWERTRAIN_COOLANT_CONTEXT_WINDOW_S)
    if len(points) < 3:
        return Outcome(NOT_READY, reason="not enough readings at cruise")

    mean = sum(v for _, v in points) / len(points)
    delta = mean - baseline
    limit = config.POWERTRAIN_COOLANT_CONTEXT_DELTA_F
    detail = {"coolant_temp": round(mean, 1),
              "baseline_coolant_temp": round(baseline, 1),
              "delta_f": round(delta, 1), "limit_f": limit,
              "baseline_days": days, "sample_count": len(points),
              "first_sample_at": points[0][0], "last_sample_at": points[-1][0],
              "vehicle_speed": speed, "engine_load": latest(inp, "engine_load"),
              "rpm": latest(inp, "rpm"), "engine_running": True,
              "data_quality": _quality(inp)}

    if delta >= limit:
        return Outcome(READY, FAILED_PENDING,
                       confidence=_confidence(len(points), delta - limit, 6.0),
                       reason=f"coolant averaging {mean:.0f}°F at cruise against "
                              f"a {baseline:.0f}°F baseline for this vehicle",
                       detail=detail, severity=C.ADVISORY)
    return Outcome(READY, PASSED, confidence=0.75,
                   reason="coolant is where it normally is at cruise",
                   detail=detail)


def _eval_charging(inp: MonitorInput) -> Outcome:
    """Is the alternator keeping up?

    Engine-running only, and the gate matters: 12.4 V on a parked car is a
    healthy battery and on a running one it is a charging system that has
    stopped working.
    """
    floor = _band("battery_voltage", "warn_low")
    if floor is None:
        return Outcome(NOT_SUPPORTED, reason="no charging floor is configured")
    if not _running(inp):
        return Outcome(INHIBITED,
                       reason="the engine is not running, where a resting "
                              "battery voltage is correct")

    points = series(inp, "battery_voltage")
    if len(points) < 3:
        return Outcome(NOT_READY,
                       reason=f"{len(points)} of 3 voltage readings so far")

    held = sustained_beyond(points, floor, above=False,
                            hold_s=config.POWERTRAIN_CHARGING_HOLD_S)
    current = points[-1][1]
    detail = {"battery_voltage": round(current, 2), "floor_v": floor,
              "minimum_v": round(min(v for _, v in points), 2),
              "held_s": None if held is None else round(held, 1),
              "required_hold_s": config.POWERTRAIN_CHARGING_HOLD_S,
              "sample_count": len(points), "rpm": latest(inp, "rpm"),
              "engine_running": True, "data_quality": _quality(inp)}

    if held is not None:
        crit = _band("battery_voltage", "crit_low")
        severity = C.CRITICAL if (crit is not None and current <= crit) \
            else C.WARNING
        return Outcome(READY, FAILED_PENDING,
                       confidence=_confidence(len(points), floor - current, 0.5),
                       reason=f"{current:.1f} V with the engine running, below "
                              f"the {floor:.1f} V floor for {held:.0f}s",
                       detail=detail, severity=severity)
    return Outcome(READY, PASSED, confidence=0.85,
                   reason=f"charging at {current:.1f} V", detail=detail)


def _eval_start_voltage(inp: MonitorInput) -> Outcome:
    """What the battery does when the starter loads it.

    The one measurement on this list that a running-voltage band can never make.
    A battery losing capacity holds 14.2 V all day and drops a little further
    every time it cranks, so the evidence exists for about two seconds per drive
    and only means anything across several of them.
    """
    events = list(inp.start_events or [])
    if len(events) < config.POWERTRAIN_START_EVENTS_MIN:
        return Outcome(NOT_READY,
                       reason=f"{len(events)} of "
                              f"{config.POWERTRAIN_START_EVENTS_MIN} recorded "
                              f"starts",
                       detail={"start_event_count": len(events)})

    volts = [float(e["min_v"]) for e in events if e.get("min_v") is not None]
    if len(volts) < config.POWERTRAIN_START_EVENTS_MIN:
        return Outcome(NOT_READY, reason="not enough starts carry a voltage",
                       detail={"start_event_count": len(volts)})

    first, last = volts[0], volts[-1]
    decline = first - last
    floor = config.POWERTRAIN_START_V_FLOOR
    detail = {"start_event_count": len(volts),
              "start_voltage_first": round(first, 2),
              "start_voltage_last": round(last, 2),
              "cranking_voltage": round(last, 2),
              "decline_v": round(decline, 2),
              "floor_v": floor,
              "required_decline_v": config.POWERTRAIN_START_V_DECLINE_V,
              "engine_running": _running(inp),
              "data_quality": _quality(inp)}

    if last <= floor:
        return Outcome(READY, FAILED_PENDING,
                       confidence=_confidence(len(volts), floor - last, 0.5),
                       reason=f"cranking voltage down to {last:.1f} V, at or "
                              f"below the {floor:.1f} V floor",
                       detail=detail, severity=C.WARNING)
    if decline >= config.POWERTRAIN_START_V_DECLINE_V:
        return Outcome(READY, FAILED_PENDING,
                       confidence=_confidence(len(volts), decline, 0.4),
                       reason=f"cranking voltage has fallen {decline:.1f} V "
                              f"across {len(volts)} recorded starts, from "
                              f"{first:.1f} to {last:.1f}",
                       detail=detail, severity=C.ADVISORY)
    return Outcome(READY, PASSED, confidence=0.75,
                   reason=f"cranking voltage steady around {last:.1f} V",
                   detail=detail)


def _eval_fuel_trim(inp: MonitorInput) -> Outcome:
    """The earliest number on the car that says something has changed.

    Warm and closed-loop only. Both gates are load-bearing: a cold engine has
    not started trimming, and an engine at wide-open throttle has stopped —
    reporting the zero it reports there as a healthy trim would be reading a
    switched-off system as a passing one.
    """
    warn_high = _band("ltft_b1", "warn_high")
    warn_low = _band("ltft_b1", "warn_low")
    if warn_high is None and warn_low is None:
        return Outcome(NOT_SUPPORTED, reason="no fuel trim band is configured")
    if not _running(inp):
        return Outcome(INHIBITED, reason="the engine is not running")
    if not _warm(inp):
        return Outcome(INHIBITED,
                       reason="the engine is not warm enough to be trimming yet")

    throttle = latest(inp, "throttle_pct")
    if throttle is not None and throttle > 55.0:
        return Outcome(INHIBITED,
                       reason="the engine is in open loop under power, where it "
                              "is not trimming at all")

    # Per SAMPLE, not just per gate. A drive alternates between closed loop and
    # open loop many times, and the zeros an engine reports at wide-open
    # throttle are not trims — folding them into the history would average a
    # switched-off system in with a working one and quietly pull every finding
    # back inside the band.
    points = [(s.at, float(s.values["ltft_b1"])) for s in usable(inp)
              if s.values.get("ltft_b1") is not None
              and (s.values.get("throttle_pct") is None
                   or s.values["throttle_pct"] <= 55.0)
              and (s.values.get("coolant_temp") is None
                   or s.values["coolant_temp"] >= config.POWERTRAIN_WARM_COOLANT_F)]
    if len(points) < config.POWERTRAIN_LTFT_MIN_SAMPLES:
        return Outcome(NOT_READY,
                       reason=f"{len(points)} of "
                              f"{config.POWERTRAIN_LTFT_MIN_SAMPLES} warm "
                              f"closed-loop readings",
                       detail={"sample_count": len(points)})

    current = points[-1][1]
    mean = sum(v for _, v in points) / len(points)
    high_held = sustained_beyond(points, warn_high, above=True,
                                 hold_s=config.POWERTRAIN_LTFT_HOLD_S) \
        if warn_high is not None else None
    low_held = sustained_beyond(points, warn_low, above=False,
                                hold_s=config.POWERTRAIN_LTFT_HOLD_S) \
        if warn_low is not None else None

    detail = {"ltft_b1": round(current, 1), "ltft_mean": round(mean, 1),
              "stft_b1": latest(inp, "stft_b1"),
              "warn_high": warn_high, "warn_low": warn_low,
              "held_s": high_held if high_held is not None else low_held,
              "required_hold_s": config.POWERTRAIN_LTFT_HOLD_S,
              "baseline_ltft_b1": inp.baselines.get("ltft_b1"),
              "coolant_temp": latest(inp, "coolant_temp"),
              "engine_load": latest(inp, "engine_load"),
              "rpm": latest(inp, "rpm"), "map_kpa": latest(inp, "map_kpa"),
              "maf_gs": latest(inp, "maf_gs"),
              "afr_wideband": latest(inp, "afr_wideband"),
              "sample_count": len(points), "data_quality": _quality(inp)}

    if high_held is not None:
        return Outcome(READY, FAILED_PENDING,
                       confidence=_confidence(len(points), current - warn_high, 5.0),
                       reason=f"long-term fuel trim at {current:+.1f}% warm and "
                              f"in closed loop, past {warn_high:+.0f}% for "
                              f"{high_held:.0f}s — the engine is being given "
                              f"more fuel than the tables expect",
                       detail=detail, severity=C.ADVISORY)
    if low_held is not None:
        return Outcome(READY, FAILED_PENDING,
                       confidence=_confidence(len(points), warn_low - current, 5.0),
                       reason=f"long-term fuel trim at {current:+.1f}% warm and "
                              f"in closed loop, past {warn_low:+.0f}% for "
                              f"{low_held:.0f}s — the engine is being given "
                              f"less fuel than the tables expect",
                       detail=detail, severity=C.ADVISORY)
    return Outcome(READY, PASSED, confidence=0.8,
                   reason=f"long-term fuel trim at {current:+.1f}%, inside its "
                          f"band", detail=detail)


# Channels worth checking for integrity. Not every registered channel: a signal
# this vehicle does not expose is absent by design, and reporting it as frozen
# would turn a capability into a fault.
_INTEGRITY_CHANNELS = ("coolant_temp", "rpm", "vehicle_speed", "battery_voltage",
                       "map_kpa", "throttle_pct", "intake_air_temp",
                       "engine_load", "oil_pressure", "oil_temp")

_BAD_STATUS = ("STALE", "OFFLINE")


def _eval_signal_integrity(inp: MonitorInput) -> Outcome:
    """Is any channel lying?

    Three ways a reading can be wrong while looking right, and the first is the
    worst: a FROZEN channel passes every range check, sits comfortably in band,
    and is completely false. It is caught only by noticing that a number which
    should wander has not moved at all.
    """
    if not _running(inp):
        return Outcome(INHIBITED,
                       reason="the engine is not running, where most channels "
                              "legitimately hold still")

    u = usable(inp)
    if len(u) < config.POWERTRAIN_FROZEN_MIN_SAMPLES:
        return Outcome(NOT_READY,
                       reason=f"{len(u)} of "
                              f"{config.POWERTRAIN_FROZEN_MIN_SAMPLES} samples "
                              f"so far", detail={"sample_count": len(u)})

    frozen, jumpy, stale = [], [], []
    for channel in _INTEGRITY_CHANNELS:
        points = series(inp, channel, window_s=config.POWERTRAIN_FROZEN_S)
        if len(points) >= config.POWERTRAIN_FROZEN_MIN_SAMPLES:
            span = points[-1][0] - points[0][0]
            values = {round(v, 4) for _, v in points}
            if span >= config.POWERTRAIN_FROZEN_S and len(values) == 1:
                frozen.append(channel)

        band = config.TELEMETRY_BANDS.get(channel) or {}
        hi = band.get("crit_high")
        lo = band.get("crit_low")
        if hi is not None and lo is not None and hi > lo:
            step_limit = (hi - lo) * config.POWERTRAIN_DISCONTINUITY_FRAC
            full = series(inp, channel)
            for (_, a), (_, b) in zip(full, full[1:]):
                if abs(b - a) > step_limit:
                    jumpy.append(channel)
                    break

        last = u[-1].statuses.get(channel)
        if last in _BAD_STATUS:
            stale.append(channel)

    affected = sorted(set(frozen) | set(jumpy) | set(stale))
    detail = {"affected_channels": affected, "frozen": sorted(set(frozen)),
              "discontinuous": sorted(set(jumpy)), "stale": sorted(set(stale)),
              "sample_count": len(u), "first_sample_at": u[0].at,
              "last_sample_at": u[-1].at, "engine_running": True,
              "data_quality": _quality(inp)}

    if affected:
        why = []
        if frozen:
            why.append(f"{', '.join(sorted(set(frozen)))} has not changed at all")
        if jumpy:
            why.append(f"{', '.join(sorted(set(jumpy)))} jumped further than any "
                       f"physical process allows")
        if stale:
            why.append(f"{', '.join(sorted(set(stale)))} stopped reporting")
        return Outcome(READY, FAILED_PENDING,
                       confidence=_confidence(len(u), len(affected), 2.0),
                       reason="; ".join(why), detail=detail, severity=C.ADVISORY)
    return Outcome(READY, PASSED, confidence=0.8,
                   reason="every checked channel is behaving believably",
                   detail=detail)


def _eval_connection(inp: MonitorInput) -> Outcome:
    """Is anything reaching RIO at all?

    System-level, and it exists so the eight monitors above can be inhibited
    wholesale rather than each inventing its own fault out of the same silence —
    the same argument the tire receiver monitor makes, and the same failure it
    prevents.
    """
    link = inp.link or {}
    u = usable(inp)
    last_at = u[-1].at if u else None
    silence = (inp.now - last_at) if last_at is not None else None

    detail = {"silent_for_s": None if silence is None else round(silence, 1),
              "last_sample_at": last_at,
              "window_s": config.POWERTRAIN_NO_DATA_S,
              "source": link.get("source"),
              "outbox_pending": link.get("outbox_pending"),
              "can_state": link.get("can_state"),
              "network_state": link.get("network_state"),
              "data_quality": _quality(inp)}

    if last_at is None:
        return Outcome(READY, FAILED_PENDING, confidence=0.9,
                       reason="no engine data has arrived at all",
                       detail=detail, severity=C.INFORMATIONAL)
    if silence is not None and silence > config.POWERTRAIN_NO_DATA_S:
        return Outcome(READY, FAILED_PENDING,
                       confidence=_confidence(len(u), silence, 30.0),
                       reason=f"no engine data for {silence:.0f}s",
                       detail=detail, severity=C.INFORMATIONAL)

    outbox = link.get("outbox_pending")
    if outbox is not None and outbox >= config.POWERTRAIN_OUTBOX_WARN:
        return Outcome(READY, FAILED_PENDING, confidence=0.8,
                       reason=f"the bridge is holding {outbox} events it has "
                              f"not been able to upload",
                       detail=detail, severity=C.INFORMATIONAL)
    if link.get("can_state") in ("bus_off", "error_passive"):
        return Outcome(READY, FAILED_PENDING, confidence=0.85,
                       reason=f"the CAN interface reports {link['can_state']}",
                       detail=detail, severity=C.INFORMATIONAL)

    return Outcome(READY, PASSED, confidence=0.85,
                   reason="engine data is arriving", detail=detail)


# ---------------------------------------------------------------------------
# The definitions
# ---------------------------------------------------------------------------

def _heal(monitor_id: str) -> HealingCriteria:
    return HealingCriteria(
        required_passing_monitor_runs=config.POWERTRAIN_HEAL_RUNS.get(
            monitor_id, 2),
        required_passing_drive_cycles=0,
        minimum_stable_duration_seconds=config.POWERTRAIN_HEAL_STABLE_S.get(
            monitor_id, 300.0))


def _confirmed(monitor_id: str) -> dict:
    out = {"qualifying_runs": config.POWERTRAIN_CONFIRM_RUNS.get(monitor_id, 2)}
    cycles = config.POWERTRAIN_CONFIRM_CYCLES.get(monitor_id, 0)
    if cycles:
        out["drive_cycles"] = cycles
    return out


MONITORS = (
    MonitorDefinition(
        monitor_id="engine.new_dtc",
        component_type="engine_ecu",
        required_inputs=("dtc_scan",),
        enabling=EnablingConditions(minimum_valid_samples=0,
                                    require_system_healthy=False),
        inhibiting_conditions=("no diagnostic scan has completed",
                               "the ECU is not answering"),
        pending_criteria={"any_code_reported": True},
        confirmed_criteria=_confirmed("engine.new_dtc"),
        healing=_heal("engine.new_dtc"),
        freeze_frame_fields=C.FF_DTC,
        evaluate=_eval_new_dtc,
        default_severity=C.WARNING),

    MonitorDefinition(
        monitor_id="engine.coolant_hard_limit",
        component_type="cooling",
        required_inputs=("coolant_temp", "engine_state"),
        enabling=EnablingConditions(minimum_valid_samples=2),
        inhibiting_conditions=("the engine is not running",
                               "no engine data is arriving"),
        pending_criteria={"at_or_above": "TELEMETRY_BANDS.coolant_temp.crit_high",
                          "sustained_s": config.POWERTRAIN_COOLANT_LIMIT_HOLD_S,
                          "learned_baseline_may_not_relax_this": True},
        confirmed_criteria=_confirmed("engine.coolant_hard_limit"),
        healing=_heal("engine.coolant_hard_limit"),
        freeze_frame_fields=C.FF_THERMAL,
        evaluate=_eval_coolant_limit,
        requires_primary=True,
        default_severity=C.CRITICAL),

    MonitorDefinition(
        monitor_id="engine.coolant_rate_of_rise",
        component_type="cooling",
        required_inputs=("coolant_temp", "engine_state"),
        enabling=EnablingConditions(
            minimum_valid_samples=config.POWERTRAIN_COOLANT_RISE_MIN_SAMPLES),
        inhibiting_conditions=("the engine is not running",
                               "the engine is still warming up",
                               "no engine data is arriving"),
        pending_criteria={"rate_f_per_min": config.POWERTRAIN_COOLANT_RISE_F_PER_MIN,
                          "window_s": config.POWERTRAIN_COOLANT_RISE_WINDOW_S,
                          "excludes": "warm-up"},
        confirmed_criteria=_confirmed("engine.coolant_rate_of_rise"),
        healing=_heal("engine.coolant_rate_of_rise"),
        freeze_frame_fields=C.FF_THERMAL,
        evaluate=_eval_coolant_rise,
        requires_primary=True,
        default_severity=C.WARNING),

    MonitorDefinition(
        monitor_id="engine.coolant_contextual",
        component_type="cooling",
        required_inputs=("coolant_temp", "vehicle_speed", "baseline"),
        enabling=EnablingConditions(minimum_valid_samples=3),
        inhibiting_conditions=("the engine is not running",
                               "the engine is not up to temperature",
                               "below cruising speed",
                               "too few days of comparable history"),
        pending_criteria={"above_own_baseline_f":
                          config.POWERTRAIN_COOLANT_CONTEXT_DELTA_F,
                          "condition": "cruise",
                          "window_s":
                          config.POWERTRAIN_COOLANT_CONTEXT_WINDOW_S,
                          "minimum_days":
                          config.POWERTRAIN_COOLANT_CONTEXT_MIN_DAYS,
                          "may_not_relax_a_fixed_limit": True},
        confirmed_criteria=_confirmed("engine.coolant_contextual"),
        healing=_heal("engine.coolant_contextual"),
        freeze_frame_fields=C.FF_THERMAL,
        evaluate=_eval_coolant_contextual,
        requires_primary=True,
        default_severity=C.ADVISORY),

    MonitorDefinition(
        monitor_id="engine.charging_voltage",
        component_type="electrical",
        required_inputs=("battery_voltage", "engine_state"),
        enabling=EnablingConditions(minimum_valid_samples=3),
        inhibiting_conditions=("the engine is not running",),
        pending_criteria={"at_or_below": "TELEMETRY_BANDS.battery_voltage.warn_low",
                          "sustained_s": config.POWERTRAIN_CHARGING_HOLD_S},
        confirmed_criteria=_confirmed("engine.charging_voltage"),
        healing=_heal("engine.charging_voltage"),
        freeze_frame_fields=C.FF_ELECTRICAL,
        evaluate=_eval_charging,
        requires_primary=True,
        default_severity=C.WARNING),

    MonitorDefinition(
        monitor_id="engine.start_voltage_trend",
        component_type="electrical",
        required_inputs=("start_events",),
        enabling=EnablingConditions(minimum_valid_samples=0),
        inhibiting_conditions=("too few recorded starts",),
        pending_criteria={"cranking_floor_v": config.POWERTRAIN_START_V_FLOOR,
                          "or_decline_v": config.POWERTRAIN_START_V_DECLINE_V,
                          "across_starts": config.POWERTRAIN_START_EVENTS_MIN},
        confirmed_criteria=_confirmed("engine.start_voltage_trend"),
        healing=_heal("engine.start_voltage_trend"),
        freeze_frame_fields=C.FF_ELECTRICAL,
        evaluate=_eval_start_voltage,
        default_severity=C.ADVISORY),

    MonitorDefinition(
        monitor_id="engine.fuel_trim_long_term",
        component_type="fuel",
        required_inputs=("ltft_b1", "coolant_temp", "throttle_pct"),
        enabling=EnablingConditions(
            minimum_valid_samples=config.POWERTRAIN_LTFT_MIN_SAMPLES),
        inhibiting_conditions=("the engine is not running",
                               "the engine is not warm",
                               "the engine is in open loop under power"),
        pending_criteria={"beyond": "TELEMETRY_BANDS.ltft_b1",
                          "sustained_s": config.POWERTRAIN_LTFT_HOLD_S,
                          "requires": "warm and closed loop"},
        confirmed_criteria=_confirmed("engine.fuel_trim_long_term"),
        healing=_heal("engine.fuel_trim_long_term"),
        freeze_frame_fields=C.FF_FUEL,
        evaluate=_eval_fuel_trim,
        requires_primary=True,
        default_severity=C.ADVISORY),

    MonitorDefinition(
        monitor_id="engine.signal_integrity",
        component_type="sensing",
        required_inputs=("all_channels",),
        enabling=EnablingConditions(
            minimum_valid_samples=config.POWERTRAIN_FROZEN_MIN_SAMPLES),
        inhibiting_conditions=("the engine is not running",),
        pending_criteria={"frozen_s": config.POWERTRAIN_FROZEN_S,
                          "or_step_fraction_of_band":
                          config.POWERTRAIN_DISCONTINUITY_FRAC,
                          "or_channel_stale": True},
        confirmed_criteria=_confirmed("engine.signal_integrity"),
        healing=_heal("engine.signal_integrity"),
        freeze_frame_fields=C.FF_SIGNAL,
        evaluate=_eval_signal_integrity,
        default_severity=C.ADVISORY),

    MonitorDefinition(
        monitor_id="engine.connection",
        component_type="link",
        required_inputs=("sample_times", "gateway_state"),
        enabling=EnablingConditions(minimum_valid_samples=0,
                                    require_system_healthy=False),
        inhibiting_conditions=(),
        pending_criteria={"no_data_for_s": config.POWERTRAIN_NO_DATA_S,
                          "or_outbox_pending": config.POWERTRAIN_OUTBOX_WARN,
                          "or_can_state": ["bus_off", "error_passive"]},
        confirmed_criteria=_confirmed("engine.connection"),
        healing=_heal("engine.connection"),
        freeze_frame_fields=C.FF_LINK,
        evaluate=_eval_connection,
        default_severity=C.INFORMATIONAL),
)

BY_ID: Dict[str, MonitorDefinition] = {m.monitor_id: m for m in MONITORS}


def definitions_view() -> List[dict]:
    from diag import shadow
    return _definitions_view(MONITORS, shadowed=shadow.is_shadowed(C.DOMAIN),
                             confirm_runs=config.POWERTRAIN_CONFIRM_RUNS,
                             confirm_cycles=config.POWERTRAIN_CONFIRM_CYCLES)
