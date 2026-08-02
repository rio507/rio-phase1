"""monitors.py — the diagnostic monitors, and the pure logic that runs them.

Each detector is a MONITOR in the OBD sense: a test that runs only when the
conditions for a meaningful result are present, reports whether it could run at
all, and — separately — what it found when it did. RIO Tire Health is not an
OBD-II system; this is the discipline, not the standard.

THE TWO FIELDS THAT MATTER MOST
-------------------------------
A monitor reports `status` and `last_result`, and they are different questions:

    status       could the monitor run, and can it now?
                 NOT_SUPPORTED / NOT_READY / RUNNING / READY /
                 INHIBITED / DATA_UNAVAILABLE
    last_result  what did it find the last time it DID run?
                 PASSED / FAILED_PENDING / FAILED_CONFIRMED / None

Folding these into one enum is the mistake that makes a diagnostic system lie.
"NOT_READY" is not a result, and a monitor that has never run has no result at
all — reporting either of those as "passed" is how a system ends up telling a
driver their tires are fine when what it means is that it has not looked.

    "I do not have enough comparable readings yet to evaluate a slow leak, but
     the current pressure is not critically low."

That sentence needs both fields. One enum cannot say it.

WHAT IS AND IS NOT IN THIS FILE
-------------------------------
In: definitions, enabling and inhibiting conditions, pass/fail criteria,
confidence, and the evaluation functions. All pure — given the same samples they
return the same outcome, with no clock of their own and no I/O.

Out: counting runs, promoting CANDIDATE to ACTIVE, healing, freeze frames,
persistence, and every decision about speech. Those are engine.py's, because
they are stateful and this file is not.

LLM FIREWALL
------------
This module imports config and the stdlib. Nothing here can read a model output
because there is nothing here to read one with. tire_diag/selftest.py asserts
it, the same way headway/live_selftest.py does for live_policy.py.
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Callable, Dict, List, Optional, Tuple

import config

from . import codes as C

# --- monitor status: could it run? -----------------------------------------
NOT_SUPPORTED = "NOT_SUPPORTED"
NOT_READY = "NOT_READY"
RUNNING = "RUNNING"
READY = "READY"
INHIBITED = "INHIBITED"
DATA_UNAVAILABLE = "DATA_UNAVAILABLE"

# --- monitor result: what did it find? -------------------------------------
PASSED = "PASSED"
FAILED_PENDING = "FAILED_PENDING"
FAILED_CONFIRMED = "FAILED_CONFIRMED"

# Statuses in which the monitor did NOT produce a verdict this tick. The
# distinction is load-bearing everywhere downstream: none of these may be
# reported as a pass, and none of them advances a confirmation or healing count.
NO_VERDICT = (NOT_SUPPORTED, NOT_READY, RUNNING, INHIBITED, DATA_UNAVAILABLE)


# ---------------------------------------------------------------------------
# What a monitor is given
# ---------------------------------------------------------------------------

@dataclass
class Sample:
    """One validated report from one sensor.

    `valid` is False for a sample the plausibility gate rejected. Rejected
    samples are KEPT rather than dropped: how often a sensor talks nonsense is
    itself the evidence the plausibility monitor runs on, and a list with the
    bad ones deleted cannot support that judgement.
    """
    corner: str
    at: float
    pressure_psi: Optional[float]
    temp_f: Optional[float]
    battery_pct: Optional[float]
    connected: bool
    valid: bool = True
    reject_reason: str = ""


@dataclass
class MonitorInput:
    """Everything a monitor may look at. Nothing else is in scope."""
    corner: Optional[str]
    now: float
    samples: List[Sample]                      # this corner, oldest first
    peers: Dict[str, List[Sample]]             # the other three
    target_psi: float
    moving: bool
    speed_mph: Optional[float]
    parked_for_s: Optional[float]
    receiver_healthy: bool
    epoch_started_at: float                    # last relearn/reset for this corner
    drive_cycle_id: Optional[str]
    # Issues already ACTIVE on this corner, by monitor id. Only one monitor
    # reads it — sensor_loss_during_decline, whose entire question is "was this
    # tire already in trouble when it went quiet".
    active_monitor_ids: Tuple[str, ...] = ()


@dataclass
class Outcome:
    """What one monitor run produced."""
    status: str
    result: Optional[str] = None          # None whenever status is in NO_VERDICT
    confidence: float = 0.0
    reason: str = ""
    detail: Dict[str, object] = field(default_factory=dict)
    severity: Optional[str] = None        # overrides the code's default
    variant: Optional[str] = None         # picks between codes sharing a monitor
    urgent: bool = False


@dataclass(frozen=True)
class EnablingConditions:
    """When this monitor is allowed to reach a verdict at all."""
    minimum_vehicle_speed_mph: Optional[float] = None
    maximum_vehicle_speed_mph: Optional[float] = None
    minimum_valid_samples: int = 1
    minimum_elapsed_seconds: float = 0.0
    require_comparable_thermal_state: bool = False
    require_stable_sensor_identity: bool = True
    require_receiver_healthy: bool = True
    require_moving: bool = False
    require_parked: bool = False


@dataclass(frozen=True)
class HealingCriteria:
    required_passing_monitor_runs: int
    required_passing_drive_cycles: int = 0
    minimum_stable_duration_seconds: float = 0.0


@dataclass(frozen=True)
class MonitorDefinition:
    """One diagnostic monitor, fully described.

    The criteria dicts are prose-as-data on purpose: they are what a service
    view prints and what a test asserts against, and a monitor whose
    confirmation rule exists only inside a function body is one nobody can
    review without reading the function.
    """
    monitor_id: str
    component_type: str                  # tire | tpms_sensor | tpms_system
    required_inputs: Tuple[str, ...]
    enabling: EnablingConditions
    inhibiting_conditions: Tuple[str, ...]
    pending_criteria: Dict[str, object]
    confirmed_criteria: Dict[str, object]
    healing: HealingCriteria
    freeze_frame_fields: Tuple[str, ...]
    evaluate: Callable[[MonitorInput], Outcome]
    default_severity: str = C.WARNING
    urgent_criteria: Optional[Dict[str, object]] = None
    per_corner: bool = True
    # Shadow mode is a property of the DEPLOYMENT, not of the monitor, so it is
    # read from config at decision time rather than frozen in here. This field
    # records only whether the monitor has ever been cleared to speak at all.
    shadow_default: bool = True


# ---------------------------------------------------------------------------
# Shared helpers. Pure, and small enough to reason about.
# ---------------------------------------------------------------------------

def valid_samples(inp: MonitorInput, max_age_s: float = None) -> List[Sample]:
    """Samples this monitor may use: plausible, connected, and recent enough.

    Age is measured against the sample, not against the poll — a receiver that
    is being asked four times a second does not make its last report newer.
    """
    max_age = config.TIRE_DIAG_SAMPLE_MAX_AGE_S if max_age_s is None else max_age_s
    return [s for s in inp.samples
            if s.valid and s.connected and s.pressure_psi is not None
            and s.at >= inp.epoch_started_at
            and (inp.now - s.at) <= max_age]


def comparable(a: Sample, b: Sample) -> bool:
    """Are two samples thermally comparable?

    ~1 PSI per 10°F. Without this gate a leak monitor comparing a warm motorway
    reading with yesterday's cold parked one finds a leak in a perfectly good
    tire, every autumn, on all four corners at once.

    A sample with no temperature is NOT comparable to anything. Cheap TPMS
    sensors report pressure only, and the honest response to "I cannot tell
    whether these two readings are thermally alike" is to decline the
    comparison, not to assume it.
    """
    if a.temp_f is None or b.temp_f is None:
        return False
    return abs(a.temp_f - b.temp_f) <= config.TIRE_DIAG_COMPARABLE_TEMP_F


def comparable_pairs(samples: List[Sample], window_s: float,
                     now: float) -> Optional[Tuple[Sample, Sample]]:
    """The oldest and newest thermally comparable samples spanning the window.

    Returns None when no such pair exists, which is the INHIBITED case and not
    a pass: "I could not make this comparison" and "I made it and found nothing"
    are different answers.
    """
    recent = [s for s in samples if (now - s.at) <= window_s]
    if len(recent) < 2:
        return None
    newest = recent[-1]
    for old in recent:
        if old is newest:
            break
        if comparable(old, newest) and (newest.at - old.at) >= window_s * 0.5:
            return old, newest
    return None


def peer_change(inp: MonitorInput, window_s: float) -> Optional[float]:
    """Mean pressure change across the other corners over the same window.

    This is the single most useful number in the whole file. Weather, altitude
    and a cold night move every tire together; a puncture moves one. Subtracting
    the peers is what turns "this tire is 4 PSI down" into "this tire is 4 PSI
    down and the others are not", which is the only version of that sentence
    worth acting on.
    """
    changes = []
    for corner, samples in inp.peers.items():
        usable = [s for s in samples
                  if s.valid and s.connected and s.pressure_psi is not None
                  and (inp.now - s.at) <= window_s]
        if len(usable) < 2:
            continue
        pair = comparable_pairs(usable, window_s, inp.now)
        if pair is None:
            continue
        old, new = pair
        changes.append(new.pressure_psi - old.pressure_psi)
    if not changes:
        return None
    return sum(changes) / len(changes)


def _gate(inp: MonitorInput, d: MonitorDefinition) -> Optional[Outcome]:
    """The enabling and inhibiting conditions, applied in one place.

    Order is deliberate and is the order of what makes an answer impossible
    first: no sensor at all, then no data, then a condition that forbids the
    comparison, then not enough evidence. A monitor's status should name the
    FIRST reason it could not judge, not whichever check happened to run last.
    """
    e = d.enabling

    if inp.corner is not None and not inp.samples:
        return Outcome(NOT_SUPPORTED, reason="no sensor has ever reported on this corner")

    if e.require_receiver_healthy and not inp.receiver_healthy:
        # Every corner silent is one receiver fault. Running four tire monitors
        # against it would manufacture four tire faults out of one box fault --
        # and that is a real failure mode, not a hypothetical: it is exactly
        # what an unguarded monitor does during a receiver outage.
        return Outcome(INHIBITED, reason="the receiver is not reporting at all")

    latest = inp.samples[-1] if inp.samples else None
    if "pressure" in d.required_inputs:
        if latest is None or not latest.connected:
            return Outcome(DATA_UNAVAILABLE, reason="the sensor is not reporting")
        if (inp.now - latest.at) > config.TIRE_DIAG_SAMPLE_MAX_AGE_S:
            return Outcome(DATA_UNAVAILABLE,
                           reason=f"the last report is {int(inp.now - latest.at)}s old")

    if e.require_moving and not inp.moving:
        return Outcome(INHIBITED, reason="the car is not moving")
    if e.require_parked and inp.moving:
        return Outcome(INHIBITED, reason="the car is moving")
    if e.minimum_vehicle_speed_mph is not None:
        if inp.speed_mph is None:
            return Outcome(INHIBITED, reason="road speed is unknown")
        if inp.speed_mph < e.minimum_vehicle_speed_mph:
            return Outcome(INHIBITED, reason="below the minimum speed for this monitor")
    if e.maximum_vehicle_speed_mph is not None and inp.speed_mph is not None \
            and inp.speed_mph > e.maximum_vehicle_speed_mph:
        return Outcome(INHIBITED, reason="above the maximum speed for this monitor")

    usable = valid_samples(inp)
    if "pressure" in d.required_inputs and len(usable) < e.minimum_valid_samples:
        return Outcome(NOT_READY,
                       reason=f"{len(usable)} of {e.minimum_valid_samples} trusted "
                              f"samples so far",
                       detail={"valid_sample_count": len(usable)})

    if e.minimum_elapsed_seconds and usable:
        span = usable[-1].at - usable[0].at
        if span < e.minimum_elapsed_seconds:
            # RUNNING, not NOT_READY: it has the samples and is waiting out the
            # window. That is a monitor in progress, and a driver asking is owed
            # "still watching" rather than "not enough data".
            return Outcome(RUNNING,
                           reason=f"collecting comparable readings "
                                  f"({int(span)}s of {int(e.minimum_elapsed_seconds)}s)",
                           detail={"valid_sample_count": len(usable),
                                   "span_s": round(span, 1)})
    return None


def _confidence(n_samples: int, margin: float, scale: float) -> float:
    """0..1, from how much evidence there is and how far past the line it went.

    Deliberately blunt. A confidence that looks precise invites being read as a
    probability, and this is not one — it is "how much would I stake on this",
    used to order findings and to decide nothing on its own.
    """
    ev = min(1.0, n_samples / 6.0)
    mag = 1.0 - math.exp(-max(0.0, margin) / max(1e-6, scale))
    return round(0.35 + 0.4 * ev + 0.25 * mag, 3)


# ---------------------------------------------------------------------------
# The monitors
# ---------------------------------------------------------------------------

def _eval_low_pressure(inp: MonitorInput) -> Outcome:
    """Absolute under-inflation against the placard.

    No thermal gate, deliberately, and it is worth being explicit about why: a
    cold tire that is 4 PSI down IS 4 PSI down. It will come up as it warms, but
    it is being driven on under-inflated right now, and that is the fact the
    driver needs. Weather is why it happened, not a reason to say nothing.

    The slow-leak monitor is the one that must not confuse the two, because it
    is claiming air has LEFT the tire, which weather does not do.
    """
    usable = valid_samples(inp)
    threshold = inp.target_psi - config.TIRE_PRESSURE_WARN_DELTA
    recent = usable[-2:]
    below = [s for s in recent if s.pressure_psi <= threshold]
    latest = usable[-1]
    peers = peer_change(inp, config.TIRE_DIAG_LEAK_WINDOW_S)

    if len(below) >= 2:
        margin = threshold - min(s.pressure_psi for s in below)
        return Outcome(
            READY, FAILED_PENDING,
            confidence=_confidence(len(usable), margin, 2.0),
            reason=f"{latest.pressure_psi:.1f} PSI against a {inp.target_psi:.1f} "
                   f"PSI target on {len(below)} consecutive reports",
            detail={"pressure_psi": latest.pressure_psi,
                    "threshold_psi": round(threshold, 1),
                    "margin_psi": round(margin, 2),
                    "valid_sample_count": len(usable),
                    "peer_change_psi": None if peers is None else round(peers, 2)})

    # Recovery has to clear the line by the hysteresis before it counts, or a
    # pressure sitting on the threshold heals and re-fails on alternate reports
    # forever.
    healed = latest.pressure_psi >= threshold + config.TIRE_DIAG_HEAL_HYSTERESIS_PSI
    return Outcome(READY, PASSED if healed else FAILED_PENDING,
                   confidence=0.6,
                   reason=("pressure is back above the threshold" if healed
                           else "one report below the threshold, not two"),
                   detail={"pressure_psi": latest.pressure_psi,
                           "threshold_psi": round(threshold, 1),
                           "valid_sample_count": len(usable)})


def _eval_critical_pressure(inp: MonitorInput) -> Outcome:
    """Dangerously low, by the placard delta or by the absolute floor.

    Two thresholds because they answer different questions. The delta says the
    tire is far from where THIS car wants it; the floor says the sidewall is
    carrying load no tire was built for, whatever the placard says.

    The urgent path is narrow on purpose. Being wrong here means interrupting a
    driver at speed, so it needs two validated samples, motion, and evidence the
    pressure is still going down — a single bad packet, a wake-up frame or an
    unknown sensor cannot satisfy any of that.
    """
    usable = valid_samples(inp)
    delta_threshold = inp.target_psi - config.TIRE_PRESSURE_CRITICAL_DELTA
    floor = config.TIRE_DIAG_CRITICAL_FLOOR_PSI
    threshold = max(delta_threshold, floor) if floor > delta_threshold else delta_threshold

    recent = usable[-2:]
    below = [s for s in recent if s.pressure_psi <= threshold]
    latest = usable[-1]

    if len(below) < 2:
        healed = latest.pressure_psi >= threshold + config.TIRE_DIAG_HEAL_HYSTERESIS_PSI
        return Outcome(READY, PASSED if healed else FAILED_PENDING,
                       confidence=0.6,
                       reason=("above the critical threshold" if healed
                               else "one report below critical, not two"),
                       detail={"pressure_psi": latest.pressure_psi,
                               "threshold_psi": round(threshold, 1),
                               "valid_sample_count": len(usable)})

    falling = (len(usable) >= 2
               and (usable[-2].pressure_psi - latest.pressure_psi)
               >= config.TIRE_DIAG_FALLING_PSI)
    urgent = bool(inp.moving and falling and len(usable) >= 2
                  and latest.pressure_psi <= threshold)

    margin = threshold - latest.pressure_psi
    return Outcome(
        READY, FAILED_PENDING,
        confidence=_confidence(len(usable), margin, 3.0),
        reason=f"{latest.pressure_psi:.1f} PSI, below the critical threshold of "
               f"{threshold:.1f} PSI" + (" and still falling" if falling else ""),
        detail={"pressure_psi": latest.pressure_psi,
                "threshold_psi": round(threshold, 1),
                "floor_psi": floor,
                "falling": falling,
                "margin_psi": round(margin, 2),
                "valid_sample_count": len(usable)},
        severity=C.CRITICAL, urgent=urgent)


def _eval_slow_leak(inp: MonitorInput) -> Outcome:
    """Air actually leaving, as distinct from a cold morning.

    Three things have to be true, and the second and third are what make this
    monitor worth having:

      1. a decline across the window
      2. between THERMALLY COMPARABLE readings, so the decline is not the
         thermometer
      3. larger than the peers' decline by a margin, so it is not the weather

    Fail any of them and the honest answer is INHIBITED or PASSED, never a
    quiet fail. This is the monitor most likely to cry wolf and the one whose
    false positive is most expensive: "your tire has a slow leak" sends somebody
    to a tire shop.
    """
    usable = valid_samples(inp, max_age_s=config.TIRE_DIAG_LEAK_WINDOW_S * 2)
    pair = comparable_pairs(usable, config.TIRE_DIAG_LEAK_WINDOW_S, inp.now)
    if pair is None:
        return Outcome(
            INHIBITED,
            reason="no two readings in the window are thermally comparable",
            detail={"valid_sample_count": len(usable)})

    old, new = pair
    change = new.pressure_psi - old.pressure_psi
    peers = peer_change(inp, config.TIRE_DIAG_LEAK_WINDOW_S)

    detail = {"change_psi": round(change, 2),
              "window_s": round(new.at - old.at, 1),
              "from_psi": old.pressure_psi, "to_psi": new.pressure_psi,
              "from_temp_f": old.temp_f, "to_temp_f": new.temp_f,
              "peer_change_psi": None if peers is None else round(peers, 2),
              "valid_sample_count": len(usable)}

    if change > -config.TIRE_DIAG_LEAK_PSI:
        return Outcome(READY, PASSED, confidence=0.7,
                       reason="no material decline across comparable readings",
                       detail=detail)

    if peers is None:
        # A decline with nothing to compare it against. Real, and not yet
        # attributable: it could be this tire or it could be the afternoon.
        return Outcome(
            INHIBITED,
            reason="the other corners have no comparable readings to judge this "
                   "against",
            detail=detail)

    excess = (peers - change)          # how much MORE this corner lost
    detail["excess_vs_peers_psi"] = round(excess, 2)
    if excess < config.TIRE_DIAG_LEAK_PEER_MARGIN_PSI:
        return Outcome(
            READY, PASSED, confidence=0.75,
            reason="the decline matches the other corners — weather, not a leak",
            detail=detail)

    return Outcome(READY, FAILED_PENDING,
                   confidence=_confidence(len(usable), excess, 1.5),
                   reason=f"down {abs(change):.1f} PSI across comparable readings "
                          f"while the other corners moved {peers:+.1f} PSI",
                   detail=detail, severity=C.ADVISORY)


def _eval_asymmetric(inp: MonitorInput) -> Outcome:
    """One corner against its axle peer.

    Narrower than the slow-leak monitor's all-peer average and more sensitive
    for it: the two tires on an axle share load, road surface and weather almost
    exactly, so a difference between them is about the tire in a way a
    front-to-rear difference is not.
    """
    usable = valid_samples(inp, max_age_s=config.TIRE_DIAG_ASYM_WINDOW_S * 2)
    peer_corner = {"FL": "FR", "FR": "FL", "RL": "RR", "RR": "RL"}.get(inp.corner)
    peer_samples = inp.peers.get(peer_corner or "", [])

    mine = comparable_pairs(usable, config.TIRE_DIAG_ASYM_WINDOW_S, inp.now)
    theirs = comparable_pairs(
        [s for s in peer_samples if s.valid and s.connected and s.pressure_psi is not None],
        config.TIRE_DIAG_ASYM_WINDOW_S, inp.now)
    if mine is None or theirs is None:
        return Outcome(INHIBITED,
                       reason="no comparable pair on this corner and its axle peer",
                       detail={"peer_corner": peer_corner})

    my_change = mine[1].pressure_psi - mine[0].pressure_psi
    their_change = theirs[1].pressure_psi - theirs[0].pressure_psi
    gap = their_change - my_change
    detail = {"change_psi": round(my_change, 2),
              "peer_corner": peer_corner,
              "peer_change_psi": round(their_change, 2),
              "gap_psi": round(gap, 2),
              "valid_sample_count": len(usable)}

    if gap >= config.TIRE_DIAG_ASYM_MARGIN_PSI:
        return Outcome(READY, FAILED_PENDING,
                       confidence=_confidence(len(usable), gap, 1.5),
                       reason=f"down {abs(my_change):.1f} PSI while the {peer_corner} "
                              f"corner moved {their_change:+.1f} PSI",
                       detail=detail, severity=C.ADVISORY)
    return Outcome(READY, PASSED, confidence=0.7,
                   reason="both corners of the axle are moving together",
                   detail=detail)


def _eval_plausibility(inp: MonitorInput) -> Outcome:
    """Is this sensor telling the truth?

    Counts the samples the validation gate rejected. One rejected packet is a
    radio; four in a window is hardware, and the difference matters because the
    remedy is different — you do not replace a sensor because a wheel arch was
    in the way once.

    Note this monitor reads INVALID samples on purpose. Everything else in this
    file filters them out; here they are the measurement.
    """
    window = config.TIRE_DIAG_IMPLAUSIBLE_WINDOW_S
    recent = [s for s in inp.samples
              if (inp.now - s.at) <= window and s.at >= inp.epoch_started_at]
    if not recent:
        return Outcome(NOT_READY, reason="no reports in the plausibility window")

    bad = [s for s in recent if not s.valid]
    detail = {"implausible_count": len(bad), "sample_count": len(recent),
              "reasons": sorted({s.reject_reason for s in bad if s.reject_reason}),
              "threshold": config.TIRE_DIAG_IMPLAUSIBLE_COUNT}

    if len(bad) >= config.TIRE_DIAG_IMPLAUSIBLE_COUNT:
        return Outcome(READY, FAILED_PENDING,
                       confidence=_confidence(len(recent), len(bad), 3.0),
                       reason=f"{len(bad)} implausible reports out of {len(recent)}",
                       detail=detail, severity=C.ADVISORY)
    if len(recent) < 3:
        return Outcome(NOT_READY, reason="too few reports to judge the sensor",
                       detail=detail)
    return Outcome(READY, PASSED, confidence=0.7,
                   reason="reports are within plausible bounds", detail=detail)


def _eval_connectivity(inp: MonitorInput) -> Outcome:
    """Is the sensor still transmitting, and is its battery going?

    Only while MOVING. A parked car's sensors are asleep by design, and a
    monitor that treats that as a fault would raise four of them every night —
    which is the single most common way a naive TPMS implementation becomes
    something the driver disables.
    """
    latest = inp.samples[-1] if inp.samples else None
    if latest is None:
        return Outcome(NOT_SUPPORTED, reason="no sensor on this corner")

    battery = latest.battery_pct
    if battery is not None and battery <= config.TIRE_BATTERY_LOW_PCT:
        return Outcome(READY, FAILED_PENDING, confidence=0.8,
                       reason=f"sensor battery at {battery:.0f}%",
                       detail={"battery_pct": battery,
                               "threshold_pct": config.TIRE_BATTERY_LOW_PCT},
                       severity=C.INFORMATIONAL, variant="LOW-BATTERY")

    silence = inp.now - latest.at
    detail = {"silent_for_s": round(silence, 1),
              "tolerance_s": config.TIRE_DIAG_MISSED_REPORT_S,
              "connected": latest.connected}

    if not latest.connected or silence > config.TIRE_DIAG_MISSED_REPORT_S:
        return Outcome(READY, FAILED_PENDING,
                       confidence=_confidence(len(inp.samples),
                                              silence - config.TIRE_DIAG_MISSED_REPORT_S,
                                              120.0),
                       reason=(f"no report for {int(silence)}s while moving"
                               if latest.connected else "the sensor is off the air"),
                       detail=detail, severity=C.INFORMATIONAL, variant="STALE")
    return Outcome(READY, PASSED, confidence=0.8,
                   reason="reporting normally", detail=detail)


def _eval_loss_during_decline(inp: MonitorInput) -> Outcome:
    """The sensor went quiet on a tire that was already going down.

    The naive reading of this situation is "that corner is now unknown", which
    silently downgrades a developing fault to an absence of information. The
    correct reading is the opposite: the last thing we knew was that the tire
    was deteriorating, and losing the ability to watch it makes that worse.

    One-trip, because waiting for a second drive cycle to mention a tire that
    was going down and can no longer be seen is not a defensible delay.
    """
    declining = any(m in inp.active_monitor_ids
                    for m in ("tire.slow_leak", "tire.asymmetric_loss",
                              "tire.low_pressure", "tire.critical_low_pressure"))
    if not declining:
        return Outcome(NOT_SUPPORTED,
                       reason="this corner has no active decline to lose sight of")

    latest = inp.samples[-1] if inp.samples else None
    if latest is None:
        return Outcome(DATA_UNAVAILABLE, reason="no reports at all from this corner")

    silence = inp.now - latest.at
    lost = (not latest.connected) or silence > config.TIRE_DIAG_MISSED_REPORT_S
    detail = {"silent_for_s": round(silence, 1),
              "last_known_pressure_psi": latest.pressure_psi,
              "active_issues": list(inp.active_monitor_ids)}

    if lost and inp.moving:
        return Outcome(READY, FAILED_PENDING, confidence=0.9,
                       reason="a tire with an active decline has stopped reporting "
                              "while moving",
                       detail=detail, severity=C.CRITICAL, urgent=True)
    if lost:
        return Outcome(READY, FAILED_PENDING, confidence=0.7,
                       reason="a tire with an active decline has stopped reporting",
                       detail=detail, severity=C.WARNING)
    return Outcome(READY, PASSED, confidence=0.8,
                   reason="the corner is still reporting", detail=detail)


def _eval_inflation(inp: MonitorInput) -> Outcome:
    """Somebody put air in.

    Not a fault, and it earns a monitor anyway: it is the only positive evidence
    this system ever gets that a pressure problem was actually dealt with.
    Without it, a fixed tire heals by the same slow path as a tire that merely
    warmed up on a motorway, and "that rear-left has been stable since you added
    air" becomes a sentence RIO cannot honestly say.

    Parked only. A step this size while moving is a sensor fault, not an
    airline.
    """
    usable = valid_samples(inp)
    if len(usable) < 2:
        return Outcome(NOT_READY, reason="need two reports to see a step")
    step = usable[-1].pressure_psi - usable[-2].pressure_psi
    detail = {"step_psi": round(step, 2),
              "from_psi": usable[-2].pressure_psi, "to_psi": usable[-1].pressure_psi,
              "threshold_psi": config.TIRE_DIAG_INFLATION_STEP_PSI}
    if step >= config.TIRE_DIAG_INFLATION_STEP_PSI:
        return Outcome(READY, FAILED_PENDING, confidence=0.85,
                       reason=f"pressure stepped up {step:.1f} PSI while parked",
                       detail=detail, severity=C.INFORMATIONAL)
    return Outcome(READY, PASSED, confidence=0.7,
                   reason="no inflation step", detail=detail)


def _eval_receiver(inp: MonitorInput) -> Outcome:
    """One fault in the box, not four in the wheels.

    System-level, and it exists so that the per-corner monitors can be inhibited
    wholesale during an outage. Four simultaneous sensor failures do not happen;
    a receiver that has lost power does.
    """
    silent, total = [], 0
    for corner, samples in inp.peers.items():
        total += 1
        latest = samples[-1] if samples else None
        if latest is None or not latest.connected \
                or (inp.now - latest.at) > config.TIRE_DIAG_MISSED_REPORT_S:
            silent.append(corner)
    detail = {"corners_silent": sorted(silent), "corners_total": total}

    if total == 0:
        return Outcome(NOT_SUPPORTED, reason="no sensors are paired at all")
    if len(silent) == total:
        return Outcome(READY, FAILED_PENDING, confidence=0.9,
                       reason="all four corners have gone silent together",
                       detail=detail, severity=C.INFORMATIONAL)
    return Outcome(READY, PASSED, confidence=0.85,
                   reason=f"{total - len(silent)} of {total} corners reporting",
                   detail=detail)


# ---------------------------------------------------------------------------
# The definitions
# ---------------------------------------------------------------------------

def _heal(monitor_id: str) -> HealingCriteria:
    return HealingCriteria(
        required_passing_monitor_runs=config.TIRE_DIAG_HEAL_RUNS.get(monitor_id, 2),
        required_passing_drive_cycles=0,
        minimum_stable_duration_seconds=config.TIRE_DIAG_HEAL_STABLE_S.get(
            monitor_id, 300.0))


MONITORS: Tuple[MonitorDefinition, ...] = (
    MonitorDefinition(
        monitor_id="tire.low_pressure",
        component_type="tire",
        required_inputs=("pressure", "target_pressure"),
        enabling=EnablingConditions(minimum_valid_samples=2),
        inhibiting_conditions=("receiver-wide outage",
                               "sensor identity unknown or relearning",
                               "no trusted samples in this epoch"),
        pending_criteria={"below": "target - TIRE_PRESSURE_WARN_DELTA",
                          "consecutive_reports": 2},
        confirmed_criteria={"qualifying_runs":
                            config.TIRE_DIAG_CONFIRM_RUNS["tire.low_pressure"]},
        healing=_heal("tire.low_pressure"),
        freeze_frame_fields=C.FF_PRESSURE,
        evaluate=_eval_low_pressure,
        default_severity=C.WARNING),

    MonitorDefinition(
        monitor_id="tire.critical_low_pressure",
        component_type="tire",
        required_inputs=("pressure", "target_pressure"),
        enabling=EnablingConditions(minimum_valid_samples=2),
        inhibiting_conditions=("receiver-wide outage",
                               "sensor identity unknown or relearning"),
        pending_criteria={"below": "max(target - CRITICAL_DELTA, CRITICAL_FLOOR)",
                          "consecutive_reports": 2},
        confirmed_criteria={"qualifying_runs":
                            config.TIRE_DIAG_CONFIRM_RUNS["tire.critical_low_pressure"]},
        urgent_criteria={"moving": True, "validated_samples": 2,
                         "falling_psi_between_reports": config.TIRE_DIAG_FALLING_PSI,
                         "rejects": ["single malformed packet", "impossible step",
                                     "receiver-wide loss", "wake-up frame",
                                     "unknown tire identity", "stale data"]},
        healing=_heal("tire.critical_low_pressure"),
        freeze_frame_fields=C.FF_PRESSURE,
        evaluate=_eval_critical_pressure,
        default_severity=C.CRITICAL),

    MonitorDefinition(
        monitor_id="tire.slow_leak",
        component_type="tire",
        required_inputs=("pressure", "temperature", "peer_pressure"),
        enabling=EnablingConditions(
            minimum_valid_samples=config.TIRE_DIAG_LEAK_MIN_SAMPLES,
            minimum_elapsed_seconds=config.TIRE_DIAG_LEAK_WINDOW_S,
            require_comparable_thermal_state=True),
        inhibiting_conditions=("no thermally comparable pair in the window",
                               "no comparable peer readings to attribute against",
                               "receiver-wide outage",
                               "relearn epoch too young"),
        pending_criteria={"decline_psi": config.TIRE_DIAG_LEAK_PSI,
                          "excess_over_peers_psi": config.TIRE_DIAG_LEAK_PEER_MARGIN_PSI,
                          "across": "thermally comparable readings only"},
        confirmed_criteria={"qualifying_runs":
                            config.TIRE_DIAG_CONFIRM_RUNS["tire.slow_leak"],
                            "drive_cycles":
                            config.TIRE_DIAG_CONFIRM_CYCLES.get("tire.slow_leak", 0)},
        healing=_heal("tire.slow_leak"),
        freeze_frame_fields=C.FF_PRESSURE,
        evaluate=_eval_slow_leak,
        default_severity=C.ADVISORY),

    MonitorDefinition(
        monitor_id="tire.asymmetric_loss",
        component_type="tire",
        required_inputs=("pressure", "temperature", "peer_pressure"),
        enabling=EnablingConditions(
            minimum_valid_samples=config.TIRE_DIAG_ASYM_MIN_SAMPLES,
            minimum_elapsed_seconds=config.TIRE_DIAG_ASYM_WINDOW_S,
            require_comparable_thermal_state=True),
        inhibiting_conditions=("no comparable pair on this corner or its peer",
                               "receiver-wide outage"),
        pending_criteria={"gap_vs_axle_peer_psi": config.TIRE_DIAG_ASYM_MARGIN_PSI},
        confirmed_criteria={"qualifying_runs":
                            config.TIRE_DIAG_CONFIRM_RUNS["tire.asymmetric_loss"]},
        healing=_heal("tire.asymmetric_loss"),
        freeze_frame_fields=C.FF_PRESSURE,
        evaluate=_eval_asymmetric,
        default_severity=C.ADVISORY),

    MonitorDefinition(
        monitor_id="tpms.sensor_plausibility",
        component_type="tpms_sensor",
        required_inputs=("raw_reports",),
        enabling=EnablingConditions(minimum_valid_samples=0,
                                    require_receiver_healthy=True),
        inhibiting_conditions=("receiver-wide outage",),
        pending_criteria={"implausible_reports": config.TIRE_DIAG_IMPLAUSIBLE_COUNT,
                          "window_s": config.TIRE_DIAG_IMPLAUSIBLE_WINDOW_S},
        confirmed_criteria={"qualifying_runs":
                            config.TIRE_DIAG_CONFIRM_RUNS["tpms.sensor_plausibility"]},
        healing=_heal("tpms.sensor_plausibility"),
        freeze_frame_fields=C.FF_SENSOR,
        evaluate=_eval_plausibility,
        default_severity=C.ADVISORY),

    MonitorDefinition(
        monitor_id="tpms.sensor_connectivity",
        component_type="tpms_sensor",
        required_inputs=("report_times", "battery"),
        enabling=EnablingConditions(minimum_valid_samples=0, require_moving=True),
        inhibiting_conditions=("the car is parked and the sensors are asleep",
                               "receiver-wide outage"),
        pending_criteria={"silent_for_s": config.TIRE_DIAG_MISSED_REPORT_S,
                          "or_battery_pct_at_or_below": config.TIRE_BATTERY_LOW_PCT},
        confirmed_criteria={"qualifying_runs":
                            config.TIRE_DIAG_CONFIRM_RUNS["tpms.sensor_connectivity"]},
        healing=_heal("tpms.sensor_connectivity"),
        freeze_frame_fields=C.FF_SENSOR,
        evaluate=_eval_connectivity,
        default_severity=C.INFORMATIONAL),

    MonitorDefinition(
        monitor_id="tire.sensor_loss_during_decline",
        component_type="tire",
        required_inputs=("report_times", "active_issues"),
        enabling=EnablingConditions(minimum_valid_samples=0),
        inhibiting_conditions=("no active decline on this corner",
                               "receiver-wide outage"),
        pending_criteria={"connectivity_lost_on_corner_with_active_decline": True},
        confirmed_criteria={"qualifying_runs": 1, "one_trip": True},
        urgent_criteria={"moving": True,
                         "active_decline_issue": True,
                         "rejects": ["receiver-wide loss", "parked sensors asleep"]},
        healing=_heal("tire.sensor_loss_during_decline"),
        freeze_frame_fields=C.FF_SENSOR,
        evaluate=_eval_loss_during_decline,
        default_severity=C.CRITICAL),

    MonitorDefinition(
        monitor_id="tire.inflation_event",
        component_type="tire",
        required_inputs=("pressure",),
        enabling=EnablingConditions(minimum_valid_samples=2, require_parked=True),
        inhibiting_conditions=("the car is moving",),
        pending_criteria={"step_up_psi": config.TIRE_DIAG_INFLATION_STEP_PSI},
        confirmed_criteria={"qualifying_runs": 1},
        healing=HealingCriteria(required_passing_monitor_runs=1),
        freeze_frame_fields=C.FF_PRESSURE,
        evaluate=_eval_inflation,
        default_severity=C.INFORMATIONAL),

    MonitorDefinition(
        monitor_id="tpms.receiver_health",
        component_type="tpms_system",
        required_inputs=("report_times",),
        enabling=EnablingConditions(require_receiver_healthy=False),
        inhibiting_conditions=(),
        pending_criteria={"all_corners_silent_for_s":
                          config.TIRE_DIAG_MISSED_REPORT_S},
        confirmed_criteria={"qualifying_runs":
                            config.TIRE_DIAG_CONFIRM_RUNS["tpms.receiver_health"]},
        healing=_heal("tpms.receiver_health"),
        freeze_frame_fields=C.FF_SYSTEM,
        evaluate=_eval_receiver,
        default_severity=C.INFORMATIONAL,
        per_corner=False),
)

BY_ID: Dict[str, MonitorDefinition] = {m.monitor_id: m for m in MONITORS}


def run(d: MonitorDefinition, inp: MonitorInput) -> Outcome:
    """Gate, then evaluate. The only way a monitor is ever run.

    Nothing calls `evaluate` directly — the gate is where "could this run"
    lives, and a monitor that could be evaluated around it would be a monitor
    whose enabling conditions are advisory.
    """
    gated = _gate(inp, d)
    if gated is not None:
        return gated
    try:
        return d.evaluate(inp)
    except Exception as e:
        # A monitor that throws is a monitor that could not judge. It must not
        # take the others down, and it must NOT be recorded as a pass.
        return Outcome(DATA_UNAVAILABLE,
                       reason=f"monitor error: {type(e).__name__}: {e}")


def definitions_view() -> List[dict]:
    """Every monitor, fully described. For the service view and the tests."""
    out = []
    for m in MONITORS:
        e = m.enabling
        out.append({
            "monitor_id": m.monitor_id,
            "component_type": m.component_type,
            "required_inputs": list(m.required_inputs),
            "enabling_conditions": {
                "minimum_vehicle_speed_mph": e.minimum_vehicle_speed_mph,
                "maximum_vehicle_speed_mph": e.maximum_vehicle_speed_mph,
                "minimum_valid_samples": e.minimum_valid_samples,
                "minimum_elapsed_seconds": e.minimum_elapsed_seconds,
                "require_comparable_thermal_state": e.require_comparable_thermal_state,
                "require_stable_sensor_identity": e.require_stable_sensor_identity,
                "require_receiver_healthy": e.require_receiver_healthy,
                "require_moving": e.require_moving,
                "require_parked": e.require_parked,
            },
            "inhibiting_conditions": list(m.inhibiting_conditions),
            "pending_criteria": dict(m.pending_criteria),
            "confirmed_criteria": dict(m.confirmed_criteria),
            "urgent_criteria": dict(m.urgent_criteria) if m.urgent_criteria else None,
            "healing_criteria": {
                "required_passing_monitor_runs": m.healing.required_passing_monitor_runs,
                "required_passing_drive_cycles": m.healing.required_passing_drive_cycles,
                "minimum_stable_duration_seconds": m.healing.minimum_stable_duration_seconds,
            },
            "freeze_frame_fields": list(m.freeze_frame_fields),
            "default_severity": m.default_severity,
            "per_corner": m.per_corner,
            "shadow_mode": bool(config.TIRE_DIAG_SHADOW_MODE or m.shadow_default),
        })
    return out
