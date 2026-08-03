"""monitors.py — the monitor contract, and the gate every monitor runs behind.

Lifted from tire_diag/monitors.py. The evaluation functions stayed behind, in
the domains; what moved here is the part that was never about tires.

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
driver everything is fine when what it means is that it has not looked.

    "I do not have enough comparable readings yet to evaluate a slow leak, but
     the current pressure is not critically low."

That sentence needs both fields. One enum cannot say it. The same sentence, in
the powertrain domain, is "I have not seen a cold start yet, so I cannot judge
the starting voltage, but the running voltage is where it should be" — which is
why this contract is here and not in either domain.

WHAT IS AND IS NOT IN THIS FILE
-------------------------------
In: the status/result vocabulary, the dataclasses a monitor is described and fed
with, the enabling/inhibiting gate, and `run`. All pure — given the same input
they return the same outcome, with no clock of their own and no I/O.

Out: counting runs, promoting CANDIDATE to ACTIVE, healing, freeze frames,
persistence, and every decision about speech. Those are runner.py's, because
they are stateful and this file is not. Also out: every pass/fail criterion,
which belongs to the domain that knows what it is measuring.

THE GATE IS THE POINT
---------------------
`run` is the only way a monitor is ever evaluated, and it applies `gate` first.
Nothing calls `evaluate` directly — a monitor that could be evaluated around the
gate would be a monitor whose enabling conditions are advisory.

LLM FIREWALL
------------
This module imports the stdlib and nothing else. Not even config: the tunables a
gate needs travel on the MonitorInput, put there by the domain's engine. There is
nothing here to read a model output with.
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Callable, Dict, List, Optional, Tuple

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
    """One reading from one subject, after the domain's plausibility gate.

    `valid` is False for a sample the gate rejected. Rejected samples are KEPT
    rather than dropped: how often a source talks nonsense is itself the
    evidence a plausibility monitor runs on, and a list with the bad ones deleted
    cannot support that judgement.

    `connected` and `valid` are different facts and are both here. "The sensor
    did not answer" and "the sensor answered nonsense" call for different
    remedies, and a system that conflated them would tell somebody to replace a
    sensor whose wire had fallen off.
    """
    subject: str
    at: float
    connected: bool = True
    valid: bool = True
    reject_reason: str = ""

    def has_primary(self) -> bool:
        """Does this sample carry the measurement its domain is mainly about?

        Overridden per domain. The gate uses it to tell "the source answered"
        from "the source answered with the one number we needed missing", which
        is DATA_UNAVAILABLE rather than a pass.
        """
        return True


@dataclass
class MonitorInput:
    """Everything a monitor may look at. Nothing else is in scope.

    The vehicle-level facts (`moving`, `speed_mph`) are here rather than in a
    domain subclass because they are facts about the car, not about the
    subsystem: a tire monitor and a start-voltage monitor both need to know
    whether the wheels are turning, and two domains deriving that separately is
    two answers to one question.

    The three string fields at the bottom exist so the GATE can produce a reason
    in the domain's own words without knowing which domain it is in. A monitor's
    status has to say what it is actually waiting for; "the subsystem is not
    reporting" is useless where "the receiver is not reporting at all" is the
    fact.
    """
    subject: Optional[str]
    now: float
    samples: List[Sample]                      # this subject, oldest first
    moving: bool = False
    speed_mph: Optional[float] = None
    system_healthy: bool = True
    epoch_started_at: float = 0.0              # last relearn/reset for this subject
    drive_cycle_id: Optional[str] = None
    # Issues already ACTIVE on this subject, by monitor id. For the monitors
    # whose question is "was this already in trouble when it went quiet".
    active_monitor_ids: Tuple[str, ...] = ()
    # How old a sample may be and still be evidence. A domain tunable, carried
    # here so the gate and valid_samples() need not reach for config.
    sample_max_age_s: float = 150.0
    no_subject_reason: str = "nothing has ever reported on this subject"
    system_unhealthy_reason: str = "the subsystem is not reporting at all"


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
    require_stable_source_identity: bool = True
    require_system_healthy: bool = True
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
    component_type: str
    required_inputs: Tuple[str, ...]
    enabling: EnablingConditions
    inhibiting_conditions: Tuple[str, ...]
    pending_criteria: Dict[str, object]
    confirmed_criteria: Dict[str, object]
    healing: HealingCriteria
    freeze_frame_fields: Tuple[str, ...]
    evaluate: Callable[[MonitorInput], Outcome]
    default_severity: str = "warning"
    urgent_criteria: Optional[Dict[str, object]] = None
    per_subject: bool = True
    # Whether the gate should insist on a fresh sample carrying the domain's
    # primary measurement. Replaces tire_diag's `"pressure" in required_inputs`
    # test, which was a string comparison standing in for a boolean.
    requires_primary: bool = False
    # Shadow mode is a property of the DEPLOYMENT, not of the monitor, so it is
    # read at decision time rather than frozen in here. This field records only
    # whether the monitor has ever been cleared to speak at all.
    shadow_default: bool = True


# ---------------------------------------------------------------------------
# Shared helpers
# ---------------------------------------------------------------------------

def valid_samples(inp: MonitorInput, max_age_s: float = None) -> List[Sample]:
    """Samples this monitor may use: plausible, connected, present and recent.

    Age is measured against the sample, not against the poll — a source that is
    being asked four times a second does not make its last report newer.
    """
    max_age = inp.sample_max_age_s if max_age_s is None else max_age_s
    return [s for s in inp.samples
            if s.valid and s.connected and s.has_primary()
            and s.at >= inp.epoch_started_at
            and (inp.now - s.at) <= max_age]


def confidence(n_samples: int, margin: float, scale: float) -> float:
    """0..1, from how much evidence there is and how far past the line it went.

    Deliberately blunt. A confidence that looks precise invites being read as a
    probability, and this is not one — it is "how much would I stake on this",
    used to order findings and to decide nothing on its own.
    """
    ev = min(1.0, n_samples / 6.0)
    mag = 1.0 - math.exp(-max(0.0, margin) / max(1e-6, scale))
    return round(0.35 + 0.4 * ev + 0.25 * mag, 3)


def gate(inp: MonitorInput, d: MonitorDefinition) -> Optional[Outcome]:
    """The enabling and inhibiting conditions, applied in one place.

    Order is deliberate and is the order of what makes an answer impossible
    first: no source at all, then a whole-subsystem outage, then no data, then a
    condition that forbids the comparison, then not enough evidence. A monitor's
    status should name the FIRST reason it could not judge, not whichever check
    happened to run last.
    """
    e = d.enabling

    if inp.subject is not None and not inp.samples:
        return Outcome(NOT_SUPPORTED, reason=inp.no_subject_reason)

    if e.require_system_healthy and not inp.system_healthy:
        # Every subject silent at once is one subsystem fault. Running the
        # per-subject monitors against it would manufacture N faults out of one
        # — and that is a real failure mode, not a hypothetical: it is exactly
        # what an unguarded monitor does during a receiver outage.
        return Outcome(INHIBITED, reason=inp.system_unhealthy_reason)

    latest = inp.samples[-1] if inp.samples else None
    if d.requires_primary:
        if latest is None or not latest.connected:
            return Outcome(DATA_UNAVAILABLE, reason="the sensor is not reporting")
        if (inp.now - latest.at) > inp.sample_max_age_s:
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
    if d.requires_primary and len(usable) < e.minimum_valid_samples:
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


def run(d: MonitorDefinition, inp: MonitorInput) -> Outcome:
    """Gate, then evaluate. The only way a monitor is ever run."""
    gated = gate(inp, d)
    if gated is not None:
        return gated
    try:
        return d.evaluate(inp)
    except Exception as e:
        # A monitor that throws is a monitor that could not judge. It must not
        # take the others down, and it must NOT be recorded as a pass.
        return Outcome(DATA_UNAVAILABLE,
                       reason=f"monitor error: {type(e).__name__}: {e}")


def definitions_view(monitors: Tuple[MonitorDefinition, ...],
                     shadowed: bool,
                     confirm_runs: Dict[str, int] = None,
                     confirm_cycles: Dict[str, int] = None) -> List[dict]:
    """Every monitor, fully described. For the service view and the tests."""
    confirm_runs = confirm_runs or {}
    confirm_cycles = confirm_cycles or {}
    out = []
    for m in monitors:
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
                "require_stable_source_identity": e.require_stable_source_identity,
                "require_system_healthy": e.require_system_healthy,
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
            "confirm_runs_required": confirm_runs.get(m.monitor_id, 2),
            "confirm_cycles_required": confirm_cycles.get(m.monitor_id, 0),
            "freeze_frame_fields": list(m.freeze_frame_fields),
            "default_severity": m.default_severity,
            "per_subject": m.per_subject,
            "shadow_mode": bool(shadowed or m.shadow_default),
        })
    return out
