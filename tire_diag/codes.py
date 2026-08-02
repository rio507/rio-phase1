"""codes.py — RIO's internal tire diagnostic identifiers.

These are RIO diagnostic codes. They are NOT OBD-II DTCs, RIO Tire Health is not
an OBD-II system, and nothing here imitates or reuses an SAE powertrain P-code.
What is borrowed is the idea that a diagnostic condition deserves a stable
identifier that a technician, a test and a log can all agree on — which a
free-text message never gives you.

    RIO-TIRE-LOW-PRESSURE-RL          not P0300, and never presented as one

Readable and namespaced on purpose: RIO-<system>-<condition>-<location>. A code
in a log should be legible without a lookup table, which is the one thing the
P-code space is worst at.

WHAT A CODE OWNS
----------------
Everything that is true of the condition regardless of which tire has it: which
monitor raises it, what it is about, how bad it is by default, what to call it
to a driver, what to call it to a technician, what to do about it, what has to
be true to confirm it, what has to be true to heal it, what evidence to freeze,
and — separately from all of that — whether RIO is allowed to say it out loud.

WHAT A CODE IS NOT
------------------
Something the driver ever sees. RIO says "your rear-left tire may have a slow
leak", never "RIO-TIRE-POSSIBLE-LEAK-RL is active". The code exists for the
service view, the log and the tests. `driver_term` is the only field in here
that is ever allowed near a sentence spoken to a person.

SPEECH ELIGIBILITY IS OFF
-------------------------
Every code below ships with speak=False. That is shadow mode and it is
deliberate: these monitors have never seen a real drive, and the tuning data
that would justify letting one interrupt a driver does not exist yet. The engine
still detects, confirms, freezes evidence and records the announcement it WOULD
have made. Turning one on is a one-line edit here, after reading those logs.

`fast_path` is the separate, narrower permission: an urgent condition that
speaks immediately through the pre-rendered clip mechanism rather than waiting
for ordinary confirmation. Only two codes have it, and both are gated by
validation rules in monitors.py that a single bad packet cannot pass.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

# Severity, in the vocabulary the conversation layer already speaks. `advisory`
# is new here and sits between informational and warning: the spec asks for it,
# and it is the level for "this is real, it is on the dashboard, and it belongs
# in a drive-start briefing rather than in the middle of one".
INFORMATIONAL = "informational"
ADVISORY = "advisory"
WARNING = "warning"
CRITICAL = "critical"

SEVERITY_RANK = {INFORMATIONAL: 1, ADVISORY: 2, WARNING: 3, CRITICAL: 4}

CORNERS = ("FL", "FR", "RL", "RR")
CORNER_SPOKEN = {"FL": "front left", "FR": "front right",
                 "RL": "rear left", "RR": "rear right"}
CORNER_TECH = {"FL": "front-left", "FR": "front-right",
               "RL": "rear-left", "RR": "rear-right"}


@dataclass(frozen=True)
class DiagnosticCode:
    """One diagnostic condition, everything that is true of it everywhere.

    Per-corner codes are generated from a template (see CODES below) so that
    four ids cannot drift apart in wording — a typo in one corner's suggested
    action is a wrong instruction that only one quarter of drivers ever see.
    """

    code: str
    monitor_id: str
    component_type: str                 # tire | tpms_sensor | tpms_system
    corner: Optional[str]               # None for system-level codes
    default_severity: str

    # What a person is told. `driver_term` is a noun phrase, not a sentence:
    # the sentence is assembled where the tire is known.
    driver_term: str
    technician_description: str
    suggested_action: str

    # Confirmation and healing live on the MONITOR, which owns the evidence.
    # These are the code's summary of them, for the service view and for tests
    # that want to assert the contract without reading the monitor.
    confirmation_summary: str
    healing_summary: str

    freeze_frame_fields: Tuple[str, ...]

    # The two permissions, deliberately separate. `speak` is ordinary
    # announcement eligibility and is False on everything (shadow mode).
    # `fast_path` is the urgent exception and is narrower still.
    speak: bool = False
    fast_path: bool = False

    def spoken_location(self) -> str:
        return CORNER_SPOKEN.get(self.corner or "", "")


# Freeze-frame field sets. Named rather than repeated, because "what evidence
# did we have at the moment we decided" is the thing a service view is actually
# for, and three codes disagreeing about it makes the log unreadable.
FF_PRESSURE = ("vehicle_speed_mph", "current_pressure_psi", "target_pressure_psi",
               "temperature_f", "peer_average_pressure_psi", "pressure_change_psi",
               "valid_sample_count", "first_valid_sample_at", "last_valid_sample_at",
               "data_quality", "sensor_battery", "receiver_status",
               "drive_cycle_id", "monitor_runs")
FF_SENSOR = ("vehicle_speed_mph", "last_valid_sample_at", "expected_reports",
             "received_reports", "missed_reports", "data_quality",
             "sensor_battery", "receiver_status", "drive_cycle_id",
             "monitor_runs", "last_known_pressure_psi")
FF_SYSTEM = ("vehicle_speed_mph", "corners_silent", "last_valid_sample_at",
             "receiver_status", "drive_cycle_id", "monitor_runs")


# --- per-corner code templates ---------------------------------------------
# (suffix, monitor, component, severity, driver term, technician text, action,
#  confirmation summary, healing summary, freeze frame, speak, fast_path)
_PER_CORNER = (
    ("LOW-PRESSURE", "tire.low_pressure", "tire", WARNING,
     "low tire pressure",
     "Measured pressure below the placard target by more than the configured "
     "warning delta, on repeated valid samples.",
     "Add air to the placard pressure when the tires are cold.",
     "Two qualifying monitor runs with valid, thermally comparable samples.",
     "Pressure back above the warning threshold plus hysteresis, held across "
     "two passing runs.",
     FF_PRESSURE, False, False),

    ("CRITICAL-PRESSURE", "tire.critical_low_pressure", "tire", CRITICAL,
     "critically low tire pressure",
     "Measured pressure below the critical delta or below the absolute floor, "
     "on validated samples.",
     "Stop driving on it. Inflate to the placard pressure or fit the spare.",
     "Two validated samples below the critical threshold; one-trip urgent path "
     "when moving and confirmed falling.",
     "Pressure back above the warning threshold plus hysteresis across three "
     "passing runs and a comparable period.",
     FF_PRESSURE, False, True),

    ("POSSIBLE-LEAK", "tire.slow_leak", "tire", ADVISORY,
     "a possible slow leak",
     "Sustained decline across thermally comparable samples, exceeding the "
     "peer decline by the configured margin.",
     "Have it checked for a nail or a leaking valve in the next day or two.",
     "Decline confirmed across comparable readings spanning the configured "
     "duration, with peers materially more stable.",
     "An inflation event, or a stable pressure across the full comparable "
     "window and two passing runs.",
     FF_PRESSURE, False, False),

    ("ASYMMETRIC-LOSS", "tire.asymmetric_loss", "tire", ADVISORY,
     "one tire losing pressure faster than the others",
     "One corner declining materially faster than its axle peer over the same "
     "comparable window.",
     "Worth checking that corner before the next long drive.",
     "Two qualifying runs with the peer difference beyond the margin.",
     "Peer difference back inside the margin across two passing runs.",
     FF_PRESSURE, False, False),

    ("SENSOR-STALE", "tpms.sensor_connectivity", "tpms_sensor", INFORMATIONAL,
     "a tire sensor that has stopped reporting",
     "Expected transmissions missed beyond the configured tolerance while the "
     "wheel was turning.",
     "Usually the sensor battery. Have it read at the next service.",
     "Missed reports beyond tolerance across two runs while moving.",
     "Reports resumed and sustained for the configured stable period.",
     FF_SENSOR, False, False),

    ("SENSOR-FAULT", "tpms.sensor_plausibility", "tpms_sensor", ADVISORY,
     "a tire sensor reporting implausible values",
     "Repeated samples outside the plausible range, or steps larger than any "
     "physical process, with no supporting evidence from peers.",
     "The sensor is suspect. Have it replaced or re-paired.",
     "Repeated implausible samples across two monitor runs.",
     "Plausible samples sustained across three passing runs.",
     FF_SENSOR, False, False),

    ("SENSOR-LOW-BATTERY", "tpms.sensor_connectivity", "tpms_sensor", INFORMATIONAL,
     "a tire sensor low on battery",
     "Sensor-reported cell charge at or below the configured threshold.",
     "Book the sensor battery for replacement — no hurry.",
     "Reported on two consecutive valid samples.",
     "Reported charge back above the threshold (i.e. the sensor was replaced).",
     FF_SENSOR, False, False),

    ("SENSOR-LOSS-DURING-DECLINE", "tire.sensor_loss_during_decline", "tire", CRITICAL,
     "a tire that was losing pressure and has now gone silent",
     "Connectivity lost on a corner carrying a validated active decline. The "
     "last known state of the tire was deteriorating and it can no longer be "
     "observed.",
     "Pull over and check that tire by hand — it cannot be monitored.",
     "Loss of a corner with an ACTIVE decline issue while moving. One-trip.",
     "Reports resumed AND pressure verified stable across two passing runs.",
     FF_SENSOR, False, True),

    ("INFLATION-EVENT", "tire.inflation_event", "tire", INFORMATIONAL,
     "air added to a tire",
     "Step increase in pressure consistent with manual inflation, while "
     "stationary.",
     "None. Recorded so pressure issues on this corner can be verified as "
     "repaired.",
     "A single validated step above the configured size while parked.",
     "Not applicable — an inflation event is an observation, not a fault.",
     FF_PRESSURE, False, False),
)

# --- system-level codes -----------------------------------------------------
_SYSTEM = (
    ("RIO-TPMS-RECEIVER-UNAVAILABLE", "tpms.receiver_health", "tpms_system",
     INFORMATIONAL,
     "no tire sensor data at all",
     "All corners silent simultaneously. Diagnosed as one receiver fault "
     "rather than four sensor faults.",
     "Check the receiver and its power before suspecting the sensors.",
     "All four corners silent past the tolerance in the same monitor run.",
     "Any corner reporting again, sustained for the stable period.",
     FF_SYSTEM, False, False),
)


def _build() -> Dict[str, DiagnosticCode]:
    out: Dict[str, DiagnosticCode] = {}
    for (suffix, monitor, component, severity, term, tech, action,
         conf, heal, ff, speak, fast) in _PER_CORNER:
        for corner in CORNERS:
            code = f"RIO-{'TPMS' if component.startswith('tpms') else 'TIRE'}-{suffix}-{corner}"
            out[code] = DiagnosticCode(
                code=code, monitor_id=monitor, component_type=component,
                corner=corner, default_severity=severity,
                driver_term=term,
                technician_description=f"{CORNER_TECH[corner]}: {tech}",
                suggested_action=action, confirmation_summary=conf,
                healing_summary=heal, freeze_frame_fields=ff,
                speak=speak, fast_path=fast)
    for (code, monitor, component, severity, term, tech, action,
         conf, heal, ff, speak, fast) in _SYSTEM:
        out[code] = DiagnosticCode(
            code=code, monitor_id=monitor, component_type=component, corner=None,
            default_severity=severity, driver_term=term,
            technician_description=tech, suggested_action=action,
            confirmation_summary=conf, healing_summary=heal,
            freeze_frame_fields=ff, speak=speak, fast_path=fast)
    return out


CODES: Dict[str, DiagnosticCode] = _build()

# monitor_id + corner -> code. The engine works in monitors and corners; the
# code is what it writes down.
_BY_MONITOR: Dict[Tuple[str, Optional[str]], List[str]] = {}
for _c in CODES.values():
    _BY_MONITOR.setdefault((_c.monitor_id, _c.corner), []).append(_c.code)


def code_for(monitor_id: str, corner: Optional[str] = None,
             variant: str = None) -> Optional[DiagnosticCode]:
    """The code a monitor raises for a corner.

    `variant` picks between codes that share a monitor — tpms.sensor_connectivity
    raises both SENSOR-STALE and SENSOR-LOW-BATTERY, because "stopped talking"
    and "about to stop talking" are one monitor's two findings and two quite
    different things to tell somebody.
    """
    candidates = _BY_MONITOR.get((monitor_id, corner)) or []
    if not candidates:
        return None
    if variant:
        for c in candidates:
            if variant.upper() in c:
                return CODES[c]
    return CODES[sorted(candidates)[0]]


def get(code: str) -> Optional[DiagnosticCode]:
    return CODES.get(code)


def speech_eligible(code: str) -> bool:
    """May RIO announce this at all? False for everything, in shadow mode."""
    c = CODES.get(code)
    return bool(c and c.speak)


def fast_path_eligible(code: str) -> bool:
    """May this take the urgent path, bypassing ordinary confirmation?

    Separate from speech_eligible and deliberately not implied by it. An urgent
    condition speaks even in shadow mode, because the whole argument for the
    fast path is that the consequence of waiting is not recoverable — but only
    after passing every validation gate in monitors.py, which no single bad
    packet can do.
    """
    c = CODES.get(code)
    return bool(c and c.fast_path)


def service_view() -> List[dict]:
    """Every code and what it means. For a diagnostic or service view only —
    never for the conversation layer, which gets driver_term and nothing else."""
    return [{
        "code": c.code,
        "monitor": c.monitor_id,
        "component": c.component_type,
        "corner": c.corner,
        "default_severity": c.default_severity,
        "driver_term": c.driver_term,
        "technician_description": c.technician_description,
        "suggested_action": c.suggested_action,
        "confirmation": c.confirmation_summary,
        "healing": c.healing_summary,
        "freeze_frame_fields": list(c.freeze_frame_fields),
        "speech_eligible": c.speak,
        "fast_path_eligible": c.fast_path,
    } for c in sorted(CODES.values(), key=lambda x: x.code)]
