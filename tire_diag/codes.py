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

WHAT THIS FILE IS NOW
---------------------
The tire domain's CONTENT. The DiagnosticCode dataclass and the CodeCatalog that
holds it moved to diag/codes.py when the powertrain monitors needed the same
shape. What is left is the part that is genuinely about tires: nine per-corner
conditions, one system-level condition, and the words for each.

The module-level functions below delegate to the catalogue. They are kept
because vehicle_health.py and app.py already call them, and because a domain
reading `C.get(code)` should not have to know whether the catalogue is a module
or an object.

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
for ordinary confirmation. Only two conditions have it, and both are gated by
validation rules in monitors.py that a single bad packet cannot pass.
"""
from __future__ import annotations

from typing import Dict, List, Optional

import config

from diag import shadow
from diag.codes import (ADVISORY, CRITICAL, INFORMATIONAL, SEVERITY_RANK,
                        WARNING, CodeCatalog, DiagnosticCode)

DOMAIN = "tires"

# The tire domain's speech clearance, declared once at import. A getter rather
# than a value so config stays the single source of truth — see diag/shadow.py.
shadow.register(DOMAIN, lambda: bool(config.TIRE_DIAG_SHADOW_MODE))

CORNERS = ("FL", "FR", "RL", "RR")
CORNER_SPOKEN = {"FL": "front left", "FR": "front right",
                 "RL": "rear left", "RR": "rear right"}
CORNER_TECH = {"FL": "front-left", "FR": "front-right",
               "RL": "rear-left", "RR": "rear-right"}


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
                subject=corner, default_severity=severity,
                driver_term=term,
                technician_description=f"{CORNER_TECH[corner]}: {tech}",
                suggested_action=action, confirmation_summary=conf,
                healing_summary=heal, freeze_frame_fields=ff,
                speak=speak, fast_path=fast,
                spoken_subject=CORNER_SPOKEN[corner])
    for (code, monitor, component, severity, term, tech, action,
         conf, heal, ff, speak, fast) in _SYSTEM:
        out[code] = DiagnosticCode(
            code=code, monitor_id=monitor, component_type=component, subject=None,
            default_severity=severity, driver_term=term,
            technician_description=tech, suggested_action=action,
            confirmation_summary=conf, healing_summary=heal,
            freeze_frame_fields=ff, speak=speak, fast_path=fast)
    return out


CATALOG = CodeCatalog(_build())

# The dictionary the tests and the service view already read by name.
CODES: Dict[str, DiagnosticCode] = CATALOG.codes


def code_for(monitor_id: str, corner: Optional[str] = None,
             variant: str = None) -> Optional[DiagnosticCode]:
    """The code a monitor raises for a corner.

    `variant` picks between codes that share a monitor — tpms.sensor_connectivity
    raises both SENSOR-STALE and SENSOR-LOW-BATTERY, because "stopped talking"
    and "about to stop talking" are one monitor's two findings and two quite
    different things to tell somebody.
    """
    return CATALOG.code_for(monitor_id, corner, variant)


def get(code: str) -> Optional[DiagnosticCode]:
    return CATALOG.get(code)


def speech_eligible(code: str) -> bool:
    """May RIO announce this at all? False for everything, in shadow mode."""
    return CATALOG.speech_eligible(code)


def fast_path_eligible(code: str) -> bool:
    """May this take the urgent path, bypassing ordinary confirmation?

    Separate from speech_eligible and deliberately not implied by it. An urgent
    condition speaks even in shadow mode, because the whole argument for the
    fast path is that the consequence of waiting is not recoverable — but only
    after passing every validation gate in monitors.py, which no single bad
    packet can do.
    """
    return CATALOG.fast_path_eligible(code)


def service_view() -> List[dict]:
    """Every code and what it means. For a diagnostic or service view only —
    never for the conversation layer, which gets driver_term and nothing else."""
    rows = CATALOG.service_view()
    for row in rows:
        # The tire half of the codebase has always called a subject a corner.
        row["corner"] = row["subject"]
    return rows
