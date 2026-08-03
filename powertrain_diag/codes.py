"""codes.py — RIO's internal engine diagnostic identifiers.

These are RIO diagnostic codes, exactly as tire_diag's are. They are NOT SAE
DTCs, they are not P-codes, and they must never be presented as one — which is a
sharper distinction here than it was for tires, because this domain sits right
next to a layer that handles genuine P-codes.

    RIO-ENGINE-COOLANT-OVER-LIMIT        RIO observed this
    P0217                                the vehicle reported this

Both can be about the same overheat, they are different claims, and §17.8 puts
them under different headings for exactly that reason. `vehicle/dtc/` owns the
second kind. This file owns the first.

WHAT THE TWO KINDS ARE FOR
--------------------------
A P-code means the ECU's own monitor tripped. A RIO code means RIO noticed
something the ECU either has no monitor for, or has not confirmed yet, or would
only report after the light comes on. The overlap is deliberate: RIO-ENGINE-
COOLANT-RISING has no equivalent P-code at all, and RIO-ENGINE-COOLANT-OVER-LIMIT
usually arrives before the ECU sets one.

SPEECH ELIGIBILITY IS OFF
-------------------------
Every code below ships with speak=False, and the domain's shadow flag
(config.VEHICLE_DIAG_SHADOW_MODE) is True. That is a stronger position than the
tire domain's: those monitors have shadow logs from real drives behind them, and
these have never seen a vehicle. Nothing here has a fast path either — there is
no engine condition in this set whose consequence is unrecoverable within the
seconds a fast path saves, and the two tire conditions that do have one earned it
by argument rather than by default.
"""
from __future__ import annotations

from typing import Dict, List, Optional

import config

from diag import shadow
from diag.codes import (ADVISORY, CRITICAL, INFORMATIONAL, SEVERITY_RANK,
                        WARNING, CodeCatalog, DiagnosticCode)

DOMAIN = "powertrain"

# This domain's speech clearance, declared once at import. A getter rather than
# a value so config stays the single source of truth — see diag/shadow.py.
shadow.register(DOMAIN, lambda: bool(config.VEHICLE_DIAG_SHADOW_MODE))

# One subject. Unlike the tires, which have four physically separate things that
# fail independently, these monitors are all about one engine — and inventing
# per-channel subjects would produce a service view of thirty rows describing
# nine questions.
SUBJECT = "engine"
SUBJECTS = (SUBJECT,)


# Freeze-frame field sets. Named rather than repeated, because "what evidence
# did we have at the moment we decided" is what a service view is for.
FF_THERMAL = ("coolant_temp", "coolant_rate_f_per_min", "oil_temp",
              "intake_air_temp", "vehicle_speed", "engine_load", "rpm",
              "engine_running", "baseline_coolant_temp", "sample_count",
              "first_sample_at", "last_sample_at", "data_quality",
              "drive_cycle_id", "monitor_runs",
              # The limit the reading was judged against, and how long it held.
              # A freeze frame with the measurement and not the threshold is
              # half a record: three weeks later nobody remembers what the
              # ceiling was that day, and the number alone cannot be argued
              # with.
              "limit_f", "limit_f_per_min", "peak_coolant_temp", "delta_f",
              "held_s", "required_hold_s", "baseline_days", "window_s")
FF_ELECTRICAL = ("battery_voltage", "cranking_voltage", "start_event_count",
                 "start_voltage_first", "start_voltage_last", "rpm",
                 "engine_running", "sample_count", "data_quality",
                 "drive_cycle_id", "monitor_runs",
                 "floor_v", "minimum_v", "decline_v", "required_decline_v",
                 "held_s", "required_hold_s")
FF_FUEL = ("ltft_b1", "stft_b1", "coolant_temp", "engine_load", "rpm",
           "map_kpa", "maf_gs", "afr_wideband", "baseline_ltft_b1",
           "sample_count", "data_quality", "drive_cycle_id", "monitor_runs",
           "ltft_mean", "warn_high", "warn_low", "held_s", "required_hold_s")
FF_SIGNAL = ("affected_channels", "sample_count", "first_sample_at",
             "last_sample_at", "data_quality", "engine_running",
             "drive_cycle_id", "monitor_runs")
FF_LINK = ("silent_for_s", "last_sample_at", "outbox_pending", "can_state",
           "network_state", "source", "drive_cycle_id", "monitor_runs")
FF_DTC = ("dtc_codes", "dtc_added", "mil_commanded_on", "dtc_count",
          "drive_cycle_id", "monitor_runs")


# (suffix, monitor, component, severity, driver term, technician text, action,
#  confirmation summary, healing summary, freeze frame)
_CODES = (
    ("DTC-REPORTED", "engine.new_dtc", "engine_ecu", WARNING,
     "a fault code from the engine computer",
     "One or more diagnostic trouble codes newly reported by the ECU, or an "
     "existing code changing status.",
     "Read the Flagged Error Codes section — the vehicle's own code is the "
     "authority here, not this finding.",
     "A single scan, because the ECU has already done the confirming.",
     "The reported set of codes is unchanged across two passing runs.",
     FF_DTC),

    ("COOLANT-OVER-LIMIT", "engine.coolant_hard_limit", "cooling", CRITICAL,
     "coolant temperature above its safe limit",
     "Coolant temperature at or above the configured critical ceiling, "
     "sustained, on a running engine with fresh data.",
     "Stop and let it cool before going any further.",
     "Sustained above the ceiling for the configured hold, across two runs.",
     "Back below the ceiling and held there across three passing runs.",
     FF_THERMAL),

    ("COOLANT-RISING", "engine.coolant_rate_of_rise", "cooling", WARNING,
     "coolant temperature climbing faster than it should",
     "Least-squares rate of rise across the window exceeding the configured "
     "limit, on a warm engine, excluding the warm-up phase.",
     "Worth stopping to check coolant level before it reaches the limit.",
     "Rate above the limit across two runs on warm, comparable data.",
     "Rate back inside the limit across three passing runs.",
     FF_THERMAL),

    ("COOLANT-ABOVE-BASELINE", "engine.coolant_contextual", "cooling", ADVISORY,
     "coolant running warmer than it normally does",
     "Conditioned coolant mean materially above this vehicle's own historical "
     "baseline for comparable conditions, while still inside every fixed band.",
     "Not urgent. Worth mentioning at the next service — a cooling system "
     "losing margin looks exactly like this first.",
     "Above the baseline delta across three runs and more than one drive.",
     "Back inside the baseline delta across three passing runs.",
     FF_THERMAL),

    ("CHARGING-LOW", "engine.charging_voltage", "electrical", WARNING,
     "a charging system that is not keeping up",
     "Control-module voltage below the configured floor with the engine "
     "running, sustained past the hold.",
     "Have the alternator and battery checked before the next cold start.",
     "Below the floor for the configured hold, across three runs.",
     "Back above the floor across three passing runs.",
     FF_ELECTRICAL),

    ("START-VOLTAGE-FALLING", "engine.start_voltage_trend", "electrical", ADVISORY,
     "a battery that is getting weaker every time it starts",
     "Cranking voltage minimum declining across the recorded start history by "
     "more than the configured margin. Running voltage may be entirely normal.",
     "Have the battery load-tested. This is the measurement that predicts a "
     "no-start, and the running voltage will not show it.",
     "A declining trend across the minimum number of recorded starts, twice.",
     "Cranking voltage recovered across two passing runs, i.e. a new battery.",
     FF_ELECTRICAL),

    ("FUEL-TRIM-HIGH", "engine.fuel_trim_long_term", "fuel", ADVISORY,
     "a long-term fuel trim that has moved away from where it sits",
     "Long-term fuel trim beyond the configured band on a warm engine in "
     "closed loop, sustained past the hold.",
     "Worth investigating for unmetered air or a fuelling shortfall before the "
     "ECU sets a code of its own.",
     "Beyond the band for the configured hold, across three runs and more "
     "than one drive.",
     "Back inside the band across three passing runs.",
     FF_FUEL),

    ("SIGNAL-SUSPECT", "engine.signal_integrity", "sensing", ADVISORY,
     "an engine sensor that is not reporting believably",
     "One or more channels frozen, discontinuous, out of plausible range, or "
     "reporting a decode error, on a running engine.",
     "The reading is suspect. Do not act on that channel until it is checked.",
     "Repeated across three runs on the same channel.",
     "Every affected channel reporting believably across three passing runs.",
     FF_SIGNAL),

    ("LINK-DEGRADED", "engine.connection", "link", INFORMATIONAL,
     "no engine data reaching RIO",
     "No usable engine channel within the configured window, or a gateway "
     "reporting a bus, network or outbox problem.",
     "Check the adapter and its connection. Nothing about the engine can be "
     "judged while this is true.",
     "No usable data past the window, across two runs.",
     "Data flowing again, sustained for the stable period.",
     FF_LINK),
)


def _build() -> Dict[str, DiagnosticCode]:
    out: Dict[str, DiagnosticCode] = {}
    for (suffix, monitor, component, severity, term, tech, action,
         conf, heal, ff) in _CODES:
        code = f"RIO-ENGINE-{suffix}"
        out[code] = DiagnosticCode(
            code=code, monitor_id=monitor, component_type=component,
            subject=SUBJECT, default_severity=severity,
            driver_term=term, technician_description=tech,
            suggested_action=action, confirmation_summary=conf,
            healing_summary=heal, freeze_frame_fields=ff,
            # Shadow mode, and no fast path anywhere in this domain. See the
            # module header on why that is a stronger position than the tire
            # domain's rather than an oversight.
            speak=False, fast_path=False,
            spoken_subject="")
    return out


CATALOG = CodeCatalog(_build())
CODES: Dict[str, DiagnosticCode] = CATALOG.codes


def code_for(monitor_id: str, subject: Optional[str] = None,
             variant: str = None) -> Optional[DiagnosticCode]:
    return CATALOG.code_for(monitor_id, subject or SUBJECT, variant)


def get(code: str) -> Optional[DiagnosticCode]:
    return CATALOG.get(code)


def speech_eligible(code: str) -> bool:
    return CATALOG.speech_eligible(code)


def fast_path_eligible(code: str) -> bool:
    return CATALOG.fast_path_eligible(code)


def service_view() -> List[dict]:
    return CATALOG.service_view()
