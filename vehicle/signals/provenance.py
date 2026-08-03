"""provenance.py — where a claim came from, travelling with the claim.

Every measurement, every diagnostic trouble code, every finding and every
sentence in a report carries one of these. It is not metadata in the decorative
sense: it is the difference between "your ECU has recorded a lean-condition
fault" and "RIO thinks your fuel trim looks high", and a driver who cannot tell
those apart has been told something false by a system that only said true things.

THE DISTINCTION THAT MATTERS MOST
---------------------------------
    ecu_reported_*        the vehicle said this
    rio_*                 RIO inferred this

Everything else is detail. A dashboard that merges the two, or a report that
prints them in the same typeface, has thrown away the only fact that tells a
mechanic which half to trust and which half to check.

WHY A CAUSE IS NEVER A PROVENANCE
---------------------------------
There is no `confirmed_cause` value in this list and there must not be one. A
DTC names a CONDITION the ECU observed, not the component that failed; P0171 is
"the mixture is lean", not "the MAF is bad". `mechanic_confirmed` and
`repair_validated` exist precisely because a cause only becomes a fact when
somebody outside this system establishes it.
"""
from __future__ import annotations

# --- the vehicle said it ---------------------------------------------------
ECU_MEASUREMENT = "ecu_measurement"
ECU_PENDING_FAULT = "ecu_reported_pending_fault"
ECU_CONFIRMED_FAULT = "ecu_reported_confirmed_fault"
ECU_PERMANENT_FAULT = "ecu_reported_permanent_fault"
ECU_FREEZE_FRAME = "ecu_freeze_frame"

# --- RIO worked it out -----------------------------------------------------
RIO_FIXED_RULE = "rio_fixed_rule"
RIO_OBSERVED_PATTERN = "rio_observed_pattern"
RIO_BASELINE_DEVIATION = "rio_baseline_deviation"
RIO_CROSS_SIGNAL = "rio_cross_signal_interpretation"

# --- a person said it ------------------------------------------------------
USER_REPORTED = "user_reported"
MECHANIC_CONFIRMED = "mechanic_confirmed"
REPAIR_VALIDATED = "repair_validated"

# --- it did not come from a vehicle at all ---------------------------------
SIMULATION = "simulation"
RECORDED_REPLAY = "recorded_replay"

ALL = (
    ECU_MEASUREMENT, ECU_PENDING_FAULT, ECU_CONFIRMED_FAULT,
    ECU_PERMANENT_FAULT, ECU_FREEZE_FRAME,
    RIO_FIXED_RULE, RIO_OBSERVED_PATTERN, RIO_BASELINE_DEVIATION,
    RIO_CROSS_SIGNAL,
    USER_REPORTED, MECHANIC_CONFIRMED, REPAIR_VALIDATED,
    SIMULATION, RECORDED_REPLAY,
)

# The ones that mean "the vehicle's own electronics reported this". Used by the
# dashboard and the report to decide which side of the ECU-reported /
# RIO-observed line a row belongs on.
ECU_REPORTED = (ECU_MEASUREMENT, ECU_PENDING_FAULT, ECU_CONFIRMED_FAULT,
                ECU_PERMANENT_FAULT, ECU_FREEZE_FRAME)

# The ones that are RIO's interpretation. Never presented as a vehicle fault.
RIO_OBSERVED = (RIO_FIXED_RULE, RIO_OBSERVED_PATTERN, RIO_BASELINE_DEVIATION,
                RIO_CROSS_SIGNAL)

# The ones that did not come from a vehicle. A finding resting on these must say
# so wherever it is displayed — a demo that presents fabricated data as measured
# is the one failure this layer cannot have, and insights.py already makes the
# same promise about its seeded history.
NOT_MEASURED = (SIMULATION, RECORDED_REPLAY)

# How each one is written when a person reads it. §17.8's two labels are the
# first two; the rest exist so a report never has to print a snake_case token.
DISPLAY = {
    ECU_MEASUREMENT: "Reported by Vehicle ECU",
    ECU_PENDING_FAULT: "Reported by Vehicle ECU — pending fault",
    ECU_CONFIRMED_FAULT: "Reported by Vehicle ECU — confirmed fault",
    ECU_PERMANENT_FAULT: "Reported by Vehicle ECU — permanent fault",
    ECU_FREEZE_FRAME: "Recorded by Vehicle ECU when the fault was set",
    RIO_FIXED_RULE: "Observed by RIO — fixed limit",
    RIO_OBSERVED_PATTERN: "Observed by RIO — pattern over time",
    RIO_BASELINE_DEVIATION: "Observed by RIO — against this vehicle's baseline",
    RIO_CROSS_SIGNAL: "Observed by RIO — several signals together",
    USER_REPORTED: "Reported by the driver",
    MECHANIC_CONFIRMED: "Confirmed by a mechanic",
    REPAIR_VALIDATED: "Validated after repair",
    SIMULATION: "Simulated — not measured from a vehicle",
    RECORDED_REPLAY: "Replayed from a recording — not measured live",
}


def is_valid(value: str) -> bool:
    return value in ALL


def display(value: str) -> str:
    return DISPLAY.get(value, value)


def is_ecu_reported(value: str) -> bool:
    return value in ECU_REPORTED


def is_rio_observed(value: str) -> bool:
    return value in RIO_OBSERVED


def is_measured(value: str) -> bool:
    """False for simulation and replay. A safety-critical conclusion must never
    rest on data that did not come from a vehicle."""
    return value not in NOT_MEASURED
