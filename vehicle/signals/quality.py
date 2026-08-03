"""quality.py — how much a reading is worth, said out loud.

telemetry.py already knows that "the sensor did not answer" and "the sensor
answered zero" are different facts about the car, and keeps `ok` separate from
`value` for exactly that reason. This is the same distinction widened to the
eleven states a value can arrive in once it has crossed a network, a decoder and
a bus.

The states are not a severity ladder and must not be sorted into one. `stale`
and `unsupported` are both "you cannot use this number", but a stale coolant
reading means the link is in trouble and an unsupported one means this vehicle
does not have that PID — a UI that treated them alike would send somebody
looking for a fault in a wire that was never there.

WHAT USES THIS
--------------
The gate on every conclusion. A finding derived from a `decode_error` value is
not a finding; a `simulation` value must never support a safety-critical claim;
an `unverified_decoder` value can be shown, labelled, and reasoned about out
loud, but not acted on.
"""
from __future__ import annotations

VALID = "valid"
STALE = "stale"
UNSUPPORTED = "unsupported"
MISSING = "missing"
INVALID_RANGE = "invalid_range"
DECODE_ERROR = "decode_error"
TRANSPORT_ERROR = "transport_error"
ESTIMATED = "estimated"
SIMULATION = "simulation"
RECORDED_REPLAY = "recorded_replay"
UNVERIFIED_DECODER = "unverified_decoder"

ALL = (VALID, STALE, UNSUPPORTED, MISSING, INVALID_RANGE, DECODE_ERROR,
       TRANSPORT_ERROR, ESTIMATED, SIMULATION, RECORDED_REPLAY,
       UNVERIFIED_DECODER)

# States in which the number is a measurement of the thing it claims to measure.
# Everything else is a fact about the pipeline, not about the car.
USABLE = (VALID, ESTIMATED, SIMULATION, RECORDED_REPLAY, UNVERIFIED_DECODER)

# States that may support a conclusion RIO acts on. Simulation and replay are
# usable — the dashboard renders them, the monitors run on them — but a monitor
# must never tell a driver to pull over because of a number nobody measured, and
# an unverified Holley decoder must never be the evidence behind a safety claim.
TRUSTWORTHY = (VALID, ESTIMATED)

# What a person reads.
DISPLAY = {
    VALID: "Valid",
    STALE: "Stale",
    UNSUPPORTED: "Not supported by this vehicle",
    MISSING: "Missing",
    INVALID_RANGE: "Out of plausible range",
    DECODE_ERROR: "Could not be decoded",
    TRANSPORT_ERROR: "Transport error",
    ESTIMATED: "Estimated",
    SIMULATION: "Simulated",
    RECORDED_REPLAY: "Replayed",
    UNVERIFIED_DECODER: "Unverified decoder",
}


def is_valid(value: str) -> bool:
    return value in ALL


def is_usable(value: str) -> bool:
    """Is this a measurement at all, or a statement about the pipeline?"""
    return value in USABLE


def is_trustworthy(value: str) -> bool:
    """May a conclusion RIO acts on rest on this reading?"""
    return value in TRUSTWORTHY


def display(value: str) -> str:
    return DISPLAY.get(value, value)
