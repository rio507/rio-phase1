"""schema.py — the canonical telemetry event, and the batch that carries it.

One shape, for everything. A CANable bridge in a car, the simulator on a laptop,
a replay of a recorded drive and a future Jetson all produce exactly this, and
the cloud cannot tell them apart except by reading `source_type` — which is
metadata, not a branch in the health logic.

    {
      "schema_version": "1.0",
      "event_id": "uuid",              deduplication identity
      "batch_id": "uuid",
      "vehicle_id": "...", "gateway_id": "...", "drive_session_id": "...",
      "signal": "powertrain.engine.coolant_temperature",
      "value": 96.0, "unit": "fahrenheit",
      "source_type": "obd2_can", "source_signal": "0105", "source_ecu": "0x7E8",
      "provenance": "ecu_measurement",
      "quality": "valid", "confidence": 1.0,
      "observed_at": "...", "received_at": "...",
      "decoder_version": "obd-standard-1.0",
      "metadata": {...}
    }

WHY observed_at AND received_at ARE BOTH REQUIRED
-------------------------------------------------
Because they answer different questions and a system with only one of them
cannot notice that it has a problem. `observed_at` is when the car did the
thing; `received_at` is when the cloud heard about it. A bridge that buffered
through a tunnel delivers a burst of events whose observed times are minutes
apart and whose received times are milliseconds apart, and every trend has to be
computed on the first and every latency on the second. Storing one and inferring
the other is how an offline buffer turns into a fabricated ten-minute spike.

Excessive skew between them is recorded rather than corrected. A gateway with a
wrong clock is a fact about the gateway, and quietly rewriting its timestamps
would destroy the only evidence of it.

WHY THE UNIT TRAVELS WITH THE VALUE
-----------------------------------
So that a mismatch is catchable. `unit` is asserted by the sender and checked
here against the registry; a value whose declared unit is not the canonical one
is converted once, on arrival, and an undeclared or unconvertible unit is a
`decode_error` rather than a number that quietly means something else. See
units.py on why this boundary is the only safe place for it.

WHAT VALIDATION IS FOR, AND WHAT IT IS NOT FOR
----------------------------------------------
It is for rejecting things that are not measurements: an unknown signal name, a
non-numeric value, a timestamp that will not parse, a provenance nobody defined.
It is NOT for deciding whether a reading is worrying — a coolant temperature of
260°F passes validation with `quality: valid` and then sets the panel on fire
downstream, which is exactly right. Validation guards the shape; config.py owns
the meaning.
"""
from __future__ import annotations

import re
import uuid
from datetime import datetime, timezone
from typing import Dict, List, Optional, Tuple

from . import provenance as P
from . import quality as Q
from . import registry as R
from . import units as U

SCHEMA_VERSION = "1.0"

# --- source types (§14.3) ---------------------------------------------------
# What produced the event. Metadata, deliberately: the health engine reads it to
# label a row and to decide how much a conclusion may lean on it, and never to
# choose which code path to run.
DASHBOARD_SIMULATOR = "dashboard_simulator"
OBD2_CAN = "obd2_can"
HOLLEY_TERMINATOR_X = "holley_terminator_x"
RECORDED_REPLAY = "recorded_replay"
FUTURE_JETSON = "future_jetson"
RIO_CUSTOM_SENSOR = "rio_custom_sensor"

SOURCE_TYPES = (DASHBOARD_SIMULATOR, OBD2_CAN, HOLLEY_TERMINATOR_X,
                RECORDED_REPLAY, FUTURE_JETSON, RIO_CUSTOM_SENSOR)

# Which provenance a source type implies when the sender did not say. A sender
# MAY override — a Holley bridge replaying a capture is recorded_replay even
# though its source type is the Holley — but a source that says nothing gets the
# honest default rather than `ecu_measurement`.
DEFAULT_PROVENANCE = {
    DASHBOARD_SIMULATOR: P.SIMULATION,
    OBD2_CAN: P.ECU_MEASUREMENT,
    HOLLEY_TERMINATOR_X: P.ECU_MEASUREMENT,
    RECORDED_REPLAY: P.RECORDED_REPLAY,
    FUTURE_JETSON: P.ECU_MEASUREMENT,
    RIO_CUSTOM_SENSOR: P.ECU_MEASUREMENT,
}

DEFAULT_QUALITY = {
    DASHBOARD_SIMULATOR: Q.SIMULATION,
    RECORDED_REPLAY: Q.RECORDED_REPLAY,
}

# How far apart observed_at and received_at may be before the gateway's clock is
# reported as suspect. Generous: a bridge that buffered through a long tunnel is
# legitimately hours behind, and that is not skew. What this catches is a clock
# in the wrong YEAR, or one running ahead of the cloud's — an event observed in
# the future is not a late event, it is a wrong one.
MAX_FUTURE_SKEW_S = 120.0

_ID_RE = re.compile(r"^[A-Za-z0-9_.:-]{1,128}$")


# ---------------------------------------------------------------------------
# Time
# ---------------------------------------------------------------------------
# Epoch floats inside, ISO-8601 on the wire. The repository runs on epoch floats
# everywhere — telemetry's rings, tires' updated_at, the diagnostic store — and
# converting at the boundary keeps it that way rather than threading a second
# time representation through code that already works.

def to_iso(ts: Optional[float]) -> Optional[str]:
    if ts is None:
        return None
    return datetime.fromtimestamp(float(ts), tz=timezone.utc).isoformat(
        timespec="milliseconds").replace("+00:00", "Z")


def from_iso(value) -> Optional[float]:
    """ISO-8601 (or a bare epoch number) -> epoch seconds. None if unparseable.

    Numbers are accepted because a bridge written in a hurry will send one, and
    refusing a valid instant over its notation would drop real vehicle data.
    """
    if value is None:
        return None
    if isinstance(value, (int, float)):
        return float(value)
    text = str(value).strip()
    if not text:
        return None
    try:
        return float(text)
    except ValueError:
        pass
    try:
        if text.endswith("Z"):
            text = text[:-1] + "+00:00"
        dt = datetime.fromisoformat(text)
        if dt.tzinfo is None:
            dt = dt.replace(tzinfo=timezone.utc)
        return dt.timestamp()
    except ValueError:
        return None


def new_id() -> str:
    return uuid.uuid4().hex


# ---------------------------------------------------------------------------
# Building
# ---------------------------------------------------------------------------

def make_event(signal: str, value: Optional[float], observed_at: float,
               source_type: str, *, vehicle_id: str, gateway_id: str,
               drive_session_id: str = None, unit: str = None,
               source_signal: str = "", source_ecu: str = "",
               provenance: str = None, quality: str = None,
               confidence: float = 1.0, decoder_version: str = "",
               event_id: str = None, batch_id: str = None,
               metadata: dict = None) -> dict:
    """One canonical event, in the shape the API accepts.

    `unit` defaults to the registry's canonical unit for the signal, which is
    what an in-process producer should send. A decoder that has NOT converted
    passes its own unit and lets normalize() do the conversion — that is the
    supported path and the only one that keeps the conversion at the boundary.
    """
    spec = R.spec(signal)
    return {
        "schema_version": SCHEMA_VERSION,
        "event_id": event_id or new_id(),
        "batch_id": batch_id or "",
        "vehicle_id": vehicle_id,
        "gateway_id": gateway_id,
        "drive_session_id": drive_session_id,
        "signal": signal,
        "value": None if value is None else float(value),
        "unit": unit or (spec.unit if spec else ""),
        "source_type": source_type,
        "source_signal": source_signal,
        "source_ecu": source_ecu,
        "provenance": provenance or DEFAULT_PROVENANCE.get(source_type,
                                                           P.ECU_MEASUREMENT),
        "quality": quality or DEFAULT_QUALITY.get(source_type, Q.VALID),
        "confidence": float(confidence),
        "observed_at": to_iso(observed_at),
        "received_at": None,
        "decoder_version": decoder_version,
        "metadata": dict(metadata or {}),
    }


def make_batch(events: List[dict], *, vehicle_id: str, gateway_id: str,
               drive_session_id: str = None, batch_id: str = None,
               sequence_start: int = None, sequence_end: int = None,
               sent_at: float = None) -> dict:
    bid = batch_id or new_id()
    for e in events:
        e.setdefault("batch_id", bid)
        if not e.get("batch_id"):
            e["batch_id"] = bid
    return {
        "schema_version": SCHEMA_VERSION,
        "batch_id": bid,
        "gateway_id": gateway_id,
        "vehicle_id": vehicle_id,
        "drive_session_id": drive_session_id,
        "sent_at": to_iso(sent_at) if sent_at else None,
        "sequence_start": sequence_start,
        "sequence_end": sequence_end,
        "events": events,
    }


# ---------------------------------------------------------------------------
# Validation and normalization
# ---------------------------------------------------------------------------

def normalize_event(raw: dict, received_at: float) -> Tuple[Optional[dict], List[str]]:
    """-> (event, errors). A returned event is safe to store and to interpret.

    Errors fall into two kinds and only one of them loses the event:

      REJECTED   the shape is wrong — no signal name, an unparseable timestamp,
                 an unknown provenance. There is nothing to store.

      DOWNGRADED the shape is right and the value is not trustworthy — a unit
                 that will not convert, a number outside any plausible range.
                 The event is KEPT with a quality that says so, because "the
                 decoder produced nonsense at 14:22" is itself a finding, and an
                 event dropped on the floor is a finding nobody can make.
    """
    errors: List[str] = []
    if not isinstance(raw, dict):
        return None, ["event is not an object"]

    signal = str(raw.get("signal") or "").strip()
    if not signal:
        return None, ["missing signal"]

    event_id = str(raw.get("event_id") or "").strip() or new_id()
    if not _ID_RE.match(event_id):
        return None, [f"malformed event_id {event_id!r}"]

    observed = from_iso(raw.get("observed_at"))
    if observed is None:
        return None, ["missing or unparseable observed_at"]

    source_type = str(raw.get("source_type") or "").strip()
    if source_type not in SOURCE_TYPES:
        return None, [f"unknown source_type {source_type!r}"]

    prov = str(raw.get("provenance") or "").strip() or \
        DEFAULT_PROVENANCE.get(source_type, P.ECU_MEASUREMENT)
    if not P.is_valid(prov):
        return None, [f"unknown provenance {prov!r}"]

    qual = str(raw.get("quality") or "").strip() or \
        DEFAULT_QUALITY.get(source_type, Q.VALID)
    if not Q.is_valid(qual):
        return None, [f"unknown quality {qual!r}"]

    value = raw.get("value")
    if value is not None:
        try:
            value = float(value)
        except (TypeError, ValueError):
            value, qual = None, Q.DECODE_ERROR
            errors.append("value is not a number")

    spec = R.spec(signal)
    unit = str(raw.get("unit") or "").strip() or (spec.unit if spec else "")

    # Conversion, exactly once, here. A sender that already converted declares
    # the canonical unit and this is a no-op.
    if spec is not None and value is not None and unit and unit != spec.unit:
        if U.can_convert(unit, spec.unit):
            value = U.convert(value, unit, spec.unit)
            unit = spec.unit
        else:
            qual = Q.DECODE_ERROR
            errors.append(f"cannot convert {unit!r} to {spec.unit!r} "
                          f"for {signal!r}")

    # Plausibility. Not a warning band — the bound outside which the number is
    # not a measurement at all. A value that fails this is kept and labelled;
    # see the docstring on why it is not dropped.
    if spec is not None and value is not None and Q.is_usable(qual) \
            and not R.in_range(signal, value):
        qual = Q.INVALID_RANGE
        errors.append(f"{signal}={value} outside the plausible range "
                      f"{spec.plausible}")

    # A clock running ahead of the cloud's is a wrong clock, not a late event.
    # Recorded, never corrected: quietly rewriting it would destroy the only
    # evidence that the gateway's clock is off.
    skew = observed - received_at
    metadata = dict(raw.get("metadata") or {})
    if skew > MAX_FUTURE_SKEW_S:
        metadata["clock_skew_s"] = round(skew, 1)
        errors.append(f"observed_at is {skew:.0f}s in the future")

    try:
        confidence = float(raw.get("confidence", 1.0))
    except (TypeError, ValueError):
        confidence = 1.0

    event = {
        "schema_version": str(raw.get("schema_version") or SCHEMA_VERSION),
        "event_id": event_id,
        "batch_id": str(raw.get("batch_id") or ""),
        "vehicle_id": str(raw.get("vehicle_id") or ""),
        "gateway_id": str(raw.get("gateway_id") or ""),
        "drive_session_id": raw.get("drive_session_id") or None,
        "signal": signal,
        "value": value,
        "unit": unit,
        "source_type": source_type,
        "source_signal": str(raw.get("source_signal") or ""),
        "source_ecu": str(raw.get("source_ecu") or ""),
        "provenance": prov,
        "quality": qual,
        "confidence": confidence,
        "observed_at": to_iso(observed),
        "observed_ts": observed,          # epoch, for everything downstream
        "received_at": to_iso(received_at),
        "received_ts": received_at,
        "decoder_version": str(raw.get("decoder_version") or ""),
        "metadata": metadata,
        # A signal nobody has registered is stored and displayed, never dropped.
        # Manufacturer channels and undecoded Holley frames are exactly this,
        # and a pipeline that discarded them would make them undiscoverable.
        "known_signal": spec is not None,
    }
    if spec is None:
        event["quality"] = qual if qual != Q.VALID else Q.UNVERIFIED_DECODER
        errors.append(f"signal {signal!r} is not in the registry")
    return event, errors


def validate_batch(raw: dict) -> List[str]:
    """Shape errors on the batch envelope itself. -> [] when it is acceptable."""
    errors = []
    if not isinstance(raw, dict):
        return ["batch is not an object"]
    if not str(raw.get("gateway_id") or "").strip():
        errors.append("missing gateway_id")
    if not str(raw.get("vehicle_id") or "").strip():
        errors.append("missing vehicle_id")
    events = raw.get("events")
    if not isinstance(events, list):
        errors.append("events is not a list")
    batch_id = str(raw.get("batch_id") or "").strip()
    if batch_id and not _ID_RE.match(batch_id):
        errors.append(f"malformed batch_id {batch_id!r}")
    return errors
