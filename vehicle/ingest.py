"""ingest.py — the front door. Authenticate, validate, deduplicate, hand on.

    POST /api/v1/vehicle-telemetry/batches
              ↓
        authenticate the gateway          auth.authenticate
        authorize it for this vehicle     auth.authorize_vehicle
        spend a rate-limit token          auth.check_rate
        refuse an oversized payload       config.VEHICLE_INGEST_MAX_*
        acknowledge an exact retry        auth.seen_batch
              ↓
        per event: normalize, convert, range-check    signals.schema
        per event: have we already got this one?      the dedup ring below
              ↓
        providers.ingested.buffer().push(...)

WHY DEDUPLICATION IS THE CLOUD'S JOB
------------------------------------
Because the bridge cannot do it. An outbox that has uploaded a batch and not
received the acknowledgement has no way to know whether the batch arrived: the
network failed somewhere, and both "before the server saw it" and "after the
server stored it" look identical from the car. The only safe behaviour for the
bridge is to retry, which means the only safe behaviour for the cloud is to
expect duplicates and to say yes to them.

So a repeated event_id is ACCEPTED — the response says `duplicate`, and the
batch is acknowledged — rather than rejected. A bridge that received an error
for a batch it had already delivered would retry it forever.

WHY A REJECTED EVENT STILL RETURNS A RESULT
-------------------------------------------
Per-event results, not a single verdict, because a batch is not atomic in any
sense that matters: one undecodable frame in two hundred good readings must not
cost the two hundred. The response names exactly which event ids were stored,
which were duplicates and which were refused and why, so a bridge can log the
refusals and stop resending them without also throwing away its backlog.

WHAT THIS CANNOT DO
-------------------
Ask the vehicle for anything. There is no command in a response, no queue for the
bridge to collect, no field a future version could quietly grow into one. The
read-only posture is enforced by there being nothing to call.
"""
from __future__ import annotations

import threading
import time
from collections import deque
from typing import Dict, List, Optional

import config

from .gateway import auth
from .providers import ingested
from .signals import quality as Q
from .signals import schema

_lock = threading.RLock()

# Event ids already stored, newest last. A deque plus a set: the set answers the
# membership question in constant time and the deque decides what falls off the
# end. Bounded because this is a prototype in one process — the durable version
# of this is a unique index in whatever storage the events land in, and the
# window here only has to be longer than any plausible retry.
_seen_ids: deque = deque()
_seen_set: set = set()


def _remember(event_id: str) -> None:
    _seen_ids.append(event_id)
    _seen_set.add(event_id)
    while len(_seen_ids) > config.VEHICLE_INGEST_DEDUP_MAX:
        _seen_set.discard(_seen_ids.popleft())


class IngestError(Exception):
    """Refused before any event was looked at. The message is safe to return."""

    def __init__(self, message: str, status: int = 400):
        super().__init__(message)
        self.status = status


def ingest_batch(raw: dict, gateway_id: str, token: str,
                 now: float = None) -> dict:
    """One batch. -> the acknowledgement, with a result per event.

    Raises IngestError only for things that make the batch unprocessable as a
    whole: bad credentials, the wrong vehicle, a rate limit, a malformed
    envelope, a payload over the ceiling. Anything wrong with an individual
    event is reported in `results` and costs only that event.
    """
    now = time.time() if now is None else float(now)

    # -- the gateway ------------------------------------------------------
    try:
        rec = auth.authenticate(gateway_id, token)
    except auth.AuthError as e:
        raise IngestError(str(e), status=401)

    errors = schema.validate_batch(raw)
    if errors:
        raise IngestError("; ".join(errors), status=400)

    try:
        auth.authorize_vehicle(rec, str(raw.get("vehicle_id") or ""))
    except auth.AuthError as e:
        raise IngestError(str(e), status=403)

    events = raw.get("events") or []
    if len(events) > config.VEHICLE_INGEST_MAX_EVENTS:
        # Refused whole rather than half-processed. A partially accepted batch
        # is the one shape an outbox's retry logic cannot reason about: it has
        # no way to know which half to send again.
        raise IngestError(
            f"batch carries {len(events)} events, over the "
            f"{config.VEHICLE_INGEST_MAX_EVENTS} limit", status=413)

    try:
        auth.check_rate(gateway_id, now=now)
    except auth.AuthError as e:
        raise IngestError(str(e), status=429)

    batch_id = str(raw.get("batch_id") or "")

    # -- an exact retry ---------------------------------------------------
    if batch_id and auth.seen_batch(gateway_id, batch_id):
        return {
            "batch_id": batch_id,
            "accepted": 0,
            "duplicates": len(events),
            "rejected": 0,
            "idempotent_replay": True,
            "results": [],
            "received_at": schema.to_iso(now),
        }

    # -- the events -------------------------------------------------------
    accepted: List[dict] = []
    results: List[dict] = []
    n_dupe = n_bad = 0

    for item in events:
        event, problems = schema.normalize_event(item, received_at=now)
        if event is None:
            n_bad += 1
            results.append({"event_id": (item or {}).get("event_id"),
                            "status": "rejected",
                            "errors": problems})
            continue

        with _lock:
            already = event["event_id"] in _seen_set
            if not already:
                _remember(event["event_id"])

        if already:
            # Accepted, not refused. See the module header: a bridge that got an
            # error for a batch it had already delivered would retry it forever.
            n_dupe += 1
            results.append({"event_id": event["event_id"], "status": "duplicate"})
            continue

        accepted.append(event)
        results.append({
            "event_id": event["event_id"],
            "status": "stored",
            # A stored event may still be one nobody should reason about. The
            # quality says which, and it travels back so a bridge can see that
            # its decoder is producing values the cloud is labelling.
            "quality": event["quality"],
            "warnings": problems or None,
        })

    applied = ingested.buffer().push(accepted) if accepted else {"applied": 0}
    auth.note_batch(gateway_id, batch_id, accepted=len(accepted), rejected=n_bad)

    return {
        "batch_id": batch_id,
        "accepted": len(accepted),
        "duplicates": n_dupe,
        "rejected": n_bad,
        "applied_as_current": applied.get("applied", 0),
        "superseded_by_newer": applied.get("superseded_by_newer", 0),
        "idempotent_replay": False,
        "results": results,
        "received_at": schema.to_iso(now),
    }


def ingest_local(events: List[dict], now: float = None) -> dict:
    """The same path, for a producer running inside this process.

    The simulator and the replay provider use this. They are NOT given a
    shortcut into the buffer: they build canonical events, those events are
    normalized, converted, range-checked and deduplicated by exactly the code a
    bridge's upload goes through, and only then are they current. That is the
    whole of what "simulation and live data use the same pipeline" means, and
    the moment there is a second way in it stops being true.

    What they skip is the network and the credentials, because there is no
    network and no gateway — inventing one would be theatre, and a token stored
    in the process that mints it proves nothing.
    """
    now = time.time() if now is None else float(now)
    accepted: List[dict] = []
    rejected = 0
    for item in events:
        event, problems = schema.normalize_event(item, received_at=now)
        if event is None:
            rejected += 1
            continue
        with _lock:
            if event["event_id"] in _seen_set:
                continue
            _remember(event["event_id"])
        accepted.append(event)
    applied = ingested.buffer().push(accepted) if accepted else {"applied": 0}
    return {"accepted": len(accepted), "rejected": rejected,
            "applied_as_current": applied.get("applied", 0),
            "superseded_by_newer": applied.get("superseded_by_newer", 0)}


def stats() -> dict:
    with _lock:
        dedup = len(_seen_set)
    out = ingested.buffer().stats()
    out["dedup_ids_held"] = dedup
    out["dedup_capacity"] = config.VEHICLE_INGEST_DEDUP_MAX
    return out


def reset_for_test() -> None:
    """Forget every event id and empty the buffer. Tests only."""
    with _lock:
        _seen_ids.clear()
        _seen_set.clear()
    ingested.buffer().clear()
