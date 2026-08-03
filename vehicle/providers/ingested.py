"""ingested.py — where pushed data becomes pulled data.

This is the join between the two halves of the system, and it is the reason the
rest of the OBD-II work does not need a second health pipeline.

    THE EXISTING PIPELINE IS PULL
    the browser polls /vehicle/telemetry, telemetry.snapshot() asks every
    provider to read(), and insights.observe() is fed as a side effect of that
    read. Nothing in it ever asked to be given anything.

    A BRIDGE IN A CAR IS PUSH
    it decodes at whatever rate the bus allows, buffers through tunnels, and
    uploads in batches when it can. Nothing about that is on the dashboard's
    clock.

Those are opposite directions. The naive resolution is a second path — a
"live data" branch that skips telemetry.py — and it is the wrong one for a
reason that only shows up months later: two paths means two sets of bands, two
staleness rules, two definitions of a trend, and eventually a live vehicle whose
dashboard disagrees with its own conversation layer.

So pushed events land in the buffer below, and the buffer answers read() when
the poll comes round. Live OBD-II, a passive Holley capture, a recorded replay
and the simulator all arrive here, which means none of them gets its own route
into the health layer and none of them can drift from the others.

OUT-OF-ORDER EVENTS ARE NOT A CORNER CASE
-----------------------------------------
They are the NORMAL consequence of the outbox working. A bridge that lost
network for two minutes uploads two minutes of readings at once, oldest first,
while the live stream continues — so the buffer receives an old value after a
newer one on the same signal, routinely. `latest` is therefore chosen by
`observed_at` and never by arrival: a late event updates the history and does
NOT overwrite a fresher reading. Getting this backwards would make every
reconnection look like the coolant temperature suddenly dropping.

THE RING IS NOT THE HISTORY
---------------------------
The ring below is minutes long and exists for one job: the early-fault snapshot
in §16.7 has to reach BACKWARDS from the moment a pending code appears, and
nothing in the repository could do that. telemetry.py's trend ring is twenty
seconds and is deliberately cleared whenever a scenario changes; insights.py
keeps daily aggregates. Between twenty seconds and one day there was nothing.
Long-term history stays where it already lives.
"""
from __future__ import annotations

import threading
import time
from typing import Dict, List, Optional

import config

from ..signals import quality as Q
from ..signals import registry as R


class IngestionBuffer:
    """Latest value per canonical signal, plus a short ring of raw events.

    One instance per process, held below. Thread-safe because ingestion arrives
    on request threads and the dashboard poll reads from another.
    """

    def __init__(self):
        self._lock = threading.RLock()
        # canonical signal name -> the most recently OBSERVED event for it
        self._latest: Dict[str, dict] = {}
        self._ring: List[dict] = []
        self._received = 0
        self._dropped_older = 0
        self._unknown: Dict[str, int] = {}
        self._sources: Dict[str, int] = {}
        self._last_received_ts: Optional[float] = None
        self._session_id: Optional[str] = None

    # -- writing ------------------------------------------------------------

    def push(self, events: List[dict]) -> dict:
        """Fold accepted canonical events in. -> a small summary for the API.

        Events are expected to have been through schema.normalize_event: they
        carry `observed_ts` and `received_ts` as floats and a validated quality.
        """
        applied = superseded = 0
        with self._lock:
            for e in events:
                signal = e.get("signal")
                ts = e.get("observed_ts")
                if not signal or ts is None:
                    continue
                self._received += 1
                self._sources[e.get("source_type", "?")] = \
                    self._sources.get(e.get("source_type", "?"), 0) + 1
                if not e.get("known_signal", True):
                    self._unknown[signal] = self._unknown.get(signal, 0) + 1

                self._ring.append(e)
                self._last_received_ts = max(self._last_received_ts or 0.0,
                                             e.get("received_ts") or 0.0)
                if e.get("drive_session_id"):
                    self._session_id = e["drive_session_id"]

                prev = self._latest.get(signal)
                if prev is not None and (prev.get("observed_ts") or 0.0) > ts:
                    # A late arrival. It belongs in the ring — it is a real
                    # reading and the history needs it — but it must not become
                    # "current", or every reconnection would look like a step
                    # change on whichever signal came out of the outbox last.
                    superseded += 1
                    self._dropped_older += 1
                    continue
                self._latest[signal] = e
                applied += 1

            self._trim()
        return {"applied": applied, "superseded_by_newer": superseded}

    def _trim(self) -> None:
        # Trimmed by RECEIVED time, not observed: a replay of last week's drive
        # must not evaporate the moment it lands, and a bridge emptying its
        # outbox is delivering old observations that are new information.
        #
        # And relative to the newest RECEIVED event rather than to the wall
        # clock, which is not a subtlety: a test driving synthetic time, and a
        # replay whose recording is stamped in the past, both hand this buffer
        # events the wall clock considers ancient. Trimming against time.time()
        # emptied the ring on the way in, so the early-fault snapshot would have
        # had nothing to reach back into on exactly the runs that needed it.
        newest = self._last_received_ts
        if newest is None:
            return
        cutoff = newest - config.VEHICLE_EVENT_RING_S
        i = 0
        for i, e in enumerate(self._ring):
            if (e.get("received_ts") or 0.0) >= cutoff:
                break
        else:
            i = len(self._ring)
        if i:
            del self._ring[:i]
        if len(self._ring) > config.VEHICLE_EVENT_RING_MAX:
            del self._ring[:len(self._ring) - config.VEHICLE_EVENT_RING_MAX]

    # -- reading ------------------------------------------------------------

    def latest(self) -> Dict[str, dict]:
        with self._lock:
            return dict(self._latest)

    def window(self, start_ts: float, end_ts: float,
               signals: List[str] = None) -> List[dict]:
        """Raw events observed in a time range, oldest first.

        This is what the early-fault snapshot reads. It answers by OBSERVED
        time, because the question is "what was the engine doing either side of
        this moment", not "what did the cloud happen to hold".
        """
        want = set(signals) if signals else None
        with self._lock:
            rows = [e for e in self._ring
                    if start_ts <= (e.get("observed_ts") or 0.0) <= end_ts
                    and (want is None or e.get("signal") in want)]
        rows.sort(key=lambda e: e.get("observed_ts") or 0.0)
        return rows

    def stats(self) -> dict:
        with self._lock:
            return {
                "events_received": self._received,
                "signals_current": len(self._latest),
                "ring_events": len(self._ring),
                "superseded_by_newer": self._dropped_older,
                "unknown_signals": dict(self._unknown),
                "source_types": dict(self._sources),
                "last_received_at": self._last_received_ts,
                "drive_session_id": self._session_id,
            }

    def capability(self) -> dict:
        """Which registry signals have ever arrived, and which never have.

        §32.7's "show it in the capability report, hide it from the normal live
        display". A signal that has never arrived is not a fault: most vehicles
        do not expose most PIDs, and a row that sits empty forever looks exactly
        like a sensor that has died.
        """
        with self._lock:
            seen = set(self._latest)
        supported, unsupported = [], []
        for spec in R.SPECS:
            (supported if spec.name in seen else unsupported).append(spec.name)
        return {"supported": sorted(supported), "unsupported": sorted(unsupported)}

    def clear(self) -> None:
        with self._lock:
            self._latest.clear()
            self._ring.clear()
            self._received = 0
            self._dropped_older = 0
            self._unknown.clear()
            self._sources.clear()
            self._last_received_ts = None
            self._session_id = None


# One buffer for the process. Same single-driver assumption as everything else
# stateful in this codebase.
_buffer = IngestionBuffer()


def buffer() -> IngestionBuffer:
    return _buffer


def readings(source_types: List[str] = None) -> List[dict]:
    """The current value of every ingested signal, as flat rows.

    Returned as plain dicts rather than telemetry.SensorReading so that this
    package does not import telemetry — the dependency runs the other way, and a
    cycle between the ingestion layer and the pipeline it feeds would be a cycle
    at import time in a module app.py loads first.

    Each row: {telemetry_id, signal, value, at, ok, quality, provenance,
               source_type, source_ecu, unit, detail}
    """
    want = set(source_types) if source_types else None
    now = time.time()
    out = []
    for signal, e in _buffer.latest().items():
        if want is not None and e.get("source_type") not in want:
            continue
        tid = R.internal(signal)
        if not tid:
            # Canonical, ingested and stored, but not a row on this dashboard.
            # Manufacturer PIDs and undecoded Holley channels live here.
            continue
        qual = e.get("quality", Q.VALID)
        age = now - (e.get("observed_ts") or now)
        usable = Q.is_usable(qual) and e.get("value") is not None
        out.append({
            "telemetry_id": tid,
            "signal": signal,
            "value": e.get("value") if usable else None,
            "at": e.get("observed_ts"),
            "ok": bool(usable) and age <= config.VEHICLE_INGEST_STALE_AFTER_S,
            "quality": qual,
            "provenance": e.get("provenance"),
            "source_type": e.get("source_type"),
            "source_ecu": e.get("source_ecu"),
            "unit": e.get("unit"),
            "decoder_version": e.get("decoder_version"),
            "detail": "" if usable else Q.display(qual),
        })
    return out
