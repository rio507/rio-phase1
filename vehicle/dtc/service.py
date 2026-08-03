"""service.py — scanning on a cadence, and the Flagged Error Codes section.

Three things that belong together because they share one clock:

    the SCHEDULE    which diagnostic service is due to be asked, and when
    the REGISTRY    what to do with the answer
    the SECTION     what the dashboard renders from it

WHY THE SCHEDULE IS NOT "ASK FOR EVERYTHING EVERY TIME"
-------------------------------------------------------
§13's bus etiquette is not a nicety. A diagnostic request occupies the bus the
car uses to run itself, and a prototype that polls flat out is the fastest way to
make a vehicle's own systems unreliable — which would be a spectacular way for a
health monitor to fail. So each service has its own cadence, the scheduler asks
for at most one thing at a time, and the cadences are in config.py where they can
be tuned without touching this file.

The pending scan is the frequent one, because it is the one carrying the product:
a code that is pending and not yet confirmed is the whole of the early-detection
story, and finding it four minutes late is four minutes of a story lost.

WHY A SCAN CAN BE ASKED FOR OUT OF TURN
---------------------------------------
§16.4 lists the moments that justify an immediate scan and they are all "the
world just changed": the drive started, the lamp state changed, the reported code
count changed, the driver asked, an abnormal pattern was detected, the drive is
ending. Those are not a cadence, they are events, and a scheduler that could only
tick would miss every one of them.

WHAT THIS FILE DOES NOT DO
--------------------------
Speak. It produces dashboard events and a section to render; the decision to
interrupt a driver stays in vehicle_health_policy.py, which imports nothing and
cannot be reached from here. §19's sentences are report and card text — they are
deliberately NOT entries in that module's LINE table, because that table is the
bounded set of things RIO can say unprompted and a pending code is not something
to interrupt somebody about.
"""
from __future__ import annotations

import threading
import time
from typing import Callable, Dict, List, Optional

import config

from ..providers import ingested
from ..signals import provenance as P
from . import catalog as C
from . import lifecycle as L
from .snapshot import SnapshotStore

# Why a scan happened. Recorded on every one, because "we asked because the lamp
# changed" and "we asked because two minutes elapsed" produce identical data and
# very different stories about it.
REASON_CADENCE = "cadence"
REASON_DRIVE_START = "drive_start"
REASON_DRIVE_END = "drive_end"
REASON_REQUESTED = "driver_requested"
REASON_MIL_CHANGED = "mil_changed"
REASON_COUNT_CHANGED = "dtc_count_changed"
REASON_PATTERN = "abnormal_pattern"


class DTCService:
    """Scanning, recording and rendering, on one clock."""

    def __init__(self, scanner: Callable = None, registry: L.DTCRegistry = None):
        self._lock = threading.RLock()
        self.registry = registry or L.DTCRegistry()
        self.snapshots = SnapshotStore(ingested.buffer)
        self._scanner = scanner
        self._last_full: Optional[float] = None
        self._last_pending: Optional[float] = None
        self._last_stored: Optional[float] = None
        self._last_mil_poll: Optional[float] = None
        self._last_count: Optional[int] = None
        self._events: List[dict] = []
        self._scans = 0
        self._no_response_since: Optional[float] = None

    def set_scanner(self, scanner: Callable) -> None:
        """Where scans come from. The mock ECU today, a bridge later.

        Injected rather than imported so that a bridge's uploaded scan and a
        simulated one go through identical code — the same argument the
        canonical telemetry path makes, applied to the diagnostic side.
        """
        with self._lock:
            self._scanner = scanner

    # ------------------------------------------------------------------
    # Scanning
    # ------------------------------------------------------------------
    def poll(self, now: float, drive_session_id: str = None) -> dict:
        """One tick. Scans only what is due, and completes waiting snapshots."""
        due = self._due(now)
        out = {"scanned": False, "due": due, "events": []}
        if due:
            out = self.scan(now, drive_session_id, reason=REASON_CADENCE,
                            services=due)
        finished = self.snapshots.complete_due(now)
        for snap in finished:
            out.setdefault("events", []).append({
                "type": L.EV_SNAPSHOT_READY,
                "vehicle_id": config.VEHICLE_ID,
                "drive_session_id": snap.get("drive_session_id"),
                "at": now,
                "payload": {"code": snap["code"],
                            "snapshot_id": snap["snapshot_id"],
                            "complete": True},
            })
        return out

    def _due(self, now: float) -> List[str]:
        with self._lock:
            due = []
            if self._last_mil_poll is None or \
                    (now - self._last_mil_poll) >= config.VEHICLE_DTC_MIL_POLL_S:
                due.append("mil")
            if self._last_pending is None or \
                    (now - self._last_pending) >= config.VEHICLE_DTC_PENDING_POLL_S:
                due.append("pending")
            if self._last_stored is None or \
                    (now - self._last_stored) >= config.VEHICLE_DTC_STORED_POLL_S:
                due.append("stored")
            return due

    def scan(self, now: float, drive_session_id: str = None,
             reason: str = REASON_REQUESTED,
             services: List[str] = None) -> dict:
        """Ask the vehicle. -> what changed.

        The mock ECU answers a whole scan at once, as a bridge's uploaded scan
        does; `services` records what the SCHEDULER wanted so the log shows why
        the request went out, without pretending the transport was finer-grained
        than it is.
        """
        with self._lock:
            scanner = self._scanner
        if scanner is None:
            return {"scanned": False, "reason": "no scanner configured",
                    "events": []}

        raw = scanner(now, drive_session_id)
        with self._lock:
            self._scans += 1
            self._last_mil_poll = now
            if services is None or "pending" in services:
                self._last_pending = now
            if services is None or "stored" in services:
                self._last_stored = now
            self._last_full = now

        if raw is None:
            # The ECU did not answer. NOT an empty result: a vehicle saying "no
            # codes" and a vehicle saying nothing are opposite facts, and
            # reporting the second as the first would be an all-clear for a car
            # nobody could reach.
            with self._lock:
                if self._no_response_since is None:
                    self._no_response_since = now
            return {"scanned": False, "ecu_responding": False,
                    "no_response_since": self._no_response_since,
                    "reason": reason, "events": []}

        with self._lock:
            self._no_response_since = None

        result = self.registry.observe_scan(
            raw, now=now,
            snapshotter=lambda code, at: self.snapshots.capture(
                code, at, drive_session_id))

        count = raw.get("dtc_count")
        with self._lock:
            count_changed = (self._last_count is not None
                             and count != self._last_count)
            self._last_count = count
            for e in result["events"]:
                e["reason"] = reason
            self._events.extend(result["events"])
            if len(self._events) > 400:
                del self._events[:len(self._events) - 400]

        return {
            "scanned": True,
            "ecu_responding": True,
            "reason": reason,
            "services": services,
            "mil": result["mil"],
            "codes_reported": result["codes_reported"],
            "dtc_count": count,
            "dtc_count_changed": count_changed,
            "events": result["events"],
        }

    def ingest_scan(self, payload: dict, now: float = None) -> dict:
        """A scan uploaded by a gateway (§25.4). Same path as a local one."""
        now = time.time() if now is None else float(now)
        return self.scan(now, payload.get("drive_session_id"),
                         reason=payload.get("reason", REASON_CADENCE),
                         services=None) if self._scanner is not None else \
            self._ingest_direct(payload, now)

    def _ingest_direct(self, payload: dict, now: float) -> dict:
        result = self.registry.observe_scan(
            payload, now=now,
            snapshotter=lambda code, at: self.snapshots.capture(
                code, at, payload.get("drive_session_id")))
        with self._lock:
            self._events.extend(result["events"])
        return {"scanned": True, "ecu_responding": True,
                "events": result["events"], "mil": result["mil"],
                "codes_reported": result["codes_reported"]}

    # ------------------------------------------------------------------
    # The Flagged Error Codes section (§17)
    # ------------------------------------------------------------------
    def card(self, rec: dict) -> dict:
        """One code, with every field §17.5 requires and nothing invented."""
        d = C.get(rec["code"])
        snap = self.snapshots.get(rec.get("snapshot_id") or "") \
            if rec.get("snapshot_id") else None
        return {
            "code": rec["code"],
            "description": d.description,
            "known": rec.get("known", False),
            "manufacturer_specific": rec.get("manufacturer_specific", False),

            "status_label": L.display_status(rec),
            "lifecycle": rec["lifecycle"],
            "dtc_category": rec.get("last_present_status"),
            "ecu_status": rec.get("ecu_status"),

            "system": d.system,
            "system_label": d.system_label,
            "source_ecu": (rec.get("freeze_frame") or {}).get("source_ecu"),

            "provenance": rec.get("provenance"),
            "provenance_label": P.display(rec.get("provenance", "")),
            "severity": rec.get("severity"),
            "severity_label": C.SEVERITY_LABEL.get(rec.get("severity", ""), ""),

            "mil_commanded_on": rec.get("mil_commanded_on"),
            "mil_at_first_detection": rec.get("mil_at_first_detection"),
            "early_detection": rec.get("early_detection", False),

            "first_seen_at": rec.get("first_seen_at"),
            "last_seen_at": rec.get("last_seen_at"),
            "first_seen_session": rec.get("first_seen_session"),
            "last_seen_session": rec.get("last_seen_session"),
            "pending_scan_count": rec.get("pending_scan_count", 0),
            "drive_cycle_count_observed": rec.get("drive_cycle_count_observed", 0),
            "recurrence_count": rec.get("recurrence_count", 0),

            # The vehicle's own record of the moment, and RIO's. Two fields,
            # never merged: one is authoritative about when the fault was set
            # and the other is what RIO happened to be watching. §17.9.
            "freeze_frame_available": bool(rec.get("freeze_frame_available")),
            "freeze_frame": rec.get("freeze_frame"),
            "rio_snapshot_id": rec.get("snapshot_id"),
            "rio_snapshot_complete": bool(snap and snap.get("complete")),

            "related_signals": list(d.related_signals),
            "what_this_means": d.driver_explanation,
            "technician_detail": d.technician_detail,
            # A list, presented as a list. Nothing in this codebase promotes an
            # entry of it to a fact — see catalog.py's header.
            "possible_causes": list(d.possible_causes),
            "cause_status": ("Confirmed" if rec.get("confirmed_cause")
                             else "Not Confirmed"),
            "confirmed_cause": rec.get("confirmed_cause"),
            "repair": rec.get("repair"),
            "status_history": list(rec.get("status_history") or []),
        }

    def section(self) -> dict:
        """Everything the Flagged Error Codes section renders (§17).

        The empty state is a sentence about the CODES, not about the car. "No
        active or pending error codes are currently being reported by the
        vehicle" is true and useful; "your vehicle is healthy" would be neither,
        because most of what can be wrong with a car sets no code at all.
        """
        records = self.registry.records()
        groups = []
        for key in L.GROUP_ORDER:
            rows = [r for r in records if L.group_of(r) == key]
            rows.sort(key=lambda r: (-C.SEVERITY_RANK.get(r["severity"], 0),
                                     -(r.get("last_seen_at") or 0.0),
                                     r["code"]))
            if rows:
                groups.append({"key": key, "title": L.GROUP_TITLE[key],
                               "cards": [self.card(r) for r in rows]})

        meta = self.registry.meta()
        with self._lock:
            scans = self._scans
            no_response_since = self._no_response_since
        return {
            "groups": groups,
            "code_count": sum(len(g["cards"]) for g in groups),
            "active_count": len(self.registry.active_codes()),
            "mil_commanded_on": self.registry.mil(),
            "scan_count": scans,
            "last_scan_at": meta.get("last_scan_at"),
            "dtc_count_reported": meta.get("dtc_count_reported"),
            "services_supported": meta.get("supported") or {},
            "ecu_responding": no_response_since is None,
            "no_response_since": no_response_since,
            "empty_state": ("No active or pending error codes are currently "
                            "being reported by the vehicle."),
            # Said in the payload rather than left to the browser, because it is
            # the one sentence in this section most likely to be paraphrased
            # into something false.
            "empty_state_caveat": ("This is not the same as the vehicle being "
                                   "mechanically healthy — most faults set no "
                                   "code at all."),
        }

    def events(self, limit: int = 100) -> List[dict]:
        with self._lock:
            return list(self._events[-limit:])

    def stats(self) -> dict:
        with self._lock:
            return {
                "scans": self._scans,
                "last_scan_at": self._last_full,
                "last_pending_scan_at": self._last_pending,
                "last_stored_scan_at": self._last_stored,
                "ecu_responding": self._no_response_since is None,
                "snapshots": self.snapshots.stats(),
                "cadence_s": {
                    "mil": config.VEHICLE_DTC_MIL_POLL_S,
                    "pending": config.VEHICLE_DTC_PENDING_POLL_S,
                    "stored": config.VEHICLE_DTC_STORED_POLL_S,
                },
            }

    def reset_for_test(self, directory: str) -> None:
        self.registry.reset_for_test(directory)
        self.snapshots.clear()
        with self._lock:
            self._last_full = self._last_pending = self._last_stored = None
            self._last_mil_poll = None
            self._last_count = None
            self._events = []
            self._scans = 0
            self._no_response_since = None


_service: Optional[DTCService] = None


def service() -> DTCService:
    global _service
    if _service is None:
        from ..producers import ecu as ecu_mod
        _service = DTCService()
        _service.set_scanner(
            lambda now, session: ecu_mod.ecu().scan(now, session))
    return _service
