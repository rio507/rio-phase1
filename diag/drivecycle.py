"""drivecycle.py — a RIO drive cycle, built on the session infrastructure.

OBD-II counts drive cycles because some faults are only meaningful across
separate warm-up-and-drive events, and because an emissions fault is never urgent
enough to justify guessing. RIO borrows the concept and uses it sparingly:
a handful of monitors require a drive cycle — the ones whose measurement inside a
single drive is mostly measuring the drive — and nothing urgent waits for one. A
critically low tire that sat through three drives before being mentioned would be
a design failure, not diagnostic rigour.

NOT A SECOND SESSION SYSTEM
---------------------------
sessions.py already knows when a drive starts and ends — the dashboard opens one
on Start Drive and closes it on End Drive, and the reaper closes the ones whose
client vanished. This module does not duplicate any of that. It listens:

    session started  ->  a drive cycle begins
    session ended    ->  it ends

The speed heuristic exists only for the case nobody told us — the car moving with
no session open, which happens on a pod restart mid-drive and in every headless
test. It is a fallback, not a parallel truth.

ONE TRACKER PER DOMAIN, ONE DRIVE
---------------------------------
Each domain's engine holds its own tracker, so cycle bookkeeping (which monitors
ran, which issues were created) stays with the domain that produced it. They
agree about when a drive started because they are both driven by the same
sessions.py hooks and the same speed, not because they coordinate.

WHAT A CYCLE RECORDS
--------------------
Enough to answer "was this fault present on more than one drive", and enough that
a freeze frame can name the drive it was captured on. Distance, speeds and
ambient range are recorded when available and left null when not — a cycle that
invents a distance it did not measure is worse than one that admits it has none.
"""
from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Dict, List, Optional


@dataclass
class DriveCycle:
    cycle_id: str
    started_at: float
    started_by: str                       # session | motion
    session_id: Optional[str] = None
    ended_at: Optional[float] = None
    ended_by: str = ""
    max_speed_mph: Optional[float] = None
    speed_sum: float = 0.0
    speed_n: int = 0
    distance_mi: Optional[float] = None
    ambient_min_f: Optional[float] = None
    ambient_max_f: Optional[float] = None
    sensors_seen: List[str] = field(default_factory=list)
    monitor_runs: int = 0
    monitor_results: Dict[str, int] = field(default_factory=dict)
    issues_at_start: List[str] = field(default_factory=list)
    issues_created: List[str] = field(default_factory=list)
    issues_resolved: List[str] = field(default_factory=list)

    def avg_speed_mph(self) -> Optional[float]:
        return round(self.speed_sum / self.speed_n, 1) if self.speed_n else None

    def to_dict(self) -> dict:
        return {
            "cycle_id": self.cycle_id,
            "started_at": self.started_at,
            "started_by": self.started_by,
            "session_id": self.session_id,
            "ended_at": self.ended_at,
            "ended_by": self.ended_by,
            "max_speed_mph": self.max_speed_mph,
            "avg_speed_mph": self.avg_speed_mph(),
            "distance_mi": self.distance_mi,
            "ambient_range_f": ([self.ambient_min_f, self.ambient_max_f]
                                if self.ambient_min_f is not None else None),
            "sensors_seen": sorted(self.sensors_seen),
            "monitor_runs": self.monitor_runs,
            "monitor_results": dict(self.monitor_results),
            "issues_at_start": list(self.issues_at_start),
            "issues_created": list(self.issues_created),
            "issues_resolved": list(self.issues_resolved),
        }


class DriveCycleTracker:
    """One tracker per domain. Single-driver, like everything else here."""

    def __init__(self, store, start_mph: float = 5.0,
                 end_parked_s: float = 300.0, id_prefix: str = "drive"):
        self._store = store
        self._start_mph = start_mph
        self._end_parked_s = end_parked_s
        self._id_prefix = id_prefix
        self.current: Optional[DriveCycle] = None
        self.completed: List[dict] = []
        self._parked_since: Optional[float] = None
        self._seq = 0

    # -- the session hooks, which are the primary signal --------------------
    def note_session_start(self, session_id: str, now: float = None,
                           active_issue_ids: List[str] = None) -> DriveCycle:
        now = time.time() if now is None else now
        if self.current is not None:
            self.end("superseded_by_new_session", now)
        return self._begin("session", now, session_id, active_issue_ids)

    def note_session_end(self, session_id: str, now: float = None) -> None:
        if self.current is not None and self.current.session_id == session_id:
            self.end("session_ended", time.time() if now is None else now)

    # -- the fallback heuristic, for when nobody told us ---------------------
    def observe(self, speed_mph: Optional[float], now: float,
                active_issue_ids: List[str] = None,
                ambient_f: float = None) -> Optional[DriveCycle]:
        """Motion, once per tick. -> the cycle if one just began."""
        began = None
        moving = speed_mph is not None and speed_mph >= self._start_mph

        if moving:
            self._parked_since = None
            if self.current is None:
                began = self._begin("motion", now, None, active_issue_ids)
        else:
            if self._parked_since is None:
                self._parked_since = now
            elif (self.current is not None
                  and self.current.started_by == "motion"
                  and (now - self._parked_since) >= self._end_parked_s):
                # Only a motion-started cycle ends this way. A cycle the session
                # opened ends when the session does — the driver sitting at
                # lights for six minutes has not finished their drive, and
                # ending it here would count one drive as two.
                self.end("parked", now)

        c = self.current
        if c is not None and speed_mph is not None:
            c.speed_sum += speed_mph
            c.speed_n += 1
            c.max_speed_mph = max(c.max_speed_mph or 0.0, speed_mph)
        if c is not None and ambient_f is not None:
            c.ambient_min_f = min(c.ambient_min_f, ambient_f) \
                if c.ambient_min_f is not None else ambient_f
            c.ambient_max_f = max(c.ambient_max_f, ambient_f) \
                if c.ambient_max_f is not None else ambient_f
        return began

    # -- bookkeeping the engine feeds in ------------------------------------
    def note_run(self, monitor_id: str, result: Optional[str]) -> None:
        if self.current is None:
            return
        self.current.monitor_runs += 1
        if result:
            self.current.monitor_results[result] = \
                self.current.monitor_results.get(result, 0) + 1

    def note_issue_created(self, issue_id: str) -> None:
        if self.current is not None and issue_id not in self.current.issues_created:
            self.current.issues_created.append(issue_id)

    def note_issue_resolved(self, issue_id: str) -> None:
        if self.current is not None and issue_id not in self.current.issues_resolved:
            self.current.issues_resolved.append(issue_id)

    def note_sensor(self, name: str) -> None:
        if self.current is not None and name not in self.current.sensors_seen:
            self.current.sensors_seen.append(name)

    # -- identity -----------------------------------------------------------
    @property
    def cycle_id(self) -> Optional[str]:
        return self.current.cycle_id if self.current else None

    def _begin(self, by: str, now: float, session_id: Optional[str],
               active_issue_ids: Optional[List[str]]) -> DriveCycle:
        self._seq += 1
        # Deterministic and readable, and derived from the clock rather than
        # from a uuid so that a log read by a person is orderable by eye.
        cid = f"{self._id_prefix}_{int(now)}_{self._seq:02d}"
        self.current = DriveCycle(
            cycle_id=cid, started_at=now, started_by=by, session_id=session_id,
            issues_at_start=list(active_issue_ids or []))
        self._store.append_event("drive_cycle_start", {
            "cycle_id": cid, "started_by": by, "session_id": session_id,
            "issues_at_start": list(active_issue_ids or [])}, at=now)
        return self.current

    def end(self, why: str, now: float = None) -> Optional[dict]:
        if self.current is None:
            return None
        now = time.time() if now is None else now
        self.current.ended_at = now
        self.current.ended_by = why
        done = self.current.to_dict()
        self.completed.append(done)
        if len(self.completed) > 50:
            del self.completed[:len(self.completed) - 50]
        self._store.append_event("drive_cycle_end", done, at=now)
        self.current = None
        self._parked_since = now
        return done

    def state(self) -> dict:
        return {"current": self.current.to_dict() if self.current else None,
                "completed": self.completed[-5:],
                "completed_count": len(self.completed)}
