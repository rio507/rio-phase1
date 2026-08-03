"""snapshot.py — what the engine was doing either side of a code appearing.

§16.7 asks for sixty seconds before and sixty seconds after the moment a pending
code is first detected. The second half is easy and the first half is the whole
point: nothing else in this system could answer it.

    telemetry.py's trend ring    20 seconds, in memory, cleared on every
                                 scenario change
    insights.py                  daily aggregates

Between twenty seconds and one day there was nothing, so "what was the fuel trim
doing in the minute before the ECU noticed" had no answer. The ingestion buffer's
ring exists for this, and this module is its only reader.

TWO HALVES, WRITTEN AT DIFFERENT TIMES
--------------------------------------
The "before" half is captured immediately, because it already happened. The
"after" half cannot be, so the snapshot is stored incomplete and filled in later
by complete_due(). Waiting for both before storing anything would mean a process
restart in that minute loses the half that cannot be reconstructed — which is
precisely the minute where a restart is most likely, because something has just
gone wrong with the car.

THIS IS NOT A FREEZE FRAME
--------------------------
An ECU freeze frame is what the VEHICLE chose to record at the moment it set the
code, and it is authoritative about that moment. This is what RIO happened to be
watching. They are stored in different fields, displayed apart, and labelled
differently (§17.9), because presenting RIO's observations with the ECU's
authority would be the same category error as presenting a possible cause as a
diagnosis.
"""
from __future__ import annotations

import threading
import time
from typing import Dict, List, Optional

import config

from ..signals import quality as Q
from ..signals import registry as R
from . import catalog as C

# The channels §16.7 asks for, as canonical names. Everything a lean-condition
# or cooling investigation would want, and nothing else — a snapshot carrying
# every registered signal would be unreadable and would imply the code had
# something to do with all of them.
DEFAULT_SIGNALS = (
    "powertrain.engine.rpm",
    "vehicle.speed",
    "powertrain.engine.coolant_temperature",
    "powertrain.engine.intake_air_temperature",
    "powertrain.engine.calculated_load",
    "powertrain.engine.throttle_position",
    "powertrain.engine.manifold_pressure",
    "powertrain.engine.mass_air_flow",
    "powertrain.fuel.short_term_trim_bank_1",
    "powertrain.fuel.long_term_trim_bank_1",
    "electrical.control_module_voltage",
)


def signals_for(code: str) -> List[str]:
    """The default set, plus whatever this code's profile adds (§18).

    Union rather than replacement: the related-signal profile says what is
    ESPECIALLY worth watching for this code, not what is the only thing worth
    watching. A misfire snapshot without road speed in it is harder to read, and
    nobody thanks you for the economy.
    """
    extra = [s for s in C.related_signals(code) if s not in DEFAULT_SIGNALS]
    return list(DEFAULT_SIGNALS) + extra


class SnapshotStore:
    """Early-fault snapshots, in memory, bounded, and completed in arrears."""

    def __init__(self, buffer_getter):
        self._lock = threading.RLock()
        self._buffer = buffer_getter
        self._snaps: Dict[str, dict] = {}
        self._order: List[str] = []
        self._seq = 0

    def capture(self, code: str, at: float,
                drive_session_id: str = None) -> Optional[dict]:
        """Take the BEFORE half now, and register the after half as pending."""
        with self._lock:
            self._seq += 1
            snapshot_id = f"snap_{int(at)}_{self._seq:03d}"
        wanted = signals_for(code)
        before = self._collect(at - config.VEHICLE_DTC_SNAPSHOT_BEFORE_S, at,
                               wanted)
        snap = {
            "snapshot_id": snapshot_id,
            "code": code,
            "detected_at": at,
            "drive_session_id": drive_session_id,
            "signals": wanted,
            "before": before,
            "after": {},
            "after_due_at": at + config.VEHICLE_DTC_SNAPSHOT_AFTER_S,
            "complete": False,
            # Every reading in here is RIO's own observation, never the ECU's
            # freeze frame. The two are displayed apart and this field is what
            # keeps them apart downstream.
            "source": "rio_recorded_history",
        }
        with self._lock:
            self._snaps[snapshot_id] = snap
            self._order.append(snapshot_id)
            while len(self._order) > config.VEHICLE_DTC_SNAPSHOT_MAX:
                self._snaps.pop(self._order.pop(0), None)
        return snap

    def complete_due(self, now: float) -> List[dict]:
        """Fill in the after half of any snapshot whose minute has passed."""
        done = []
        with self._lock:
            pending = [s for s in self._snaps.values()
                       if not s["complete"] and now >= s["after_due_at"]]
        for snap in pending:
            after = self._collect(snap["detected_at"], snap["after_due_at"],
                                  snap["signals"])
            with self._lock:
                snap["after"] = after
                snap["complete"] = True
            done.append(snap)
        return done

    def _collect(self, start: float, end: float,
                 signals: List[str]) -> Dict[str, dict]:
        """Every reading of every wanted signal in a window, plus a summary.

        The series is kept as well as the summary. A minimum and a maximum
        answer "how bad did it get"; the series answers "what shape was it",
        and a fuel trim that ramped and one that spiked have the same maximum
        and mean completely different things.
        """
        rows = self._buffer().window(start, end, signals)
        out: Dict[str, dict] = {}
        for e in rows:
            if e.get("value") is None or not Q.is_usable(e.get("quality", "")):
                continue
            name = e["signal"]
            entry = out.setdefault(name, {
                "signal": name,
                "telemetry_id": R.internal(name),
                "unit": e.get("unit"),
                "series": [], "min": None, "max": None,
                "first": None, "last": None, "n": 0,
            })
            v = float(e["value"])
            entry["series"].append([round(e["observed_ts"], 3), round(v, 3)])
            entry["n"] += 1
            entry["min"] = v if entry["min"] is None else min(entry["min"], v)
            entry["max"] = v if entry["max"] is None else max(entry["max"], v)
            if entry["first"] is None:
                entry["first"] = v
            entry["last"] = v
        return out

    def get(self, snapshot_id: str) -> Optional[dict]:
        with self._lock:
            snap = self._snaps.get(snapshot_id)
            return dict(snap) if snap else None

    def for_code(self, code: str) -> List[dict]:
        code = (code or "").upper()
        with self._lock:
            return [dict(s) for s in self._snaps.values() if s["code"] == code]

    def stats(self) -> dict:
        with self._lock:
            return {"snapshots": len(self._snaps),
                    "complete": sum(1 for s in self._snaps.values()
                                    if s["complete"]),
                    "capacity": config.VEHICLE_DTC_SNAPSHOT_MAX}

    def clear(self) -> None:
        with self._lock:
            self._snaps.clear()
            self._order.clear()
            self._seq = 0
