"""ecu.py — a simulated ECU that answers diagnostic scans.

physics.py models what the engine is DOING. This models what the engine's
computer has DECIDED about it, and they are separate on purpose: an ECU sets a
code on evidence RIO cannot see (misfire counts, monitor completion, trip
counters), and pretending to derive one from the other would be inventing a
mechanism no real vehicle has.

So this is a scripted timeline. Each scenario is a list of moments and what the
ECU reports from that moment on, which is exactly what a scan of a real vehicle
returns: a list of codes and a lamp state, with no explanation of how they got
there.

WHAT A SCAN LOOKS LIKE
----------------------
    {"pending": ["P0171"], "stored": [], "permanent": [],
     "mil": False, "dtc_count": 1,
     "supported": {"pending": True, "stored": True, "permanent": False},
     "freeze_frames": {"P0171": {...}}}

`supported` is not decoration. A vehicle that does not answer Mode 0A has no
permanent codes, which is a different fact from having none — and an
implementation that treated an unsupported service as an empty result would
report "no permanent codes" about a car it never asked.

READ-ONLY
---------
Modes 01, 02, 03, 07, 09 and 0A. There is no Mode 04 here, no method that could
become one, and no way for a caller to ask this object to change anything about
the vehicle it models. Even the simulated ECU refuses to have the capability,
because a mock with a clear-codes method is a mock somebody will one day wire to
something real.
"""
from __future__ import annotations

import threading
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

# One freeze frame, as an ECU records it: the conditions when the fault was set.
# Deliberately sparse — a real freeze frame carries the handful of PIDs that
# vehicle supports and no more, and a mock that returned everything would let
# the UI be built against a richness the hardware does not offer.
_LEAN_FREEZE = {
    "powertrain.engine.rpm": 842.0,
    "vehicle.speed": 0.0,
    "powertrain.engine.coolant_temperature": 197.0,
    "powertrain.engine.calculated_load": 24.0,
    "powertrain.fuel.short_term_trim_bank_1": 9.4,
    "powertrain.fuel.long_term_trim_bank_1": 18.8,
    "source_ecu": "0x7E8",
}

_COOLANT_FREEZE = {
    "powertrain.engine.rpm": 1880.0,
    "vehicle.speed": 61.0,
    "powertrain.engine.coolant_temperature": 246.0,
    "powertrain.engine.calculated_load": 58.0,
    "source_ecu": "0x7E8",
}


@dataclass(frozen=True)
class Moment:
    """From `at` seconds into the scenario, this is what a scan returns."""
    at: float
    pending: Tuple[str, ...] = ()
    stored: Tuple[str, ...] = ()
    permanent: Tuple[str, ...] = ()
    mil: bool = False


@dataclass(frozen=True)
class EcuScenario:
    name: str
    label: str
    timeline: Tuple[Moment, ...]
    # Which diagnostic services this vehicle answers. Mode 0A is genuinely
    # absent on a lot of 2008-2010 cars, and the prototype has to handle that
    # gracefully rather than reporting an empty list as an all-clear.
    supports_pending: bool = True
    supports_stored: bool = True
    supports_permanent: bool = True
    supports_freeze_frame: bool = True
    freeze_frames: Dict[str, dict] = field(default_factory=dict)


_LEAN_FF = {"P0171": _LEAN_FREEZE}
_COOL_FF = {"P0217": _COOLANT_FREEZE}


SCENARIOS: Tuple[EcuScenario, ...] = (
    EcuScenario("healthy", "No Codes", (Moment(0.0),)),

    # The product advantage, in its simplest form: the ECU has noticed and the
    # dashboard has not lit up.
    EcuScenario("pending_mil_off", "Pending Code, Light Off", (
        Moment(0.0),
        Moment(20.0, pending=("P0171",)),
    ), freeze_frames=_LEAN_FF),

    # The same code seen again on a later scan. Not a promotion and not a
    # prediction of one — just a recurrence, which is more informative than a
    # first sighting and is not the same as confirmation.
    EcuScenario("pending_repeated", "Pending Code Seen Again", (
        Moment(0.0),
        Moment(20.0, pending=("P0171",)),
        Moment(140.0, pending=("P0171",)),
    ), freeze_frames=_LEAN_FF),

    # Pending, then the ECU confirms it and commands the lamp.
    EcuScenario("pending_to_confirmed", "Pending Becomes Confirmed", (
        Moment(0.0),
        Moment(20.0, pending=("P0171",)),
        Moment(120.0, stored=("P0171",), mil=True),
    ), freeze_frames=_LEAN_FF),

    # Confirmed and permanent. A permanent code survives a battery
    # disconnection and cannot be cleared by a scan tool, which is exactly why
    # §16.6 says to display it separately and offer no clearing control.
    EcuScenario("permanent", "Permanent Code", (
        Moment(0.0),
        Moment(20.0, stored=("P0217",), mil=True),
        Moment(90.0, stored=("P0217",), permanent=("P0217",), mil=True),
    ), freeze_frames=_COOL_FF),

    # The lamp coming on with no new code: the ECU confirming something it had
    # been watching.
    EcuScenario("mil_on", "Warning Light Comes On", (
        Moment(0.0, pending=("P0300",)),
        Moment(60.0, stored=("P0300",), mil=True),
    )),

    # A pending code that simply stops being reported. It stays in the history
    # and RIO does not call it repaired.
    EcuScenario("code_disappears", "Code Disappears", (
        Moment(0.0),
        Moment(20.0, pending=("P0171",)),
        Moment(120.0),
    ), freeze_frames=_LEAN_FF),

    # ...and one that comes back, which is the reason keeping the history
    # matters at all.
    EcuScenario("code_recurs", "Code Returns", (
        Moment(0.0),
        Moment(20.0, pending=("P0171",)),
        Moment(120.0),
        Moment(220.0, pending=("P0171",)),
    ), freeze_frames=_LEAN_FF),

    # A well-formed code with no catalogue entry. It must survive whole.
    EcuScenario("unknown_code", "Unrecognised Code", (
        Moment(0.0),
        Moment(20.0, stored=("P0468",), mil=True),
    )),

    # Manufacturer-specific. RIO can say the vehicle reported it and must not
    # guess what it means.
    EcuScenario("manufacturer_code", "Manufacturer-Specific Code", (
        Moment(0.0),
        Moment(20.0, pending=("P1614",)),
    )),

    # Several at once, of different severities, so the section's ordering and
    # grouping are exercised rather than assumed.
    EcuScenario("multiple", "Several Codes At Once", (
        Moment(0.0),
        Moment(20.0, pending=("P0171", "P0135")),
        Moment(100.0, pending=("P0135",), stored=("P0171", "P0300"), mil=True),
    ), freeze_frames=_LEAN_FF),

    # A vehicle that does not answer Mode 0A. "No permanent codes" and "we
    # could not ask" are different answers.
    EcuScenario("no_permanent_support", "No Mode 0A Support", (
        Moment(0.0),
        Moment(20.0, stored=("P0171",), mil=True),
    ), supports_permanent=False, freeze_frames=_LEAN_FF),

    # An ECU that has stopped answering diagnostic requests entirely. Not the
    # same as a healthy car, and the dashboard must not read it as one.
    EcuScenario("no_response", "ECU Not Answering", ()),
)

BY_NAME: Dict[str, EcuScenario] = {s.name: s for s in SCENARIOS}
DEFAULT = SCENARIOS[0].name


def resolve(name: str) -> str:
    return name if name in BY_NAME else DEFAULT


def catalogue() -> List[dict]:
    return [{"name": s.name, "label": s.label} for s in SCENARIOS]


class MockEcu:
    """A vehicle's diagnostic side, on a script."""

    def __init__(self, scenario: str = None):
        self._lock = threading.RLock()
        self._scenario = resolve(scenario or DEFAULT)
        self._t0: Optional[float] = None

    @property
    def scenario(self) -> str:
        with self._lock:
            return self._scenario

    def scenarios(self) -> List[dict]:
        return catalogue()

    def set_scenario(self, name: str, now: float = None) -> bool:
        if name not in BY_NAME:
            return False
        with self._lock:
            self._scenario = name
            self._t0 = now
        return True

    def reset(self) -> None:
        with self._lock:
            self._t0 = None

    def responding(self, now: float) -> bool:
        """False when the ECU is not answering diagnostic requests at all."""
        with self._lock:
            return bool(BY_NAME[self._scenario].timeline)

    def scan(self, now: float, drive_session_id: str = None) -> Optional[dict]:
        """One complete diagnostic scan. None when the ECU is not answering.

        None rather than an empty scan, because they mean opposite things: an
        empty scan is a vehicle saying "no codes", and no answer is a vehicle
        saying nothing. A caller that conflated them would report an all-clear
        for a car it could not reach.
        """
        with self._lock:
            if self._t0 is None:
                self._t0 = now
            sc = BY_NAME[self._scenario]
            elapsed = now - self._t0

        if not sc.timeline:
            return None

        moment = sc.timeline[0]
        for m in sc.timeline:
            if elapsed >= m.at:
                moment = m
            else:
                break

        pending = list(moment.pending) if sc.supports_pending else []
        stored = list(moment.stored) if sc.supports_stored else []
        permanent = list(moment.permanent) if sc.supports_permanent else []

        frames = {}
        if sc.supports_freeze_frame:
            for code in set(stored) | set(pending):
                if code in sc.freeze_frames:
                    frames[code] = dict(sc.freeze_frames[code])

        return {
            "pending": pending,
            "stored": stored,
            "permanent": permanent,
            "mil": bool(moment.mil),
            # Mode 01 PID 01's own count. Reported separately from the code
            # lists on purpose: a disagreement between them is a real and
            # informative fault, and deriving one from the other would hide it.
            "dtc_count": len(stored),
            "supported": {
                "pending": sc.supports_pending,
                "stored": sc.supports_stored,
                "permanent": sc.supports_permanent,
                "freeze_frame": sc.supports_freeze_frame,
            },
            # Mode 02. The conditions the VEHICLE recorded when it set the
            # fault — kept apart from RIO's own snapshot everywhere downstream,
            # because one is authoritative about that moment and the other is
            # what RIO happened to be watching.
            "freeze_frames": frames,
            "drive_session_id": drive_session_id,
            "scenario": sc.name,
            "at": now,
        }

    def stats(self) -> dict:
        with self._lock:
            return {"scenario": self._scenario, "started_at": self._t0,
                    "scenarios": catalogue()}


_ecu = MockEcu()


def ecu() -> MockEcu:
    return _ecu
