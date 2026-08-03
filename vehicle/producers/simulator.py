"""simulator.py — the dashboard simulator, going out the same door as a car.

The point of this file is what it does NOT do. It does not write into the
telemetry buffer, it does not construct a SensorReading, and it does not have a
provider of its own. It builds canonical events and hands them to
vehicle.ingest.ingest_local(), which normalizes them, converts their units,
range-checks them and deduplicates them with exactly the code a bridge's upload
goes through.

That is the whole of what "simulated data uses the same pipeline as live data"
means. It is easy to say and easy to quietly not do — a simulator that reached
into the buffer directly would look identical from the dashboard and would prove
nothing at all, because the first time a real bridge connected it would be
exercising code that had never run.

WHAT IT SKIPS, AND WHY THAT IS HONEST
-------------------------------------
The network and the credentials. There is no network and there is no gateway;
inventing one would be theatre, and a token minted by the process that checks it
proves nothing. Everything downstream of "a well-formed event arrived" is
identical, and everything upstream of it is the part a bridge is for.

DRIVEN, NOT FREE-RUNNING
------------------------
tick() takes the time. No thread, no wall clock of its own, and every scenario is
a pure function of elapsed time — so a whole drive can be run at machine speed in
a test, exactly as tire_diag/selftest.py's Car harness does, and the same code
produces the same numbers when the dashboard drives it at 1 Hz.

THE CADENCE IS PER SIGNAL
-------------------------
§12 asks for different rates per channel — rpm at 2-5 Hz, coolant at 0.2-0.5 Hz —
because that is what a bounded OBD scheduler can actually achieve on one bus.
Emitting every channel on every tick would produce a stream no real vehicle could
ever produce and would let a monitor be tuned against a luxury the hardware does
not offer. So each signal has a period, and a tick emits only the ones that are
due.
"""
from __future__ import annotations

import threading
from typing import Dict, List, Optional

import config

from .. import ingest
from ..signals import registry as R
from ..signals import schema as S
from . import physics

# How often each canonical signal is emitted, in seconds. Straight from §12's
# suggested rates, taking the slower end of each range: a prototype that polls
# at the top of every band is a prototype that floods the bus, and §13's whole
# bus-etiquette section exists because that is the failure mode.
_PERIOD_S: Dict[str, float] = {
    "rpm": 0.5,
    "vehicle_speed": 1.0,
    "throttle_pct": 0.5,
    "map_kpa": 0.5,
    "maf_gs": 0.5,
    "engine_load": 1.0,
    "coolant_temp": 3.0,
    "intake_air_temp": 3.0,
    "stft_b1": 1.0,
    "ltft_b1": 1.0,
    "battery_voltage": 2.0,
    # Holley-only channels. A Terminator X publishes these on its own broadcast
    # cycle rather than being asked, so they are quicker than the polled ones.
    "oil_pressure": 1.0,
    "oil_temp": 3.0,
    "fuel_pressure": 1.0,
    "afr_target": 1.0,
    "afr_wideband": 0.5,
}

_DEFAULT_PERIOD_S = 1.0


class Simulator:
    """One simulated vehicle, emitting canonical events on demand."""

    def __init__(self, scenario: str = None,
                 source_type: str = S.DASHBOARD_SIMULATOR,
                 decoder_version: str = "simulator-1.0"):
        self._lock = threading.RLock()
        self._scenario = physics.resolve(scenario or physics.DEFAULT)
        self._source_type = source_type
        self._decoder_version = decoder_version
        self._t0: Optional[float] = None
        self._last_emit: Dict[str, float] = {}
        self._previous: Dict[str, Optional[float]] = {}
        self._session_id: Optional[str] = None
        self._ticks = 0
        self._emitted = 0

    # -- control ------------------------------------------------------------

    @property
    def scenario(self) -> str:
        with self._lock:
            return self._scenario

    def scenarios(self) -> List[dict]:
        return physics.catalogue()

    def set_scenario(self, name: str, now: float = None) -> bool:
        """Switch scenarios, and restart this one's clock.

        The clock restarts because every scenario is a function of elapsed time:
        a warm-up resumed at t=3600 is not a warm-up, and an overheat that
        remembered where it got to an hour ago would jump 60°F between two
        consecutive readings — which the range check would then correctly refuse
        as an impossible step.
        """
        if name not in physics.BY_NAME:
            return False
        with self._lock:
            self._scenario = name
            self._t0 = now
            self._last_emit.clear()
            self._previous.clear()
        return True

    def set_session(self, session_id: Optional[str]) -> None:
        with self._lock:
            self._session_id = session_id

    def reset(self) -> None:
        with self._lock:
            self._t0 = None
            self._last_emit.clear()
            self._previous.clear()
            self._ticks = 0
            self._emitted = 0

    # -- production ---------------------------------------------------------

    def build(self, now: float) -> List[dict]:
        """The canonical events due at `now`. Pure but for the emission clocks.

        Returned rather than sent, so a caller can inspect them, corrupt them
        (see faults.py) or write them to a recording before they are ingested.
        """
        with self._lock:
            if self._t0 is None:
                self._t0 = now
            sc = physics.BY_NAME[self._scenario]
            elapsed = now - self._t0
            previous = dict(self._previous)
            session = self._session_id
            self._ticks += 1

        values, dropped = physics.sample(sc, elapsed, previous)

        with self._lock:
            self._previous = dict(values)

        events: List[dict] = []
        for telemetry_id, value in values.items():
            name = R.canonical(telemetry_id)
            if name is None:
                continue
            period = _PERIOD_S.get(telemetry_id, _DEFAULT_PERIOD_S)
            with self._lock:
                last = self._last_emit.get(telemetry_id)
                if last is not None and (now - last) < period:
                    continue
                self._last_emit[telemetry_id] = now

            spec = R.spec(name)
            if telemetry_id in dropped:
                # The sensor did not answer. Sent as an event with a null value
                # and a quality that says why, rather than simply omitted:
                # silence and "I asked and got nothing" are different facts, and
                # a pipeline that could not tell them apart would make a dead
                # sensor indistinguishable from a channel nobody polls.
                events.append(S.make_event(
                    name, None, now, self._source_type,
                    vehicle_id=config.VEHICLE_ID, gateway_id="simulator",
                    drive_session_id=session, quality="missing",
                    source_signal=spec.obd_pid if spec else "",
                    decoder_version=self._decoder_version))
                continue
            if value is None:
                continue
            events.append(S.make_event(
                name, value, now, self._source_type,
                vehicle_id=config.VEHICLE_ID, gateway_id="simulator",
                drive_session_id=session,
                source_signal=spec.obd_pid if spec else "",
                source_ecu="0x7E8" if (spec and spec.obd_pid) else "",
                decoder_version=self._decoder_version,
                metadata={"scenario": sc.name}))

        with self._lock:
            self._emitted += len(events)
        return events

    def tick(self, now: float) -> dict:
        """Build this instant's events and push them through the front door."""
        events = self.build(now)
        if not events:
            return {"accepted": 0, "rejected": 0, "built": 0}
        out = ingest.ingest_local(events, now=now)
        out["built"] = len(events)
        return out

    def stats(self) -> dict:
        with self._lock:
            return {"scenario": self._scenario, "ticks": self._ticks,
                    "events_built": self._emitted,
                    "source_type": self._source_type,
                    "drive_session_id": self._session_id,
                    "started_at": self._t0}


# One simulator for the process, like every other stateful thing here.
_simulator = Simulator()


def simulator() -> Simulator:
    return _simulator
