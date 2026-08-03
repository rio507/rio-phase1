"""producers — the things that make vehicle data, and the pump that runs them.

    physics.py     one engine model, shared by the mock and the simulator
    simulator.py   canonical events from that model, through the front door
    replay.py      a recorded drive, through the same front door

THE PUMP
--------
An in-process producer has no clock of its own, on purpose: `tick(now)` takes the
time so a whole drive can be run at machine speed in a test and at 1 Hz on the
dashboard with the same code producing the same numbers.

Something has to call it, and that something is telemetry.snapshot() — but only
on a RECORDING snapshot. This is the same discipline snapshot() already applies
to the trend ring and the insight engine, and it exists for the same reason: a
conversation turn reads the pipeline at whatever cadence the driver is talking,
and a producer advanced by being asked about would report a drive whose length
depended on how chatty somebody was.

WHY THE FAULT INJECTOR SITS HERE
--------------------------------
Between building the events and ingesting them, which is exactly where a link
lives. A producer builds what the car did; the injector decides what survives the
trip; ingestion decides what to believe. Putting the injector inside the producer
would have let it fabricate readings, and putting it inside ingestion would have
made the cloud complicit in its own confusion.
"""
from __future__ import annotations

from typing import Optional

from .. import faults, ingest
from . import replay as replay_mod
from . import simulator as simulator_mod

# Deliberately NOT `from .simulator import simulator`, and no accessor of that
# name here either. Either one rebinds the submodule attribute on this package
# to a function, so `from vehicle.producers import simulator` hands a caller the
# function and `simulator.Simulator` fails with a message about a function
# having no attribute — a genuinely baffling five minutes, and one this comment
# exists to prevent somebody spending twice.
#
# The accessors live on the submodules: producers.simulator.simulator() and
# producers.replay.replay().


def _pump(events, now: float) -> dict:
    if not events:
        # Even with nothing to send, a cleared fault may have a backlog to
        # release. An outbox does not wait for new data to drain.
        events = []
    out_events = faults.injector().apply(events, now)
    if not out_events:
        return {"built": len(events), "accepted": 0, "rejected": 0,
                "sent": 0}
    res = ingest.ingest_local(out_events, now=now)
    res["built"] = len(events)
    res["sent"] = len(out_events)
    return res


def pump_simulation(now: float) -> dict:
    """One tick of the simulator, through the injector and the front door."""
    return _pump(simulator_mod.simulator().build(now), now)


def pump_replay(now: float) -> dict:
    """One tick of the replay, through the injector and the front door."""
    return _pump(replay_mod.replay().build(now), now)


def stats() -> dict:
    return {"simulator": simulator_mod.simulator().stats(),
            "replay": replay_mod.replay().stats(),
            "faults": faults.injector().stats()}
