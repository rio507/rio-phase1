"""faults.py — breaking the link on purpose.

§27.9 asks the simulator to reproduce twenty-odd conditions, and they are not all
the same kind of thing. Half are about the ENGINE — high coolant, fuel-trim
drift, a frozen sensor — and those are scenarios in producers/physics.py, because
they are facts about the car. The other half are about the LINK:

    gateway disconnect      the bridge stopped talking
    cloud disconnect        the bridge is fine and cannot upload
    duplicate upload        a retry the bridge could not know was unnecessary
    out-of-order timestamps a backlog arriving behind the live stream
    clock skew              a gateway whose clock is wrong

No physics model can produce any of those, because none of them is about the
engine. They are properties of the transport, and a simulator that could only
break the car would leave the entire recovery path — the part most likely to be
wrong and least likely to be exercised — untested until a real tunnel.

WHY CLOUD DISCONNECT IS NOT GATEWAY DISCONNECT
----------------------------------------------
They look identical from the dashboard for the first few seconds and they are
opposite problems. A gateway that has stopped talking has stopped MEASURING: that
data does not exist and never will. A gateway that cannot upload is still
measuring and still buffering, and every one of those readings arrives later.
Conflating them means either mourning data that is about to turn up, or
cheerfully waiting for data that is gone. So the first drops events and the
second holds them, and the held ones are released on reconnect — which is what
produces the out-of-order arrivals the buffer has to handle.

THIS IS DEVELOPMENT MACHINERY
-----------------------------
Enabled explicitly, never by default, and it can only ever make the pipeline's
job harder. There is no fault here that fabricates a reading: the worst it does
is duplicate, delay, reorder or drop events that a producer genuinely built.
"""
from __future__ import annotations

import threading
from dataclasses import dataclass, field
from typing import Dict, List, Optional

from .signals import schema as S

# --- the fault vocabulary ---------------------------------------------------
NONE = "none"
GATEWAY_DISCONNECT = "gateway_disconnect"
CLOUD_DISCONNECT = "cloud_disconnect"
DUPLICATE_UPLOAD = "duplicate_upload"
OUT_OF_ORDER = "out_of_order"
CLOCK_SKEW = "clock_skew"

ALL = (NONE, GATEWAY_DISCONNECT, CLOUD_DISCONNECT, DUPLICATE_UPLOAD,
       OUT_OF_ORDER, CLOCK_SKEW)

DESCRIPTION = {
    NONE: "No injected fault.",
    GATEWAY_DISCONNECT: "The bridge has stopped measuring. Events are lost, "
                        "not delayed — this data will never arrive.",
    CLOUD_DISCONNECT: "The bridge is measuring and cannot upload. Events are "
                      "held and delivered on reconnect, oldest first.",
    DUPLICATE_UPLOAD: "Every batch is delivered twice, as a bridge retrying "
                      "one it never saw acknowledged would.",
    OUT_OF_ORDER: "Each tick's events are delivered after the following "
                  "tick's, as a backlog draining behind a live stream does.",
    CLOCK_SKEW: "The gateway's clock runs ahead of the cloud's.",
}

# How far ahead the skewed clock runs. Comfortably past
# schema.MAX_FUTURE_SKEW_S, so the condition is actually reached rather than
# sitting just inside tolerance and proving nothing.
_SKEW_S = 600.0


@dataclass
class _State:
    mode: str = NONE
    held: List[dict] = field(default_factory=list)
    deferred: List[dict] = field(default_factory=list)
    last_batch: List[dict] = field(default_factory=list)
    reorder_holding: bool = False
    dropped: int = 0
    duplicated: int = 0
    reordered: int = 0


class FaultInjector:
    """Applied to a producer's events on their way to the front door."""

    def __init__(self):
        self._lock = threading.RLock()
        self._s = _State()

    # -- control ------------------------------------------------------------

    @property
    def mode(self) -> str:
        with self._lock:
            return self._s.mode

    def set_mode(self, mode: str) -> bool:
        """-> False on an unknown mode. Leaving a fault releases what it held.

        Held events are RELEASED rather than discarded when cloud disconnect is
        cleared, because that is what a reconnecting bridge does: the outbox
        empties. Discarding them would make the fault a data-loss fault, which
        is the other one.
        """
        if mode not in ALL:
            return False
        with self._lock:
            previous = self._s.mode
            self._s.mode = mode
            if previous == CLOUD_DISCONNECT and mode != CLOUD_DISCONNECT:
                self._s.deferred.extend(self._s.held)
                self._s.held = []
        return True

    def reset(self) -> None:
        with self._lock:
            self._s = _State()

    def stats(self) -> dict:
        with self._lock:
            return {
                "mode": self._s.mode,
                "description": DESCRIPTION.get(self._s.mode, ""),
                "held": len(self._s.held),
                "pending_release": len(self._s.deferred),
                "dropped": self._s.dropped,
                "duplicated": self._s.duplicated,
                "reordered": self._s.reordered,
                "modes": [{"name": m, "description": DESCRIPTION[m]} for m in ALL],
            }

    # -- the transform ------------------------------------------------------

    def apply(self, events: List[dict], now: float) -> List[dict]:
        """-> the events that actually reach ingestion this tick."""
        with self._lock:
            mode = self._s.mode
            # Anything a cleared fault released goes out first, oldest first,
            # exactly as an outbox drains.
            out: List[dict] = []
            if self._s.deferred:
                out.extend(self._s.deferred)
                self._s.deferred = []

            if mode == GATEWAY_DISCONNECT:
                # Lost, not delayed. A bridge that has stopped measuring has no
                # backlog to deliver later.
                self._s.dropped += len(events)
                return out

            if mode == CLOUD_DISCONNECT:
                # Measured and buffered. These arrive on reconnect, which is
                # what produces the out-of-order case the buffer must survive.
                self._s.held.extend(events)
                return out

            if mode == DUPLICATE_UPLOAD:
                # A genuine retry: the SAME event ids, because that is what
                # makes it a duplicate rather than a second measurement. An
                # injector that re-minted the ids would be testing nothing —
                # deduplication is by event_id and new ids are new events.
                self._s.duplicated += len(events)
                out.extend(events)
                out.extend([dict(e) for e in events])
                return out

            if mode == OUT_OF_ORDER:
                # Alternate ticks are held and then delivered BEHIND the
                # following tick's, so the buffer sees a stale reading arrive
                # after a fresh one on the same signal — which is what a backlog
                # draining behind a live stream actually looks like.
                #
                # Every event is still sent exactly once. The obvious
                # implementation — re-send the previous tick's events alongside
                # this tick's — does not work and fails in an instructive way:
                # those events keep their ids, deduplication correctly discards
                # them, and the reordering that was supposed to be under test
                # never reaches the buffer at all.
                self._s.reorder_holding = not self._s.reorder_holding
                if self._s.reorder_holding:
                    self._s.last_batch = list(events)
                    return out
                backlog = self._s.last_batch
                self._s.last_batch = []
                self._s.reordered += len(backlog)
                out.extend(events)          # newer
                out.extend(backlog)         # older, arriving behind
                return out

            if mode == CLOCK_SKEW:
                skewed = []
                for e in events:
                    e = dict(e)
                    ts = S.from_iso(e.get("observed_at"))
                    if ts is not None:
                        e["observed_at"] = S.to_iso(ts + _SKEW_S)
                    skewed.append(e)
                out.extend(skewed)
                return out

            out.extend(events)
            return out


_injector = FaultInjector()


def injector() -> FaultInjector:
    return _injector
