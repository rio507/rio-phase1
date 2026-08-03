"""replay.py — a drive, recorded once and run again as often as needed.

Two halves that only make sense together:

    Recorder    every canonical event a producer built, to a JSONL file
    Replay      that file, back through the same front door

WHY RECORD CANONICAL EVENTS AND NOT CAN FRAMES
----------------------------------------------
Raw frames are recorded too — that is the bridge's job, and §29.1 keeps them for
seven days for exactly the debugging a decoder needs. But a raw capture can only
be replayed through the decoder that produced it, so a decoder change invalidates
every recording ever made, and the recordings that matter most are the ones from
the drive where something odd happened.

Canonical events survive that. A recorded drive replays through the ingestion
API, the bands, the monitors, the insight engine and the conversation layer, and
it keeps working when the decoder is rewritten. It is the difference between a
regression test and a museum piece.

WHAT IS REWRITTEN ON REPLAY, AND WHAT IS NOT
--------------------------------------------
    event_id      re-minted. Deduplication is by event_id and it works: a
                  second replay of the same file would otherwise be silently
                  discarded in full, which is the single most confusing failure
                  this feature could have.

    observed_at   shifted so the recording starts now. A drive from last Tuesday
                  replayed with its original timestamps is a drive every
                  staleness rule correctly refuses.

    source_type   recorded_replay
    provenance    recorded_replay
    quality       recorded_replay, where it was previously a measurement

Those four exist so that nothing downstream can mistake a replay for a live
vehicle. Everything else — the values, the units, the ECU, the decoder version,
the RELATIVE timing — is exactly what was recorded. A replay that quietly
smoothed its own timing would be useless for reproducing the bug you recorded it
to reproduce.

ACCELERATED PLAYBACK
--------------------
`speed` compresses the relative timing. A two-hour drive at 120x is a minute, and
every monitor sees the same sequence of values in the same order — which is what
makes an hour-long slow-leak window testable in CI. What it does NOT do is change
the observed_at spacing: a monitor asking "how long did this take" gets the
answer the drive gave, not the answer the test runner gave.
"""
from __future__ import annotations

import json
import os
import threading
from typing import Dict, List, Optional

from .. import ingest
from ..signals import provenance as P
from ..signals import quality as Q
from ..signals import schema as S


class Recorder:
    """Canonical events to a JSONL file, one per line, as they are produced."""

    def __init__(self, path: str):
        self.path = path
        self._lock = threading.Lock()
        self._count = 0
        os.makedirs(os.path.dirname(path) or ".", exist_ok=True)

    def write(self, events: List[dict]) -> int:
        if not events:
            return 0
        with self._lock:
            with open(self.path, "a") as fh:
                for e in events:
                    fh.write(json.dumps(e, separators=(",", ":")) + "\n")
            self._count += len(events)
        return len(events)

    def stats(self) -> dict:
        with self._lock:
            return {"path": self.path, "events": self._count}


def read_log(path: str) -> List[dict]:
    """Every event in a recording, in the order it was written.

    One bad line does not cost the file — the same discipline insights.py and
    the diagnostic store already apply, and for the same reason: a recording is
    evidence, and evidence with a torn page is still evidence.
    """
    out = []
    try:
        with open(path) as fh:
            for line in fh:
                line = line.strip()
                if not line:
                    continue
                try:
                    out.append(json.loads(line))
                except Exception:
                    continue
    except FileNotFoundError:
        return []
    return out


class Replay:
    """A recording, played back through the canonical ingestion path."""

    def __init__(self):
        self._lock = threading.RLock()
        self._events: List[dict] = []
        self._base_ts: Optional[float] = None      # first observed_at in the file
        self._started_at: Optional[float] = None   # wall time playback began
        self._cursor = 0
        self._speed = 1.0
        self._path = ""
        self._loops = 0
        self._loop = False
        self._session_id: Optional[str] = None

    # -- control ------------------------------------------------------------

    def load(self, path: str) -> dict:
        """-> a summary of what was loaded. Nothing is sent until start()."""
        events = read_log(path)
        stamped = []
        for e in events:
            ts = S.from_iso(e.get("observed_at"))
            if ts is None:
                continue
            stamped.append((ts, e))
        stamped.sort(key=lambda pair: pair[0])
        with self._lock:
            self._events = [e for _, e in stamped]
            self._base_ts = stamped[0][0] if stamped else None
            self._cursor = 0
            self._started_at = None
            self._path = path
            self._loops = 0
        return self.stats()

    def start(self, now: float, speed: float = 1.0, loop: bool = False,
              session_id: str = None) -> dict:
        with self._lock:
            self._started_at = now
            self._cursor = 0
            self._speed = max(0.01, float(speed))
            self._loop = bool(loop)
            self._session_id = session_id
        return self.stats()

    def stop(self) -> dict:
        with self._lock:
            self._started_at = None
        return self.stats()

    @property
    def running(self) -> bool:
        with self._lock:
            return self._started_at is not None and bool(self._events)

    # -- production ---------------------------------------------------------

    def build(self, now: float) -> List[dict]:
        """The recorded events whose moment has arrived, rewritten for today."""
        with self._lock:
            if self._started_at is None or not self._events \
                    or self._base_ts is None:
                return []
            elapsed = (now - self._started_at) * self._speed
            out = []
            while self._cursor < len(self._events):
                e = self._events[self._cursor]
                offset = (S.from_iso(e.get("observed_at")) or 0.0) - self._base_ts
                if offset > elapsed:
                    break
                self._cursor += 1
                out.append(self._rewrite(e, self._started_at + offset / self._speed))
            if self._cursor >= len(self._events):
                if self._loop:
                    self._cursor = 0
                    self._started_at = now
                    self._loops += 1
                else:
                    self._started_at = None
            return out

    def _rewrite(self, e: dict, observed_at: float) -> dict:
        """One recorded event, made safe to mistake for nothing.

        See the module header on why exactly these four fields change and no
        others.
        """
        out = dict(e)
        out["event_id"] = S.new_id()
        out["observed_at"] = S.to_iso(observed_at)
        out["received_at"] = None
        out["source_type"] = S.RECORDED_REPLAY
        out["provenance"] = P.RECORDED_REPLAY
        # A quality that was already a complaint stays a complaint: a decode
        # error in the recording is a decode error on replay, and upgrading it
        # to "replayed" would hide the very thing the recording captured.
        if out.get("quality") in (Q.VALID, None, ""):
            out["quality"] = Q.RECORDED_REPLAY
        meta = dict(out.get("metadata") or {})
        meta["replayed_from"] = os.path.basename(self._path)
        meta["original_observed_at"] = e.get("observed_at")
        out["metadata"] = meta
        return out

    def tick(self, now: float) -> dict:
        events = self.build(now)
        if not events:
            return {"accepted": 0, "rejected": 0, "built": 0,
                    "running": self.running}
        out = ingest.ingest_local(events, now=now)
        out["built"] = len(events)
        out["running"] = self.running
        return out

    def stats(self) -> dict:
        with self._lock:
            total = len(self._events)
            span = None
            if total and self._base_ts is not None:
                last = S.from_iso(self._events[-1].get("observed_at"))
                span = None if last is None else round(last - self._base_ts, 1)
            return {
                "path": self._path,
                "events": total,
                "cursor": self._cursor,
                "running": self._started_at is not None and bool(self._events),
                "speed": self._speed,
                "loop": self._loop,
                "loops_completed": self._loops,
                "recorded_span_s": span,
                "drive_session_id": self._session_id,
            }


_replay = Replay()


def replay() -> Replay:
    return _replay
