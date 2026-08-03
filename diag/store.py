"""store.py — the diagnostic record, on disk. One Store per domain.

Two things, and the difference between them is the point:

    <prefix>_state.json     what is true NOW. Rewritten whole, atomically.
                            Active issues, monitor counters, relearn epochs.

    <prefix>_events.jsonl   what HAPPENED. Append-only. Never rewritten to say
                            something different, and never trimmed because a
                            problem went away.

Why on disk at all
------------------
Because a diagnostic system whose findings die with the process is a status
light, not a diagnostic system. Three of the required properties are impossible
without persistence:

  - restarting RIO must not erase an active issue
  - restarting RIO must not repeat an alert the driver already heard
  - a recurring problem must stay traceable across drives

The last one is the one that would be quietly lost. "This is the second confirmed
pressure-loss issue on the same tire this month" is only sayable if the first one
is still written down after it was fixed.

And the corollary, which matters more: clearing a cache or restarting the process
must never be able to mark a problem as repaired. Repair is something the healing
criteria establish by observation. It is not something a lost file can assert on
the car's behalf, so the loader below treats a missing state file as "no history
yet", never as "everything is fine".

WHY THIS IS A CLASS NOW
-----------------------
It was a module of globals, which worked exactly as long as there was one
domain. Two domains sharing one file would interleave a coolant finding and a
tire finding in the same state dictionary, keyed by monitor ids that happen not
to collide — a property nobody declared and nothing enforces. Separate stores
make the separation structural instead of lucky.

Following insights.py's pattern, deliberately — same atomic replace, same
one-bad-line-does-not-cost-the-file parsing. This process is polled several times
a second and a half-written state file read on the next tick would take the panel
down with a JSONDecodeError.
"""
from __future__ import annotations

import json
import os
import time
from typing import Dict, List, Optional

STATE_VERSION = 1

DEFAULT_MAX_EVENTS = 4000


def blank_state() -> dict:
    return {"version": STATE_VERSION, "issues": {}, "monitors": {},
            "epochs": {}, "meta": {}}


class Store:
    """Atomic state plus an append-only event log, scoped to one directory."""

    def __init__(self, directory: str, prefix: str,
                 max_events: int = DEFAULT_MAX_EVENTS):
        self._dir = directory
        self._prefix = prefix
        self.max_events = max_events
        self._recompute()

    def _recompute(self) -> None:
        self._state_path = os.path.join(self._dir, f"{self._prefix}_state.json")
        self._events_path = os.path.join(self._dir, f"{self._prefix}_events.jsonl")

    def _ensure_dir(self) -> None:
        os.makedirs(self._dir, exist_ok=True)

    # -- state ---------------------------------------------------------------

    def load_state(self) -> dict:
        """The persisted record, or a blank one.

        A missing or corrupt file is "we have no history", never "the car is
        fine". Those are different claims and only one of them is safe to make
        on no evidence.
        """
        try:
            with open(self._state_path) as fh:
                data = json.load(fh)
        except Exception:
            return blank_state()
        if not isinstance(data, dict) or data.get("version") != STATE_VERSION:
            # A version we do not understand is not something to guess at. Keep
            # the file (it is evidence) and start a fresh record beside it.
            return blank_state()
        for key in ("issues", "monitors", "epochs", "meta"):
            data.setdefault(key, {})
        return data

    def save_state(self, state: dict) -> None:
        self._ensure_dir()
        state["version"] = STATE_VERSION
        state.setdefault("meta", {})["saved_at"] = time.time()
        tmp = self._state_path + ".tmp"
        with open(tmp, "w") as fh:
            json.dump(state, fh, separators=(",", ":"))
        os.replace(tmp, self._state_path)

    # -- events --------------------------------------------------------------

    def append_event(self, kind: str, payload: dict, at: float = None) -> dict:
        """One line in the permanent record.

        Every lifecycle transition, every freeze frame, every announcement and
        every announcement RIO would have made in shadow mode. Append-only: a
        later reading does not get to rewrite what we believed at the time,
        which is the whole value of a freeze frame.
        """
        self._ensure_dir()
        event = {"at": time.time() if at is None else float(at), "kind": kind}
        event.update(payload or {})
        with open(self._events_path, "a") as fh:
            fh.write(json.dumps(event, separators=(",", ":")) + "\n")
        return event

    def read_events(self, limit: int = 200, kinds: Optional[List[str]] = None,
                    issue_id: str = None) -> List[dict]:
        """Recent events, newest last. Diagnostics, tests and the service view."""
        out = []
        try:
            with open(self._events_path) as fh:
                for line in fh:
                    line = line.strip()
                    if not line:
                        continue
                    try:
                        e = json.loads(line)
                    except Exception:
                        # One bad line must not cost the whole history.
                        continue
                    if kinds and e.get("kind") not in kinds:
                        continue
                    if issue_id and e.get("issue_id") != issue_id:
                        continue
                    out.append(e)
        except FileNotFoundError:
            return []
        return out[-limit:]

    def trim_events(self, max_events: int = None) -> int:
        """Cap the log by AGE-ordered count, never by content.

        Nothing here decides that an event is uninteresting. The only thing that
        ever removes a line is that the file has grown past the configured size,
        and then it is the oldest lines that go — never the resolved ones, never
        the ones about a fault that was fixed.
        """
        cap = self.max_events if max_events is None else max_events
        try:
            with open(self._events_path) as fh:
                lines = fh.readlines()
        except FileNotFoundError:
            return 0
        if len(lines) <= cap:
            return 0
        keep = lines[-cap:]
        tmp = self._events_path + ".tmp"
        with open(tmp, "w") as fh:
            fh.writelines(keep)
        os.replace(tmp, self._events_path)
        return len(lines) - len(keep)

    # -- introspection and tests ---------------------------------------------

    def paths(self) -> dict:
        return {"state": self._state_path, "events": self._events_path,
                "dir": self._dir}

    def reset_for_test(self, directory: str) -> None:
        """Point this store somewhere disposable. Tests only.

        Explicit rather than a module-level flag: a test that forgets to call
        this would otherwise write into the real diagnostic history, and a test
        suite that can fabricate a car's fault record is worse than no test
        suite.

        Mutates in place rather than returning a new Store, because an engine
        built before the call is holding a reference to this one.
        """
        self._dir = directory
        self._recompute()
        os.makedirs(self._dir, exist_ok=True)
        for p in (self._state_path, self._events_path):
            try:
                os.remove(p)
            except FileNotFoundError:
                pass
