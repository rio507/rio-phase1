"""store.py — the diagnostic record, on disk.

Two things, and the difference between them is the point:

    tire_diag_state.json    what is true NOW. Rewritten whole, atomically.
                            Active issues, monitor counters, relearn epochs.

    tire_diag_events.jsonl  what HAPPENED. Append-only. Never rewritten to say
                            something different, and never trimmed because a
                            problem went away.

Why on disk at all
------------------
Because a diagnostic system whose findings die with the process is a status
light, not a diagnostic system. Three of the properties the spec asks for are
impossible without persistence:

  - restarting RIO must not erase an active issue
  - restarting RIO must not repeat an alert the driver already heard
  - a recurring problem must stay traceable across drives

The last one is the one that would be quietly lost. "This is the second
confirmed pressure-loss issue on the same tire this month" is only sayable if
the first one is still written down after it was fixed.

And the corollary, which matters more: clearing a cache or restarting the
process must never be able to mark a problem as repaired. Repair is something
the healing criteria establish by observation. It is not something a lost file
can assert on the car's behalf, so the loader below treats a missing state file
as "no history yet", never as "everything is fine".

Following insights.py's pattern, deliberately — same directory, same atomic
replace, same one-bad-line-does-not-cost-the-file parsing. This process is
polled several times a second and a half-written state file read on the next
tick would take the panel down with a JSONDecodeError.
"""
from __future__ import annotations

import json
import os
import time
from typing import Dict, List, Optional

import config

_DIR = config.TIRE_DIAG_DIR
_STATE_PATH = os.path.join(_DIR, "tire_diag_state.json")
_EVENTS_PATH = os.path.join(_DIR, "tire_diag_events.jsonl")

STATE_VERSION = 1


def _ensure_dir() -> None:
    os.makedirs(_DIR, exist_ok=True)


def _blank_state() -> dict:
    return {"version": STATE_VERSION, "issues": {}, "monitors": {},
            "epochs": {}, "meta": {}}


def load_state() -> dict:
    """The persisted record, or a blank one.

    A missing or corrupt file is "we have no history", never "the car is fine".
    Those are different claims and only one of them is safe to make on no
    evidence.
    """
    try:
        with open(_STATE_PATH) as fh:
            data = json.load(fh)
    except Exception:
        return _blank_state()
    if not isinstance(data, dict) or data.get("version") != STATE_VERSION:
        # A version we do not understand is not something to guess at. Keep the
        # file (it is evidence) and start a fresh record beside it.
        return _blank_state()
    for key in ("issues", "monitors", "epochs", "meta"):
        data.setdefault(key, {})
    return data


def save_state(state: dict) -> None:
    _ensure_dir()
    state["version"] = STATE_VERSION
    state.setdefault("meta", {})["saved_at"] = time.time()
    tmp = _STATE_PATH + ".tmp"
    with open(tmp, "w") as fh:
        json.dump(state, fh, separators=(",", ":"))
    os.replace(tmp, _STATE_PATH)


def append_event(kind: str, payload: dict, at: float = None) -> dict:
    """One line in the permanent record.

    Every lifecycle transition, every freeze frame, every announcement and every
    announcement RIO would have made in shadow mode. Append-only: a later
    reading does not get to rewrite what we believed at the time, which is the
    whole value of a freeze frame.
    """
    _ensure_dir()
    event = {"at": time.time() if at is None else float(at), "kind": kind}
    event.update(payload or {})
    with open(_EVENTS_PATH, "a") as fh:
        fh.write(json.dumps(event, separators=(",", ":")) + "\n")
    return event


def read_events(limit: int = 200, kinds: Optional[List[str]] = None,
                issue_id: str = None) -> List[dict]:
    """Recent events, newest last. Diagnostics, tests and the service view."""
    out = []
    try:
        with open(_EVENTS_PATH) as fh:
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


def trim_events(max_events: int = None) -> int:
    """Cap the log by AGE-ordered count, never by content.

    Nothing here decides that an event is uninteresting. The only thing that
    ever removes a line is that the file has grown past the configured size, and
    then it is the oldest lines that go — never the resolved ones, never the
    ones about a tire that was fixed.
    """
    cap = config.TIRE_DIAG_MAX_EVENTS if max_events is None else max_events
    try:
        with open(_EVENTS_PATH) as fh:
            lines = fh.readlines()
    except FileNotFoundError:
        return 0
    if len(lines) <= cap:
        return 0
    keep = lines[-cap:]
    tmp = _EVENTS_PATH + ".tmp"
    with open(tmp, "w") as fh:
        fh.writelines(keep)
    os.replace(tmp, _EVENTS_PATH)
    return len(lines) - len(keep)


def paths() -> dict:
    return {"state": _STATE_PATH, "events": _EVENTS_PATH, "dir": _DIR}


def reset_for_test(directory: str) -> None:
    """Point the store somewhere disposable. Tests only.

    Explicit rather than a module-level flag: a test that forgets to call this
    would otherwise write into the real diagnostic history, and a test suite
    that can fabricate a car's fault record is worse than no test suite.
    """
    global _DIR, _STATE_PATH, _EVENTS_PATH
    _DIR = directory
    _STATE_PATH = os.path.join(_DIR, "tire_diag_state.json")
    _EVENTS_PATH = os.path.join(_DIR, "tire_diag_events.jsonl")
    os.makedirs(_DIR, exist_ok=True)
    for p in (_STATE_PATH, _EVENTS_PATH):
        try:
            os.remove(p)
        except FileNotFoundError:
            pass
