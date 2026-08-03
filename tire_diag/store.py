"""store.py — the tire diagnostic record, on disk.

The implementation moved to diag/store.py when the engine monitors needed the
same thing. What is left here is the tire domain's own Store — its directory,
its file prefix, its event cap — plus the module-level functions the rest of the
codebase already calls.

Those functions are not legacy cruft to be tidied away later. app.py's
/vehicle/diagnostics/events reads `read_events`, the selftests call
`reset_for_test`, and keeping the module API means the extraction of diag/
changed no caller anywhere. The wrapper is three lines per function and it buys
a refactor that bisects cleanly.

Everything the old header said about WHY this is on disk still holds and now
lives in diag/store.py, where both domains can read it.
"""
from __future__ import annotations

from typing import List, Optional

import config

from diag.store import STATE_VERSION, Store, blank_state

# The tire domain's store. Held as a module-level instance rather than rebuilt
# per call so that reset_for_test can move it and every engine already holding a
# reference follows — see Store.reset_for_test on why it mutates in place.
STORE = Store(config.TIRE_DIAG_DIR, "tire_diag",
              max_events=config.TIRE_DIAG_MAX_EVENTS)


def _blank_state() -> dict:
    return blank_state()


def load_state() -> dict:
    return STORE.load_state()


def save_state(state: dict) -> None:
    STORE.save_state(state)


def append_event(kind: str, payload: dict, at: float = None) -> dict:
    return STORE.append_event(kind, payload, at=at)


def read_events(limit: int = 200, kinds: Optional[List[str]] = None,
                issue_id: str = None) -> List[dict]:
    return STORE.read_events(limit=limit, kinds=kinds, issue_id=issue_id)


def trim_events(max_events: int = None) -> int:
    return STORE.trim_events(max_events)


def paths() -> dict:
    return STORE.paths()


def reset_for_test(directory: str) -> None:
    """Point the store somewhere disposable. Tests only.

    Explicit rather than a module-level flag: a test that forgets to call this
    would otherwise write into the real diagnostic history, and a test suite
    that can fabricate a car's fault record is worse than no test suite.
    """
    STORE.reset_for_test(directory)
