"""store.py — the powertrain diagnostic record, on disk.

Its own Store, in its own files, beside the tire domain's. Separate because two
domains sharing one state file would interleave a coolant finding and a tire
finding in the same dictionary, keyed by monitor ids that happen not to collide —
a property nobody declared and nothing enforces. See diag/store.py.
"""
from __future__ import annotations

from typing import List, Optional

import config

from diag.store import Store, blank_state

STORE = Store(config.VEHICLE_DIAG_DIR, "powertrain_diag",
              max_events=config.VEHICLE_DIAG_MAX_EVENTS)


def load_state() -> dict:
    return STORE.load_state()


def save_state(state: dict) -> None:
    STORE.save_state(state)


def append_event(kind: str, payload: dict, at: float = None) -> dict:
    return STORE.append_event(kind, payload, at=at)


def read_events(limit: int = 200, kinds: Optional[List[str]] = None,
                issue_id: str = None) -> List[dict]:
    return STORE.read_events(limit=limit, kinds=kinds, issue_id=issue_id)


def paths() -> dict:
    return STORE.paths()


def reset_for_test(directory: str) -> None:
    """Point the store somewhere disposable. Tests only."""
    STORE.reset_for_test(directory)
