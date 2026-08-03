"""shadow.py — per-domain speech clearance.

Shadow mode means: detect, confirm, freeze the evidence, compose the exact words
RIO would have said, write all of it down, and say nothing. It is how a monitor
earns the right to interrupt somebody — by first producing a log that answers
"how often would this have interrupted somebody" from real drives.

WHY THIS IS NOT ONE BOOLEAN ANY MORE
------------------------------------
It was `config.TIRE_DIAG_SHADOW_MODE`, read in four places, and that was correct
while there was one domain. With two, a single flag makes an impossible demand:
the tire monitors have shadow logs from real drives and may one day be cleared,
while the engine monitors have never seen a vehicle at all. Clearing the first
would clear the second, silently, in the same edit.

So clearance is per domain, and a domain registers a GETTER rather than a value —
config is the source of truth and stays editable at runtime, which is what the
selftests rely on when they assert the shipped default.

THE ONE EXCEPTION
-----------------
An urgent fast-path code speaks even in shadow mode, because the whole argument
for the fast path is that the consequence of waiting is not recoverable. That
decision is not made here: it is `CodeCatalog.fast_path_eligible`, gated by
validation rules in the domain's monitors that no single bad reading can pass.
This module only answers "is ordinary announcement cleared for this domain".
"""
from __future__ import annotations

from typing import Callable, Dict

# domain -> () -> bool. A callable rather than a bool so config stays the single
# source of truth and a test that flips a config flag is immediately reflected.
_FLAGS: Dict[str, Callable[[], bool]] = {}


def register(domain: str, getter: Callable[[], bool]) -> None:
    """Declare how this domain's shadow flag is read. Called at import time."""
    _FLAGS[domain] = getter


def is_shadowed(domain: str) -> bool:
    """Is ordinary announcement suppressed for this domain?

    An unregistered domain is shadowed. That default is deliberate and it is the
    safe direction: a domain that forgot to declare itself stays quiet, rather
    than gaining a voice by omission.
    """
    getter = _FLAGS.get(domain)
    if getter is None:
        return True
    try:
        return bool(getter())
    except Exception:
        return True


def registered() -> Dict[str, bool]:
    """Every domain and its current clearance. For the service view."""
    return {d: is_shadowed(d) for d in sorted(_FLAGS)}
