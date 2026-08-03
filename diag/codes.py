"""codes.py — a diagnostic identifier, and what is true of it everywhere.

Lifted from tire_diag/codes.py with `corner` widened to `subject`. Everything
that file said about why codes exist still applies and is not repeated here; the
one paragraph worth restating is the one that keeps being got wrong:

    A code is not something the driver ever sees. RIO says "your rear-left tire
    may have a slow leak", never "RIO-TIRE-POSSIBLE-LEAK-RL is active".
    `driver_term` is the only field in here that is ever allowed near a sentence
    spoken to a person.

WHAT A SUBJECT IS
-----------------
The thing a code is about, within its domain: a corner for tires, a channel or a
system for the engine, None for a finding about the whole subsystem. It is a
plain string because it ends up in an issue id, in a log a person reads, and in
a code name — and a structured identifier would be none of those things.

TWO PERMISSIONS, DELIBERATELY SEPARATE
--------------------------------------
`speak` is ordinary announcement eligibility. `fast_path` is the urgent exception
that bypasses ordinary confirmation. Neither implies the other, and the second is
narrower than the first: a fast-path code speaks even in shadow mode, because the
whole argument for the fast path is that the consequence of waiting is not
recoverable — but only after passing validation gates that no single bad reading
can satisfy.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

# Severity, in the vocabulary the conversation layer already speaks. `advisory`
# sits between informational and warning: the level for "this is real, it is on
# the dashboard, and it belongs in a drive-start briefing rather than in the
# middle of one".
INFORMATIONAL = "informational"
ADVISORY = "advisory"
WARNING = "warning"
CRITICAL = "critical"

SEVERITY_RANK = {INFORMATIONAL: 1, ADVISORY: 2, WARNING: 3, CRITICAL: 4}


@dataclass(frozen=True)
class DiagnosticCode:
    """One diagnostic condition, everything that is true of it everywhere."""

    code: str
    monitor_id: str
    component_type: str
    subject: Optional[str]              # None for domain-level codes
    default_severity: str

    # What a person is told. `driver_term` is a noun phrase, not a sentence:
    # the sentence is assembled where the subject is known.
    driver_term: str
    technician_description: str
    suggested_action: str

    # Confirmation and healing live on the MONITOR, which owns the evidence.
    # These are the code's summary of them, for the service view and for tests
    # that want to assert the contract without reading the monitor.
    confirmation_summary: str
    healing_summary: str

    freeze_frame_fields: Tuple[str, ...]

    speak: bool = False
    fast_path: bool = False

    # Kept for the domains whose subject is a physical location and whose
    # conversation layer wants to say it out loud. Empty when the subject is not
    # a place — "battery voltage" has no spoken location and must not invent one.
    spoken_subject: str = ""

    def spoken_location(self) -> str:
        return self.spoken_subject


class CodeCatalog:
    """Every code a domain can raise, and the lookups the engine needs.

    A class rather than a module of globals because there are two domains now.
    Module-level CODES worked exactly as long as there was one of them, and the
    failure mode of keeping it would have been silent: the second domain's codes
    landing in the first domain's dictionary, and `code_for` returning a tire
    code to an engine monitor whose ids happened to sort earlier.
    """

    def __init__(self, codes: Dict[str, DiagnosticCode]):
        self._codes = dict(codes)
        # (monitor_id, subject) -> [code]. The engine works in monitors and
        # subjects; the code is what it writes down.
        self._by_monitor: Dict[Tuple[str, Optional[str]], List[str]] = {}
        for c in self._codes.values():
            self._by_monitor.setdefault((c.monitor_id, c.subject), []).append(c.code)

    # -- lookups ------------------------------------------------------------

    @property
    def codes(self) -> Dict[str, DiagnosticCode]:
        return self._codes

    def get(self, code: str) -> Optional[DiagnosticCode]:
        return self._codes.get(code)

    def code_for(self, monitor_id: str, subject: Optional[str] = None,
                 variant: str = None) -> Optional[DiagnosticCode]:
        """The code a monitor raises for a subject.

        `variant` picks between codes that share a monitor — a connectivity
        monitor raises both "stopped talking" and "about to stop talking",
        because those are one monitor's two findings and two quite different
        things to tell somebody.
        """
        candidates = self._by_monitor.get((monitor_id, subject)) or []
        if not candidates:
            return None
        if variant:
            for c in candidates:
                if variant.upper() in c:
                    return self._codes[c]
        return self._codes[sorted(candidates)[0]]

    # -- the two permissions -------------------------------------------------

    def speech_eligible(self, code: str) -> bool:
        """May RIO announce this at all? False for everything, in shadow mode."""
        c = self._codes.get(code)
        return bool(c and c.speak)

    def fast_path_eligible(self, code: str) -> bool:
        """May this take the urgent path, bypassing ordinary confirmation?

        Separate from speech_eligible and deliberately not implied by it. See
        the module header.
        """
        c = self._codes.get(code)
        return bool(c and c.fast_path)

    # -- the service view ----------------------------------------------------

    def service_view(self) -> List[dict]:
        """Every code and what it means. For a diagnostic or service view only —
        never for the conversation layer, which gets driver_term and nothing
        else."""
        return [{
            "code": c.code,
            "monitor": c.monitor_id,
            "component": c.component_type,
            "subject": c.subject,
            "default_severity": c.default_severity,
            "driver_term": c.driver_term,
            "technician_description": c.technician_description,
            "suggested_action": c.suggested_action,
            "confirmation": c.confirmation_summary,
            "healing": c.healing_summary,
            "freeze_frame_fields": list(c.freeze_frame_fields),
            "speech_eligible": c.speak,
            "fast_path_eligible": c.fast_path,
        } for c in sorted(self._codes.values(), key=lambda x: x.code)]
