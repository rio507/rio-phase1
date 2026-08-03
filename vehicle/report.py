"""report.py — "RIO, run a diagnostic report."

The difference between this and the quick health check is not length. The quick
check answers from what RIO already knows; this one GOES AND ASKS THE VEHICLE.
That distinction is the whole reason it is a job with progress states rather
than a function that returns a dict: asking a car takes seconds, the driver is
watching a button, and "Checking Pending Codes" is a more honest thing to show
them than a spinner.

    Requesting Vehicle Data
    Checking Pending Codes
    Checking Stored Codes
    Checking Permanent Codes
    Retrieving Freeze-Frame Data
    Reviewing Live Signals
    Comparing Recent History
    Generating Summary
    Complete

WHAT THE REPORT IS ORGANISED AROUND
-----------------------------------
Not severity, and not subsystem. Provenance. §28's sections split on WHO IS
MAKING THE CLAIM — what the vehicle reported, then what RIO observed — because
that is the split a driver and a mechanic both need and the one a merged list
destroys. A report that sorted by severity would put a RIO baseline deviation
above a confirmed ECU fault, and a reader would have no way to tell which of
them the car itself had said.

WHAT IT MUST NOT DO
-------------------
Diagnose. Every possible cause in here is a list item, every RIO finding says it
is RIO's, and nothing in this file promotes either to a fact. The plain-language
summary is assembled deterministically from the same fields — it is not model
output and cannot be, because a summary generated from the report could restate
its hedges as conclusions and nothing downstream would know.

The conversation layer may narrate this. It may not add to it.
"""
from __future__ import annotations

import threading
import time
from typing import Callable, Dict, List, Optional

import config

from .dtc import catalog as C
from .dtc import lifecycle as L
from .signals import provenance as P
from .signals import quality as Q
from .signals import registry as R

# §27.8's progress states, in order.
STAGES = (
    "Requesting Vehicle Data",
    "Checking Pending Codes",
    "Checking Stored Codes",
    "Checking Permanent Codes",
    "Retrieving Freeze-Frame Data",
    "Reviewing Live Signals",
    "Comparing Recent History",
    "Generating Summary",
    "Complete",
)

PENDING = "pending"
RUNNING = "running"
COMPLETE = "complete"
FAILED = "failed"


class Report:
    """One diagnostic report, from request to summary."""

    def __init__(self, report_id: str, vehicle_id: str,
                 drive_session_id: str = None):
        self._lock = threading.RLock()
        self.report_id = report_id
        self.vehicle_id = vehicle_id
        self.drive_session_id = drive_session_id
        self.requested_at = time.time()
        self.completed_at: Optional[float] = None
        self.status = PENDING
        self.stage = STAGES[0]
        self.stage_index = 0
        self.error: Optional[str] = None
        self.body: Dict = {}

    def _advance(self, stage: str) -> None:
        with self._lock:
            self.stage = stage
            self.stage_index = STAGES.index(stage)
            self.status = RUNNING

    def progress(self) -> dict:
        with self._lock:
            return {
                "report_id": self.report_id,
                "vehicle_id": self.vehicle_id,
                "drive_session_id": self.drive_session_id,
                "status": self.status,
                "stage": self.stage,
                "stage_index": self.stage_index,
                "stages": list(STAGES),
                "requested_at": self.requested_at,
                "completed_at": self.completed_at,
                "error": self.error,
            }

    def view(self) -> dict:
        out = self.progress()
        with self._lock:
            out["report"] = dict(self.body) if self.body else None
        return out

    # ------------------------------------------------------------------
    def run(self, *, dtc_service, telemetry_snapshot: Callable,
            vehicle_state: Callable, health_issues: Callable,
            insight_feed: Callable, gateways: Callable,
            capability: Callable, now: float = None) -> dict:
        """Do the work. Everything it needs is passed in.

        Injected rather than imported, so this module cannot reach into the
        conversation layer, the announcement policy or a model — and so the
        whole report is testable without a server.
        """
        now = time.time() if now is None else now
        try:
            self._advance("Requesting Vehicle Data")
            snap = telemetry_snapshot() or {}
            state = vehicle_state(snap) or {}
            gws = gateways() or []

            # One scan covers Modes 03, 07 and 0A, exactly as a bridge's scan
            # does. The stages are still walked one at a time, because what the
            # driver is watching is a list of questions being asked, and
            # collapsing them into "scanning..." would be less honest rather
            # than more efficient.
            self._advance("Checking Pending Codes")
            scan = dtc_service.scan(now, self.drive_session_id,
                                    reason="driver_requested")
            self._advance("Checking Stored Codes")
            self._advance("Checking Permanent Codes")
            self._advance("Retrieving Freeze-Frame Data")

            self._advance("Reviewing Live Signals")
            signals = _signal_section(snap, capability())

            self._advance("Comparing Recent History")
            observations = _observation_section(health_issues(), insight_feed())

            self._advance("Generating Summary")
            section = dtc_service.section()
            body = {
                "report_id": self.report_id,
                "generated_at": now,
                "vehicle": _vehicle_section(self.vehicle_id, snap, state, gws,
                                            scan, self.drive_session_id),
                "early_detection": _early_section(section),
                "confirmed_faults": _confirmed_section(section),
                "operating_data": signals,
                "rio_observations": observations,
            }
            body["summary"] = _summary(body)
            with self._lock:
                self.body = body
                self.status = COMPLETE
                self.stage = "Complete"
                self.stage_index = STAGES.index("Complete")
                self.completed_at = time.time()
        except Exception as e:
            with self._lock:
                self.status = FAILED
                self.error = f"{type(e).__name__}: {e}"
                self.completed_at = time.time()
            print(f"[report] {self.report_id} failed: {self.error}", flush=True)
        return self.view()


# ---------------------------------------------------------------------------
# The sections (§28)
# ---------------------------------------------------------------------------

def _vehicle_section(vehicle_id: str, snap: dict, state: dict,
                     gateways: List[dict], scan: dict,
                     session_id: Optional[str]) -> dict:
    decoders = sorted({r.get("decoder_version") for r in []} - {None})
    return {
        "vehicle_id": vehicle_id,
        "data_source": snap.get("source"),
        "data_source_label": snap.get("source_label"),
        "gateways": gateways,
        "drive_session_id": session_id,
        "vehicle_state": state.get("state"),
        "vehicle_state_label": state.get("label"),
        "vehicle_state_why": state.get("why"),
        "connection": snap.get("connection"),
        "last_data_at": snap.get("updated_at"),
        "engine_running": snap.get("engine_running"),
        "ecu_responding": scan.get("ecu_responding", True),
        "services_supported": (scan.get("services") or None),
        "decoder_versions": decoders,
    }


def _early_section(section: dict) -> dict:
    """§28.2 — the part no code reader can produce.

    Pending faults RIO saw before the lamp came on, with the counts that say
    how seriously to take each one. This section is first in the report for the
    same reason its group is first in the dashboard: it is the only part of the
    document that could not have been obtained by plugging in a scan tool.
    """
    cards = [c for g in section.get("groups", [])
             if g["key"] == L.GROUP_EARLY for c in g["cards"]]
    repeated = [c for g in section.get("groups", [])
                if g["key"] == L.GROUP_REPEATED_PENDING for c in g["cards"]]
    return {
        "detected_before_warning_light": cards,
        "repeated_pending": repeated,
        "count": len(cards) + len(repeated),
        "mil_commanded_on": section.get("mil_commanded_on"),
        "note": ("A pending code means the vehicle's computer has observed a "
                 "condition and has not confirmed that it is persistent. It is "
                 "not a diagnosis and it does not name a failed part."),
    }


def _confirmed_section(section: dict) -> dict:
    """§28.3 — what the vehicle has actually confirmed."""
    def group(key):
        return [c for g in section.get("groups", []) if g["key"] == key
                for c in g["cards"]]

    active = group(L.GROUP_ACTIVE)
    permanent = group(L.GROUP_PERMANENT)
    previous = group(L.GROUP_PREVIOUS)
    with_frames = [c for c in active + permanent if c.get("freeze_frame_available")]
    return {
        "active_or_confirmed": active,
        "permanent": permanent,
        "previously_observed": previous,
        "count": len(active) + len(permanent),
        "freeze_frames_available": len(with_frames),
        "services_supported": section.get("services_supported") or {},
        "note": ("Permanent codes cannot be cleared by a scan tool and clear "
                 "themselves only after the vehicle's own monitors pass. RIO "
                 "offers no way to clear any code."),
    }


def _signal_section(snap: dict, capability: dict) -> dict:
    """§28.4 — what is reading now, what is not, and what never was."""
    current, missing, stale, invalid = [], [], [], []
    for row in (snap.get("rows") or []):
        if row["id"].startswith("tire_"):
            continue
        entry = {"signal": R.canonical(row["id"]) or row["id"],
                 "telemetry_id": row["id"], "label": row["label"],
                 "value": row["value"], "value_text": row["value_text"],
                 "unit": row["units"], "status": row["status"],
                 "trend": row["trend"]}
        status = row["status"]
        if status == "NO DATA":
            missing.append(entry)
        elif status in ("STALE", "OFFLINE"):
            stale.append(entry)
        elif status in ("WARNING", "CRITICAL"):
            invalid.append(entry)
            current.append(entry)
        else:
            current.append(entry)
    return {
        "current": current,
        "missing": missing,
        "stale": stale,
        "out_of_band": invalid,
        # A signal this vehicle has never produced is NOT missing. Most
        # vehicles do not expose most PIDs, and a row that reads "no data"
        # forever looks exactly like a sensor that has died.
        "never_supported": capability.get("unsupported", []),
        "supported": capability.get("supported", []),
    }


def _observation_section(issues: List[dict], insights: List[dict]) -> dict:
    """§28.5 — RIO's own findings, kept visibly apart from the vehicle's.

    Every entry here is an inference. The section is separate, the provenance is
    on each item, and the wording says "RIO observed" — because the one thing
    this report cannot afford is a reader who cannot tell which half the car
    itself said.
    """
    rio = []
    gaps = []
    for issue in issues:
        if issue.get("domain") == "diagnostics":
            # Codes are the vehicle's, and they have their own sections above.
            continue
        itype = issue.get("type") or ""
        if itype.endswith("_unavailable") or "not_ready" in itype \
                or "not_evaluated" in itype:
            # "Nothing is reporting from the tire sensors" is not an observation
            # ABOUT the car — it is a statement about what RIO was able to look
            # at. Listing it among the findings makes a report of an unmonitored
            # vehicle read as a report of a troubled one, and buries the actual
            # findings underneath a list of things nobody checked.
            #
            # It absolutely still appears. §28 has no section for "things we
            # quietly left out", and a report that dropped its own blind spots
            # would be the most misleading document this system could produce.
            gaps.append({"domain": issue.get("domain"),
                         "message": issue.get("message"),
                         "type": itype})
            continue
        rio.append({
            "type": issue.get("type"),
            "domain": issue.get("domain"),
            "severity": issue.get("severity"),
            "message": issue.get("message"),
            "observation_window": issue.get("observation_window"),
            "suggested_action": issue.get("suggested_action") or None,
            "provenance": P.RIO_OBSERVED_PATTERN,
            "provenance_label": P.display(P.RIO_OBSERVED_PATTERN),
            "evidence": issue.get("evidence") or {},
        })
    return {
        "findings": rio,
        "count": len(rio),
        # What RIO could not look at. Named, so the report says what it does not
        # cover instead of leaving the reader to assume it covered everything.
        "coverage_gaps": gaps,
        "gap_count": len(gaps),
        "recent_history": insights[:12],
        # Seeded demo history is flagged all the way out to here. A report that
        # presented fabricated history as measured is the one failure this
        # layer cannot have — see insights.py's header.
        "history_includes_seeded": any(e.get("seeded") for e in insights[:12]),
        "note": ("These are RIO's own observations, not faults reported by the "
                 "vehicle. They are shown separately for that reason."),
    }


# ---------------------------------------------------------------------------
# The plain-language summary (§28.6)
# ---------------------------------------------------------------------------

def _summary(body: dict) -> str:
    """Assembled from the report's own fields. Deterministic, and hedged.

    Not model output, and it must not become model output. A summary generated
    FROM this document could restate its hedges as conclusions — "the MAF is
    faulty" out of a list of five possible causes — and nothing downstream would
    be able to tell that it had. The conversation layer may narrate this
    sentence; it may not replace it.
    """
    early = body["early_detection"]
    confirmed = body["confirmed_faults"]
    obs = body["rio_observations"]
    vehicle = body["vehicle"]
    parts: List[str] = []

    if not vehicle.get("ecu_responding", True):
        return ("The vehicle's computer did not answer RIO's diagnostic "
                "requests, so no code information is available for this "
                "report. That is a fault in the connection, not a statement "
                "about the engine.")

    n_conf = confirmed["count"]
    if n_conf:
        codes = ", ".join(c["code"] for c in
                          confirmed["active_or_confirmed"] + confirmed["permanent"])
        parts.append(f"The vehicle is reporting {n_conf} confirmed fault"
                     f"{'' if n_conf == 1 else 's'} ({codes}).")
    else:
        parts.append("The vehicle is not currently reporting a confirmed "
                     "engine fault.")

    n_early = early["count"]
    if n_early:
        early_codes = ", ".join(
            c["code"] for c in early["detected_before_warning_light"]
            + early["repeated_pending"])
        lamp = "before the check-engine light came on" \
            if not early.get("mil_commanded_on") else "as pending"
        parts.append(f"RIO detected {n_early} pending code"
                     f"{'' if n_early == 1 else 's'} {lamp} ({early_codes}). "
                     f"A pending code is a condition the computer has observed "
                     f"and not confirmed.")

    if obs["count"]:
        worst = obs["findings"][0]
        more = obs["count"] - 1
        tail = f" There {'is' if more == 1 else 'are'} {more} further " \
               f"observation{'' if more == 1 else 's'}." if more > 0 else ""
        parts.append(f"Separately, RIO observed: {worst['message']}{tail}")

    missing = body["operating_data"]["missing"] + body["operating_data"]["stale"]
    if missing:
        names = ", ".join(m["label"] for m in missing[:3])
        parts.append(f"{len(missing)} signal"
                     f"{'' if len(missing) == 1 else 's'} were not reporting "
                     f"during this report ({names}), so nothing could be judged "
                     f"from them.")

    if obs["gap_count"]:
        # Said out loud, in the summary, not buried in a field. A report whose
        # headline is "nothing is wrong" while three subsystems were never
        # looked at is the most confidently misleading thing this system could
        # produce, and this sentence is what stops it.
        domains = ", ".join(sorted({g["domain"] for g in obs["coverage_gaps"]}))
        parts.append(f"RIO was not able to evaluate {domains} during this "
                     f"report, so nothing here should be read as an all-clear "
                     f"for {'that' if obs['gap_count'] == 1 else 'those'}.")

    if n_conf or n_early or obs["count"]:
        parts.append("No physical cause has been confirmed. RIO will keep "
                     "monitoring and report any change.")

    return " ".join(parts)


# ---------------------------------------------------------------------------
# The registry of reports
# ---------------------------------------------------------------------------

class ReportStore:
    """Reports, newest last, bounded. Kept as vehicle history (§29.1)."""

    def __init__(self, limit: int = 50):
        self._lock = threading.RLock()
        self._reports: Dict[str, Report] = {}
        self._order: List[str] = []
        self._limit = limit
        self._seq = 0

    def create(self, vehicle_id: str, drive_session_id: str = None) -> Report:
        with self._lock:
            self._seq += 1
            rid = f"report_{int(time.time())}_{self._seq:03d}"
            report = Report(rid, vehicle_id, drive_session_id)
            self._reports[rid] = report
            self._order.append(rid)
            while len(self._order) > self._limit:
                self._reports.pop(self._order.pop(0), None)
            return report

    def get(self, report_id: str) -> Optional[Report]:
        with self._lock:
            return self._reports.get(report_id)

    def latest(self) -> Optional[Report]:
        with self._lock:
            return self._reports.get(self._order[-1]) if self._order else None

    def list(self) -> List[dict]:
        with self._lock:
            return [self._reports[r].progress() for r in reversed(self._order)]

    def clear(self) -> None:
        with self._lock:
            self._reports.clear()
            self._order.clear()
            self._seq = 0


_store = ReportStore()


def store() -> ReportStore:
    return _store
