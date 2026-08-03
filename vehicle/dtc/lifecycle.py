"""lifecycle.py — what happens to a diagnostic trouble code over time.

    not_observed
          ↓
    pending_first_seen  ──────────────┐
          ↓                           │
    pending_repeated                  │
          ↓                           │
    confirmed_or_stored               │
          ↓                           │
    permanent_if_applicable           │
          ↓                           ↓
    no_longer_reported  ←─────────────┘
          ↓
    repair_validated

THREE AXES, HELD APART
----------------------
The same discipline diag/runner.py applies to monitor findings, and it matters
here for the same reason:

    ecu_status    what the vehicle said on the LAST scan
                  pending / stored / permanent / absent
    lifecycle     what has happened to this code over its life
                  the states above
    communication what the driver has been told about it

They are genuinely independent. A code can be absent from the latest scan and
still be the most important thing on the dashboard; a confirmed code can be one
the driver was told about an hour ago; a pending code can be brand new and worth
mentioning while its ECU status has not changed at all. Folding any two together
makes at least one of those unanswerable.

WHAT RIO MUST NOT ASSUME (§16.2)
--------------------------------
Emissions monitors commonly use a two-trip confirmation, and it is very tempting
to build that assumption in. It is not universal, and every one of these is
false in general:

    every DTC requires two drives
    every pending code becomes confirmed
    every confirmed code turns the light on immediately
    every pending code is a major failure
    a DTC identifies the failed component

So this module reports the ECU's ACTUAL state and its own observation counts. It
never predicts a promotion, never counts down to one, and never says a code
"will" become confirmed.

EARLY DETECTION IS THE PRODUCT
------------------------------
`early_detection` is set when a code is first seen PENDING while the malfunction
indicator lamp is off. That is the entire advantage this feature has over a code
reader: the ECU has noticed something and the dashboard has not lit up yet. It is
recorded at first sight and never recalculated, because "was this caught before
the light came on" is a fact about that moment and stays true afterwards.

A CODE THAT GOES AWAY IS NOT A CODE THAT WAS FIXED
--------------------------------------------------
`no_longer_reported` means exactly what it says. `repair_validated` is a
different state and is only ever reached with evidence from outside this module —
a recorded repair, a mechanic's confirmation, or sustained absence together with
the related signals returning to normal. Nothing in here can promote the first to
the second on its own, and a system that did would tell drivers their car was
fixed because a monitor had not run.
"""
from __future__ import annotations

import threading
import time
from typing import Dict, List, Optional, Tuple

import config

from diag.store import Store

from ..signals import provenance as P
from . import catalog as C

# --- lifecycle states (§16.5) ----------------------------------------------
NOT_OBSERVED = "not_observed"
PENDING_FIRST_SEEN = "pending_first_seen"
PENDING_REPEATED = "pending_repeated"
CONFIRMED = "confirmed_or_stored"
PERMANENT = "permanent_if_applicable"
NO_LONGER_REPORTED = "no_longer_reported"
REPAIR_VALIDATED = "repair_validated"

# --- what the ECU said on the last scan ------------------------------------
ECU_PENDING = "pending"
ECU_STORED = "stored"
ECU_PERMANENT = "permanent"
ECU_ABSENT = "absent"

_PROVENANCE = {
    ECU_PENDING: P.ECU_PENDING_FAULT,
    ECU_STORED: P.ECU_CONFIRMED_FAULT,
    ECU_PERMANENT: P.ECU_PERMANENT_FAULT,
    ECU_ABSENT: P.ECU_CONFIRMED_FAULT,
}

# --- §17.6 status labels ----------------------------------------------------
LABEL_DETECTED_EARLY = "Detected Early"
LABEL_PENDING = "Pending"
LABEL_REPEATED_PENDING = "Repeated Pending"
LABEL_ACTIVE = "Active"
LABEL_CONFIRMED = "Confirmed"
LABEL_PERMANENT = "Permanent"
LABEL_PREVIOUSLY_OBSERVED = "Previously Observed"
LABEL_NO_LONGER_REPORTED = "No Longer Reported"
LABEL_REPAIR_VALIDATED = "Repair Validated"

# --- §17.3 display groups, in the order the section renders them ------------
GROUP_EARLY = "newly_detected_before_warning_light"
GROUP_ACTIVE = "active_or_confirmed"
GROUP_REPEATED_PENDING = "repeated_pending"
GROUP_PERMANENT = "permanent"
GROUP_PREVIOUS = "previously_observed"

GROUP_ORDER = (GROUP_EARLY, GROUP_ACTIVE, GROUP_REPEATED_PENDING,
               GROUP_PERMANENT, GROUP_PREVIOUS)

GROUP_TITLE = {
    GROUP_EARLY: "Newly Detected Before Warning Light",
    GROUP_ACTIVE: "Active or Confirmed Codes",
    GROUP_REPEATED_PENDING: "Repeated Pending Codes",
    GROUP_PERMANENT: "Permanent Codes",
    GROUP_PREVIOUS: "Previously Observed Codes",
}

# --- §26 event types --------------------------------------------------------
EV_PENDING_FIRST = "vehicle.dtc.pending_first_seen"
EV_PENDING_REPEATED = "vehicle.dtc.pending_repeated"
EV_PROMOTED = "vehicle.dtc.promoted_to_confirmed"
EV_PERMANENT = "vehicle.dtc.permanent_detected"
EV_ABSENT = "vehicle.dtc.no_longer_reported"
EV_MIL_CHANGED = "vehicle.dtc.mil_status_changed"
EV_SNAPSHOT_READY = "vehicle.dtc.early_detection_snapshot_ready"
EV_SCAN_COMPLETED = "vehicle.dtc.scan_completed"
EV_RETURNED = "vehicle.dtc.returned"


def _blank_communication() -> dict:
    """The ledger. Every field is about the DRIVER, none about the car."""
    return {"firstToldAt": None, "lastToldAt": None, "lastStatusTold": None,
            "announcementCount": 0, "shadowProposals": 0,
            "lastShadowProposalAt": None}


class DTCRegistry:
    """Every code this vehicle has ever reported, and what became of it."""

    def __init__(self, store: Store = None, load: bool = True):
        self._lock = threading.RLock()
        self.store = store or Store(config.VEHICLE_DIAG_DIR, "vehicle_dtc",
                                    max_events=config.VEHICLE_DIAG_MAX_EVENTS)
        self._state = self.store.load_state() if load else {
            "version": 1, "issues": {}, "monitors": {}, "epochs": {}, "meta": {}}
        # `issues` is the Store's own key and is reused rather than renamed, so
        # one loader serves both this and the monitor domains.
        self._state.setdefault("issues", {})
        self._state.setdefault("meta", {})
        self._mil: Optional[bool] = None
        self._last_scan_at: Optional[float] = None
        self._scan_count = 0

    # ------------------------------------------------------------------
    # Reading a scan
    # ------------------------------------------------------------------
    def observe_scan(self, scan: dict, now: float = None,
                     snapshotter=None) -> dict:
        """One completed DTC scan. -> what changed, as dashboard events.

        `scan` is what the ECU said, already decoded:

            {"pending": ["P0171"], "stored": [], "permanent": [],
             "mil": False, "dtc_count": 1,
             "drive_session_id": "...", "freeze_frames": {"P0171": {...}},
             "supported": {"pending": True, "stored": True, "permanent": False}}

        `supported` matters and is not decoration: a vehicle that does not
        answer Mode 0A has no permanent codes, which is a different fact from
        having none. Treating an unsupported service as an empty result would
        quietly report "no permanent codes" about a car nobody asked.
        """
        now = time.time() if now is None else float(now)
        events: List[dict] = []

        with self._lock:
            self._scan_count += 1
            self._last_scan_at = now
            session = scan.get("drive_session_id")
            mil = bool(scan.get("mil"))
            supported = dict(scan.get("supported") or {})

            if self._mil is not None and mil != self._mil:
                events.append(self._event(EV_MIL_CHANGED, None, now, session,
                                          {"mil_commanded_on": mil,
                                           "was": self._mil}))
            self._mil = mil

            seen: Dict[str, str] = {}
            for code in (scan.get("permanent") or []):
                seen[code.upper()] = ECU_PERMANENT
            for code in (scan.get("stored") or []):
                # Stored outranks pending when both report the same code: the
                # ECU has confirmed it, and describing it as pending afterwards
                # would understate what the vehicle already decided.
                seen.setdefault(code.upper(), ECU_STORED)
            for code in (scan.get("pending") or []):
                seen.setdefault(code.upper(), ECU_PENDING)

            for code, ecu_status in sorted(seen.items()):
                events.extend(self._apply(code, ecu_status, mil, now, session,
                                          scan, snapshotter))

            # Codes we knew about that this scan did not mention. Only for the
            # services the vehicle actually answered — a code cannot be "no
            # longer reported" by a service that was never asked.
            for code, rec in self._state["issues"].items():
                if code in seen:
                    continue
                if rec["lifecycle"] in (NO_LONGER_REPORTED, REPAIR_VALIDATED,
                                        NOT_OBSERVED):
                    continue
                if not self._service_answered(rec, supported):
                    continue
                rec["status_history"].append(
                    {"at": now, "from": rec.get("ecu_status"), "to": ECU_ABSENT,
                     "mil_commanded_on": mil, "drive_session_id": session})
                rec["ecu_status"] = ECU_ABSENT
                rec["lifecycle"] = NO_LONGER_REPORTED
                rec["absent_since"] = now
                rec["last_absent_at"] = now
                rec["last_absent_scan"] = self._scan_count
                events.append(self._event(EV_ABSENT, code, now, session, {
                    "was": rec.get("last_present_status")}))
                self.store.append_event("dtc_no_longer_reported", {
                    "code": code, "was": rec.get("last_present_status"),
                    "drive_session_id": session}, at=now)

            self._state["meta"]["mil"] = mil
            self._state["meta"]["last_scan_at"] = now
            self._state["meta"]["scan_count"] = self._scan_count
            self._state["meta"]["dtc_count_reported"] = scan.get("dtc_count")
            self._state["meta"]["supported"] = supported
            self.store.save_state(self._state)

        events.append(self._event(EV_SCAN_COMPLETED, None, now,
                                  scan.get("drive_session_id"),
                                  {"codes": sorted(seen),
                                   "mil_commanded_on": mil,
                                   "supported": supported}))
        return {"at": now, "events": events, "mil": mil,
                "codes_reported": sorted(seen), "scan_count": self._scan_count}

    @staticmethod
    def _service_answered(rec: dict, supported: dict) -> bool:
        """Did this scan actually ask the service that would have found it?"""
        was = rec.get("last_present_status")
        if was == ECU_PERMANENT:
            return bool(supported.get("permanent", True))
        if was == ECU_STORED:
            return bool(supported.get("stored", True))
        if was == ECU_PENDING:
            return bool(supported.get("pending", True))
        return True

    def _apply(self, code: str, ecu_status: str, mil: bool, now: float,
               session: Optional[str], scan: dict, snapshotter) -> List[dict]:
        events: List[dict] = []
        rec = self._state["issues"].get(code)
        freeze = (scan.get("freeze_frames") or {}).get(code)

        if rec is None:
            rec = self._new(code, ecu_status, mil, now, session)
            self._state["issues"][code] = rec
            first_time = True
        else:
            first_time = False

        was_lifecycle = rec["lifecycle"]
        was_status = rec.get("ecu_status")
        returning = was_lifecycle in (NO_LONGER_REPORTED, REPAIR_VALIDATED)

        rec["ecu_status"] = ecu_status
        rec["last_present_status"] = ecu_status
        rec["last_seen_at"] = now
        rec["last_seen_session"] = session
        rec["mil_commanded_on"] = mil
        rec["absent_since"] = None
        if session and session not in rec["sessions"]:
            rec["sessions"].append(session)
        rec["drive_cycle_count_observed"] = len(rec["sessions"])

        if freeze and not rec.get("freeze_frame"):
            # ECU-recorded conditions at the moment the fault was set. Kept
            # visibly apart from RIO's own snapshot — see §17.9. One is what the
            # vehicle chose to record; the other is what RIO happened to be
            # watching, and presenting them alike would overstate both.
            rec["freeze_frame"] = dict(freeze)
            rec["freeze_frame_available"] = True

        if returning:
            rec["recurrence_count"] = rec.get("recurrence_count", 0) + 1
            rec["previous_absences"].append(rec.get("last_absent_at") or now)
            events.append(self._event(EV_RETURNED, code, now, session, {
                "recurrence_count": rec["recurrence_count"]}))
            self.store.append_event("dtc_returned", {
                "code": code, "recurrence_count": rec["recurrence_count"],
                "drive_session_id": session}, at=now)

        if ecu_status == ECU_PENDING:
            rec["pending_scan_count"] = rec.get("pending_scan_count", 0) + 1
            if first_time or returning or was_lifecycle == NOT_OBSERVED:
                rec["lifecycle"] = PENDING_FIRST_SEEN
                # The whole product advantage, recorded once and never
                # recalculated: "was this caught before the light came on" is a
                # fact about this moment and stays true afterwards.
                rec["early_detection"] = not mil
                rec["mil_at_first_detection"] = mil
                events.append(self._event(EV_PENDING_FIRST, code, now, session, {
                    "early_detection": rec["early_detection"],
                    "mil_commanded_on": mil}))
                self.store.append_event("dtc_pending_first_seen", {
                    "code": code, "early_detection": rec["early_detection"],
                    "mil_commanded_on": mil, "drive_session_id": session}, at=now)
                if snapshotter is not None:
                    snap = snapshotter(code, now)
                    if snap:
                        rec["snapshot_id"] = snap.get("snapshot_id")
                        events.append(self._event(
                            EV_SNAPSHOT_READY, code, now, session,
                            {"snapshot_id": snap.get("snapshot_id"),
                             "signals": snap.get("signals", [])}))
            elif was_lifecycle in (PENDING_FIRST_SEEN, PENDING_REPEATED):
                rec["lifecycle"] = PENDING_REPEATED
                if was_lifecycle == PENDING_FIRST_SEEN:
                    events.append(self._event(EV_PENDING_REPEATED, code, now,
                                              session, {
                        "pending_scan_count": rec["pending_scan_count"]}))
                    self.store.append_event("dtc_pending_repeated", {
                        "code": code,
                        "pending_scan_count": rec["pending_scan_count"],
                        "drive_session_id": session}, at=now)

        elif ecu_status == ECU_STORED:
            if was_lifecycle != CONFIRMED or returning:
                # The pending first-seen time is PRESERVED. It is the evidence
                # that RIO saw this before the vehicle confirmed it, and
                # overwriting it on promotion would destroy the only record of
                # the thing this feature exists to do.
                rec["lifecycle"] = CONFIRMED
                rec["confirmed_at"] = now
                events.append(self._event(EV_PROMOTED, code, now, session, {
                    "first_seen_at": rec["first_seen_at"],
                    "was_detected_early": rec.get("early_detection", False),
                    "mil_commanded_on": mil}))
                self.store.append_event("dtc_promoted_to_confirmed", {
                    "code": code, "first_seen_at": rec["first_seen_at"],
                    "was_detected_early": rec.get("early_detection", False),
                    "drive_session_id": session}, at=now)

        elif ecu_status == ECU_PERMANENT:
            if was_lifecycle != PERMANENT:
                rec["lifecycle"] = PERMANENT
                rec["permanent_at"] = now
                events.append(self._event(EV_PERMANENT, code, now, session, {}))
                self.store.append_event("dtc_permanent_detected", {
                    "code": code, "drive_session_id": session}, at=now)

        if was_status != ecu_status:
            rec["status_history"].append(
                {"at": now, "from": was_status, "to": ecu_status,
                 "mil_commanded_on": mil, "drive_session_id": session})
        return events

    def _new(self, code: str, ecu_status: str, mil: bool, now: float,
             session: Optional[str]) -> dict:
        definition = C.get(code)
        return {
            "dtc_event_id": f"dtc_{code}_{int(now)}",
            "code": code,
            "known": C.is_known(code),
            "manufacturer_specific": C.is_manufacturer_specific(code),
            "system": definition.system,
            "description": definition.description,
            "severity": definition.severity,
            "health_severity": C.health_severity(definition.severity),
            # None, not the incoming status. The record starts as "we had never
            # heard of this", so the FIRST sighting is a real transition and
            # lands in the status history — which §17.10's timeline opens with.
            # Pre-filling it here made the first thing that ever happened to a
            # code the one thing its history did not record.
            "ecu_status": None,
            "last_present_status": None,
            "lifecycle": NOT_OBSERVED,
            "provenance": _PROVENANCE.get(ecu_status, P.ECU_CONFIRMED_FAULT),
            "first_seen_at": now,
            "first_seen_session": session,
            "last_seen_at": now,
            "last_seen_session": session,
            "confirmed_at": None,
            "permanent_at": None,
            "absent_since": None,
            "last_absent_at": None,
            "resolved_at": None,
            "mil_commanded_on": mil,
            "mil_at_first_detection": mil,
            "early_detection": False,
            "pending_scan_count": 0,
            "sessions": [],
            "drive_cycle_count_observed": 0,
            "recurrence_count": 0,
            "previous_absences": [],
            "freeze_frame": None,
            "freeze_frame_available": False,
            "snapshot_id": None,
            "status_history": [],
            # Filled in only by a person or by validated evidence. Nothing in
            # this module writes it.
            "confirmed_cause": None,
            "repair": None,
            "communication": _blank_communication(),
        }

    def _event(self, kind: str, code: Optional[str], now: float,
               session: Optional[str], payload: dict) -> dict:
        body = {"code": code} if code else {}
        body.update(payload)
        rec = self._state["issues"].get(code) if code else None
        if rec is not None:
            body.setdefault("provenance", rec.get("provenance"))
            body.setdefault("status", rec.get("ecu_status"))
        return {"type": kind, "vehicle_id": config.VEHICLE_ID,
                "drive_session_id": session, "at": now, "payload": body}

    # ------------------------------------------------------------------
    # Repair and validation — the only way out of no_longer_reported
    # ------------------------------------------------------------------
    def record_repair(self, code: str, description: str, by: str = "driver",
                      now: float = None) -> bool:
        """A repair was performed. NOT a claim that it worked.

        Recording work done and validating that it fixed something are separate
        events on purpose. A part replaced is evidence; a code that stays away
        afterwards is the confirmation, and only the pair justifies telling a
        driver their car is fixed.
        """
        now = time.time() if now is None else now
        with self._lock:
            rec = self._state["issues"].get((code or "").upper())
            if rec is None:
                return False
            rec["repair"] = {"description": description, "by": by, "at": now}
            self.store.append_event("dtc_repair_recorded", {
                "code": rec["code"], "description": description, "by": by},
                at=now)
            self.store.save_state(self._state)
            return True

    def validate_repair(self, code: str, evidence: str, by: str = "mechanic",
                        now: float = None) -> bool:
        """Only with evidence from outside this module. See the module header."""
        now = time.time() if now is None else now
        with self._lock:
            rec = self._state["issues"].get((code or "").upper())
            if rec is None or rec["lifecycle"] not in (NO_LONGER_REPORTED,
                                                       REPAIR_VALIDATED):
                return False
            rec["lifecycle"] = REPAIR_VALIDATED
            rec["resolved_at"] = now
            rec["confirmed_cause"] = evidence
            rec["provenance"] = P.REPAIR_VALIDATED
            self.store.append_event("dtc_repair_validated", {
                "code": rec["code"], "evidence": evidence, "by": by}, at=now)
            self.store.save_state(self._state)
            return True

    def note_announced(self, code: str, status: str, now: float = None) -> bool:
        now = time.time() if now is None else now
        with self._lock:
            rec = self._state["issues"].get((code or "").upper())
            if rec is None:
                return False
            comm = rec["communication"]
            if comm["firstToldAt"] is None:
                comm["firstToldAt"] = now
            comm["lastToldAt"] = now
            comm["lastStatusTold"] = status
            comm["announcementCount"] += 1
            self.store.append_event("dtc_announced", {
                "code": rec["code"], "status": status}, at=now)
            self.store.save_state(self._state)
            return True

    def note_shadow_proposal(self, code: str, text: str, reason: str,
                             now: float = None) -> bool:
        now = time.time() if now is None else now
        with self._lock:
            rec = self._state["issues"].get((code or "").upper())
            if rec is None:
                return False
            comm = rec["communication"]
            comm["shadowProposals"] += 1
            comm["lastShadowProposalAt"] = now
            self.store.append_event("dtc_shadow_proposal", {
                "code": rec["code"], "would_have_said": text,
                "policy_reason": reason}, at=now)
            self.store.save_state(self._state)
            return True

    # ------------------------------------------------------------------
    # Views
    # ------------------------------------------------------------------
    def records(self) -> List[dict]:
        with self._lock:
            return [dict(r) for r in self._state["issues"].values()]

    def get(self, code: str) -> Optional[dict]:
        with self._lock:
            rec = self._state["issues"].get((code or "").upper())
            return dict(rec) if rec else None

    def mil(self) -> Optional[bool]:
        return self._mil

    def meta(self) -> dict:
        with self._lock:
            return dict(self._state.get("meta") or {})

    def active_codes(self) -> List[dict]:
        """Codes the vehicle is currently reporting, worst first."""
        rows = [r for r in self.records()
                if r["lifecycle"] not in (NO_LONGER_REPORTED, REPAIR_VALIDATED,
                                          NOT_OBSERVED)]
        rows.sort(key=lambda r: (-C.SEVERITY_RANK.get(r["severity"], 0),
                                 r["code"]))
        return rows

    def reset_for_test(self, directory: str) -> None:
        self.store.reset_for_test(directory)
        with self._lock:
            self._state = {"version": 1, "issues": {}, "monitors": {},
                           "epochs": {}, "meta": {}}
            self._mil = None
            self._scan_count = 0
            self._last_scan_at = None


# ---------------------------------------------------------------------------
# Display grouping (§17.3)
# ---------------------------------------------------------------------------

def display_status(rec: dict) -> str:
    """§17.6's label for one record.

    Early detection wins over every other label while the code is still pending,
    because it is the thing that distinguishes RIO from a code reader and it is
    the first thing the section is meant to show.
    """
    life = rec.get("lifecycle")
    if life == REPAIR_VALIDATED:
        return LABEL_REPAIR_VALIDATED
    if life == NO_LONGER_REPORTED:
        return LABEL_NO_LONGER_REPORTED
    if life == PERMANENT:
        return LABEL_PERMANENT
    if life == CONFIRMED:
        return LABEL_CONFIRMED if not rec.get("mil_commanded_on") else LABEL_ACTIVE
    if life == PENDING_REPEATED:
        return LABEL_REPEATED_PENDING
    if life == PENDING_FIRST_SEEN:
        return LABEL_DETECTED_EARLY if rec.get("early_detection") else LABEL_PENDING
    return LABEL_PREVIOUSLY_OBSERVED


def group_of(rec: dict) -> str:
    life = rec.get("lifecycle")
    if life in (NO_LONGER_REPORTED, REPAIR_VALIDATED, NOT_OBSERVED):
        return GROUP_PREVIOUS
    if life == PERMANENT:
        return GROUP_PERMANENT
    if life == CONFIRMED:
        return GROUP_ACTIVE
    if life == PENDING_REPEATED:
        return GROUP_REPEATED_PENDING
    if life == PENDING_FIRST_SEEN:
        return GROUP_EARLY if rec.get("early_detection") else GROUP_ACTIVE
    return GROUP_PREVIOUS
