"""runner.py — the stateful half. Samples in, monitor results and Issues out.

Lifted from tire_diag/engine.py with every reference to a corner, a pressure or
a receiver removed. This is where counting monitor runs, promoting a CANDIDATE to
ACTIVE, healing an ACTIVE back to RESOLVED, freezing evidence at the moment of
confirmation, tracking recurrence and keeping the communication ledger all live.
The monitors stay pure; everything that has to remember something is here.

THREE INDEPENDENT DIMENSIONS
----------------------------
The single most important structural decision in this file, and the one most
easily got wrong:

    lifecycle       CANDIDATE -> ACTIVE -> RESOLVED
                    what the DIAGNOSTIC believes about the car

    severity        informational / advisory / warning / critical
                    how bad it is

    communication   firstToldAt, lastToldAt, lastSeverityTold, lastBriefedAt,
                    acknowledgedAt, announcementCount, monitoringActive,
                    resolutionMentionedAt
                    what the DRIVER has been told

ANNOUNCED is not a lifecycle state. Neither is CRITICAL, WORSENED or
DATA_UNAVAILABLE. Folding any of them in is how a system ends up unable to answer
"is this still wrong?" separately from "have I mentioned it?" — and those
questions have genuinely different answers all the time. A finding can be worse
and already announced; a resolved finding can still be worth mentioning once.

This is OBD's own separation: storing a diagnostic condition and commanding the
malfunction lamp are different decisions, made by different logic, on different
evidence.

WHAT A DOMAIN HAS TO SUPPLY
---------------------------
Class attributes describing itself, and five methods. Nothing else — if a domain
finds itself overriding a lifecycle method, that is the signal that something
generic has been left domain-specific by mistake.

    DOMAIN, SUBJECTS, MONITORS, CATALOG, STORE          identity and content
    CONFIRM_RUNS, CONFIRM_CYCLES, HEAL_*, SAMPLE_*      the tunables
    _enabled()                                          is this domain switched on
    _ingest(snapshot, now)                              read this domain's snapshot
    _validate(sample, subject)                          what is implausible here
    _system_healthy(now)                                what a whole-subsystem outage is
    _build_input(d, subject, now, ctx, system_healthy)  what a monitor may see
    _freeze_evidence(issue, d, out, now, ctx)           conditions at the time

WHAT FEEDS THIS
---------------
observe() is called from the real poll cadences only. It is NOT called from a
conversation turn. Reports are deduplicated by their own timestamp, so being
polled ten times between two source transmissions produces one sample, not ten.
Without that, "valid sample count" would measure the poll rate rather than the
evidence — and every confirmation count downstream would be a measure of how
talkative the driver is.

LLM FIREWALL
------------
Imports the stdlib and this package. No model, no network, no prompt, nothing
that could reach one. The speech DECISION is not made here either — this file
only reports whether a code is eligible to be spoken; the timing and the words
remain vehicle_health_policy.py's, which imports nothing at all.
"""
from __future__ import annotations

import threading
import time
from typing import Dict, List, Optional, Tuple

from . import codes as C
from . import monitors as M
from . import shadow

# --- lifecycle -------------------------------------------------------------
CANDIDATE = "CANDIDATE"
ACTIVE = "ACTIVE"
RESOLVED = "RESOLVED"


def blank_communication() -> dict:
    """The ledger. Every field is about the DRIVER, none about the car."""
    return {
        "firstToldAt": None,
        "lastToldAt": None,
        "lastSeverityTold": None,
        "lastBriefedAt": None,
        "acknowledgedAt": None,
        "announcementCount": 0,
        # "I am watching this" — set when an issue is confirmed, cleared when it
        # resolves. It is what lets RIO answer "are you keeping an eye on it?"
        # without implying she is about to speak again.
        "monitoringActive": False,
        "resolutionMentionedAt": None,
        # Shadow mode: what RIO WOULD have said, and how often. This is the
        # tuning data the whole shadow deployment exists to produce.
        "shadowProposals": 0,
        "lastShadowProposalAt": None,
    }


class DiagnosticEngine:
    """One per domain per process. Single-driver, like everything else here."""

    # --- what a domain declares about itself -------------------------------
    DOMAIN = "base"
    SUBJECTS: Tuple[str, ...] = ()
    MONITORS: Tuple[M.MonitorDefinition, ...] = ()
    CATALOG: Optional[C.CodeCatalog] = None
    STORE = None

    # How many samples per subject are held in memory. Enough to span the
    # longest monitor window at the source's real report rate, and no more: this
    # is working evidence, not history. History is the event log, on disk.
    MAX_SAMPLES = 96

    SAMPLE_MAX_AGE_S = 150.0
    MIN_RUN_SPACING_S = 20.0
    RESOLVED_RETAIN_DAYS = 180.0

    CONFIRM_RUNS: Dict[str, int] = {}
    CONFIRM_CYCLES: Dict[str, int] = {}

    # Monitors whose evidence is SILENCE. A monitor waiting for a new sample
    # before noticing that no sample came would wait forever, which is a real
    # and slightly funny bug to have in a connectivity monitor.
    TIME_BASED_MONITORS: Tuple[str, ...] = ()

    # Reasons the gate gives, in this domain's own words. A monitor's status has
    # to say what it is actually waiting for.
    NO_SUBJECT_REASON = "nothing has ever reported on this subject"
    SYSTEM_UNHEALTHY_REASON = "the subsystem is not reporting at all"

    # A second key under which the subject appears in views and issue records,
    # for a domain whose subject has a name its consumers already use.
    SUBJECT_ALIAS: Optional[str] = None

    # The id prefix for this domain's drive cycles.
    CYCLE_PREFIX = "drive"

    def __init__(self, load: bool = True, cycles=None):
        self._lock = threading.RLock()
        # Working evidence, in memory only. Deliberately NOT persisted: after a
        # restart a trend monitor genuinely does not have comparable readings
        # any more, and pretending otherwise by reloading samples would let it
        # report READY on evidence it cannot actually see. It goes NOT_READY,
        # which is the honest answer.
        self._samples: Dict[str, List[M.Sample]] = {s: [] for s in self.SUBJECTS}
        self._last_report_at: Dict[str, float] = {}
        self._last_run_at: Dict[str, float] = {}
        # Monitor status/result, held apart. Status is about whether it could
        # run; result is about what it found. Result survives a restart; status
        # does not, because status is a statement about evidence this process
        # holds.
        self._status: Dict[str, str] = {}
        self._status_reason: Dict[str, str] = {}
        self.cycles = cycles if cycles is not None else self._make_cycles()

        from .store import blank_state
        self._state = self.STORE.load_state() if load else blank_state()
        self._state.setdefault("issues", {})
        self._state.setdefault("monitors", {})
        self._state.setdefault("epochs", {})

        # Every monitor starts NOT_READY after a load: the results came back off
        # disk, the evidence did not.
        for key in self._state["monitors"]:
            self._status[key] = M.NOT_READY
            self._status_reason[key] = "restarted — no samples in this process yet"

    def _make_cycles(self):
        from .drivecycle import DriveCycleTracker
        return DriveCycleTracker(self.STORE, id_prefix=self.CYCLE_PREFIX)

    # ------------------------------------------------------------------
    # Hooks a domain fills in
    # ------------------------------------------------------------------
    def _enabled(self) -> bool:
        return True

    def _ingest(self, snapshot: dict, now: float) -> Dict[str, bool]:
        """New reports only. -> {subject: True} for subjects that actually spoke."""
        raise NotImplementedError

    def _validate(self, sample: M.Sample, subject: str) -> None:
        """The plausibility gate. Runs on every reading, before any monitor.

        A sample the gate rejects is KEPT with valid=False: how often a source
        talks nonsense is exactly what a plausibility monitor runs on, and a list
        with the bad ones deleted cannot support that judgement.
        """

    def _system_healthy(self, now: float) -> bool:
        """False only when the whole subsystem has gone quiet.

        This is the guard that stops one subsystem fault becoming N subject
        faults, and it is checked before any per-subject monitor runs.
        """
        return True

    def _build_input(self, d: M.MonitorDefinition, subject: Optional[str],
                     now: float, ctx: dict, system_healthy: bool) -> M.MonitorInput:
        raise NotImplementedError

    def _freeze_evidence(self, issue: dict, d: M.MonitorDefinition,
                         out: M.Outcome, now: float, ctx: dict) -> dict:
        """The domain-specific half of a freeze frame. Numbers, never sentences."""
        return {}

    # ------------------------------------------------------------------
    # Sample intake
    # ------------------------------------------------------------------
    def observe(self, snapshot: dict, now: float = None,
                moving: bool = None, speed_mph: float = None,
                session_id: str = None) -> dict:
        """One poll. -> a summary of what ran and what changed."""
        if not self._enabled():
            return {"enabled": False}
        now = time.time() if now is None else float(now)
        ctx = {"moving": bool(moving), "speed_mph": speed_mph,
               "session_id": session_id}

        with self._lock:
            new = self._ingest(snapshot, now)
            self.cycles.observe(speed_mph, now,
                                active_issue_ids=self.active_issue_ids())
            system_healthy = self._system_healthy(now)
            ran = self._run_monitors(now, ctx, system_healthy, new)
            if ran["changed"]:
                self.STORE.save_state(self._state)
            return {
                "enabled": True,
                "domain": self.DOMAIN,
                "new_samples": new,
                "system_healthy": system_healthy,
                # The tire half has always called this receiver health. Kept so
                # a caller written against the old shape still reads.
                "receiver_healthy": system_healthy,
                "drive_cycle_id": self.cycles.cycle_id,
                "runs": ran["runs"],
                "changed": ran["changed"],
            }

    def _push(self, subject: str, sample: M.Sample) -> None:
        """Validate and ring-buffer one sample. Domains call this from _ingest."""
        self._validate(sample, subject)
        ring = self._samples.setdefault(subject, [])
        ring.append(sample)
        if len(ring) > self.MAX_SAMPLES:
            del ring[:len(ring) - self.MAX_SAMPLES]

    # ------------------------------------------------------------------
    # Running the monitors
    # ------------------------------------------------------------------
    def _run_monitors(self, now: float, ctx: dict, system_healthy: bool,
                      new: Dict[str, bool]) -> dict:
        runs, changed = [], False

        for d in self.MONITORS:
            targets = self.SUBJECTS if d.per_subject else (None,)
            for subject in targets:
                key = self._key(d.monitor_id, subject)

                # A run needs NEW evidence. Two runs off the same sample are one
                # run counted twice, and every confirmation count in this file
                # would then be a measure of poll rate rather than persistence.
                fresh = bool(new.get(subject)) if subject else bool(new)
                spaced = (now - self._last_run_at.get(key, -1e9)) \
                    >= self.MIN_RUN_SPACING_S
                # A subsystem outage is exactly when a monitor's STATUS most
                # needs to be right, and it is the one time no new evidence will
                # ever arrive to trigger a run. Without this, a monitor keeps
                # advertising the READY it had before the link died — which is
                # the opposite of what `status` means, and it means it for as
                # long as the outage lasts.
                #
                # The run itself changes nothing: the gate inhibits on the way
                # in, so no verdict is reached and no count advances. What it
                # updates is the honest answer to "could you look right now".
                outage = not system_healthy and spaced
                if not fresh and not outage \
                        and not (spaced and self._needs_time_based_run(d)):
                    continue

                inp = self._build_input(d, subject, now, ctx, system_healthy)
                out = M.run(d, inp)
                self._last_run_at[key] = now
                self._status[key] = out.status
                self._status_reason[key] = out.reason
                self.cycles.note_run(d.monitor_id, out.result)

                row = {"monitor": d.monitor_id, "subject": subject,
                       "status": out.status, "result": out.result,
                       "reason": out.reason}
                if self.SUBJECT_ALIAS:
                    row[self.SUBJECT_ALIAS] = subject
                runs.append(row)

                rec = self._state["monitors"].setdefault(key, {
                    "monitor_id": d.monitor_id, "subject": subject,
                    "last_result": None, "last_result_at": None,
                    "fail_runs": 0, "pass_runs": 0, "runs": 0,
                    "fail_cycles": [], "pass_since": None})
                rec["runs"] = rec.get("runs", 0) + 1

                if out.status in M.NO_VERDICT:
                    # No verdict. The previous last_result stands untouched --
                    # "I could not look" must never overwrite "I looked and it
                    # was fine", in either direction.
                    continue

                rec["last_result"] = out.result
                rec["last_result_at"] = now
                if self._apply(d, subject, out, rec, now, ctx):
                    changed = True

        return {"runs": runs, "changed": changed}

    def _needs_time_based_run(self, d: M.MonitorDefinition) -> bool:
        return d.monitor_id in self.TIME_BASED_MONITORS

    def _active_monitor_ids(self, subject: Optional[str]) -> Tuple[str, ...]:
        return tuple(
            i["monitor_id"] for i in self._state["issues"].values()
            if i.get("lifecycle") == ACTIVE and i.get("subject") == subject)

    def _epoch_started_at(self, subject: Optional[str]) -> float:
        return (self._state["epochs"].get(subject or "system") or {}).get(
            "started_at", 0.0)

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------
    def _apply(self, d: M.MonitorDefinition, subject: Optional[str],
               out: M.Outcome, rec: dict, now: float, ctx: dict) -> bool:
        """One verdict, applied to the record. -> True if anything changed."""
        code = self.CATALOG.code_for(d.monitor_id, subject, out.variant)
        if code is None:
            return False
        issue_id = f"{d.monitor_id}:{subject or 'system'}:{code.code}"
        issue = self._state["issues"].get(issue_id)

        if out.result == M.PASSED:
            rec["fail_runs"] = 0
            rec["pass_runs"] = rec.get("pass_runs", 0) + 1
            if rec.get("pass_since") is None:
                rec["pass_since"] = now
            return self._on_pass(d, issue_id, issue, rec, now, out)

        # FAILED_PENDING from here down.
        rec["pass_runs"] = 0
        rec["pass_since"] = None
        rec["fail_runs"] = rec.get("fail_runs", 0) + 1
        cycle = self.cycles.cycle_id
        if cycle and cycle not in rec["fail_cycles"]:
            rec["fail_cycles"].append(cycle)

        return self._on_fail(d, code, issue_id, issue, rec, out, now, ctx)

    def _on_fail(self, d, code, issue_id, issue, rec, out, now, ctx) -> bool:
        confirm_runs = self.CONFIRM_RUNS.get(d.monitor_id, 2)
        confirm_cycles = self.CONFIRM_CYCLES.get(d.monitor_id, 0)
        severity = out.severity or code.default_severity

        # The urgent one-trip path. It skips the RUN COUNT, not the validation:
        # every gate in the domain's monitors has already been passed to get
        # here, and those gates are what a single malformed reading, a wake-up
        # frame or a subsystem-wide loss cannot get through.
        urgent = bool(out.urgent and self.CATALOG.fast_path_eligible(code.code))

        enough_runs = rec["fail_runs"] >= confirm_runs
        enough_cycles = len(rec["fail_cycles"]) >= confirm_cycles
        confirmable = urgent or (enough_runs and enough_cycles)

        if issue is None:
            issue = self._new_issue(issue_id, d, code, severity, now, out)
            self._state["issues"][issue_id] = issue
            self.STORE.append_event("issue_candidate", {
                "issue_id": issue_id, "code": code.code, "monitor": d.monitor_id,
                "subject": code.subject, "domain": self.DOMAIN,
                "severity": severity,
                "confidence": out.confidence, "reason": out.reason,
                "fail_runs": rec["fail_runs"], "detail": out.detail}, at=now)

        issue["last_seen_at"] = now
        issue["confidence"] = out.confidence
        issue["reason"] = out.reason
        issue["detail"] = out.detail
        # Any progress toward healing is void the moment the monitor fails
        # again. Leaving it stale would leave an issue that is actively failing
        # still advertising "2 of 2 passing runs" — which the announcement
        # policy reads to decide whether to stay quiet, so a stale value there
        # would silence a fault that is getting worse.
        issue["pass_runs"] = 0
        issue["healing_progress"] = {}
        issue["fail_runs"] = rec["fail_runs"]
        issue["fail_cycles"] = list(rec["fail_cycles"])
        issue["urgent"] = urgent

        was = issue["lifecycle"]
        prev_severity = issue["severity"]

        if issue["lifecycle"] == RESOLVED:
            # It came back. A recurrence is a new occurrence of a KNOWN problem,
            # which is more informative than a first occurrence, not less — so
            # the counter goes up and the history stays.
            issue["recurrence"]["count"] += 1
            issue["recurrence"]["previous_resolved_at"].append(issue.get("resolved_at"))
            issue["resolved_at"] = None
            issue["lifecycle"] = CANDIDATE
            issue["communication"]["resolutionMentionedAt"] = None
            self.STORE.append_event("issue_recurred", {
                "issue_id": issue_id, "code": code.code, "domain": self.DOMAIN,
                "recurrence_count": issue["recurrence"]["count"]}, at=now)

        if confirmable and issue["lifecycle"] != ACTIVE:
            issue["lifecycle"] = ACTIVE
            issue["confirmed_at"] = now
            issue["severity"] = severity
            issue["communication"]["monitoringActive"] = True
            self._freeze_frame(issue, d, out, now, ctx,
                               why="confirmed", urgent=urgent)
            self.cycles.note_issue_created(issue_id)
            self.STORE.append_event("issue_confirmed", {
                "issue_id": issue_id, "code": code.code, "monitor": d.monitor_id,
                "subject": code.subject, "domain": self.DOMAIN,
                "severity": severity,
                "confidence": out.confidence, "reason": out.reason,
                "urgent": urgent, "fail_runs": rec["fail_runs"],
                "fail_cycles": list(rec["fail_cycles"]),
                "drive_cycle_id": self.cycles.cycle_id}, at=now)
        elif issue["lifecycle"] == ACTIVE and \
                C.SEVERITY_RANK.get(severity, 0) > C.SEVERITY_RANK.get(prev_severity, 0):
            # A material increase in severity gets its own snapshot. The first
            # freeze frame describes the fault we found; this one describes the
            # fault it became, and overwriting the first would destroy the only
            # record of what it looked like when it started.
            issue["severity"] = severity
            self._freeze_frame(issue, d, out, now, ctx,
                               why="severity_increase", urgent=urgent)
            self.STORE.append_event("issue_severity_increase", {
                "issue_id": issue_id, "code": code.code, "domain": self.DOMAIN,
                "from": prev_severity, "to": severity,
                "reason": out.reason}, at=now)

        return issue["lifecycle"] != was or issue["severity"] != prev_severity or True

    def _on_pass(self, d, issue_id, issue, rec, now, out) -> bool:
        """A passing run. Healing is the only thing that can resolve an issue.

        Never on one good sample. A warm tire on a motorway reads a PSI or two
        higher than the same tire cold, and a system that resolves a low-pressure
        issue on that reading has not observed a repair — it has observed the
        weather, and will re-raise the same issue tonight. The equivalent in the
        powertrain domain is a coolant temperature that fell because the car
        reached a motorway, not because anything was fixed.
        """
        if issue is None or issue["lifecycle"] == RESOLVED:
            return False

        heal = d.healing
        runs_ok = rec["pass_runs"] >= heal.required_passing_monitor_runs
        stable_ok = (rec.get("pass_since") is not None
                     and (now - rec["pass_since"]) >= heal.minimum_stable_duration_seconds)
        issue["pass_runs"] = rec["pass_runs"]
        issue["healing_progress"] = {
            "passing_runs": rec["pass_runs"],
            "required_runs": heal.required_passing_monitor_runs,
            "stable_for_s": (None if rec.get("pass_since") is None
                             else round(now - rec["pass_since"], 1)),
            "required_stable_s": heal.minimum_stable_duration_seconds,
        }

        if not (runs_ok and stable_ok):
            return True

        if issue["lifecycle"] == CANDIDATE:
            # A pending condition that self-clears. It never became an Issue the
            # driver could be told about, and it leaves a trace anyway — a
            # candidate that keeps appearing and clearing is itself a finding.
            issue["lifecycle"] = RESOLVED
            issue["resolved_at"] = now
            self.STORE.append_event("candidate_cleared", {
                "issue_id": issue_id, "code": issue["code"],
                "domain": self.DOMAIN,
                "passing_runs": rec["pass_runs"]}, at=now)
            return True

        issue["lifecycle"] = RESOLVED
        issue["resolved_at"] = now
        issue["communication"]["monitoringActive"] = False
        self.cycles.note_issue_resolved(issue_id)
        self.STORE.append_event("issue_resolved", {
            "issue_id": issue_id, "code": issue["code"], "domain": self.DOMAIN,
            "passing_runs": rec["pass_runs"],
            "stable_for_s": issue["healing_progress"]["stable_for_s"],
            "reason": out.reason,
            "recurrence_count": issue["recurrence"]["count"],
            "drive_cycle_id": self.cycles.cycle_id}, at=now)
        return True

    def _new_issue(self, issue_id, d, code, severity, now, out) -> dict:
        issue = {
            "issue_id": issue_id,
            "code": code.code,
            "domain": self.DOMAIN,
            "monitor_id": d.monitor_id,
            "component": code.component_type,
            "subject": code.subject,
            "lifecycle": CANDIDATE,
            "severity": severity,
            "created_at": now,
            "confirmed_at": None,
            "resolved_at": None,
            "last_seen_at": now,
            "confidence": out.confidence,
            "reason": out.reason,
            "detail": out.detail,
            "fail_runs": 0,
            "fail_cycles": [],
            "pass_runs": 0,
            "healing_progress": {},
            "urgent": False,
            "freeze_frames": [],
            "recurrence": {"count": 0, "first_seen_at": now,
                           "previous_resolved_at": []},
            "communication": blank_communication(),
        }
        if self.SUBJECT_ALIAS:
            issue[self.SUBJECT_ALIAS] = code.subject
        return issue

    # ------------------------------------------------------------------
    # Freeze frames
    # ------------------------------------------------------------------
    def _freeze_frame(self, issue: dict, d: M.MonitorDefinition, out: M.Outcome,
                      now: float, ctx: dict, why: str, urgent: bool) -> dict:
        """The evidence, as it stood at the moment of the decision.

        Never rewritten by a later reading. That is the whole value of it: three
        weeks later the question is not what the sensor reads now, it is what we
        were looking at when we decided — and a "freeze frame" that tracked the
        current value would answer the wrong one.

        The identity fields are built here; everything measured comes from the
        domain. Raw source identifiers are deliberately absent: a sensor id is a
        stable identifier for a physical object that travels with the car, it is
        of no use in a sentence, and the conversation layer has no business
        holding one.
        """
        frame = {
            "issue_id": issue["issue_id"],
            "code": issue["code"],
            "domain": self.DOMAIN,
            "captured_at": now,
            "capture_reason": why,
            "triggering_monitor": d.monitor_id,
            "drive_cycle_id": self.cycles.cycle_id,
            "urgent_path": urgent,
            "moving": bool(ctx.get("moving")),
            "monitor_runs": self._state["monitors"].get(
                self._key(d.monitor_id, issue.get("subject")), {}).get("runs", 0),
            "monitor_detail": dict(out.detail),
            "confidence": out.confidence,
            "severity": issue["severity"],
        }
        frame.update(self._freeze_evidence(issue, d, out, now, ctx))

        # Only the fields this code declared, plus the identity ones. A freeze
        # frame that carried every field for every code would make the service
        # view unreadable and would imply evidence the monitor never looked at.
        wanted = set(d.freeze_frame_fields)
        keep = {k: v for k, v in frame.items()
                if k in wanted or k in ("issue_id", "code", "domain",
                                        "captured_at", "capture_reason",
                                        "triggering_monitor", "drive_cycle_id",
                                        "urgent_path", "moving",
                                        "monitor_detail", "confidence",
                                        "severity")}
        issue["freeze_frames"].append(keep)
        self.STORE.append_event("freeze_frame", keep, at=now)
        return keep

    # ------------------------------------------------------------------
    # The communication ledger — separate from everything above
    # ------------------------------------------------------------------
    def note_announced(self, issue_id: str, severity: str, now: float = None) -> bool:
        """RIO said it out loud. Nothing about the car changed."""
        now = time.time() if now is None else now
        with self._lock:
            issue = self._state["issues"].get(issue_id)
            if issue is None:
                return False
            comm = issue["communication"]
            if comm["firstToldAt"] is None:
                comm["firstToldAt"] = now
            comm["lastToldAt"] = now
            comm["lastSeverityTold"] = severity
            comm["announcementCount"] += 1
            self.STORE.append_event("announced", {
                "issue_id": issue_id, "code": issue["code"],
                "domain": self.DOMAIN, "severity": severity,
                "announcement_count": comm["announcementCount"]}, at=now)
            self.STORE.save_state(self._state)
            return True

    def note_shadow_proposal(self, issue_id: str, text: str, severity: str,
                             reason: str, now: float = None) -> bool:
        """What RIO WOULD have said. The reason shadow mode exists.

        Recorded in exactly the place a real announcement would be recorded, so
        that turning a monitor on later changes one flag and nothing else — and
        so the log answers "how often would this have interrupted somebody" for
        real, from real drives, before anyone is interrupted.
        """
        now = time.time() if now is None else now
        with self._lock:
            issue = self._state["issues"].get(issue_id)
            if issue is None:
                return False
            comm = issue["communication"]
            comm["shadowProposals"] += 1
            comm["lastShadowProposalAt"] = now
            self.STORE.append_event("shadow_proposal", {
                "issue_id": issue_id, "code": issue["code"],
                "domain": self.DOMAIN, "severity": severity,
                "would_have_said": text, "policy_reason": reason,
                "proposal_count": comm["shadowProposals"]}, at=now)
            self.STORE.save_state(self._state)
            return True

    def note_briefed(self, issue_ids: List[str], now: float = None) -> None:
        """Mentioned in a drive-start briefing, which is not an interruption."""
        now = time.time() if now is None else now
        with self._lock:
            for iid in issue_ids:
                issue = self._state["issues"].get(iid)
                if issue:
                    issue["communication"]["lastBriefedAt"] = now
            self.STORE.save_state(self._state)

    def note_acknowledged(self, issue_id: str, now: float = None) -> bool:
        with self._lock:
            issue = self._state["issues"].get(issue_id)
            if issue is None:
                return False
            issue["communication"]["acknowledgedAt"] = \
                time.time() if now is None else now
            self.STORE.save_state(self._state)
            return True

    # ------------------------------------------------------------------
    # Reset and relearn
    # ------------------------------------------------------------------
    def relearn(self, subject: str = None, reason: str = "", by: str = "driver",
                now: float = None) -> dict:
        """Sources replaced, parts changed, or a baseline deliberately reset.

        Deletes nothing. A relearn says "stop comparing against what came
        before", not "that never happened" — the history is what makes "this is
        the second time on this one" answerable, and a reset that erased it
        would let a chronic problem look new forever.

        Trend monitors go NOT_READY, because they genuinely are. Absolute
        threshold monitoring stays available the moment a reliable reading
        exists: a tire at 12 PSI is at 12 PSI whether or not we have learned its
        new sensor, and a relearn must never suppress a validated critical
        condition.
        """
        now = time.time() if now is None else now
        subjects = [subject] if subject else list(self.SUBJECTS)
        with self._lock:
            previous_map = {s: len(self._samples.get(s) or []) for s in self.SUBJECTS}
            for s in subjects:
                self._state["epochs"][s] = {
                    "started_at": now, "reason": reason, "by": by,
                    "previous_sample_count": previous_map.get(s, 0),
                }
                self._samples[s] = []
                self._last_report_at.pop(s, None)
                for d in self.MONITORS:
                    if not d.per_subject:
                        continue
                    key = self._key(d.monitor_id, s)
                    rec = self._state["monitors"].get(key)
                    if rec is not None:
                        # Counters reset; last_result does NOT. What the monitor
                        # last found is a fact about the past and survives.
                        rec["fail_runs"] = 0
                        rec["pass_runs"] = 0
                        rec["pass_since"] = None
                        rec["fail_cycles"] = []
                    self._status[key] = M.NOT_READY
                    self._status_reason[key] = "relearning — no samples in this epoch"
            self.STORE.append_event("relearn", {
                "domain": self.DOMAIN, "subjects": subjects,
                "reason": reason, "by": by,
                "previous_sample_counts": previous_map,
                "active_issues_preserved": self.active_issue_ids()}, at=now)
            self.STORE.save_state(self._state)
            out = {"subjects": subjects, "at": now, "by": by, "reason": reason}
            if self.SUBJECT_ALIAS:
                out[self.SUBJECT_ALIAS + "s"] = subjects
            return out

    # ------------------------------------------------------------------
    # Views
    # ------------------------------------------------------------------
    @staticmethod
    def _key(monitor_id: str, subject: Optional[str]) -> str:
        return f"{monitor_id}|{subject or 'system'}"

    def active_issue_ids(self) -> List[str]:
        return [i["issue_id"] for i in self._state["issues"].values()
                if i.get("lifecycle") == ACTIVE]

    def issues(self, lifecycle: str = None) -> List[dict]:
        out = [dict(i) for i in self._state["issues"].values()
               if lifecycle is None or i.get("lifecycle") == lifecycle]
        out.sort(key=lambda i: (-C.SEVERITY_RANK.get(i["severity"], 0),
                                -(i.get("confidence") or 0.0), i["issue_id"]))
        return out

    def monitor_view(self) -> List[dict]:
        """Status and last_result, side by side and never conflated.

        This is the readiness view, and the reason it exists in this shape: a
        monitor that has not run has NO result, and a caller must be able to see
        that rather than infer a pass from silence.
        """
        out = []
        for d in self.MONITORS:
            targets = self.SUBJECTS if d.per_subject else (None,)
            for subject in targets:
                key = self._key(d.monitor_id, subject)
                rec = self._state["monitors"].get(key, {})
                row = {
                    "monitor_id": d.monitor_id,
                    "domain": self.DOMAIN,
                    "subject": subject,
                    "component": d.component_type,
                    "status": self._status.get(key, M.NOT_READY),
                    "status_reason": self._status_reason.get(
                        key, "has not run in this process yet"),
                    "last_result": rec.get("last_result"),
                    "last_result_at": rec.get("last_result_at"),
                    "runs": rec.get("runs", 0),
                    "fail_runs": rec.get("fail_runs", 0),
                    "pass_runs": rec.get("pass_runs", 0),
                    "fail_cycles": rec.get("fail_cycles", []),
                    "confirm_runs_required":
                        self.CONFIRM_RUNS.get(d.monitor_id, 2),
                    "confirm_cycles_required":
                        self.CONFIRM_CYCLES.get(d.monitor_id, 0),
                    "epoch_started_at": self._epoch_started_at(subject) or None,
                }
                if self.SUBJECT_ALIAS:
                    row[self.SUBJECT_ALIAS] = subject
                out.append(row)
        return out

    def status_of(self, monitor_id: str, subject: str = None) -> Tuple[str, Optional[str]]:
        """-> (status, last_result). The two fields, for a caller that wants both."""
        key = self._key(monitor_id, subject)
        rec = self._state["monitors"].get(key, {})
        return self._status.get(key, M.NOT_READY), rec.get("last_result")

    def state(self) -> dict:
        return {
            "domain": self.DOMAIN,
            "shadow_mode": shadow.is_shadowed(self.DOMAIN),
            "drive_cycle": self.cycles.state(),
            "monitors": self.monitor_view(),
            "issues": self.issues(),
            "active": self.issues(ACTIVE),
            "candidates": self.issues(CANDIDATE),
            "epochs": dict(self._state["epochs"]),
            "samples_held": {s: len(v) for s, v in self._samples.items()},
        }

    def prune_resolved(self, now: float = None) -> int:
        """Drop resolved issues past the retention window from the working set.

        The event log keeps them regardless — this only bounds the state file.
        Nothing here is allowed to make a problem look repaired: an issue is
        removed only long after it resolved, and only from the working set.
        """
        now = time.time() if now is None else now
        cutoff = now - self.RESOLVED_RETAIN_DAYS * 86400.0
        drop = [k for k, i in self._state["issues"].items()
                if i.get("lifecycle") == RESOLVED
                and (i.get("resolved_at") or now) < cutoff]
        for k in drop:
            del self._state["issues"][k]
        if drop:
            self.STORE.save_state(self._state)
        return len(drop)
