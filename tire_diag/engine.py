"""engine.py — the runner. Samples in, monitor results and Issues out.

This is where the stateful half lives: counting monitor runs, promoting a
CANDIDATE to ACTIVE, healing an ACTIVE back to RESOLVED, freezing evidence at
the moment of confirmation, tracking recurrence, and keeping the communication
ledger. monitors.py stays pure; everything that has to remember something is
here.

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
DATA_UNAVAILABLE. Folding any of them in is how a system ends up unable to
answer "is this still wrong?" separately from "have I mentioned it?" — and those
questions have genuinely different answers all the time. A tire can be worse and
already announced; a resolved issue can still be worth mentioning once.

This is OBD's own separation: storing a diagnostic condition and commanding the
malfunction lamp are different decisions, made by different logic, on different
evidence.

WHAT FEEDS THIS
---------------
observe() is called from the real poll cadences only — /vehicle/tires and
/vehicle/health/announcement. It is NOT called from a conversation turn. Reports
are deduplicated by their own timestamp, so being polled ten times between two
sensor transmissions produces one sample, not ten. Without that, "valid sample
count" would measure the poll rate rather than the evidence.

LLM FIREWALL
------------
Imports config, the stdlib and this package. No model, no network, no prompt,
nothing that could reach one. The speech DECISION is not made here either — this
file only reports whether a code is eligible to be spoken; the timing and the
words remain vehicle_health_policy.py's, which imports nothing at all.
"""
from __future__ import annotations

import threading
import time
from typing import Dict, List, Optional, Tuple

import config

from . import codes as C
from . import monitors as M
from . import store
from .drivecycle import DriveCycleTracker

# --- lifecycle -------------------------------------------------------------
CANDIDATE = "CANDIDATE"
ACTIVE = "ACTIVE"
RESOLVED = "RESOLVED"

# How many samples per corner are held in memory. Enough to span the longest
# monitor window at the sensor's real report rate, and no more: this is working
# evidence, not history. History is the event log, and it is on disk.
_MAX_SAMPLES = 96


def _blank_communication() -> dict:
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


class TireDiagnosticEngine:
    """One per process. Single-driver, like nav's route registry and _last_talk."""

    def __init__(self, load: bool = True):
        self._lock = threading.RLock()
        # Working evidence, in memory only. Deliberately NOT persisted: after a
        # restart a trend monitor genuinely does not have comparable readings
        # any more, and pretending otherwise by reloading samples would let it
        # report READY on evidence it cannot actually see. It goes NOT_READY,
        # which is the honest answer and is what test 19 asserts.
        self._samples: Dict[str, List[M.Sample]] = {c: [] for c in C.CORNERS}
        self._last_report_at: Dict[str, float] = {}
        self._last_run_at: Dict[str, float] = {}
        # Monitor status/result, split per amendment 2. Status is about whether
        # it could run; result is about what it found. Result survives a
        # restart; status does not, because status is a statement about evidence
        # this process holds.
        self._status: Dict[str, str] = {}
        self._status_reason: Dict[str, str] = {}
        self.cycles = DriveCycleTracker()

        self._state = store.load_state() if load else store._blank_state()
        self._state.setdefault("issues", {})
        self._state.setdefault("monitors", {})
        self._state.setdefault("epochs", {})

        # Every monitor starts NOT_READY after a load: the results came back off
        # disk, the evidence did not.
        for key, rec in self._state["monitors"].items():
            self._status[key] = M.NOT_READY
            self._status_reason[key] = "restarted — no samples in this process yet"

    # ------------------------------------------------------------------
    # Sample intake
    # ------------------------------------------------------------------
    def observe(self, snapshot: dict, now: float = None,
                moving: bool = None, speed_mph: float = None,
                session_id: str = None) -> dict:
        """One poll. -> a summary of what ran and what changed.

        `snapshot` is tires.snapshot(). Reports are deduped by updated_at, so
        polling faster than the sensors transmit adds no evidence — which is the
        entire reason the enabling conditions can be trusted.
        """
        if not getattr(config, "TIRE_DIAG_ENABLED", True):
            return {"enabled": False}
        now = time.time() if now is None else float(now)

        with self._lock:
            new = self._ingest(snapshot, now)
            self.cycles.observe(speed_mph, now,
                                active_issue_ids=self.active_issue_ids())
            receiver_healthy = self._receiver_healthy(now)
            ran = self._run_monitors(now, bool(moving), speed_mph,
                                     receiver_healthy, new)
            if ran["changed"]:
                store.save_state(self._state)
            return {
                "enabled": True,
                "new_samples": new,
                "receiver_healthy": receiver_healthy,
                "drive_cycle_id": self.cycles.cycle_id,
                "runs": ran["runs"],
                "changed": ran["changed"],
            }

    def _ingest(self, snapshot: dict, now: float) -> Dict[str, bool]:
        """New reports only. -> {corner: True} for corners that actually spoke.

        Validation happens here, once, so every monitor sees the same verdict on
        the same packet. A sample the gate rejects is KEPT with valid=False:
        how often a sensor talks nonsense is exactly what the plausibility
        monitor runs on, and a list with the bad ones deleted cannot support
        that judgement.
        """
        new: Dict[str, bool] = {}
        for row in (snapshot.get("tires") or []):
            corner = row.get("corner")
            if corner not in C.CORNERS:
                continue
            raw = row.get("raw") or {}
            at = raw.get("updated_at")
            connected = bool(raw.get("connected"))

            if at is None:
                continue
            if self._last_report_at.get(corner) == at:
                continue          # same transmission, seen again by a faster poll
            self._last_report_at[corner] = at

            sample = M.Sample(
                corner=corner, at=float(at),
                pressure_psi=raw.get("pressure_psi"),
                temp_f=raw.get("temp_f"),
                battery_pct=raw.get("battery_pct"),
                connected=connected)
            self._validate(sample, corner)
            ring = self._samples[corner]
            ring.append(sample)
            if len(ring) > _MAX_SAMPLES:
                del ring[:len(ring) - _MAX_SAMPLES]
            new[corner] = True
            self.cycles.note_sensor(corner)
        return new

    def _validate(self, s: M.Sample, corner: str) -> None:
        """The plausibility gate. Runs on every packet, before any monitor.

        Two rejections, and one retraction:

          out of range     a tire pressure is not 90 PSI and not 0.4 PSI. This
                           is a malformed packet, full stop.

          impossible step  more than TIRE_IMPLAUSIBLE_STEP_PSI since the last
                           good report. Rejected PROVISIONALLY, because the one
                           thing that legitimately produces a step this large is
                           the thing we most need to catch: a tire going down
                           fast.

          retraction       if the NEXT report is consistent with the rejected
                           one, the step was real and both are accepted. Two
                           reports corroborating each other is the same
                           discipline as everything else here — a single frame
                           never establishes anything, and a single frame never
                           suppresses anything either.
        """
        lo, hi = config.TIRE_PLAUSIBLE_RANGE_PSI
        if not s.connected:
            return
        if s.pressure_psi is None:
            s.valid = False
            s.reject_reason = "no pressure in the report"
            return
        if not (lo <= s.pressure_psi <= hi):
            s.valid = False
            s.reject_reason = "outside any plausible tire pressure"
            return

        ring = self._samples[corner]
        prev_good = next((x for x in reversed(ring)
                          if x.valid and x.pressure_psi is not None), None)
        step = (abs(s.pressure_psi - prev_good.pressure_psi)
                if prev_good is not None else 0.0)
        step_implausible = step > config.TIRE_IMPLAUSIBLE_STEP_PSI

        # Retraction is checked BEFORE the step is rejected, not after. Order
        # matters: this report may be implausible against the last GOOD one and
        # still corroborate the one provisionally rejected before it, which is
        # precisely the signature of a real, fast pressure loss.
        #
        # Two reports that agree with each other establish a level;
        # BOTH become valid, not just the earlier one. Accepting only the
        # provisionally-rejected sample and leaving this one rejected would mean
        # a tire going down fast never accumulates two consecutive valid
        # readings at the new level — which is exactly what the critical monitor
        # needs, so the urgent path would be unreachable by construction.
        last = ring[-1] if ring else None
        if (last is not None and not last.valid
                and last.reject_reason.startswith("impossible")
                and last.pressure_psi is not None
                and abs(s.pressure_psi - last.pressure_psi)
                <= config.TIRE_IMPLAUSIBLE_STEP_PSI):
            last.valid = True
            last.reject_reason = "corroborated by the next report"
            s.valid = True
            s.reject_reason = "corroborates the previous report"
            return

        if step_implausible:
            s.valid = False
            s.reject_reason = f"impossible {step:.1f} PSI step in one report"

    def _receiver_healthy(self, now: float) -> bool:
        """False only when EVERY corner is silent.

        This is the guard that stops one receiver fault becoming four tire
        faults, and it is checked before any per-corner monitor runs.
        """
        heard = 0
        for corner in C.CORNERS:
            ring = self._samples[corner]
            latest = ring[-1] if ring else None
            if latest is not None and latest.connected \
                    and (now - latest.at) <= config.TIRE_DIAG_MISSED_REPORT_S:
                heard += 1
        return heard > 0

    # ------------------------------------------------------------------
    # Running the monitors
    # ------------------------------------------------------------------
    def _run_monitors(self, now: float, moving: bool, speed_mph: Optional[float],
                      receiver_healthy: bool, new: Dict[str, bool]) -> dict:
        runs, changed = [], False

        for d in M.MONITORS:
            targets = C.CORNERS if d.per_corner else (None,)
            for corner in targets:
                key = self._key(d.monitor_id, corner)

                # A run needs NEW evidence. Two runs off the same sample are one
                # run counted twice, and every confirmation count in this file
                # would then be a measure of poll rate rather than persistence.
                fresh = bool(new.get(corner)) if corner else bool(new)
                spaced = (now - self._last_run_at.get(key, -1e9)) \
                    >= config.TIRE_DIAG_MIN_RUN_SPACING_S
                if not fresh and not (spaced and self._needs_time_based_run(d, key)):
                    continue

                inp = self._build_input(d, corner, now, moving, speed_mph,
                                        receiver_healthy)
                out = M.run(d, inp)
                self._last_run_at[key] = now
                self._status[key] = out.status
                self._status_reason[key] = out.reason
                self.cycles.note_run(d.monitor_id, out.result)

                runs.append({"monitor": d.monitor_id, "corner": corner,
                             "status": out.status, "result": out.result,
                             "reason": out.reason})

                rec = self._state["monitors"].setdefault(key, {
                    "monitor_id": d.monitor_id, "corner": corner,
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
                if self._apply(d, corner, out, rec, now, moving, speed_mph):
                    changed = True

        return {"runs": runs, "changed": changed}

    @staticmethod
    def _needs_time_based_run(d: M.MonitorDefinition, key: str) -> bool:
        """Which monitors must run even when no report arrived.

        Connectivity and receiver health are the ones whose evidence is SILENCE.
        A monitor waiting for a new sample before noticing that no sample came
        would wait forever, which is a real and slightly funny bug to have in a
        connectivity monitor.
        """
        return d.monitor_id in ("tpms.sensor_connectivity", "tpms.receiver_health",
                                "tire.sensor_loss_during_decline")

    def _build_input(self, d: M.MonitorDefinition, corner: Optional[str],
                     now: float, moving: bool, speed_mph: Optional[float],
                     receiver_healthy: bool) -> M.MonitorInput:
        if corner is None:
            peers = {c: list(self._samples[c]) for c in C.CORNERS}
            samples: List[M.Sample] = []
        else:
            samples = list(self._samples[corner])
            peers = {c: list(self._samples[c]) for c in C.CORNERS if c != corner}

        epoch = (self._state["epochs"].get(corner or "system") or {}).get("started_at", 0.0)
        active_ids = tuple(
            i["monitor_id"] for i in self._state["issues"].values()
            if i.get("lifecycle") == ACTIVE and i.get("corner") == corner)

        return M.MonitorInput(
            corner=corner, now=now, samples=samples, peers=peers,
            target_psi=config.TIRE_TARGET_PSI.get(corner or "", 35.0),
            moving=moving, speed_mph=speed_mph,
            parked_for_s=None, receiver_healthy=receiver_healthy,
            epoch_started_at=epoch, drive_cycle_id=self.cycles.cycle_id,
            active_monitor_ids=active_ids)

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------
    def _apply(self, d: M.MonitorDefinition, corner: Optional[str],
               out: M.Outcome, rec: dict, now: float, moving: bool,
               speed_mph: Optional[float]) -> bool:
        """One verdict, applied to the record. -> True if anything changed."""
        code = C.code_for(d.monitor_id, corner, out.variant)
        if code is None:
            return False
        issue_id = f"{d.monitor_id}:{corner or 'system'}:{code.code}"
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

        return self._on_fail(d, code, issue_id, issue, rec, out, now, moving,
                             speed_mph)

    def _on_fail(self, d, code, issue_id, issue, rec, out, now, moving,
                 speed_mph) -> bool:
        confirm_runs = config.TIRE_DIAG_CONFIRM_RUNS.get(d.monitor_id, 2)
        confirm_cycles = config.TIRE_DIAG_CONFIRM_CYCLES.get(d.monitor_id, 0)
        severity = out.severity or code.default_severity

        # The urgent one-trip path. It skips the RUN COUNT, not the validation:
        # every gate in monitors.py has already been passed to get here, and
        # those gates are what a single malformed packet, a wake-up frame, a
        # receiver-wide loss or an unknown sensor cannot get through.
        urgent = bool(out.urgent and C.fast_path_eligible(code.code))

        enough_runs = rec["fail_runs"] >= confirm_runs
        enough_cycles = len(rec["fail_cycles"]) >= confirm_cycles
        confirmable = urgent or (enough_runs and enough_cycles)

        if issue is None:
            issue = self._new_issue(issue_id, d, code, severity, now, out)
            self._state["issues"][issue_id] = issue
            store.append_event("issue_candidate", {
                "issue_id": issue_id, "code": code.code, "monitor": d.monitor_id,
                "corner": code.corner, "severity": severity,
                "confidence": out.confidence, "reason": out.reason,
                "fail_runs": rec["fail_runs"], "detail": out.detail}, at=now)

        issue["last_seen_at"] = now
        issue["confidence"] = out.confidence
        issue["reason"] = out.reason
        issue["detail"] = out.detail
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
            store.append_event("issue_recurred", {
                "issue_id": issue_id, "code": code.code,
                "recurrence_count": issue["recurrence"]["count"]}, at=now)

        if confirmable and issue["lifecycle"] != ACTIVE:
            issue["lifecycle"] = ACTIVE
            issue["confirmed_at"] = now
            issue["severity"] = severity
            issue["communication"]["monitoringActive"] = True
            self._freeze(issue, d, out, now, moving, speed_mph,
                         why="confirmed", urgent=urgent)
            self.cycles.note_issue_created(issue_id)
            store.append_event("issue_confirmed", {
                "issue_id": issue_id, "code": code.code, "monitor": d.monitor_id,
                "corner": code.corner, "severity": severity,
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
            self._freeze(issue, d, out, now, moving, speed_mph,
                         why="severity_increase", urgent=urgent)
            store.append_event("issue_severity_increase", {
                "issue_id": issue_id, "code": code.code,
                "from": prev_severity, "to": severity,
                "reason": out.reason}, at=now)

        return issue["lifecycle"] != was or issue["severity"] != prev_severity or True

    def _on_pass(self, d, issue_id, issue, rec, now, out) -> bool:
        """A passing run. Healing is the only thing that can resolve an issue.

        Never on one good sample. A warm tire on a motorway reads a PSI or two
        higher than the same tire cold, and a system that resolves a low-pressure
        issue on that reading has not observed a repair — it has observed the
        weather, and will re-raise the same issue tonight.
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
            store.append_event("candidate_cleared", {
                "issue_id": issue_id, "code": issue["code"],
                "passing_runs": rec["pass_runs"]}, at=now)
            return True

        issue["lifecycle"] = RESOLVED
        issue["resolved_at"] = now
        issue["communication"]["monitoringActive"] = False
        self.cycles.note_issue_resolved(issue_id)
        store.append_event("issue_resolved", {
            "issue_id": issue_id, "code": issue["code"],
            "passing_runs": rec["pass_runs"],
            "stable_for_s": issue["healing_progress"]["stable_for_s"],
            "reason": out.reason,
            "recurrence_count": issue["recurrence"]["count"],
            "drive_cycle_id": self.cycles.cycle_id}, at=now)
        return True

    def _new_issue(self, issue_id, d, code, severity, now, out) -> dict:
        return {
            "issue_id": issue_id,
            "code": code.code,
            "monitor_id": d.monitor_id,
            "component": code.component_type,
            "corner": code.corner,
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
            "communication": _blank_communication(),
        }

    # ------------------------------------------------------------------
    # Freeze frames
    # ------------------------------------------------------------------
    def _freeze(self, issue: dict, d: M.MonitorDefinition, out: M.Outcome,
                now: float, moving: bool, speed_mph: Optional[float],
                why: str, urgent: bool) -> dict:
        """The evidence, as it stood at the moment of the decision.

        Never rewritten by a later reading. That is the whole value of it: three
        weeks later the question is not what the tire reads now, it is what we
        were looking at when we decided — and a "freeze frame" that tracked the
        current value would answer the wrong one.

        Raw radio identifiers are deliberately absent. A sensor id is a stable
        identifier for a physical object that travels with the car, it is of no
        use in a sentence, and the conversation layer has no business holding
        one.
        """
        corner = issue.get("corner")
        samples = [s for s in self._samples.get(corner or "", [])
                   if s.valid and s.pressure_psi is not None] if corner else []
        latest = samples[-1] if samples else None
        peers = []
        for c in C.CORNERS:
            if c == corner:
                continue
            ring = [s for s in self._samples[c] if s.valid and s.pressure_psi is not None]
            if ring:
                peers.append(ring[-1].pressure_psi)

        frame = {
            "issue_id": issue["issue_id"],
            "code": issue["code"],
            "captured_at": now,
            "capture_reason": why,
            "triggering_monitor": d.monitor_id,
            "drive_cycle_id": self.cycles.cycle_id,
            "urgent_path": urgent,
            "vehicle_speed_mph": speed_mph,
            "moving": bool(moving),
            "current_pressure_psi": latest.pressure_psi if latest else None,
            "target_pressure_psi": config.TIRE_TARGET_PSI.get(corner or "", None),
            "temperature_f": latest.temp_f if latest else None,
            "peer_average_pressure_psi": (round(sum(peers) / len(peers), 2)
                                          if peers else None),
            "pressure_change_psi": out.detail.get("change_psi"),
            "valid_sample_count": len(samples),
            "first_valid_sample_at": samples[0].at if samples else None,
            "last_valid_sample_at": latest.at if latest else None,
            "data_quality": self._data_quality(corner),
            "sensor_battery": self._battery_word(latest),
            "receiver_status": ("healthy" if self._receiver_healthy(now)
                                else "not reporting"),
            "monitor_runs": self._state["monitors"].get(
                self._key(d.monitor_id, corner), {}).get("runs", 0),
            "monitor_detail": dict(out.detail),
            "confidence": out.confidence,
            "severity": issue["severity"],
        }
        # Only the fields this code declared, plus the identity ones. A freeze
        # frame that carried every field for every code would make the service
        # view unreadable and would imply evidence the monitor never looked at.
        wanted = set(d.freeze_frame_fields)
        keep = {k: v for k, v in frame.items()
                if k in wanted or k in ("issue_id", "code", "captured_at",
                                        "capture_reason", "triggering_monitor",
                                        "drive_cycle_id", "urgent_path", "moving",
                                        "monitor_detail", "confidence", "severity")}
        issue["freeze_frames"].append(keep)
        store.append_event("freeze_frame", keep, at=now)
        return keep

    def _data_quality(self, corner: Optional[str]) -> str:
        if not corner:
            return "unknown"
        ring = self._samples.get(corner) or []
        if not ring:
            return "none"
        recent = ring[-8:]
        bad = sum(1 for s in recent if not s.valid)
        if bad == 0:
            return "good"
        return "degraded" if bad < len(recent) / 2 else "poor"

    @staticmethod
    def _battery_word(sample: Optional[M.Sample]) -> str:
        if sample is None or sample.battery_pct is None:
            return "unknown"
        if sample.battery_pct <= config.TIRE_DYING_BATTERY_PCT:
            return "failing"
        if sample.battery_pct <= config.TIRE_BATTERY_LOW_PCT:
            return "low"
        return "good"

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
            store.append_event("announced", {
                "issue_id": issue_id, "code": issue["code"],
                "severity": severity,
                "announcement_count": comm["announcementCount"]}, at=now)
            store.save_state(self._state)
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
            store.append_event("shadow_proposal", {
                "issue_id": issue_id, "code": issue["code"],
                "severity": severity, "would_have_said": text,
                "policy_reason": reason,
                "proposal_count": comm["shadowProposals"]}, at=now)
            store.save_state(self._state)
            return True

    def note_briefed(self, issue_ids: List[str], now: float = None) -> None:
        """Mentioned in a drive-start briefing, which is not an interruption."""
        now = time.time() if now is None else now
        with self._lock:
            for iid in issue_ids:
                issue = self._state["issues"].get(iid)
                if issue:
                    issue["communication"]["lastBriefedAt"] = now
            store.save_state(self._state)

    def note_acknowledged(self, issue_id: str, now: float = None) -> bool:
        with self._lock:
            issue = self._state["issues"].get(issue_id)
            if issue is None:
                return False
            issue["communication"]["acknowledgedAt"] = \
                time.time() if now is None else now
            store.save_state(self._state)
            return True

    # ------------------------------------------------------------------
    # Reset and relearn
    # ------------------------------------------------------------------
    def relearn(self, corner: str = None, reason: str = "", by: str = "driver",
                now: float = None) -> dict:
        """Sensors replaced, tires rotated, or a baseline deliberately reset.

        Deletes nothing. A relearn says "stop comparing against what came
        before", not "that never happened" — the history is what makes "this is
        the second time on this tire" answerable, and a reset that erased it
        would let a chronic problem look new forever.

        Trend monitors go NOT_READY, because they genuinely are. Absolute
        pressure monitoring stays available the moment a reliable reading
        exists: a tire at 12 PSI is at 12 PSI whether or not we have learned its
        new sensor, and a relearn must never suppress a validated critical
        condition.
        """
        now = time.time() if now is None else now
        corners = [corner] if corner else list(C.CORNERS)
        with self._lock:
            previous_map = {c: len(self._samples.get(c) or []) for c in C.CORNERS}
            for c in corners:
                self._state["epochs"][c] = {
                    "started_at": now, "reason": reason, "by": by,
                    "previous_sample_count": previous_map.get(c, 0),
                }
                self._samples[c] = []
                self._last_report_at.pop(c, None)
                for d in M.MONITORS:
                    if not d.per_corner:
                        continue
                    key = self._key(d.monitor_id, c)
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
            store.append_event("relearn", {
                "corners": corners, "reason": reason, "by": by,
                "previous_sample_counts": previous_map,
                "active_issues_preserved": self.active_issue_ids()}, at=now)
            store.save_state(self._state)
            return {"corners": corners, "at": now, "by": by, "reason": reason}

    # ------------------------------------------------------------------
    # Views
    # ------------------------------------------------------------------
    @staticmethod
    def _key(monitor_id: str, corner: Optional[str]) -> str:
        return f"{monitor_id}|{corner or 'system'}"

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
        for d in M.MONITORS:
            targets = C.CORNERS if d.per_corner else (None,)
            for corner in targets:
                key = self._key(d.monitor_id, corner)
                rec = self._state["monitors"].get(key, {})
                out.append({
                    "monitor_id": d.monitor_id,
                    "corner": corner,
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
                        config.TIRE_DIAG_CONFIRM_RUNS.get(d.monitor_id, 2),
                    "confirm_cycles_required":
                        config.TIRE_DIAG_CONFIRM_CYCLES.get(d.monitor_id, 0),
                    "epoch_started_at": (self._state["epochs"].get(
                        corner or "system") or {}).get("started_at"),
                })
        return out

    def status_of(self, monitor_id: str, corner: str = None) -> Tuple[str, Optional[str]]:
        """-> (status, last_result). The two fields, for a caller that wants both."""
        key = self._key(monitor_id, corner)
        rec = self._state["monitors"].get(key, {})
        return self._status.get(key, M.NOT_READY), rec.get("last_result")

    def state(self) -> dict:
        return {
            "shadow_mode": bool(config.TIRE_DIAG_SHADOW_MODE),
            "drive_cycle": self.cycles.state(),
            "monitors": self.monitor_view(),
            "issues": self.issues(),
            "active": self.issues(ACTIVE),
            "candidates": self.issues(CANDIDATE),
            "epochs": dict(self._state["epochs"]),
            "samples_held": {c: len(v) for c, v in self._samples.items()},
        }

    def prune_resolved(self, now: float = None) -> int:
        """Drop resolved issues past the retention window from the ACTIVE record.

        The event log keeps them regardless — this only bounds the state file.
        Nothing here is allowed to make a problem look repaired: an issue is
        removed only long after it resolved, and only from the working set.
        """
        now = time.time() if now is None else now
        cutoff = now - config.TIRE_DIAG_RESOLVED_RETAIN_DAYS * 86400.0
        drop = [k for k, i in self._state["issues"].items()
                if i.get("lifecycle") == RESOLVED
                and (i.get("resolved_at") or now) < cutoff]
        for k in drop:
            del self._state["issues"][k]
        if drop:
            store.save_state(self._state)
        return len(drop)


# ---------------------------------------------------------------------------
# The one engine
# ---------------------------------------------------------------------------

_engine: Optional[TireDiagnosticEngine] = None


def engine() -> TireDiagnosticEngine:
    global _engine
    if _engine is None:
        _engine = TireDiagnosticEngine()
    return _engine


def reset_engine(load: bool = True) -> TireDiagnosticEngine:
    """Rebuild from disk. Used by the restart tests, and by nothing else."""
    global _engine
    _engine = TireDiagnosticEngine(load=load)
    return _engine


def observe(snapshot: dict, **kw) -> dict:
    return engine().observe(snapshot, **kw)


def active_issues() -> List[dict]:
    return engine().issues(ACTIVE)


def state() -> dict:
    return engine().state()
