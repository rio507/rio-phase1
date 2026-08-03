"""engine.py — the tire domain, bound to the generic diagnostic runner.

The lifecycle used to live here. It now lives in diag/runner.py, because the
powertrain monitors needed exactly the same one and two copies of "when does a
CANDIDATE become ACTIVE" is two answers to a question that has one. What is left
in this file is the five things that are genuinely about tires:

    _ingest            a tires.snapshot() has four corners in it
    _validate          a tire pressure is not 90 PSI and not 0.4 PSI
    _system_healthy    every corner silent at once is one receiver, not four tires
    _build_input       a tire monitor may see its peers and its placard target
    _freeze_evidence   "the conditions at the time" means pressures and peers

Everything else — confirmation counts, healing, freeze-frame identity fields,
recurrence, the communication ledger, relearn, the views — is inherited and is
not restated here. diag/runner.py's header is the place to read about the three
independent dimensions and why ANNOUNCED is not a lifecycle state.

WHAT FEEDS THIS
---------------
observe() is called from the real poll cadences only — /vehicle/tires and
/vehicle/health/announcement. It is NOT called from a conversation turn. Reports
are deduplicated by their own timestamp, so being polled ten times between two
sensor transmissions produces one sample, not ten. Without that, "valid sample
count" would measure the poll rate rather than the evidence.

LLM FIREWALL
------------
Imports config, the stdlib, diag and this package. No model, no network, no
prompt, nothing that could reach one. The speech DECISION is not made here
either — this file only reports whether a code is eligible to be spoken; the
timing and the words remain vehicle_health_policy.py's, which imports nothing at
all.
"""
from __future__ import annotations

import time
from typing import Dict, List, Optional

import config

from diag.runner import ACTIVE, CANDIDATE, RESOLVED, DiagnosticEngine

from . import codes as C
from . import monitors as M
from . import store
from .drivecycle import DriveCycleTracker

__all__ = ["CANDIDATE", "ACTIVE", "RESOLVED", "TireDiagnosticEngine",
           "engine", "reset_engine", "observe", "active_issues", "state"]


class TireDiagnosticEngine(DiagnosticEngine):
    """One per process. Single-driver, like nav's route registry and _last_talk."""

    DOMAIN = C.DOMAIN
    SUBJECTS = C.CORNERS
    MONITORS = M.MONITORS
    CATALOG = C.CATALOG
    STORE = store.STORE
    SUBJECT_ALIAS = "corner"

    # Enough to span the longest monitor window at the sensor's real report
    # rate, and no more: this is working evidence, not history.
    MAX_SAMPLES = 96

    SAMPLE_MAX_AGE_S = config.TIRE_DIAG_SAMPLE_MAX_AGE_S
    MIN_RUN_SPACING_S = config.TIRE_DIAG_MIN_RUN_SPACING_S
    RESOLVED_RETAIN_DAYS = config.TIRE_DIAG_RESOLVED_RETAIN_DAYS

    CONFIRM_RUNS = config.TIRE_DIAG_CONFIRM_RUNS
    CONFIRM_CYCLES = config.TIRE_DIAG_CONFIRM_CYCLES

    # The monitors whose evidence is SILENCE. A monitor waiting for a new sample
    # before noticing that no sample came would wait forever, which is a real
    # and slightly funny bug to have in a connectivity monitor.
    TIME_BASED_MONITORS = ("tpms.sensor_connectivity", "tpms.receiver_health",
                           "tire.sensor_loss_during_decline")

    NO_SUBJECT_REASON = "no sensor has ever reported on this corner"
    SYSTEM_UNHEALTHY_REASON = "the receiver is not reporting at all"

    def _make_cycles(self):
        return DriveCycleTracker()

    def _enabled(self) -> bool:
        return bool(getattr(config, "TIRE_DIAG_ENABLED", True))

    # ------------------------------------------------------------------
    # Sample intake
    # ------------------------------------------------------------------
    def _ingest(self, snapshot: dict, now: float) -> Dict[str, bool]:
        """New reports only. -> {corner: True} for corners that actually spoke.

        Validation happens in _push, once, so every monitor sees the same verdict
        on the same packet.
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

            self._push(corner, M.Sample(
                subject=corner, at=float(at),
                pressure_psi=raw.get("pressure_psi"),
                temp_f=raw.get("temp_f"),
                battery_pct=raw.get("battery_pct"),
                connected=connected))
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

        ring = self._samples.setdefault(corner, [])
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
        # Two reports that agree with each other establish a level; BOTH become
        # valid, not just the earlier one. Accepting only the
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

    def _system_healthy(self, now: float) -> bool:
        """False only when EVERY corner is silent.

        This is the guard that stops one receiver fault becoming four tire
        faults, and it is checked before any per-corner monitor runs.
        """
        heard = 0
        for corner in C.CORNERS:
            ring = self._samples.get(corner) or []
            latest = ring[-1] if ring else None
            if latest is not None and latest.connected \
                    and (now - latest.at) <= config.TIRE_DIAG_MISSED_REPORT_S:
                heard += 1
        return heard > 0

    # Kept under its original name: it reads better at the call sites in this
    # file, and the freeze frame reports "receiver_status", not "system status".
    _receiver_healthy = _system_healthy

    # ------------------------------------------------------------------
    # What a tire monitor may see
    # ------------------------------------------------------------------
    def _build_input(self, d: M.MonitorDefinition, corner: Optional[str],
                     now: float, ctx: dict, system_healthy: bool) -> M.MonitorInput:
        if corner is None:
            peers = {c: list(self._samples.get(c) or []) for c in C.CORNERS}
            samples: List[M.Sample] = []
        else:
            samples = list(self._samples.get(corner) or [])
            peers = {c: list(self._samples.get(c) or [])
                     for c in C.CORNERS if c != corner}

        return M.MonitorInput(
            subject=corner, now=now, samples=samples, peers=peers,
            target_psi=config.TIRE_TARGET_PSI.get(corner or "", 35.0),
            moving=bool(ctx.get("moving")), speed_mph=ctx.get("speed_mph"),
            parked_for_s=None, system_healthy=system_healthy,
            epoch_started_at=self._epoch_started_at(corner),
            drive_cycle_id=self.cycles.cycle_id,
            active_monitor_ids=self._active_monitor_ids(corner),
            sample_max_age_s=self.SAMPLE_MAX_AGE_S,
            no_subject_reason=self.NO_SUBJECT_REASON,
            system_unhealthy_reason=self.SYSTEM_UNHEALTHY_REASON)

    # ------------------------------------------------------------------
    # Freeze frames
    # ------------------------------------------------------------------
    def _freeze_evidence(self, issue: dict, d: M.MonitorDefinition,
                         out: M.Outcome, now: float, ctx: dict) -> dict:
        """What we were looking at when we decided, in PSI and °F.

        Raw radio identifiers are deliberately absent. A sensor id is a stable
        identifier for a physical object that travels with the car, it is of no
        use in a sentence, and the conversation layer has no business holding
        one.
        """
        corner = issue.get("subject")
        samples = [s for s in (self._samples.get(corner or "") or [])
                   if s.valid and s.pressure_psi is not None] if corner else []
        latest = samples[-1] if samples else None
        peers = []
        for c in C.CORNERS:
            if c == corner:
                continue
            ring = [s for s in (self._samples.get(c) or [])
                    if s.valid and s.pressure_psi is not None]
            if ring:
                peers.append(ring[-1].pressure_psi)

        return {
            "vehicle_speed_mph": ctx.get("speed_mph"),
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
            "receiver_status": ("healthy" if self._system_healthy(now)
                                else "not reporting"),
        }

    def _freeze(self, issue: dict, d: M.MonitorDefinition, out: M.Outcome,
                now: float, moving: bool, speed_mph: Optional[float],
                why: str, urgent: bool) -> dict:
        """The tire-shaped call signature, kept for callers that had it.

        The engine itself goes through _freeze_frame with a context dict; this
        exists because a test that wants to force a severity-increase snapshot
        should not have to know the framework's internal shape.
        """
        return self._freeze_frame(issue, d, out, now,
                                  {"moving": bool(moving), "speed_mph": speed_mph},
                                  why=why, urgent=urgent)

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
    # Relearn, in the tire domain's own vocabulary
    # ------------------------------------------------------------------
    def relearn(self, corner: str = None, reason: str = "", by: str = "driver",
                now: float = None) -> dict:
        """Sensors replaced, tires rotated, or a baseline deliberately reset.

        Deletes nothing. Trend monitors go NOT_READY because they genuinely are;
        absolute pressure monitoring stays available the moment a reliable
        reading exists, and a relearn must never suppress a validated critical
        condition. See DiagnosticEngine.relearn.
        """
        return super().relearn(subject=corner, reason=reason, by=by, now=now)


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
