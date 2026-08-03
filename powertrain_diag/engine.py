"""engine.py — the powertrain domain, bound to the generic diagnostic runner.

The second instance of diag/. Everything about lifecycle, confirmation, healing,
freeze frames, recurrence, the communication ledger and shadow proposals is
inherited and none of it is restated here — diag/runner.py's header is where
that lives. What this file supplies is the five hooks:

    _ingest            a telemetry snapshot is one instant of the whole engine
    _validate          a channel outside its plausible range is not a reading
    _system_healthy    nothing arriving at all is one link fault, not nine
    _build_input       an engine monitor may see this vehicle's own history
    _freeze_evidence   "the conditions at the time" means the whole channel set

ONE SAMPLE PER TICK, NOT ONE PER CHANNEL
----------------------------------------
The opposite of the tire domain, and for the opposite reason. Four tires fail
independently, so a corner is a subject. An engine does not: almost every monitor
here is cross-signal, and a coolant reading without the road speed and load that
go with it cannot answer any of the questions being asked. So a sample carries
every channel at one instant, and there is one subject.

WHY START EVENTS ARE PERSISTED WHEN SAMPLES ARE NOT
---------------------------------------------------
diag/runner.py deliberately does not persist samples: after a restart a trend
monitor genuinely does not have comparable readings any more, and reloading them
would let it report READY on evidence it cannot see.

A start event is not a sample. It is a derived observation of a discrete thing
that happened — the starter loaded the battery and it sagged to 9.4 V — and its
whole value is across drives, days apart. Persisting it is the same decision
insights.py makes about daily baselines, and the honest test is whether the
record could be recomputed from live evidence. A sample could; a start event
that happened last Tuesday could not.

WHERE THE BASELINES COME FROM
-----------------------------
insights.py, which has been accumulating conditioned daily means since long
before this domain existed. §22.4's "vehicle-specific baseline layer" is that
file, and building a second one here would produce two answers to "what does
this car normally do".

LLM FIREWALL
------------
Imports config, the stdlib, diag, insights and this package. No model, no
network, no prompt. The speech decision is not made here either.
"""
from __future__ import annotations

import time
from typing import Dict, List, Optional

import config
import insights

from diag.drivecycle import DriveCycleTracker
from diag.runner import ACTIVE, CANDIDATE, RESOLVED, DiagnosticEngine

from . import codes as C
from . import monitors as M
from . import store

__all__ = ["CANDIDATE", "ACTIVE", "RESOLVED", "PowertrainDiagnosticEngine",
           "engine", "reset_engine", "observe", "active_issues", "state"]

# Channels lifted out of a telemetry snapshot. Everything the nine monitors read,
# and nothing else — a sample carrying every row would put eight tire pressures
# into an engine freeze frame.
_CHANNELS = ("rpm", "vehicle_speed", "coolant_temp", "oil_temp", "oil_pressure",
             "intake_air_temp", "battery_voltage", "map_kpa", "maf_gs",
             "throttle_pct", "engine_load", "stft_b1", "ltft_b1",
             "fuel_pressure", "afr_wideband", "afr_target")

# The conditioned baselines these monitors ask insights.py for.
_BASELINE_KEYS = ("coolant_temp@cruise", "coolant_temp", "battery_voltage@idle",
                  "ltft_b1", "oil_pressure@idle", "fuel_pressure@load")


class PowertrainDiagnosticEngine(DiagnosticEngine):
    """One per process. Single-driver, like everything else stateful here."""

    DOMAIN = C.DOMAIN
    SUBJECTS = C.SUBJECTS
    MONITORS = M.MONITORS
    CATALOG = C.CATALOG
    STORE = store.STORE

    # Engine channels arrive at 1 Hz or faster, so the ring spans a few minutes
    # rather than the tire domain's many hours.
    MAX_SAMPLES = 240

    SAMPLE_MAX_AGE_S = config.POWERTRAIN_SAMPLE_MAX_AGE_S
    MIN_RUN_SPACING_S = config.POWERTRAIN_MIN_RUN_SPACING_S
    RESOLVED_RETAIN_DAYS = config.TIRE_DIAG_RESOLVED_RETAIN_DAYS

    CONFIRM_RUNS = config.POWERTRAIN_CONFIRM_RUNS
    CONFIRM_CYCLES = config.POWERTRAIN_CONFIRM_CYCLES

    # The monitors whose evidence is SILENCE. Both of them have to be able to
    # run when nothing arrived, which is exactly the case they exist for.
    TIME_BASED_MONITORS = ("engine.connection", "engine.new_dtc")

    NO_SUBJECT_REASON = "no engine data has ever arrived"
    SYSTEM_UNHEALTHY_REASON = "no engine data is arriving at all"

    CYCLE_PREFIX = "engine_drive"

    def __init__(self, load: bool = True):
        # Set before super().__init__ because the base constructor may reach for
        # the cycle tracker, and because _build_input can be called before the
        # first observe() in a test.
        self._context: Dict[str, dict] = {"dtc": {}, "link": {}}
        self._cranking_min_v: Optional[float] = None
        self._was_cranking = False
        super().__init__(load=load)
        self._state.setdefault("meta", {}).setdefault("start_events", [])

    def _make_cycles(self):
        return DriveCycleTracker(store,
                                 start_mph=config.TIRE_DIAG_DRIVE_START_MPH,
                                 end_parked_s=config.TIRE_DIAG_DRIVE_END_PARKED_S,
                                 id_prefix=self.CYCLE_PREFIX)

    def _enabled(self) -> bool:
        return bool(getattr(config, "VEHICLE_DIAG_ENABLED", True))

    # ------------------------------------------------------------------
    # Intake
    # ------------------------------------------------------------------
    def observe(self, snapshot: dict, now: float = None, moving: bool = None,
                speed_mph: float = None, session_id: str = None,
                dtc: dict = None, link: dict = None) -> dict:
        """One poll, plus the two things a telemetry snapshot cannot carry.

        The DTC view and the link state come from other subsystems and are
        passed in rather than fetched, so this engine cannot reach sideways into
        the DTC service or the gateway registry — the same reason the monitors
        are handed their baselines instead of reading them.
        """
        self._context = {"dtc": dict(dtc or {}), "link": dict(link or {})}
        return super().observe(snapshot, now=now, moving=moving,
                               speed_mph=speed_mph, session_id=session_id)

    def _ingest(self, snapshot: dict, now: float) -> Dict[str, bool]:
        """One telemetry snapshot -> one sample of the whole engine."""
        rows = {r["id"]: r for r in (snapshot.get("rows") or [])}
        if not rows:
            return {}

        values: Dict[str, Optional[float]] = {}
        statuses: Dict[str, str] = {}
        for channel in _CHANNELS:
            row = rows.get(channel)
            if row is None:
                continue
            statuses[channel] = row.get("status") or ""
            values[channel] = row.get("value")

        running = bool(snapshot.get("engine_running"))
        sample = M.Sample(subject=C.SUBJECT, at=now, connected=True,
                          values=values, statuses=statuses,
                          engine_running=running)
        self._push(C.SUBJECT, sample)
        self._note_start_event(sample, now)
        self.cycles.note_sensor("ecu")
        return {C.SUBJECT: True}

    def _validate(self, s: M.Sample, subject: str) -> None:
        """A channel outside its plausible range is not a reading.

        Per channel, not per sample: one bad coolant value must not discard the
        rpm that came with it. The sample as a whole is only rejected when
        nothing in it is usable, which is the case the connection monitor is
        for.
        """
        from vehicle.signals import registry as R

        rejected = []
        for channel, value in list(s.values.items()):
            if value is None:
                continue
            name = R.canonical(channel)
            if name and not R.in_range(name, value):
                s.values[channel] = None
                rejected.append(channel)
        if rejected:
            s.reject_reason = ("outside any plausible range: "
                               + ", ".join(sorted(rejected)))
        if not s.has_primary():
            s.valid = False
            if not s.reject_reason:
                s.reject_reason = "no channel in this sample carried a value"

    def _note_start_event(self, s: M.Sample, now: float) -> None:
        """Catch the two seconds the starter is loading the battery.

        The evidence for the start-voltage monitor exists only while cranking,
        and only once per drive. Missing it means waiting for the next journey,
        so it is recorded here on the intake path rather than left to a monitor
        that might not run at the right instant.
        """
        rpm = s.values.get("rpm")
        volts = s.values.get("battery_voltage")
        if rpm is None:
            return
        cranking = 50.0 < rpm < config.TELEMETRY_ENGINE_RUNNING_RPM

        if cranking:
            self._was_cranking = True
            if volts is not None:
                self._cranking_min_v = volts if self._cranking_min_v is None \
                    else min(self._cranking_min_v, volts)
            return

        if self._was_cranking and s.engine_running:
            if self._cranking_min_v is not None:
                events = self._state.setdefault("meta", {}).setdefault(
                    "start_events", [])
                events.append({"at": now, "min_v": round(self._cranking_min_v, 3),
                               "drive_cycle_id": self.cycles.cycle_id})
                keep = config.POWERTRAIN_START_EVENTS_KEEP
                if len(events) > keep:
                    del events[:len(events) - keep]
                self.STORE.append_event("start_event", {
                    "min_v": round(self._cranking_min_v, 3),
                    "drive_cycle_id": self.cycles.cycle_id}, at=now)
        self._was_cranking = False
        self._cranking_min_v = None

    def _system_healthy(self, now: float) -> bool:
        """False when nothing usable has arrived inside the window.

        The guard that stops one dead link becoming nine engine faults. The
        connection monitor is exempt from it — see require_system_healthy on
        that definition — because a link outage is the thing it reports.
        """
        ring = self._samples.get(C.SUBJECT) or []
        for s in reversed(ring):
            if s.valid and s.has_primary():
                return (now - s.at) <= config.POWERTRAIN_NO_DATA_S
        return False

    # ------------------------------------------------------------------
    # What an engine monitor may see
    # ------------------------------------------------------------------
    def _baselines(self) -> tuple:
        """This vehicle's own history, from insights.py. Never fatal."""
        means: Dict[str, float] = {}
        days: Dict[str, int] = {}
        try:
            for key in _BASELINE_KEYS:
                mean, n_days, _seeded = insights.baseline(key)
                if mean is not None:
                    means[key] = mean
                    days[key] = n_days
        except Exception as e:
            print(f"[powertrain_diag] baselines unavailable: "
                  f"{type(e).__name__}: {e}", flush=True)
        return means, days

    def _build_input(self, d: M.MonitorDefinition, subject: Optional[str],
                     now: float, ctx: dict, system_healthy: bool) -> M.MonitorInput:
        means, days = self._baselines()
        link = dict(self._context.get("link") or {})
        link.setdefault("source", None)
        return M.MonitorInput(
            subject=subject, now=now,
            samples=list(self._samples.get(subject or C.SUBJECT) or []),
            moving=bool(ctx.get("moving")), speed_mph=ctx.get("speed_mph"),
            system_healthy=system_healthy,
            epoch_started_at=self._epoch_started_at(subject),
            drive_cycle_id=self.cycles.cycle_id,
            active_monitor_ids=self._active_monitor_ids(subject),
            sample_max_age_s=self.SAMPLE_MAX_AGE_S,
            no_subject_reason=self.NO_SUBJECT_REASON,
            system_unhealthy_reason=self.SYSTEM_UNHEALTHY_REASON,
            baselines=means, baseline_days=days,
            start_events=list(self._state.get("meta", {}).get("start_events") or []),
            dtc=dict(self._context.get("dtc") or {}),
            link=link)

    # ------------------------------------------------------------------
    # Freeze frames
    # ------------------------------------------------------------------
    def _freeze_evidence(self, issue: dict, d: M.MonitorDefinition,
                         out: M.Outcome, now: float, ctx: dict) -> dict:
        """What we were looking at when we decided, in this engine's units.

        Assembled from the monitor's own detail plus the current channel set, so
        a coolant finding carries the road speed and load that make it readable
        three weeks later — which is the difference between a freeze frame and a
        number.
        """
        ring = [s for s in (self._samples.get(C.SUBJECT) or []) if s.valid]
        latest = ring[-1] if ring else None
        frame: Dict[str, object] = {
            "engine_running": bool(latest.engine_running) if latest else False,
            "sample_count": len(ring),
            "first_sample_at": ring[0].at if ring else None,
            "last_sample_at": latest.at if latest else None,
            "vehicle_speed": ctx.get("speed_mph"),
            "data_quality": "good" if ring else "none",
        }
        if latest is not None:
            for channel in _CHANNELS:
                if channel in latest.values:
                    frame.setdefault(channel, latest.values[channel])
        # The monitor's own measurements win: they are what the decision was
        # actually made on, and the channel set is context around them.
        for key, value in (out.detail or {}).items():
            frame[key] = value
        return frame


# ---------------------------------------------------------------------------
# The one engine
# ---------------------------------------------------------------------------

_engine: Optional[PowertrainDiagnosticEngine] = None


def engine() -> PowertrainDiagnosticEngine:
    global _engine
    if _engine is None:
        _engine = PowertrainDiagnosticEngine()
    return _engine


def reset_engine(load: bool = True) -> PowertrainDiagnosticEngine:
    """Rebuild from disk. Used by the restart tests, and by nothing else."""
    global _engine
    _engine = PowertrainDiagnosticEngine(load=load)
    return _engine


def observe(snapshot: dict, **kw) -> dict:
    return engine().observe(snapshot, **kw)


def active_issues() -> List[dict]:
    return engine().issues(ACTIVE)


def state() -> dict:
    return engine().state()
