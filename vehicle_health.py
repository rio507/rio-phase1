"""vehicle_health.py — the Voice Context API. What RIO knows about the car.

Where this sits
---------------
                 tires.py            telemetry.py         (later: OBD-II,
              TireHealthProvider  TelemetryProvider(s)      RIO Connect, ECU)
                     │                    │                       │
                     └──────────┬─────────┴───────────────────────┘
                                ▼
                          THIS FILE — normalize
                    one vocabulary, one severity ladder,
                    one issue list, worst first
                                │
                 ┌──────────────┴──────────────┐
                 ▼                             ▼
        llm_interface.py                vehicle_health_policy.py
     compact summary every turn        deterministic: is this worth
     full structure when the router     interrupting for, and when
     says it is a health question              │
                 │                             ▼
                 ▼                    static/rio_health.js
          RIO answers                  the existing arbiter,
          conversationally             VEHICLE_HEALTH priority

This is one more context provider alongside navigation, vision, conversation and
driver memory. It answers five questions and nothing else: how is the car, what
is wrong with it, how bad is that, how would you say it in English, and what
should the driver do about it.

Not tires
---------
The word "tire" appears in one class below and nowhere else. Everything above
`TireSource` is domain-agnostic: a source has `available()`, `state()` and
`issues()`, and adding battery chemistry, oil pressure, coolant, fuel pressure,
OBD-II or a real ECU is a new class and a `register_source()` call. The LLM
interface does not change, the policy does not change, the prompt does not
change, the client does not change. `EngineSource` already proves it — the
Holley channels arrive through the same three methods the tires do.

What must never come out of here
--------------------------------
UI state. Not a CSS class, not a colour, not a glyph, not a poll interval, not
whether a banner is showing. The dashboard's job is to render what tires.py and
telemetry.py already worded; this file's job is to describe the CAR. Those are
different questions and the normalized structure below answers only the second.

Truthfulness
------------
Every issue carries an explicit `observation_window` naming exactly how far back
the evidence goes, and the top-level context carries `history_depth` saying what
is known at all. This is not decoration. RIO is a language model with a strong
prior for narrative — "your rear-left has been losing pressure for weeks" is a
much more natural sentence than the truth, which is that the sensor reports a
24-hour trend and nothing older. The window travels with the data so the prompt
can forbid inventing durations and the model can see the actual bound rather
than guessing at one.
"""
from __future__ import annotations

from typing import List, Optional

import config
import telemetry
import tires
import vehicle_health_policy as policy

# The severity vocabulary is the policy's, imported rather than restated. There
# is exactly one ladder in this feature and the module that acts on it owns it.
INFORMATIONAL = policy.INFORMATIONAL
ADVISORY = policy.ADVISORY
WARNING = policy.WARNING
CRITICAL = policy.CRITICAL
SEVERITY_RANK = policy.SEVERITY_RANK

# Overall status, in the spec's own lowercase vocabulary. `unknown` is not a
# fourth severity — it is the honest answer when nothing is reporting, and it is
# deliberately not "normal": a car with no sensors on it is not a healthy car,
# it is a car nobody can see.
NORMAL = "normal"
UNKNOWN = "unknown"

_STATUS_BY_RANK = {0: NORMAL, 1: INFORMATIONAL, 2: ADVISORY, 3: WARNING,
                   4: CRITICAL}

# Windows, written as a person says them, because they are read out. A window is
# a claim about the DATA, so each one is tied to the thing that produced it:
# TIRE_TREND is 24 h because TireReading.trend_psi_24h is a 24-hour delta, and
# ENGINE_TREND is the telemetry ring, which is seconds long.
W_NOW = "this moment only"
W_TIRE_TREND = "the last 24 hours"
W_ENGINE_TREND = f"the last {int(config.TELEMETRY_TREND_WINDOW_S)} seconds"
W_SESSION = "since the engine was started"


# ---------------------------------------------------------------------------
# What a source returns
# ---------------------------------------------------------------------------

class HealthSource:
    """Interface. Implement these three and RIO can talk about your subsystem.

    `available()` is asked separately from the other two for the reason it is in
    tires.py and telemetry.py: "this car has no coolant sensor" and "this car
    has a coolant sensor and it is fine" are different answers and RIO says
    different things about them.
    """

    domain = "base"
    label = "Subsystem"

    def refresh(self) -> None:
        """Drop whatever was read last. Called once at the top of every public
        entry point below, which is what makes a context build ONE read per
        source: available(), state() and issues() all want the same snapshot and
        must all see the same one, or the issue list can disagree with the state
        it was derived from. A time-based cache would do it too and would also
        make the whole layer untestable at machine speed."""

    def available(self) -> bool:
        raise NotImplementedError

    def state(self) -> dict:
        """The normalized picture of this subsystem, for the full context. Plain
        numbers and plain words — no formatting, no units baked into strings,
        nothing the dashboard invented."""
        raise NotImplementedError

    def issues(self, ctx: dict) -> List[dict]:
        """Everything currently wrong, worst first is not required — the caller
        sorts. `ctx` carries the facts a source cannot know on its own; today
        that is whether the car is moving."""
        raise NotImplementedError


_SOURCES: List[HealthSource] = []


def register_source(source: HealthSource) -> None:
    """Add a subsystem. Called at import time below; a plugin would call it too."""
    _SOURCES.append(source)


def sources() -> List[HealthSource]:
    return list(_SOURCES)


# ---------------------------------------------------------------------------
# Issue construction
# ---------------------------------------------------------------------------

def make_issue(key: str, domain: str, type: str, severity: str, message: str,
               observation_window: str, suggested_action: str = "",
               location: str = "", magnitude: float = 0.0,
               value: Optional[float] = None, unit: str = "",
               spoken_fallback: str = "", evidence: Optional[dict] = None) -> dict:
    """One fault, in the shape both consumers expect.

    `key` is identity and it has to be STABLE: the announcement policy's whole
    "say it once" rests on the same fault producing the same key on every tick,
    so it is built from what the fault IS (domain, location, kind) and never
    from a value or a timestamp.

    `magnitude` is how bad, in the issue's own units, higher worse. It is only
    ever compared against itself across ticks — it is the worsening test, not a
    ranking between issues, and comparing PSI against °F would be meaningless.
    """
    return {
        "key": key,
        "domain": domain,
        "type": type,
        "severity": severity,
        "severity_rank": SEVERITY_RANK.get(severity, 0),
        "message": message,
        "observation_window": observation_window,
        "suggested_action": suggested_action,
        "location": location,
        "magnitude": round(float(magnitude), 3),
        "value": (round(float(value), 2) if value is not None else None),
        "unit": unit,
        "spoken_fallback": spoken_fallback,
        "evidence": evidence or {},
    }


# ---------------------------------------------------------------------------
# Tires
# ---------------------------------------------------------------------------

_CORNER_KEY = {"FL": "front_left", "FR": "front_right",
               "RL": "rear_left", "RR": "rear_right"}
_CORNER_SPOKEN = {"FL": "front left", "FR": "front right",
                  "RL": "rear left", "RR": "rear right"}

# How long a monitor's evidence reaches back, said the way a person says it.
# Derived from the evidence the monitor actually used rather than written as a
# constant per issue type: the whole point of observation_window is that RIO
# cannot claim more history than the data has, and a hardcoded phrase would be a
# claim about the data rather than a report of it.

def _window_phrase(seconds: Optional[float]) -> str:
    if not seconds or seconds <= 0:
        return W_NOW
    if seconds < 120:
        return f"the last {int(round(seconds / 10.0)) * 10} seconds"
    if seconds < 5400:
        return f"the last {int(round(seconds / 60.0))} minutes"
    if seconds < 86400:
        return f"the last {int(round(seconds / 3600.0))} hours"
    return f"the last {int(round(seconds / 86400.0))} days"


# What each monitor's finding is called in the conversation layer, and which
# line in vehicle_health_policy.LINE it would use if it were ever cleared to
# speak. `type` is the announcement-path key; the driver-facing noun phrase
# comes from the code catalogue, which owns the wording.
_MONITOR_TYPE = {
    "tire.low_pressure": "low_pressure",
    "tire.critical_low_pressure": "critical_low_pressure",
    "tire.slow_leak": "possible_slow_leak",
    "tire.asymmetric_loss": "asymmetric_pressure_loss",
    "tpms.sensor_connectivity": "tire_sensor_quiet",
    "tpms.sensor_plausibility": "tire_sensor_implausible",
    "tire.sensor_loss_during_decline": "tire_sensor_lost_driving",
    "tire.inflation_event": "inflation_event",
    "tpms.receiver_health": "tire_sensors_all_silent",
}


class TireSource(HealthSource):
    """The four corners, as the DIAGNOSTIC ENGINE understands them.

    What changed, and why it matters
    --------------------------------
    This class used to read tires.snapshot() and turn each corner's
    instantaneous state into an issue. That was wrong in a way that only showed
    up once anyone looked closely: a single reading became a `warning`, and one
    missed poll became a `critical`. A tire that read low on one wake-up frame
    was, as far as RIO was concerned, a tire with a problem.

    Now the issues come from tire_diag, where a finding has to survive its
    monitor's confirmation criteria before it is an Issue at all. tires.py's
    classification is still exactly what the dashboard renders — it is the
    instantaneous view and it is correct as that — but it is no longer what RIO
    reasons about.

    Readiness is part of the answer
    -------------------------------
    A monitor that has not run is not a monitor that passed. This class reports
    that distinction rather than smoothing it over, because the alternative is
    RIO saying "everything looks good" about four tires she has not yet been
    able to evaluate. That is the single most damaging sentence this system
    could produce, and it is the one a naive implementation produces by default.
    """

    domain = "tires"
    label = "Tires"

    def __init__(self):
        self._snap = None

    def refresh(self) -> None:
        self._snap = None

    def _read(self) -> dict:
        # tires.snapshot() is a pure read of the provider -- no history, no
        # insight engine -- so the only reason to hold it is to keep one context
        # build from asking twice. It is NOT what produces issues any more; it
        # is the current-value view that sits alongside them.
        if self._snap is None:
            self._snap = tires.snapshot()
        return self._snap

    def available(self) -> bool:
        try:
            return bool(self._read().get("available"))
        except Exception:
            return False

    def state(self) -> dict:
        snap = self._read()
        out = {}
        for t in snap.get("tires", []):
            raw = t.get("raw") or {}
            out[_CORNER_KEY.get(t["corner"], t["corner"])] = {
                "pressure": raw.get("pressure_psi"),
                "target_pressure": raw.get("target_psi"),
                "temperature": raw.get("temp_f"),
                "status": _tire_status_word(t["state"]),
                "trend": _tire_trend_word(raw.get("trend_psi_24h")),
                "change_over_24h": raw.get("trend_psi_24h"),
                "sensor": "reporting" if raw.get("connected") else "not reporting",
            }
        return {"pressure_unit": "psi", "temperature_unit": "fahrenheit",
                "corners": out,
                # The readiness view, in the conversation context. This is what
                # lets RIO say "I do not have enough comparable readings yet to
                # evaluate a slow leak, but the current pressure is not
                # critically low" -- a sentence that is impossible to say
                # honestly without both halves of it.
                "monitors": _monitor_readiness()}

    def issues(self, ctx: dict) -> List[dict]:
        try:
            from tire_diag import engine as diag
            active = diag.active_issues()
        except Exception as e:
            print(f"[vehicle_health] diagnostic engine unavailable: "
                  f"{type(e).__name__}: {e}", flush=True)
            return []

        out = [_issue_from_diagnostic(i) for i in active]

        # Nothing confirmed. Before saying so, check that anything has actually
        # been EVALUATED -- "no confirmed faults" and "no monitor has managed to
        # run" are different answers and only one of them is reassuring.
        if not out:
            unevaluated = _unevaluated_issue()
            if unevaluated is not None:
                out.append(unevaluated)
        return out


def _tire_status_word(state: str) -> str:
    """tires.py's instantaneous state -> the spec's lowercase status vocabulary.

    This describes the CURRENT READING of a corner, which is a different claim
    from whether a diagnostic monitor has confirmed anything about it. Both are
    in the context, side by side and clearly labelled, because "the pressure is
    low right now" and "a low-pressure fault is confirmed" are both true things
    a driver might be asking about and they are not the same thing.
    """
    return {"NORMAL": "normal", "WARNING": "warning", "CRITICAL": "critical",
            "BATTERY_LOW": "normal", "STALE": "unknown",
            "DISCONNECTED": "unknown", "NO_DATA": "unknown"}.get(state, "unknown")


def _tire_trend_word(change: Optional[float]) -> Optional[str]:
    """A direction only where the data supports one.

    None, not "stable", when there is no trend to report: a sensor that has been
    powered up for four minutes has no 24-hour history, and telling RIO the
    pressure is "stable" over a window that does not exist is precisely the
    fabrication the observation_window fields exist to prevent.
    """
    if change is None:
        return None
    if change <= config.TIRE_TREND_LEAK_PSI_24H:
        return "dropping"
    if change <= -0.5:
        return "drifting down"
    if change >= 0.5:
        return "rising"
    return "steady"


def _monitor_readiness() -> dict:
    """Status and last_result per monitor, summarised for the conversation.

    Deliberately not the full per-corner matrix: thirty-three rows of monitor
    state in a conversation turn is a cost paid on a question nobody asked. What
    RIO needs is which monitors can currently judge, which are still gathering,
    and why the ones that cannot, cannot.
    """
    try:
        from tire_diag import engine as diag
        from tire_diag import monitors as M
    except Exception:
        return {}

    rows = diag.engine().monitor_view()
    by_monitor: dict = {}
    for r in rows:
        entry = by_monitor.setdefault(r["monitor_id"], {
            "statuses": {}, "results": {}, "reasons": []})
        entry["statuses"][r["status"]] = entry["statuses"].get(r["status"], 0) + 1
        key = r["last_result"] or "never_run"
        entry["results"][key] = entry["results"].get(key, 0) + 1
        if r["status"] in M.NO_VERDICT and r["status_reason"]:
            if r["status_reason"] not in entry["reasons"]:
                entry["reasons"].append(r["status_reason"])

    out = {}
    for monitor_id, entry in by_monitor.items():
        # The worst status wins the summary: one corner that cannot be evaluated
        # is the thing worth saying, not the three that could.
        order = (M.DATA_UNAVAILABLE, M.INHIBITED, M.NOT_READY, M.RUNNING,
                 M.NOT_SUPPORTED, M.READY)
        status = next((s for s in order if s in entry["statuses"]), M.NOT_READY)
        out[monitor_id] = {
            "status": status,
            "evaluated_corners": entry["results"].get("PASSED", 0)
                                 + entry["results"].get("FAILED_PENDING", 0)
                                 + entry["results"].get("FAILED_CONFIRMED", 0),
            "never_run_corners": entry["results"].get("never_run", 0),
            "why_not": entry["reasons"][:2],
        }
    return out


# The monitors a driver is actually asking about when they ask whether their
# tires are okay. If NEITHER of these has judged anything, "everything looks
# normal" is not an answer RIO is entitled to give, however many other monitors
# have run — a connectivity monitor reporting that all four sensors are talking
# says nothing whatsoever about the pressure in the tires.
_CORE_MONITORS = ("tire.low_pressure", "tire.critical_low_pressure")


def _unevaluated_issue() -> Optional[dict]:
    """"I have not been able to look at this yet", as an issue.

    The spec's rule, and the one this whole readiness apparatus exists to
    enforce: do not describe a system as healthy merely because its monitor has
    not run. A car whose pressure monitors are NOT_READY is a car nobody has
    evaluated, and saying "all four are where they should be" about it is the
    single most damaging sentence this feature could produce.

    Informational, so it never speaks. It exists to stop the one-line summary
    that goes into EVERY conversation turn from claiming an all-clear nobody
    established.
    """
    try:
        from tire_diag import engine as diag
    except Exception:
        return None

    rows = diag.engine().monitor_view()
    if not rows:
        return None

    core = [r for r in rows if r["monitor_id"] in _CORE_MONITORS]
    if any(r["last_result"] is not None for r in core):
        return None

    reasons = []
    for r in core:
        if r["status_reason"] and r["status_reason"] not in reasons:
            reasons.append(r["status_reason"])
    judged = sorted({r["monitor_id"] for r in rows
                     if r["last_result"] is not None})
    return make_issue(
        key="tires:monitors:not_evaluated",
        domain="tires", type="tire_monitors_not_ready",
        severity=INFORMATIONAL,
        message="Tire pressure has not been evaluated yet — "
                + (reasons[0] if reasons else "not enough trusted samples")
                + ". This is not the same as the tires being fine.",
        observation_window=W_NOW,
        suggested_action="",
        spoken_fallback="I haven't been able to evaluate your tires yet.",
        evidence={"reasons": reasons[:3],
                  "monitors_with_a_verdict": judged,
                  "monitors_total": len(rows)})


def _issue_from_diagnostic(i: dict) -> dict:
    """One confirmed diagnostic Issue, in the conversation layer's vocabulary.

    The diagnostic code never crosses this boundary as a code. RIO says "your
    rear-left tire may have a slow leak"; RIO-TIRE-POSSIBLE-LEAK-RL rides along
    in a field the service view reads and the prompt never shows.
    """
    from tire_diag import codes as C

    code = C.get(i.get("code") or "")
    detail = i.get("detail") or {}
    corner = i.get("corner")
    spoken_loc = _CORNER_SPOKEN.get(corner or "", "")
    name = {"FL": "Front Left", "FR": "Front Right",
            "RL": "Rear Left", "RR": "Rear Right"}.get(corner or "", "")

    window_s = detail.get("window_s") or detail.get("span_s")
    if i.get("monitor_id") == "tpms.sensor_plausibility":
        window_s = detail.get("window_s") or config.TIRE_DIAG_IMPLAUSIBLE_WINDOW_S

    severity = i.get("severity") or (code.default_severity if code else WARNING)
    message = _diagnostic_message(i, code, detail, name)

    # Speech eligibility, decided here and enforced by the policy. Two gates:
    # the per-code flag (every one of which is False in shadow mode) and the
    # global shadow switch. The urgent fast path is the documented exception and
    # is separately gated -- it is the only thing that can speak while the
    # monitors are still being evaluated.
    fast = bool(i.get("urgent")) and C.fast_path_eligible(i.get("code") or "")
    allowed = fast or (not config.TIRE_DIAG_SHADOW_MODE
                       and C.speech_eligible(i.get("code") or ""))

    issue = make_issue(
        key=i["issue_id"],
        domain="tires",
        type=_MONITOR_TYPE.get(i.get("monitor_id"), "tire_fault"),
        severity=severity,
        message=message,
        observation_window=_window_phrase(window_s),
        suggested_action=(code.suggested_action if code else ""),
        location=spoken_loc,
        magnitude=_diagnostic_magnitude(detail),
        value=detail.get("pressure_psi") or detail.get("to_psi"),
        unit="psi" if ("pressure_psi" in detail or "to_psi" in detail) else "",
        spoken_fallback=message,
        evidence={
            "confirmed_at": i.get("confirmed_at"),
            "confidence": i.get("confidence"),
            "monitor": i.get("monitor_id"),
            "monitor_runs_failed": i.get("fail_runs"),
            "drive_cycles_failed": len(i.get("fail_cycles") or []),
            "recurrence_count": (i.get("recurrence") or {}).get("count", 0),
            "measurement": {k: v for k, v in detail.items()
                            if k not in ("reasons",)},
        })
    # Fields the announcement path needs and the LLM never sees -- _issue_view()
    # in context() drops everything except the documented set.
    issue["issue_id"] = i["issue_id"]
    issue["code"] = i.get("code")
    # Consecutive passing monitor runs. The announcement policy uses this to
    # decline to REMIND about a fault that is currently recovering — see
    # vehicle_health_policy.R_HEALING. It has to travel on the issue because the
    # policy imports nothing and cannot ask the engine anything.
    issue["healing_runs"] = int((i.get("healing_progress") or {})
                                .get("passing_runs", 0) or 0)
    issue["announce_allowed"] = allowed
    issue["fast_path"] = fast
    issue["audio"] = (_FAST_PATH_CLIP.get(i.get("monitor_id"), "tts")
                      if fast else "tts")
    issue["lifecycle"] = i.get("lifecycle")
    return issue


# Pre-rendered clips for the urgent fast path, by monitor. Same mechanism the
# headway red tier uses and for the same reason: an ElevenLabs round trip is
# 300-800 ms, and the entire argument for a fast path is that waiting is what it
# exists to avoid. Rendered by tools/render_alerts.py.
_FAST_PATH_CLIP = {
    "tire.critical_low_pressure": "tire_critical",
    "tire.sensor_loss_during_decline": "tire_sensor_lost",
}


def _diagnostic_message(i: dict, code, detail: dict, name: str) -> str:
    """One sentence of plain English about a confirmed diagnostic finding.

    Deterministic, and about MEANING rather than numbers -- the figures are in
    `evidence` for RIO to reach for when they help. It also carries the honest
    hedge: the code's driver_term says "a possible slow leak", not "a slow leak",
    because that is what the evidence supports and the prompt is not allowed to
    upgrade it.
    """
    term = code.driver_term if code else "a fault"
    where = f"{name} " if name else ""
    monitor = i.get("monitor_id")

    if monitor == "tire.slow_leak":
        change = abs(detail.get("change_psi") or 0.0)
        peers = detail.get("peer_change_psi")
        peer_txt = (f" while the other corners moved {peers:+.1f} PSI"
                    if peers is not None else "")
        return (f"{where}shows {term}: down {change:.1f} PSI across thermally "
                f"comparable readings{peer_txt}.")
    if monitor == "tire.asymmetric_loss":
        return (f"{where}is losing pressure faster than the "
                f"{detail.get('peer_corner', 'other')} corner on the same axle "
                f"— a {abs(detail.get('gap_psi') or 0.0):.1f} PSI difference.")
    if monitor == "tire.critical_low_pressure":
        psi = detail.get("pressure_psi")
        falling = " and still falling" if detail.get("falling") else ""
        return (f"{where}is at {psi:.0f} PSI{falling} — below the level it is "
                f"safe to drive on.")
    if monitor == "tire.low_pressure":
        psi = detail.get("pressure_psi")
        thr = detail.get("threshold_psi")
        return (f"{where}is at {psi:.1f} PSI, under the {thr:.1f} PSI it should "
                f"not go below.")
    if monitor == "tire.sensor_loss_during_decline":
        return (f"{where}was losing pressure and its sensor has stopped "
                f"reporting — that corner cannot be watched any more.")
    if monitor == "tpms.sensor_connectivity":
        return f"The {name.lower()} sensor has {term}."
    if monitor == "tpms.sensor_plausibility":
        n = detail.get("implausible_count")
        return (f"The {name.lower()} sensor is {term} — {n} implausible reports "
                f"in the window.")
    if monitor == "tpms.receiver_health":
        return "Nothing is reporting from any tire sensor — that is the receiver."
    if monitor == "tire.inflation_event":
        return (f"{where}was inflated by {detail.get('step_psi', 0):.1f} PSI.")
    return f"{where}{term}."


def _diagnostic_magnitude(detail: dict) -> float:
    """How bad, in the issue's own units. Compared only against itself."""
    for key in ("margin_psi", "excess_vs_peers_psi", "gap_psi", "silent_for_s",
                "implausible_count", "step_psi"):
        v = detail.get(key)
        if v is not None:
            return abs(float(v))
    return 0.0


# ---------------------------------------------------------------------------
# Engine and electrical
# ---------------------------------------------------------------------------
# Proof that the layer is not about tires: this source is generic over
# telemetry.ALL_SENSORS and gains a channel whenever that table does. Battery,
# oil pressure, coolant and fuel pressure — the spec's "future" list — arrive
# through it today because the Holley mock already reports them.

# Channels a driver would want interpreted rather than recited, and how to say
# what they are for. Anything not listed still produces an issue when it leaves
# its band; it just gets a plainer sentence.
_CHANNEL_MEANING = {
    "battery_voltage": "the charging system",
    "oil_pressure": "oil pressure",
    "oil_temp": "oil temperature",
    "coolant_temp": "coolant temperature",
    "fuel_pressure": "fuel pressure",
    "intake_air_temp": "intake air temperature",
    "afr_wideband": "the air-fuel mixture",
}

_CHANNEL_ACTION = {
    "battery_voltage": "Have the alternator and battery checked before the next "
                       "cold start.",
    "oil_pressure": "Stop and shut the engine off — running without oil pressure "
                    "damages it in seconds.",
    "coolant_temp": "Pull over and let it cool before going any further.",
    "fuel_pressure": "Have the pump and filter looked at.",
}


class EngineSource(HealthSource):
    """Everything telemetry.py reports, minus the tire channels.

    The tire channels are excluded because TireSource already describes those
    corners in far more detail, and an issue list that says both "rear left
    pressure is low" and "Rear Left Pressure is WARNING" is a list that has
    counted one fault twice.
    """

    domain = "engine"
    label = "Engine & Electrical"

    def __init__(self):
        self._snap = None

    def refresh(self) -> None:
        self._snap = None

    def _read(self) -> dict:
        if self._snap is None:
            # record=False — see telemetry.snapshot's docstring. A conversation
            # turn must not add samples to the trend window.
            self._snap = telemetry.snapshot(record=False)
        return self._snap

    def available(self) -> bool:
        try:
            return bool(self._read().get("available"))
        except Exception:
            return False

    def state(self) -> dict:
        snap = self._read()
        channels = {}
        for row in snap.get("rows", []):
            if row["id"].startswith("tire_"):
                continue
            channels[row["id"]] = {
                "label": row["label"],
                "value": row["value"],
                "unit": row["units"],
                "status": row["status"].lower(),
                "trend": row["trend"],
            }
        # runtime is None rather than 0 when it is not known. The runtime clock
        # is advanced by the dashboard's poll, and this is a read that does not
        # advance it (telemetry.snapshot(record=False)); reporting 0.0 on a
        # running engine would tell RIO it had just been started, which is a
        # fact about the poll and not about the car.
        runtime = snap.get("runtime_s")
        return {
            "engine_running": snap.get("engine_running"),
            "running_for_seconds": runtime if runtime else None,
            "channels": channels,
        }

    def issues(self, ctx: dict) -> List[dict]:
        snap = self._read()
        out: List[dict] = []
        for row in snap.get("rows", []):
            if row["id"].startswith("tire_"):
                continue
            status = row["status"]
            if status not in ("WARNING", "CRITICAL"):
                continue
            sev = CRITICAL if status == "CRITICAL" else WARNING
            what = _CHANNEL_MEANING.get(row["id"], row["label"].lower())
            direction = "high" if _over_high(row) else "low"
            # `trend` is the only claim here about anything but the present
            # instant, so it is the only thing that widens the window.
            window = W_ENGINE_TREND if row["trend"] in ("up", "down", "abnormal") \
                else W_NOW
            message = (f"{row['label']} is {direction} at "
                       f"{row['value_text']}{_unit_suffix(row['units'])}"
                       + (f", and still moving the wrong way"
                          if row["trend"] == "abnormal" else "")
                       + f". In plain terms: {what} is outside its normal range.")
            out.append(make_issue(
                key=f"engine:{row['id']}:{direction}",
                domain=self.domain,
                type=f"{row['id']}_{'critical' if sev == CRITICAL else 'warning'}",
                severity=sev,
                message=message,
                observation_window=window,
                suggested_action=_CHANNEL_ACTION.get(row["id"], ""),
                location="",
                magnitude=_channel_magnitude(row),
                value=row["value"],
                unit=row["units"],
                spoken_fallback=f"{row['label']} is outside its normal range.",
                evidence={"channel": row["id"], "value": row["value"],
                          "unit": row["units"], "trend": row["trend"],
                          "detail": row.get("detail", "")}))
        return out


def _unit_suffix(unit: str) -> str:
    if not unit:
        return ""
    return unit if unit.startswith("°") or unit.startswith(":") else " " + unit


def _over_high(row: dict) -> bool:
    band = telemetry.band(row["id"])
    v = row.get("value")
    if v is None:
        return False
    for edge in ("crit_high", "warn_high"):
        if band.get(edge) is not None and v >= band[edge]:
            return True
    return False


def _channel_magnitude(row: dict) -> float:
    """How far past the nearest exceeded limit, in the channel's own units."""
    band = telemetry.band(row["id"])
    v = row.get("value")
    if v is None:
        return 0.0
    worst = 0.0
    for edge in ("crit_high", "warn_high"):
        if band.get(edge) is not None and v >= band[edge]:
            worst = max(worst, v - band[edge])
    for edge in ("crit_low", "warn_low"):
        if band.get(edge) is not None and v <= band[edge]:
            worst = max(worst, band[edge] - v)
    return worst


# ---------------------------------------------------------------------------
# Diagnostic trouble codes — what the VEHICLE says
# ---------------------------------------------------------------------------
# The proof that this layer was the right seam: a whole new subsystem arrives as
# a class with three methods and a register_source() call. Not one line of
# llm_interface.py, vehicle_health_policy.py, rio_prompts.py or the browser
# changes, and RIO can talk about diagnostic trouble codes.

class DTCSource(HealthSource):
    """The vehicle's own diagnostic trouble codes.

    Kept as its own domain rather than folded into `engine`, because the split
    that matters in this whole feature is WHO IS MAKING THE CLAIM. A P-code is
    the vehicle saying something; everything in EngineSource and
    PowertrainSource is RIO saying something. §17.8 puts them under different
    headings and this is the same line drawn one layer earlier, where the
    conversation context is built — so the prompt cannot merge them by accident.
    """

    domain = "diagnostics"
    label = "Vehicle Diagnostics"

    def __init__(self):
        self._section = None

    def refresh(self) -> None:
        self._section = None

    def _read(self) -> dict:
        if self._section is None:
            from vehicle.dtc import service as dtc
            self._section = dtc.service().section()
        return self._section

    def available(self) -> bool:
        try:
            sec = self._read()
            # A scan has completed AND the vehicle answered. Before that, RIO
            # genuinely does not know whether this car has codes, and the
            # generic unavailable-issue in issues() says exactly that rather
            # than letting an empty list read as an all-clear.
            return bool(sec.get("scan_count")) and bool(sec.get("ecu_responding"))
        except Exception:
            return False

    def state(self) -> dict:
        sec = self._read()
        cards = [c for g in sec.get("groups", []) for c in g["cards"]]
        return {
            "malfunction_indicator_lamp": ("on" if sec.get("mil_commanded_on")
                                           else "off"),
            "codes_reported": len(cards),
            "scan_count": sec.get("scan_count"),
            "last_scan_at": sec.get("last_scan_at"),
            "services_supported": sec.get("services_supported") or {},
            "codes": [{
                "code": c["code"],
                "description": c["description"],
                "status": c["status_label"],
                "category": c["dtc_category"],
                "system": c["system_label"],
                "detected_before_warning_light": c["early_detection"],
                "first_seen_at": c["first_seen_at"],
                "times_seen_pending": c["pending_scan_count"],
                "drives_observed": c["drive_cycle_count_observed"],
                "freeze_frame_available": c["freeze_frame_available"],
                "what_this_means": c["what_this_means"],
                "possible_causes": c["possible_causes"],
                "cause_confirmed": c["confirmed_cause"] is not None,
            } for c in cards],
        }

    def issues(self, ctx: dict) -> List[dict]:
        from vehicle.dtc import catalog as DC
        from vehicle.dtc import lifecycle as DL
        from vehicle.dtc import service as dtc

        out: List[dict] = []
        svc = dtc.service()
        for rec in svc.registry.active_codes():
            card = svc.card(rec)
            severity = DC.health_severity(rec.get("severity", ""))
            out.append(_issue_from_code(card, severity))
        return out


def _issue_from_code(card: dict, severity: str) -> dict:
    """One reported code, in the conversation layer's vocabulary.

    The wording is the careful part. A DTC names a CONDITION and the temptation
    is to phrase it as a diagnosis, because a diagnosis is a better sentence.
    §16.3 forbids it by example: "RIO detected a pending engine fault before the
    check-engine light turned on" is approved and "this component has failed" is
    not, and the difference is the whole credibility of the feature.
    """
    code = card["code"]
    desc = card["description"]
    early = card.get("early_detection")
    status = card.get("status_label")

    if early and card["lifecycle"] == "pending_first_seen":
        message = (f"The engine computer has flagged {code} — {desc} — and the "
                   f"check-engine light is still off. It has observed the "
                   f"condition and has not confirmed that it is persistent.")
        spoken = ("I've picked up a pending engine code before the "
                  "check-engine light came on.")
    elif card["lifecycle"] == "pending_repeated":
        message = (f"{code} ({desc}) has appeared again on a later scan. Still "
                   f"not a confirmed diagnosis, but the recurrence makes it "
                   f"worth looking at.")
        spoken = "That pending engine code has shown up again."
    elif card["lifecycle"] == "permanent_if_applicable":
        message = (f"{code} ({desc}) is stored as a permanent code. It cannot "
                   f"be cleared by a scan tool and will clear itself only after "
                   f"the vehicle's own monitors pass.")
        spoken = "There's a permanent code stored in the engine computer."
    else:
        lamp = ("with the check-engine light on" if card.get("mil_commanded_on")
                else "with the check-engine light off")
        message = (f"The engine computer has confirmed {code} — {desc} — "
                   f"{lamp}.")
        spoken = "The engine computer has confirmed a fault code."

    causes = card.get("possible_causes") or []
    if causes:
        message += (" Possible causes include " + ", ".join(causes[:3]).lower()
                    + " — none of them confirmed.")

    return make_issue(
        key=f"dtc:{code}",
        domain="diagnostics",
        type=f"dtc_{card['lifecycle']}",
        severity=severity,
        message=message,
        # The window a code supports is exactly when it was first and last
        # seen. Nothing about a DTC licenses a claim about weeks.
        observation_window=_window_phrase(
            (card.get("last_seen_at") or 0) - (card.get("first_seen_at") or 0)),
        suggested_action=("Have the code read and investigated at a workshop."
                          if severity in (WARNING, CRITICAL) else ""),
        location="",
        magnitude=float(card.get("pending_scan_count") or 0)
        + float(card.get("recurrence_count") or 0),
        spoken_fallback=spoken,
        evidence={
            "code": code,
            "status": status,
            "detected_before_warning_light": early,
            "check_engine_light": ("on" if card.get("mil_commanded_on")
                                   else "off"),
            "times_seen_pending": card.get("pending_scan_count"),
            "drives_observed": card.get("drive_cycle_count_observed"),
            "freeze_frame_available": card.get("freeze_frame_available"),
            "possible_causes": causes,
            "cause_confirmed": card.get("confirmed_cause") is not None,
            "reported_by": "vehicle ECU",
        })


# ---------------------------------------------------------------------------
# Powertrain monitors — what RIO says about the engine
# ---------------------------------------------------------------------------

class PowertrainSource(HealthSource):
    """Confirmed findings from the engine monitors.

    The same relationship TireSource has to tire_diag: the panel shows the
    instantaneous reading, and this shows what a monitor has actually confirmed
    across enough evidence to be worth saying out loud.
    """

    domain = "powertrain"
    label = "Engine Diagnostics"

    def available(self) -> bool:
        try:
            from powertrain_diag import engine as diag
            return any(m["last_result"] is not None
                       for m in diag.engine().monitor_view())
        except Exception:
            return False

    def state(self) -> dict:
        try:
            from powertrain_diag import engine as diag
            from diag import monitors as M
        except Exception:
            return {}
        rows = diag.engine().monitor_view()
        out = {}
        for r in rows:
            out[r["monitor_id"]] = {
                "status": r["status"],
                "last_result": r["last_result"],
                "why_not": (r["status_reason"] if r["status"] in M.NO_VERDICT
                            else None),
            }
        return {"monitors": out}

    def issues(self, ctx: dict) -> List[dict]:
        try:
            from powertrain_diag import codes as PC
            from powertrain_diag import engine as diag
        except Exception as e:
            print(f"[vehicle_health] powertrain engine unavailable: "
                  f"{type(e).__name__}: {e}", flush=True)
            return []

        out = []
        for i in diag.active_issues():
            if i.get("monitor_id") == "engine.new_dtc":
                # The codes reach RIO through DTCSource, in far more detail and
                # with the right provenance. This monitor exists so a reported
                # code takes part in the diagnostic lifecycle and the
                # communication ledger — surfacing it here as well would be an
                # issue list that has counted one fault twice, exactly as
                # EngineSource excludes the tire channels.
                continue
            code = PC.get(i.get("code") or "")
            detail = i.get("detail") or {}
            out.append(make_issue(
                key=i["issue_id"],
                domain=self.domain,
                type=i.get("monitor_id", "engine_fault").replace(".", "_"),
                severity=i.get("severity") or WARNING,
                message=_powertrain_message(i, code),
                observation_window=_window_phrase(
                    detail.get("window_s") or detail.get("held_s")),
                suggested_action=(code.suggested_action if code else ""),
                location="",
                magnitude=_powertrain_magnitude(detail),
                value=detail.get("coolant_temp") or detail.get("battery_voltage")
                or detail.get("ltft_b1"),
                unit=("°F" if detail.get("coolant_temp") is not None
                      else "V" if detail.get("battery_voltage") is not None
                      else "%" if detail.get("ltft_b1") is not None else ""),
                spoken_fallback=(code.driver_term if code
                                 else "something on the engine"),
                evidence={
                    "monitor": i.get("monitor_id"),
                    "confirmed_at": i.get("confirmed_at"),
                    "confidence": i.get("confidence"),
                    "monitor_runs_failed": i.get("fail_runs"),
                    "drive_cycles_failed": len(i.get("fail_cycles") or []),
                    "recurrence_count": (i.get("recurrence") or {}).get("count", 0),
                    "observed_by": "RIO",
                    "measurement": detail,
                }))
        return out


def _powertrain_message(issue: dict, code) -> str:
    """One sentence about a confirmed engine finding.

    Always says it is RIO's observation. A finding phrased like a code — "fault
    detected in the cooling system" — is a finding a reader will attribute to the
    vehicle, and RIO does not get to borrow the ECU's authority.
    """
    term = code.driver_term if code else "a fault"
    reason = issue.get("reason") or ""
    return f"RIO has observed {term}: {reason}."


def _powertrain_magnitude(detail: dict) -> float:
    """How bad, in the finding's own units. Compared only against itself."""
    for key in ("delta_f", "held_s", "decline_v", "coolant_rate_f_per_min",
                "silent_for_s"):
        v = detail.get(key)
        if v is not None:
            try:
                return abs(float(v))
            except (TypeError, ValueError):
                continue
    return 0.0


_ENGINE = EngineSource()

register_source(TireSource())
register_source(_ENGINE)
# Order matters only for readability of the unavailable-issues; the issue list
# itself is sorted worst-first in issues().
register_source(DTCSource())
register_source(PowertrainSource())


def _refresh_all() -> None:
    """One read per source per public call. See HealthSource.refresh."""
    for src in _SOURCES:
        try:
            src.refresh()
        except Exception:
            pass


# ---------------------------------------------------------------------------
# Is the car moving
# ---------------------------------------------------------------------------

def _driving() -> tuple:
    """-> (driving, why). The one cross-domain fact the sources need.

    Conservative on purpose: no telemetry means NOT driving. The only thing this
    flag can do is promote a quiet sensor to a critical announcement, and
    guessing that the car is moving when nobody knows would be RIO interrupting
    a parked driver to tell them about a sensor that went to sleep — which is
    exactly what TPMS sensors do when the car is parked.
    """
    try:
        snap = _ENGINE._read()
    except Exception:
        return False, "no telemetry"
    if not snap.get("available"):
        return False, "no telemetry"
    speed = None
    for row in snap.get("rows", []):
        if row["id"] == "vehicle_speed":
            speed = row.get("value")
            break
    if speed is None:
        return False, "no road speed"
    if speed >= config.HEALTH_DRIVING_MPH:
        return True, f"{speed:.0f} mph"
    return False, f"{speed:.0f} mph"


# ---------------------------------------------------------------------------
# The public API
# ---------------------------------------------------------------------------

def issues() -> List[dict]:
    """Everything currently wrong with the car, worst first.

    This is what the announcement policy is given, and it is the only thing it
    is given. Sorted here rather than there so the policy stays a pure function
    of an ordered list.
    """
    if not getattr(config, "VEHICLE_HEALTH_ENABLED", True):
        return []
    _refresh_all()
    driving, _ = _driving()
    ctx = {"driving": driving}
    out: List[dict] = []
    for src in _SOURCES:
        try:
            if not src.available():
                # A registered subsystem that is not reporting is a FACT about
                # the car, not an absence of one. Without this, unplugging the
                # TPMS receiver would make RIO say "everything looks good" — the
                # single worst thing this feature could do, and the difference
                # between "all four tires are fine" and "I can't see your tires"
                # is the whole reason `available()` is a separate question.
                out.append(make_issue(
                    key=f"{src.domain}:unavailable",
                    domain=src.domain, type=f"{src.domain}_unavailable",
                    severity=INFORMATIONAL,
                    message=f"Nothing is reporting from {src.label.lower()} — "
                            f"there is no data to judge that from.",
                    observation_window=W_NOW,
                    suggested_action="",
                    spoken_fallback=f"I've got nothing from {src.label.lower()}."))
                continue
            out.extend(src.issues(ctx) or [])
        except Exception as e:
            # A source that throws is a source that is not there. RIO says less;
            # she does not take the conversation down with her.
            print(f"[vehicle_health] source {src.domain} failed: "
                  f"{type(e).__name__}: {e}", flush=True)
    out.sort(key=lambda i: (-i["severity_rank"], -i["magnitude"], i["key"]))
    return out


def overall_status(issue_list: Optional[List[dict]] = None,
                   any_source: Optional[bool] = None) -> str:
    """normal | informational | warning | critical | unknown."""
    if any_source is None:
        any_source = any(_safe_available(s) for s in _SOURCES)
    if not any_source:
        return UNKNOWN
    issue_list = issues() if issue_list is None else issue_list
    worst = max((i["severity_rank"] for i in issue_list), default=0)
    return _STATUS_BY_RANK.get(worst, NORMAL)


def _safe_available(src: HealthSource) -> bool:
    try:
        return bool(src.available())
    except Exception:
        return False


def _history_depth(live_domains: List[str]) -> str:
    """What is actually known about the past, stated so RIO cannot exceed it.

    Assembled from the windows that are genuinely available rather than written
    as a constant, because the honest sentence changes when a source drops out.
    """
    parts = []
    if "tires" in live_domains:
        parts.append("tire pressure trends cover the last 24 hours")
    if "engine" in live_domains:
        parts.append(f"engine channel trends cover only "
                     f"{int(config.TELEMETRY_TREND_WINDOW_S)} seconds")
    if not parts:
        return "No history at all is available — nothing is reporting."
    joined = ", and ".join(parts)
    return ("This is live data from the current drive: "
            + joined
            + ". Nothing older than that is available to you.")


def context(full: bool = True) -> dict:
    """The normalized Vehicle Health state, in the spec's shape.

    `full=False` returns the same top-level judgement without the per-channel
    detail — see compact(), which is what every ordinary conversation turn gets.

    Never contains UI state. See the module header.
    """
    if not getattr(config, "VEHICLE_HEALTH_ENABLED", True):
        return {"vehicle_health": {"overall_status": UNKNOWN,
                                   "data_available": False,
                                   "issues": []}}

    # issues() first: it is what refreshes the sources, and everything below
    # must be read out of the SAME snapshot it was derived from.
    issue_list = issues()
    live = [s for s in _SOURCES if _safe_available(s)]
    live_domains = [s.domain for s in live]
    driving, driving_why = _driving()
    status = overall_status(issue_list, any_source=bool(live))

    body = {
        "overall_status": status,
        "data_available": bool(live),
        "subsystems_reporting": live_domains,
        "moving": driving,
        "history_depth": _history_depth(live_domains),
        "issue_count": len(issue_list),
        "issues": [_issue_view(i) for i in issue_list[:config.HEALTH_MAX_ISSUES]],
        "summary": summary_line(status, issue_list, bool(live)),
    }
    if len(issue_list) > config.HEALTH_MAX_ISSUES:
        # Never silently truncate. A list that was cut has to say so, or the
        # absence of a fault reads as its absence from the car.
        body["issues_omitted"] = len(issue_list) - config.HEALTH_MAX_ISSUES

    if full:
        for src in live:
            try:
                body[src.domain] = src.state()
            except Exception as e:
                print(f"[vehicle_health] state {src.domain} failed: "
                      f"{type(e).__name__}: {e}", flush=True)
        body["suggested_actions"] = [i["suggested_action"] for i in issue_list
                                     if i["suggested_action"]][:3]
        body["driving_state"] = driving_why

    return {"vehicle_health": body}


def _issue_view(issue: dict) -> dict:
    """The issue as the LLM sees it. Policy-only fields (key, magnitude,
    spoken_fallback) are dropped: they are bookkeeping for the cooldown, they
    mean nothing in a sentence, and every field in a prompt is a field a model
    can be tempted to read out loud."""
    return {
        "type": issue["type"],
        "domain": issue["domain"],
        # The single most important field in this structure, and the one worth
        # spending a token on in every turn: did the VEHICLE say this, or did
        # RIO work it out? A model given a merged list will merge them in its
        # sentence, and "your car is reporting a fault" about a RIO baseline
        # deviation is a false statement assembled from true ones.
        "reported_by": ("the vehicle's own computer"
                        if issue["domain"] == "diagnostics" else "RIO"),
        "severity": issue["severity"],
        "where": issue["location"] or None,
        "message": issue["message"],
        "observation_window": issue["observation_window"],
        "suggested_action": issue["suggested_action"] or None,
        "evidence": issue["evidence"],
    }


def summary_line(status: Optional[str] = None,
                 issue_list: Optional[List[dict]] = None,
                 any_source: Optional[bool] = None) -> str:
    """One line of English. The whole of what an ordinary turn is told.

    Deterministic — this is the compact injection and it goes into EVERY
    conversation turn, so it has to be cheap, stable and impossible to
    misread.
    """
    if issue_list is None:
        issue_list = issues()          # refreshes; see HealthSource.refresh
    if any_source is None:
        any_source = any(_safe_available(s) for s in _SOURCES)
    if status is None:
        status = overall_status(issue_list, any_source=any_source)

    if status == UNKNOWN:
        return "No vehicle health data is reporting right now."
    if not issue_list:
        return "Everything reporting normal."
    worst = issue_list[0]
    more = len(issue_list) - 1
    tail = f" (+{more} more)" if more > 0 else ""
    return f"{worst['message']}{tail}"


def compact() -> dict:
    """The cheap summary injected on every conversation turn.

    Three facts and one sentence. Anything larger than this on a turn about the
    weather is a cost paid on every turn for a question that was not asked --
    the full structure is one router classification away.
    """
    issue_list = issues()              # refreshes; see HealthSource.refresh
    live = any(_safe_available(s) for s in _SOURCES)
    status = overall_status(issue_list, any_source=live)
    return {
        "overall_status": status,
        "active_issues": len(issue_list),
        "summary": summary_line(status, issue_list, live),
    }


def compact_line() -> str:
    """compact(), as the one line of text that actually goes into the prompt."""
    c = compact()
    return (f"VEHICLE HEALTH: {c['overall_status']} · "
            f"{c['active_issues']} active issue"
            f"{'' if c['active_issues'] == 1 else 's'} · {c['summary']}")
