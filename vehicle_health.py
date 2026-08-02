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
WARNING = policy.WARNING
CRITICAL = policy.CRITICAL
SEVERITY_RANK = policy.SEVERITY_RANK

# Overall status, in the spec's own lowercase vocabulary. `unknown` is not a
# fourth severity — it is the honest answer when nothing is reporting, and it is
# deliberately not "normal": a car with no sensors on it is not a healthy car,
# it is a car nobody can see.
NORMAL = "normal"
UNKNOWN = "unknown"

_STATUS_BY_RANK = {0: NORMAL, 1: INFORMATIONAL, 2: WARNING, 3: CRITICAL}

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

# tires.py's note is the classification, in its own words. Keying off it rather
# than re-deriving from the numbers is what keeps the two from disagreeing: if
# the panel says "Losing Pressure Fast", RIO says the tire is losing air fast,
# and there is no second copy of the threshold to drift.
#
#   note -> (type, severity, window, action)
_TIRE_NOTE = {
    "Possible Blowout": (
        "possible_blowout", CRITICAL, W_NOW,
        "Pull over somewhere safe and look at the tire before driving further."),
    "Losing Pressure Fast": (
        "rapid_pressure_loss", CRITICAL, W_TIRE_TREND,
        "Stop and check it — this is a puncture, not a slow leak."),
    "Pressure Critical": (
        "critical_low_pressure", CRITICAL, W_NOW,
        "Get air into it before driving further, or fit the spare."),
    "Overheating": (
        "tire_overheating", CRITICAL, W_NOW,
        "Slow down and let it cool before carrying on."),
    "Pressure Low": (
        "low_pressure", WARNING, W_NOW,
        "Top it up at the next stop."),
    "Pressure High": (
        "high_pressure", WARNING, W_NOW,
        "Let a little out once the tires are cold."),
    "Running Hot": (
        "tire_running_hot", WARNING, W_NOW,
        "Worth checking the pressure and the brake on that corner."),
    "Pressure Dropping": (
        "possible_slow_leak", WARNING, W_TIRE_TREND,
        "Have it checked for a nail or a leaking valve in the next day or two."),
    "Sensor Battery Low": (
        "sensor_battery_low", INFORMATIONAL, W_NOW,
        "Book the sensor battery for replacement — no hurry."),
}

# States that are about the SENSOR rather than the tire. Handled apart because
# the severity of "I cannot see this corner" depends on whether the car is
# moving, which is the one thing the tire provider cannot know.
_TIRE_QUIET = {
    "DISCONNECTED": ("tire_sensor_disconnected", "The sensor on the {name} tire "
                     "has stopped answering."),
    "NO_DATA": ("tire_sensor_missing", "There is no sensor reporting from the "
                "{name} corner."),
    "STALE": ("tire_sensor_stale", "The {name} sensor's last reading is old "
              "enough that it may not reflect the tire now."),
}


class TireSource(HealthSource):
    """The four corners, read through tires.snapshot().

    Reads only. Every threshold, every state name and the wording of every note
    is decided in tires.py against config.py, and nothing here second-guesses
    any of it — this class turns that classification into the vocabulary the
    conversation and the announcement policy share.
    """

    domain = "tires"
    label = "Tires"

    def __init__(self):
        self._snap = None

    def refresh(self) -> None:
        self._snap = None

    def _read(self) -> dict:
        # tires.snapshot() is a pure read of the provider — no history, no
        # insight engine — so the only reason to hold it is to keep one context
        # build from asking twice.
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
                "corners": out}

    def issues(self, ctx: dict) -> List[dict]:
        snap = self._read()
        driving = bool(ctx.get("driving"))
        out: List[dict] = []

        # All four quiet at once is one receiver fault, not four tire faults --
        # the same judgement tires._banners makes, and pointing the driver at
        # four wheels when the thing to look at is the box would be worse than
        # saying nothing.
        tire_rows = snap.get("tires", [])
        quiet = [t for t in tire_rows if t["state"] in _TIRE_QUIET]
        if tire_rows and len(quiet) == len(tire_rows):
            all_stale = all(t["state"] == "STALE" for t in tire_rows)
            return [make_issue(
                key="tires:receiver:stale" if all_stale else "tires:receiver:silent",
                domain=self.domain,
                type="tire_sensors_all_stale" if all_stale else "tire_sensors_all_silent",
                severity=WARNING if driving else INFORMATIONAL,
                message=("Every tire reading is old enough that it may not "
                         "describe the tires now."
                         if all_stale else
                         "Nothing is reporting from any of the four tire sensors."),
                observation_window=W_NOW,
                suggested_action="The receiver, not the tires — check that first.",
                spoken_fallback=("Your tire readings have all gone stale."
                                 if all_stale else "I've lost all four tire sensors."),
                evidence={"corners": [t["corner"] for t in quiet],
                          "ages": [t.get("age_text", "") for t in tire_rows]})]

        for t in tire_rows:
            corner = t["corner"]
            raw = t.get("raw") or {}
            name = t.get("name", corner)
            spoken_loc = _CORNER_SPOKEN.get(corner, name.lower())

            if t["state"] in _TIRE_QUIET:
                type_, template = _TIRE_QUIET[t["state"]]
                # The spec's "sensor disconnected while driving" critical. A
                # sensor that drops out in the driveway is a dead battery; one
                # that drops out at speed is a corner of the car nobody can see,
                # on the channel whose failure mode is a blowout.
                if t["state"] == "DISCONNECTED" and driving:
                    sev, type_ = CRITICAL, "tire_sensor_lost_driving"
                elif t["state"] == "DISCONNECTED":
                    sev = WARNING
                else:
                    sev = INFORMATIONAL
                out.append(make_issue(
                    key=f"tires:{corner}:{t['state'].lower()}",
                    domain=self.domain, type=type_, severity=sev,
                    message=template.format(name=name.lower()),
                    observation_window=W_NOW,
                    suggested_action=("Worth a look at that sensor when you stop."
                                      if sev != INFORMATIONAL else ""),
                    location=spoken_loc,
                    magnitude=float(SEVERITY_RANK[sev]),
                    spoken_fallback=f"I've lost the sensor on the {spoken_loc} tire.",
                    evidence={"corner": corner, "detail": t.get("detail", "")}))
                continue

            note = tires.bare_note(t)
            if note not in _TIRE_NOTE:
                continue
            type_, sev, window, action = _TIRE_NOTE[note]
            out.append(make_issue(
                key=f"tires:{corner}:{type_}",
                domain=self.domain, type=type_, severity=sev,
                message=_tire_message(name, type_, raw),
                observation_window=window,
                suggested_action=action,
                location=spoken_loc,
                magnitude=_tire_magnitude(type_, raw),
                value=_tire_value(type_, raw),
                unit=("psi" if "pressure" in type_ or "blowout" in type_
                      else "f" if "heat" in type_ or "hot" in type_ else ""),
                spoken_fallback=_tire_message(name, type_, raw),
                evidence={"corner": corner,
                          "pressure_psi": raw.get("pressure_psi"),
                          "target_psi": raw.get("target_psi"),
                          "temp_f": raw.get("temp_f"),
                          "change_psi_24h": raw.get("trend_psi_24h"),
                          "detail": t.get("detail", "")}))
        return out


def _tire_status_word(state: str) -> str:
    """tires.py's state -> the spec's lowercase status vocabulary."""
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


def _tire_message(name: str, type_: str, raw: dict) -> str:
    """One sentence of plain English about one corner.

    Deterministic, and deliberately about MEANING rather than numbers: the
    figure is in `evidence` for RIO to reach for if it helps the driver, and
    this sentence is what the announcement path falls back on if a type has no
    line of its own.
    """
    psi = raw.get("pressure_psi")
    target = raw.get("target_psi")
    temp = raw.get("temp_f")
    change = raw.get("trend_psi_24h")
    low = f"{name} is {abs(psi - target):.1f} PSI under its target" \
        if (psi is not None and target is not None) else f"{name} is low"

    if type_ == "possible_blowout":
        return f"{name} has lost most of its air and is at {psi:.0f} PSI."
    if type_ == "rapid_pressure_loss":
        return (f"{name} has lost {abs(change):.1f} PSI in the last day — "
                f"that is a puncture rather than a slow leak.")
    if type_ == "critical_low_pressure":
        return f"{low} — low enough not to keep driving on."
    if type_ == "tire_overheating":
        return f"{name} is at {temp:.0f}°F, hot enough to damage the tire."
    if type_ == "low_pressure":
        return f"{low}."
    if type_ == "high_pressure":
        return f"{name} is over-inflated by {abs(psi - target):.1f} PSI."
    if type_ == "tire_running_hot":
        return f"{name} is running hotter than the others, at {temp:.0f}°F."
    if type_ == "possible_slow_leak":
        return (f"{name} has dropped {abs(change):.1f} PSI over the last "
                f"24 hours, which looks like a slow leak.")
    if type_ == "sensor_battery_low":
        return f"The sensor in the {name.lower()} tire is low on battery."
    return f"{name} needs attention."


def _tire_magnitude(type_: str, raw: dict) -> float:
    """How bad, in the units of the thing that is wrong. Higher is worse."""
    psi, target = raw.get("pressure_psi"), raw.get("target_psi")
    temp, change = raw.get("temp_f"), raw.get("trend_psi_24h")
    if type_ in ("possible_blowout", "critical_low_pressure", "low_pressure"):
        return max(0.0, (target - psi)) if (psi is not None and target is not None) else 0.0
    if type_ == "high_pressure":
        return max(0.0, (psi - target)) if (psi is not None and target is not None) else 0.0
    if type_ == "rapid_pressure_loss" or type_ == "possible_slow_leak":
        return abs(change) if change is not None else 0.0
    if type_ in ("tire_overheating", "tire_running_hot"):
        return max(0.0, temp - config.TIRE_TEMP_WARN_F) if temp is not None else 0.0
    return 0.0


def _tire_value(type_: str, raw: dict) -> Optional[float]:
    """The one number the spoken line takes, if it takes one."""
    if type_ in ("tire_overheating", "tire_running_hot"):
        return raw.get("temp_f")
    return raw.get("pressure_psi")


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


_ENGINE = EngineSource()

register_source(TireSource())
register_source(_ENGINE)


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
