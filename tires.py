"""tires.py — TireHealthProvider: the seam between whatever measures the tires
and the rest of the system.

Where this sits now
-------------------
The top-down tire graphic this file was written for is gone; the Vehicle Health
column is a telemetry list. Nothing below changed when that happened, which was
the point of the split. telemetry.py's TireTelemetryProvider calls snapshot()
and turns each corner into two ordinary telemetry rows, so the per-corner target
pressures, the nine named scenarios and the slow-leak detector all still run —
they just come out as channels in a list instead of numbers beside a picture of
a car. The tire thresholds stay in the tire half of config.py, and this remains
the only place in the codebase that knows what 29 PSI means.

The provider split
------------------
A TireHealthProvider answers exactly one question: *what are the four tires
reading right now?* It returns physics — pressure, temperature, a 24-hour
pressure trend, whether the sensor answered, how much battery that sensor has
left — and nothing else. It does not decide whether 29 PSI is a problem, does
not choose a word for it, and does not know a dashboard exists.

    hardware / mock  ──►  TireReading × 4          (this file, providers)
                          raw measurement only

    TireReading × 4  ──►  state, wording, banners  (this file, snapshot())
                          thresholds from config.py

    snapshot()       ──►  JSON the browser prints verbatim

Phase 1 ships one provider: MockTireProvider. Later there will be a Bluetooth
one, an RF TPMS receiver, an ESP32, RIO Connect. Each of those is a class with
`available()` and `read()` and nothing else changes — not this file's
normalizer, not the endpoint, not one line of the UI.

What the UI is allowed to know
------------------------------
Nothing. Every number the dashboard shows arrives here already formatted, every
state is a string it maps to a colour, and every sentence in the alert banner is
composed here. That is not tidiness for its own sake: a threshold duplicated in
JavaScript is a threshold that will disagree with the one in config.py the first
time somebody tunes it, and a tire panel that disagrees with itself is worse
than no tire panel. The browser is a renderer. If you find yourself wanting to
compute something in static/rio_vehicle.js, it belongs in this file or in
telemetry.py.

RIO does talk about tires now — and still not from this file
------------------------------------------------------------
The earlier version of this header said tire-aware speech was a later phase and
would go out through the speech arbiter when it arrived. It has arrived, and the
sentence about this file is unchanged: there is no arbiter call here and no
voice path here. vehicle_health.py reads snapshot() and normalizes it into the
context RIO reasons about; vehicle_health_policy.py decides whether a fault is
worth interrupting for and owns the words; static/rio_health.js submits that
decision to the existing arbiter at VEHICLE_HEALTH priority. What this file
gained is two extra CRITICAL notes (Possible Blowout, Losing Pressure Fast) so
that the dashboard and RIO cannot describe the same tire differently.

The rule the split protects is unchanged: this is still the only place in the
codebase that knows what 29 PSI means.
"""
from __future__ import annotations

import threading
import time
from dataclasses import dataclass, replace
from datetime import datetime
from typing import Dict, List, Optional

import config

CORNERS = ("FL", "FR", "RL", "RR")

CORNER_NAMES = {
    "FL": "Front Left",
    "FR": "Front Right",
    "RL": "Rear Left",
    "RR": "Rear Right",
}

# How the seven states rank against each other. The column headline and the
# alert banner are both "worst thing on the car", so the ordering has to be
# explicit rather than implied by the order of a chain of ifs.
#
# BATTERY_LOW sits below the no-contact states on purpose: a sensor with a dying
# battery is still telling you the pressure, and a tire you cannot hear from at
# all is the more urgent of the two. Both sit below an actual pressure problem,
# which is the only thing here that can strand a driver.
SEVERITY = {
    "NORMAL": 0,
    "BATTERY_LOW": 1,
    "STALE": 2,
    "DISCONNECTED": 2,
    "NO_DATA": 2,
    "WARNING": 3,
    "CRITICAL": 4,
}

# The warning triangle, with a NON-BREAKING space after it. The readout blocks
# beside the vehicle are about 85px wide, and with an ordinary space the glyph
# wraps onto a line of its own -- where it reads as a bullet point rather than
# as part of the phrase it belongs to. Defined once because the banner strips it
# back off, and a glyph that is added with one kind of space and removed with
# another is a bug that only shows up in the states nobody looks at.
_WARN_GLYPH = "\u26a0\u00a0"
_WARNED = ("WARNING", "CRITICAL", "BATTERY_LOW")


# ---------------------------------------------------------------------------
# What a provider returns
# ---------------------------------------------------------------------------

@dataclass
class TireReading:
    """One corner, as measured. No judgement, no formatting.

    Every field except `corner` is optional because every field is genuinely
    absent in some real hardware state: an unpaired sensor has no pressure, a
    cheap TPMS has no temperature, a receiver that has just powered up has no
    24-hour history to compute a trend from.
    """
    corner: str
    pressure_psi: Optional[float] = None
    temp_f: Optional[float] = None
    # Change over the last 24 hours. Negative means losing air. This is what
    # separates "this tire is a bit low" from "this tire has a hole in it", and
    # it is the reason a slow leak can be called before it becomes a warning.
    trend_psi_24h: Optional[float] = None
    # Did the sensor answer at all this cycle.
    connected: bool = True
    battery_pct: Optional[float] = None
    # When this reading was taken, epoch seconds. Not when it was asked for --
    # staleness is measured against the measurement, not against the poll.
    updated_at: Optional[float] = None


class TireHealthProvider:
    """Interface. Implement these two and the rest of the system works.

    `available()` is asked separately from `read()` because "there is no TPMS on
    this car" and "there is a TPMS and all four tires are fine" are different
    answers and the dashboard says different things about them.
    """

    name = "base"
    label = "Provider"

    def available(self) -> bool:
        raise NotImplementedError

    def read(self) -> List[TireReading]:
        """Current readings, one per corner. May be short or empty: a corner
        with no entry is reported as NO_DATA rather than silently skipped, so
        all four corners always have a row and a sensor going quiet is visible
        instead of invisible."""
        raise NotImplementedError


# ---------------------------------------------------------------------------
# The mock
# ---------------------------------------------------------------------------

def _base_set(now: float) -> Dict[str, TireReading]:
    """A healthy car. Every scenario is this, with one thing done to it.

    Deliberately not four identical numbers: real tires never agree to the
    decimal, and a mock that produces 35.0 across the board makes the panel look
    like a placeholder even when it is working.
    """
    return {
        "FL": TireReading("FL", 35.2, 84.0, -0.1, True, 92.0, now),
        "FR": TireReading("FR", 34.8, 83.0, -0.2, True, 88.0, now),
        "RL": TireReading("RL", 33.1, 87.0, -0.1, True, 95.0, now),
        "RR": TireReading("RR", 33.4, 85.0, 0.0, True, 71.0, now),
    }


def _sc_all_normal(now, elapsed):
    return _base_set(now)


def _sc_one_low(now, elapsed):
    s = _base_set(now)
    # ~3.7 under a 33.0 target: past the warning delta, well short of critical.
    s["RL"] = replace(s["RL"], pressure_psi=29.3, temp_f=88.0, trend_psi_24h=-0.6)
    return s


def _sc_one_critical(now, elapsed):
    s = _base_set(now)
    s["RL"] = replace(s["RL"], pressure_psi=25.6, temp_f=94.0, trend_psi_24h=-1.1)
    return s


def _sc_slow_leak(now, elapsed):
    """A tire actually losing air while you watch it.

    The pressure walks down with wall-clock time rather than sitting still,
    because the whole point of this scenario is the transition: it starts inside
    the normal band and is only flagged by the 24-hour trend, then crosses the
    warning delta about ninety seconds later. A frozen number would exercise the
    threshold but not the thing the threshold is for.

    The drop is capped just short of the critical delta on purpose. Left to run,
    this scenario would eventually become the one_critical scenario, and two
    entries in the menu that converge on the same picture are one entry.
    """
    s = _base_set(now)
    drop = min(4.4, 0.9 * (elapsed / 60.0))
    s["RL"] = replace(s["RL"],
                      pressure_psi=round(31.6 - drop, 1),
                      temp_f=86.0,
                      trend_psi_24h=-2.4)
    return s


def _sc_overheating(now, elapsed):
    s = _base_set(now)
    s["RR"] = replace(s["RR"], temp_f=171.0, pressure_psi=35.9, trend_psi_24h=0.4)
    return s


def _sc_sensor_disconnected(now, elapsed):
    s = _base_set(now)
    # The sensor did not answer. Everything it would have said is gone with it;
    # a provider that kept serving the last pressure it heard would be inventing
    # data, which is the one thing this layer must never do.
    s["FR"] = TireReading("FR", None, None, None, False, None, now - 240.0)
    return s


def _sc_battery_low(now, elapsed):
    s = _base_set(now)
    s["RR"] = replace(s["RR"], battery_pct=8.0)
    return s


def _sc_stale_data(now, elapsed):
    s = _base_set(now)
    # Everything still reads plausibly -- it is just old. This is the failure
    # that looks most like success, which is exactly why it gets its own state
    # instead of being folded into DISCONNECTED.
    old = now - (config.TIRE_STALE_AFTER_S + 720.0)
    return {c: replace(r, updated_at=old) for c, r in s.items()}


def _sc_no_sensors(now, elapsed):
    return {}


class MockTireProvider(TireHealthProvider):
    """Named scenarios instead of random numbers.

    A provider that jitters plausible values is useless for building a UI: you
    cannot get back to the state you were just looking at. Every scenario here is
    reproducible, and the only thing that moves on its own is the slow leak,
    which moves because that is what it is demonstrating.
    """

    name = "mock"
    label = "Mock Provider"

    # Labels are what fits in the selector at the foot of the column, which is
    # narrow on purpose -- a dev control that pushes the panel wider than the
    # column it is testing has changed the thing it was meant to observe.
    SCENARIOS = (
        ("all_normal",          "All Normal",     _sc_all_normal),
        ("one_low",             "One Low",        _sc_one_low),
        ("one_critical",        "One Critical",   _sc_one_critical),
        ("slow_leak",           "Slow Leak",      _sc_slow_leak),
        ("overheating",         "Overheating",    _sc_overheating),
        ("sensor_disconnected", "Sensor Offline", _sc_sensor_disconnected),
        ("battery_low",         "Battery Low",    _sc_battery_low),
        ("stale_data",          "Stale Data",     _sc_stale_data),
        ("no_sensors",          "No Sensors",     _sc_no_sensors),
    )

    def __init__(self, scenario: str = None):
        self._lock = threading.Lock()
        self._scenario = self._resolve(scenario or config.TIRE_DEFAULT_SCENARIO)
        self._set_at = time.time()

    @classmethod
    def _resolve(cls, name: str) -> str:
        return name if any(n == name for n, _, _ in cls.SCENARIOS) else "all_normal"

    @classmethod
    def scenarios(cls) -> List[dict]:
        return [{"name": n, "label": lbl} for n, lbl, _ in cls.SCENARIOS]

    @property
    def scenario(self) -> str:
        with self._lock:
            return self._scenario

    def set_scenario(self, name: str) -> bool:
        """-> True if the name was known. An unknown name changes nothing: the
        caller gets False and the panel keeps showing what it was showing."""
        if not any(n == name for n, _, _ in self.SCENARIOS):
            return False
        with self._lock:
            self._scenario = name
            # Reset, so the slow leak restarts from the top every time it is
            # selected rather than resuming wherever it got to an hour ago.
            self._set_at = time.time()
        return True

    def available(self) -> bool:
        return self.scenario != "no_sensors"

    def read(self) -> List[TireReading]:
        with self._lock:
            name, set_at = self._scenario, self._set_at
        build = next(fn for n, _, fn in self.SCENARIOS if n == name)
        now = time.time()
        return list(build(now, now - set_at).values())


# ---------------------------------------------------------------------------
# The one live provider
# ---------------------------------------------------------------------------

def _build_provider() -> TireHealthProvider:
    kind = getattr(config, "TIRE_PROVIDER", "mock")
    if kind != "mock":
        # Not fatal. A car with a misconfigured provider name should still show
        # a tire panel that says it has nothing, not a 500 on every poll.
        print(f"[tires] unknown provider {kind!r}; falling back to mock", flush=True)
    return MockTireProvider()


_provider: TireHealthProvider = _build_provider()


def provider() -> TireHealthProvider:
    return _provider


# ---------------------------------------------------------------------------
# Formatting. Every string the dashboard prints is made here.
# ---------------------------------------------------------------------------

def _psi_text(v: Optional[float]) -> str:
    # One decimal, not a whole number. A slow leak is sub-PSI per hour, and a
    # panel that rounds to 32 for three hours before jumping to 31 hides the one
    # thing the driver would have wanted to see coming.
    return "--" if v is None else f"{v:.1f}"


def _temp_text(v: Optional[float]) -> str:
    return "--" if v is None else f"{v:.0f}°F"


def _signed_psi(v: float) -> str:
    # U+2212 MINUS SIGN, not a hyphen: at 10px mono a hyphen in front of a
    # number reads as punctuation.
    return f"{'−' if v < 0 else '+'}{abs(v):.1f} PSI"


def _age_text(age: float) -> str:
    if age < 60:
        return f"{int(age)}s ago"
    if age < 3600:
        return f"{int(age / 60)}m ago"
    return f"{int(age / 3600)}h ago"


def _clock(ts: Optional[float]) -> str:
    return "--:--:--" if not ts else datetime.fromtimestamp(ts).strftime("%H:%M:%S")


# ---------------------------------------------------------------------------
# Classification
# ---------------------------------------------------------------------------

def _classify(r: Optional[TireReading], now: float) -> tuple:
    """-> (state, note, detail). One corner, everything decided.

    Order matters and is the order of what you would want to be told first. A
    tire that is not talking cannot be assessed, so contact is checked before
    pressure; pressure is checked before temperature because a flat tire ends
    the drive and a warm one usually does not; battery is last because it is the
    only thing here that is about the sensor rather than the tire.
    """
    if r is None:
        return "NO_DATA", "No Data", "No sensor paired to this corner"

    if not r.connected:
        seen = ("Last contact " + _age_text(now - r.updated_at)) if r.updated_at \
            else "No sensor contact"
        return "DISCONNECTED", "Sensor Offline", seen

    if r.pressure_psi is None:
        return "NO_DATA", "No Data", "Sensor paired but reporting nothing"

    age = now - r.updated_at if r.updated_at else 0.0
    if age > config.TIRE_STALE_AFTER_S:
        return "STALE", "Stale Data", f"Last reading {_age_text(age)}"

    target = config.TIRE_TARGET_PSI.get(r.corner, 35.0)
    delta = r.pressure_psi - target
    against = f"{_psi_text(r.pressure_psi)} PSI against a {target:.1f} PSI target"

    # Two refinements of CRITICAL, added when RIO gained a mouth for this panel.
    # Both are the SAME state and the same severity as the pressure-critical
    # branch below -- the column colours and the banner do not change -- and they
    # exist because "the tire is dangerously low" and "the tire is failing right
    # now" call for different words out loud, and the dashboard and RIO must not
    # be able to disagree about which one it is. Ordered worst-first, so a tire
    # that is both is described as the worse of the two.
    if r.pressure_psi <= config.TIRE_BLOWOUT_PSI:
        return "CRITICAL", "Possible Blowout", \
            f"{_psi_text(r.pressure_psi)} PSI · pull over as soon as it is safe"
    if r.trend_psi_24h is not None and r.trend_psi_24h <= config.TIRE_RAPID_LOSS_PSI_24H:
        return "CRITICAL", "Losing Pressure Fast", \
            f"{_signed_psi(r.trend_psi_24h)} over the last 24 hours · {against}"

    if delta <= -config.TIRE_PRESSURE_CRITICAL_DELTA:
        return "CRITICAL", "Pressure Critical", against + " · do not drive on it"
    if r.temp_f is not None and r.temp_f >= config.TIRE_TEMP_CRITICAL_F:
        return "CRITICAL", "Overheating", f"{_temp_text(r.temp_f)} · pull over and let it cool"
    if delta <= -config.TIRE_PRESSURE_WARN_DELTA:
        return "WARNING", "Pressure Low", against
    if delta >= config.TIRE_PRESSURE_HIGH_DELTA:
        return "WARNING", "Pressure High", against
    if r.temp_f is not None and r.temp_f >= config.TIRE_TEMP_WARN_F:
        return "WARNING", "Running Hot", f"{_temp_text(r.temp_f)} · above the normal running range"
    # Still inside the pressure band, but going the wrong way fast enough that
    # it will not be by tomorrow. This is the branch that catches a puncture on
    # the day it happens instead of the morning it strands somebody.
    if r.trend_psi_24h is not None and r.trend_psi_24h <= config.TIRE_TREND_LEAK_PSI_24H:
        return "WARNING", "Pressure Dropping", \
            f"{_signed_psi(r.trend_psi_24h)} over the last 24 hours"
    if r.battery_pct is not None and r.battery_pct <= config.TIRE_BATTERY_LOW_PCT:
        return "BATTERY_LOW", "Sensor Battery Low", \
            f"Sensor battery at {r.battery_pct:.0f}%"

    return "NORMAL", "", ""


def _normalize(r: Optional[TireReading], corner: str, now: float) -> dict:
    state, note, detail = _classify(r, now)
    target = config.TIRE_TARGET_PSI.get(corner, 35.0)
    has_link = bool(r and r.connected and r.pressure_psi is not None)
    batt = r.battery_pct if r else None
    age = (now - r.updated_at) if (r and r.updated_at) else None

    return {
        "corner": corner,
        "name": CORNER_NAMES[corner],
        "state": state,
        "severity": SEVERITY[state],
        # --- the numbers, already written out ---
        "pressure_text": _psi_text(r.pressure_psi if r else None),
        "pressure_unit": "PSI" if has_link else "",
        "temp_text": _temp_text(r.temp_f if r else None),
        "target_text": f"{target:.1f} PSI",
        "trend_text": (_signed_psi(r.trend_psi_24h) + " / 24h")
                      if (r and r.trend_psi_24h is not None) else "",
        "link_text": "Link" if has_link else "No Link",
        "link_state": "ok" if has_link else "lost",
        "battery_text": f"{batt:.0f}%" if batt is not None else "--",
        "battery_state": ("unknown" if batt is None
                          else "low" if batt <= config.TIRE_BATTERY_LOW_PCT else "ok"),
        "age_text": _age_text(age) if age is not None else "",
        # The glyph travels with the note so the browser never has to know which
        # states deserve a warning triangle.
        "note": ((_WARN_GLYPH + note) if state in _WARNED else note) if note else "",
        "detail": detail,
        # Raw values ride along for anything that is not the dashboard -- the
        # session log, a future arbiter, a test. The UI ignores them.
        "raw": {
            "pressure_psi": r.pressure_psi if r else None,
            "temp_f": r.temp_f if r else None,
            "trend_psi_24h": r.trend_psi_24h if r else None,
            "connected": bool(r.connected) if r else False,
            "battery_pct": batt,
            "target_psi": target,
            "updated_at": r.updated_at if r else None,
            "age_s": round(age, 1) if age is not None else None,
        },
    }


# ---------------------------------------------------------------------------
# Headline and banners
# ---------------------------------------------------------------------------

_QUIET = ("STALE", "DISCONNECTED", "NO_DATA")


def _headline(tires: List[dict], any_reading: bool) -> tuple:
    """-> (status, status_text). Worst thing on the car, in four words."""
    if not any_reading:
        return "UNAVAILABLE", "Sensor Data Unavailable"

    worst = max(t["severity"] for t in tires)

    if worst >= SEVERITY["CRITICAL"]:
        return "CRITICAL", "Critical Tire Issue"
    if worst >= SEVERITY["WARNING"]:
        return "ATTENTION", "Attention Needed"
    if worst >= SEVERITY["STALE"]:
        # All four gone quiet at once is not four tire faults, it is one
        # receiver fault, and saying "attention needed" about the tires would be
        # pointing at the wrong thing.
        if all(t["state"] in _QUIET for t in tires):
            if all(t["state"] == "STALE" for t in tires):
                return "UNAVAILABLE", "Sensor Data Stale"
            return "UNAVAILABLE", "Sensor Data Unavailable"
        return "ATTENTION", "Attention Needed"
    if worst >= SEVERITY["BATTERY_LOW"]:
        return "ATTENTION", "Attention Needed"
    return "NORMAL", "All Tires Normal"


# BATTERY_LOW banners are amber rather than dim, to agree with the tire block
# they refer to: the note on that corner is already amber, and a banner quieter
# than the thing it is explaining reads as a second, lesser problem.
_LEVEL = {"CRITICAL": "critical", "WARNING": "warning", "BATTERY_LOW": "warning"}


def _banners(tires: List[dict]) -> List[dict]:
    """Which issues get a banner under the vehicle.

    One, unless there is more than one critical -- in which case all of them,
    because "your rear left is flat" while the front right is also flat is a
    dangerously incomplete sentence. Everything else is a list the driver would
    read at 60 mph, which is to say would not read.
    """
    issues = [t for t in tires if t["severity"] > 0]
    if not issues:
        return []

    # All four corners quiet at once is one fault in the receiver, not four
    # faults in the tires. Picking the first corner and naming it would point
    # the driver at a wheel when the thing to look at is the box.
    if all(t["state"] in _QUIET for t in tires):
        if all(t["state"] == "STALE" for t in tires):
            ages = [t["age_text"] for t in tires if t["age_text"]]
            return [{"corner": None, "level": "info",
                     "title": "Tire Sensor Data Stale",
                     "detail": ("All four sensors last reported " + ages[0])
                               if ages else "All four sensors have stopped reporting"}]
        return [{"corner": None, "level": "info",
                 "title": "No Tire Sensor Data",
                 "detail": "Nothing is reporting from any corner"}]

    issues.sort(key=lambda t: -t["severity"])

    criticals = [t for t in issues if t["state"] == "CRITICAL"]
    chosen = criticals if len(criticals) > 1 else issues[:1]

    out = []
    for t in chosen:
        bare = t["note"].replace(_WARN_GLYPH, "")
        out.append({
            "corner": t["corner"],
            "level": _LEVEL.get(t["state"], "info"),
            # Plain space here: the banner runs the full width of the column
            # and has no wrapping problem to solve.
            "title": ("⚠ " if t["state"] in _WARNED else "")
                     + f"{t['name']} {bare}",
            "detail": t["detail"],
        })
    return out


# ---------------------------------------------------------------------------
# The one thing app.py calls
# ---------------------------------------------------------------------------

def snapshot() -> dict:
    """Everything the Vehicle Health column needs, in the shape it renders it.

    Nothing in here requires the caller to know which provider produced it, and
    nothing downstream is allowed to compute anything from it.
    """
    p = _provider
    now = time.time()

    try:
        readings = {r.corner: r for r in (p.read() or []) if r.corner in CORNERS}
    except Exception as e:
        # A provider that throws is a provider that is not there. The panel says
        # so; it does not take the dashboard down with it.
        print(f"[tires] provider {p.name} read failed: {type(e).__name__}: {e}",
              flush=True)
        readings = {}

    tires = [_normalize(readings.get(c), c, now) for c in CORNERS]
    any_reading = bool(readings) and any(t["raw"]["pressure_psi"] is not None
                                         for t in tires)
    status, status_text = _headline(tires, any_reading)

    stamps = [r.updated_at for r in readings.values() if r.updated_at]
    newest = max(stamps) if stamps else None
    age = (now - newest) if newest else None
    stale = age is not None and age > config.TIRE_STALE_AFTER_S
    if newest is None:
        updated_display = "No Data"
    elif stale:
        updated_display = f"{_clock(newest)} · {_age_text(age)}"
    else:
        updated_display = _clock(newest)

    scenario = getattr(p, "scenario", None)
    return {
        "provider": p.name,
        "provider_label": p.label,
        "available": p.available(),
        "status": status,
        "status_text": status_text,
        "updated_at": newest,
        "updated_display": updated_display,
        "stale": stale,
        "tires": tires,
        "banners": _banners(tires),
        # The browser takes its own cadence from the server so the poll rate is
        # one number in one place like every other tunable.
        "poll_ms": config.TIRE_POLL_MS,
        "scenario": scenario,
        "scenarios": MockTireProvider.scenarios() if scenario is not None else [],
    }


def bare_note(tire: dict) -> str:
    """The note on a normalized corner, without the warning glyph.

    The glyph travels with the note so the browser never has to know which
    states deserve one (see _normalize). Anything that wants to READ the note --
    the banner builder, and now vehicle_health.py, which keys its issue types
    off it -- has to take it back off, and doing that in one place is what stops
    the non-breaking space in _WARN_GLYPH from being a bug in every caller that
    tries it with an ordinary one.
    """
    return (tire.get("note") or "").replace(_WARN_GLYPH, "").strip()


def current_scenario() -> Optional[str]:
    return getattr(_provider, "scenario", None)


def scenarios() -> List[dict]:
    return MockTireProvider.scenarios() if hasattr(_provider, "scenario") else []


def set_scenario(name: str) -> bool:
    """-> True if it took. False on a provider with no scenarios (real hardware
    has exactly one scenario, which is whatever the tires are actually doing)."""
    setter = getattr(_provider, "set_scenario", None)
    return bool(setter and setter(name))
