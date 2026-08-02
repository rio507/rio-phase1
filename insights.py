"""insights.py — the layer that notices things before a warning light would.

What this is for
----------------
A warning light is a threshold that has already been crossed. By the time the
battery light comes on the alternator has failed; by the time the temperature
light comes on the coolant is already gone. Everything in this file exists to
say something one step earlier than that — not "the voltage is low" but "the
voltage has been sliding for three weeks and it did not use to."

That requires exactly one thing a dashboard normally does not have: memory. So
this module keeps a rolling history on disk, builds a per-channel baseline of
what *this* car normally does, and reports the difference.

    telemetry.py frame ──► observe()  ──► daily baselines  (baselines.json)
                                    └──►  detectors ──► Observation
                                                            │
                                                        Narrator
                                                            │
                                                    entries (insights.jsonl)
                                                            │
                                                    snapshot() ──► the panel

The narrator seam
-----------------
Detectors produce an `Observation`: a kind, a channel, a severity and a dict of
*facts* — numbers, never sentences. Turning those into English is a separate
step behind the `Narrator` interface. Today that is `TemplateNarrator`, which is
deterministic, offline and free. When RIO's intelligence layer takes over the
phrasing, an `LLMNarrator` implements the same one method and nothing else in
this file, in telemetry.py, or in the browser changes. The detectors are the
pipeline; the wording is a rendering of it.

That split is the whole point of doing it this way now. If the sentences were
generated inside the detectors, swapping in a model would mean rewriting the
detectors, and every one of them would have to be re-verified against a
non-deterministic component.

Insights do not speak
---------------------
Deliberately, and permanently as far as this phase is concerned. There is no
arbiter call in this file and no voice path out of it. A predictive observation
is a line in a log the driver reads when parked — it is *not* an alert, it does
not interrupt, and it must never become the reason RIO opens its mouth. The
firewall that keeps the Vehicle Health column silent applies here unchanged.
See the header of static/rio_vehicle.js.

Seeded history
--------------
A brand new install has no past, and a predictive panel with no past has nothing
to show. On an empty history this module seeds a demo: four weeks of daily
baselines and a handful of log entries. Every seeded day and every seeded entry
is flagged, the flag rides all the way out to the payload, and the panel prints
it. Observations computed from a window that contains seeded days are flagged
too. A demo that presents fabricated history as measured is the one failure this
layer cannot have.
"""
from __future__ import annotations

import json
import os
import threading
import time
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from typing import Dict, List, Optional

import config

_DIR = config.INSIGHTS_DIR
_LOG_PATH = os.path.join(_DIR, "insights.jsonl")
_BASELINE_PATH = os.path.join(_DIR, "baselines.json")

_lock = threading.RLock()


# ---------------------------------------------------------------------------
# What a detector produces
# ---------------------------------------------------------------------------

@dataclass
class Observation:
    """One thing worth saying, before anybody has decided how to say it.

    `facts` holds numbers and nothing else. That is what makes the narrator
    swappable: a template and a language model are given the same evidence and
    differ only in how they phrase it. A detector that writes prose into this
    dataclass has welded the two layers back together.
    """
    key: str                       # dedupe/cooldown identity, e.g. "drift:battery_voltage"
    kind: str                      # nominal | event | deviation | drift | response
    sensor: Optional[str]
    severity: str                  # info | notice | attention | critical
    icon: str
    facts: Dict = field(default_factory=dict)
    # True when the evidence behind this observation includes seeded demo days.
    seeded: bool = False


class Narrator:
    """Observation -> one plain-English sentence.

    The entire surface an LLM narration layer has to implement. It gets facts
    and returns a sentence; it does not decide whether there is something to
    say, which channel it is about, or how serious it is.
    """

    name = "base"

    def narrate(self, obs: Observation) -> str:
        raise NotImplementedError


# ---------------------------------------------------------------------------
# Formatting helpers used by the template narrator
# ---------------------------------------------------------------------------

def _fmt(value: Optional[float], precision: int = 1, unit: str = "") -> str:
    if value is None:
        return "--"
    text = f"{value:.{max(0, precision)}f}"
    if not unit:
        return text
    # Degrees and colons hug the number; everything else takes a space, the way
    # a person would write it.
    if unit.startswith("°") or unit.startswith(":"):
        return text + unit
    return text + " " + unit


def _span_phrase(days: float) -> str:
    """How a person says a length of time. "over the past three weeks", not
    "over 21.4 days" — this log is meant to read like Apple Health."""
    if days < 1.5:
        return "the past day"
    if days < 10:
        return f"the past {int(round(days))} days"
    weeks = days / 7.0
    words = {1: "one", 2: "two", 3: "three", 4: "four", 5: "five", 6: "six"}
    n = int(round(weeks))
    if n <= 1:
        return "the past week"
    if n >= 8:
        return f"the past {int(round(days / 30.0))} months"
    return f"the past {words.get(n, str(n))} weeks"


def _lower_label(label: str) -> str:
    """"Battery Voltage" -> "battery voltage", but "MAP" stays "MAP".

    Acronyms are left alone because "the map has declined" is a sentence about
    a road atlas.
    """
    parts = []
    for word in label.split():
        parts.append(word if (word.isupper() and len(word) > 1) else word.lower())
    return " ".join(parts)


_CONDITION_PHRASE = {
    "idle": "at idle",
    "load": "under load",
    "cruise": "at cruising speed",
}


class TemplateNarrator(Narrator):
    """The deterministic narrator. No model, no network, no variance.

    Every sentence here is assembled from the facts a detector measured. That
    makes the whole pipeline testable: given a history, the log is reproducible
    to the character, which is not true the moment a model is doing the writing.
    """

    name = "template"

    def narrate(self, obs: Observation) -> str:
        f = obs.facts
        label = f.get("label") or (obs.sensor or "Sensor")
        unit = f.get("unit", "")
        prec = int(f.get("precision", 1))
        cond = _CONDITION_PHRASE.get(f.get("condition") or "")

        if obs.kind == "nominal":
            return "Vehicle operating within expected parameters."

        if obs.kind == "event":
            return f.get("text") or f"{label} changed state."

        if obs.kind == "deviation":
            delta = float(f.get("delta", 0.0))
            direction = "above" if delta > 0 else "below"
            where = f" {cond}" if cond else ""
            return (f"{label} is consistently {_fmt(abs(delta), prec, unit)} "
                    f"{direction} your normal baseline{where}.")

        if obs.kind == "drift":
            change = float(f.get("change", 0.0))
            days = float(f.get("days", 0.0))
            verb = "risen" if change > 0 else "declined"
            where = f" {cond}" if cond else ""
            # First character only. str.capitalize() lowercases everything after
            # it, which turns "0.5 V" into "0.5 v" and "5°F" into "5°f" — the
            # units are the part of this sentence a mechanic actually reads.
            sentence = (f"{_lower_label(label)} has gradually {verb} "
                        f"{_fmt(abs(change), prec, unit)} over {_span_phrase(days)}"
                        f"{where}.")
            return sentence[:1].upper() + sentence[1:]

        if obs.kind == "response":
            drop = float(f.get("drop", 0.0))
            return (f"{label} drops {_fmt(abs(drop), prec, unit)} during "
                    f"aggressive acceleration.")

        return f"{label} observation."


# The live narrator. One line to change when RIO's intelligence layer is ready
# to take over the phrasing; nothing above or below it moves.
_narrator: Narrator = TemplateNarrator()


def narrator() -> Narrator:
    return _narrator


def set_narrator(n: Narrator) -> None:
    global _narrator
    with _lock:
        _narrator = n


# ---------------------------------------------------------------------------
# Conditioned channels
# ---------------------------------------------------------------------------
# A baseline of oil pressure across a whole drive is nearly useless: it is an
# average of idle and 4000 rpm, and it moves whenever the driving does. A
# baseline of oil pressure *at idle* is a number that only changes when the
# engine does, which is exactly the signal this feature is looking for.
#
# Each entry is (baseline key suffix, gate). The gate is checked against the
# frame, so a sample only lands in the conditioned bucket when the car was
# genuinely in that state.

def _cond_idle(f) -> bool:
    rpm = f.get("rpm")
    return rpm is not None and 400.0 <= rpm <= 1100.0


def _cond_load(f) -> bool:
    tp = f.get("throttle_pct")
    return tp is not None and tp >= 55.0


def _cond_cruise(f) -> bool:
    spd = f.get("vehicle_speed")
    return spd is not None and spd >= 25.0


CONDITIONS = {
    "oil_pressure":    [("idle", _cond_idle)],
    "battery_voltage": [("idle", _cond_idle)],
    "fuel_pressure":   [("load", _cond_load)],
    "coolant_temp":    [("cruise", _cond_cruise)],
}


def _split_key(key: str):
    """"oil_pressure@idle" -> ("oil_pressure", "idle")."""
    if "@" in key:
        sensor, cond = key.split("@", 1)
        return sensor, cond
    return key, None


# ---------------------------------------------------------------------------
# Persistence
# ---------------------------------------------------------------------------
# Two files, both small, both rewritten whole. The log is append-mostly and
# trimmed on write; the baselines are a single JSON object replaced atomically.
# Atomically matters: this process is polled several times a second and a
# half-written baselines file read back on the next tick would take the whole
# panel down with a JSONDecodeError.

def _ensure_dir() -> None:
    os.makedirs(_DIR, exist_ok=True)


def _write_json_atomic(path: str, payload: dict) -> None:
    _ensure_dir()
    tmp = path + ".tmp"
    with open(tmp, "w") as fh:
        json.dump(payload, fh, separators=(",", ":"))
    os.replace(tmp, path)


def _load_baselines() -> dict:
    try:
        with open(_BASELINE_PATH) as fh:
            data = json.load(fh)
    except Exception:
        data = {}
    data.setdefault("days", {})
    data.setdefault("meta", {})
    return data


def _load_entries() -> List[dict]:
    out = []
    try:
        with open(_LOG_PATH) as fh:
            for line in fh:
                line = line.strip()
                if not line:
                    continue
                try:
                    out.append(json.loads(line))
                except Exception:
                    # One bad line must not cost the whole history.
                    pass
    except FileNotFoundError:
        pass
    return out


def _write_entries(entries: List[dict]) -> None:
    _ensure_dir()
    entries = sorted(entries, key=lambda e: e.get("at", 0.0))[-config.INSIGHTS_MAX_ENTRIES:]
    tmp = _LOG_PATH + ".tmp"
    with open(tmp, "w") as fh:
        for e in entries:
            fh.write(json.dumps(e, separators=(",", ":")) + "\n")
    os.replace(tmp, _LOG_PATH)


def _day_key(ts: float) -> str:
    return datetime.fromtimestamp(ts).strftime("%Y-%m-%d")


# ---------------------------------------------------------------------------
# The engine
# ---------------------------------------------------------------------------

class InsightEngine:
    """Rolling baselines, deviation and drift, on a persisted history.

    One instance per process. Everything it knows that matters survives a
    restart, because a baseline that resets when the server bounces is not a
    baseline — it is the current reading with extra steps.
    """

    def __init__(self):
        self._lock = threading.RLock()
        self._baselines = _load_baselines()
        self._entries = _load_entries()
        # When each observation key last fired, so a channel that is genuinely
        # 7°F high does not write that sentence 3600 times an hour.
        self._last_fired: Dict[str, float] = {}
        for e in self._entries:
            k = e.get("key")
            if k:
                self._last_fired[k] = max(self._last_fired.get(k, 0.0), e.get("at", 0.0))
        # Short-lived, in-memory only: state transitions are about *this* run.
        self._prev_status: Dict[str, str] = {}
        self._prev_running: Optional[bool] = None
        self._dirty_baselines = False
        self._last_baseline_flush = 0.0
        self._seeded_days = set()
        for day, rec in self._baselines.get("days", {}).items():
            if rec.get("_seeded"):
                self._seeded_days.add(day)

        if config.INSIGHTS_SEED_DEMO and not self._entries and not self._baselines["days"]:
            self._seed_demo()

    # -- accumulation ------------------------------------------------------

    def _bucket(self, day: str, key: str) -> dict:
        days = self._baselines.setdefault("days", {})
        rec = days.setdefault(day, {})
        return rec.setdefault(key, {"n": 0, "sum": 0.0, "sumsq": 0.0,
                                    "min": None, "max": None})

    def _accumulate(self, frame: dict) -> None:
        """Fold one telemetry frame into today's running statistics.

        Only while the engine is running. A parked car contributes hours of
        zeroes to every pressure channel, and a baseline poisoned with those
        would make a healthy running engine look permanently high.
        """
        if not frame.get("engine_running"):
            return

        day = _day_key(frame["at"])
        values = frame.get("values", {})

        for sensor, value in values.items():
            if value is None:
                continue
            b = self._bucket(day, sensor)
            b["n"] += 1
            b["sum"] += value
            b["sumsq"] += value * value
            b["min"] = value if b["min"] is None else min(b["min"], value)
            b["max"] = value if b["max"] is None else max(b["max"], value)

            for suffix, gate in CONDITIONS.get(sensor, ()):
                if gate(values):
                    cb = self._bucket(day, f"{sensor}@{suffix}")
                    cb["n"] += 1
                    cb["sum"] += value
                    cb["sumsq"] += value * value
                    cb["min"] = value if cb["min"] is None else min(cb["min"], value)
                    cb["max"] = value if cb["max"] is None else max(cb["max"], value)

        meta = self._baselines.setdefault("meta", {})
        for sensor, m in (frame.get("meta") or {}).items():
            meta[sensor] = m

        self._dirty_baselines = True

    def _flush(self, now: float, force: bool = False) -> None:
        """Write the baselines out, at most once every 30 s.

        The frame rate here is 1 Hz and the file is rewritten whole; flushing on
        every frame would be a disk write per second for a number that changes
        in the fourth decimal place.
        """
        if not self._dirty_baselines:
            return
        if not force and (now - self._last_baseline_flush) < 30.0:
            return
        self._trim_days()
        _write_json_atomic(_BASELINE_PATH, self._baselines)
        self._dirty_baselines = False
        self._last_baseline_flush = now

    def _trim_days(self) -> None:
        days = self._baselines.get("days", {})
        keep = config.INSIGHTS_DRIFT_WINDOW_DAYS + 7
        if len(days) <= keep:
            return
        for day in sorted(days.keys())[:-keep]:
            days.pop(day, None)
            self._seeded_days.discard(day)

    # -- reading the history ----------------------------------------------

    def _daily_means(self, key: str, exclude_today: str = None) -> List[tuple]:
        """-> [(day_key, mean, n)] oldest first, days with too few samples
        dropped. A day with nine samples in it is a day the engine ran for nine
        seconds, and averaging that against a two-hour drive would let a single
        cold start rewrite the baseline."""
        out = []
        for day in sorted(self._baselines.get("days", {}).keys()):
            if exclude_today and day == exclude_today:
                continue
            rec = self._baselines["days"][day].get(key)
            if not rec or rec.get("n", 0) < config.INSIGHTS_MIN_SAMPLES_PER_DAY:
                continue
            out.append((day, rec["sum"] / rec["n"], rec["n"]))
        return out

    def _baseline_mean(self, key: str, today: str):
        """-> (mean, n_days, seeded) over every prior day, sample-weighted.

        Weighted by sample count rather than a plain mean of means, so a
        five-minute errand does not carry the same weight as a two-hour drive.
        """
        rows = self._daily_means(key, exclude_today=today)
        if not rows:
            return None, 0, False
        total_n = sum(n for _, _, n in rows)
        mean = sum(m * n for _, m, n in rows) / total_n
        seeded = any(day in self._seeded_days for day, _, _ in rows)
        return mean, len(rows), seeded

    def _today_mean(self, key: str, today: str):
        rec = self._baselines.get("days", {}).get(today, {}).get(key)
        if not rec or rec.get("n", 0) < config.INSIGHTS_MIN_SAMPLES_PER_DAY:
            return None, 0
        return rec["sum"] / rec["n"], rec["n"]

    # -- detectors ---------------------------------------------------------

    def _detect_events(self, frame: dict) -> List[Observation]:
        """State changes as they happen. The only detector that is about *now*.

        These are the cheap, legible entries that give the log a pulse between
        the slow observations — the engine started, the coolant came up to
        temperature, a channel went out of band.
        """
        out = []
        running = bool(frame.get("engine_running"))
        statuses = frame.get("statuses", {})
        meta = frame.get("meta", {})
        values = frame.get("values", {})

        if self._prev_running is not None and running != self._prev_running:
            out.append(Observation(
                key="event:engine_run",
                kind="event", sensor=None,
                severity="info", icon="◈",
                facts={"text": "Engine started." if running else "Engine stopped."},
            ))
        self._prev_running = running

        for sensor, status in statuses.items():
            prev = self._prev_status.get(sensor)
            self._prev_status[sensor] = status
            if prev is None or prev == status:
                continue

            m = meta.get(sensor, {})
            label = m.get("label", sensor)
            unit = m.get("unit", "")
            prec = int(m.get("precision", 1))
            value = values.get(sensor)

            # Warmed up. The one transition a driver actually waits for, and the
            # spec asks for it by name.
            if prev == "WARMING" and status in ("NORMAL", "CHARGING"):
                out.append(Observation(
                    key=f"event:warm:{sensor}",
                    kind="event", sensor=sensor, severity="info", icon="◈",
                    facts={"text": f"{label} reached operating temperature "
                                   f"at {_fmt(value, prec, unit)}."},
                ))
            elif status in ("WARNING", "CRITICAL") and prev not in ("WARNING", "CRITICAL"):
                out.append(Observation(
                    key=f"event:band:{sensor}:{status}",
                    kind="event", sensor=sensor,
                    severity="attention" if status == "WARNING" else "critical",
                    icon="⚠",
                    facts={"text": f"{label} moved out of its normal range "
                                   f"at {_fmt(value, prec, unit)}."},
                ))
            elif prev in ("WARNING", "CRITICAL") and status not in ("WARNING", "CRITICAL"):
                out.append(Observation(
                    key=f"event:recover:{sensor}",
                    kind="event", sensor=sensor, severity="info", icon="◈",
                    facts={"text": f"{label} returned to its normal range "
                                   f"at {_fmt(value, prec, unit)}."},
                ))
        return out

    def _detect_deviation(self, frame: dict) -> List[Observation]:
        """Today, against every day before it.

        This is the "consistently 7°F above your normal baseline" detector. It
        compares means rather than instantaneous values on purpose: a single hot
        reading in traffic is weather, and the thing worth a sentence is the
        channel that has settled somewhere new.
        """
        out = []
        today = _day_key(frame["at"])
        meta = frame.get("meta", {})

        for key in sorted(self._baselines.get("days", {}).get(today, {}).keys()):
            sensor, cond = _split_key(key)
            delta_min = config.INSIGHTS_DEVIATION_DELTA.get(sensor)
            if delta_min is None:
                continue

            now_mean, _ = self._today_mean(key, today)
            if now_mean is None:
                continue
            base_mean, n_days, seeded = self._baseline_mean(key, today)
            if base_mean is None or n_days < 2:
                continue

            delta = now_mean - base_mean
            if abs(delta) < delta_min:
                continue

            m = meta.get(sensor, {})
            out.append(Observation(
                key=f"deviation:{key}",
                kind="deviation", sensor=sensor,
                # Attention, never critical. A channel sitting off its baseline
                # is a thing to look into, not a thing to pull over for — the
                # bands in telemetry.py are what say the second thing.
                severity="attention" if abs(delta) >= 2 * delta_min else "notice",
                icon="◆",
                facts={"label": m.get("label", sensor), "unit": m.get("unit", ""),
                       "precision": int(m.get("precision", 1)),
                       "delta": delta, "baseline": base_mean,
                       "current": now_mean, "days": n_days, "condition": cond},
                seeded=seeded,
            ))
        return out

    def _detect_drift(self, frame: dict) -> List[Observation]:
        """The slow one. Slope across daily baselines over four weeks.

        Nothing else in RIO can see this. Every band in config.py is a snapshot
        judgement, and a battery that has fallen from 14.4 to 13.9 over a month
        passes every one of them on every single day of that month. The fit is
        least-squares over day index, and it is deliberately a fit rather than a
        first-to-last difference: one unusually cold morning at either end of
        the window would otherwise invent a trend or hide one.
        """
        out = []
        today = _day_key(frame["at"])
        meta = frame.get("meta", {})
        window_start = datetime.fromtimestamp(frame["at"]) - timedelta(
            days=config.INSIGHTS_DRIFT_WINDOW_DAYS)

        for sensor, min_change in config.INSIGHTS_DRIFT_DELTA.items():
            # Today is excluded. It is a partial day — possibly four minutes of
            # a cold start — and dropping it onto the end of a four-week fit
            # lets one unfinished morning invert the slope of a month. Drift is
            # a statement about completed days; what today is doing is the
            # deviation detector's question, not this one's.
            rows = [r for r in self._daily_means(sensor, exclude_today=today)
                    if datetime.strptime(r[0], "%Y-%m-%d") >= window_start]
            if len(rows) < config.INSIGHTS_DRIFT_MIN_DAYS:
                continue

            base_day = datetime.strptime(rows[0][0], "%Y-%m-%d")
            xs = [(datetime.strptime(d, "%Y-%m-%d") - base_day).days for d, _, _ in rows]
            ys = [m for _, m, _ in rows]
            slope = _slope(xs, ys)
            if slope is None:
                continue

            span_days = xs[-1] - xs[0]
            change = slope * span_days
            if abs(change) < min_change:
                continue

            m = meta.get(sensor, {})
            seeded = any(d in self._seeded_days for d, _, _ in rows)
            out.append(Observation(
                key=f"drift:{sensor}",
                kind="drift", sensor=sensor,
                severity="attention", icon="↗" if change > 0 else "↘",
                facts={"label": m.get("label", sensor), "unit": m.get("unit", ""),
                       "precision": int(m.get("precision", 1)),
                       "change": change, "days": span_days,
                       "slope_per_day": slope, "samples": len(rows),
                       "first": ys[0], "last": ys[-1], "condition": None},
                seeded=seeded,
            ))
        return out

    def _detect_response(self, frame: dict) -> List[Observation]:
        """How a channel behaves under load, against how it behaves at rest.

        "Fuel pressure drops during aggressive acceleration" is not a threshold
        — the pressure never leaves its band — it is the gap between two
        conditioned baselines. Same machinery as deviation, pointed sideways.
        """
        out = []
        today = _day_key(frame["at"])
        meta = frame.get("meta", {})

        for sensor, cond in (("fuel_pressure", "load"),):
            loaded, _ = self._today_mean(f"{sensor}@{cond}", today)
            resting, _ = self._today_mean(sensor, today)
            if loaded is None or resting is None:
                continue
            drop = resting - loaded
            floor = config.INSIGHTS_DEVIATION_DELTA.get(sensor, 3.0)
            if drop < floor:
                continue
            m = meta.get(sensor, {})
            out.append(Observation(
                key=f"response:{sensor}:{cond}",
                kind="response", sensor=sensor, severity="notice", icon="◆",
                facts={"label": m.get("label", sensor), "unit": m.get("unit", ""),
                       "precision": int(m.get("precision", 1)),
                       "drop": drop, "loaded": loaded, "resting": resting},
            ))
        return out

    def _detect_nominal(self, frame: dict, fired: List[Observation]) -> List[Observation]:
        """"Nothing to report" is a report.

        A health log that only ever writes when something is wrong leaves the
        driver unable to tell a healthy car from a broken sensor. This entry is
        the difference between "no news" and "no data", and it is rate-limited
        hard because it is the least interesting line in the file.
        """
        if fired:
            return []
        if not frame.get("engine_running"):
            return []
        statuses = frame.get("statuses", {})
        if any(s in ("WARNING", "CRITICAL") for s in statuses.values()):
            return []
        return [Observation(key="nominal", kind="nominal", sensor=None,
                            severity="info", icon="✓", facts={})]

    # -- the loop ----------------------------------------------------------

    def observe(self, frame: dict) -> None:
        """One telemetry frame in. Called on every telemetry poll.

        Cheap by design: accumulation is arithmetic on a dict, the detectors run
        over at most a few dozen daily aggregates, and the disk is touched twice
        a minute.
        """
        if not config.INSIGHTS_ENABLED:
            return
        now = frame.get("at") or time.time()
        with self._lock:
            self._accumulate(frame)

            fired: List[Observation] = []
            fired += self._detect_events(frame)
            fired += self._detect_deviation(frame)
            fired += self._detect_drift(frame)
            fired += self._detect_response(frame)
            fired += self._detect_nominal(frame, fired)

            new = []
            for obs in fired:
                cooldown = (config.INSIGHTS_NOMINAL_COOLDOWN_S
                            if obs.kind == "nominal" else config.INSIGHTS_COOLDOWN_S)
                last = self._last_fired.get(obs.key, 0.0)
                if now - last < cooldown:
                    continue
                self._last_fired[obs.key] = now
                new.append(self._record(obs, now))

            if new:
                self._entries.extend(new)
                self._entries = sorted(self._entries,
                                       key=lambda e: e["at"])[-config.INSIGHTS_MAX_ENTRIES:]
                _write_entries(self._entries)

            self._flush(now, force=bool(new))

    def _record(self, obs: Observation, at: float) -> dict:
        """Observation -> the stored entry. The narrator runs exactly here.

        The text is frozen into the log at write time rather than re-narrated on
        read, so history does not silently rewrite itself the day the narrator
        is swapped. `narrator` records which one wrote it.
        """
        return {
            "at": at,
            "key": obs.key,
            "kind": obs.kind,
            "sensor": obs.sensor,
            "severity": obs.severity,
            "icon": obs.icon,
            "text": _narrator.narrate(obs),
            "facts": obs.facts,
            "narrator": _narrator.name,
            "seeded": bool(obs.seeded),
        }

    # -- demo seed ---------------------------------------------------------

    def _seed_demo(self) -> None:
        """Four weeks of baselines and a few days of log, all flagged.

        The baselines are not decoration. The battery series below genuinely
        declines, so the drift detector *computes* "battery voltage has
        gradually declined over the past four weeks" from data rather than
        being handed the sentence. That is the point of seeding history instead
        of seeding conclusions — the pipeline is exercised end to end, and when
        real history replaces this the same code path produces the real answer.
        """
        now = time.time()
        n_days = config.INSIGHTS_DRIFT_WINDOW_DAYS
        # Half an hour of 1 Hz samples: a plausible day's driving, and
        # comfortably past INSIGHTS_MIN_SAMPLES_PER_DAY so the seeded days
        # actually qualify as days.
        samples = 1800

        for i in range(n_days, 0, -1):
            day = _day_key(now - i * 86400.0)
            age = i / float(n_days)      # 1.0 oldest, ~0 newest
            # A charging system quietly on its way out: 14.45 four weeks ago,
            # 13.95 now. Every one of those days passes the 13.2 warn band.
            batt = 13.95 + 0.50 * age
            # Coolant sitting a few degrees higher than it used to, the way a
            # tired thermostat or a fouling radiator behaves.
            coolant = 196.0 - 4.0 * age
            oil_p = 46.0 + 1.0 * age
            fuel_p = 57.6
            values = {
                "battery_voltage": batt,
                "coolant_temp": coolant,
                "oil_pressure": oil_p,
                "fuel_pressure": fuel_p,
                "oil_temp": 208.0,
                "intake_air_temp": 92.0,
                "afr_wideband": 14.66,
                "battery_voltage@idle": batt - 0.05,
                "oil_pressure@idle": 44.0 + 1.2 * age,
                "coolant_temp@cruise": coolant + 1.5,
                "fuel_pressure@load": fuel_p - 2.2,
            }
            rec = {}
            for key, mean in values.items():
                rec[key] = {"n": samples, "sum": mean * samples,
                            "sumsq": mean * mean * samples,
                            "min": mean - 1.0, "max": mean + 1.0}
            rec["_seeded"] = True
            self._baselines.setdefault("days", {})[day] = rec
            self._seeded_days.add(day)

        meta = self._baselines.setdefault("meta", {})
        meta.update({
            "battery_voltage": {"label": "Battery Voltage", "unit": "V", "precision": 1},
            "coolant_temp": {"label": "Coolant Temp", "unit": "°F", "precision": 0},
            "oil_pressure": {"label": "Oil Pressure", "unit": "PSI", "precision": 0},
            "fuel_pressure": {"label": "Fuel Pressure", "unit": "PSI", "precision": 0},
            "oil_temp": {"label": "Oil Temp", "unit": "°F", "precision": 0},
            "intake_air_temp": {"label": "Intake Air Temp", "unit": "°F", "precision": 0},
            "afr_wideband": {"label": "Wideband O₂", "unit": ":1", "precision": 1},
        })

        # A handful of past entries so the log is not empty above whatever the
        # live engine writes in this session. Offsets are hours back from now.
        seeded_log = [
            (26.0,  "info",      "✓", "Vehicle health excellent across all monitored systems."),
            (27.5,  "notice",    "◆", "Fuel pressure held steady through a full throttle pull."),
            (50.0,  "info",      "◈", "Coolant reached operating temperature in 3 minutes 40 seconds."),
            (52.0,  "attention", "◆", "Charging voltage averaged 14.1 V across the drive."),
            (74.0,  "info",      "✓", "Vehicle operating within expected parameters."),
            (99.0,  "notice",    "◆", "Fuel pressure dipped briefly during heavy throttle."),
            (121.0, "info",      "◈", "Oil temperature stabilised at 205 °F on the motorway."),
        ]
        entries = []
        for hours, severity, icon, text in seeded_log:
            entries.append({
                "at": now - hours * 3600.0,
                "key": f"seed:{hours}",
                "kind": "event",
                "sensor": None,
                "severity": severity,
                "icon": icon,
                "text": text,
                "facts": {},
                "narrator": "seed",
                "seeded": True,
            })
        self._entries = entries
        _write_entries(self._entries)
        self._dirty_baselines = True
        self._flush(now, force=True)

    # -- read side ---------------------------------------------------------

    def feed(self, limit: int = None) -> List[dict]:
        """Newest first, already worded and already timestamped for display."""
        limit = limit or config.INSIGHTS_FEED_LIMIT
        now = time.time()
        with self._lock:
            rows = sorted(self._entries, key=lambda e: e.get("at", 0.0), reverse=True)[:limit]
        return [_present(e, now) for e in rows]

    def stats(self) -> dict:
        with self._lock:
            days = self._baselines.get("days", {})
            return {
                "days_of_history": len(days),
                "seeded_days": len(self._seeded_days),
                "entries": len(self._entries),
                "narrator": _narrator.name,
            }


def _slope(xs: List[float], ys: List[float]) -> Optional[float]:
    """Least-squares slope, units per x. None when x never varies."""
    n = len(xs)
    if n < 2:
        return None
    mx = sum(xs) / n
    my = sum(ys) / n
    denom = sum((x - mx) ** 2 for x in xs)
    if denom <= 1e-9:
        return None
    return sum((x - mx) * (y - my) for x, y in zip(xs, ys)) / denom


# ---------------------------------------------------------------------------
# Display formatting. Every string the panel prints is made here.
# ---------------------------------------------------------------------------

def _when_text(ts: float, now: float) -> str:
    """"09:41" today, "Yesterday" yesterday, "5 Days Ago" before that.

    Calendar days, not elapsed hours: something at 23:50 last night is
    "Yesterday" at 00:10 even though it was twenty minutes ago, which is how a
    person reads a log.
    """
    then = datetime.fromtimestamp(ts)
    today = datetime.fromtimestamp(now).date()
    days = (today - then.date()).days
    if days <= 0:
        return then.strftime("%H:%M")
    if days == 1:
        return "Yesterday"
    if days < 7:
        return f"{days} Days Ago"
    if days < 14:
        return "Last Week"
    weeks = days // 7
    if weeks < 5:
        return f"{weeks} Weeks Ago"
    return then.strftime("%d %b")


def _present(entry: dict, now: float) -> dict:
    return {
        "at": entry.get("at"),
        "when": _when_text(entry.get("at", now), now),
        "kind": entry.get("kind", "event"),
        "sensor": entry.get("sensor"),
        "severity": entry.get("severity", "info"),
        "icon": entry.get("icon", "◈"),
        "text": entry.get("text", ""),
        # Flagged all the way to the screen. See the module header.
        "seeded": bool(entry.get("seeded")),
        "narrator": entry.get("narrator", "template"),
    }


# ---------------------------------------------------------------------------
# The two things the rest of the system calls
# ---------------------------------------------------------------------------

_engine: Optional[InsightEngine] = None


def engine() -> InsightEngine:
    global _engine
    if _engine is None:
        with _lock:
            if _engine is None:
                _engine = InsightEngine()
    return _engine


def observe(frame: dict) -> None:
    """Fold one telemetry frame in. Never raises: an insight layer that can take
    the telemetry endpoint down with it is worse than no insight layer."""
    try:
        engine().observe(frame)
    except Exception as e:
        print(f"[insights] observe failed: {type(e).__name__}: {e}", flush=True)


def snapshot() -> dict:
    """Everything the VEHICLE INSIGHTS section renders, already worded."""
    try:
        eng = engine()
        return {
            "available": True,
            "entries": eng.feed(),
            "stats": eng.stats(),
            "poll_ms": config.INSIGHTS_POLL_MS,
        }
    except Exception as e:
        print(f"[insights] snapshot failed: {type(e).__name__}: {e}", flush=True)
        return {"available": False, "entries": [], "stats": {},
                "poll_ms": config.INSIGHTS_POLL_MS}
