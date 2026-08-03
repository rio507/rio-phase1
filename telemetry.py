"""telemetry.py — every sensor on the car, in one array, judged in one place.

This is the generalisation of tires.py. That file proved the shape: a provider
returns physics, this layer decides what the physics means, and the browser
prints the result without ever holding a threshold. Everything below is the same
contract widened from four tires to the whole vehicle.

    hardware / mock  ──►  SensorReading[]           (providers, below)
                          raw measurement only

    SensorReading[]  ──►  rolling window per channel (this file)
                          slope, bands, modes, severity
                                   │
                                   ├──►  rows[] ──► GET /vehicle/telemetry
                                   │
                                   └──►  insights.observe(frame)

A provider answers one question: *what is this sensor reading right now?* It
does not know what NORMAL means, does not choose a word, and does not know a
dashboard exists. Phase 1 ships two: MockHolleyProvider, which models a Holley
Terminator X Stealth EFI, and TireTelemetryProvider, which is tires.py wearing
this interface. A real Holley serial reader, a CAN bus, an ESP32 and RIO Connect
all land beside them as classes with `available()` and `read()`, and not one
line of this file's normalizer, the endpoint, or the UI changes when they do.

Adding a sensor
---------------
Add a row to SENSORS. Optionally add a band to config.TELEMETRY_BANDS and a
trend delta to config.TELEMETRY_TREND_DELTA. That is the whole procedure — the
panel is generated from the array, so boost, transmission temperature,
differential temperature, brake temperature and suspension travel arrive without
anybody touching the layout.

Trend is a slope, not a difference
----------------------------------
`trend` comes from a least-squares fit over TELEMETRY_TREND_WINDOW_S, not from
comparing this sample to the last one. That is not a refinement, it is the
difference between an arrow and a coin flip: every channel here carries a little
sensor noise, and single-sample differencing on a noisy signal produces a
direction that reverses on almost every poll. Exactly the problem the headway
warnings had, solved the same way.

What the UI is allowed to know
------------------------------
Nothing. Every number arrives already formatted, every state is a string the
browser maps to a CSS class, and every glyph is chosen here. A threshold
duplicated in JavaScript is a threshold that will disagree with config.py the
first time somebody tunes it. See static/rio_vehicle.js.

RIO does not speak about telemetry
----------------------------------
Deliberately, in this phase. There is no arbiter call in this file, none in
insights.py, and none in the column's JavaScript. A predictive insight is not an
alert — see the header of insights.py. Telemetry-aware speech is a later phase
and will go out through the speech arbiter at coaching priority like everything
else with a mouth.
"""
from __future__ import annotations

import threading
import time
from collections import deque
from dataclasses import dataclass
from datetime import datetime
from typing import Dict, List, Optional, Tuple

import config
import insights
import tires
from vehicle.producers import physics


# ---------------------------------------------------------------------------
# What a provider returns
# ---------------------------------------------------------------------------

@dataclass
class SensorReading:
    """One channel, as measured. No judgement, no formatting.

    `ok` is separate from a null value because "the sensor did not answer" and
    "the sensor answered zero" are different facts about the car, and a panel
    that conflates them will one day tell a driver their oil pressure is fine
    when the wire has fallen off.
    """
    id: str
    value: Optional[float] = None
    ok: bool = True
    at: Optional[float] = None
    # Set only by providers that have already classified the channel themselves
    # — tires.py is the case that exists. Everything else is judged here against
    # config.TELEMETRY_BANDS.
    status_override: Optional[str] = None
    detail: str = ""


class TelemetryProvider:
    """Interface. Implement these two and the rest of the system works.

    `available()` is asked separately from `read()` for the same reason it is in
    tires.py: "there is no ECU on this car" and "there is an ECU and everything
    is fine" are different answers and the banner says different things about
    them.
    """

    name = "base"
    label = "Provider"

    def available(self) -> bool:
        raise NotImplementedError

    def read(self) -> List[SensorReading]:
        """Current readings. May be short: a channel with no entry is reported
        as NO DATA rather than silently dropped, so the row list is stable and
        a sensor going quiet is visible instead of invisible."""
        raise NotImplementedError


# ---------------------------------------------------------------------------
# The sensor catalogue
# ---------------------------------------------------------------------------
# Label, unit and precision only. Not one threshold — those are in config.py, so
# that tuning a limit and adding a channel are separate operations on separate
# files.
#
# `group` is what keeps this readable as it grows. Twenty rows in one flat list
# is a wall; twenty rows under five headings is a telemetry screen. Order within
# a group is the order given here.

@dataclass(frozen=True)
class SensorSpec:
    id: str
    label: str
    unit: str
    precision: int
    group: str


ENGINE = "Engine"
ELECTRICAL = "Electrical"
INDUCTION = "Induction & Fuel"
DRIVELINE = "Driveline"
CHASSIS = "Chassis"

GROUP_ORDER = (ELECTRICAL, ENGINE, INDUCTION, DRIVELINE, CHASSIS)

SENSORS: Tuple[SensorSpec, ...] = (
    SensorSpec("battery_voltage",  "Battery Voltage",  "V",    1, ELECTRICAL),

    SensorSpec("coolant_temp",     "Coolant Temp",     "°F",   0, ENGINE),
    SensorSpec("oil_pressure",     "Oil Pressure",     "PSI",  0, ENGINE),
    SensorSpec("oil_temp",         "Oil Temp",         "°F",   0, ENGINE),
    SensorSpec("fuel_pressure",    "Fuel Pressure",    "PSI",  0, ENGINE),

    SensorSpec("afr_target",       "Air Fuel Ratio",   ":1",   1, INDUCTION),
    SensorSpec("afr_wideband",     "Wideband O₂",      ":1",   1, INDUCTION),
    SensorSpec("map_kpa",          "MAP",              "kPa",  0, INDUCTION),
    SensorSpec("maf_gs",           "Mass Air Flow",    "g/s",  1, INDUCTION),
    SensorSpec("throttle_pct",     "Throttle Position", "%",   0, INDUCTION),
    SensorSpec("engine_load",      "Engine Load",      "%",    0, INDUCTION),
    SensorSpec("intake_air_temp",  "Intake Air Temp",  "°F",   0, INDUCTION),
    # The two fuel trims are what make a lean condition visible BEFORE the ECU
    # sets a code for it. They are the single most useful pair on this panel for
    # the early-detection story: a long-term trim that has been climbing for a
    # week is a vacuum leak nobody has noticed yet, and it passes every band on
    # every one of those days.
    SensorSpec("stft_b1",          "Short Term Fuel Trim", "%", 1, INDUCTION),
    SensorSpec("ltft_b1",          "Long Term Fuel Trim",  "%", 1, INDUCTION),

    SensorSpec("rpm",              "Engine RPM",       "",     0, DRIVELINE),
    SensorSpec("vehicle_speed",    "Vehicle Speed",    "mph",  0, DRIVELINE),
)

# Tire channels are generated rather than typed out, because there are eight of
# them and a typo in one corner's id is a row that silently never updates.
_TIRE_LABEL = {"FL": "Front Left", "FR": "Front Right",
               "RL": "Rear Left", "RR": "Rear Right"}
_TIRE_SENSORS = tuple(
    s for corner in tires.CORNERS for s in (
        SensorSpec(f"tire_pressure_{corner}", f"{_TIRE_LABEL[corner]} Pressure",
                   "PSI", 1, CHASSIS),
        SensorSpec(f"tire_temp_{corner}", f"{_TIRE_LABEL[corner]} Temp",
                   "°F", 0, CHASSIS),
    )
)

ALL_SENSORS: Tuple[SensorSpec, ...] = SENSORS + _TIRE_SENSORS
SPEC_BY_ID: Dict[str, SensorSpec] = {s.id: s for s in ALL_SENSORS}

# Which config key a generated tire channel borrows its trend delta and band
# from, so eight ids do not need eight near-identical config entries.
_BASE_KEY = {}
for _c in tires.CORNERS:
    _BASE_KEY[f"tire_pressure_{_c}"] = "tire_pressure"
    _BASE_KEY[f"tire_temp_{_c}"] = "tire_temp"


def _cfg_key(sensor_id: str) -> str:
    return _BASE_KEY.get(sensor_id, sensor_id)


# ---------------------------------------------------------------------------
# Status vocabulary
# ---------------------------------------------------------------------------
# Same shape as tires.py's SEVERITY and for the same reason: the banner is
# "worst thing on the car", so the ordering has to be explicit rather than
# implied by a chain of ifs.
#
# The contextual modes all rank zero. An engine at 900 rpm is not a problem to
# be ranked, it is idling, and a car at 0 mph is stopped — those words describe
# the state, they do not grade it.

STATUS_SEVERITY = {
    "NORMAL": 0,
    "IDLE": 0,
    "STOPPED": 0,
    "WARMING": 0,
    "CHARGING": 0,
    "CRANKING": 0,
    "SENSOR LOW": 1,
    "STALE": 2,
    "OFFLINE": 2,
    "NO DATA": 2,
    "WARNING": 3,
    "CRITICAL": 4,
}

# Statuses for a channel that is parked. These show a flat dash instead of an
# arrow: the trend of an engine that is idling is not information, it is noise
# with a direction attached, and the spec draws exactly this — RPM IDLE —,
# Throttle IDLE —, Speed STOPPED —.
#
# WARMING and CHARGING are deliberately NOT in this list even though they are
# also contextual modes. A coolant temperature that is warming is the one
# channel on the panel where the direction is the whole story, and a dash there
# would hide the only thing the driver is waiting to see.
PARKED_STATUSES = ("IDLE", "STOPPED", "CRANKING")
QUIET_STATUSES = ("OFFLINE", "NO DATA", "STALE")

TREND_GLYPH = {"up": "↑", "down": "↓", "stable": "→", "abnormal": "⚠", "none": "—"}


def _status_class(status: str) -> str:
    """"NO DATA" -> "no-data". The browser needs a class name, and computing one
    from a display string in JavaScript is exactly the kind of small decision
    that belongs on this side of the wire."""
    return status.lower().replace(" ", "-")


# ---------------------------------------------------------------------------
# The mock ECU
# ---------------------------------------------------------------------------
# The engine model itself moved to vehicle/producers/physics.py, and it moved
# for one reason: the canonical simulator needs the same model. Two engine models
# drift, and "the simulator produces what the mock produces" would have become a
# thing somebody hoped rather than a thing that was true. Now both call
# physics.sample() and the identical numbers go down the direct path and the
# ingestion path.
#
# What is left here is the PROVIDER — the two methods this file's interface asks
# for, and the scenario selection that is dev-only and disappears the moment a
# real source is live.

Scenario = physics.Scenario
SCENARIOS = physics.SCENARIOS


class MockHolleyProvider(TelemetryProvider):
    """A Holley Terminator X Stealth EFI that is not there yet.

    Emits the channel set a Terminator X actually publishes, with values that
    agree with each other. Named scenarios rather than randomness, for the same
    reason MockTireProvider uses them: a scenario is reproducible and a random
    walk is not.
    """

    name = "mock_holley"
    label = "Holley Terminator X (Mock)"

    def __init__(self, scenario: str = None):
        self._lock = threading.Lock()
        self._scenario = physics.resolve(scenario or config.TELEMETRY_DEFAULT_SCENARIO)
        self._set_at = time.time()
        # The last sample, for the scenarios whose fault is defined by what did
        # NOT change — a frozen sensor reports the value it last had, and that
        # is impossible to model without remembering it.
        self._previous: Dict[str, Optional[float]] = {}

    @classmethod
    def scenarios(cls) -> List[dict]:
        return physics.catalogue()

    @property
    def scenario(self) -> str:
        with self._lock:
            return self._scenario

    def set_scenario(self, name: str) -> bool:
        if name not in physics.BY_NAME:
            return False
        with self._lock:
            self._scenario = name
            # Reset the clock: warm-up must restart from cold every time it is
            # selected rather than resuming wherever it got to an hour ago.
            self._set_at = time.time()
            self._previous = {}
        return True

    def available(self) -> bool:
        return True

    def read(self) -> List[SensorReading]:
        with self._lock:
            name, set_at = self._scenario, self._set_at
            previous = dict(self._previous)
        now = time.time()
        values, dropped = physics.sample(physics.BY_NAME[name], now - set_at,
                                         previous)
        with self._lock:
            self._previous = dict(values)

        out = []
        for sensor_id, value in values.items():
            if sensor_id in dropped:
                # The sensor did not answer. Everything it would have said goes
                # with it — a provider that kept serving the last value it heard
                # would be inventing data, which is the one thing this layer
                # must never do.
                out.append(SensorReading(sensor_id, None, ok=False, at=now,
                                         detail="Sensor not responding"))
                continue
            out.append(SensorReading(sensor_id, value, ok=True, at=now))
        # A channel that did not answer at all still has to be REPORTED as not
        # answering, or a dropout looks identical to a channel nobody asked
        # about. Absent-because-unsupported is a different case and is handled
        # by simply not being here — see physics.Scenario.unsupported.
        for sensor_id in dropped:
            if sensor_id not in values:
                out.append(SensorReading(sensor_id, None, ok=False, at=now,
                                         detail="Sensor not responding"))
        return out


# ---------------------------------------------------------------------------
# tires.py, wearing this interface
# ---------------------------------------------------------------------------

class TireTelemetryProvider(TelemetryProvider):
    """The tire sensors as eight ordinary telemetry channels.

    The top-down tire graphic is gone, but the provider behind it is not: it
    still has nine named scenarios, a leak that develops while you watch, and
    the only classification logic in the codebase that understands per-corner
    target pressures. Rather than duplicate that here, this adapter calls
    tires.py's own classifier and translates its state names into this file's
    vocabulary. The tire thresholds stay in the tire half of config.py, where
    they belong, and there is exactly one place that knows what 29 PSI means.
    """

    name = "tires"
    label = "TPMS"

    # tires.py's states -> the status words this panel prints. The two
    # vocabularies are close but not identical, and mapping them explicitly is
    # what keeps either of them free to change.
    STATE_MAP = {
        "NORMAL": "NORMAL",
        "WARNING": "WARNING",
        "CRITICAL": "CRITICAL",
        "BATTERY_LOW": "SENSOR LOW",
        "STALE": "STALE",
        "DISCONNECTED": "OFFLINE",
        "NO_DATA": "NO DATA",
    }

    def available(self) -> bool:
        return tires.provider().available()

    def read(self) -> List[SensorReading]:
        snap = tires.snapshot()
        now = time.time()
        out = []
        for t in snap.get("tires", []):
            corner = t["corner"]
            status = self.STATE_MAP.get(t["state"], "NO DATA")
            raw = t.get("raw", {})
            at = raw.get("updated_at") or now
            detail = t.get("detail") or t.get("note") or ""

            out.append(SensorReading(f"tire_pressure_{corner}", raw.get("pressure_psi"),
                                     ok=status not in QUIET_STATUSES, at=at,
                                     status_override=status, detail=detail))
            # The temperature carries the pressure's status only when the
            # sensor is the problem. A tire that is 4 PSI low is not a tire
            # whose temperature is wrong, and colouring that row amber would
            # report one fault as two.
            temp_status = status if status in QUIET_STATUSES else None
            out.append(SensorReading(f"tire_temp_{corner}", raw.get("temp_f"),
                                     ok=status not in QUIET_STATUSES, at=at,
                                     status_override=temp_status, detail=detail))
        return out


# ---------------------------------------------------------------------------
# The live providers
# ---------------------------------------------------------------------------

class IngestedProvider(TelemetryProvider):
    """Everything that arrived over the canonical ingestion API.

    This is the class that makes "the health engine does not care where a signal
    came from" true rather than aspirational. A CANable bridge on a real car, a
    passive Holley capture, a replay of a recorded drive and the simulator all
    POST the same canonical events; they land in vehicle/providers/ingested.py's
    buffer; and this reads that buffer through the same two methods
    MockHolleyProvider implements. Not one line below this class knows the
    difference.

    `source_types` narrows it to one producer, which is what the dashboard's
    source selector switches. Left empty it accepts whatever is arriving, which
    is the right behaviour on a car where the bridge is reporting OBD-II and
    Holley at once.

    available() is FALSE when nothing has arrived, and that is the important
    half. "There is no bridge connected" and "there is a bridge and the engine
    is fine" are different answers, the banner says different things about them,
    and vehicle_health.py raises an explicit `engine_unavailable` issue rather
    than letting an empty panel read as a healthy one.
    """

    def __init__(self, name: str, label: str, source_types=()):
        self.name = name
        self.label = label
        self._source_types = tuple(source_types)

    def available(self) -> bool:
        from vehicle.providers import ingested
        return bool(ingested.readings(list(self._source_types) or None))

    def read(self) -> List[SensorReading]:
        from vehicle.providers import ingested
        out = []
        for row in ingested.readings(list(self._source_types) or None):
            out.append(SensorReading(
                row["telemetry_id"], row["value"], ok=row["ok"], at=row["at"],
                detail=row["detail"]))
        return out


# ---------------------------------------------------------------------------
# Which producer the pipeline is listening to
# ---------------------------------------------------------------------------
# The spec's source selector, and the reason it is a registry rather than an if
# statement: `config.TELEMETRY_PROVIDER` used to be read and then ignored — any
# value but "mock" printed a warning and produced the mock anyway — so there was
# no way to point this pipeline at anything, and "switching sources does not
# require a backend restart" was not a property the code had.
#
# Note what does NOT change when the source does: the bands, the trend window,
# the staleness rule, the insight engine, the conversation layer, the browser.
# The provider changes and nothing else, which is the whole claim.

@dataclass(frozen=True)
class Source:
    name: str
    label: str
    build: object            # () -> TelemetryProvider
    detail: str = ""
    # An in-process producer that has to be advanced before it can be read.
    # None for the mock, which computes from the clock on every read, and for
    # the live sources, whose producer is a bridge in a car.
    pump: object = None       # (now: float) -> dict, or None


def _pump_simulation(now):
    from vehicle import producers
    return producers.pump_simulation(now)


def _pump_replay(now):
    from vehicle import producers
    return producers.pump_replay(now)


SOURCES: Tuple[Source, ...] = (
    Source("mock_holley", "Holley Terminator X (Mock)",
           lambda: MockHolleyProvider(),
           "In-process mock, read directly. The original development path."),
    Source("simulation", "Simulation",
           lambda: IngestedProvider("simulation", "Simulation",
                                    ("dashboard_simulator",)),
           "The same physics, pushed through the canonical ingestion API.",
           pump=_pump_simulation),
    Source("live_obd", "Live OBD-II",
           lambda: IngestedProvider("live_obd", "Live OBD-II", ("obd2_can",)),
           "A bridge on a CAN OBD-II vehicle."),
    Source("live_holley", "Live Holley",
           lambda: IngestedProvider("live_holley", "Live Holley",
                                    ("holley_terminator_x",)),
           "A bridge listening passively to a Holley bus."),
    Source("replay", "Recorded Replay",
           lambda: IngestedProvider("replay", "Recorded Replay",
                                    ("recorded_replay",)),
           "A recorded canonical log, played back.",
           pump=_pump_replay),
)

SOURCE_BY_NAME: Dict[str, Source] = {s.name: s for s in SOURCES}


def _resolve_source(name: str) -> str:
    if name in SOURCE_BY_NAME:
        return name
    # Not fatal. A misconfigured source name should still show a telemetry panel
    # that says what it has, not a 500 on every poll.
    print(f"[telemetry] unknown source {name!r}; falling back to "
          f"{SOURCES[0].name}", flush=True)
    return SOURCES[0].name


_source_name: str = _resolve_source(getattr(config, "VEHICLE_SOURCE_DEFAULT",
                                            "mock_holley"))
_source_lock = threading.Lock()


def _build_providers(source_name: str) -> List[TelemetryProvider]:
    """The ECU source, plus the tire receiver.

    The tires are not a source and are not switched: TPMS is a separate
    subsystem with its own radio, and a car whose OBD bridge was unplugged still
    has four tires being watched. Bundling them into the source selector would
    make unplugging the bridge look like losing the tires.
    """
    return [SOURCE_BY_NAME[source_name].build(), TireTelemetryProvider()]


_providers: List[TelemetryProvider] = _build_providers(_source_name)


def providers() -> List[TelemetryProvider]:
    return _providers


def source() -> str:
    return _source_name


def sources() -> List[dict]:
    return [{"name": s.name, "label": s.label, "detail": s.detail}
            for s in SOURCES]


def set_source(name: str) -> bool:
    """Point the pipeline at a different producer. -> False on an unknown name.

    Clears the trend history and the runtime clock, for the same reason
    set_scenario does: a slope fitted across the moment the source changed is a
    slope across a discontinuity, and it would report a spectacular direction
    for a channel that simply came from somewhere else.
    """
    global _providers, _source_name
    if name not in SOURCE_BY_NAME:
        return False
    with _source_lock:
        if name != _source_name:
            _source_name = name
            _providers = _build_providers(name)
            _history.reset()
            _runtime.reset()
    return True


def _ecu() -> Optional[TelemetryProvider]:
    """The active source provider, whatever it is.

    Used only for the banner's label and for the dev scenario controls, both of
    which ask politely: a provider with no scenarios (which is every real one)
    simply has no set_scenario, and the selector disappears from the panel.
    """
    return _providers[0] if _providers else None


# ---------------------------------------------------------------------------
# Rolling history — the thing trend is computed from
# ---------------------------------------------------------------------------

class _History:
    """A short ring per channel. Seconds, not days — days are insights.py's job.

    This exists for exactly one purpose: to give the slope fit something to fit.
    It is deliberately in memory and deliberately small; a trend arrow does not
    need to survive a restart, and one that tried to would be showing a
    direction measured before the engine was last switched off.
    """

    def __init__(self):
        self._lock = threading.Lock()
        self._rings: Dict[str, deque] = {}

    def push(self, sensor_id: str, value: Optional[float], at: float) -> None:
        if value is None:
            return
        with self._lock:
            ring = self._rings.setdefault(sensor_id, deque())
            ring.append((at, value))
            cutoff = at - config.TELEMETRY_TREND_WINDOW_S
            while ring and ring[0][0] < cutoff:
                ring.popleft()

    def window(self, sensor_id: str) -> List[Tuple[float, float]]:
        with self._lock:
            return list(self._rings.get(sensor_id, ()))

    def reset(self) -> None:
        with self._lock:
            self._rings.clear()


_history = _History()


def _slope_per_window(sensor_id: str) -> Optional[float]:
    """Least-squares change across the whole window, in the channel's units.

    Returned as total change over the window rather than a rate, because the
    threshold it is compared against (TELEMETRY_TREND_DELTA) is easiest to
    reason about that way: "how much would this have to move in twenty seconds
    before I would call it moving".
    """
    pts = _history.window(sensor_id)
    if len(pts) < config.TELEMETRY_TREND_MIN_SAMPLES:
        return None
    t0 = pts[0][0]
    xs = [p[0] - t0 for p in pts]
    ys = [p[1] for p in pts]
    n = len(xs)
    mx = sum(xs) / n
    my = sum(ys) / n
    denom = sum((x - mx) ** 2 for x in xs)
    if denom <= 1e-9:
        return None
    slope = sum((x - mx) * (y - my) for x, y in zip(xs, ys)) / denom
    span = xs[-1] - xs[0]
    if span <= 0:
        return None
    return slope * span


# ---------------------------------------------------------------------------
# Classification
# ---------------------------------------------------------------------------

def _band(sensor_id: str) -> dict:
    return config.TELEMETRY_BANDS.get(_cfg_key(sensor_id), {})


def band(sensor_id: str) -> dict:
    """The configured limits for a channel. Read-only.

    Public because vehicle_health.py needs to know HOW FAR past a limit a
    reading is, not just that it is past one — "how much worse did this get" is
    the whole of the announcement policy's worsening test. Going through here
    rather than reading config.TELEMETRY_BANDS directly keeps the tire channels'
    key aliasing (_cfg_key) in one place.
    """
    return dict(_band(sensor_id))


def _classify(spec: SensorSpec, r: Optional[SensorReading],
              now: float, engine_running: bool) -> Tuple[str, str]:
    """-> (status, detail). One channel, everything decided.

    Order is the order of what you would want to be told first: a channel that
    is not answering cannot be assessed, so contact is checked before value;
    value is checked against the critical band before the warning band; and the
    contextual modes are last, because a description only applies to a channel
    that has nothing wrong with it.
    """
    if r is None:
        return "NO DATA", "No sensor reporting on this channel"
    if r.status_override:
        return r.status_override, r.detail
    if not r.ok:
        return "OFFLINE", r.detail or "Sensor not responding"
    if r.value is None:
        return "NO DATA", r.detail or "Channel present but reporting nothing"

    age = now - r.at if r.at else 0.0
    if age > config.TELEMETRY_STALE_AFTER_S:
        return "STALE", f"Last reading {age:.0f}s ago"

    band = _band(spec.id)
    # A band marked `running` is meaningless on a stopped engine: oil pressure
    # is 0 PSI at key-on and that is correct. Skipping the band here rather
    # than special-casing each channel is what keeps this generic.
    apply_band = engine_running or not band.get("running")

    if apply_band:
        v = r.value
        crit_low, crit_high = band.get("crit_low"), band.get("crit_high")
        warn_low, warn_high = band.get("warn_low"), band.get("warn_high")
        if crit_low is not None and v <= crit_low:
            return "CRITICAL", f"{_fmt(v, spec)} against a {_fmt(crit_low, spec)} floor"
        if crit_high is not None and v >= crit_high:
            return "CRITICAL", f"{_fmt(v, spec)} against a {_fmt(crit_high, spec)} limit"
        if warn_low is not None and v <= warn_low:
            return "WARNING", f"{_fmt(v, spec)} against a {_fmt(warn_low, spec)} floor"
        if warn_high is not None and v >= warn_high:
            return "WARNING", f"{_fmt(v, spec)} against a {_fmt(warn_high, spec)} limit"

    for label, low, high in config.TELEMETRY_MODES.get(_cfg_key(spec.id), ()):
        if low is not None and r.value < low:
            continue
        if high is not None and r.value >= high:
            continue
        return label, ""

    return "NORMAL", ""


def _trend(spec: SensorSpec, status: str, value: Optional[float]) -> str:
    """-> up | down | stable | abnormal | none.

    `abnormal` is not "the value is bad" — the status column already says that.
    It is "the value is bad *and still moving the wrong way*", which is the one
    combination that means the situation is not under control. A coolant
    temperature at 228°F and falling and a coolant temperature at 228°F and
    climbing call for different decisions, and the arrow is the only place on
    the row where that difference appears.
    """
    if status in QUIET_STATUSES or value is None:
        return "none"
    # A parked channel has no meaningful direction. See PARKED_STATUSES.
    if status in PARKED_STATUSES:
        return "none"

    change = _slope_per_window(spec.id)
    if change is None:
        return "none"

    delta = config.TELEMETRY_TREND_DELTA.get(_cfg_key(spec.id))
    # A channel with no delta configured never claims a direction. Silence is
    # the right default for a sensor nobody has decided a meaningful step for.
    if delta is None or abs(change) < delta:
        return "stable"

    direction = "up" if change > 0 else "down"

    if status in ("WARNING", "CRITICAL"):
        band = _band(spec.id)
        high = band.get("warn_high") if band.get("warn_high") is not None else band.get("crit_high")
        low = band.get("warn_low") if band.get("warn_low") is not None else band.get("crit_low")
        over = high is not None and value >= high
        under = low is not None and value <= low
        if (over and direction == "up") or (under and direction == "down"):
            return "abnormal"
    return direction


# ---------------------------------------------------------------------------
# Formatting. Every string the dashboard prints is made here.
# ---------------------------------------------------------------------------

def _fmt(value: Optional[float], spec: SensorSpec) -> str:
    if value is None:
        return "--"
    return f"{value:.{spec.precision}f}"


def _clock(ts: Optional[float]) -> str:
    return "--:--:--" if not ts else datetime.fromtimestamp(ts).strftime("%H:%M:%S")


def _runtime_text(seconds: float) -> str:
    if seconds <= 0:
        return "00:00"
    m, s = divmod(int(seconds), 60)
    h, m = divmod(m, 60)
    return f"{h}:{m:02d}:{s:02d}" if h else f"{m:02d}:{s:02d}"


# ---------------------------------------------------------------------------
# Engine runtime
# ---------------------------------------------------------------------------

class _Runtime:
    """How long the engine has been running this time.

    Wall-clock accumulation while rpm is above the running threshold, reset the
    moment it drops. It is one of the three things the spec puts in the banner,
    and it is the one that makes the other two legible: "last update 14:22:07"
    means something quite different on an engine that has been running for two
    hours than on one that started forty seconds ago.
    """

    def __init__(self):
        self._lock = threading.Lock()
        self._started_at: Optional[float] = None

    def update(self, running: bool, now: float) -> float:
        with self._lock:
            if not running:
                self._started_at = None
                return 0.0
            if self._started_at is None:
                self._started_at = now
            return max(0.0, now - self._started_at)

    def peek(self, running: bool, now: float) -> float:
        """The same answer update() would give, without starting the clock.

        A read-only snapshot must not be able to decide that the engine started
        at the moment somebody asked RIO a question.
        """
        with self._lock:
            if not running or self._started_at is None:
                return 0.0
            return max(0.0, now - self._started_at)

    def reset(self) -> None:
        with self._lock:
            self._started_at = None


_runtime = _Runtime()


# ---------------------------------------------------------------------------
# Normalization
# ---------------------------------------------------------------------------

def _normalize(spec: SensorSpec, r: Optional[SensorReading],
               now: float, engine_running: bool) -> dict:
    status, detail = _classify(spec, r, now, engine_running)
    value = r.value if r else None
    trend = _trend(spec, status, value)
    severity = STATUS_SEVERITY.get(status, 0)

    return {
        # The shape the spec asks for, plus the strings that keep arithmetic out
        # of the browser.
        "id": spec.id,
        "label": spec.label,
        "group": spec.group,
        "value": round(value, 3) if value is not None else None,
        "value_text": _fmt(value, spec),
        "units": spec.unit,
        "status": status,
        "status_class": _status_class(status),
        "trend": trend,
        "trend_glyph": TREND_GLYPH.get(trend, "—"),
        "severity": severity,
        "detail": detail,
    }


# ---------------------------------------------------------------------------
# Headline
# ---------------------------------------------------------------------------

def _headline(rows: List[dict], any_reading: bool) -> Tuple[str, str]:
    """-> (status, status_text). Worst thing on the car, in three words.

    The spec's three bands exactly: blue normal, amber attention, red immediate.
    A channel that has gone quiet is neither — it is the absence of information,
    and saying ATTENTION RECOMMENDED about a wire that has fallen off points the
    driver at the engine when the thing to look at is the loom.
    """
    if not any_reading:
        return "UNAVAILABLE", "Telemetry Unavailable"

    worst = max((r["severity"] for r in rows), default=0)
    if worst >= STATUS_SEVERITY["CRITICAL"]:
        return "CRITICAL", "Immediate Issue"
    if worst >= STATUS_SEVERITY["WARNING"]:
        return "ATTENTION", "Attention Recommended"
    if worst >= STATUS_SEVERITY["STALE"]:
        quiet = [r for r in rows if r["status"] in QUIET_STATUSES]
        if len(quiet) == len(rows):
            return "UNAVAILABLE", "Telemetry Unavailable"
        return "ATTENTION", "Sensor Not Reporting"
    if worst >= STATUS_SEVERITY["SENSOR LOW"]:
        return "ATTENTION", "Attention Recommended"
    return "NORMAL", "All Systems Normal"


# ---------------------------------------------------------------------------
# The one thing app.py calls
# ---------------------------------------------------------------------------

def snapshot(record: bool = True) -> dict:
    """Everything the Vehicle Health column renders, in the shape it renders it.

    Nothing in here requires the caller to know which provider produced what,
    and nothing downstream is allowed to compute anything from it.

    `record=False` makes this a pure read.

    This is not a micro-optimisation, it is a correctness fix for a second
    caller. A snapshot normally has three side effects: it pushes every value
    into the 20 s trend ring, it advances the engine runtime clock, and it hands
    a frame to the insight engine. Those are all correct ONCE PER POLL, at the
    dashboard's 1 Hz cadence, because they are what makes a trend a trend.

    vehicle_health.py reads this on a conversation turn and on the announcement
    poll — cadences that have nothing to do with the sample rate. Left recording,
    a driver who asked three questions in a row would fit a slope across nine
    samples of a 20 s window, and the arrow on the panel would report a direction
    caused by having been asked about. So the conversation layer reads with
    record=False and observes without disturbing.
    """
    now = time.time()
    readings: Dict[str, SensorReading] = {}
    live_providers = 0

    # An in-process producer is advanced HERE, and only on a recording snapshot.
    # Same discipline as the trend ring and the insight engine below, and for
    # the same reason: a conversation turn reads this at whatever cadence the
    # driver is talking, and a simulator advanced by being asked about would
    # report a drive whose length depended on how chatty somebody was.
    #
    # Never fatal. A producer that throws is a producer that has nothing to say
    # this tick; the panel reports what it has rather than going down.
    if record:
        pump = SOURCE_BY_NAME[_source_name].pump
        if pump is not None:
            try:
                pump(now)
            except Exception as e:
                print(f"[telemetry] producer {_source_name} tick failed: "
                      f"{type(e).__name__}: {e}", flush=True)

    for p in _providers:
        try:
            if not p.available():
                continue
            live_providers += 1
            for r in (p.read() or []):
                if r.id in SPEC_BY_ID:
                    readings[r.id] = r
        except Exception as e:
            # A provider that throws is a provider that is not there. The panel
            # says so; it does not take the dashboard down with it.
            print(f"[telemetry] provider {p.name} read failed: "
                  f"{type(e).__name__}: {e}", flush=True)

    rpm_reading = readings.get("rpm")
    engine_running = bool(rpm_reading and rpm_reading.ok
                          and rpm_reading.value is not None
                          and rpm_reading.value >= config.TELEMETRY_ENGINE_RUNNING_RPM)

    if record:
        for sensor_id, r in readings.items():
            _history.push(sensor_id, r.value, r.at or now)

    rows = [_normalize(spec, readings.get(spec.id), now, engine_running)
            for spec in ALL_SENSORS]

    any_reading = any(r["value"] is not None for r in rows)
    status, status_text = _headline(rows, any_reading)

    stamps = [r.at for r in readings.values() if r.at]
    newest = max(stamps) if stamps else None
    age = (now - newest) if newest else None
    stale = age is not None and age > config.TELEMETRY_STALE_AFTER_S

    runtime_s = _runtime.update(engine_running, now) if record \
        else _runtime.peek(engine_running, now)

    # Hand the frame to the insight engine on the telemetry cadence rather than
    # on the insights cadence. The baselines want every sample; the log wants to
    # be read once every fifteen seconds. Those are different jobs and they get
    # different clocks.
    if record:
        insights.observe(_frame(rows, readings, engine_running, now))

    ecu = _ecu()
    return {
        "provider": ecu.name if ecu else "none",
        "provider_label": ecu.label if ecu else "No Provider",
        # Which producer the pipeline is listening to, and what else it could
        # listen to. The interpretation below this line is identical for every
        # one of them — see the SOURCES table.
        "source": _source_name,
        "source_label": SOURCE_BY_NAME[_source_name].label,
        "sources": sources(),
        "available": live_providers > 0 and any_reading,
        "status": status,
        "status_text": status_text,
        # The three things the spec puts in the banner.
        "connection": "Live" if (live_providers and not stale and any_reading) else "No Link",
        "connection_state": "live" if (live_providers and not stale and any_reading) else "lost",
        "updated_at": newest,
        "updated_display": _clock(newest) if newest else "No Data",
        "runtime_display": _runtime_text(runtime_s) if engine_running else "Engine Off",
        "runtime_s": round(runtime_s, 1),
        "engine_running": engine_running,
        "stale": stale,
        "groups": [g for g in GROUP_ORDER if any(r["group"] == g for r in rows)],
        "rows": rows,
        # The browser takes its cadence from the server so the poll rate is one
        # number in one place like every other tunable.
        "poll_ms": config.TELEMETRY_POLL_MS,
        "scenario": current_scenario(),
        "scenarios": scenarios(),
        # The tire mock keeps its own scenario list. Both selectors are dev-only
        # and both disappear the moment a provider with no scenarios is live.
        "tire_scenario": tires.current_scenario(),
        "tire_scenarios": tires.scenarios(),
    }


def _frame(rows: List[dict], readings: Dict[str, SensorReading],
           engine_running: bool, now: float) -> dict:
    """What insights.py is given. Numbers and labels, never sentences.

    Deliberately not the row list: rows carry formatted strings and CSS class
    names, and an insight engine that reads those would be parsing the UI back
    out of itself. It gets the values and the metadata it needs to phrase them,
    and nothing else.
    """
    values = {}
    statuses = {}
    meta = {}
    for row in rows:
        values[row["id"]] = row["value"]
        statuses[row["id"]] = row["status"]
        spec = SPEC_BY_ID[row["id"]]
        meta[row["id"]] = {"label": spec.label, "unit": spec.unit,
                           "precision": spec.precision}
    return {"at": now, "engine_running": engine_running,
            "values": values, "statuses": statuses, "meta": meta}


# ---------------------------------------------------------------------------
# Dev scenario control
# ---------------------------------------------------------------------------

def current_scenario() -> Optional[str]:
    ecu = _ecu()
    return getattr(ecu, "scenario", None) if ecu else None


def scenarios() -> List[dict]:
    """What the active source can be told to do. Empty for every real one.

    Asked of the provider rather than assumed of the class: a live bridge has
    exactly one scenario, which is whatever the engine is actually doing, and
    the selector removes itself from the panel when this list is empty.
    """
    ecu = _ecu()
    fn = getattr(ecu, "scenarios", None) if ecu else None
    return fn() if callable(fn) else []


def set_scenario(name: str) -> bool:
    """-> True if it took. False on a provider with no scenarios (real hardware
    has exactly one scenario, which is whatever the engine is actually doing).

    Clears the trend history: a slope fitted across the moment the scenario
    changed is a slope across a discontinuity, and it would show a spectacular
    arrow on a channel that simply jumped.
    """
    ecu = _ecu()
    setter = getattr(ecu, "set_scenario", None) if ecu else None
    if not callable(setter) or not setter(name):
        return False
    _history.reset()
    _runtime.reset()
    return True


def set_tire_scenario(name: str) -> bool:
    """Switch the tire mock, and forget the trend window while doing it.

    tires.set_scenario() on its own is not enough now that the tire channels are
    rows in this list: jumping a corner from 25.6 to 33.1 PSI leaves a ring
    holding both, and the fit across that step reports a dramatic direction for
    a tire that did not move at all. Same reason set_scenario() above resets —
    a slope across a discontinuity is not a trend.
    """
    if not tires.set_scenario(name):
        return False
    _history.reset()
    return True
