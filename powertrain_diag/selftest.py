"""Verification for the engine monitors.

    python -m powertrain_diag.selftest

diag/selftest.py already proves the framework is generic and
tire_diag/selftest.py proves the discipline holds. What this adds is the nine
engine monitors: that each one fires on the condition it is for, that each one
DECLINES on the conditions it cannot judge, and — the one that matters most —
that a learned baseline can never relax a fixed limit.

     1  the hard coolant limit fires on a sustained overheat
     2  ...and not on a single spike
     3  ...and fires anyway with an absurd learned baseline (§5.5)
     4  rate of rise catches a fast climb the ceiling has not seen yet
     5  ...and is INHIBITED during warm-up, where a fast climb is normal
     6  the contextual monitor finds what every fixed band passes
     7  ...and is INHIBITED below cruising speed
     8  charging voltage fires running, and is INHIBITED stopped
     9  start voltage needs several starts, then reports the decline
    10  fuel trim fires warm and in closed loop
    11  ...and is INHIBITED cold, and INHIBITED under power
    12  a frozen channel is caught, though every range check passes
    13  no data at all is one link finding, not nine engine faults
    14  the DTC monitor reports the vehicle's codes without judging them
    15  the lifecycle is the framework's, and survives a restart
    16  shadow mode is per domain, and this domain has no fast path

Plus the LLM firewall. No GPU, no models, no network, ~2 s.

The harness builds telemetry snapshots straight from vehicle/producers/physics.py
rather than going through telemetry.snapshot(). That is deliberate: these are
unit tests of nine monitors, and driving them through the classification layer
would make a band change in config.py look like a monitor regression.
"""
import ast
import os
import shutil
import sys
import tempfile

sys.path.insert(0, "/workspace/rio-phase1")

import config                                        # noqa: E402

import insights                                      # noqa: E402
from vehicle.producers import physics                # noqa: E402

from . import store                                  # noqa: E402

_TMP = tempfile.mkdtemp(prefix="powertrain_selftest_")
store.reset_for_test(_TMP)

from diag import monitors as DM                      # noqa: E402
from diag import shadow                              # noqa: E402

from . import codes as C                             # noqa: E402
from . import engine as E                            # noqa: E402
from . import monitors as M                          # noqa: E402

_results = []

# Baselines the monitors are handed. Swapped in wholesale so a test can state
# exactly what this car is supposed to normally do, rather than depending on
# whatever four weeks of seeded history happens to contain.
_BASELINES = {}


def _fake_baseline(key, now=None):
    if key in _BASELINES:
        return _BASELINES[key], 30, False
    return None, 0, False


insights.baseline = _fake_baseline


def check(condition, label, detail=""):
    _results.append((bool(condition), label, detail))
    print(f"  [{'PASS' if condition else 'FAIL'}] {label}"
          + (f"  -- {detail}" if detail else ""))
    return bool(condition)


def head(title):
    print(f"\n{title}")


# ---------------------------------------------------------------------------
# Harness
# ---------------------------------------------------------------------------

class Car:
    """An engine being driven, in synthetic time.

    Wall-clock is useless here: a fuel-trim monitor needs two minutes of warm
    closed-loop running and a healing rule needs half an hour, so a real-time
    test would take an afternoon. Every clock in this stack takes its time from
    the caller precisely so this is possible.
    """

    def __init__(self, scenario="cruise", t0=1_700_400_000.0, load=False):
        store.reset_for_test(_TMP)
        self.engine = E.reset_engine(load=load)
        self.t = t0
        self.scenario = physics.BY_NAME[scenario]
        self.elapsed = 0.0
        self.previous = {}
        self.speed = 62.0
        self.dtc = {}
        self.link = {}

    def set_scenario(self, name):
        self.scenario = physics.BY_NAME[name]
        self.elapsed = 0.0
        self.previous = {}
        return self

    def snapshot(self):
        values, dropped = physics.sample(self.scenario, self.elapsed,
                                         self.previous)
        self.previous = dict(values)
        rows = []
        for channel in E._CHANNELS:
            if channel in dropped:
                rows.append({"id": channel, "value": None, "status": "OFFLINE"})
                continue
            if channel not in values:
                # Absent because this vehicle does not expose it. Not a fault.
                continue
            v = values[channel]
            rows.append({"id": channel, "value": v,
                         "status": "NORMAL" if v is not None else "NO DATA"})
        rpm = values.get("rpm") or 0.0
        return {"rows": rows,
                "engine_running": rpm >= config.TELEMETRY_ENGINE_RUNNING_RPM}

    def step(self, dt=2.0, n=1, feed=True):
        for _ in range(n):
            self.t += dt
            self.elapsed += dt
            snap = self.snapshot() if feed else {"rows": []}
            self.engine.observe(snap, now=self.t, moving=self.speed > 5,
                                speed_mph=self.speed, dtc=self.dtc,
                                link=self.link)
        return self

    def run_for(self, seconds, dt=2.0, feed=True):
        return self.step(dt, n=max(1, int(round(seconds / dt))), feed=feed)

    def monitor(self, monitor_id):
        for m in self.engine.monitor_view():
            if m["monitor_id"] == monitor_id:
                return m
        return None

    def issue(self, monitor_id):
        for i in self.engine.issues():
            if i["monitor_id"] == monitor_id:
                return i
        return None


def _clear_baselines():
    _BASELINES.clear()


# ---------------------------------------------------------------------------
# 1-3. The fixed coolant ceiling
# ---------------------------------------------------------------------------

def test_01_hard_limit():
    head("1 -- the fixed coolant limit fires on a sustained overheat")
    _clear_baselines()
    car = Car("overheating").run_for(120)
    m = car.monitor("engine.coolant_hard_limit")
    check(m["last_result"] == DM.FAILED_PENDING, "the monitor failed",
          str(m["last_result"]))
    iss = car.issue("engine.coolant_hard_limit")
    check(iss is not None and iss["lifecycle"] == E.ACTIVE,
          "and it confirmed to an ACTIVE issue",
          iss["lifecycle"] if iss else "missing")
    check(iss and iss["severity"] == "critical", "at critical severity",
          iss["severity"] if iss else "")
    frames = (iss or {}).get("freeze_frames") or []
    check(frames, "with freeze-frame evidence")
    if frames:
        f = frames[0]
        check(f.get("coolant_temp") is not None and f.get("limit_f") is not None,
              "carrying the reading and the limit it was judged against",
              f"{f.get('coolant_temp')} vs {f.get('limit_f')}")
        check(f.get("vehicle_speed") is not None,
              "and the context that makes it readable three weeks later")


def test_01b_healthy_cruise_is_quiet():
    head("1b -- a healthy drive produces no findings at all")
    _BASELINES.clear()
    _BASELINES["coolant_temp@cruise"] = 195.0
    car = Car("cruise")
    car.dtc = {"scanned": True, "responding": True, "codes": [], "mil": False,
               "count": 0}
    car.run_for(400, dt=2.0)

    raised = [i["code"] for i in car.engine.issues()]
    check(not raised,
          "ten minutes of ordinary cruising raises nothing — a monitor set that "
          "cries wolf on a healthy car is one a driver learns to ignore, and "
          "that is the failure mode none of the individual tests would catch",
          str(raised))

    judged = [m for m in car.engine.monitor_view() if m["last_result"] is not None]
    check(len(judged) >= 5,
          "while at least five monitors actually reached a verdict — silence "
          "because nothing ran is not the same as silence because nothing is "
          "wrong", f"{len(judged)} of {len(car.engine.monitor_view())}")
    passed = [m["monitor_id"] for m in judged if m["last_result"] == DM.PASSED]
    check(len(passed) == len(judged), "and every verdict was a pass",
          str([m["monitor_id"] for m in judged if m["last_result"] != DM.PASSED]))


def test_02_no_fire_on_a_spike():
    head("2 -- ...and not on a single reading past the limit")
    _clear_baselines()
    car = Car("cruise").run_for(60)
    # One absurd sample, then back to normal. The hold is what separates an
    # engine from a sensor, and it is the whole reason this monitor is not just
    # the panel's own row.
    car.engine.observe({"rows": [{"id": "coolant_temp", "value": 250.0,
                                  "status": "NORMAL"},
                                 {"id": "rpm", "value": 1850.0,
                                  "status": "NORMAL"}],
                        "engine_running": True},
                       now=car.t + 2, moving=True, speed_mph=62.0)
    car.t += 2
    car.run_for(30)
    m = car.monitor("engine.coolant_hard_limit")
    check(m["last_result"] == DM.PASSED,
          "one reading past the limit is not sustained, so the monitor passes",
          str(m["last_result"]))
    check(car.issue("engine.coolant_hard_limit") is None,
          "and no issue was created at all")


def test_03_baseline_cannot_relax_a_fixed_limit():
    head("3 -- §5.5: a learned baseline may never relax a fixed limit")
    # A baseline claiming this car normally runs at 250°F. If the fixed limit
    # consulted history at all, this would silence a genuine overheat — which is
    # the single most dangerous thing a learning system can do.
    _BASELINES.clear()
    _BASELINES["coolant_temp@cruise"] = 250.0
    _BASELINES["coolant_temp"] = 250.0
    car = Car("overheating").run_for(120)
    m = car.monitor("engine.coolant_hard_limit")
    check(m["last_result"] == DM.FAILED_PENDING,
          "the hard limit still fires with an absurd baseline in place",
          str(m["last_result"]))
    iss = car.issue("engine.coolant_hard_limit")
    check(iss and iss["severity"] == "critical",
          "at full severity — nothing learned reduced it",
          iss["severity"] if iss else "missing")

    ctx = car.monitor("engine.coolant_contextual")
    check(ctx["last_result"] == DM.PASSED,
          "while the CONTEXTUAL monitor, which does consult the baseline, "
          "correctly finds nothing unusual against it", str(ctx["last_result"]))
    check(True, "the two monitors disagree, and that is exactly right: one "
                "reports the car is past its safe limit, the other that this is "
                "normal for this car. Both are true and only one is actionable.")

    # And the structural claim: the fixed-limit monitor does not read baselines.
    src = open(os.path.join(os.path.dirname(os.path.abspath(__file__)),
                            "monitors.py")).read()
    body = src[src.index("def _eval_coolant_limit"):src.index("def _eval_coolant_rise")]
    check("baseline" not in body,
          "and _eval_coolant_limit never touches inp.baselines at all")
    _clear_baselines()


# ---------------------------------------------------------------------------
# 4-5. Rate of rise
# ---------------------------------------------------------------------------

def test_04_rate_of_rise():
    head("4 -- rate of rise catches a climb the ceiling has not seen yet")
    _clear_baselines()
    car = Car("coolant_rapid_rise").run_for(200, dt=2.0)
    m = car.monitor("engine.coolant_rate_of_rise")
    check(m["last_result"] == DM.FAILED_PENDING, "the rate monitor failed",
          str(m["last_result"]))
    hard = car.monitor("engine.coolant_hard_limit")
    check(hard["last_result"] == DM.PASSED,
          "while the fixed ceiling has NOT been reached — which is the whole "
          "point of having a rate monitor", str(hard["last_result"]))
    iss = car.issue("engine.coolant_rate_of_rise")
    check(iss and iss["severity"] == "warning", "warning severity",
          iss["severity"] if iss else "missing")
    detail = (iss or {}).get("detail") or {}
    check(detail.get("coolant_rate_f_per_min", 0) > 0,
          "with the measured rate on the record",
          str(detail.get("coolant_rate_f_per_min")))


def test_05_warmup_is_not_a_fault():
    head("5 -- ...and is INHIBITED during warm-up, where a fast climb is normal")
    _clear_baselines()
    car = Car("warmup")
    car.speed = 0.0
    car.run_for(60, dt=2.0)
    m = car.monitor("engine.coolant_rate_of_rise")
    check(m["status"] == DM.INHIBITED,
          "the rate monitor declines to judge a warming engine", m["status"])
    check("warming up" in (m["status_reason"] or ""),
          "and says why", m["status_reason"])
    check(car.issue("engine.coolant_rate_of_rise") is None,
          "no finding is raised — without this gate it would fire on every "
          "single journey, which is the fastest way to make a driver stop "
          "reading the panel")


# ---------------------------------------------------------------------------
# 6-7. The contextual monitor
# ---------------------------------------------------------------------------

def test_06_contextual_finds_what_bands_pass():
    head("6 -- the contextual monitor finds what every fixed band passes")
    _BASELINES.clear()
    _BASELINES["coolant_temp@cruise"] = 193.0
    car = Car("cruise")
    car.run_for(60, dt=2.0)
    # A car sitting 12°F above its own history, and comfortably inside every
    # band in config.py. No fixed threshold in this codebase can see this.
    for i in range(70):
        car.t += 2
        car.engine.observe(
            {"rows": [{"id": "coolant_temp", "value": 205.0, "status": "NORMAL"},
                      {"id": "vehicle_speed", "value": 62.0, "status": "NORMAL"},
                      {"id": "rpm", "value": 1850.0, "status": "NORMAL"},
                      {"id": "engine_load", "value": 30.0, "status": "NORMAL"}],
             "engine_running": True},
            now=car.t, moving=True, speed_mph=62.0)
    m = car.monitor("engine.coolant_contextual")
    check(m["last_result"] == DM.FAILED_PENDING,
          "the contextual monitor failed", str(m["last_result"]))
    hard = car.monitor("engine.coolant_hard_limit")
    check(hard["last_result"] == DM.PASSED,
          "while the fixed ceiling passes", str(hard["last_result"]))
    warn = config.TELEMETRY_BANDS["coolant_temp"]["warn_high"]
    check(205.0 < warn,
          f"and 205°F is inside the panel's own warning band ({warn}°F) too — "
          f"nothing else in this codebase could have noticed")


def test_07_contextual_declines_off_condition():
    head("7 -- ...and is INHIBITED below cruising speed")
    _BASELINES.clear()
    _BASELINES["coolant_temp@cruise"] = 193.0
    car = Car("normal_idle")
    car.speed = 0.0
    car.run_for(60, dt=2.0)
    m = car.monitor("engine.coolant_contextual")
    check(m["status"] == DM.INHIBITED,
          "the cruise baseline does not apply to a car that is not cruising",
          m["status"])
    check("cruising speed" in (m["status_reason"] or ""), "and says so",
          m["status_reason"])

    _BASELINES.clear()
    car2 = Car("cruise").run_for(60, dt=2.0)
    m2 = car2.monitor("engine.coolant_contextual")
    check(m2["status"] == DM.NOT_READY,
          "and with no history at all it is NOT_READY, never a pass",
          m2["status"])
    check(m2["last_result"] is None,
          "with NO result — a monitor that has not judged has nothing to report")


# ---------------------------------------------------------------------------
# 8-9. Electrical
# ---------------------------------------------------------------------------

def test_08_charging():
    head("8 -- charging voltage fires running, and declines stopped")
    _clear_baselines()
    car = Car("charging_fault")
    car.speed = 0.0
    car.run_for(120, dt=2.0)
    m = car.monitor("engine.charging_voltage")
    check(m["last_result"] == DM.FAILED_PENDING, "a failing alternator is found",
          str(m["last_result"]))
    iss = car.issue("engine.charging_voltage")
    check(iss is not None, "and raises an issue")
    floor = config.TELEMETRY_BANDS["battery_voltage"]["warn_low"]
    check((iss or {}).get("detail", {}).get("floor_v") == floor,
          "judged against the panel's own floor, not a copy of it",
          f"{(iss or {}).get('detail', {}).get('floor_v')} vs {floor}")

    # Engine off: 12.6 V is a healthy resting battery, and a monitor that judged
    # it against the running floor would report a fault on every parked car.
    car2 = Car("cruise")
    car2.speed = 0.0
    for i in range(20):
        car2.t += 2
        car2.engine.observe(
            {"rows": [{"id": "rpm", "value": 0.0, "status": "NORMAL"},
                      {"id": "battery_voltage", "value": 12.6, "status": "NORMAL"}],
             "engine_running": False},
            now=car2.t, moving=False, speed_mph=0.0)
    m2 = car2.monitor("engine.charging_voltage")
    check(m2["status"] == DM.INHIBITED,
          "a stopped engine inhibits the charging monitor", m2["status"])
    check(car2.issue("engine.charging_voltage") is None,
          "so a parked car reports no charging fault")


def test_09_start_voltage():
    head("9 -- start voltage needs several starts, then reports the decline")
    _clear_baselines()
    car = Car("start_voltage_decline")
    car.speed = 0.0
    car.run_for(120, dt=2.0)
    m = car.monitor("engine.start_voltage_trend")
    check(m["status"] == DM.NOT_READY,
          "one or two starts is not a trend", m["status"])
    check(m["last_result"] is None, "and produces no result")

    car.run_for(400, dt=2.0)
    events = car.engine._state["meta"]["start_events"]
    check(len(events) >= config.POWERTRAIN_START_EVENTS_MIN,
          f"after several drives there are "
          f"{config.POWERTRAIN_START_EVENTS_MIN}+ recorded starts",
          str(len(events)))
    m = car.monitor("engine.start_voltage_trend")
    check(m["last_result"] == DM.FAILED_PENDING,
          "and the decline is reported", str(m["last_result"]))

    charging = car.monitor("engine.charging_voltage")
    check(charging["last_result"] == DM.PASSED,
          "while the RUNNING voltage monitor passes throughout — this is the "
          "measurement a running-voltage band can never make",
          str(charging["last_result"]))

    # Persisted, because its value is across drives. See engine.py's header on
    # why this is not the same decision as persisting samples.
    reborn = E.reset_engine(load=True)
    check(len(reborn._state["meta"]["start_events"]) >= 4,
          "and the start history survives a restart",
          str(len(reborn._state["meta"].get("start_events") or [])))


# ---------------------------------------------------------------------------
# 10-11. Fuel trim
# ---------------------------------------------------------------------------

def test_10_fuel_trim():
    head("10 -- fuel trim fires warm and in closed loop")
    _clear_baselines()
    car = Car("fuel_trim_drift")
    car.speed = 0.0
    car.run_for(600, dt=2.0)
    m = car.monitor("engine.fuel_trim_long_term")
    check(m["last_result"] == DM.FAILED_PENDING, "a drifting trim is found",
          str(m["last_result"]))
    iss = car.issue("engine.fuel_trim_long_term")
    check(iss and iss["severity"] == "advisory",
          "at advisory severity — real, on the dashboard, and never worth "
          "interrupting a drive for", iss["severity"] if iss else "missing")
    detail = (iss or {}).get("detail") or {}
    check(detail.get("ltft_b1") is not None and detail.get("stft_b1") is not None,
          "with both trims recorded, because they mean different things",
          f"ltft={detail.get('ltft_b1')} stft={detail.get('stft_b1')}")


def test_11_fuel_trim_gates():
    head("11 -- ...and is INHIBITED cold, and INHIBITED under power")
    _clear_baselines()
    cold = Car("warmup")
    cold.speed = 0.0
    cold.run_for(40, dt=2.0)
    m = cold.monitor("engine.fuel_trim_long_term")
    check(m["status"] == DM.INHIBITED, "a cold engine is not trimming yet",
          m["status"])
    check("warm" in (m["status_reason"] or ""), "and says so", m["status_reason"])

    # Wide-open throttle, where the ECU stops trimming and reports zero.
    loaded = Car("cruise")
    loaded.speed = 80.0
    for i in range(30):
        loaded.t += 2
        loaded.engine.observe(
            {"rows": [{"id": "coolant_temp", "value": 198.0, "status": "NORMAL"},
                      {"id": "throttle_pct", "value": 88.0, "status": "NORMAL"},
                      {"id": "ltft_b1", "value": 0.0, "status": "NORMAL"},
                      {"id": "rpm", "value": 4200.0, "status": "NORMAL"}],
             "engine_running": True},
            now=loaded.t, moving=True, speed_mph=80.0)
    m2 = loaded.monitor("engine.fuel_trim_long_term")
    check(m2["status"] == DM.INHIBITED,
          "an engine in open loop under power is not trimming either — reading "
          "the zero it reports there as a healthy trim would be calling a "
          "switched-off system a passing one", m2["status"])
    check("open loop" in (m2["status_reason"] or ""), "and says so",
          m2["status_reason"])


# ---------------------------------------------------------------------------
# 12-13. Integrity and the link
# ---------------------------------------------------------------------------

def test_12_frozen_channel():
    head("12 -- a frozen channel is caught, though every range check passes")
    _clear_baselines()
    car = Car("frozen_signal").run_for(200, dt=2.0)
    m = car.monitor("engine.signal_integrity")
    check(m["last_result"] == DM.FAILED_PENDING, "the stuck sensor is found",
          str(m["last_result"]))
    iss = car.issue("engine.signal_integrity")
    detail = (iss or {}).get("detail") or {}
    check("coolant_temp" in (detail.get("frozen") or []),
          "and it names the channel", str(detail.get("frozen")))

    band = config.TELEMETRY_BANDS["coolant_temp"]
    frozen_at = detail.get("coolant_temp")
    check(frozen_at is None or frozen_at < band["warn_high"],
          "the frozen value is INSIDE its band — every plausibility check "
          "passes and the reading is completely false, which is why this is the "
          "nastiest sensor failure there is", str(frozen_at))

    hard = car.monitor("engine.coolant_hard_limit")
    check(hard["last_result"] == DM.PASSED,
          "and the coolant monitors are cheerfully passing on it, which is "
          "exactly the situation the integrity monitor exists to flag",
          str(hard["last_result"]))


def test_13_no_data_is_one_finding():
    head("13 -- no data at all is one link finding, not nine engine faults")
    _clear_baselines()
    car = Car("cruise").run_for(60, dt=2.0)
    check(car.monitor("engine.connection")["last_result"] == DM.PASSED,
          "data is flowing to begin with")

    # The bridge goes quiet. Time passes, and nothing arrives.
    for i in range(8):
        car.t += 20
        car.engine.observe({"rows": []}, now=car.t, moving=True, speed_mph=62.0,
                           link={"source": "live_obd", "can_state": "active"})

    conn = car.monitor("engine.connection")
    check(conn["last_result"] == DM.FAILED_PENDING,
          "the connection monitor reports the silence", str(conn["last_result"]))
    others = [m for m in car.engine.monitor_view()
              if m["monitor_id"] not in ("engine.connection", "engine.new_dtc")
              and m["last_result"] == DM.FAILED_PENDING
              and m["monitor_id"] != "engine.signal_integrity"]
    inhibited = [m for m in car.engine.monitor_view()
                 if m["status"] == DM.INHIBITED]
    check(inhibited, "and the engine monitors are INHIBITED rather than failing",
          str(len(inhibited)))
    check(any("no engine data" in (m["status_reason"] or "")
              for m in car.engine.monitor_view()),
          "naming the link as the reason")

    issues = [i for i in car.engine.issues()
              if i["monitor_id"] != "engine.connection"]
    check(not [i for i in issues if i["lifecycle"] == E.ACTIVE],
          "no engine fault is invented out of the silence",
          str([i["code"] for i in issues if i["lifecycle"] == E.ACTIVE]))

    # A bridge that is measuring and cannot upload is a different finding.
    car2 = Car("cruise").run_for(60, dt=2.0)
    car2.link = {"source": "live_obd", "outbox_pending": 900,
                 "network_state": "disconnected"}
    car2.run_for(30, dt=2.0)
    m = car2.monitor("engine.connection")
    check(m["last_result"] == DM.FAILED_PENDING,
          "a bridge holding a large outbox is reported too",
          str(m["last_result"]))


# ---------------------------------------------------------------------------
# 14. The DTC monitor
# ---------------------------------------------------------------------------

def test_14_dtc_monitor():
    head("14 -- the DTC monitor reports the vehicle's codes without judging them")
    _clear_baselines()
    car = Car("cruise").run_for(30, dt=2.0)
    m = car.monitor("engine.new_dtc")
    check(m["status"] == DM.NOT_READY,
          "with no scan completed it is NOT_READY, not a pass", m["status"])

    car.dtc = {"scanned": True, "responding": True, "codes": [], "mil": False,
               "count": 0}
    car.run_for(20, dt=2.0)
    check(car.monitor("engine.new_dtc")["last_result"] == DM.PASSED,
          "a scan reporting nothing is a pass")

    car.dtc = {"scanned": True, "responding": True, "codes": ["P0171"],
               "added": ["P0171"], "mil": False, "count": 0,
               "worst_health_severity": "advisory"}
    car.run_for(30, dt=2.0)
    m = car.monitor("engine.new_dtc")
    check(m["last_result"] == DM.FAILED_PENDING, "a reported code is a finding",
          str(m["last_result"]))
    iss = car.issue("engine.new_dtc")
    check(iss and iss["lifecycle"] == E.ACTIVE,
          "confirmed on a single scan, because the ECU has already done the "
          "confirming", iss["lifecycle"] if iss else "missing")
    check(iss and iss["severity"] == "advisory",
          "at the severity the DTC catalogue translated, not one invented here",
          iss["severity"] if iss else "")
    check("P0171" in (iss or {}).get("reason", ""),
          "naming the code", (iss or {}).get("reason", "")[:60])

    car.dtc = {"scanned": True, "responding": False}
    car.run_for(20, dt=2.0)
    check(car.monitor("engine.new_dtc")["status"] == DM.DATA_UNAVAILABLE,
          "an ECU that stops answering is DATA_UNAVAILABLE, never a pass — "
          "'no codes' and 'no answer' are opposite facts",
          car.monitor("engine.new_dtc")["status"])


# ---------------------------------------------------------------------------
# 15-16. Framework and shadow
# ---------------------------------------------------------------------------

def test_15_lifecycle_is_the_frameworks():
    head("15 -- the lifecycle is the framework's, and survives a restart")
    _clear_baselines()
    car = Car("overheating")
    car.run_for(20, dt=2.0)
    iss = car.issue("engine.coolant_hard_limit")
    check(iss is None or iss["lifecycle"] == E.CANDIDATE,
          "one failing run is a CANDIDATE, not an ACTIVE issue",
          (iss or {}).get("lifecycle", "none yet"))

    car.run_for(120, dt=2.0)
    iss = car.issue("engine.coolant_hard_limit")
    check(iss and iss["lifecycle"] == E.ACTIVE, "repeated failures confirm it",
          (iss or {}).get("lifecycle", "missing"))

    reborn = E.reset_engine(load=True)
    after = [i for i in reborn.issues(E.ACTIVE)
             if i["monitor_id"] == "engine.coolant_hard_limit"]
    check(after, "and it survives a restart")
    rows = [m for m in reborn.monitor_view()
            if m["monitor_id"] == "engine.coolant_hard_limit"]
    check(rows and rows[0]["status"] == DM.NOT_READY,
          "with the monitor NOT_READY — the results came off disk, the evidence "
          "did not", rows[0]["status"] if rows else "")
    check(rows and rows[0]["last_result"] == DM.FAILED_PENDING,
          "while what it FOUND survives, because that is a fact about the past",
          str(rows[0]["last_result"]) if rows else "")

    # Healing, which is the framework's and is not reimplemented here.
    car2 = Car("overheating").run_for(120, dt=2.0)
    check(car2.issue("engine.coolant_hard_limit")["lifecycle"] == E.ACTIVE,
          "an issue is ACTIVE before healing")
    car2.set_scenario("cruise")
    car2.run_for(config.POWERTRAIN_HEAL_STABLE_S["engine.coolant_hard_limit"] * 2,
                 dt=10.0)
    done = car2.issue("engine.coolant_hard_limit")
    check(done["lifecycle"] == E.RESOLVED,
          "and resolves only after its configured healing criteria pass",
          done["lifecycle"])
    check(done.get("freeze_frames"),
          "with its evidence retained after resolution")


def test_16_shadow_is_per_domain():
    head("16 -- shadow mode is per domain, and this domain has no fast path")
    check(config.VEHICLE_DIAG_SHADOW_MODE is True,
          "the powertrain domain ships shadowed")
    check(shadow.is_shadowed(C.DOMAIN) is True, "and reads as shadowed")
    speaking = [c for c in C.CODES.values() if c.speak]
    check(not speaking, "no engine code is cleared to speak",
          str([c.code for c in speaking]))
    fast = [c for c in C.CODES.values() if c.fast_path]
    check(not fast,
          "and NONE has a fast path — there is no engine condition here whose "
          "consequence is unrecoverable within the seconds a fast path saves, "
          "and the two tire codes that do have one earned it by argument",
          str([c.code for c in fast]))

    # Two domains, two flags, and clearing one must not clear the other.
    check(shadow.is_shadowed("tires") is True,
          "the tire domain is shadowed too, independently")
    saved = config.VEHICLE_DIAG_SHADOW_MODE
    config.VEHICLE_DIAG_SHADOW_MODE = False
    check(shadow.is_shadowed(C.DOMAIN) is False
          and shadow.is_shadowed("tires") is True,
          "clearing the powertrain domain leaves the tire domain shadowed — "
          "which is the whole reason clearance stopped being one boolean")
    config.VEHICLE_DIAG_SHADOW_MODE = saved

    # Separate stores, so the two domains cannot interleave findings.
    from tire_diag import store as tire_store
    check(store.paths()["state"] != tire_store.paths()["state"],
          "and the two domains write to separate state files",
          f"{os.path.basename(store.paths()['state'])} vs "
          f"{os.path.basename(tire_store.paths()['state'])}")


# ---------------------------------------------------------------------------
# LLM firewall
# ---------------------------------------------------------------------------

FORBIDDEN = ["openai", "llm_interface", "vision", "visual_qa", "app",
             "requests", "httpx", "torch", "transformers", "voice",
             "vehicle_health", "vehicle_health_policy"]


def test_firewall():
    head("F -- no monitor, lifecycle or speech decision can reach a model")
    root = os.path.dirname(os.path.abspath(__file__))
    for name in ("monitors.py", "engine.py", "codes.py", "store.py"):
        tree = ast.parse(open(os.path.join(root, name)).read())
        imported = set()
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                imported.update(a.name.split(".")[0] for a in node.names)
            elif isinstance(node, ast.ImportFrom):
                if node.module:
                    imported.add(node.module.split(".")[0])
                imported.update(a.name for a in node.names
                                if node.level and not node.module)
        bad = sorted(imported & set(FORBIDDEN))
        check(not bad, f"{name} imports nothing that could reach a model",
              str(bad))

    src = open(os.path.join(root, "monitors.py")).read()
    tree = ast.parse(src)
    docstrings = set()
    for node in ast.walk(tree):
        body = getattr(node, "body", None)
        if isinstance(node, (ast.Module, ast.ClassDef, ast.FunctionDef)) and body \
                and isinstance(body[0], ast.Expr) \
                and isinstance(body[0].value, ast.Constant) \
                and isinstance(body[0].value.value, str):
            docstrings.add(id(body[0].value))
    surface = []
    for node in ast.walk(tree):
        if isinstance(node, ast.Name):
            surface.append(node.id)
        elif isinstance(node, ast.Attribute):
            surface.append(node.attr)
        elif isinstance(node, ast.Constant) and isinstance(node.value, str) \
                and id(node) not in docstrings:
            surface.append(node.value)
    text = " ".join(surface).lower()
    for word in ("gpt", "prompt", "completion", "openai", "llm"):
        check(word not in text,
              f"monitors.py never mentions {word!r} in executable code")

    # And the read-only posture, in this domain too.
    for name in ("monitors.py", "engine.py", "codes.py"):
        body = open(os.path.join(root, name)).read().lower()
        for token in ("mode_04", "clear_dtc", "clear_codes", "actuator_test",
                      "ecu_write"):
            check(token not in body, f"{name} contains no {token!r}")


# ---------------------------------------------------------------------------

def main():
    print("=" * 74)
    print("RIO powertrain diagnostics — nine engine monitors, shadow mode")
    print("=" * 74)
    print(f"  store: {_TMP}")

    tests = [
        test_01_hard_limit,
        test_01b_healthy_cruise_is_quiet,
        test_02_no_fire_on_a_spike,
        test_03_baseline_cannot_relax_a_fixed_limit,
        test_04_rate_of_rise,
        test_05_warmup_is_not_a_fault,
        test_06_contextual_finds_what_bands_pass,
        test_07_contextual_declines_off_condition,
        test_08_charging,
        test_09_start_voltage,
        test_10_fuel_trim,
        test_11_fuel_trim_gates,
        test_12_frozen_channel,
        test_13_no_data_is_one_finding,
        test_14_dtc_monitor,
        test_15_lifecycle_is_the_frameworks,
        test_16_shadow_is_per_domain,
        test_firewall,
    ]
    for t in tests:
        try:
            t()
        except Exception as e:
            import traceback
            check(False, f"{t.__name__} raised", f"{type(e).__name__}: {e}")
            traceback.print_exc()

    passed = sum(1 for ok, _, _ in _results if ok)
    total = len(_results)
    print("\n" + "=" * 74)
    print(f"{passed}/{total} checks passed")
    if passed != total:
        print("\nFAILURES:")
        for ok, name, detail in _results:
            if not ok:
                print(f"  - {name}  {detail}")
    print("=" * 74)
    shutil.rmtree(_TMP, ignore_errors=True)
    return 0 if passed == total else 1


if __name__ == "__main__":
    raise SystemExit(main())
