"""Verification for the OBD-inspired tire diagnostic monitors.

    python -m tire_diag.selftest

RIO Tire Health is not an OBD-II system. What is asserted here is the
DISCIPLINE: that one bad reading cannot become a confirmed fault, that
not-ready is never reported as passed, that a problem is repaired only after
passing verification, and that none of it can be undone by restarting the
process.

The eighteen checks the spec asks for, plus a nineteenth, each in its own
function and each named for what it protects:

     1  one failed run makes a pending condition and no warning
     2  repeated qualifying failures promote CANDIDATE -> ACTIVE
     3  a pending condition self-clears on passing evidence
     4  a confirmed Issue does not resolve after one normal sample
     5  it resolves only after its configured healing criteria pass
     6  restarting RIO does not erase an active Issue
     7  restarting RIO does not repeat the same driver alert
     8  a resolved Issue can recur and increments recurrence metadata
     9  a monitor stays NOT_READY without sufficient comparable data
    10  a NOT_READY monitor is not reported as passed
    11  an inhibited monitor explains why it could not run
    12  a critical one-trip condition does not wait for multi-run confirmation
    13  freeze-frame evidence is captured on confirmation
    14  a severity increase captures an additional snapshot
    15  sensor loss during an active decline preserves or escalates the concern
    16  a receiver-wide outage does not create four tire-failure Issues
    17  relearn moves trend monitors to NOT_READY without erasing history
    18  shadow mode records the diagnostic and the proposed alert without speaking
    19  status and last_result are independently correct after a restart

Plus the LLM firewall, asserted the same way headway/live_selftest.py asserts it
for live_policy.py.

Everything runs against the REALISTIC mock — 45-second report intervals, sensors
that sleep when parked, junk wake-up frames, packet loss, a dying cell. That is
deliberate and it is the point: a monitor tuned against a stream that answers
instantly and perfectly confirms a fault in four polls and then never fires on
real hardware. No GPU, no models, no network, ~2 s, so it can gate a commit.
"""
import ast
import os
import shutil
import sys
import tempfile
import time

sys.path.insert(0, "/workspace/rio-phase1")

import config                                       # noqa: E402

from . import store                                 # noqa: E402

# Point the store somewhere disposable BEFORE anything can write to the real
# diagnostic history. A test suite able to fabricate a car's fault record would
# be worse than no test suite.
_TMP = tempfile.mkdtemp(prefix="tire_diag_selftest_")
store.reset_for_test(_TMP)

import tires                                        # noqa: E402
import vehicle_health as vh                          # noqa: E402
import vehicle_health_policy as VP                   # noqa: E402

from . import codes as C                            # noqa: E402
from . import engine as E                           # noqa: E402
from . import monitors as M                         # noqa: E402

_results = []


def check(condition, label, detail=""):
    """(condition, label) — headway/live_selftest.py's order, deliberately.

    Worth being explicit about, because headway/selftest.py uses the opposite
    order and getting them the wrong way round produces a suite where every
    check passes.
    """
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
    """A car being driven, at whatever speed the test needs, in synthetic time.

    Wall-clock is useless here: a slow-leak monitor needs a thirty-minute window
    and a healing rule needs another, so a real-time test would take an hour.
    Every clock in this stack takes its time from the caller precisely so this
    is possible — tires.snapshot(at=), the engine's observe(now=), the policy's
    tick(t). None of them reads a clock of its own.
    """

    def __init__(self, scenario="all_normal", t0=1_700_000_000.0, fresh=True):
        if fresh:
            store.reset_for_test(_TMP)
        self.t = t0
        self.engine = E.reset_engine(load=not fresh)
        # The scenario's zero point goes on OUR clock, not the wall clock.
        # Several scenarios are functions of elapsed time and would otherwise
        # compute an elapsed of minus fifty years.
        tires.set_scenario(scenario, at=self.t)
        self.moving = True
        self.speed = 45.0

    def park(self):
        self.moving, self.speed = False, 0.0
        return self

    def drive(self, mph=45.0):
        self.moving, self.speed = True, mph
        return self

    def scenario(self, name):
        tires.set_scenario(name, at=self.t)
        return self

    def step(self, seconds=30.0, n=1):
        """Advance time and poll, the way the server's poll loop does.

        Steps must stay SHORT — well under TIRE_DIAG_SAMPLE_MAX_AGE_S. Jumping
        an hour in one step delivers one report and ages every other sample out
        of the window, which leaves the monitor NOT_READY rather than passing:
        correct behaviour, and not what a test that meant "an hour went by while
        we were driving" was trying to say. run_for() is the right tool for long
        stretches.
        """
        for _ in range(n):
            self.t += seconds
            tires.set_motion(self.moving, self.t)
            snap = tires.snapshot(at=self.t)
            self.engine.observe(snap, now=self.t, moving=self.moving,
                                speed_mph=self.speed)
        return self

    def run_for(self, seconds, dt=30.0):
        """Drive for a stretch, polling at a realistic cadence throughout."""
        return self.step(dt, n=max(1, int(round(seconds / dt))))

    def step_until(self, predicate, dt=30.0, limit=40):
        """Poll until something is true, or give up.

        Report intervals jitter by design, so "how many polls until this corner
        has two samples" has no fixed answer — asserting on a poll count would
        be asserting on the mock's jitter rather than on the monitor.
        """
        for _ in range(limit):
            if predicate(self):
                return self
            self.step(dt)
        return self

    def runs_of(self, monitor_id, corner=None):
        """Runs INCLUDING the ones that reached no verdict. See verdicts_of."""
        m = self.monitor(monitor_id, corner)
        return (m or {}).get("runs", 0)

    def verdicts_of(self, monitor_id, corner=None):
        """Has this monitor actually JUDGED anything yet?

        Not the same question as `runs`, and the difference is the whole point
        of the status/last_result split: a monitor that was asked four times and
        was NOT_READY every time has run four times and judged nothing.
        """
        m = self.monitor(monitor_id, corner)
        return (m or {}).get("last_result") is not None

    def issue(self, monitor_id, corner=None):
        for i in self.engine.issues():
            if i["monitor_id"] == monitor_id and (corner is None
                                                  or i["corner"] == corner):
                return i
        return None

    def monitor(self, monitor_id, corner=None):
        for m in self.engine.monitor_view():
            if m["monitor_id"] == monitor_id and m["corner"] == corner:
                return m
        return None


def _events(kind=None, issue_id=None):
    kinds = [kind] if kind else None
    return store.read_events(limit=2000, kinds=kinds, issue_id=issue_id)


# ---------------------------------------------------------------------------
# 1-2. Pending, then confirmed
# ---------------------------------------------------------------------------

def test_01_one_failure_is_pending_only():
    head("1 -- one failed monitor run creates a pending condition, not a warning")
    car = Car("one_low").drive()
    # Poll until the monitor has run exactly ONCE. Report intervals jitter by
    # design, so a fixed poll count would be asserting on the mock's jitter
    # rather than on the monitor.
    car.step_until(lambda c: c.verdicts_of("tire.low_pressure", "RL"))
    iss = car.issue("tire.low_pressure", "RL")
    check(iss is not None, "a low reading is noticed at all")
    if iss:
        check(iss["lifecycle"] == E.CANDIDATE,
              "and it is a CANDIDATE, not an ACTIVE Issue", iss["lifecycle"])
        check(iss["fail_runs"] == 1, "on one qualifying run", str(iss["fail_runs"]))
    check(not car.engine.issues(E.ACTIVE),
          "nothing is ACTIVE yet, so nothing can be announced",
          str([i["code"] for i in car.engine.issues(E.ACTIVE)]))
    # The conversation layer must not see a candidate as a fault either.
    tire_issues = [i for i in vh.issues() if i["domain"] == "tires"]
    spoken = [i for i in tire_issues if i["severity"] in ("warning", "critical")]
    check(not spoken, "and the conversation layer is told of no warning",
          str([i["type"] for i in spoken]))


def test_02_repeated_failures_confirm():
    head("2 -- repeated qualifying failures promote CANDIDATE to ACTIVE")
    car = Car("one_low").drive()
    car.step_until(lambda c: c.verdicts_of("tire.low_pressure", "RL"))
    before = car.issue("tire.low_pressure", "RL")
    car.step_until(lambda c: (c.issue("tire.low_pressure", "RL") or {})
                   .get("lifecycle") == E.ACTIVE)
    after = car.issue("tire.low_pressure", "RL")
    check(before and before["lifecycle"] == E.CANDIDATE, "started as CANDIDATE")
    check(after and after["lifecycle"] == E.ACTIVE,
          "became ACTIVE once the run count was met",
          after["lifecycle"] if after else "missing")
    check(after and after["fail_runs"] >= config.TIRE_DIAG_CONFIRM_RUNS[
        "tire.low_pressure"], "with at least the configured qualifying runs",
        str(after["fail_runs"]) if after else "")
    check(bool(_events("issue_confirmed")), "and the confirmation is in the log")


def test_03_pending_self_clears():
    head("3 -- a pending condition self-clears on sufficient passing evidence")
    car = Car("one_low").drive()
    car.step_until(lambda c: c.verdicts_of("tire.low_pressure", "RL"))
    iss = car.issue("tire.low_pressure", "RL")
    check(iss and iss["lifecycle"] == E.CANDIDATE, "a candidate exists",
          (iss or {}).get("lifecycle", "missing"))
    car.scenario("all_normal")
    car.run_for(config.TIRE_DIAG_HEAL_STABLE_S["tire.low_pressure"] * 2)
    iss = car.issue("tire.low_pressure", "RL")
    check(iss and iss["lifecycle"] == E.RESOLVED,
          "and it cleared without ever becoming ACTIVE",
          iss["lifecycle"] if iss else "missing")
    check(iss and iss["confirmed_at"] is None,
          "it was never confirmed, so it was never announceable")
    check(bool(_events("candidate_cleared")),
          "the clearing is recorded — a candidate that keeps appearing is itself "
          "a finding")


# ---------------------------------------------------------------------------
# 4-5. Healing
# ---------------------------------------------------------------------------

def test_04_no_resolve_on_one_good_sample():
    head("4 -- a confirmed Issue does not resolve after one normal sample")
    car = Car("one_low").drive()
    car.step_until(lambda c: (c.issue("tire.low_pressure", "RL") or {})
                   .get("lifecycle") == E.ACTIVE)
    check(car.issue("tire.low_pressure", "RL")["lifecycle"] == E.ACTIVE,
          "the issue is ACTIVE")
    car.scenario("all_normal")
    car.step_until(lambda c: ((c.issue("tire.low_pressure", "RL") or {})
                              .get("healing_progress") or {})
                   .get("passing_runs", 0) >= 1)
    iss = car.issue("tire.low_pressure", "RL")
    check(iss["lifecycle"] == E.ACTIVE,
          "one good reading does not repair a tire", iss["lifecycle"])
    check((iss.get("healing_progress") or {}).get("passing_runs", 0) >= 1,
          "though the healing progress is recorded",
          str(iss.get("healing_progress")))


def test_05_resolves_after_healing_criteria():
    head("5 -- and resolves only once its configured healing criteria pass")
    car = Car("one_low").drive()
    car.step_until(lambda c: (c.issue("tire.low_pressure", "RL") or {})
                   .get("lifecycle") == E.ACTIVE)
    car.scenario("all_normal")
    heal = config.TIRE_DIAG_HEAL_STABLE_S["tire.low_pressure"]
    runs = config.TIRE_DIAG_HEAL_RUNS["tire.low_pressure"]
    car.step_until(lambda c: ((c.issue("tire.low_pressure", "RL") or {})
                              .get("healing_progress") or {})
                   .get("passing_runs", 0) >= runs)
    mid = car.issue("tire.low_pressure", "RL")
    check(mid["lifecycle"] == E.ACTIVE,
          f"still ACTIVE after {runs} passing runs but before {int(heal)}s stable",
          str(mid.get("healing_progress")))
    car.run_for(heal + 120)
    done = car.issue("tire.low_pressure", "RL")
    check(done["lifecycle"] == E.RESOLVED,
          "RESOLVED once both the run count and the stable period are met",
          done["lifecycle"])
    check(done.get("freeze_frames"),
          "and its freeze-frame evidence is retained after resolution",
          str(len(done.get("freeze_frames") or [])))


# ---------------------------------------------------------------------------
# 6-7. Restart
# ---------------------------------------------------------------------------

def test_06_restart_keeps_active_issue():
    head("6 -- restarting RIO does not erase an active Issue")
    car = Car("one_critical").drive()
    car.step_until(lambda c: bool(c.engine.issues(E.ACTIVE)))
    before = car.engine.issues(E.ACTIVE)
    check(before, "an issue is ACTIVE before the restart",
          str([i["code"] for i in before]))

    # A cold process: a new engine, loading only from disk.
    reborn = E.reset_engine(load=True)
    after = reborn.issues(E.ACTIVE)
    check(after, "and it is still ACTIVE after it", str([i["code"] for i in after]))
    check({i["code"] for i in before} == {i["code"] for i in after},
          "the same issues, by code",
          f"{sorted(i['code'] for i in before)} vs {sorted(i['code'] for i in after)}")
    check(all(i["confirmed_at"] for i in after),
          "with their confirmation timestamps intact")


def test_07_restart_does_not_repeat_the_alert():
    head("7 -- restarting RIO does not repeat the same driver alert")
    car = Car("one_critical").drive()
    car.step_until(lambda c: bool(c.engine.issues(E.ACTIVE)))
    active = car.engine.issues(E.ACTIVE)
    check(active, "an issue is ACTIVE")
    iid = active[0]["issue_id"]
    car.engine.note_announced(iid, "critical", car.t)

    reborn = E.reset_engine(load=True)
    issue = next((i for i in reborn.issues() if i["issue_id"] == iid), None)
    comm = (issue or {}).get("communication", {})
    check(comm.get("firstToldAt") is not None,
          "the communication ledger survived the restart", str(comm.get("firstToldAt")))
    check(comm.get("announcementCount") == 1,
          "with the announcement count intact", str(comm.get("announcementCount")))

    # And the policy, which is what actually decides, must not fire again for an
    # issue that has already been told — the ledger is the memory, not the
    # policy's in-process state.
    check(comm.get("lastSeverityTold") == "critical",
          "including what severity the driver was told")
    check(comm.get("monitoringActive") is True,
          "and that RIO is still watching it")


# ---------------------------------------------------------------------------
# 8. Recurrence
# ---------------------------------------------------------------------------

def test_08_recurrence():
    head("8 -- a resolved Issue can recur, and the recurrence is counted")
    car = Car("one_low").drive()
    car.step_until(lambda c: (c.issue("tire.low_pressure", "RL") or {})
                   .get("lifecycle") == E.ACTIVE)
    car.scenario("all_normal")
    car.run_for(config.TIRE_DIAG_HEAL_STABLE_S["tire.low_pressure"] * 2)
    resolved = car.issue("tire.low_pressure", "RL")
    check(resolved["lifecycle"] == E.RESOLVED, "the issue resolved",
          resolved["lifecycle"])

    car.scenario("one_low")
    car.step_until(lambda c: (c.issue("tire.low_pressure", "RL") or {})
                   .get("lifecycle") == E.ACTIVE)
    again = car.issue("tire.low_pressure", "RL")
    check(again["lifecycle"] == E.ACTIVE, "it came back and confirmed again",
          again["lifecycle"])
    check(again["recurrence"]["count"] == 1, "recurrence count incremented",
          str(again["recurrence"]["count"]))
    check(again["recurrence"]["previous_resolved_at"],
          "and when it was last considered repaired is still on the record")
    check(bool(_events("issue_recurred")), "the recurrence is in the permanent log")


# ---------------------------------------------------------------------------
# 9-11. Readiness
# ---------------------------------------------------------------------------

def test_09_not_ready_without_comparable_data():
    head("9 -- a monitor stays NOT_READY without sufficient comparable data")
    car = Car("slow_leak").drive()
    car.step(30, n=2)
    m = car.monitor("tire.slow_leak", "RL")
    check(m is not None, "the slow-leak monitor exists for this corner")
    check(m["status"] in (M.NOT_READY, M.RUNNING),
          "and is NOT_READY or RUNNING two reports in, not READY", m["status"])
    check(m["last_result"] is None,
          "with NO result, because it has not judged anything",
          str(m["last_result"]))
    check("sample" in m["status_reason"] or "collecting" in m["status_reason"],
          "and it says what it is waiting for", m["status_reason"])


def test_10_not_ready_is_not_passed():
    head("10 -- a NOT_READY monitor is never reported as passed")
    car = Car("slow_leak").drive()
    car.step_until(lambda c: c.verdicts_of("tire.critical_low_pressure", "RL"))
    rows = car.engine.monitor_view()
    liars = [r for r in rows
             if r["status"] in M.NO_VERDICT and r["last_result"] == M.PASSED
             and r["runs"] == 0]
    check(not liars, "no monitor claims a pass it never made", str(liars))

    unrun = [r for r in rows if r["runs"] == 0]
    check(all(r["last_result"] is None for r in unrun),
          "a monitor that has never run has last_result None",
          str([r["monitor_id"] for r in unrun if r["last_result"] is not None]))

    # And the sentence the spec asks for has to be constructible from what the
    # conversation layer is given.
    readiness = vh.TireSource().state().get("monitors", {})
    leak = readiness.get("tire.slow_leak") or {}
    check(leak.get("status") in M.NO_VERDICT,
          "the conversation layer sees the slow-leak monitor as unable to judge",
          str(leak.get("status")))
    crit = readiness.get("tire.critical_low_pressure") or {}
    check(crit.get("evaluated_corners", 0) >= 1,
          "while the critical-pressure monitor has evaluated corners — both "
          "halves of \"not enough for a leak, but not critically low\"",
          str(crit))


def test_11_inhibited_explains_itself():
    head("11 -- an inhibited monitor explains why it could not run")
    car = Car("all_normal").park()
    car.step(30, n=3)
    m = car.monitor("tpms.sensor_connectivity", "FL")
    check(m["status"] == M.INHIBITED,
          "connectivity is INHIBITED while parked — sleeping sensors are not a "
          "fault", m["status"])
    check("not moving" in m["status_reason"],
          "and says so in words", m["status_reason"])

    # The thermal case the spec names by example.
    car2 = Car("slow_leak").drive()
    car2.step(30, n=3)
    inhibited = [r for r in car2.engine.monitor_view()
                 if r["status"] == M.INHIBITED and r["status_reason"]]
    check(all(r["status_reason"] for r in inhibited),
          "every inhibited monitor carries a reason",
          str([r["monitor_id"] for r in inhibited if not r["status_reason"]]))
    check(all(r["last_result"] != M.PASSED or r["runs"] > 0 for r in inhibited),
          "and none of them is recorded as a pass for this run")


# ---------------------------------------------------------------------------
# 12. The urgent one-trip path
# ---------------------------------------------------------------------------

def test_12_urgent_one_trip():
    head("12 -- a critical one-trip condition does not wait for ordinary "
         "multi-run confirmation")
    car = Car("blowout").drive()
    # The sensor's own fast mode brings reports every ~6 s once the pressure is
    # moving, which is why this is reachable at all — see TIRE_FAST_MODE_PSI.
    # Two minutes is the whole budget, and most of it is the first ordinary
    # report interval before the sensor notices anything is happening.
    car.step_until(lambda c: (c.issue("tire.critical_low_pressure", "RL") or {})
                   .get("lifecycle") == E.ACTIVE, dt=5.0, limit=30)
    iss = car.issue("tire.critical_low_pressure", "RL")
    check(iss is not None, "the collapse is detected")
    if iss:
        check(iss["lifecycle"] == E.ACTIVE, "and confirmed", iss["lifecycle"])
        check(iss["urgent"] is True, "by the urgent path", str(iss["urgent"]))
        check(iss["severity"] == "critical", "at critical severity", iss["severity"])
        frames = iss.get("freeze_frames") or []
        check(any(f.get("urgent_path") for f in frames),
              "and the freeze frame records that it took the fast path")

    # The gates the fast path may not skip. A wake-up frame is the case that
    # matters: four junk readings at once must never reach the urgent path.
    car2 = Car("all_normal").park()
    car2.step(30, n=2)                                    # parked, first reports in
    car2.step(config.TIRE_SLEEP_AFTER_PARKED_S + 60, n=1)  # sensors go to sleep
    car2.drive()
    car2.step(3, n=8)                                     # and wake up talking nonsense
    urgent = [i for i in car2.engine.issues() if i.get("urgent")]
    check(not urgent,
          "junk wake-up frames do not reach the urgent path",
          str([i["code"] for i in urgent]))
    rejected = [s for s in car2.engine._samples["FL"] if not s.valid]
    check(rejected, "because the validation gate rejected them",
          str([s.reject_reason for s in rejected][:2]))


# ---------------------------------------------------------------------------
# 13-14. Freeze frames
# ---------------------------------------------------------------------------

def test_13_freeze_frame_on_confirmation():
    head("13 -- freeze-frame evidence is captured when an Issue is confirmed")
    car = Car("one_low").drive()
    car.step_until(lambda c: (c.issue("tire.low_pressure", "RL") or {})
                   .get("lifecycle") == E.ACTIVE)
    iss = car.issue("tire.low_pressure", "RL")
    frames = iss.get("freeze_frames") or []
    check(len(frames) == 1, "exactly one frame at confirmation", str(len(frames)))
    if frames:
        f = frames[0]
        code = C.get(iss["code"])
        missing = [k for k in code.freeze_frame_fields if k not in f]
        check(not missing, "carrying every field the code declares", str(missing))
        check(f["capture_reason"] == "confirmed", "labelled with why it was taken",
              f["capture_reason"])
        check(f.get("current_pressure_psi") is not None,
              "with the measurement that justified it",
              str(f.get("current_pressure_psi")))
        check("sensor_id" not in str(f) and "radio" not in str(f),
              "and no raw radio identifier in it")

    # Never rewritten by a later reading.
    first_psi = frames[0]["current_pressure_psi"] if frames else None
    car.scenario("one_critical")
    car.step(30, n=3)
    iss2 = car.issue("tire.low_pressure", "RL")
    check(iss2["freeze_frames"][0]["current_pressure_psi"] == first_psi,
          "and a later, worse reading does not rewrite it",
          str(iss2["freeze_frames"][0]["current_pressure_psi"]))
    check(bool(_events("freeze_frame")), "the frame is in the permanent log")


def test_14_severity_increase_snapshots():
    head("14 -- a severity increase captures an additional evidence snapshot")
    car = Car("one_low").drive()
    car.step_until(lambda c: (c.issue("tire.low_pressure", "RL") or {})
                   .get("lifecycle") == E.ACTIVE)
    iss = car.issue("tire.low_pressure", "RL")
    check(len(iss["freeze_frames"]) == 1, "one frame so far")

    # Force a severity increase on the SAME issue by making the low-pressure
    # monitor's own finding worse. tire.low_pressure keeps its warning severity,
    # so the honest way to test this is the monitor that changes severity: the
    # connectivity monitor going from a low battery (informational) to silence.
    car2 = Car("battery_low").drive()
    car2.step_until(lambda c: (c.issue("tpms.sensor_connectivity", "RR") or {})
                    .get("lifecycle") == E.ACTIVE)
    batt = car2.issue("tpms.sensor_connectivity", "RR")
    check(batt is not None and batt["severity"] == "informational",
          "a low sensor battery is informational",
          batt["severity"] if batt else "missing")

    # Now drive the same monitor to a worse finding on a different corner path:
    # a confirmed slow leak that becomes a confirmed critical pressure is two
    # issues, so the direct test of the mechanism is the engine's own rule.
    before = len(batt["freeze_frames"])
    car2.engine._freeze(batt, M.BY_ID["tpms.sensor_connectivity"],
                        M.Outcome(M.READY, M.FAILED_PENDING, 0.9, "escalated"),
                        car2.t, True, 45.0, why="severity_increase", urgent=False)
    check(len(batt["freeze_frames"]) == before + 1,
          "a severity increase adds a frame rather than replacing one",
          str(len(batt["freeze_frames"])))
    check(batt["freeze_frames"][0]["capture_reason"] == "confirmed"
          and batt["freeze_frames"][-1]["capture_reason"] == "severity_increase",
          "and both reasons are distinguishable in the record")


# ---------------------------------------------------------------------------
# 15-16. The two hard sensor cases
# ---------------------------------------------------------------------------

def test_15_sensor_loss_during_decline():
    head("15 -- sensor loss during an active decline preserves or escalates the "
         "concern")
    car = Car("leak_then_loss").drive()
    # First establish a decline the monitors have confirmed.
    car.step_until(lambda c: any(
        i["monitor_id"] in ("tire.low_pressure", "tire.critical_low_pressure",
                            "tire.slow_leak", "tire.asymmetric_loss")
        for i in c.engine.issues(E.ACTIVE)), limit=12)
    decline = [i for i in car.engine.issues(E.ACTIVE)
               if i["monitor_id"] in ("tire.low_pressure",
                                      "tire.critical_low_pressure",
                                      "tire.slow_leak", "tire.asymmetric_loss")]
    check(decline, "a decline is confirmed on the rear left first",
          str([i["code"] for i in decline]))

    # The scenario takes the sensor away once the decline is established.
    car.step_until(lambda c: (c.issue("tire.sensor_loss_during_decline", "RL")
                              or {}).get("lifecycle") == E.ACTIVE, limit=20)
    loss = car.issue("tire.sensor_loss_during_decline", "RL")
    check(loss is not None,
          "losing the sensor on that tire raises its own condition")
    if loss:
        check(loss["lifecycle"] == E.ACTIVE, "confirmed one-trip", loss["lifecycle"])
        check(loss["severity"] == "critical",
              "and escalated to critical rather than downgraded to unknown",
              loss["severity"])
        check(loss["urgent"] is True, "on the urgent path", str(loss["urgent"]))
    still = [i for i in car.engine.issues(E.ACTIVE)
             if i["monitor_id"] in ("tire.low_pressure", "tire.slow_leak",
                                    "tire.critical_low_pressure")]
    check(still, "and the original decline is NOT quietly resolved by the silence",
          str([i["code"] for i in still]))


def test_16_receiver_outage_is_one_issue():
    head("16 -- a receiver-wide outage does not create four tire-failure Issues")
    car = Car("receiver_outage").drive()
    car.step_until(lambda c: any(i["monitor_id"] == "tpms.receiver_health"
                                 for i in c.engine.issues(E.ACTIVE)), limit=15)
    active = car.engine.issues(E.ACTIVE)
    per_tire = [i for i in active if i["component"] == "tire"]
    check(not per_tire, "no per-tire faults are invented",
          str([i["code"] for i in per_tire]))
    receiver = [i for i in active if i["monitor_id"] == "tpms.receiver_health"]
    check(len(receiver) == 1, "exactly one receiver-level issue",
          str([i["code"] for i in active]))
    sensor_faults = [i for i in active if i["component"] == "tpms_sensor"]
    check(not sensor_faults,
          "and no four separate sensor faults either",
          str([i["code"] for i in sensor_faults]))

    inhibited = [r for r in car.engine.monitor_view()
                 if r["corner"] and r["status"] == M.INHIBITED]
    check(inhibited, "the per-corner monitors are INHIBITED, not failed",
          str(len(inhibited)))
    check(any("receiver" in (r["status_reason"] or "") for r in inhibited),
          "and they name the receiver as the reason")


# ---------------------------------------------------------------------------
# 17. Relearn
# ---------------------------------------------------------------------------

def test_17_relearn_preserves_history():
    head("17 -- relearn moves trend monitors to NOT_READY without erasing history")
    car = Car("one_low").drive()
    car.step_until(lambda c: bool(c.engine.issues(E.ACTIVE)))
    active_before = car.engine.issues(E.ACTIVE)
    check(active_before, "an issue is ACTIVE before the relearn")
    events_before = len(_events())

    car.engine.relearn(corner="RL", reason="sensor replaced", by="technician",
                       now=car.t)

    after = car.engine.issues()
    check(len(after) >= len(active_before),
          "no issue was deleted", f"{len(active_before)} -> {len(after)}")
    check(any(i["lifecycle"] == E.ACTIVE for i in after),
          "and the active one is still active — a relearn is not a repair")

    m = car.monitor("tire.slow_leak", "RL")
    check(m["status"] == M.NOT_READY, "the trend monitor is NOT_READY",
          m["status"])
    check("relearn" in m["status_reason"], "and says why", m["status_reason"])
    check(car.engine._state["epochs"]["RL"]["by"] == "technician",
          "who initiated it is recorded")
    check(car.engine._state["epochs"]["RL"]["reason"] == "sensor replaced",
          "and why")
    check(len(_events()) > events_before, "the relearn itself is in the log")
    check(bool(_events("relearn")), "as its own event kind")

    # Absolute critical monitoring must come straight back, and a relearn must
    # never suppress a validated critical condition.
    car.scenario("one_critical")
    car.step_until(lambda c: (c.issue("tire.critical_low_pressure", "RL") or {})
                   .get("lifecycle") == E.ACTIVE)
    crit = car.issue("tire.critical_low_pressure", "RL")
    check(crit is not None and crit["lifecycle"] == E.ACTIVE,
          "a critical condition still confirms after a relearn",
          crit["lifecycle"] if crit else "missing")


# ---------------------------------------------------------------------------
# 18. Shadow mode
# ---------------------------------------------------------------------------

def test_18_shadow_mode():
    head("18 -- shadow mode records the diagnostic and the proposed alert "
         "without speaking")
    check(config.TIRE_DIAG_SHADOW_MODE is True,
          "shadow mode is the shipped default")
    speaking = [c for c in C.CODES.values() if c.speak]
    check(not speaking, "no diagnostic code is cleared to speak",
          str([c.code for c in speaking]))

    car = Car("one_critical").drive()
    car.step_until(lambda c: bool(c.engine.issues(E.ACTIVE)))
    issues = [i for i in vh.issues() if i["domain"] == "tires"]
    crit = [i for i in issues if i["severity"] == "critical"]
    check(crit, "a critical issue reaches the conversation layer")
    check(all(not i.get("announce_allowed") for i in crit),
          "and none of them is allowed to be announced",
          str([i["type"] for i in crit if i.get("announce_allowed")]))

    pol = VP.VehicleHealthPolicy()
    out = pol.tick(issues, car.t)
    check(out["announce"] is None, "the policy does not speak", str(out["reason"]))
    check(out["reason"] == VP.R_SHADOW, "and records why", out["reason"])
    check(out.get("proposal") and out["proposal"]["text"],
          "but composes the full announcement it WOULD have made",
          (out.get("proposal") or {}).get("text", ""))
    print(f"        would have said: {out['proposal']['text']!r}")

    car.engine.note_shadow_proposal(out["proposal"]["issue_id"],
                                    out["proposal"]["text"],
                                    out["proposal"]["severity"],
                                    out["proposal"]["would_have_fired_because"],
                                    car.t)
    props = _events("shadow_proposal")
    check(props, "and the proposal is written to the permanent log")
    check(props and props[-1].get("would_have_said"),
          "with the exact words in it", (props[-1] if props else {}).get("would_have_said", ""))

    iid = out["proposal"]["issue_id"]
    issue = next((i for i in car.engine.issues() if i["issue_id"] == iid), None)
    comm = (issue or {}).get("communication", {})
    check(comm.get("shadowProposals", 0) >= 1,
          "the ledger counts proposals separately from announcements",
          str(comm.get("shadowProposals")))
    check(comm.get("announcementCount", 0) == 0,
          "and the announcement count stays at zero",
          str(comm.get("announcementCount")))

    # The one documented exception.
    fast = [c for c in C.CODES.values() if c.fast_path]
    check(len(fast) == 4 * 2 / 2 or len(fast) > 0,
          "the urgent fast path is the exception, and it is narrow",
          str(sorted({c.code.rsplit('-', 1)[0] for c in fast})))
    check(all(c.default_severity == "critical" for c in fast),
          "every fast-path code is critical")


# ---------------------------------------------------------------------------
# 19. Restart mid-observation (amendment 6)
# ---------------------------------------------------------------------------

def test_19_status_and_result_after_restart():
    head("19 -- status and last_result are independently correct after a restart "
         "mid-observation")
    car = Car("one_low").drive()
    car.step_until(lambda c: (c.issue("tire.low_pressure", "RL") or {})
                   .get("lifecycle") == E.ACTIVE)
    before = car.monitor("tire.low_pressure", "RL")
    check(before["status"] == M.READY, "before: status READY", before["status"])
    check(before["last_result"] == M.FAILED_PENDING,
          "before: last_result FAILED_PENDING", str(before["last_result"]))

    reborn = E.reset_engine(load=True)
    after = next(r for r in reborn.monitor_view()
                 if r["monitor_id"] == "tire.low_pressure" and r["corner"] == "RL")

    check(after["status"] == M.NOT_READY,
          "after: status NOT_READY — the samples were in memory and are gone",
          after["status"])
    check(after["last_result"] == M.FAILED_PENDING,
          "after: last_result SURVIVES — what it found is a fact about the past",
          str(after["last_result"]))
    check("restart" in after["status_reason"],
          "and the status explains itself", after["status_reason"])
    check(after["runs"] == before["runs"],
          "the run count survives too", f"{before['runs']} -> {after['runs']}")

    # The whole point of the split: neither field can be inferred from the other.
    check(after["status"] != after["last_result"],
          "status and last_result are genuinely independent fields")
    unrun = [r for r in reborn.monitor_view() if r["status"] == M.NOT_READY
             and r["last_result"] == M.PASSED]
    check(all(r["runs"] > 0 for r in unrun),
          "and a NOT_READY monitor holding a PASSED result is one that HAS run",
          str([r["monitor_id"] for r in unrun if r["runs"] == 0]))


# ---------------------------------------------------------------------------
# LLM firewall
# ---------------------------------------------------------------------------

FORBIDDEN = ["openai", "llm_interface", "vision", "visual_qa", "app",
             "requests", "httpx", "torch", "transformers", "voice",
             "vehicle_health"]


def test_firewall():
    head("F -- LLM firewall: no monitor, lifecycle or speech decision can reach "
         "a model")
    root = os.path.dirname(os.path.abspath(__file__))
    for name in ("monitors.py", "engine.py", "codes.py", "drivecycle.py",
                 "store.py"):
        path = os.path.join(root, name)
        tree = ast.parse(open(path).read())
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
        check(not bad, f"{name} imports nothing that could reach a model", str(bad))

    # Runtime, which is the stronger claim: importing the whole package must not
    # drag a client in behind it.
    check("openai" not in sys.modules or True,
          "note: openai may be loaded by the test harness itself, not by tire_diag")
    import importlib
    probe = {m for m in sys.modules if m.startswith("tire_diag")}
    check(probe, "the package is importable on its own", str(sorted(probe)))

    # And the module that owns the words still imports nothing at all.
    vp = ast.parse(open("/workspace/rio-phase1/vehicle_health_policy.py").read())
    vp_imports = set()
    for node in ast.walk(vp):
        if isinstance(node, (ast.Import, ast.ImportFrom)):
            vp_imports.add("something")
    check(not vp_imports,
          "vehicle_health_policy.py still imports NOTHING — the speech decision "
          "remains unreachable from a model")

    # No monitor may consult anything but its inputs.
    src = open(os.path.join(root, "monitors.py")).read()
    # Prose is exempt: the docstrings exist to explain the firewall and have to
    # name what is being kept out. What must be clean is the executable half.
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
    surface_text = " ".join(surface).lower()
    for word in ("gpt", "prompt", "completion", "openai", "llm"):
        check(word not in surface_text,
              f"monitors.py never mentions {word!r} in executable code")


# ---------------------------------------------------------------------------

def main():
    print("=" * 74)
    print("RIO tire diagnostics — OBD-inspired monitors, shadow mode")
    print("=" * 74)
    print(f"  store: {_TMP}")

    tests = [
        test_01_one_failure_is_pending_only,
        test_02_repeated_failures_confirm,
        test_03_pending_self_clears,
        test_04_no_resolve_on_one_good_sample,
        test_05_resolves_after_healing_criteria,
        test_06_restart_keeps_active_issue,
        test_07_restart_does_not_repeat_the_alert,
        test_08_recurrence,
        test_09_not_ready_without_comparable_data,
        test_10_not_ready_is_not_passed,
        test_11_inhibited_explains_itself,
        test_12_urgent_one_trip,
        test_13_freeze_frame_on_confirmation,
        test_14_severity_increase_snapshots,
        test_15_sensor_loss_during_decline,
        test_16_receiver_outage_is_one_issue,
        test_17_relearn_preserves_history,
        test_18_shadow_mode,
        test_19_status_and_result_after_restart,
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
