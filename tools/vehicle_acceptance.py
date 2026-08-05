"""vehicle_acceptance.py — §27.9's condition list, driven end to end.

    python tools/vehicle_acceptance.py [--family engine|link|codes] [-v]

WHAT THIS IS, AND WHY IT IS NOT ANOTHER SELFTEST
------------------------------------------------
vehicle.selftest, powertrain_diag.selftest and the rest prove that each layer
does its own job. This proves something they cannot: that a CONDITION — an
overheating engine, a bridge in a tunnel, a pending code the lamp has not shown
— travels the whole distance from the producer that invents it to the surface a
driver actually reads, and arrives saying the right thing.

So nothing here reaches into a module and calls one function. Every condition is
established at the producer, pushed through the real ingestion front door, read
back through the real telemetry pipeline, observed by the real monitors, scanned
by the real DTC service, and finally judged on what came out. Where a selftest
asks "does the lifecycle promote a twice-seen pending code", this asks "if the
ECU reports P0171 twice, does the panel end up with a card that says Repeated
Pending" — and it gets there the same way the running system would.

§27.9's conditions are not all the same kind of thing, and the three families
here are the split that matters:

  A. ENGINE  (15)  facts about the CAR      — scenarios in producers/physics.py
  B. LINK     (5)  facts about the TRANSPORT — modes in vehicle/faults.py
  C. CODES   (13)  what the VEHICLE REPORTS  — scripts in producers/ecu.py

THE CLOCK, WHICH IS THE ONE PIECE OF MACHINERY IN HERE
-------------------------------------------------------
Every freshness rule in this pipeline is measured against wall time, and
config.TELEMETRY_STALE_AFTER_S is six seconds. That leaves two honest ways to
drive a scenario and one dishonest one.

Driving it in real time is honest and unusable: `fuel_trim_drift` needs about
ten minutes of engine running before long-term trim is out of band long enough
to count, and there are thirty-three conditions.

Replaying ten minutes of simulated time instantly against the real clock is the
dishonest one, and it fails in a way that looks like a passing test. Every
reading lands with a timestamp minutes old, every row goes stale, `engine_running`
comes back False, and eight of the nine engine monitors report INHIBITED — "the
engine is not running". Nothing raises; the harness simply stops testing the
thing it claims to test.

So the clock is controlled, and simulated time is advanced one tick per second
of scenario. Freshness then means what it means on a running vehicle, a
120-second hold really is 120 seconds, and the monitors see the stream they
would see on a car. This is the only global patch in the file and it is undone
in a finally block.

WHAT THE ASSERTIONS ARE ALLOWED TO SAY
--------------------------------------
Not a number. There is no threshold, band, hold time or severity constant in
this file, and that is not tidiness — a harness that restated `240°F` would pass
forever after somebody changed the limit to 250, because it would be agreeing
with its own copy rather than with the system. So the claims are structural:
which monitor reached a verdict, whether that verdict was a failing one, which
group a code landed in, what the status label says. The numbers stay in
config.py, where exactly one copy of each of them lives.

OPEN FINDINGS
-------------
A condition whose outcome is wrong is reported as a FINDING rather than quietly
encoded as expected. The whole reason powertrain_diag ships in shadow mode is
that its monitors have never seen a vehicle; this harness is one of the few
places their false positives can show up before a driver hears one, and a
harness that wrote today's false positive into its own expectations would be
the thing that stops them ever being found.
"""
import argparse
import os
import sys
import tempfile
import time

sys.path.insert(0, "/workspace/rio-phase1")

# ---------------------------------------------------------------------------
# The clock. Installed before anything that reads it is imported.
# ---------------------------------------------------------------------------
_real_time = time.time


class Clock:
    """Simulated wall time. Callable, so it can BE time.time."""

    def __init__(self, start: float):
        self.t = float(start)

    def __call__(self) -> float:
        return self.t

    def set(self, t: float) -> None:
        self.t = float(t)


CLOCK = Clock(_real_time())
time.time = CLOCK

import config                                              # noqa: E402
import telemetry                                           # noqa: E402
import vehicle_health                                      # noqa: E402
import insights                                            # noqa: E402
from diag import shadow as diag_shadow                     # noqa: E402
from vehicle import faults, ingest, producers              # noqa: E402
from vehicle import report as vehicle_report               # noqa: E402
from vehicle import state as vehicle_state_mod             # noqa: E402
from vehicle.dtc import service as dtc_service             # noqa: E402
from vehicle.gateway import auth as gateway_auth           # noqa: E402
from vehicle.producers import ecu as ecu_mod               # noqa: E402
from vehicle.producers import simulator as sim_mod         # noqa: E402
from vehicle.providers import ingested                     # noqa: E402
from vehicle.signals import provenance as P                # noqa: E402
from vehicle.signals import quality as Q                   # noqa: E402
import powertrain_diag.engine as powertrain                # noqa: E402

_results = []
_findings = []
_TMP = tempfile.mkdtemp(prefix="vehicle_acceptance_")
_run = 0
VERBOSE = False


def check(condition, label, detail=""):
    ok = bool(condition)
    _results.append((ok, label, detail))
    if VERBOSE or not ok:
        print(f"    [{'PASS' if ok else 'FAIL'}] {label}"
              + (f"  -- {detail}" if detail else ""))
    return ok


def finding(condition_name, title, detail):
    """Something true and wrong. Reported, never asserted away."""
    _findings.append((condition_name, title, detail))


def head(title):
    print(f"\n{title}")
    print("-" * len(title))


# ---------------------------------------------------------------------------
# The rig
# ---------------------------------------------------------------------------

def _reset(scenario: str, ecu_scenario: str, t0: float) -> None:
    """A clean vehicle, a clean link and a clean diagnostic history.

    All five of these matter and the reason is one bug each. The ingestion
    buffer keeps the previous condition's readings, which arrive dated in a
    future this condition has not reached yet. The fault injector keeps its
    outbox. The simulator keeps its elapsed time, so `warmup` starts warm. The
    powertrain engine keeps confirmation counts, so the second condition
    confirms on evidence the first one gathered. And the DTC scheduler keeps
    ABSOLUTE last-scanned timestamps — leave those and the next condition is
    never due for a scan, its section stays empty, and every code assertion in
    family C passes by finding nothing.
    """
    global _run
    _run += 1
    ingest.reset_for_test()
    faults.injector().reset()
    telemetry.set_source("simulation")
    sim = sim_mod.simulator()
    sim.reset()
    sim.set_scenario(scenario, now=t0)
    powertrain.reset_engine(load=False)
    ecu_mod.ecu().reset()
    ecu_mod.ecu().set_scenario(ecu_scenario, now=t0)
    dtc_service.service().reset_for_test(os.path.join(_TMP, f"dtc{_run}"))


def drive(scenario="cruise", ecu_scenario="healthy", ticks=60, fault=None,
          release_at=None, moving=True, speed_mph=45.0):
    """Run one condition for `ticks` simulated seconds. -> everything observable.

    One tick is one second and does what the running server does in a second:
    the producer builds its events, they go through the injector and the front
    door, the monitors observe the resulting snapshot, and the DTC scheduler
    gets a chance to ask for whatever is due.

    `release_at` clears the injected fault partway through, which is the only
    way to test a recovery — a bridge that is still in the tunnel has not
    reconnected yet, and the interesting half of cloud_disconnect is what
    arrives afterwards.
    """
    # Time only moves forward, including BETWEEN conditions. Restarting each
    # condition at the real clock would rewind simulated time by however long
    # the previous one ran — `fuel_trim_drift` alone advances it ten minutes —
    # and every provider not reset by _reset() would then be holding readings
    # stamped in the future. The tire mock is exactly that provider, and the
    # symptom is a panel that reads stale for reasons the condition under test
    # had nothing to do with. No vehicle's clock goes backwards; neither does
    # this one.
    t0 = max(CLOCK.t + 1.0, _real_time())
    CLOCK.set(t0)
    _reset(scenario, ecu_scenario, t0)
    if fault:
        faults.injector().set_mode(fault)

    pump_total = {"built": 0, "sent": 0, "accepted": 0, "rejected": 0,
                  "applied_as_current": 0, "superseded_by_newer": 0}
    for i in range(ticks):
        CLOCK.set(t0 + i)
        if release_at is not None and i == release_at:
            faults.injector().set_mode(faults.NONE)
        res = producers.pump_simulation(CLOCK.t)
        for k in pump_total:
            pump_total[k] += res.get(k) or 0
        snap = telemetry.snapshot(record=False)
        powertrain.observe(snap, now=CLOCK.t, moving=moving,
                           speed_mph=speed_mph)
        dtc_service.service().poll(CLOCK.t)

    snap = telemetry.snapshot(record=False)
    monitors = {m["monitor_id"]: m for m in powertrain.engine().monitor_view()}
    return {
        "t0": t0,
        "snapshot": snap,
        "monitors": monitors,
        "failed": sorted(m for m, v in monitors.items()
                         if v["last_result"] and "FAIL" in v["last_result"]),
        "passed": sorted(m for m, v in monitors.items()
                         if v["last_result"] == "PASSED"),
        "no_verdict": sorted(m for m, v in monitors.items()
                             if v["last_result"] is None),
        "pump": pump_total,
        "injector": faults.injector().stats(),
        "section": dtc_service.service().section(),
        "capability": ingested.buffer().capability(),
        "latest": ingested.buffer().latest(),
        "issues": vehicle_health.issues(),
    }


def run_report(session_id=None) -> dict:
    """The real §27.8 job, wired exactly as app.py wires it."""
    report = vehicle_report.store().create(config.VEHICLE_ID, session_id)
    return report.run(
        dtc_service=dtc_service.service(),
        telemetry_snapshot=lambda: telemetry.snapshot(record=False),
        vehicle_state=lambda snap: vehicle_state_mod.derive(
            snap, False, powertrain.engine().cycles.state().get("current")),
        health_issues=vehicle_health.issues,
        insight_feed=lambda: insights.snapshot().get("entries", []),
        gateways=lambda: gateway_auth.gateways(config.VEHICLE_ID),
        capability=lambda: ingested.buffer().capability(),
    )


# ---------------------------------------------------------------------------
# Invariants — asserted on EVERY condition, whatever it was meant to show
# ---------------------------------------------------------------------------

def invariants(name: str, out: dict) -> None:
    """The four things that must hold no matter what the car is doing.

    These are the claims worth more than any single scenario's expected
    outcome, because they are the ones whose failure would be invisible: a
    fabricated reading looks like a reading, and a monitor that passes a channel
    it never received looks like good news.
    """
    # 1. Nothing is invented. Everything in the buffer came from the simulator
    #    and says so, in both fields, so that no consumer can mistake it for a
    #    measurement of a real vehicle.
    bad_prov = [s for s, e in out["latest"].items()
                if e.get("provenance") != P.SIMULATION]
    check(not bad_prov, f"{name}: every reading carries simulation provenance",
          ", ".join(sorted(bad_prov)[:3]))
    bad_qual = [s for s, e in out["latest"].items()
                if Q.is_trustworthy(e.get("quality"))]
    check(not bad_qual,
          f"{name}: and a quality no monitor may treat as trustworthy",
          ", ".join(sorted(bad_qual)[:3]))

    # 2. The powertrain domain is shadowed, so nothing it found may speak. This
    #    is re-asserted per condition rather than once, because the interesting
    #    case is precisely the condition that produced a critical finding.
    check(diag_shadow.is_shadowed("powertrain"),
          f"{name}: the powertrain domain is still shadowed — a finding here "
          f"may be written down and may not interrupt anyone")

    # 3. Missing is not healthy. A monitor whose inputs never arrived must have
    #    NO verdict; reporting PASSED for a channel it never received is the
    #    single most dangerous thing a diagnostic layer can do, because it is
    #    indistinguishable from good news.
    for mid, m in out["monitors"].items():
        if m["last_result"] == "PASSED" and m["status"] in ("NOT_READY",
                                                            "INHIBITED"):
            check(False,
                  f"{name}: {mid} reported PASSED while {m['status']}",
                  m["status_reason"])


def report_invariants(name: str, view: dict) -> None:
    """A report must complete, and must never promote a possibility to a fact."""
    check(view.get("status") == "complete",
          f"{name}: the diagnostic report completes",
          str(view.get("error") or view.get("status")))
    body = view.get("report") or {}
    if not body:
        return
    check(bool(body.get("summary")),
          f"{name}: with a summary assembled from its own fields")
    # Every code the report carries still says its cause is unconfirmed. The
    # report is a document about evidence, and there is nothing in this system
    # that can confirm a cause except a person.
    cards = (body.get("early_detection", {}).get("detected_before_warning_light", [])
             + body.get("early_detection", {}).get("repeated_pending", [])
             + body.get("confirmed_faults", {}).get("active_or_confirmed", [])
             + body.get("confirmed_faults", {}).get("permanent", []))
    unconfirmed = [c for c in cards
                   if (c.get("cause_status") or "Not Confirmed") != "Not Confirmed"]
    check(not unconfirmed,
          f"{name}: and no code in it claims a confirmed cause",
          ", ".join(c.get("code", "?") for c in unconfirmed[:3]))
    # RIO's findings are labelled as RIO's, in the section that says so.
    obs = body.get("rio_observations", {})
    unlabelled = [o for o in (obs.get("observations") or [])
                  if not o.get("provenance")]
    check(not unlabelled,
          f"{name}: and every RIO observation carries its provenance",
          str(len(unlabelled)))


# ---------------------------------------------------------------------------
# A. ENGINE — facts about the car
# ---------------------------------------------------------------------------
# ticks: how long the condition needs to become visible. These differ by an
# order of magnitude and that is the point — a coolant limit is reached in
# seconds and a long-term fuel trim drift takes ten minutes, and a harness that
# gave both the same window would silently stop testing the slow ones.
ENGINE = (
    # (scenario, ticks, must_fail, note)
    ("normal_idle", 60, (), "a healthy idling engine"),
    ("warmup", 60, (), "a cold engine coming up to temperature"),
    ("cruise", 60, (), "steady cruise"),
    ("city", 60, (), "stop-start city driving"),
    ("aggressive", 60, (), "hard driving, still nothing wrong"),
    ("charging_fault", 60, ("engine.charging_voltage",),
     "the alternator is not holding the bus up"),
    # All three coolant monitors are expected here, and they are three
    # different claims: past the absolute limit, climbing too fast, and hot
    # relative to what this vehicle normally runs at.
    ("overheating", 60, ("engine.coolant_hard_limit",
                         "engine.coolant_rate_of_rise",
                         "engine.coolant_contextual"),
     "coolant past the limit and still climbing"),
    ("coolant_rapid_rise", 60, ("engine.coolant_rate_of_rise",),
     "coolant climbing fast while still in range"),
    ("start_voltage_decline", 400, ("engine.start_voltage_trend",),
     "a battery that is weaker at every start"),
    ("fuel_trim_drift", 620, ("engine.fuel_trim_long_term",),
     "long-term trim walking out of its band"),
    ("sensor_dropout", 60, ("engine.signal_integrity",),
     "two channels stop arriving"),
    ("unsupported_pids", 60, (), "PIDs this ECU never supported"),
    ("frozen_signal", 130, ("engine.signal_integrity",),
     "a channel that has stopped changing"),
    # The impossible value is rejected at the door, so the channel goes silent
    # rather than showing a number — and it is signal integrity that notices.
    ("invalid_value", 60, ("engine.signal_integrity",),
     "a reading outside physical possibility"),
    ("ecu_no_response", 60, ("engine.connection",), "the ECU stops answering"),
)

# The scenarios where nothing is wrong with the engine. `warmup` belongs here:
# a cold engine coming up to temperature is a car in perfect health, and a
# monitor that fires during it fires on every first drive of the day.
HEALTHY = {"normal_idle", "warmup", "cruise", "city", "aggressive"}


def run_engine():
    head("A -- ENGINE: conditions that are facts about the car")
    for scenario, ticks, must_fail, note in ENGINE:
        out = drive(scenario=scenario, ticks=ticks)
        print(f"  {scenario:<24} {note}")
        invariants(scenario, out)

        for monitor in must_fail:
            m = out["monitors"].get(monitor, {})
            check(monitor in out["failed"],
                  f"{scenario}: {monitor} reaches a failing verdict",
                  m.get("status_reason", "no such monitor"))

        # Anything that failed and was not the point of this scenario. Reported
        # as a finding rather than failing the run, and the distinction is the
        # whole reason powertrain_diag ships in shadow mode: these monitors have
        # never seen a vehicle, collecting their false positives is exactly what
        # shadow mode is for, and a harness that hard-failed on every one of
        # them could not be used as a gate on anything else. What IS gated is
        # the contract underneath — provenance, missing-is-not-healthy, the
        # labels, the plumbing — because those are settled.
        for m in out["failed"]:
            if m in must_fail:
                continue
            where = "on a healthy engine" if scenario in HEALTHY \
                else "unrelated to this condition"
            finding(scenario, f"{m} reports a fault {where}",
                    out["monitors"][m]["status_reason"])

        # Per-scenario claims that are about the SURFACE, not the monitor.
        if scenario == "overheating":
            check(out["snapshot"]["status"] == "CRITICAL",
                  "overheating: the panel headline reaches CRITICAL",
                  out["snapshot"]["status"])

        if scenario == "sensor_dropout":
            # The dropped channels must go quiet, not to zero, and the monitors
            # that needed them must have no verdict rather than a pass.
            rows = {r["id"]: r for r in out["snapshot"]["rows"]}
            gone = [c for c in ("coolant_temp", "oil_pressure")
                    if rows.get(c, {}).get("value") is None]
            check(len(gone) == 2,
                  "sensor_dropout: a dropped channel reads as no data, not 0",
                  f"silent: {gone}")
            check("engine.coolant_hard_limit" in out["no_verdict"],
                  "sensor_dropout: and the monitor that needed it has NO "
                  "verdict — 'we could not look' is not 'it is fine'",
                  out["monitors"]["engine.coolant_hard_limit"]["status_reason"])

        if scenario == "unsupported_pids":
            cap = out["capability"]
            for signal in ("powertrain.engine.mass_air_flow",
                           "powertrain.fuel.long_term_trim_bank_1"):
                check(signal in cap["unsupported"],
                      f"unsupported_pids: {signal.split('.')[-1]} is reported "
                      f"as unsupported, not as missing data", "")
            check("engine.fuel_trim_long_term" in out["no_verdict"],
                  "unsupported_pids: a monitor whose PID this ECU does not "
                  "support returns no verdict rather than a pass",
                  out["monitors"]["engine.fuel_trim_long_term"]["status_reason"])

        if scenario == "invalid_value":
            rows = {r["id"]: r for r in out["snapshot"]["rows"]}
            coolant = rows.get("coolant_temp", {})
            check(coolant.get("value") is None,
                  "invalid_value: an impossible reading never reaches the "
                  "panel as a number", str(coolant.get("value")))

        if scenario == "ecu_no_response":
            check(out["snapshot"]["engine_running"] is False,
                  "ecu_no_response: the engine does not read as running")
            invented = [m for m in out["failed"] if m != "engine.connection"]
            check(not invented,
                  "ecu_no_response: one dead link produces one finding, not "
                  "nine engine faults", ", ".join(invented))


# ---------------------------------------------------------------------------
# B. LINK — facts about the transport
# ---------------------------------------------------------------------------

def run_link():
    head("B -- LINK: conditions no physics model can produce")

    # Gateway disconnect. The bridge stopped MEASURING: this data does not
    # exist and never will.
    out = drive(fault=faults.GATEWAY_DISCONNECT, ticks=60)
    print("  gateway_disconnect       the bridge stopped measuring")
    invariants("gateway_disconnect", out)
    check(out["pump"]["accepted"] == 0,
          "gateway_disconnect: nothing arrives", str(out["pump"]))
    check(out["injector"].get("dropped", 0) > 0,
          "gateway_disconnect: and the events are counted as LOST",
          str(out["injector"].get("dropped")))
    check("engine.connection" in out["failed"],
          "gateway_disconnect: the connection monitor is what notices")
    invented = [m for m in out["failed"] if m != "engine.connection"]
    check(not invented,
          "gateway_disconnect: and no engine fault is invented from silence",
          ", ".join(invented))

    # Cloud disconnect. The bridge is fine and cannot upload: every reading is
    # still being taken and arrives late. The half that matters is the recovery.
    out = drive(fault=faults.CLOUD_DISCONNECT, ticks=60, release_at=45)
    print("  cloud_disconnect         the bridge cannot upload, then can")
    invariants("cloud_disconnect", out)
    check(out["pump"]["accepted"] >= out["pump"]["built"] * 0.9,
          "cloud_disconnect: everything held is delivered on reconnect — the "
          "difference between mourning data that is about to turn up and "
          "waiting for data that is gone",
          f"built {out['pump']['built']}, accepted {out['pump']['accepted']}")
    check(out["injector"].get("held", 0) == 0,
          "cloud_disconnect: with nothing left in the outbox",
          str(out["injector"].get("held")))

    # Duplicate upload. A retry the bridge could not know was unnecessary.
    out = drive(fault=faults.DUPLICATE_UPLOAD, ticks=60)
    print("  duplicate_upload         a bridge retrying an unacknowledged batch")
    invariants("duplicate_upload", out)
    check(out["pump"]["sent"] == 2 * out["pump"]["built"],
          "duplicate_upload: every event is delivered twice",
          f"built {out['pump']['built']}, sent {out['pump']['sent']}")
    check(out["pump"]["accepted"] == out["pump"]["built"],
          "duplicate_upload: and exactly half are accepted — deduplication by "
          "event_id, so a retry cannot double-count a fault",
          f"accepted {out['pump']['accepted']}")

    # Out of order. A backlog draining behind the live stream.
    out = drive(fault=faults.OUT_OF_ORDER, ticks=60)
    print("  out_of_order             a backlog arriving behind live data")
    invariants("out_of_order", out)
    check(out["injector"].get("reordered", 0) > 0,
          "out_of_order: events are genuinely delivered out of sequence",
          str(out["injector"].get("reordered")))
    check(out["pump"]["accepted"] == out["pump"]["built"],
          "out_of_order: every event still arrives — out of sequence is not "
          "lost", f"built {out['pump']['built']}, "
                  f"accepted {out['pump']['accepted']}")
    check(out["snapshot"]["stale"] is False,
          "out_of_order: and a late arrival never overwrites a fresher one — "
          "the panel stays current rather than stepping backwards",
          f"stale={out['snapshot']['stale']}")

    # Clock skew. A gateway whose clock is wrong.
    out = drive(fault=faults.CLOCK_SKEW, ticks=60)
    print("  clock_skew               the gateway's clock runs ahead")
    invariants("clock_skew", out)
    skewed = [e for e in out["latest"].values()
              if (e.get("metadata") or {}).get("clock_skew_s")]
    check(len(skewed) > 0,
          "clock_skew: the skew is recorded on the event rather than corrected "
          "— rewriting the timestamp would destroy the only evidence the "
          "gateway's clock is wrong", f"{len(skewed)} events carry it")
    # The consequence that matters, and the one this harness was written to
    # catch: a future-dated reading must not read as permanently fresh.
    check(out["snapshot"]["stale"] is True,
          "clock_skew: and a reading stamped in the future does not read as "
          "live — distance from now, not elapsed since",
          f"stale={out['snapshot']['stale']} "
          f"conn={out['snapshot']['connection_state']}")


# ---------------------------------------------------------------------------
# C. CODES — what the vehicle itself reports
# ---------------------------------------------------------------------------
# (ecu_scenario, ticks, expectations)
#
# `group` is the §17.3 group the code must land in and `label` its §17.6 status
# label. Both are strings the SERVER produced; this file never computes one.
CODES = (
    ("healthy", 300, {"codes": 0, "responding": True}),
    ("pending_mil_off", 300, {"codes": 1, "code": "P0171",
                              "label": "Repeated Pending", "early": True,
                              "mil": False}),
    ("pending_repeated", 300, {"codes": 1, "code": "P0171",
                               "group": "Repeated Pending Codes"}),
    ("pending_to_confirmed", 300, {"codes": 1, "code": "P0171",
                                   "label": "Active", "mil": True}),
    ("permanent", 300, {"codes": 1, "code": "P0217", "label": "Permanent",
                        "group": "Permanent Codes"}),
    ("mil_on", 300, {"codes": 1, "code": "P0300", "label": "Active",
                     "mil": True}),
    ("code_disappears", 300, {"codes": 1, "code": "P0171",
                              "label": "No Longer Reported",
                              "group": "Previously Observed Codes"}),
    ("code_recurs", 300, {"codes": 1, "code": "P0171"}),
    ("unknown_code", 300, {"codes": 1, "code": "P0468"}),
    ("manufacturer_code", 300, {"codes": 1, "code": "P1614",
                                "manufacturer": True}),
    ("multiple", 300, {"codes": 3}),
    ("no_permanent_support", 300, {"codes": 1, "code": "P0171"}),
    ("no_response", 300, {"codes": 0, "responding": False}),
)


def _cards(section):
    return [(g["title"], c) for g in section["groups"] for c in g["cards"]]


def run_codes():
    head("C -- CODES: what the vehicle itself reports")
    for name, ticks, want in CODES:
        out = drive(ecu_scenario=name, ticks=ticks)
        section = out["section"]
        cards = _cards(section)
        by_code = {c["code"]: (title, c) for title, c in cards}
        print(f"  {name:<24} {section['code_count']} code(s)")
        invariants(name, out)

        check(section["code_count"] == want["codes"],
              f"{name}: the section carries {want['codes']} code(s)",
              f"got {section['code_count']}: "
              f"{[c['code'] for _, c in cards]}")

        if "responding" in want:
            check(section["ecu_responding"] is want["responding"],
                  f"{name}: ecu_responding is {want['responding']} — "
                  f"'no answer' and 'no codes' are different claims",
                  str(section["ecu_responding"]))

        code = want.get("code")
        if code:
            check(code in by_code, f"{name}: {code} is present",
                  str(sorted(by_code)))
            if code in by_code:
                title, card = by_code[code]
                if "label" in want:
                    check(card["status_label"] == want["label"],
                          f"{name}: {code} is labelled {want['label']!r}",
                          card["status_label"])
                if "group" in want:
                    check(title == want["group"],
                          f"{name}: {code} is grouped under {want['group']!r}",
                          title)
                if "early" in want:
                    check(bool(card.get("early_detection")) is want["early"],
                          f"{name}: {code} is marked as detected before the "
                          f"warning light — the claim no code reader can make",
                          str(card.get("early_detection")))
                if want.get("manufacturer"):
                    check(bool(card.get("manufacturer_specific")),
                          f"{name}: {code} is flagged manufacturer-specific")
                    check(not card.get("possible_causes"),
                          f"{name}: and RIO offers no causes for it rather "
                          f"than guessing",
                          str(card.get("possible_causes")))
                # True of every card, in every scenario.
                check((card.get("cause_status") or "Not Confirmed")
                      == "Not Confirmed",
                      f"{name}: {code} states its cause is not confirmed",
                      str(card.get("cause_status")))
                check(bool(card.get("provenance_label")),
                      f"{name}: {code} says the VEHICLE reported it",
                      str(card.get("provenance_label")))

        if "mil" in want:
            check(section["mil_commanded_on"] is want["mil"],
                  f"{name}: the warning light is "
                  f"{'on' if want['mil'] else 'off'}",
                  str(section["mil_commanded_on"]))

        if name == "no_response":
            check(section["ecu_responding"] is False,
                  "no_response: an ECU that will not answer is never reported "
                  "as a car with no codes")

        if name == "no_permanent_support":
            check(section["services_supported"].get("permanent") is False,
                  "no_permanent_support: 'we could not ask about permanent "
                  "codes' is recorded, not reported as 'there are none'",
                  str(section["services_supported"]))

        # The report is the other surface these codes reach, so it is built
        # here rather than in a family of its own — a report of a vehicle with
        # no codes proves much less than one of a vehicle with three.
        report_invariants(name, run_report())


# ---------------------------------------------------------------------------

def main() -> int:
    global VERBOSE
    ap = argparse.ArgumentParser(description="§27.9 acceptance harness")
    ap.add_argument("--family", choices=("engine", "link", "codes"),
                    action="append", help="run only these (repeatable)")
    ap.add_argument("-v", "--verbose", action="store_true",
                    help="print every check, not only failures")
    args = ap.parse_args()
    VERBOSE = args.verbose
    families = args.family or ["engine", "link", "codes"]

    print("RIO vehicle health -- §27.9 acceptance harness")
    print(f"  {len(ENGINE)} engine conditions, 5 link conditions, "
          f"{len(CODES)} code conditions")
    print("  simulated clock; one tick = one second of scenario")

    started = _real_time()
    try:
        if "engine" in families:
            run_engine()
        if "link" in families:
            run_link()
        if "codes" in families:
            run_codes()
    finally:
        time.time = _real_time

    passed = sum(1 for ok, _, _ in _results if ok)
    total = len(_results)
    failed = [(l, d) for ok, l, d in _results if not ok]

    if _findings:
        head("OPEN FINDINGS -- true, wrong, and not asserted away")
        for name, title, detail in _findings:
            print(f"  [{name}] {title}")
            print(f"      {detail}")

    print("\n" + "=" * 74)
    print(f"{passed}/{total} checks passed"
          f"   ({_real_time() - started:.0f}s)")
    if _findings:
        print(f"{len(_findings)} open finding(s) -- see above")
    print("=" * 74)
    if failed:
        print("\nFAILED:")
        for label, detail in failed:
            print(f"  - {label}" + (f"  -- {detail}" if detail else ""))
    return 0 if not failed else 1


if __name__ == "__main__":
    sys.exit(main())
