"""Acceptance tests for the Vehicle Health conversation layer.

    python -m tools.vehicle_health_selftest
    python -m tools.vehicle_health_selftest --model     # + real RIO answers

Runs the real code in-process: the mock providers, the context layer, the
announcement policy, the router. No HTTP, no microphone, no speakers, and
without --model no network at all — which is what makes the whole of the
decision half of this feature runnable in CI.

Seven parts, separated by what each one is able to prove:

  A. CONTEXT over the spec's six scenarios — all healthy, one low, slow leak,
     critical pressure, sensor disconnected, no sensor data. What RIO is told,
     and whether it is true.

  B. ANNOUNCEMENT POLICY, scripted straight into VehicleHealthPolicy. The
     policy is pure, so its edge cases can be driven exactly rather than hoped
     for out of a mock: announce once, stay quiet after, remind, escalate on
     worsening, re-arm on resolve-and-return, hold the gap, and never speak at
     all below the threshold.

  C. TRUTHFULNESS — every issue carries an observation_window, and the window
     is the one the data actually supports.

  D. SPOKEN FORM — every line in the table is fully written out for speech.
     A digit reaching a synthesiser is a bug.

  E. ROUTER — the spec's eight driver questions land on the health path, and
     the questions that were already routed somewhere still go there.

  F. LLM FIREWALL — vehicle_health_policy.py must not be able to reach a model.
     Same check headway/live_selftest.py runs against live_policy.py, and for
     the same reason.

  G. SPEECH PRIORITY — the ladder, read out of the JavaScript. Static, because
     the behavioural half of this needs a browser or node: `node
     tools/nav_selftest.js` drives the real arbiter through the same cases and
     is where the interrupt semantics are actually proved.

--model additionally puts the spec's questions through the real conversation
path and PRINTS the answers. Nothing there is asserted beyond "she said
something": "does this sound like a knowledgeable passenger rather than a
diagnostic scanner" is not a thing to fake with a regex, and the point of
printing it in full is that a person reads it.
"""
import argparse
import ast
import os
import re
import sys
import tempfile
import time

sys.path.insert(0, "/workspace/rio-phase1")

from dotenv import load_dotenv                          # noqa: E402

load_dotenv("/workspace/rio-phase1/.env")

import config                                           # noqa: E402
import router                                           # noqa: E402
import telemetry                                        # noqa: E402
import tires                                            # noqa: E402
import vehicle_health as vh                             # noqa: E402
import vehicle_health_policy as P                       # noqa: E402

from tire_diag import engine as diag                    # noqa: E402
from tire_diag import store as diag_store               # noqa: E402
from powertrain_diag import engine as pdiag             # noqa: E402
from powertrain_diag import store as pdiag_store        # noqa: E402
from vehicle.dtc import service as dtc_service          # noqa: E402
from vehicle.producers import ecu as mock_ecu           # noqa: E402

# Every diagnostic record goes somewhere disposable. A test suite able to write
# into a car's real fault history would be worse than no test suite.
diag_store.reset_for_test(tempfile.mkdtemp(prefix="vh_selftest_"))
pdiag_store.reset_for_test(tempfile.mkdtemp(prefix="vh_selftest_pt_"))
dtc_service.service().reset_for_test(tempfile.mkdtemp(prefix="vh_selftest_dtc_"))

ROOT = "/workspace/rio-phase1"

PASS, FAIL = [], []


def check(cond, label, detail=""):
    (PASS if cond else FAIL).append(label)
    print(f"  [{'PASS' if cond else 'FAIL'}] {label}" + (f"  -- {detail}" if detail else ""))
    return bool(cond)


def head(title):
    print(f"\n{title}")


# Synthetic time, so a monitor's confirmation criteria can actually be met.
_CLOCK = [1_800_000_000.0]


def scenario(tire_name, ecu_name="normal_idle", drive_s=420.0, dt=30.0,
             moving=None, dtc_name="healthy"):
    """Put the car in a known state, DRIVE IT, and read the context out of it.

    The driving is the change this spec brought, and it is not a detail. RIO's
    tire context now comes from tire_diag, where a finding has to survive its
    monitor's confirmation criteria before it becomes an Issue at all — so a
    scenario that is merely selected produces no issues, correctly. Getting a
    warning out of it requires the same evidence a real car would have to
    supply: several reports, spaced at the sensor's real cadence, agreeing with
    each other.

    That is exactly what the old version of this function could not express,
    and why it asserted a behaviour (one reading, one issue) that has since been
    deliberately removed.
    """
    telemetry.set_scenario(ecu_name)
    eng = diag.reset_engine(load=False)
    # The powertrain monitors and the DTC layer are driven too, because app.py's
    # announcement poll drives all three and a harness that only fed one would
    # be testing a system that does not exist. Without this the two new health
    # sources are never available, and "everything is normal" becomes an answer
    # about a car nobody has looked at.
    peng = pdiag.reset_engine(load=False)
    svc = dtc_service.service()
    svc.reset_for_test(tempfile.mkdtemp(prefix="vh_dtc_"))
    mock_ecu.ecu().set_scenario(dtc_name, now=_CLOCK[0])
    t = _CLOCK[0]
    _CLOCK[0] += drive_s + 600.0        # never reuse a window between scenarios
    tires.set_scenario(tire_name, at=t)

    if moving is None:
        moving = ecu_name not in ("normal_idle",)
    speed = 45.0 if moving else 0.0

    for _ in range(max(1, int(drive_s / dt))):
        t += dt
        tires.set_motion(moving, t)
        eng.observe(tires.snapshot(at=t), now=t, moving=moving, speed_mph=speed)
        svc.scan(t, "vh-selftest", reason="test")
        peng.observe(telemetry.snapshot(record=False), now=t, moving=moving,
                     speed_mph=speed,
                     dtc={"scanned": True, "responding": True,
                          "codes": [r["code"] for r in svc.registry.active_codes()],
                          "mil": svc.registry.mil(), "count": 0,
                          "worst_health_severity": "advisory"},
                     link={"source": "mock_holley"})

    return vh.context(full=True)["vehicle_health"]


# ===========================================================================
# A. Context — the six scenarios from the spec
# ===========================================================================

def run_context():
    head("A -- vehicle health context over the spec's six scenarios")

    # --- nothing evaluated yet: the honest answer is not "fine" ---
    diag.reset_engine(load=False)
    tires.set_scenario("all_normal")
    fresh = vh.context(full=False)["vehicle_health"]
    check(fresh["overall_status"] == "informational",
          "before any monitor has run, the car is NOT reported as normal",
          fresh["overall_status"])
    # The tire monitors say so in as many words. The summary now leads with
    # whichever unevaluated subsystem sorts first — there are three of them on a
    # cold start, and that is more honest rather than less.
    full = vh.context(full=True)["vehicle_health"]
    messages = " | ".join(i["message"] for i in full["issues"])
    check("not the same as the tires being fine" in messages,
          "and the tire monitors say so in as many words", messages[:90])
    check("normal" not in fresh["summary"].lower()
          and "fine" not in fresh["summary"].lower(),
          "while the summary refuses to claim anything is fine",
          fresh["summary"])
    unavailable = {i["domain"] for i in vh.issues()
                   if i["type"].endswith("_unavailable")
                   or "not_ready" in i["type"]}
    check(len(unavailable) >= 2,
          "and every subsystem that has not been looked at says so — a health "
          "layer that only admitted the gaps it happened to notice would be "
          "reporting on the parts it found easiest to check",
          str(sorted(unavailable)))

    # --- all tires healthy ---
    st = scenario("all_normal")
    check(st["overall_status"] == "normal",
          "all healthy, once the monitors have actually run -> normal",
          st["overall_status"])
    check(st["issue_count"] == 0, "all healthy -> no issues", str(st["issue_count"]))
    check("normal" in st["summary"].lower(),
          "all healthy -> a summary that says so", st["summary"])
    corners = st["tires"]["corners"]
    check(set(corners) == {"front_left", "front_right", "rear_left", "rear_right"},
          "all four corners are present in the structure")
    check(all(c["pressure"] is not None for c in corners.values()),
          "every corner carries a pressure")

    # --- one tire low ---
    st = scenario("one_low")
    check(st["overall_status"] == "warning",
          "one low, CONFIRMED across monitor runs -> warning",
          st["overall_status"])
    check(st["issue_count"] >= 1, "one low -> at least one confirmed issue",
          str(st["issue_count"]))
    iss = st["issues"][0]
    check(iss["severity"] == "warning", "one low -> severity warning",
          iss["severity"])
    check(iss["where"] == "rear left", "one low -> names the corner", str(iss["where"]))
    check(corners_status(st, "rear_left") == "warning",
          "and the structure marks that corner, not another one")

    # --- slow leak ---
    # Driven long enough for the pressure to cross the warning threshold, which
    # is what the low-pressure monitor confirms. The slow-leak MONITOR needs a
    # thirty-minute thermally-comparable window and is covered end to end in
    # tire_diag/selftest.py; what is asserted here is that whatever is confirmed
    # arrives with an observation window derived from its own evidence.
    st = scenario("slow_leak", drive_s=600.0)
    check(st["issue_count"] >= 1, "slow leak -> something is confirmed",
          str(st["issue_count"]))
    windows = {i["observation_window"] for i in st["issues"]}
    check(all(w for w in windows), "and every issue carries a window", str(windows))
    check(st["tires"]["corners"]["rear_left"]["trend"] == "dropping",
          "slow leak -> the corner reports a dropping trend",
          str(st["tires"]["corners"]["rear_left"]["trend"]))

    # --- critical pressure ---
    st = scenario("one_critical")
    check(st["overall_status"] == "critical", "critical pressure -> overall critical",
          st["overall_status"])
    check(st["issues"][0]["severity"] == "critical",
          "critical pressure -> the issue is critical", st["issues"][0]["severity"])
    check(st["issues"][0]["suggested_action"],
          "critical pressure -> carries a suggested action",
          str(st["issues"][0]["suggested_action"]))

    # --- sensor disconnected, parked and moving ---
    st = scenario("sensor_disconnected", "normal_idle", moving=False)
    parked_sev = [i["severity"] for i in st["issues"]]
    check("critical" not in parked_sev,
          "a sensor offline while PARKED is never critical — sleeping sensors "
          "are not a fault", str(parked_sev))

    st = scenario("sensor_disconnected", "cruise", moving=True)
    types = [i["type"] for i in st["issues"]]
    check(any("sensor" in t for t in types),
          "a sensor offline while MOVING is reported", str(types))
    check(st["moving"] is True, "and the context knows the car is moving")

    # --- no sensor data ---
    st = scenario("no_sensors", "normal_idle", moving=False)
    types = [i["type"] for i in st["issues"]]
    check("tires_unavailable" in types,
          "no sensors -> says so rather than reporting healthy tires", str(types))
    check("normal" not in st["summary"].lower(),
          "no sensors -> the summary does NOT claim everything is fine",
          st["summary"])
    check("tires" not in st["subsystems_reporting"],
          "no sensors -> tires are not listed as reporting")

    # --- no UI state anywhere in the payload ---
    st = scenario("one_critical", "cruise", moving=True)
    blob = repr(st)
    leaked = [w for w in ("status_class", "poll_ms", "trend_glyph", "banners",
                          "scenario", "veh-", "css", "colour", "color")
              if w in blob]
    check(not leaked, "no UI state reaches the conversation context", str(leaked))


def corners_status(state, corner):
    return state["tires"]["corners"][corner]["status"]


# ===========================================================================
# B. The announcement policy
# ===========================================================================
# Scripted, not driven through the mocks: the policy is pure, so every branch
# can be reached exactly rather than by finding a scenario that happens to
# produce it. The mock-driven half is covered in A and in the end-to-end check
# at the bottom of this section.

def issue(key="tires:RL:critical_low_pressure", severity="critical",
          magnitude=7.0, type_="critical_low_pressure", location="rear left",
          value=26.0, unit="psi", healing_runs=0):
    out = vh.make_issue(key=key, domain="tires", type=type_, severity=severity,
                        message="scripted", observation_window=vh.W_NOW,
                        location=location, magnitude=magnitude, value=value,
                        unit=unit, spoken_fallback="Something is wrong.")
    # Consecutive passing monitor runs. Above zero means the fault is measurably
    # better right now, whatever its lifecycle still says.
    out["healing_runs"] = healing_runs
    return out


def run_policy():
    head("B -- announcement policy: once, then quiet")

    # --- informational and warning never speak on their own ---
    pol = P.VehicleHealthPolicy()
    r = pol.tick([issue(severity="informational", magnitude=1.0)], 0.0)
    check(r["announce"] is None, "informational never speaks automatically",
          r["reason"])
    r = pol.tick([issue(severity="warning", magnitude=3.0)], 10.0)
    check(r["announce"] is None, "warning never speaks automatically", r["reason"])
    check(r["reason"] == P.R_BELOW_THRESHOLD,
          "and the reason recorded is the threshold, not a cooldown", r["reason"])

    # --- critical speaks exactly once ---
    pol = P.VehicleHealthPolicy()
    spoken = []
    for i in range(20):
        r = pol.tick([issue()], i * 3.0)
        if r["announce"]:
            spoken.append(r["announce"])
    check(len(spoken) == 1, "critical speaks exactly once over a minute of polling",
          f"{len(spoken)} announcements")
    check(spoken and spoken[0]["reason"] == P.R_FIRST,
          "and the first one is reported as first_time")

    # --- the fault stays visible: it is still in the issue list ---
    r = pol.tick([issue()], 100.0)
    check(r["announce"] is None and r["reason"] == P.R_ALREADY,
          "the issue is still tracked and still suppressed, not forgotten",
          r["reason"])

    # --- reminder after the configured interval ---
    r = pol.tick([issue()], 3.0 + P.REMIND_S)
    check(r["announce"] is not None and r["announce"]["reason"] == P.R_REMINDER,
          f"a reminder is allowed after {int(P.REMIND_S)}s",
          r["reason"])

    # --- ...but not while the fault is recovering ---
    # The shadow log's first finding: the driver added air, the pressure was
    # good from that moment, and the reminder fired four and a half minutes
    # before the healing criteria finished verifying it. RIO would have told
    # them to pull over for a tire they had just dealt with.
    pol = P.VehicleHealthPolicy()
    pol.tick([issue()], 0.0)
    r = pol.tick([issue(healing_runs=1)], P.REMIND_S + 10)
    check(r["announce"] is None,
          "no reminder while the fault is passing its monitor again",
          r["reason"])
    check(r["reason"] == P.R_HEALING,
          "and the silence is attributed to healing, not to a cooldown",
          r["reason"])
    # One passing run is NOT resolution -- the issue is still ACTIVE and still
    # on the dashboard. The bar for staying quiet is deliberately lower than the
    # bar for declaring a repair.
    r = pol.tick([issue(healing_runs=0)], P.REMIND_S + 20)
    check(r["announce"] is not None and r["announce"]["reason"] == P.R_REMINDER,
          "and if it starts failing again the reminder comes straight back",
          r["reason"])

    # --- worsening beats the cooldown ---
    pol = P.VehicleHealthPolicy()
    pol.tick([issue(magnitude=7.0)], 0.0)
    r = pol.tick([issue(magnitude=7.2)], 30.0)
    check(r["announce"] is None, "drifting slightly worse does NOT re-announce",
          r["reason"])
    r = pol.tick([issue(magnitude=11.0)], 60.0)
    check(r["announce"] is not None and r["announce"]["reason"] == P.R_WORSENED,
          "getting materially worse does re-announce", r["reason"])

    # --- rank escalation beats the cooldown ---
    pol = P.VehicleHealthPolicy()
    pol.tick([issue(key="k", severity="critical", magnitude=1.0)], 0.0)
    r = pol.tick([issue(key="k", severity="critical", magnitude=1.0)], 30.0)
    check(r["announce"] is None, "unchanged -> silent")

    # --- resolve, then return ---
    pol = P.VehicleHealthPolicy()
    pol.tick([issue()], 0.0)
    pol.tick([], 30.0)
    r = pol.tick([issue()], 45.0)
    check(r["announce"] is None,
          "a fault that flickers back within the clear window does not repeat",
          r["reason"])
    pol = P.VehicleHealthPolicy()
    pol.tick([issue()], 0.0)
    pol.tick([], 30.0)
    r = pol.tick([issue()], 30.0 + P.RESOLVED_CLEAR_S + 5)
    check(r["announce"] is not None and r["announce"]["reason"] == P.R_RETURNED,
          "a fault that genuinely resolved and came back is a NEW event",
          r["reason"])

    # --- two criticals do not stack ---
    pol = P.VehicleHealthPolicy()
    a = issue(key="a", magnitude=9.0)
    b = issue(key="b", magnitude=8.0, type_="tire_overheating",
              location="front right", value=190.0, unit="f")
    r1 = pol.tick([a, b], 0.0)
    r2 = pol.tick([a, b], 5.0)
    check(r1["announce"] and r1["announce"]["key"] == "a",
          "the worse of two criticals goes first")
    check(r2["announce"] is None and r2["reason"] == P.R_MIN_GAP,
          "the second waits rather than talking over the first", r2["reason"])
    r3 = pol.tick([a, b], 5.0 + P.MIN_GAP_S)
    check(r3["announce"] and r3["announce"]["key"] == "b",
          "and follows once there is room for it")

    # --- the driver asking resets the cooldown, and buys quiet ---
    pol = P.VehicleHealthPolicy()
    pol.tick([issue()], 0.0)
    pol.note_status_request(100.0)
    r = pol.tick([issue()], 105.0)
    check(r["announce"] is None and r["reason"] == P.R_POST_REQUEST,
          "RIO does not parrot back the fault she was just asked about",
          r["reason"])
    r = pol.tick([issue()], 100.0 + P.POST_REQUEST_QUIET_S + 5)
    check(r["announce"] is not None,
          "but the cooldown really was reset — it can speak again after",
          r["reason"])

    # --- end to end through the mock, which is what actually ships ---
    head("B -- the same, driven through the mock provider and the monitors")
    scenario("all_normal")
    pol = P.VehicleHealthPolicy()
    r = pol.tick(vh.issues(), 0.0)
    check(r["announce"] is None, "a healthy car produces no announcement",
          r["reason"])

    # A confirmed critical tire. Note what the policy does with it now: the
    # decision is made in full and then NOT carried out, because shadow mode is
    # the shipped default and no diagnostic code has been cleared to speak.
    scenario("one_critical", "cruise", moving=True)
    issues = vh.issues()
    r = pol.tick(issues, 30.0)
    check(r["announce"] is None,
          "a confirmed critical tire still does not speak — shadow mode",
          r["reason"])
    check(r["reason"] == P.R_SHADOW, "and the reason is recorded as such",
          r["reason"])
    said = (r.get("proposal") or {}).get("text", "")
    print(f"        RIO would have said: {said!r}")
    check("rear left" in said, "the proposal names the corner", said)
    check(all(not i.get("announce_allowed") for i in issues
              if i["domain"] == "tires"),
          "and every tire issue is marked not-announceable")

    for i in range(5):
        r2 = pol.tick(vh.issues(), 33.0 + i * 3)
        if r2.get("proposal"):
            break
    check(not r2.get("proposal"),
          "and it does not re-propose on the next five polls — a proposal "
          "consumes the cooldown exactly as an announcement would",
          r2.get("reason", ""))


# ===========================================================================
# C. Truthfulness
# ===========================================================================

def run_truthfulness():
    head("C -- RIO is only given claims the data supports")

    windows = set()
    for tire_sc in ("one_low", "slow_leak", "one_critical", "sensor_disconnected",
                    "stale_data", "overheating", "battery_low", "no_sensors"):
        st = scenario(tire_sc, "cruise", moving=True)
        for iss in st["issues"]:
            ok = bool(iss.get("observation_window"))
            windows.add(iss.get("observation_window"))
            if not ok:
                check(False, f"{tire_sc}: every issue carries an observation_window",
                      iss["type"])
                return
    check(True, "every issue in every scenario carries an observation_window",
          f"{len(windows)} distinct windows")

    # The windows must be ones the data can actually support. The tire trend is
    # a 24-hour delta and the engine ring is TELEMETRY_TREND_WINDOW_S seconds;
    # anything claiming more than that is the fabrication this whole field
    # exists to prevent.
    # Windows are now DERIVED from the evidence each monitor actually used, so
    # the allowed set is a shape rather than a fixed list: "this moment only",
    # or a span in seconds/minutes/hours that the monitor genuinely spanned.
    # Anything naming a week or a month would be a claim about data that does
    # not exist.
    bad = [w for w in windows
           if w != vh.W_NOW and not re.match(
               r"^the last \d+ (seconds|minutes|hours|days)$", w)]
    check(not bad, "and every window is one the evidence can support", str(bad))
    check(not any("week" in w or "month" in w or "year" in w for w in windows),
          "with no window claiming more history than exists", str(sorted(windows)))

    st = scenario("slow_leak", "cruise", moving=True)
    depth = st["history_depth"]
    check("24 hours" in depth, "history_depth names the real tire window", depth)
    check("week" not in depth.lower() and "month" not in depth.lower(),
          "and does not offer a longer one")

    # The prompt has to actually forbid it, or the field is decoration.
    import rio_prompts
    prompt = rio_prompts.RIO_SYSTEM_PROMPT
    check("observation_window" in prompt,
          "the system prompt names observation_window")
    check("past few weeks" in prompt or "past the" in prompt or "may not go past" in prompt,
          "and forbids exceeding it in so many words")
    check("Interpret. Never recite." in prompt,
          "and asks for interpretation rather than recitation")

    # A corner with no trend data must not be given one.
    st = scenario("all_normal")
    for name, c in st["tires"]["corners"].items():
        if c["change_over_24h"] is None and c["trend"] is not None:
            check(False, "a corner with no history is given no trend", name)
            return
    check(True, "a corner with no history is given no trend")


# ===========================================================================
# D. Spoken form
# ===========================================================================

def run_spoken():
    head("D -- announcements are written to be heard, not read")

    check(P.spoken_psi(29) == "twenty-nine P S I", "29 -> 'twenty-nine P S I'",
          P.spoken_psi(29))
    check(P.spoken_psi(29.3) == "twenty-nine P S I",
          "29.3 rounds the way a person says it, like nav's distances",
          P.spoken_psi(29.3))
    check(P.spoken_int(171) == "one hundred seventy-one", "171 in words",
          P.spoken_int(171))
    check(P.spoken_value(12.1, "V") == "twelve point one volts",
          "voltage keeps the tenth, because the tenth is the meaning",
          P.spoken_value(12.1, "V"))

    digits = re.compile(r"\d")
    bad = []
    for type_ in P.LINE:
        text = P.compose({"type": type_, "location": "rear left", "value": 29.0,
                          "unit": "f" if ("heat" in type_ or "hot" in type_
                                          or "coolant" in type_) else "psi"})
        if digits.search(text):
            bad.append((type_, text))
        if not text:
            bad.append((type_, "<empty>"))
    check(not bad, f"no digit reaches the synthesiser in any of {len(P.LINE)} lines",
          str(bad))

    # A type with no line still says something rather than nothing.
    text = P.compose({"type": "future_domain_fault_nobody_wrote_a_line_for",
                      "spoken_fallback": "Something's up with the gearbox."})
    check(text == "Something's up with the gearbox.",
          "an unknown fault type falls back to a sentence rather than silence",
          text)
    check(P.compose({"type": "unknown"}) == "",
          "and a fault with nothing sayable at all is refused, not mumbled")

    # No diagnostic-scanner vocabulary in any line. The bible bans it and this
    # is the one path where a template could smuggle it in.
    banned = ("warning", "critical", "sensor reading", "code", "fault code",
              "psi.", "telemetry", "status")
    hits = [(k, w) for k, v in P.LINE.items() for w in banned if w in v.lower()]
    check(not hits, "no line sounds like a diagnostic scanner", str(hits))


# ===========================================================================
# E. Router
# ===========================================================================

SPEC_QUESTIONS = [
    "How are my tires?",
    "Is everything okay?",
    "Which tire is low?",
    "How much air is in my tires?",
    "Are my tires healthy?",
    "Is there anything wrong?",
    "Give me a vehicle health report.",
    "What should I be worried about?",
]

OTHER_QUESTIONS = [
    ("What do you see?", "scene_description"),
    ("What kind of car is that on the left?", "specific_object_question"),
    ("What does that sign say?", "read_visible_text"),
    ("Where are we?", "location_or_landmark_question"),
    ("Hey.", "non_visual_question"),
    ("Play some music", "non_visual_question"),
    ("How far to the next exit?", "non_visual_question"),
]


def run_router():
    head("E -- the router, with the model fallback off (offline, deterministic)")

    missed = []
    for q in SPEC_QUESTIONS:
        r = router.classify(q, use_model=False)
        if r["request_type"] != router.VEHICLE_HEALTH:
            missed.append((q, r["request_type"]))
    check(not missed, "all eight of the spec's driver questions route to health",
          str(missed))

    wrong = []
    for q, expected in OTHER_QUESTIONS:
        r = router.classify(q, use_model=False)
        if r["request_type"] != expected:
            wrong.append((q, r["request_type"], expected))
    check(not wrong, "and nothing that was already routed elsewhere moved",
          str(wrong))

    r = router.classify("How are my tires?", use_model=False)
    check(not router.is_visual(r["request_type"]),
          "a health question never reaches the camera path")
    check(not r["requires_full_frame"] and not r["requires_object_crop"],
          "and asks for no frame and no crop")
    check(router.is_vehicle_health(r["request_type"]),
          "is_vehicle_health identifies it")

    # There is ONE router. A second classifier would eventually disagree with
    # this one, so assert that health lives in the same module as everything
    # else rather than in a file of its own.
    src = open(os.path.join(ROOT, "router.py")).read()
    check("vehicle_health_question" in src,
          "the health type lives in the existing router, not a second one")
    others = [f for f in os.listdir(ROOT)
              if f.endswith(".py") and "router" in f and f != "router.py"]
    check(not others, "there is no second router module", str(others))


# ===========================================================================
# F. LLM firewall
# ===========================================================================
# Identical in shape to headway/live_selftest.py's run_firewall, and stricter:
# live_policy.py is allowed math and headway.state, this module is allowed
# nothing at all.

FORBIDDEN_NAMES = ["openai", "config", "tires", "telemetry", "vehicle_health",
                   "llm_interface", "app", "requests", "httpx", "vision",
                   "visual_qa", "insights", "json", "os", "sys"]


def run_firewall():
    head("F -- LLM firewall: vehicle_health_policy.py cannot reach a model")

    path = os.path.join(ROOT, "vehicle_health_policy.py")
    src = open(path).read()
    tree = ast.parse(src)

    imported = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            imported.update(a.name.split(".")[0] for a in node.names)
        elif isinstance(node, ast.ImportFrom):
            if node.module:
                imported.add(node.module.split(".")[0])
            imported.update(a.name for a in node.names if node.level and not node.module)

    for mod in FORBIDDEN_NAMES:
        check(mod not in imported, f"does not import {mod!r}")
    check(imported == set(), "imports NOTHING AT ALL — a stronger claim than "
                             "'does not call a model'", f"imports: {sorted(imported)}")

    calls = [n.func.id for n in ast.walk(tree)
             if isinstance(n, ast.Call) and isinstance(n.func, ast.Name)]
    for builtin in ("open", "eval", "exec", "compile", "__import__", "input"):
        check(builtin not in calls, f"never calls {builtin}()")

    # Nothing that even NAMES a model anywhere in the executable half. Prose is
    # exempt — the docstrings exist precisely to explain the firewall and would
    # have to name what is being kept out — so this looks at identifiers,
    # attributes and non-docstring string literals rather than at the file's
    # text, which is the difference between checking the code and checking the
    # comments.
    docstrings = set()
    for node in ast.walk(tree):
        if isinstance(node, (ast.Module, ast.ClassDef, ast.FunctionDef,
                             ast.AsyncFunctionDef)):
            body = getattr(node, "body", None) or []
            if body and isinstance(body[0], ast.Expr) \
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

    for word in ("openai", "gpt", "claude", "llm", "model", "prompt",
                 "completion", "http", "socket"):
        check(word not in surface_text,
              f"the word {word!r} appears nowhere in the executable code")

    # And the module is genuinely importable on its own, with nothing loaded.
    check(P.VehicleHealthPolicy().tick([], 0.0)["announce"] is None,
          "and it runs standalone: an empty tick decides silence")

    # The decision path really is the deterministic one: app.py must tick the
    # policy from the endpoint, not ask anything else.
    app_src = open(os.path.join(ROOT, "app.py")).read()
    check("_health_policy.tick(" in app_src,
          "app.py ticks the deterministic policy for announcements")
    idx = app_src.find("def vehicle_health_announcement_endpoint")
    body = app_src[idx:idx + 1400] if idx >= 0 else ""
    check(idx >= 0 and "llm_interface" not in body and "client." not in body,
          "and the announcement endpoint touches no model")


# ===========================================================================
# G. Speech priority
# ===========================================================================
# Static. The behavioural proof — pre-emption, supersede-by-group, queue order —
# is `node tools/nav_selftest.js`, which drives the real arbiter through the
# same cases. This asserts the ladder itself and that nobody kept a private copy
# of it, which are the two things that break silently when a tier is inserted.

def run_priority():
    head("G -- the speech ladder (static; run `node tools/nav_selftest.js` for behaviour)")

    speech = open(os.path.join(ROOT, "static", "rio_speech.js")).read()
    m = re.search(r"var P = \{([^}]*)\}", speech)
    check(bool(m), "rio_speech.js declares the priority table")
    if not m:
        return
    table = dict(re.findall(r"(\w+):\s*(\d+)", m.group(1)))
    order = {k: int(v) for k, v in table.items()}
    print(f"        {order}")

    check("VEHICLE_HEALTH" in order, "there is a VEHICLE_HEALTH tier")
    check(order.get("SAFETY", 99) < order.get("VEHICLE_HEALTH", 0),
          "vehicle health never outranks a safety warning")
    check(order.get("VEHICLE_HEALTH", 99) < order.get("TURN_NEAR", 0),
          "and does outrank the near-tier turn — a blowout beats a junction")
    check(order.get("TURN_NEAR", 99) < order.get("NAV", 0) < order.get("CONVO", 0),
          "the rest of the ladder is unchanged: near turn > nav > conversation")

    # The navigation side of the ladder lives in the speech planner now, not
    # in the panel: rio_nav.js paints and plays audio, rio_navplan.js decides
    # what is worth saying and at which tier.
    nav = open(os.path.join(ROOT, "static", "rio_nav.js")).read()
    plan = open(os.path.join(ROOT, "static", "rio_navplan.js")).read()
    check("priority: 2" not in nav and "priority: 3" not in nav
          and "priority: 2" not in plan and "priority: 3" not in plan,
          "neither the nav panel nor the planner holds a hardcoded priority number")
    check("P.TURN_NEAR" in plan and "P.NAV" in plan,
          "the planner names the tiers, and takes them from the arbiter")

    # The planner carries a fallback table for the case where it is built
    # without an arbiter (the tests do exactly that). A fallback that disagreed
    # with rio_speech.js would silently reorder the ladder, so it is checked
    # against the real one rather than trusted.
    fb = re.search(r"arbiter\.P\)\s*\|\|\s*\{([^}]*)\}", plan)
    check(bool(fb), "the planner's fallback priority table is findable")
    if fb:
        fallback = {k: int(v) for k, v in re.findall(r"(\w+):\s*(\d+)", fb.group(1))}
        check(all(order.get(k) == v for k, v in fallback.items()),
              f"and it agrees with rio_speech.js ({fallback})")

    health = open(os.path.join(ROOT, "static", "rio_health.js")).read()
    check("RIO.speech.say(" in health,
          "rio_health.js goes through the existing arbiter")
    check("RIO.speech.P.VEHICLE_HEALTH" in health, "at the vehicle health tier")
    check("makeArbiter" not in health and "new Audio()" in health,
          "and builds no arbiter of its own — one mouth, one queue")
    check("group: 'health'" in health,
          "with its own supersede group, so a worse fault replaces a lesser one")


# ===========================================================================
# H. Dashboard and conversation agree
# ===========================================================================

def run_sync():
    head("H -- the dashboard and the conversation describe the same car")

    # tires.py's headline status and the context's overall status are computed
    # from the same classification, and this is the check that keeps them that
    # way: a threshold added to one and not the other shows up here first.
    # The relationship the two now have, and it is NOT equality.
    #
    # The panel is the instantaneous view: this is what the sensor says right
    # now. The context is the diagnostic view: this is what has been confirmed.
    # They differ, legitimately, in exactly one direction — the panel can show a
    # problem the monitors have not confirmed yet, because that is what "one
    # reading is not a fault" means. What must never happen is the reverse: a
    # confirmed diagnostic issue that the panel is not showing would be RIO
    # telling a driver about something they cannot then look at.
    order = {"NORMAL": 0, "UNAVAILABLE": 1, "ATTENTION": 2, "CRITICAL": 3}
    ctx_order = {"normal": 0, "informational": 1, "advisory": 2, "warning": 2,
                 "critical": 3, "unknown": 1}
    for tire_sc in ("all_normal", "one_low", "one_critical", "overheating",
                    "sensor_disconnected", "battery_low", "no_sensors"):
        st = scenario(tire_sc, "cruise", moving=True)
        panel = tires.snapshot()["status"]
        # The TIRE half of the context. The overall status now also carries the
        # vehicle's own codes and the engine monitors' findings, and comparing
        # those against a tire panel would be comparing two different questions.
        tire_issues = [i for i in vh.issues() if i["domain"] == "tires"]
        ctx = vh._STATUS_BY_RANK.get(
            max((i["severity_rank"] for i in tire_issues), default=0), "normal")
        ok = ctx_order.get(ctx, 0) <= order.get(panel, 0)
        if not check(ok, f"{tire_sc}: the context never outruns the panel "
                         f"(panel {panel}, context {ctx})"):
            return

    # And when a fault IS confirmed, both name the same corner.
    st = scenario("one_critical", "cruise", moving=True)
    banners = tires.snapshot()["banners"]
    issues = [i for i in vh.issues() if i["domain"] == "tires"]
    check(banners and issues,
          "a confirmed critical fault produces both a banner and an issue")
    check(any("Rear Left" in b["title"] for b in banners)
          and any(i["location"] == "rear left" for i in issues),
          "and both of them name the same corner")


# ===========================================================================
# I. Real answers (--model)
# ===========================================================================

def run_model():
    head("I -- real conversation turns. Read these; nothing here is asserted "
         "beyond 'she answered'")

    import llm_interface

    plans = [
        ("all_normal", "cruise", "How are my tires?"),
        ("slow_leak", "cruise", "How are my tires?"),
        ("slow_leak", "cruise", "Which tire is low?"),
        ("one_critical", "cruise", "Is there anything wrong?"),
        ("no_sensors", "cruise", "Are my tires healthy?"),
        ("all_normal", "cruise", "Hey."),
    ]
    for tire_sc, ecu_sc, question in plans:
        # DRIVE the scenario, do not merely select it. RIO's tire context now
        # comes from confirmed diagnostics, so a freshly-selected scenario
        # honestly produces "I haven't been able to evaluate your tires yet" —
        # correct, and not what these examples are for.
        scenario(tire_sc, ecu_sc, moving=True)
        route = router.classify(question, use_model=False)
        t0 = time.time()
        try:
            reply = "".join(llm_interface.generate_stream(question, route)).strip()
        except Exception as e:
            check(False, f"{tire_sc}: {question}", f"{type(e).__name__}: {e}")
            continue
        ms = (time.time() - t0) * 1000
        print(f"\n    scenario: {tire_sc} / {ecu_sc}   route: {route['request_type']}")
        print(f"    driver:   {question}")
        print(f"    RIO:      {reply}")
        print(f"              ({ms:.0f} ms)")
        check(bool(reply) or question == "Hey.", f"{tire_sc}: RIO answered")
        # The one mechanical thing worth asserting about an answer: she must not
        # have invented a longer history than the data has.
        for phrase in ("past few weeks", "past week", "for weeks", "last month",
                       "over the years", "past several days"):
            if phrase in reply.lower():
                check(False, f"{tire_sc}: no invented timespan", phrase)
                break


# ===========================================================================

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", action="store_true",
                    help="also run real conversation turns through the LLM")
    args = ap.parse_args()

    print("=" * 72)
    print("RIO vehicle health — conversation layer + critical announcements")
    print("=" * 72)

    run_context()
    run_policy()
    run_truthfulness()
    run_spoken()
    run_router()
    run_firewall()
    run_priority()
    run_sync()
    if args.model:
        run_model()

    print("\n" + "=" * 72)
    total = len(PASS) + len(FAIL)
    print(f"{len(PASS)}/{total} checks passed")
    if FAIL:
        print("\nFAILED:")
        for f in FAIL:
            print(f"  - {f}")
    print("=" * 72)
    return 1 if FAIL else 0


if __name__ == "__main__":
    sys.exit(main())
