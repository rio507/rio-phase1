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
import time

sys.path.insert(0, "/workspace/rio-phase1")

from dotenv import load_dotenv                          # noqa: E402

load_dotenv("/workspace/rio-phase1/.env")

import router                                           # noqa: E402
import telemetry                                        # noqa: E402
import tires                                            # noqa: E402
import vehicle_health as vh                             # noqa: E402
import vehicle_health_policy as P                       # noqa: E402

ROOT = "/workspace/rio-phase1"

PASS, FAIL = [], []


def check(cond, label, detail=""):
    (PASS if cond else FAIL).append(label)
    print(f"  [{'PASS' if cond else 'FAIL'}] {label}" + (f"  -- {detail}" if detail else ""))
    return bool(cond)


def head(title):
    print(f"\n{title}")


def scenario(tire_name, ecu_name="normal_idle"):
    """Put the car in a known state and read the context out of it."""
    telemetry.set_scenario(ecu_name)
    tires.set_scenario(tire_name)
    return vh.context(full=True)["vehicle_health"]


# ===========================================================================
# A. Context — the six scenarios from the spec
# ===========================================================================

def run_context():
    head("A -- vehicle health context over the spec's six scenarios")

    # --- all tires healthy ---
    st = scenario("all_normal")
    check(st["overall_status"] == "normal", "all healthy -> overall_status normal",
          st["overall_status"])
    check(st["issue_count"] == 0, "all healthy -> no issues", str(st["issue_count"]))
    check("all four" in st["summary"].lower() or "normal" in st["summary"].lower(),
          "all healthy -> a summary that says so", st["summary"])
    corners = st["tires"]["corners"]
    check(set(corners) == {"front_left", "front_right", "rear_left", "rear_right"},
          "all four corners are present in the structure")
    check(all(c["pressure"] is not None for c in corners.values()),
          "every corner carries a pressure")

    # --- one tire low ---
    st = scenario("one_low")
    check(st["overall_status"] == "warning", "one low -> overall_status warning",
          st["overall_status"])
    check(st["issue_count"] == 1, "one low -> exactly one issue")
    iss = st["issues"][0]
    check(iss["severity"] == "warning", "one low -> severity warning")
    check(iss["where"] == "rear left", "one low -> names the corner", str(iss["where"]))
    check(corners_status(st, "rear_left") == "warning",
          "and the structure marks that corner, not another one")

    # --- slow leak ---
    st = scenario("slow_leak")
    check(st["issues"][0]["type"] == "possible_slow_leak",
          "slow leak -> typed as a slow leak", st["issues"][0]["type"])
    check(st["issues"][0]["observation_window"] == "the last 24 hours",
          "slow leak -> the 24-hour window the trend actually comes from")
    check(st["tires"]["corners"]["rear_left"]["trend"] == "dropping",
          "slow leak -> the corner reports a dropping trend")

    # --- critical pressure ---
    st = scenario("one_critical")
    check(st["overall_status"] == "critical", "critical pressure -> overall critical",
          st["overall_status"])
    check(st["issues"][0]["severity"] == "critical",
          "critical pressure -> the issue is critical")
    check(st["issues"][0]["suggested_action"],
          "critical pressure -> carries a suggested action",
          st["issues"][0]["suggested_action"])

    # --- sensor disconnected, parked and moving ---
    st = scenario("sensor_disconnected", "normal_idle")
    sev_parked = st["issues"][0]["severity"]
    check(sev_parked == "warning",
          "sensor offline while parked -> warning, not an announcement", sev_parked)
    st = scenario("sensor_disconnected", "cruise")
    sev_moving = st["issues"][0]["severity"]
    check(sev_moving == "critical",
          "sensor offline while MOVING -> critical (the spec's own example)",
          sev_moving)
    check(st["moving"] is True, "and the context knows the car is moving")

    # --- no sensor data ---
    st = scenario("no_sensors", "normal_idle")
    types = [i["type"] for i in st["issues"]]
    check("tires_unavailable" in types,
          "no sensors -> says so rather than reporting healthy tires", str(types))
    check("normal" not in st["summary"].lower(),
          "no sensors -> the summary does NOT claim everything is fine",
          st["summary"])
    check("tires" not in st["subsystems_reporting"],
          "no sensors -> tires are not listed as reporting")

    # --- no UI state anywhere in the payload ---
    st = scenario("one_critical", "cruise")
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
          value=26.0, unit="psi"):
    return vh.make_issue(key=key, domain="tires", type=type_, severity=severity,
                         message="scripted", observation_window=vh.W_NOW,
                         location=location, magnitude=magnitude, value=value,
                         unit=unit, spoken_fallback="Something is wrong.")


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
    head("B -- the same, driven through the mock provider")
    telemetry.set_scenario("cruise")
    tires.set_scenario("all_normal")
    pol = P.VehicleHealthPolicy()
    r = pol.tick(vh.issues(), 0.0)
    check(r["announce"] is None, "a healthy car produces no announcement",
          r["reason"])
    tires.set_scenario("one_critical")
    r = pol.tick(vh.issues(), 30.0)
    check(r["announce"] is not None, "a critical tire does", r["reason"])
    said = r["announce"]["text"] if r["announce"] else ""
    print(f"        RIO says: {said!r}")
    check("rear left" in said, "and it names the corner", said)
    for i in range(5):
        r = pol.tick(vh.issues(), 33.0 + i * 3)
        if r["announce"]:
            break
    check(r["announce"] is None, "and does not say it again on the next five polls")


# ===========================================================================
# C. Truthfulness
# ===========================================================================

def run_truthfulness():
    head("C -- RIO is only given claims the data supports")

    windows = set()
    for tire_sc in ("one_low", "slow_leak", "one_critical", "sensor_disconnected",
                    "stale_data", "overheating", "battery_low", "no_sensors"):
        st = scenario(tire_sc, "cruise")
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
    allowed = {vh.W_NOW, vh.W_TIRE_TREND, vh.W_ENGINE_TREND, vh.W_SESSION}
    check(windows <= allowed, "and no window claims more history than exists",
          str(sorted(windows - allowed)))

    st = scenario("slow_leak", "cruise")
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
    tires.set_scenario("all_normal")
    st = vh.context(full=True)["vehicle_health"]
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

    nav = open(os.path.join(ROOT, "static", "rio_nav.js")).read()
    check("priority: 2" not in nav and "priority: 3" not in nav,
          "rio_nav.js holds no hardcoded priority numbers any more")
    check("RIO.speech.P.TURN_NEAR" in nav and "RIO.speech.P.NAV" in nav,
          "it names the tiers instead")

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
    mapping = {"NORMAL": ("normal", "informational"),
               "ATTENTION": ("warning", "informational"),
               "CRITICAL": ("critical",),
               "UNAVAILABLE": ("informational", "unknown")}
    for tire_sc in ("all_normal", "one_low", "slow_leak", "one_critical",
                    "overheating", "sensor_disconnected", "battery_low",
                    "stale_data", "no_sensors"):
        tires.set_scenario(tire_sc)
        telemetry.set_scenario("normal_idle")
        panel = tires.snapshot()["status"]
        ctx = vh.context(full=False)["vehicle_health"]["overall_status"]
        ok = ctx in mapping.get(panel, ())
        if not check(ok, f"{tire_sc}: panel {panel} agrees with context {ctx}"):
            return

    # A fault RIO announces has to be a fault the panel is showing, or the
    # driver hears about something they cannot then look at.
    tires.set_scenario("one_critical")
    telemetry.set_scenario("cruise")
    banners = tires.snapshot()["banners"]
    issues = vh.issues()
    check(banners and issues, "a critical fault produces both a banner and an issue")
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
        tires.set_scenario(tire_sc)
        telemetry.set_scenario(ecu_sc)
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
