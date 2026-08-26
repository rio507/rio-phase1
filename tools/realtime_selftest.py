"""Acceptance tests for RIO's live conversation and its one escalation.

    python -m tools.realtime_selftest
    python -m tools.realtime_selftest --live    # + the real models, once each

Without --live this runs entirely offline against stubs: the session payload,
the tool dispatch, every way the reasoning model can fail, and the firewall
that keeps both new models away from anything that speaks deterministically.
The behavioural half — arbitration, barge-in, the tool bridge — is
`node tools/realtime_selftest.js`, because that logic lives in the browser.

Six parts:

  A. SESSION — what the model is actually told: RIO's own personality, the
     voice, barge-in, and Whisper for transcripts so a drive's records all come
     from one transcriber.
  B. CONFIG — both model ids come from config and can be overridden from the
     environment, checked by starting a fresh interpreter rather than by
     reading the source and hoping.
  C. TOOL DISPATCH — an unknown name and unreadable arguments are answered, not
     raised. A model can emit either.
  D. FAILURE — timeout, outage, refusal, empty answer. All of them end as
     `ok: false`, which is a thing RIO is told how to handle.
  E. FIREWALL — nothing that speaks deterministically can reach either model.
     Same check headway and navigation already run on their own policy code.
  F. ENDPOINTS — the two handlers, including the switched-off path.
"""
import argparse
import inspect
import json
import os
import re
import subprocess
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

import config          # noqa: E402
import realtime        # noqa: E402

PASS, FAIL = [], []


def ok(cond, what):
    (PASS if cond else FAIL).append(what)
    print(("  ok    " if cond else "  FAIL  ") + what)


def section(name):
    print(f"\n=== {name} ===")


# ---------------------------------------------------------------------------
# A. The session
# ---------------------------------------------------------------------------
def run_session():
    section("A. the session — what RIO is actually told")
    s = realtime.session_config()

    ok(s["model"] == config.OPENAI_REALTIME_MODEL,
       f"the live model is the configured one ({s['model']})")
    ok(s["audio"]["output"]["voice"] == config.OPENAI_REALTIME_VOICE,
       f"the voice is config, not code ({s['audio']['output']['voice']})")
    ok(config.OPENAI_REALTIME_VOICE in ("cedar", "marin"),
       "and it is one of the two voices chosen for her")

    td = s["audio"]["input"]["turn_detection"]
    ok(td["type"] == "server_vad", "the server listens for turns itself")
    ok(td.get("interrupt_response") is True,
       "BARGE-IN: the driver talking stops her, without a button")
    ok(td.get("create_response") is True,
       "and a finished sentence gets an answer without one either")

    ok(s["audio"]["input"]["transcription"]["model"] == config.OPENAI_STT_MODEL,
       f"transcripts still come from Whisper ({config.OPENAI_STT_MODEL}) — the "
       "session log, /last_talk and the router all read what it produced")

    tools = [t["name"] for t in s["tools"]]
    ok(tools[:2] == [realtime.TOOL_NAME, realtime.LOOK_TOOL_NAME],
       f"the session carries her tools ({tools})")
    schema = s["tools"][0]
    ok("question" in schema["parameters"]["properties"],
       "the escalation takes a question")
    ok("route" in schema["description"] or "car" in schema["description"],
       "and is told what NOT to use it for — the car and the route are answered "
       "elsewhere")
    look = s["tools"][1]
    ok("camera" in look["description"].lower(),
       "the visual tool says plainly that it is the camera")
    ok("cannot see" in look["description"].lower(),
       "and that she cannot see anything without it")

    instr = s["instructions"]
    ok(config.SYSTEM_PROMPT.strip()[:200] in instr,
       "RIO's own personality prompt carries over verbatim")
    ok("interrupted" in instr.lower(),
       "with the live-only part added: she expects to be interrupted")
    ok("never announce a turn" in instr.lower(),
       "and is told never to announce a turn on her own initiative")
    ok("ok: false" in instr,
       "and what to do when the tool fails, which is: carry on")
    ok("you cannot see" in instr.lower(),
       "she is told outright that she has no eyes without the tool")
    ok(len(instr) > len(config.SYSTEM_PROMPT),
       "the addendum adds to the prompt rather than replacing it")


# ---------------------------------------------------------------------------
# B. Config
# ---------------------------------------------------------------------------
def run_config():
    section("B. config — both model ids in one place, overridable")
    ok(config.OPENAI_REALTIME_MODEL == "gpt-realtime-2.1",
       f"realtime model defaults to gpt-realtime-2.1 ({config.OPENAI_REALTIME_MODEL})")
    ok(config.OPENAI_REASONING_MODEL == "gpt-5.6-sol",
       f"reasoning model defaults to gpt-5.6-sol ({config.OPENAI_REASONING_MODEL})")
    ok(config.OPENAI_STT_MODEL == "whisper-1",
       "Whisper is still configured, and still what every transcript comes from")

    # Proved by starting a fresh interpreter with the environment set, rather
    # than by reading the source for os.getenv and believing it.
    env = dict(os.environ,
               OPENAI_REALTIME_MODEL="test-realtime-id",
               OPENAI_REASONING_MODEL="test-reasoning-id",
               OPENAI_REALTIME_VOICE="marin")
    out = subprocess.run(
        [sys.executable, "-c",
         "import config;print(config.OPENAI_REALTIME_MODEL, "
         "config.OPENAI_REASONING_MODEL, config.OPENAI_REALTIME_VOICE)"],
        env=env, cwd=REPO, capture_output=True, text=True, timeout=120)
    got = (out.stdout or "").strip()
    ok(got == "test-realtime-id test-reasoning-id marin",
       f"all three are overridable from the environment ({got or out.stderr[:80]})")

    src = inspect.getsource(realtime)
    ok("gpt-realtime" not in src and "gpt-5" not in src,
       "and no model id is hardcoded in realtime.py")


# ---------------------------------------------------------------------------
# C. Tool dispatch
# ---------------------------------------------------------------------------
def run_dispatch():
    section("C. dispatch — a model can emit anything, so nothing here raises")
    for args, why in (
        ("not json", "unreadable arguments"),
        (None, "no arguments at all"),
        ({}, "arguments with no question"),
        ({"question": "   "}, "a blank question"),
        (12345, "arguments of the wrong type"),
    ):
        r = realtime.run_tool(realtime.TOOL_NAME, args)
        ok(r.get("ok") is False, f"{why} -> ok:false, not an exception")

    r = realtime.run_tool("some_other_tool", {"question": "x"})
    ok(r.get("ok") is False and r.get("note") == "unknown tool",
       "a tool name that is not in its own list is answered, not raised")

    for args, why in (("not json", "unreadable arguments"),
                      ({}, "no question")):
        r = realtime.run_tool(realtime.LOOK_TOOL_NAME, args)
        ok(r.get("ok") is False, f"look with {why} -> ok:false, not an exception")


# ---------------------------------------------------------------------------
# D. Failure
# ---------------------------------------------------------------------------
class _FakeResponses:
    def __init__(self, behaviour):
        self.behaviour = behaviour
        self.calls = []

    def create(self, **kw):
        self.calls.append(kw)
        return self.behaviour(kw)


class _FakeClient:
    def __init__(self, behaviour):
        self.responses = _FakeResponses(behaviour)
        self.timeouts = []

    def with_options(self, timeout=None, **kw):
        self.timeouts.append(timeout)
        return self


class _Result:
    def __init__(self, text):
        self.output_text = text


def _with_client(behaviour):
    fake = _FakeClient(behaviour)
    realtime._client = fake
    return fake


def run_failure():
    section("D. failure — every way it can go wrong ends the same way")
    original = realtime._client
    try:
        fake = _with_client(lambda kw: _Result("Because the map says so."))
        r = realtime.escalate("why", context="driver asked about the route")
        ok(r["ok"] and r["answer"] == "Because the map says so.",
           "a good answer comes back as text for RIO to speak")
        ok(r["model"] == config.OPENAI_REASONING_MODEL,
           "recorded against the model that produced it")
        sent = fake.responses.calls[-1]
        ok(sent["model"] == config.OPENAI_REASONING_MODEL,
           "the reasoning model is the one asked")
        ok("driver asked about the route" in sent["input"],
           "and the conversation context goes with the question")
        ok("READ ALOUD" in sent["instructions"] or "read aloud" in sent["instructions"].lower(),
           "asked for prose to be spoken — no markdown, no citations read out")
        ok(fake.timeouts and fake.timeouts[-1] == config.REALTIME_TOOL_TIMEOUT_S,
           f"under the configured timeout ({config.REALTIME_TOOL_TIMEOUT_S}s), "
           "which aborts the request rather than abandoning it")
        ok(any(t.get("type") == "web_search" for t in (sent.get("tools") or [])),
           "with web search available for anything that changes")

        def boom(kw):
            raise TimeoutError("Request timed out.")
        _with_client(boom)
        r = realtime.escalate("anything")
        ok(r["ok"] is False and "Timeout" in r["note"],
           "a timeout -> ok:false, and RIO carries on")

        def refuse(kw):
            raise RuntimeError("model_not_found")
        _with_client(refuse)
        ok(realtime.escalate("anything")["ok"] is False,
           "an unavailable model -> ok:false, and RIO carries on")

        _with_client(lambda kw: _Result("   "))
        ok(realtime.escalate("anything")["ok"] is False,
           "an empty answer -> ok:false rather than a moment of silence")

        _with_client(lambda kw: _Result("fine"))
        old = config.REALTIME_WEB_SEARCH
        try:
            config.REALTIME_WEB_SEARCH = False
            realtime.escalate("anything")
            ok(not (realtime._client.responses.calls[-1].get("tools") or []),
               "web search is a config switch, and off means off")
        finally:
            config.REALTIME_WEB_SEARCH = old
    finally:
        realtime._client = original


# ---------------------------------------------------------------------------
# E. Firewall
# ---------------------------------------------------------------------------
def _imports_of(path):
    """Top-level module names a file imports, from its AST.

    Reading the imports rather than grepping the text: vehicle_health_policy.py
    contains the string "openai" in the docstring that PROMISES it never calls
    one, and a substring check cannot tell a promise from a breach. This is the
    same method tools/vehicle_health_selftest.py already uses on that file.
    """
    import ast

    tree = ast.parse(open(path).read())
    names = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            names.update(a.name.split(".")[0] for a in node.names)
        elif isinstance(node, ast.ImportFrom) and node.module:
            names.add(node.module.split(".")[0])
    return names


def run_firewall():
    section("E. firewall — no model can reach a deterministic voice")

    # Nothing that speaks from a fixed table may import a model — neither the
    # new ones nor the old chat path.
    forbidden = {"realtime", "openai", "llm_interface", "visual_qa"}
    for rel in ("navigation/speech.py", "navigation/anchors.py",
                "vehicle_health_policy.py", "headway/live_policy.py"):
        names = _imports_of(os.path.join(REPO, rel))
        hits = sorted(names & forbidden)
        ok(not hits, f"{rel} imports no model path" + (f" (found {hits})" if hits else ""))

    # ...and the other direction, which is the new one: the live session must
    # have no way to REACH a warning or an instruction. It cannot generate one
    # if it cannot import the thing that makes them.
    live_imports = _imports_of(os.path.join(REPO, "realtime.py"))
    # The line is between READING state and DECIDING to announce.
    #
    # vehicle_health.py builds the context an ordinary conversation turn
    # already gets (llm_interface._health_block calls the same function), so
    # reading it from a live session is the same read by a different mouth.
    # vehicle_health_policy.py is the thing that decides whether a driver gets
    # interrupted, and THAT must stay unreachable — along with the navigation
    # speech planner, the headway policy and the TTS path.
    unreachable = {"navigation", "headway", "vehicle_health_policy",
                   "tires", "telemetry", "voice"}
    hits = sorted(live_imports & unreachable)
    ok(not hits,
       "realtime.py cannot reach anything that DECIDES to speak — the headway "
       "policy, the health announcement policy, navigation or the TTS path"
       + (f" (found {hits})" if hits else ""))
    ok("vehicle_health_policy" not in live_imports,
       "in particular it cannot reach the policy that interrupts a driver")
    # visual_qa and router ARE allowed, and the distinction is the whole point
    # of the firewall rather than an exception to it: they are conversation.
    # The visual path is already model-driven, already speaks at the same
    # conversation tier, and already yields to every warning. What must stay
    # unreachable is the DETERMINISTIC half — the code that decides a gap is
    # closing or that a turn is four seconds away.
    # base64/io/wave arrived with clip rendering: turning PCM deltas into a WAV
    # so the pre-rendered warnings can be produced in the same voice.
    ok(live_imports <= {"json", "os", "threading", "time", "typing", "openai",
                        "config", "visual_qa", "router", "vehicle_health",
                        "base64", "io", "wave"},
       f"and imports only read-side code and stdlib ({sorted(live_imports)})")

    # The instructions are the other half of it: the model is told in words,
    # because a model that CAN say a sentence has to be told not to.
    addendum = realtime.LIVE_ADDENDUM.lower()
    ok("never announce a turn" in addendum,
       "RIO is told never to announce a turn herself — the navigation system "
       "calls it, and two voices calling one turn is the failure this prevents")
    ok("never invent anything about the car, the route, or the road" in addendum,
       "and never to invent anything about the car, the route or the road")
    ok("interrupt" in addendum,
       "and that other voices cutting her off is correct, not a fault to "
       "apologise for")


# ---------------------------------------------------------------------------
# F. Endpoints
# ---------------------------------------------------------------------------
def run_endpoints():
    section("F. endpoints — minting a credential, and running one tool")
    import app as app_mod

    old = config.REALTIME_ENABLED
    try:
        config.REALTIME_ENABLED = False
        r = app_mod.realtime_session_endpoint(session_id=None)
        ok(r.get("enabled") is False and "error" in r,
           "switched off -> the panel is told so and falls back to hold-to-talk")
    finally:
        config.REALTIME_ENABLED = old

    original = realtime._client
    try:
        _with_client(lambda kw: _Result("A lean condition."))
        r = app_mod.realtime_tool_endpoint(
            {"name": realtime.TOOL_NAME, "arguments": {"question": "P0171?"}},
            session_id=None)
        ok(r.get("ok") and "lean" in r.get("answer", "").lower(),
           "the tool endpoint runs the escalation and returns text")

        r = app_mod.realtime_tool_endpoint({"name": "nope", "arguments": {}},
                                           session_id=None)
        ok(r.get("ok") is False, "and refuses a tool it does not have, calmly")
    finally:
        realtime._client = original

    st = realtime.status()
    ok(st["model"] == config.OPENAI_REALTIME_MODEL and
       st["reasoning_model"] == config.OPENAI_REASONING_MODEL,
       "status() reports both model ids without calling anything")


# ---------------------------------------------------------------------------
# G. Looking
# ---------------------------------------------------------------------------
class _FakeSession:
    def __init__(self, referent=None, clarification=None):
        self._ref, self._clar = referent, clarification

    def active_referent(self):
        return self._ref

    def pending_clarification(self):
        return self._clar


class _FakeAnswer:
    def __init__(self, text, meta=None):
        self._text = text
        self.meta = meta or {"question": "x"}

    def text(self):
        if isinstance(self._text, Exception):
            raise self._text
        return self._text


class _FakeVisualQA:
    """Stands in for the camera pipeline: it records what it was asked."""

    def __init__(self, text="A white Lexus, two cars ahead.", meta=None,
                 session=None):
        self.text = text
        self.meta = meta
        self.session = session or _FakeSession()
        self.calls = []

    def get_session(self, key):
        self.calls.append(("get_session", key))
        return self.session

    def answer(self, session_key, question, route=None):
        self.calls.append(("answer", session_key, question, route))
        return _FakeAnswer(self.text, self.meta)


def _with_visual(fake):
    sys.modules["visual_qa"] = fake
    return fake


def run_look():
    section("G. looking — the camera pipeline, reached from a live session")
    real = sys.modules.get("visual_qa")
    try:
        fake = _with_visual(_FakeVisualQA(meta={"route": {"request_type": "scene_description"}}))
        r = realtime.run_tool(realtime.LOOK_TOOL_NAME,
                              {"question": "what's that car ahead"},
                              session_key="drive_1")
        ok(r.get("ok") and "Lexus" in r["answer"],
           "a visual question comes back as text for RIO to speak")
        answered = [c for c in fake.calls if c[0] == "answer"][0]
        ok(answered[1] == "drive_1",
           "asked against the SESSION's frame ring, not a global one — a live "
           "question and a recorded one look at the same few seconds of road")
        ok(answered[2] == "what's that car ahead",
           "with the driver's own words, so 'the black one' still resolves")
        ok(r.get("meta") is not None,
           "and the turn's decision chain rides back for the drive log")

        # The route override. The model called `look`, so the question IS
        # visual; a classifier that disagrees must not produce an answer with
        # no picture behind it.
        fake = _with_visual(_FakeVisualQA())
        realtime.run_tool(realtime.LOOK_TOOL_NAME,
                          {"question": "how are you doing today"})
        route = [c for c in fake.calls if c[0] == "answer"][0][3]
        ok(route["request_type"] == "scene_description",
           f"a question the router reads as non-visual is forced to a look "
           f"({route['request_type']})")
        ok(route.get("method") == "forced_by_look_tool",
           "and the override is recorded rather than disguised as a routing decision")

        fake = _with_visual(_FakeVisualQA())
        realtime.run_tool(realtime.LOOK_TOOL_NAME, {"question": "what is that"})
        route = [c for c in fake.calls if c[0] == "answer"][0][3]
        ok(route.get("method") != "forced_by_look_tool",
           "...but a question that already reads as visual is left alone")

        # Failure. All of it ends as ok:false, which RIO knows how to handle.
        _with_visual(_FakeVisualQA(text=""))
        r = realtime.run_tool(realtime.LOOK_TOOL_NAME, {"question": "what do you see"})
        ok(r.get("ok") is False and r.get("note") == "nothing to see",
           "no frames in the ring -> she says she cannot see, rather than "
           "describing a road she has not been shown")

        _with_visual(_FakeVisualQA(text=RuntimeError("no detector")))
        r = realtime.run_tool(realtime.LOOK_TOOL_NAME, {"question": "what do you see"})
        ok(r.get("ok") is False and "RuntimeError" in r.get("note", ""),
           "the visual stack falling over -> ok:false, and the turn survives")

        broken = _FakeVisualQA()
        broken.get_session = lambda key: (_ for _ in ()).throw(KeyError("no session"))
        _with_visual(broken)
        r = realtime.run_tool(realtime.LOOK_TOOL_NAME, {"question": "what do you see"})
        ok(r.get("ok") is False, "no session at all -> ok:false")
    finally:
        if real is not None:
            sys.modules["visual_qa"] = real
        else:
            sys.modules.pop("visual_qa", None)


# ---------------------------------------------------------------------------
# H. Dictation
# ---------------------------------------------------------------------------
def run_dictation():
    section("H. dictation — deterministic lines in RIO's own voice")
    r = realtime.verbatim_response("Back off — now.")
    ok(r["conversation"] == "none",
       "a dictated line is OUT OF BAND — it never enters the conversation, "
       "because a warning is a fact about the car and not something RIO said")
    ok(r["output_modalities"] == ["audio"], "audio only")
    ok(r["instructions"].endswith("Back off — now."),
       "with the policy's exact words at the end of the instruction")
    for phrase in ("word for word", "Add nothing", "Do not rephrase",
                   "not addressed to you"):
        ok(phrase in r["instructions"],
           f"and an instruction that says {phrase!r}")

    ok(config.VOICE_BACKEND == "realtime",
       f"RIO's active voice is the live session ({config.VOICE_BACKEND})")
    ok(config.VOICE_FALLBACK_BACKEND == "elevenlabs",
       "with ElevenLabs kept as the fallback, complete and off the active path")

    import voice
    gen = voice.synthesize_stream("anything", backend="realtime")
    try:
        next(gen)
        ok(False, "this process must not pretend it can produce the live voice")
    except voice.VoiceUnavailable:
        ok(True, "and this process refuses to fake the live voice rather than "
                 "silently substituting the other one")
    except Exception as e:
        ok(False, f"unexpected error: {type(e).__name__}")

    ok(config.REALTIME_SPEAK_TIMEOUT_MS <= 1000,
       f"the dictation budget is short ({config.REALTIME_SPEAK_TIMEOUT_MS} ms) — "
       "a warning that arrives late has stopped being a warning")
    ok(set(config.REALTIME_SPEECH_CHANNELS) == {"nav", "health", "headway"},
       "every deterministic channel has a switch")

    # The browser must not hold its own copy of the verbatim instruction.
    nav_js = open(os.path.join(REPO, "static", "rio_realtime.js")).read()
    ok("session.verbatim_instruction" in nav_js,
       "the panel takes the verbatim instruction from the session payload, so "
       "there is one copy of it and it is this one")

    # ...and the clip renderer must render in the same voice by default.
    renderer = open(os.path.join(REPO, "tools", "render_alerts.py")).read()
    ok("config.VOICE_BACKEND" in renderer,
       "the pre-rendered clips are rendered in whatever RIO's active voice is")
    ok("CLIP_RENDER_ATTEMPTS" in renderer and "_transcribe" in renderer,
       "and are transcribed after transcoding, because a clip is written once "
       "and played for months")


# ---------------------------------------------------------------------------
# I. Awareness
# ---------------------------------------------------------------------------
PENDING_CONTEXT = {
    "vehicle_health": {
        "overall_status": "attention",
        "data_available": True,
        "subsystems_reporting": ["engine", "tires"],
        "moving": True,
        "history_depth": "Live data from the current drive only.",
        "issue_count": 2,
        "issues": [
            {"type": "engine.new_dtc", "domain": "diagnostics",
             "reported_by": "the vehicle's own computer",
             "severity": "attention", "where": "engine",
             "code": "P0171", "lifecycle": "pending_first_seen",
             "confirmed": False,
             "message": "The car has picked up a pending engine code and has "
                        "not confirmed that it is persistent.",
             "observation_window": "seen once, in this drive"},
            {"type": "tire.pressure_low", "domain": "tires",
             "reported_by": "RIO", "severity": "informational",
             "where": "rear left", "confirmed": True,
             "message": "The rear left is a little below the others.",
             "observation_window": "the last 40 minutes"},
        ],
        "summary": "One pending engine code, not yet confirmed, and a soft rear left.",
    }
}


def run_awareness():
    section("I. awareness — what she can find out when asked")
    s = realtime.session_config()
    names = [t["name"] for t in s["tools"]]
    ok(names == [realtime.TOOL_NAME, realtime.LOOK_TOOL_NAME,
                 realtime.NAV_TOOL_NAME, realtime.VEHICLE_TOOL_NAME],
       f"four tools: think, look, route, car ({names})")

    nav_desc = s["tools"][2]["description"]
    veh_desc = s["tools"][3]["description"]
    ok("ETA" in nav_desc and "next turn" in nav_desc,
       "the route tool says what it answers")
    ok("not yet confirmed" in veh_desc or "not confirmed" in veh_desc,
       "and the car tool names pending codes as part of its job")
    for desc, what in ((nav_desc, "route"), (veh_desc, "car")):
        ok("not" in desc.lower() and ("announce" in desc.lower()
                                      or "itself" in desc.lower()),
           f"the {what} tool's own description says it is for answering, not announcing")

    instr = s["instructions"]
    ok("YOU ANSWER. YOU DO NOT ANNOUNCE." in instr,
       "the boundary is stated in the instructions, in those words")
    for phrase in ("never announce a turn", "not a cue", "Unasked, you say nothing"):
        ok(phrase.lower() in instr.lower(), f"and spelled out: {phrase!r}")
    for phrase in ("Say only what the data supports", "Keep the provenance",
                   "detected but NOT CONFIRMED"):
        ok(phrase in instr, f"truthfulness rule carried over: {phrase!r}")

    # vehicle_status must PASS THROUGH the existing builder, not summarise it.
    import vehicle_health
    real_context = vehicle_health.context
    try:
        vehicle_health.context = lambda full=True: PENDING_CONTEXT
        r = realtime.vehicle_status()
        ok(r["ok"], "vehicle_status answers from the conversation-layer context")
        issues = r["vehicle"]["issues"]
        ok(len(issues) == 2, "every issue survives — nothing is summarised away")
        pending = issues[0]
        ok(pending["lifecycle"] == "pending_first_seen" and pending["confirmed"] is False,
           "a pending code arrives as PENDING, with its lifecycle intact")
        ok("not confirmed" in pending["message"],
           "and its own words say it has not been confirmed")
        ok(pending["reported_by"] == "the vehicle's own computer"
           and issues[1]["reported_by"] == "RIO",
           "provenance is preserved per issue — the ECU and RIO are different "
           "claims and stay that way")
        ok(all("observation_window" in i for i in issues),
           "as is how far back each observation actually goes")
        ok("not confirmed" in (r.get("rules") or "").lower(),
           "and the result restates the rules to the model")
    finally:
        vehicle_health.context = real_context

    # ...and it does not build a parallel source.
    src = inspect.getsource(realtime.vehicle_status)
    ok("vehicle_health.context" in src,
       "it calls the existing builder rather than assembling its own view")
    ok("issues" not in src.replace('body.get("issues")', ""),
       "and does not reshape what comes back")

    # nav_status is deliberately NOT answered here.
    r = realtime.run_tool(realtime.NAV_TOOL_NAME, {})
    ok(r["ok"] is False and "panel" in r["note"],
       "nav_status is refused server-side: progress lives in the browser, and "
       "a second source would be a second answer")


# ---------------------------------------------------------------------------
# Live
# ---------------------------------------------------------------------------
SPOKEN_NUMBERS = {
    "zero": "0", "one": "1", "two": "2", "three": "3", "four": "4", "five": "5",
    "six": "6", "seven": "7", "eight": "8", "nine": "9", "ten": "10",
    "eleven": "11", "twelve": "12", "twenty": "20", "thirty": "30",
    "twentysix": "26", "twenty six": "26",
}


def spoken_norm(text: str) -> str:
    """Compare what was SAID, not how a transcriber chose to spell it.

    Whisper writes "twenty-six P S I" as "26 psi". That is a rendering
    difference, not a paraphrase, and a verbatim check that fails on it would
    be a check nobody keeps. Numbers are folded to digits and single letters
    are joined, and NOTHING else is forgiven: a changed, added or dropped word
    still fails.
    """
    t = re.sub(r"[^a-z0-9 ]+", " ", (text or "").lower())
    t = re.sub(r"\s+", " ", t).strip()
    for word, digit in sorted(SPOKEN_NUMBERS.items(), key=lambda kv: -len(kv[0])):
        t = re.sub(r"\b" + word.replace(" ", r"\s+") + r"\b", digit, t)
    # "p s i" -> "psi": a spelled-out acronym is the same sound either way.
    t = re.sub(r"\b(?:([a-z]) )+([a-z])\b",
               lambda m: m.group(0).replace(" ", ""), t)
    return " ".join(t.split())


def run_live():
    section("LIVE — both models, once each, against the real account")
    from openai import OpenAI

    c = OpenAI()
    for mid in (config.OPENAI_REALTIME_MODEL, config.OPENAI_REASONING_MODEL):
        try:
            c.models.retrieve(mid)
            ok(True, f"{mid} is available on this account")
        except Exception as e:
            # Requirement: flag it, never substitute.
            ok(False, f"{mid} is NOT available: {type(e).__name__} — "
                      "this must be flagged, not swapped for another model")

    t0 = time.time()
    try:
        s = realtime.mint_client_secret()
        ok(bool(s.get("client_secret")) and s["model"] == config.OPENAI_REALTIME_MODEL,
           f"a live session mints in {time.time()-t0:.1f}s "
           f"(voice {s['voice']}, tool {s['tool']})")
    except Exception as e:
        ok(False, f"session mint failed: {type(e).__name__}: {str(e)[:160]}")

    t0 = time.time()
    r = realtime.run_tool(realtime.TOOL_NAME,
                          {"question": "What does an OBD-II code P0171 mean?",
                           "context": "the driver is in a 2015 Camaro"})
    ok(r.get("ok"), f"deep_dive answers in {time.time()-t0:.1f}s")
    if r.get("ok"):
        answer = r["answer"]
        print(f"    \"{answer[:150]}...\"")
        ok("#" not in answer and "](" not in answer and "*" not in answer[:200],
           "in prose meant to be spoken — no markdown reaching a voice")


def run_verbatim():
    section("VERBATIM — what it was asked to say, and what it actually said")
    import io

    from openai import OpenAI

    from headway import live_policy

    client = OpenAI()
    lines = [
        live_policy.LINE_TEXT[live_policy.LINE_TOO_CLOSE],
        "Turn left by the Shell station.",
        "The rear left tire is down to twenty-six P S I.",
        "Right here.",
    ]
    for line in lines:
        got = realtime.render_speech(line)
        if not got.get("ok"):
            ok(False, f"could not render {line!r}: {got.get('note')}")
            continue
        # Two different claims, checked separately:
        #   what the model SAYS it said — catches a paraphrase outright;
        #   what a different model HEARS in the audio — catches the first
        #   claim being wrong, which is the only reason to do both.
        ok(spoken_norm(got["transcript"]) == spoken_norm(line),
           f"its own transcript matches: {line!r}")
        buf = io.BytesIO(got["wav"])
        buf.name = "line.wav"
        heard = client.audio.transcriptions.create(
            model=config.OPENAI_STT_MODEL, file=buf).text
        same = spoken_norm(heard) == spoken_norm(line)
        ok(same, f"and Whisper hears the same words back"
                 + ("" if same else f" — heard {heard.strip()!r}"))


def run_latency():
    section("LATENCY — text to first audio, the old way and the new")
    import statistics

    import voice

    line = "You're too close."
    runs = 3

    el = []
    for _ in range(runs):
        t0 = time.time()
        stream = voice.synthesize_stream(line, backend="elevenlabs")
        next(iter(stream))
        el.append((time.time() - t0) * 1000)

    warm = []
    t_conn = time.time()
    with realtime.client().realtime.connect(
            model=config.OPENAI_REALTIME_MODEL) as conn:
        conn.session.update(session={
            "type": "realtime", "output_modalities": ["audio"],
            "audio": {"output": {"voice": config.OPENAI_REALTIME_VOICE}}})
        connect_ms = (time.time() - t_conn) * 1000
        for _ in range(runs):
            t0 = time.time()
            conn.response.create(response=realtime.verbatim_response(line))
            for event in conn:
                if event.type == "response.output_audio.delta":
                    warm.append((time.time() - t0) * 1000)
                    break
                if event.type == "error":
                    break
            for event in conn:
                if event.type == "response.done":
                    break

    el_med = statistics.median(el)
    rt_med = statistics.median(warm) if warm else float("inf")
    print(f"    elevenlabs (fallback) : {el_med:6.0f} ms   {[round(x) for x in el]}")
    print(f"    dictated (warm)       : {rt_med:6.0f} ms   {[round(x) for x in warm]}")
    print(f"    opening a session     : {connect_ms:6.0f} ms  (only when none is open)")
    print(f"    pre-rendered clip     :      0 ms  (local file, no network)")

    ok(bool(warm), "the live voice speaks a dictated warning")
    ok(rt_med < config.REALTIME_SPEAK_TIMEOUT_MS,
       f"and starts inside the dictation budget ({rt_med:.0f} ms of "
       f"{config.REALTIME_SPEAK_TIMEOUT_MS} ms), so the fallback does not fire "
       "on a healthy session")
    # Not a pass/fail: the number itself is the deliverable. What must hold is
    # that the fallback is quicker, which is why it is the fallback.
    print(f"    -> dictation costs {rt_med - el_med:+.0f} ms against the "
          f"synthesiser on a warm session")
    ok(el_med < rt_med,
       f"the fallback is the faster path ({el_med:.0f} ms), which is why the "
       "most time-critical lines are pre-rendered rather than either")


# ---------------------------------------------------------------------------
# The full chain, against the running server
# ---------------------------------------------------------------------------
def _ask_live(conn, question, tool_handler, timeout_s=120.0):
    """Ask a real session a question, answer whatever it reaches for, and
    return (everything it said, which tools it called).

    This plays the part of the browser: the panel answers nav_status locally
    and forwards the rest to the server, so `tool_handler` does both.

    The one subtlety is turn-taking. A tool call arrives INSIDE a response that
    is still active, so the result is queued as an item immediately but the
    follow-up response is only asked for once that response is done —
    otherwise the API refuses it, which is what a first cut of this function
    did to itself.
    """
    conn.conversation.item.create(item={
        "type": "message", "role": "user",
        "content": [{"type": "input_text", "text": question}]})
    conn.response.create()

    said, called, need_followup = [], [], False
    t0 = time.time()
    for event in conn:
        if time.time() - t0 > timeout_s:
            break
        if event.type == "response.function_call_arguments.done":
            called.append(event.name)
            result = tool_handler(event.name, event.arguments)
            conn.conversation.item.create(item={
                "type": "function_call_output",
                "call_id": event.call_id,
                "output": json.dumps(result)})
            need_followup = True
        elif event.type == "response.output_audio_transcript.done":
            if event.transcript:
                said.append(event.transcript.strip())
        elif event.type == "response.done":
            if need_followup:
                need_followup = False
                conn.response.create()      # safe now: nothing is active
                continue
            break
        elif event.type == "error":
            said.append(f"[error: {getattr(event, 'error', '')}]")
            break
    return " ".join(said).strip(), called


def run_chain(base: str = "http://127.0.0.1:8888"):
    section("CHAIN — a real session, against the running server, over HTTP")
    import httpx

    # The panel's half of the tool bridge, and the scripted state it would be
    # reading. nav_status is answered here because that is where the tracker
    # lives; everything else goes to the server exactly as the browser sends it.
    nav_state = {
        "ok": True, "routing": True, "destination": "Griffith Observatory",
        "distance_remaining_m": 4200, "minutes_remaining": 9,
        "next_maneuver": {"instruction": "Turn left onto Vermont Avenue",
                          "direction": "LEFT", "road_name": "Vermont Avenue",
                          "distance_m": 60, "seconds_away": 4,
                          "state": "IMMINENT"},
        "maneuvers_left": 5, "route_state": "ON_ROUTE", "gps_state": "GPS_OK",
        "arrived": False,
        "rules": "Answer the question. Do NOT announce this maneuver — the "
                 "navigation system calls it out loud itself.",
    }
    pending_vehicle = {"ok": True, "vehicle": PENDING_CONTEXT["vehicle_health"],
                       "rules": ("Only claims this data supports. Keep who "
                                 "reported what. A code detected but not "
                                 "confirmed stays not confirmed.")}
    use_pending = {"on": False}
    calls = []

    def tool_handler(name, arguments):
        calls.append(name)
        if name == realtime.NAV_TOOL_NAME:
            return nav_state
        if name == realtime.VEHICLE_TOOL_NAME and use_pending["on"]:
            return pending_vehicle
        r = httpx.post(f"{base}/realtime/tool", timeout=90,
                       json={"name": name, "arguments":
                             json.loads(arguments or "{}")})
        return r.json()

    # 5. FRAME SUPPLY, over HTTP, exactly as the browser posts it.
    with open("/tmp/road.jpg", "rb") as fh:
        frame = fh.read()
    for i in range(3):
        httpx.post(f"{base}/headway_frame", timeout=60,
                   files={"image": ("frame.jpg", frame, "image/jpeg")},
                   data={"v_host": "14.0", "v_host_age_s": "0.2",
                         "frame_t": str(i)})
    scene = httpx.get(f"{base}/scene", timeout=30).json()
    ok(len(scene.get("objects") or []) > 0,
       f"the frame ring fills from posted frames — {len(scene.get('objects') or [])} "
       "objects in the scene graph")

    with realtime.client().realtime.connect(
            model=config.OPENAI_REALTIME_MODEL) as conn:
        conn.session.update(session=dict(realtime.session_config(),
                                         output_modalities=["audio"]))

        # 1. A visual question must reach the camera.
        said, used = _ask_live(conn, "What's ahead of us right now?", tool_handler)
        print(f"    Q: what's ahead of us right now?\n    A: {said[:220]}")
        ok(realtime.LOOK_TOOL_NAME in used, f"she looks ({used})")
        ok(any(w in said.lower() for w in
               ("car", "sedan", "suv", "street", "road", "traffic", "ahead",
                "lane", "building")),
           "and answers from what the camera actually returned")

        # 2. A route question must reach the tracker.
        calls.clear()
        said, used = _ask_live(conn, "How far is it and where are we headed?",
                               tool_handler)
        print(f"    Q: how far is it and where are we headed?\n    A: {said[:220]}")
        ok(realtime.NAV_TOOL_NAME in used, f"she checks the route ({used})")
        ok("griffith" in said.lower(),
           "and names the real destination rather than a plausible one")

        # 4. ANTI-DOUBLE-SPEAK: the same state, with a turn four seconds away.
        ok(not re.search(r"\b(turn left|turn right|take the next)\b", said.lower()),
           "and does NOT call the turn, which the navigation system is about "
           "to call itself")

        # 3. A vehicle question with a pending code must keep the provenance.
        use_pending["on"] = True
        said, used = _ask_live(conn, "Is anything wrong with the car?", tool_handler)
        print(f"    Q: is anything wrong with the car?\n    A: {said[:260]}")
        ok(realtime.VEHICLE_TOOL_NAME in used, f"she checks the car ({used})")
        low = said.lower()
        ok(any(p in low for p in ("not confirm", "hasn't confirmed", "has not confirmed",
                                  "pending", "not yet confirmed")),
           "and preserves detected-but-not-confirmed rather than upgrading it "
           "to a fault")
        ok(not re.search(r"\b\d+\s*(psi|pounds)\b", low),
           "and invents no numbers the data did not contain")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--live", action="store_true",
                    help="also call both real models once")
    ap.add_argument("--chain", action="store_true",
                    help="drive a real session against the running server")
    args = ap.parse_args()

    run_session()
    run_config()
    run_dispatch()
    run_failure()
    run_firewall()
    run_endpoints()
    run_look()
    run_dictation()
    run_awareness()
    if args.live:
        run_live()
        run_verbatim()
        run_latency()
    if args.chain:
        run_chain()

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
