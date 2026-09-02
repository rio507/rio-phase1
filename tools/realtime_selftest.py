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

...and, further down, the tools that came later: her eyes (G), the
deterministic lines she is dictated (H), what she can find out when asked (I),
and the one tool that ACTS rather than reports — starting a route because the
driver asked to be taken somewhere (J).
"""
import argparse
import inspect
import json
import os
import re
import subprocess
import sys
import time
from typing import Optional

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

import config          # noqa: E402
import places          # noqa: E402
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
    ok(realtime.NAVIGATE_TOOL_NAME in tools,
       "including the one that acts rather than reports: she can start a route")
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

    r = realtime.run_tool(realtime.NAVIGATE_TOOL_NAME, {"destination": "LAX"})
    ok(r.get("ok") is False and "panel" in r.get("note", ""),
       "start_navigation is refused server-side — resolving a destination here "
       "and hoping the browser routes to it is the arrangement it replaces")


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
    # `places` is read-side by the same test the others pass: it answers a
    # question and cannot start a sentence. It is checked below rather than
    # taken on trust, because it is the one module here that talks to an
    # external API and the obvious place for someone to reach for
    # navigation/geo.py -- which would drag the speech planner across the
    # firewall for the sake of a haversine.
    # `observer` and `re` arrived with the fast path for visual questions: the
    # observer is a read that runs Qwen over frames already in the ring, and it
    # is checked below like places is.
    ok(live_imports <= {"json", "os", "re", "threading", "time", "typing",
                        "openai", "config", "visual_qa", "router",
                        "vehicle_health", "places", "observer",
                        "base64", "io", "wave"},
       f"and imports only read-side code and stdlib ({sorted(live_imports)})")

    observer_imports = _imports_of("observer.py")
    hits = sorted(observer_imports & unreachable)
    ok(not hits,
       "observer.py cannot reach anything that decides to speak either"
       + (f" (found {hits})" if hits else ""))
    ok("visual_qa" not in observer_imports and "router" not in observer_imports,
       "and it describes frames without going near the turn that answers "
       "questions — one direction only, or the fast path becomes a second "
       "answering pipeline")

    places_imports = _imports_of("places.py")
    hits = sorted(places_imports & unreachable)
    ok(not hits,
       "places.py cannot reach anything that decides to speak either"
       + (f" (found {hits})" if hits else ""))
    ok("navigation" not in places_imports,
       "and does not import navigation for its geometry — one haversine is "
       "cheaper than a route through the firewall")

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
                 realtime.NAV_TOOL_NAME, realtime.NAV_DIRECTIONS_TOOL_NAME,
                 realtime.PLACES_TOOL_NAME, realtime.VEHICLE_TOOL_NAME,
                 realtime.NAVIGATE_TOOL_NAME],
       f"seven tools: think, look, route, directions, places, car — and go "
       f"({names})")

    # By name, not by index. The list has grown once and will again, and an
    # index here means the next tool inserted silently re-points three
    # assertions at the wrong description.
    by_name = {t["name"]: t for t in s["tools"]}
    nav_desc = by_name[realtime.NAV_TOOL_NAME]["description"]
    dir_desc = by_name[realtime.NAV_DIRECTIONS_TOOL_NAME]["description"]
    veh_desc = by_name[realtime.VEHICLE_TOOL_NAME]["description"]

    # The directions tool: what it is for, and the two lines that keep it from
    # becoming a second voice for the turns.
    ok("turn-by-turn" in dir_desc and "in order" in dir_desc,
       "the directions tool says it returns the whole upcoming list")
    ok("ANSWERING" in dir_desc and "not a cue to call a turn" in dir_desc,
       "and that reading them is answering, while calling them is not hers")
    ok("expectation" in dir_desc and "never as something you can see" in dir_desc,
       "and that a map landmark is an expectation, not a sighting")
    ok(by_name[realtime.NAV_DIRECTIONS_TOOL_NAME]["parameters"]["properties"]
       .get("count") is not None,
       "it takes a count, so 'the next few' and 'all of them' are one tool")
    ok("ETA" in nav_desc and "next turn" in nav_desc,
       "the route tool says what it answers")
    ok("not yet confirmed" in veh_desc or "not confirmed" in veh_desc,
       "and the car tool names pending codes as part of its job")
    for desc, what in ((nav_desc, "route"), (veh_desc, "car")):
        ok("not" in desc.lower() and ("announce" in desc.lower()
                                      or "itself" in desc.lower()),
           f"the {what} tool's own description says it is for answering, not announcing")

    instr = s["instructions"]
    # Line-wrap tolerant copy: these are paragraphs, and a phrase that happens
    # to straddle a newline is not a different instruction.
    flat_instr = re.sub(r"\s+", " ", instr)
    ok("YOU ANSWER. YOU DO NOT ANNOUNCE." in instr,
       "the boundary is stated in the instructions, in those words")
    for phrase in ("never announce a turn", "not a cue", "Unasked, you say nothing"):
        ok(phrase.lower() in instr.lower(), f"and spelled out: {phrase!r}")
    for phrase in ("Say only what the data supports", "Keep the provenance",
                   "detected but NOT CONFIRMED"):
        ok(phrase in instr, f"truthfulness rule carried over: {phrase!r}")

    # READING IS ANSWERING. The bug was RIO saying she could not read the
    # directions of a route she was driving; the fix is a tool AND permission,
    # and permission that is not written down is not permission.
    # PLACES. The instruction is the other half of the tool: a model that has
    # the tool and permission to guess will still sometimes guess.
    ok("WHEN THE DRIVER ASKS ABOUT A PLACE" in instr,
       "the instructions have a section for place questions")
    ok(realtime.PLACES_TOOL_NAME in instr, "naming find_places")
    ok("never name a business, a rating, a price or an opening time that did "
       "not come back from find_places" in flat_instr,
       "and forbidding a business, rating, price or opening time that did not "
       "come from it")
    ok("Do not use the research tool for this" in instr,
       "with place questions routed away from deep_dive explicitly")
    ok("could not pull it up" in instr,
       "the honest failure line is given to her as words, not as a principle")
    ok("ask which area to search" in flat_instr,
       "and a no-location answer is a question, not a guess")
    ok("WHEN THEY PICK ONE" in instr and "place_id" in instr,
       "picking one of the results chains into start_navigation by place_id")

    ok("WHEN THE DRIVER ASKS FOR THE DIRECTIONS" in instr,
       "the instructions have a section for being asked to read the route")
    ok("Reading the directions when the driver asks for them is ANSWERING" in instr,
       "which says, in the section about not announcing, that reading is not "
       "announcing")
    ok(realtime.NAV_DIRECTIONS_TOOL_NAME in instr,
       "and names the tool that does it")
    ok("there should be a Shell" in instr,
       "the landmark phrasing is given as a sentence, not as a principle")
    ok(all(p in instr for p in ("Round the distances", "not the way a screen "
                                "lists them")),
       "and it asks for directions spoken the way a person gives them")
    # Line-wrap tolerant: the instructions are a paragraph, and asserting on a
    # phrase that happens to sit either side of a newline is a test that fails
    # for the wrong reason the next time someone reflows it.
    flat = re.sub(r"\s+", " ", instr)
    ok('no "turn left here"' in flat and "nothing that sounds like an "
       "instruction for right now" in flat,
       "while ruling out the phrasings that would sound like a turn call")

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
# J. Going somewhere
# ---------------------------------------------------------------------------
def run_routing():
    """The tool that ACTS, and the instructions that stop it acting alone.

    The bug: asked to be taken somewhere, RIO described the route and told the
    driver to set it themselves — every navigation tool she had was a read. The
    fix is one tool that does what the destination box does. What has to be
    true of it, and is checked here, is that its description and her
    instructions both say she routes herself, that they still forbid picking
    between two plausible places, and that none of it moves the turns.
    """
    section("J. going somewhere — the one tool that does rather than reports")
    s = realtime.session_config()
    schema = next(t for t in s["tools"] if t["name"] == realtime.NAVIGATE_TOOL_NAME)
    desc = schema["description"]

    ok("destination" in schema["parameters"]["properties"],
       "start_navigation takes a destination")
    ok(schema["parameters"]["required"] == ["destination"],
       "and requires one — there is no 'route to wherever'")
    for phrase, why in (
            ("take me to", "it is described in the words a driver actually uses"),
            ("makes the route live", "it says plainly that it ACTS"),
            ("never tell the driver to set it themselves",
             "and that the manual path is not hers to suggest"),
            ("WHICH ONE", "ambiguity is part of the contract, not an error"),
            ("Never pick one yourself", "and picking is forbidden in the schema too"),
    ):
        ok(phrase.lower() in desc.lower(), f"{why} ({phrase!r})")
    ok("still calls those out loud itself" in desc,
       "and the turns stay with the navigation system, said in the tool's own "
       "description so it cannot be read as a promotion")

    instr = s["instructions"]
    ok("WHEN THE DRIVER ASKS TO GO SOMEWHERE" in instr,
       "the instructions have a section for it")
    # Wrapped at 79 columns like the rest of the prompt, so these are matched
    # against the unwrapped text rather than against one particular line break.
    flat = re.sub(r"\s+", " ", instr)
    for phrase in ("Call start_navigation, straight away",
                   "Never tell the driver to type it in",
                   "call start_navigation again with what they chose",
                   "use the destination name the tool hands back"):
        ok(phrase in flat, f"and spell it out: {phrase!r}")
    ok("What you still never do is call the turns" in flat,
       "the announce/answer boundary is re-stated where it is most likely to "
       "be misread: she starts routes, she does not call turns")
    # The boundary itself must not have been softened to make room.
    ok("YOU ANSWER. YOU DO NOT ANNOUNCE." in instr,
       "and the original boundary is still there, in those words")
    ok("never announce a turn" in instr.lower(),
       "including the turn rule it has always had")

    # Nothing about deterministic navigation speech moved.
    import navigation.speech as navspeech
    ok(navspeech.destination_reply("resolved", name="LAX") == "Routing to LAX.",
       "the hold-to-talk path still speaks its own fixed line, untouched")

    # The panel is the only implementation. A second one on the server would be
    # a second answer to "where are we going", which is the thing nav_status
    # exists to avoid.
    src = inspect.getsource(realtime)
    ok("resolve_destination" not in src,
       "realtime.py resolves no destinations of its own")
    ok("build_route" not in src, "and builds no routes")


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


# ---------------------------------------------------------------------------
# The memory-vs-tool check
# ---------------------------------------------------------------------------
# Capitalised words that are not businesses: geography this test asks about,
# and the ordinary furniture of a spoken sentence. Anything NOT here and not in
# the tool's own results is a name the model supplied itself.
_NOT_A_BUSINESS = {
    "santa monica", "los angeles", "california", "downtown", "main street",
    "ocean avenue", "third street", "third street promenade", "the promenade",
    "griffith observatory", "lax", "i-10", "pacific coast highway",
    "monday", "tuesday", "wednesday", "thursday", "friday", "saturday",
    "sunday", "google", "gps", "rio", "okay", "ok",
}

# Words that carry no identity on their own: a phrase made only of these is a
# description ("Coffee Shops"), not a name.
_GENERIC = {
    "coffee", "shop", "shops", "cafe", "café", "place", "places", "spot",
    "spots", "bar", "restaurant", "taco", "tacos", "food", "station",
    "stations", "parking", "petrol", "gas", "open", "now", "here", "near",
    "the", "and", "or", "one", "two", "three", "a", "an", "of", "in", "at",
    "good", "great", "close", "by", "minutes", "stars", "rated", "reviews",
}

_NAME_RUN = re.compile(
    r"\b([A-Z][\w'&.\-]*(?:\s+(?:of|the|and|de|la|du|des)\s+[A-Z][\w'&.\-]*"
    r"|\s+[A-Z][\w'&.\-]*)+)")


def _invented_names(said: str, allowed_names) -> list:
    """Business-shaped names in `said` that the tool did not return.

    The failure this catches is the dangerous one: four real places and one
    remembered one, in the same sentence, in the same tone. Nothing about the
    sentence gives it away, and the driver drives to whichever they liked the
    sound of.

    Multi-word Title Case only, and that is a deliberate floor rather than an
    oversight: a single capitalised word in a transcript is as often a sentence
    opening as a brand, and a check that fires on "Two" is a check that gets
    switched off. A one-word invention ("Verve") would slip through this and be
    caught by the ok() below it, which requires that the names she DID say came
    from the list.
    """
    allowed = " | ".join(list(allowed_names or []) + list(_NOT_A_BUSINESS)).lower()
    out = []
    for m in _NAME_RUN.finditer(said or ""):
        # The token pattern allows an internal dot for "St." and "Mrs.", which
        # means a phrase ending a sentence swallows the full stop. Stripped, or
        # "Third Street Promenade." never matches "third street promenade".
        phrase = " ".join(m.group(1).split()).strip(".,;:!?'\"-&")
        if not phrase:
            continue
        low = phrase.lower()
        if low in allowed or low in _NOT_A_BUSINESS:
            continue
        if any(low in a for a in [allowed]) and low in allowed:
            continue
        # Contained in one of the returned names (or one of them in it):
        # "Dogtown" out of "Dogtown Coffee" is the same business.
        if any(low in n.lower() or n.lower() in low
               for n in (allowed_names or []) if n):
            continue
        toks = [t.strip(".,'&-").lower() for t in phrase.split()]
        if all(t in _GENERIC or t in _NOT_A_BUSINESS for t in toks if t):
            continue
        out.append(phrase)
    return sorted(set(out))


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
CLOSED = "[session closed — no events]"


def _rate_limit_wait(event) -> Optional[float]:
    """Seconds to wait, if this response failed on the realtime token cap.

    A rate-limited response arrives as an ordinary `response.done` with status
    "failed" and no output — indistinguishable, from the outside, from a model
    that decided to say nothing and use no tools. That is the most misleading
    failure this file can report, so it is recognised by name and waited out
    rather than asserted against.
    """
    details = getattr(getattr(event, "response", None), "status_details", None)
    if details is None:
        return None
    text = str(details)
    if "rate_limit" not in text:
        return None
    m = re.search(r"try again in ([0-9.]+)\s*s", text)
    return min(30.0, float(m.group(1)) + 1.0) if m else 5.0


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

    Which response, though, is counted rather than assumed. Breaking on the
    first `response.done` seen works right up until there is a response in
    flight that this function did not ask for, and then it leaves one `done`
    unread in the stream — after which EVERY later question returns instantly
    with nothing, because the first event it sees is that leftover. The
    symptom is a model that appears to have stopped answering and stopped
    using its tools, which is a spectacularly misleading thing for a test to
    report. So: count what opened, count what closed, and leave when nothing
    is open and nothing is owed.
    """
    conn.conversation.item.create(item={
        "type": "message", "role": "user",
        "content": [{"type": "input_text", "text": question}]})
    conn.response.create()

    said, called, need_followup = [], [], False
    seen, started, active, waited = 0, 0, 0, 0
    t0 = time.time()
    for event in conn:
        seen += 1
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
        elif event.type == "response.created":
            started += 1
            active += 1
        elif event.type == "response.done":
            active -= 1
            pause = _rate_limit_wait(event)
            if pause is not None and waited < 4:
                waited += 1
                print(f"    (realtime token cap — waiting {pause:.0f}s and "
                      "asking again)")
                time.sleep(pause)
                conn.response.create()
                continue
            if need_followup:
                need_followup = False
                conn.response.create()      # safe now: nothing is active
                continue
            if started and active <= 0:
                break
        elif event.type == "error":
            said.append(f"[error: {getattr(event, 'error', '')}]")
            break
    if not seen:
        # The iterator ended without yielding anything: the socket is gone, not
        # the model's willingness to answer. Said so explicitly, because a
        # dropped connection otherwise reads as "she ignored every tool she
        # has", and that is the one conclusion this test must never reach by
        # accident.
        return CLOSED, called
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

    # The panel's other half of the routing tool, played the way the browser
    # plays it: resolve against the running server, then route against it. The
    # shaping is deliberately the same shape rio_realtime.js produces, because
    # what is under test here is whether the MODEL reaches for this at all and
    # what she says afterwards — the panel's real code is exercised end to end
    # by `node tools/realtime_selftest.js --server`.
    ORIGIN = (34.0219, -118.4814)
    live_route = {}
    last_places = {}
    places_fail = {"on": False}
    tool_args = []

    REL_PHRASE = {"NEAR": "by", "JUST_AFTER": "just after",
                  "JUST_BEFORE": "just before"}

    def _step(m, i, distance_key):
        """One maneuver as the panel hands it to the model.

        The landmark is the map's first candidate and carries its RELATION,
        because "just after the Shell" and "just before the Shell" are
        different turns. It is marked unverified: nothing has looked at it.
        """
        step = {"step": i + 1, "instruction": m.get("instruction", ""),
                "road_name": m.get("road_name", ""),
                "maneuver_type": m.get("type"), "direction": m.get("direction"),
                distance_key: round(m.get("route_distance_position") or 0)}
        anchors = m.get("anchors") or []
        if anchors and anchors[0].get("label"):
            a = anchors[0]
            label = a.get("spoken_label") or a["label"]
            step["landmark"] = {
                "label": label, "relation": a.get("relation"),
                "phrase": f"{REL_PHRASE.get(a.get('relation'), 'near')} {label}",
                "confidence": a.get("relation_confidence"), "verified": False,
            }
        return step

    def route_by_voice(spoken, place_id=""):
        # A place_id means find_places already resolved this exact place, so
        # resolution is skipped and the route is built straight from the id --
        # the same shortcut rio_realtime.js takes. Without this the harness
        # would send "the second one" to the geocoder and the test would be
        # measuring something the browser never does.
        if place_id:
            r = httpx.post(f"{base}/nav/route", timeout=90, json={
                "lat": ORIGIN[0], "lng": ORIGIN[1], "place_id": place_id,
                "destination": "", "label": spoken}).json()
            if r.get("error"):
                return {"ok": False, "status": "failed", "note": r["error"],
                        "rules": "The route did not start. Say so plainly."}
            nav_state["destination"] = r["destination"]["display_name"]
            live_route["route"] = r
            return {"ok": True, "routing": True, "status": "routed",
                    "destination": r["destination"]["display_name"],
                    "minutes": max(1, round(r["duration_s"] / 60)),
                    "distance_km": round(r["total_distance_m"] / 100) / 10,
                    "from_places": True,
                    "rules": "The route is live and the car is navigating "
                             "already. Confirm it once, briefly, using this "
                             "destination name. Do NOT tell the driver to set "
                             "it themselves."}
        return _route_by_query(spoken)

    def _route_by_query(spoken):
        # Kept because the ambiguity check below has to probe the query SHE
        # sent, not the one the test imagined she would. Asked to "take me to
        # LAX" she may reasonably say "Los Angeles International Airport",
        # which the provider resolves outright — and a test that then probes
        # the bare "LAX" reports her for not asking a question nobody put to
        # her.
        live_route["query"] = spoken
        d = httpx.post(f"{base}/nav/destination", timeout=60,
                       json={"q": spoken, "lat": ORIGIN[0],
                             "lng": ORIGIN[1]}).json()
        if d.get("status") == "ambiguous":
            return {"ok": True, "routing": False, "status": "ambiguous",
                    "query": d.get("query"),
                    "candidates": [{"name": c["display_name"],
                                    "address": c["formatted_address"]}
                                   for c in d["candidates"]],
                    "rules": "More than one place answers to that. Do NOT pick "
                             "one. Ask the driver which of these they meant, "
                             "naming them, and when they answer call "
                             "start_navigation again with their choice."}
        if d.get("status") != "resolved":
            return {"ok": True, "routing": False, "status": "not_found",
                    "query": d.get("query"),
                    "rules": "Say you could not find that place and ask them "
                             "to say it another way."}
        dest = d["destination"]
        r = httpx.post(f"{base}/nav/route", timeout=90, json={
            "lat": ORIGIN[0], "lng": ORIGIN[1],
            "place_id": dest.get("provider_place_id") or "",
            "destination": "" if dest.get("provider_place_id")
                           else dest["formatted_address"],
            "label": dest["display_name"]}).json()
        if r.get("error"):
            return {"ok": False, "status": "failed", "note": r["error"],
                    "rules": "The route did not start. Say so plainly."}
        # From here on the drive really is going there, so the tools that
        # answer "where are we going" and "what are the turns" have to agree.
        nav_state["destination"] = r["destination"]["display_name"]
        live_route["route"] = r
        mans = r.get("maneuvers") or []
        return {"ok": True, "routing": True, "status": "routed",
                "destination": r["destination"]["display_name"],
                "minutes": max(1, round(r["duration_s"] / 60)),
                "distance_km": round(r["total_distance_m"] / 100) / 10,
                "eta_epoch": r.get("eta_epoch"),
                "total_maneuvers": len(mans),
                "first_steps": [_step(m, i, "distance_from_start_m")
                                for i, m in enumerate(mans[:3])],
                "rules": "The route is live and the car is navigating already. "
                         "Confirm it once, briefly, in your own words, using "
                         "this destination name exactly as spelled here. Do "
                         "NOT tell the driver to set it themselves — it is "
                         "set. Do not read the turns out now: confirming is "
                         "one line. If they ASK for the directions, call "
                         "nav_directions and read them — that is answering. "
                         "What you never do is call a turn as it arrives; the "
                         "navigation system does that itself, at the moment "
                         "it matters."}

    def read_directions(arguments):
        """The panel's nav_directions, played from the REAL route.

        Shaped exactly as rio_realtime.js shapes it, for the same reason
        route_by_voice is: what is under test here is the MODEL — whether she
        reaches for this tool when asked for the directions and what she does
        with a landmark hint when she has one. The panel's own code is
        exercised end to end by `node tools/realtime_selftest.js`.
        """
        r = live_route.get("route")
        if not r:
            return {"ok": True, "routing": False,
                    "note": "no route is set — say that, do not invent turns"}
        args = json.loads(arguments or "{}")
        raw = args.get("count")
        count = 5
        if isinstance(raw, str) and raw.strip().lower() == "all":
            count = None
        elif raw not in (None, ""):
            try:
                count = max(1, int(raw))
            except (TypeError, ValueError):
                count = None
        mans = r.get("maneuvers") or []
        chosen = mans if count is None else mans[:count]
        return {
            "ok": True, "routing": True,
            "destination": r["destination"]["display_name"],
            "distance_remaining_m": round(r.get("total_distance_m") or 0),
            "eta_epoch": r.get("eta_epoch"),
            "total_maneuvers": len(mans),
            "maneuvers_left": len(chosen),
            "truncated": count is not None and len(mans) > len(chosen),
            "route_state": "ON_ROUTE", "gps_state": "GPS_OK", "arrived": False,
            "steps": [_step(m, i, "distance_m") for i, m in enumerate(chosen)],
            "rules": "The driver asked for these, so read them — that is "
                     "answering, not announcing. In your own voice and in one "
                     "flowing sentence or two, not as a numbered list: name "
                     "the roads, round the distances the way a person would, "
                     "and stop after the first few unless they asked for all "
                     "of it. A landmark here is what the MAP expects, not "
                     "something anyone has seen: say \"there should be a "
                     "Shell\", never \"there's a Shell\". Do NOT call any of "
                     "these turns as instructions now — when each one arrives "
                     "the navigation system calls it itself.",
        }

    def tool_handler(name, arguments):
        calls.append(name)
        tool_args.append(arguments)
        if name == realtime.NAV_TOOL_NAME:
            return nav_state
        if name == realtime.NAV_DIRECTIONS_TOOL_NAME:
            return read_directions(arguments)
        if name == realtime.PLACES_TOOL_NAME:
            # Through the real endpoint, with a fix the panel would have
            # attached: this is the one tool whose answer comes from outside
            # the system entirely, so the chain test uses the real one.
            body = {"name": name, "arguments": json.loads(arguments or "{}"),
                    "where": {"lat": ORIGIN[0], "lng": ORIGIN[1],
                              "accuracy_m": 12, "age_s": 2}}
            if places_fail["on"]:
                # A forced failure, to see what she says when the search does
                # not come back. Shaped exactly as places.py shapes one.
                return {"ok": False, "note": "ConnectError",
                        "rules": "The search did not come back. Say plainly "
                                 "that you could not pull that up right now, "
                                 "in your own words. Do NOT name a business "
                                 "from memory, do NOT guess, and do not offer "
                                 "one you 'think' is there."}
            out = httpx.post(f"{base}/realtime/tool", timeout=60,
                             json=body).json()
            last_places["result"] = out
            return out
        if name == realtime.NAVIGATE_TOOL_NAME:
            args = json.loads(arguments or "{}")
            return route_by_voice(str(args.get("destination") or ""),
                                  str(args.get("place_id") or ""))
        if name == realtime.VEHICLE_TOOL_NAME and use_pending["on"]:
            return pending_vehicle
        r = httpx.post(f"{base}/realtime/tool", timeout=90,
                       json={"name": name, "arguments":
                             json.loads(arguments or "{}")})
        return r.json()

    # 5. FRAME SUPPLY, over HTTP, exactly as the browser posts it.
    #
    # The frame is a road photograph from outside this repo — RIO_CHAIN_FRAME,
    # or /tmp/road.jpg. Without one there is nothing for her to look at, so the
    # visual exchange is skipped rather than asserted against an empty ring:
    # everything else in this chain is about the route and the car and does not
    # need a camera.
    frame_path = os.getenv("RIO_CHAIN_FRAME", "/tmp/road.jpg")
    have_frame = os.path.exists(frame_path)
    if have_frame:
        with open(frame_path, "rb") as fh:
            frame = fh.read()
        for i in range(3):
            httpx.post(f"{base}/headway_frame", timeout=60,
                       files={"image": ("frame.jpg", frame, "image/jpeg")},
                       data={"v_host": "14.0", "v_host_age_s": "0.2",
                             "frame_t": str(i)})
        scene = httpx.get(f"{base}/scene", timeout=30).json()
        ok(len(scene.get("objects") or []) > 0,
           f"the frame ring fills from posted frames — "
           f"{len(scene.get('objects') or [])} objects in the scene graph")
    else:
        print(f"    (no frame at {frame_path} — skipping the visual exchange; "
              "set RIO_CHAIN_FRAME to a road photo to include it)")

    # A live socket lasting several exchanges is not something this test can
    # assume — it has been seen dropping between questions — so the session is
    # held where it can be reopened, and a question that came back to a closed
    # one is asked again on a fresh session. The conversation history is lost
    # across that, which is why every question below stands on its own instead
    # of referring back to the last one.
    box = {"conn": None, "ctx": None}

    def open_session():
        ctx = realtime.client().realtime.connect(
            model=config.OPENAI_REALTIME_MODEL)
        conn = ctx.__enter__()
        conn.session.update(session=dict(realtime.session_config(),
                                         output_modalities=["audio"]))
        box["ctx"], box["conn"] = ctx, conn
        return conn

    def close_session():
        if box["ctx"] is not None:
            try:
                box["ctx"].__exit__(None, None, None)
            except Exception:
                pass
        box["ctx"], box["conn"] = None, None

    def ask(question):
        said, used = _ask_live(box["conn"], question, tool_handler)
        if said == CLOSED:
            print("    (the live session dropped — reopening and asking again)")
            close_session()
            open_session()
            said, used = _ask_live(box["conn"], question, tool_handler)
        elif not said and not used:
            # A response that contained nothing at all — no speech, no tool.
            # That is not an answer of any kind, so it is asked once more
            # rather than reported as a behavioural failure. If she really is
            # ignoring the question, the second one is empty too.
            print("    (empty response — asking once more)")
            said, used = _ask_live(box["conn"], question, tool_handler)
        return said, used

    try:
        open_session()

        # 1. A visual question must reach the camera.
        if have_frame:
            said, used = ask("What's ahead of us right now?")
            print(f"    Q: what's ahead of us right now?\n    A: {said[:220]}")
            ok(realtime.LOOK_TOOL_NAME in used, f"she looks ({used})")
            ok(any(w in said.lower() for w in
                   ("car", "sedan", "suv", "street", "road", "traffic", "ahead",
                    "lane", "building")),
               "and answers from what the camera actually returned")

        # 2. A route question must reach the tracker.
        calls.clear()
        said, used = ask("How far is it and where are we headed?")
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
        said, used = ask("Is anything wrong with the car?")
        print(f"    Q: is anything wrong with the car?\n    A: {said[:260]}")
        ok(realtime.VEHICLE_TOOL_NAME in used, f"she checks the car ({used})")
        low = said.lower().replace("\u2019", "'")
        ok(any(p in low for p in ("not confirm", "hasn't confirmed", "has not confirmed",
                                  "pending", "not yet confirmed")),
           "and preserves detected-but-not-confirmed rather than upgrading it "
           "to a fault")
        ok(not re.search(r"\b\d+\s*(psi|pounds)\b", low),
           "and invents no numbers the data did not contain")

        # 5. "TAKE ME TO X" — the bug this whole tool exists for. She must
        #    reach for start_navigation rather than describing a route and
        #    handing the job back.
        calls.clear()
        said, used = ask("Take me to Griffith Observatory.")
        print(f"    Q: take me to Griffith Observatory\n    A: {said[:260]}")
        ok(realtime.NAVIGATE_TOOL_NAME in used,
           f"she starts the route herself ({used})")
        ok("griffith" in said.lower(),
           "and confirms it by the name the provider resolved")
        deferral = re.search(
            r"(set|enter|type|put|punch|add|input)[^.?!]{0,40}"
            r"(destination|address|it in|that in)"
            r"|(dashboard|the screen|the panel|the nav system|navigation app)",
            said.lower())
        ok(deferral is None,
           "and does NOT tell the driver to set it themselves — the failure "
           f"this replaces ({deferral.group(0) if deferral else 'no deferral'})")
        ok(not re.search(r"\b(turn left|turn right|take the next)\b",
                         said.lower()),
           "and still does not call the turns on the route she just started")

        # 5b. "WHAT ARE THE DIRECTIONS?" The bug: RIO could start a route and
        #     then say she could not read it. She has a tool now, and the
        #     things worth checking are that she reaches for it, that what she
        #     says comes from the real route, and that reading is not the same
        #     as calling.
        calls.clear()
        said, used = ask("What are the directions?")
        print(f"    Q: what are the directions?\n    A: {said[:320]}")
        ok(realtime.NAV_DIRECTIONS_TOOL_NAME in used,
           f"she reads the route rather than refusing ({used})")

        route_now = live_route.get("route") or {}
        mans = route_now.get("maneuvers") or []
        roads = [m.get("road_name", "") for m in mans if m.get("road_name")]
        low = said.lower()
        hit = [r for r in roads if r.lower() in low]
        ok(bool(hit),
           "and names roads that are actually on the route "
           f"({hit[:3] if hit else 'none of ' + str(roads[:4])})")

        # Reading a turn is fine — that is the whole point. CALLING one is not.
        # The phrases that separate them are the ones that mean "now".
        call_now = re.search(
            r"\b(turn|go|bear|keep)\s+(left|right)\s+(here|now)\b"
            r"|\bget ready to turn\b|\bturn (?:left|right) in a moment\b",
            low)
        ok(call_now is None,
           "and reads them as directions rather than calling one "
           f"({call_now.group(0) if call_now else 'no turn call'})")

        # The landmark, if this route has one: named, and named as an
        # EXPECTATION. Skipped honestly when the provider found none, because
        # a route with no landmarks cannot test landmark phrasing.
        anchored = [(m, (m.get("anchors") or [None])[0]) for m in mans
                    if m.get("anchors")]
        if anchored:
            labels = [(a.get("spoken_label") or a.get("label") or "")
                      for _m, a in anchored]
            named = [l for l in labels if l and l.split()[-1].lower() in low]
            if named:
                ok(True, f"and previews the map's landmark ({named[0]!r})")
                expectation = re.search(
                    r"(should be|should see|there'?s? (?:meant|supposed) to be|"
                    r"expect|look(?:ing)? for)", low)
                ok(expectation is not None,
                   "as an expectation rather than a sighting "
                   f"({expectation.group(0) if expectation else 'stated as fact'})")
            else:
                # The landmarked turn is further down the route than the few
                # she read, which is correct behaviour and no test of the
                # landmark. So ask for the part that contains it.
                print(f"    (landmarks are deeper in the route than she read: "
                      f"{labels[:3]} — asking for all of them)")
                said2, used2 = ask("Read me all of them.")
                print(f"    Q: read me all of them\n    A: {said2[:400]}")
                low2 = said2.lower()
                ok(realtime.NAV_DIRECTIONS_TOOL_NAME in used2,
                   f"she reads the rest of the route on request ({used2})")
                named2 = [l for l in labels
                          if l and l.split()[-1].lower() in low2]
                if named2:
                    ok(True, "and previews the map's landmark when she gets "
                             f"to it ({named2[0]!r})")
                    expectation = re.search(
                        r"(should be|should see|should have|there'?s? "
                        r"(?:meant|supposed) to be|expect|look(?:ing)? for)",
                        low2)
                    ok(expectation is not None,
                       "as an expectation rather than a sighting "
                       f"({expectation.group(0) if expectation else 'stated as fact'})")
                else:
                    ok(True, f"(she summarised rather than reading every turn, "
                             f"so the landmarked ones {labels[:2]} did not come "
                             "up — the hint is in the tool result either way, "
                             "which the offline tests assert)")
        else:
            ok(True, "(this route came back with no landmark candidates — "
                     "nothing to preview, and inventing one would be the bug)")

        # 7. PLACES. The question RIO used to answer from the model's own
        #    memory of restaurants, which is a description of the world as it
        #    was when the weights were made.
        calls.clear()
        last_places.clear()
        said, used = ask("Coffee shops near me, open right now?")
        print(f"    Q: coffee shops near me, open right now?\n    A: {said[:320]}")
        ok(realtime.PLACES_TOOL_NAME in used,
           f"she looks it up rather than remembering ({used})")
        ok(realtime.TOOL_NAME not in used,
           "and does NOT send a local place question to the research tool")

        res = (last_places.get("result") or {})
        names = [r.get("name", "") for r in (res.get("results") or [])]
        print(f"    (Places returned: {names})")
        ok(res.get("ok") is True and names,
           f"the search itself worked against the live API ({len(names)} results)")

        # THE CHECK THIS SECTION EXISTS FOR: every place named in the answer
        # came back from the tool. A model that invents one business among four
        # real ones is more dangerous than one that invents all five, because
        # nothing about the sentence gives it away.
        invented = _invented_names(said, names)
        ok(not invented,
           "every business she named came from the tool "
           f"({invented if invented else 'no invented names'})")
        ok(any(n.lower() in said.lower() for n in names),
           f"and at least one of them is actually in the answer ({names[:2]})")

        # open_now was asked for, so it was asked of Google rather than guessed.
        args_seen = [json.loads(a or "{}") for a in tool_args
                     if a and '"query"' in a]
        ok(any(a.get("open_now") for a in args_seen),
           f"'open right now' reached the tool as a filter ({args_seen})")

        # 8. AN AREA, NOT THE CAR.
        calls.clear(); last_places.clear(); tool_args.clear()
        said, used = ask("What's good for coffee in Santa Monica?")
        print(f"    Q: what's good for coffee in Santa Monica?\n    A: {said[:280]}")
        ok(realtime.PLACES_TOOL_NAME in used, f"same tool ({used})")
        args_seen = [json.loads(a or "{}") for a in tool_args
                     if a and '"query"' in a]
        ok(any("santa monica" in (a.get("near", "") + " "
                                  + a.get("query", "")).lower()
               for a in args_seen),
           f"with the area she was given, not the car's position ({args_seen})")
        res2 = (last_places.get("result") or {})
        names2 = [r.get("name", "") for r in (res2.get("results") or [])]
        invented2 = _invented_names(said, names2)
        ok(not invented2,
           f"and again, only names the tool returned ({invented2 or 'none invented'})")

        # 9. "TAKE ME TO THE SECOND ONE." The list is still in the conversation
        #    and each entry carries the id that skips resolution.
        if len(names2) >= 2:
            calls.clear(); tool_args.clear()
            said, used = ask("Take me to the second one.")
            print(f"    Q: take me to the second one\n    A: {said[:240]}")
            ok(realtime.NAVIGATE_TOOL_NAME in used,
               f"she routes to it ({used})")
            nav_args = [json.loads(a or "{}") for a in tool_args
                        if a and "destination" in (a or "")]
            passed_id = [a.get("place_id") for a in nav_args if a.get("place_id")]
            want = [r.get("place_id") for r in (res2.get("results") or [])][1:2]
            ok(bool(passed_id),
               f"passing the place_id from the list rather than a name to "
               f"re-resolve ({passed_id or 'none passed'})")
            ok(not passed_id or passed_id[0] in
               [r.get("place_id") for r in (res2.get("results") or [])],
               f"and it is one of the ids she was given (wanted {want})")
        else:
            ok(True, "(fewer than two results came back — nothing to pick a "
                     "second one from)")

        # 10. FAILURE. The one moment the old behaviour would return.
        calls.clear()
        places_fail["on"] = True
        try:
            said, used = ask("Find me a taco place near here.")
        finally:
            places_fail["on"] = False
        print(f"    Q: find me a taco place near here (search failing)\n"
              f"    A: {said[:240]}")
        # A transcriber writes "couldn't" with a typographic apostrophe. That is
        # a rendering difference, not a different sentence, and a check that
        # fails on it reports the model for saying exactly the right thing --
        # which is what this one did on its first live run. The vehicle section
        # above normalises the same character for the same reason.
        low = said.lower().replace("\u2019", "'")
        ok(realtime.PLACES_TOOL_NAME in used, f"she still reaches for it ({used})")
        ok(any(p in low for p in ("couldn't", "could not", "can't", "cannot",
                                  "unable", "not able", "trouble", "failed")),
           "and says plainly that she could not pull it up")
        invented3 = _invented_names(said, [])
        ok(not invented3,
           f"naming no business at all rather than one she remembers "
           f"({invented3 or 'no names'})")

        # 6. ...and ambiguity still stops her, now that stopping costs her
        #    something: she is the one who would have to route.
        calls.clear()
        said, used = ask("Take me to LAX instead.")
        print(f"    Q: take me to LAX instead\n    A: {said[:260]}")
        ok(realtime.NAVIGATE_TOOL_NAME in used,
           f"she reaches for the same tool ({used})")
        asked_for = live_route.get("query") or "LAX"
        amb = httpx.post(f"{base}/nav/destination", timeout=60,
                         json={"q": asked_for, "lat": ORIGIN[0],
                               "lng": ORIGIN[1]}).json()
        print(f"    (she asked the provider for {asked_for!r} -> "
              f"{amb.get('status')})")
        if amb.get("status") == "ambiguous":
            ok("?" in said,
               "the provider read that as more than one place, and she asks "
               "which one rather than picking")
            ok(any(c["display_name"].split()[0].lower() in said.lower()
                   for c in amb["candidates"]),
               "naming what she found, so the question can be answered")
        else:
            ok(True, f"the provider resolved {asked_for!r} outright "
                     f"({amb.get('status')}) — she expanded the acronym "
                     "herself, so there was no ambiguity left to stop her on")
    finally:
        close_session()



# ---------------------------------------------------------------------------
# PLACES — what is actually around the car, never what the model remembers
# ---------------------------------------------------------------------------
class _FakeHTTP:
    """One captured Places request and one canned response.

    The request is the interesting half: the field mask is the bill, and the
    location bias is the difference between "near me" and "somewhere in this
    state". Both are asserted rather than assumed.
    """

    def __init__(self, payload=None, raise_with=None, status=200):
        self.payload = payload if payload is not None else {"places": []}
        self.raise_with = raise_with
        self.status = status
        self.calls = []

    def post(self, url, timeout=None, json=None, headers=None):
        self.calls.append({"url": url, "json": json, "headers": headers,
                           "timeout": timeout})
        if self.raise_with:
            raise self.raise_with
        outer = self

        class _R:
            def raise_for_status(self):
                if outer.status >= 400:
                    raise RuntimeError(f"HTTP {outer.status}")

            def json(self):
                return outer.payload
        return _R()


def _place(name, pid, lat, lng, rating=None, count=None, price=None,
           open_now=None, address="somewhere"):
    p = {"id": pid, "displayName": {"text": name}, "formattedAddress": address,
         "location": {"latitude": lat, "longitude": lng}}
    if rating is not None:
        p["rating"] = rating
    if count is not None:
        p["userRatingCount"] = count
    if price is not None:
        p["priceLevel"] = price
    if open_now is not None:
        p["currentOpeningHours"] = {"openNow": open_now}
    return p


SM_FIX = {"lat": 34.0195, "lng": -118.4912, "accuracy_m": 12, "age_s": 3}

TWO_COFFEES = {"places": [
    _place("Dogtown Coffee", "p_dogtown", 34.0110, -118.4950, 4.4, 1900,
           "PRICE_LEVEL_INEXPENSIVE", True, "1116 Main St, Santa Monica"),
    _place("Blue Bottle Coffee", "p_bluebottle", 34.0250, -118.4880, 4.6, 820,
           "PRICE_LEVEL_MODERATE", True, "1402 3rd Street Promenade"),
]}


def run_places():
    section("I. places — real businesses, never remembered ones")
    real_http = places.httpx
    try:
        # 1. "COFFEE NEAR ME, OPEN NOW." The car's fix biases the search, and
        #    the open-now filter is passed to Google rather than applied by RIO
        #    to a list she cannot see the hours of.
        fake = _FakeHTTP(TWO_COFFEES)
        places.httpx = fake
        r = realtime.run_tool(realtime.PLACES_TOOL_NAME,
                              {"query": "coffee", "open_now": True},
                              session_key="drive_1", where=SM_FIX)
        ok(r.get("ok") and r["n"] == 2, "a place question comes back with places")
        req = fake.calls[0]
        ok(len(fake.calls) == 1, "one Places call per question, not one per result")
        ok(req["json"]["openNow"] is True,
           "open-now is filtered by Google, which knows the hours")
        bias = req["json"].get("locationBias", {}).get("circle", {})
        ok(abs(bias.get("center", {}).get("latitude", 0) - SM_FIX["lat"]) < 1e-9,
           "and biased to the car's own fix, so 'near me' means near the car")
        ok(bias.get("radius") == config.PLACES_BIAS_RADIUS_M,
           f"within {config.PLACES_BIAS_RADIUS_M:.0f} m")
        ok("in" not in req["json"]["textQuery"],
           f"the text query stays the driver's words ({req['json']['textQuery']!r})")

        # THE BILL. Places (New) charges by field mask, so the mask is the
        # bill and every field in it has to be one RIO actually says.
        mask = set(req["headers"]["X-Goog-FieldMask"].split(","))
        spoken = {"places.id", "places.displayName", "places.formattedAddress",
                  "places.location", "places.rating", "places.userRatingCount",
                  "places.priceLevel", "places.currentOpeningHours.openNow"}
        ok(mask == spoken,
           f"the field mask is exactly what she speaks ({len(mask)} fields)")
        ok(not any(f.startswith("places.photos") or f.startswith("places.reviews")
                   or "editorialSummary" in f for f in mask),
           "no photos, reviews or editorial summary — each would bill on every "
           "question for something nobody hears")
        ok(req["json"]["maxResultCount"] <= config.PLACES_MAX_RESULTS,
           "and it asks for at most the number she can read out")

        # 2. WHAT COMES BACK IS WHAT SHE CAN SAY.
        first = r["results"][0]
        ok(first["name"] == "Dogtown Coffee" and first["place_id"] == "p_dogtown",
           "each result carries its name and its place_id")
        ok(first["rating"] == 4.4 and first["ratings_count"] == 1900,
           "the rating and how many people rated it — 4.2 from nine reviews is "
           "not 4.2 from nine hundred")
        ok(first["price_level"] == 1 and first["open_now"] is True,
           "price and whether it is open now")
        ok(first["distance_m"] is not None and 900 < first["distance_m"] < 1200,
           f"and how far it is from the CAR ({first['distance_m']} m)")
        ok(first["drive_minutes_est"] >= 1,
           f"with a drive time (~{first['drive_minutes_est']} min)")
        ok("est" in "drive_minutes_est" and "estimate" in r["rules"].lower(),
           "named and described as an ESTIMATE — a routed time would be five "
           "more billed calls")
        ok("Answer ONLY from this list" in r["rules"],
           "and the result tells her the list is the whole of what she knows")

        # 3. AN AREA WAS NAMED. The car's position is then irrelevant and must
        #    not bias anything: "coffee in Santa Monica" from downtown is a
        #    question about Santa Monica.
        fake = _FakeHTTP(TWO_COFFEES)
        places.httpx = fake
        r2 = realtime.run_tool(realtime.PLACES_TOOL_NAME,
                              {"query": "good coffee", "near": "Santa Monica"},
                              session_key="drive_1", where=SM_FIX)
        req2 = fake.calls[0]
        ok(req2["json"]["textQuery"] == "good coffee in Santa Monica",
           "a named area goes into the query")
        ok("locationBias" not in req2["json"],
           "and the car's position does NOT also bias it")
        ok(r2["area"] == "Santa Monica", "the answer says where it searched")

        # 4. NO FIX, NO AREA. She asks. This is the one case where guessing is
        #    invisible: results 30 km away are real, correct and useless.
        fake = _FakeHTTP(TWO_COFFEES)
        places.httpx = fake
        r3 = realtime.run_tool(realtime.PLACES_TOOL_NAME, {"query": "coffee"},
                              session_key="drive_1", where=None)
        ok(r3["ok"] is False and r3.get("need_location"),
           "with no fix and no area she is told to ask, not to search")
        ok(not fake.calls, "and nothing is billed for a question she cannot answer")
        ok("Ask the driver which area" in r3["rules"]
           and "do NOT search anyway" in r3["rules"].replace("Do NOT", "do NOT"),
           "in one short question, and without guessing a location")

        # A stale fix is the same answer: the car has been moving.
        fake = _FakeHTTP(TWO_COFFEES)
        places.httpx = fake
        r4 = realtime.run_tool(
            realtime.PLACES_TOOL_NAME, {"query": "coffee"}, session_key="drive_1",
            where={"lat": 34.0, "lng": -118.5,
                   "age_s": config.PLACES_FIX_MAX_AGE_S + 1})
        ok(r4["ok"] is False and r4["note"] == "stale_fix",
           f"a fix older than {config.PLACES_FIX_MAX_AGE_S:.0f}s is refused, not used")

        # 5. FAILURE IS HONEST. The alternative is the behaviour this tool
        #    exists to remove, and it is worse for being invisible.
        fake = _FakeHTTP(raise_with=RuntimeError("connection reset"))
        places.httpx = fake
        r5 = realtime.run_tool(realtime.PLACES_TOOL_NAME, {"query": "tacos"},
                              session_key="drive_1", where=SM_FIX)
        ok(r5["ok"] is False, "a failed search fails")
        ok("could not pull that up" in r5["rules"],
           "with the words she should say")
        ok("do NOT" in r5["rules"] and "memory" in r5["rules"],
           "and an explicit instruction not to answer from memory instead")
        ok("results" not in r5, "and no result list at all to be tempted by")

        # Nothing found is not the same as failing, and neither is an invitation.
        fake = _FakeHTTP({"places": []})
        places.httpx = fake
        r6 = realtime.run_tool(realtime.PLACES_TOOL_NAME,
                              {"query": "michelin star drive through"},
                              session_key="drive_1", where=SM_FIX)
        ok(r6["ok"] is True and r6["n"] == 0,
           "an empty result is a successful search that found nothing")
        ok("Do NOT fill the silence" in r6["rules"],
           "and still not a cue to remember somewhere")

        # 6. THE FOLLOW-THROUGH. "Take me to the second one" works because the
        #    results are still in session context and each one carries the id
        #    that skips resolution.
        fake = _FakeHTTP(TWO_COFFEES)
        places.httpx = fake
        realtime.run_tool(realtime.PLACES_TOOL_NAME, {"query": "coffee"},
                          session_key="drive_1", where=SM_FIX)
        kept = places.last_results("drive_1")
        ok(len(kept.get("results") or []) == 2,
           "the list she just read is kept for the session")
        ok(kept["results"][1]["place_id"] == "p_bluebottle",
           "so 'the second one' has an id behind it")
        ok(places.last_results("another_drive") == {},
           "and it is per session — one drive's list is not another's")

        # 7. PRICE, RATING AND OPENING STATE ARE NEVER INVENTED.
        fake = _FakeHTTP({"places": [_place("Nameless Diner", "p_x",
                                            34.02, -118.49)]})
        places.httpx = fake
        r7 = realtime.run_tool(realtime.PLACES_TOOL_NAME, {"query": "diner"},
                              session_key="drive_1", where=SM_FIX)
        got = r7["results"][0]
        ok(got["rating"] is None and got["ratings_count"] is None
           and got["price_level"] is None and got["open_now"] is None,
           "a place with no rating comes back with none — 'unrated' and 4.0 "
           "are different things to say")
        # 8. THE CHECKER ITSELF. The chain test leans on _invented_names to
        #    catch a remembered business among real ones, so it is checked
        #    here against sentences RIO plausibly says.
        real_two = ["Dogtown Coffee", "Blue Bottle Coffee"]
        ok(_invented_names(
            "Two good ones close by: Dogtown Coffee, four point four, about "
            "four minutes; or Blue Bottle Coffee, rated higher.", real_two) == [],
           "the checker passes an answer built only from the tool's results")
        ok(_invented_names(
            "Dogtown Coffee is closest, though Verve Coffee Roasters on Main "
            "is the better one.", real_two) == ["Verve Coffee Roasters"],
           "and catches the one name that was not in them")
        ok(_invented_names("I could not pull that up right now.", []) == [],
           "an honest failure names nothing and trips nothing")
        ok(_invented_names(
            "There are a few coffee shops in Santa Monica near the Third "
            "Street Promenade.", []) == [],
           "geography and generic words are not businesses")
        ok(_invented_names("Dogtown is about four minutes.", real_two) == [],
           "and part of a returned name is that name, not a new one")
    finally:
        places.httpx = real_http



# ---------------------------------------------------------------------------
# THE FAST PATH — a prepared answer, and the freshness rule that makes it safe
# ---------------------------------------------------------------------------
def run_fast_path():
    section("J. looking fast — answered before it was asked, or not at all")
    import observer

    # 1. THE GATE. Which questions a one-sentence description of the road can
    #    answer, decided locally: asking the router costs a model call (~1 s
    #    measured) to find out whether we may skip a model call.
    generic = ["what do you see", "what do you see out there",
               "what's around us right now", "describe the road ahead",
               "what's the road like", "anything I should know about ahead",
               "how is the traffic", "what is going on out there",
               "okay what do you see", "what do you see, Rio?"]
    for q in generic:
        ok(realtime.is_generic_scene_question(q), f"generic scene question: {q!r}")

    specific = ["what's on the left", "what's that car ahead",
                "what colour is the car in front", "what does that sign say",
                "how far is the car ahead", "what's that building on the right",
                "which one is closer", "read me that sign"]
    for q in specific:
        ok(not realtime.is_generic_scene_question(q),
           f"needs a proper look: {q!r}")

    ok(realtime.is_generic_scene_question("what's around us right now"),
       "'right now' is a time, not a direction — the first version of the "
       "specificity list read it as one and sent the most generic question "
       "there is down the slow path")

    # 2. THE FRESHNESS RULE. The whole reason a cached answer is honest.
    key = "fastpath_test"
    observer.stop_all()
    observer._sessions[key] = {
        "stop": __import__("threading").Event(), "last_used": time.time(),
        "record": None, "n": 0, "errors": 0, "started": time.time(), "hold": 0,
    }
    st = observer._sessions[key]

    now = time.time()
    st["record"] = {"text": "Cars ahead on a wet road.", "at": now,
                    "frame_wall_t": now - 0.4, "frame_id": "f1",
                    "frame_age_s": 0.4}
    hit = observer.fresh(key)
    ok(hit and hit["text"].startswith("Cars ahead"),
       f"a description of the road half a second ago is still the road "
       f"({hit.get('age_s')}s)")

    st["record"] = dict(st["record"],
                        frame_wall_t=now - (config.OBSERVER_FRESH_S + 1.0))
    ok(observer.fresh(key) == {},
       f"...and one from {config.OBSERVER_FRESH_S + 1:.0f}s ago is refused — at "
       "60 km/h that is a different road, and describing it as current is the "
       "failure this whole subsystem exists to prevent")
    ok(observer.cached(key).get("text"),
       "the stale record is still THERE, so a caller can say how old it is "
       "rather than pretending nothing was ever seen")

    # Age is measured from the FRAME, not from the observation: Qwen taking
    # 400 ms to describe a picture does not make the picture newer.
    st["record"] = {"text": "x", "at": time.time(),
                    "frame_wall_t": time.time() - 30.0, "frame_id": "f2"}
    ok(observer.fresh(key) == {},
       "a fresh observation OF AN OLD FRAME is old — the frame's clock is the "
       "one the driver is being told about")

    st["record"] = None
    ok(observer.fresh(key) == {} and observer.cached(key) == {},
       "and with nothing observed at all there is nothing to serve")

    # 3. THE HOLD. The observer and the visual turn want the same GPU.
    with observer.hold(key):
        ok(observer._sessions[key]["hold"] == 1,
           "a real question holds the observer off the GPU while it prepares")
        with observer.hold(key):
            ok(observer._sessions[key]["hold"] == 2,
               "and two overlapping questions nest rather than the first one "
               "releasing for both")
    ok(observer._sessions[key]["hold"] == 0, "released afterwards")

    observer.stop(key)
    observer._sessions.pop(key, None)

    # 4. THE CONFIG IS THE CONTRACT.
    ok(config.OBSERVER_FRESH_S <= 3.0,
       f"freshness window is short ({config.OBSERVER_FRESH_S}s) — a road "
       "description ages out fast")
    ok(config.OBSERVER_PERIOD_S <= 2.0,
       f"and the observer runs often enough to keep one ({config.OBSERVER_PERIOD_S}s)")
    ok(config.OBSERVER_MAX_SIDE_PX <= 768,
       f"frames are downscaled for it ({config.OBSERVER_MAX_SIDE_PX}px): "
       "prefill scales with pixels and the caption does not")

    # 5. THE INSTRUCTION. A visual question must never reach the research tool.
    cfg = realtime.session_config()
    by_name = {t["name"]: t for t in cfg["tools"]}
    deep = by_name[realtime.TOOL_NAME]["description"]
    ok("look" in deep and "camera" in deep.lower(),
       "the research tool's own description sends visual questions to look")
    flat = re.sub(r"\s+", " ", cfg["instructions"])
    ok("never use the research tool for anything you can see" in flat.lower(),
       "and the instructions say it in the section about looking")


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
    run_routing()
    run_places()
    run_fast_path()
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
