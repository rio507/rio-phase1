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
    ok(tools == [realtime.TOOL_NAME],
       f"exactly one tool, and it is the escalation ({tools})")
    schema = s["tools"][0]
    ok("question" in schema["parameters"]["properties"],
       "which takes a question")
    ok("route" in schema["description"] or "car" in schema["description"],
       "and is told what NOT to use it for — the car and the route are answered "
       "elsewhere")

    instr = s["instructions"]
    ok(config.SYSTEM_PROMPT.strip()[:200] in instr,
       "RIO's own personality prompt carries over verbatim")
    ok("interrupted" in instr.lower(),
       "with the live-only part added: she expects to be interrupted")
    ok("navigation" in instr.lower() and "never" in instr.lower(),
       "and is told never to give navigation instructions")
    ok("ok: false" in instr,
       "and what to do when the tool fails, which is: carry on")
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
    unreachable = {"navigation", "headway", "vehicle_health", "vehicle_health_policy",
                   "tires", "telemetry", "voice"}
    hits = sorted(live_imports & unreachable)
    ok(not hits,
       "realtime.py cannot reach navigation, headway, vehicle health or the TTS "
       "path" + (f" (found {hits})" if hits else ""))
    ok(live_imports <= {"json", "os", "threading", "time", "typing", "openai",
                        "config"},
       f"and imports only what a session needs ({sorted(live_imports)})")

    # The instructions are the other half of it: the model is told in words,
    # because a model that CAN say a sentence has to be told not to.
    addendum = realtime.LIVE_ADDENDUM.lower()
    ok("never give navigation instructions" in addendum,
       "RIO is told never to give navigation instructions herself")
    ok("sensors" in addendum and "invent" in addendum,
       "and never to invent anything about the car's sensors")
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
# Live
# ---------------------------------------------------------------------------
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


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--live", action="store_true",
                    help="also call both real models once")
    args = ap.parse_args()

    run_session()
    run_config()
    run_dispatch()
    run_failure()
    run_firewall()
    run_endpoints()
    if args.live:
        run_live()

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
