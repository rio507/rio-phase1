"""Does the gpt-live-1 backend actually work in the car's shape?

    python -m tools.live_selftest
    python -m tools.live_selftest --only tools
    python -m tools.live_selftest --quick

Five questions, and they are the five that would each individually make the
backend unshippable if the answer were no:

  session    does a session mint WITH DELEGATION CONFIGURED? Not "does it
             mint" -- a session that comes up without the backend attached is
             a RIO who can talk and cannot think, and it comes up perfectly
             happily.
  tools      does each of RIO's tools round-trip through the backend? The
             call has to arrive nested in a delegation envelope, the result
             has to go back as two events, and she has to then SAY something
             about it. Every one of those three is a place the loop stops
             silently.
  verbatim   does a deterministic line still arrive word for word, in Gleam?
             This is the one the migration is most likely to quietly break,
             because commentary.append is documented as free to paraphrase.
  bargein    can the driver cut her off, and does the session recover to
             answer the next question? Full duplex is the headline feature
             and this is the only claim in it that matters in a car.
  arbiter    are the announcement priorities untouched? Deliberately NOT a
             live test -- it shells out to the existing JS selftest, because
             the arbiter is not supposed to have changed and the way to prove
             that is to run its own suite unmodified.

Exit code is the number of failures, so CI can read it.
"""
import argparse
import asyncio
import json
import subprocess
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
sys.path.insert(0, str(Path(__file__).resolve().parent))

import config                                               # noqa: E402
import live                                                 # noqa: E402
from live_harness import (LiveSession, ScriptedMic, RATE,   # noqa: E402
                          say_as_driver, silence)

OK, BAD = "ok  ", "FAIL"


def norm(s: str) -> str:
    return " ".join("".join(c.lower() if (c.isalnum() or c.isspace()) else " "
                            for c in (s or "")).split())


# WHAT EACH TOOL IS ASKED AND WHAT IT IS HANDED BACK.
#
# The payloads are plausible rather than real: this is testing the LOOP, not
# the tool. A test that ran the real camera would fail when the camera failed
# and tell you nothing about delegation.
TOOL_CASES = [
    ("vehicle_status", "How are my tires?",
     {"tires": {"fl": 31, "fr": 31, "rl": 30, "rr": 30, "unit": "psi"},
      "rules": ["all four are within the normal range"]}),
    ("nav_status", "How far is it to the Getty now?",
     {"destination": "The Getty Center", "eta_min": 14,
      "next": "turn right onto Sepulveda", "in_m": 400}),
    ("nav_directions", "Read me the directions.",
     {"steps": ["Right onto Sepulveda", "Merge onto the 405 north",
                "Exit at Getty Center Drive"]}),
    ("find_places", "Is there a coffee shop near here?",
     {"places": [{"name": "Blue Bottle", "dist_m": 320, "open": True}]}),
    ("start_navigation", "Take me to the Getty.",
     {"started": True, "destination": "The Getty Center", "eta_min": 16}),
    ("look", "What kind of car is in front of us?",
     {"answer": "A grey Audi wagon, maybe three car lengths ahead."}),
    ("deep_dive", "Look up why carmakers switched to electric power steering.",
     {"answer": "Mostly fuel economy: a hydraulic pump runs off the engine "
                "all the time, an electric motor only draws when you steer."}),
]


async def t_session() -> list:
    """Mint, and check the backend really is attached."""
    fails = []
    mic = ScriptedMic(silence(3000))
    s = LiveSession(mic)
    try:
        await s.open()
    except Exception as e:
        return [f"session did not mint: {e}"]
    sess = s.session or {}
    dele = sess.get("delegation") or {}
    resp = dele.get("responses") or {}
    print(f'  model={sess.get("model")} voice='
          f'{((sess.get("audio") or {}).get("output") or {}).get("voice")}')
    print(f'  delegation={dele.get("type")} backend={resp.get("model")} '
          f'tools={len(resp.get("tools") or [])}')
    if sess.get("model") != config.GPT_LIVE_MODEL:
        fails.append(f'model is {sess.get("model")}')
    voice = ((sess.get("audio") or {}).get("output") or {}).get("voice")
    if voice != config.GPT_LIVE_VOICE:
        fails.append(f"voice is {voice}, wanted {config.GPT_LIVE_VOICE}")
    if dele.get("type") != config.GPT_LIVE_DELEGATION:
        fails.append(f'delegation type is {dele.get("type")!r}')
    if resp.get("model") != config.GPT_LIVE_BACKEND_MODEL:
        fails.append(f'backend model is {resp.get("model")!r}')
    if len(resp.get("tools") or []) != len(live.backend_tools()):
        fails.append(f'backend got {len(resp.get("tools") or [])} tools, '
                     f"sent {len(live.backend_tools())}")
    await s.close()
    return fails


async def _one_tool(name: str, question: str, payload: dict) -> tuple:
    """One question, one expected tool, one spoken answer afterwards."""
    audio = np.concatenate([silence(400), say_as_driver(question),
                            silence(9000)])
    mic = ScriptedMic(audio)
    seen = []

    def tool_fn(called, args):
        seen.append(called)
        return payload

    s = LiveSession(mic, tool_fn=tool_fn)
    await s.open()
    await s.wait_quiet(max_s=40.0, quiet_s=2.5)
    await s.close()
    return seen, s.said.strip(), s.errors


async def t_tools(quick: bool) -> list:
    fails = []
    cases = TOOL_CASES[:2] if quick else TOOL_CASES
    for name, q, payload in cases:
        try:
            seen, said, errs = await _one_tool(name, q, payload)
        except Exception as e:
            fails.append(f"{name}: {type(e).__name__}: {e}")
            continue
        hit = name in seen
        spoke = bool(said)
        print(f'  {OK if (hit and spoke) else BAD} {name:<17} '
              f'called={seen or "-"} said={said[:58]!r}')
        if not hit:
            fails.append(f"{name}: backend called {seen or 'nothing'}")
        elif not spoke:
            fails.append(f"{name}: tool returned but she said nothing")
        if errs:
            fails.append(f"{name}: {errs[0]}")
    return fails


# The deterministic lines, from the modules that own them, so this test and
# the policy that fires them cannot drift apart.
def _det_lines() -> list:
    from headway import live_policy
    lines = [live_policy.LINE_TEXT[k] for k in ("calm", "escalate")
             if k in live_policy.LINE_TEXT]
    lines.append("In 300 feet, turn right onto Ocean Avenue.")
    lines.append("Front left tire is down to 24 PSI.")
    return lines


async def t_verbatim(quick: bool) -> list:
    """Does a dictated line arrive word for word, in her voice?"""
    fails = []
    lines = _det_lines()[:2] if quick else _det_lines()
    mic = ScriptedMic(silence(60000))
    s = LiveSession(mic)
    await s.open()
    await asyncio.sleep(1.0)
    for i, line in enumerate(lines):
        s.said = ""
        s.first_audio_t = None
        s.last_audio_t = None
        t0 = time.time()
        s.say_line(line, event_id=f"v{i}")
        await s.wait_quiet(max_s=15.0, quiet_s=0.9)
        said = s.said.strip()
        exact = norm(said) == norm(line)
        ms = (s.first_audio_t - t0) * 1000 if s.first_audio_t else None
        print(f'  {OK if exact else BAD} '
              f'[{f"{ms:.0f} ms" if ms else "no audio":>9}] {said[:52]!r}')
        if not exact:
            fails.append(f"not verbatim: wanted {line!r} got {said!r}")
        if ms is None:
            fails.append(f"no audio for {line!r}")
    await s.close()
    return fails


async def t_bargein() -> list:
    """Cut her off mid-sentence, then ask something else.

    The claim under test is not "she stops" -- it is that the session is still
    usable afterwards. An interruption that leaves the session wedged is worse
    than one that never happened, and that is the failure the old backend's
    resume rule exists to bound.
    """
    fails = []
    # A long line to talk over, then the driver cutting in 1.2 s later.
    interrupt = say_as_driver("Actually, how are my tires?")
    mic = ScriptedMic(np.concatenate([silence(1500), interrupt,
                                      silence(14000)]))
    seen = []
    s = LiveSession(mic, tool_fn=lambda n, a: (seen.append(n) or
                                               {"tires": "all fine"}))
    await s.open()
    await asyncio.sleep(0.6)
    s.say_line("Let me tell you about the history of this road, which is a "
               "very long story that goes back many decades and involves a "
               "great many people and a great many arguments about where it "
               "ought to run.", event_id="long")
    # PATIENT, because "she finished the line first" and "she ignored the
    # driver" are different outcomes and a short window reports them as the
    # same one. A verbatim directive is an instruction to say a specific
    # sentence, and she may reasonably finish it before turning to the
    # question -- what must not happen is the question going unanswered.
    await s.wait_quiet(max_s=45.0, quiet_s=3.5)
    said = s.said.strip()
    print(f'  heard={s.heard.strip()[:50]!r}')
    print(f'  said ={said[:80]!r}')
    print(f'  tools={seen}')
    if not said:
        fails.append("she never spoke at all")
    if "tire" not in said.lower() and not seen:
        fails.append("the interrupting question was never answered")
    await s.close()
    return fails


def t_arbiter() -> list:
    """The arbiter's own suite, unmodified, as the proof it did not change."""
    js = Path(__file__).resolve().parent / "one_voice_selftest.js"
    if not js.exists():
        return ["tools/one_voice_selftest.js is missing"]
    r = subprocess.run(["node", str(js)], capture_output=True, text=True,
                       cwd=str(Path(__file__).resolve().parent.parent))
    tail = (r.stdout or r.stderr).strip().splitlines()[-3:]
    for ln in tail:
        print(f"  {ln}")
    return [] if r.returncode == 0 else [f"one_voice_selftest exit {r.returncode}"]


async def main_async(args) -> int:
    if config.VOICE_BACKEND != "gpt_live":
        print(f"[live_selftest] VOICE_BACKEND is {config.VOICE_BACKEND!r}; "
              "this suite tests the gpt_live backend. Re-running against it "
              "anyway — the session config is read from live.py either way.")
    picks = [x.strip() for x in (args.only or "").split(",") if x.strip()]
    todo = [("session", t_session()),
            ("tools", t_tools(args.quick)),
            ("verbatim", t_verbatim(args.quick)),
            ("bargein", t_bargein())]
    fails = {}
    for name, coro in todo:
        if picks and name not in picks:
            coro.close()
            continue
        print(f"\n== {name}")
        try:
            fails[name] = await coro
        except Exception as e:
            fails[name] = [f"{type(e).__name__}: {e}"]
        for f in fails[name]:
            print(f"  {BAD} {f}")
    if not picks or "arbiter" in picks:
        print("\n== arbiter")
        fails["arbiter"] = t_arbiter()
        for f in fails["arbiter"]:
            print(f"  {BAD} {f}")
    n = sum(len(v) for v in fails.values())
    print("\n" + "-" * 58)
    for k, v in fails.items():
        print(f'  {OK if not v else BAD} {k:<10} {len(v)} failure(s)')
    print(f'  {"PASS" if n == 0 else "FAIL"}: {n} failure(s) total')
    return n


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--only", default="",
                    help="session,tools,verbatim,bargein,arbiter")
    ap.add_argument("--quick", action="store_true",
                    help="two tools and two lines instead of all of them")
    return asyncio.run(main_async(ap.parse_args()))


if __name__ == "__main__":
    raise SystemExit(main())
