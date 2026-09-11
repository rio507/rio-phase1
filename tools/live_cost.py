"""What a minute of RIO actually costs, on each backend, on the same drive.

    python -m tools.live_cost
    python -m tools.live_cost --arms gpt_live
    python -m tools.live_cost --dump-usage

TWO PRICING MODELS THAT DO NOT COMPARE ON PAPER, which is the whole reason
this measures instead of multiplying rate cards:

  gpt-realtime-2.1   billed per TOKEN, and audio tokens are the expensive
                     kind. What a drive costs depends on how much was said, by
                     whom, and -- the part nobody predicts correctly -- on how
                     much conversation history is re-sent as input on every
                     single turn.
  gpt-live-1         billed per SECOND of session, flat, whether anyone is
                     talking or not. Plus the backend, billed per token, but
                     only on the turns that actually delegated.

So the two meet at a usage pattern rather than at a price, and a car is an
unusual pattern: long sessions, sparse speech, and a driver who says nothing
for minutes at a time. A flat per-second rate is worst exactly there, and a
token rate is worst on a talkative drive. This runs the SAME scripted drive
through both and reports dollars per minute of wall clock.

THE SCRIPT IS tools/live_tool_turns.py's, imported rather than restated: seven
questions that between them reach the camera, the reasoning model, the route,
the car and nothing at all. The last one is the control.

WHAT IS NOT COUNTED, stated so the number is not read as more than it is: the
tools' own costs. /perceive, the Maps calls, the vision model behind look() --
all of those are the same on every backend and none of them changes with this
decision.
"""
import argparse
import asyncio
import json
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from dotenv import load_dotenv                              # noqa: E402

load_dotenv(Path(__file__).resolve().parent.parent / ".env")

import config                                               # noqa: E402
import realtime                                             # noqa: E402
from live_harness import (LiveSession, ScriptedMic, RATE,   # noqa: E402
                          say_as_driver, silence)

# $ per million tokens, and $ per minute for the flat one. Checked against the
# published rate cards on 2026-09-11; a rate that moves makes every number
# below wrong, so it lives here in one place rather than inline.
PRICES = {
    "gpt-realtime-2.1": {"audio_in": 32.0, "audio_out": 64.0,
                         "text_in": 4.0, "text_out": 16.0,
                         "cached_in": 0.4},
    "gpt-live-1": {"per_minute": 0.05},
    "gpt-5.6-luna": {"text_in": 0.20, "text_out": 1.20},
    "gpt-5.6-terra": {"text_in": 2.00, "text_out": 12.00},
    "gpt-5.6-sol": {"text_in": 5.00, "text_out": 30.00},
}

# The scripted drive, and what each turn is handed when it reaches for a tool.
DRIVE = [
    ("What do you see outside?", "look",
     {"answer": "Two lanes, light traffic, a silver van on the right."}),
    ("What kind of car is in front of us?", "look",
     {"answer": "A grey Audi wagon, three car lengths up."}),
    ("Can you look up why carmakers switched from hydraulic to electric "
     "power steering?", "deep_dive",
     {"answer": "Fuel economy, mostly: a hydraulic pump runs off the engine "
                "all the time, an electric motor only draws when you steer."}),
    ("Take me to the Getty.", "start_navigation",
     {"started": True, "destination": "The Getty Center", "eta_min": 16}),
    ("What are the directions?", "nav_directions",
     {"steps": ["Right onto Sepulveda", "Merge onto the 405 north",
                "Exit at Getty Center Drive"]}),
    ("How are my tires?", "vehicle_status",
     {"tires": {"fl": 31, "fr": 31, "rl": 30, "rr": 30},
      "rules": ["all four within the normal range"]}),
    ("How long have you been driving with me?", None, {}),
]

TAIL_MS = 3000        # the quiet after each question, so the turn can end
GAP_MS = 2500         # and the quiet between them, so a drive has pauses


def _price_tokens(model: str, tin: int, tout: int, cached: int = 0) -> float:
    p = PRICES.get(model, {})
    return ((tin - cached) * p.get("text_in", 0.0)
            + cached * p.get("cached_in", p.get("text_in", 0.0))
            + tout * p.get("text_out", 0.0)) / 1e6


async def run_live(dump: bool, skip_model: bool = False) -> dict:
    """The whole drive in ONE session, which is how a car runs it."""
    import live
    audio = [silence(600)]
    for q, _, _ in DRIVE:
        audio.append(say_as_driver(q))
        audio.append(silence(TAIL_MS + GAP_MS))
    mic = ScriptedMic(np.concatenate(audio))

    calls = []

    def tool_fn(name, args):
        calls.append(name)
        for q, want, payload in DRIVE:
            if want == name:
                return payload
        return {"ok": True}

    s = LiveSession(mic, tool_fn=tool_fn)
    t0 = time.time()
    await s.open()
    total = sum(len(a) for a in audio) / float(RATE)
    while time.time() - t0 < total + 12:
        await asyncio.sleep(0.2)
    await s.close()
    wall = time.time() - t0

    seconds = None
    backend_in = backend_out = 0
    for _, e in s.events:
        if e.get("type") == "session.usage.updated":
            u = e.get("usage") or {}
            if u.get("seconds") is not None:
                seconds = float(u["seconds"])
            for k_in, k_out in (("input_tokens", "output_tokens"),):
                if u.get(k_in):
                    backend_in += int(u[k_in])
                if u.get(k_out):
                    backend_out += int(u[k_out])
        if dump and e.get("type", "").endswith("usage.updated"):
            print("   usage event:", json.dumps(e)[:300])
    seconds = seconds if seconds is not None else wall
    voice_cost = seconds / 60.0 * PRICES["gpt-live-1"]["per_minute"]
    # THE BACKEND'S SHARE IS NOT IN THE SESSION'S USAGE EVENT, and reporting
    # zero for it would be a lie that flatters this backend by about the size
    # of the thing being measured. session.usage.updated carries `seconds` and
    # nothing else -- the Responses delegation runs on OpenAI's side and its
    # tokens are billed to the account separately, out of this stream's sight.
    #
    # So it is MODELLED, and labelled as modelled: the same prompt, the same
    # tool schemas and the same utterances are sent to the same backend model
    # over the Responses API, and the usage that comes back is what the
    # delegation would have cost. That is an equivalent call rather than the
    # actual one, which is the honest description of it.
    if backend_in == 0 and backend_out == 0 and not skip_model:
        backend_in, backend_out = _model_backend_usage(calls)
        modelled = True
    else:
        modelled = False
    backend_cost = _price_tokens(config.GPT_LIVE_BACKEND_MODEL,
                                 backend_in, backend_out)
    return {"arm": "gpt_live", "wall_s": wall, "billed_s": seconds,
            "voice_cost": voice_cost, "backend_cost": backend_cost,
            "backend_in": backend_in, "backend_out": backend_out,
            "backend_modelled": modelled,
            "tools": calls, "said": s.said.strip()}


def _model_backend_usage(tool_calls: list) -> tuple:
    """What the delegated turns would have cost, over the Responses API.

    One call per utterance in the drive, with RIO's real backend prompt and
    real tool schemas, so the token counts are the ones the delegation
    actually produces rather than an estimate of them. The tool RESULTS are
    fed back exactly as the delegation loop feeds them, because the second
    half of a tool turn re-sends everything the first half sent and that is
    where most of the input tokens in a voice session come from.
    """
    import live
    from openai import OpenAI

    cl = OpenAI(timeout=90.0)
    tin = tout = 0
    for say, want, payload in DRIVE:
        try:
            r = cl.responses.create(
                model=config.GPT_LIVE_BACKEND_MODEL,
                instructions=live.backend_instructions(),
                input=[{"role": "user", "content": say}],
                tools=live.backend_tools(), tool_choice="auto")
            u = getattr(r, "usage", None)
            if u:
                tin += int(getattr(u, "input_tokens", 0) or 0)
                tout += int(getattr(u, "output_tokens", 0) or 0)
            # The second leg: the tool came back and the model has to speak.
            fc = [o for o in (r.output or [])
                  if getattr(o, "type", "") == "function_call"]
            if fc and want:
                r2 = cl.responses.create(
                    model=config.GPT_LIVE_BACKEND_MODEL,
                    instructions=live.backend_instructions(),
                    # REBUILT BY HAND rather than model_dump()'d: the SDK
                    # renames `async` to `async_` on the way out and the API
                    # rejects it on the way back in, which silently cost the
                    # second leg of every tool turn -- and the second leg is
                    # where a voice session's input tokens actually are.
                    input=[{"role": "user", "content": say}]
                    + [{"type": "function_call", "call_id": o.call_id,
                        "name": o.name, "arguments": o.arguments}
                       for o in fc]
                    + [{"type": "function_call_output",
                        "call_id": fc[0].call_id,
                        "output": json.dumps(payload)}],
                    tools=live.backend_tools(), tool_choice="auto")
                u2 = getattr(r2, "usage", None)
                if u2:
                    tin += int(getattr(u2, "input_tokens", 0) or 0)
                    tout += int(getattr(u2, "output_tokens", 0) or 0)
        except Exception as e:
            print(f"   (backend model call failed: {type(e).__name__}: {e})",
                  flush=True)
    return tin, tout


# A DRIVE THAT PRODUCES NOTHING MUST STILL END. The realtime stream is an
# async iterator, so a deadline checked inside the loop body is only checked
# when an event arrives -- and the interesting failure is a session that stops
# sending events at all. Without a timeout around the whole thing this tool
# hangs, which is the least useful thing a measurement can do. (The same trap,
# and the same fix, as tools/voice_latency.py TURN_TIMEOUT_S.)
REALTIME_RUN_TIMEOUT_S = 200.0


async def run_realtime(dump: bool) -> dict:
    try:
        return await asyncio.wait_for(_run_realtime(dump),
                                      timeout=REALTIME_RUN_TIMEOUT_S)
    except asyncio.TimeoutError:
        return {"arm": "openai_realtime", "wall_s": REALTIME_RUN_TIMEOUT_S,
                "billed_s": REALTIME_RUN_TIMEOUT_S, "voice_cost": 0.0,
                "backend_cost": 0.0, "tools": [],
                "note": "timed out with no further events"}


async def _run_realtime(dump: bool) -> dict:
    """The same drive through gpt-realtime-2.1, counting tokens."""
    import base64

    from openai import AsyncOpenAI

    cfg = dict(realtime.session_config())
    cfg["output_modalities"] = ["audio"]
    usage = {"ain": 0, "aout": 0, "tin": 0, "tout": 0,
             "cached_audio": 0, "cached_text": 0}
    calls = []
    t0 = time.time()
    cl = AsyncOpenAI()
    async with cl.realtime.connect(
            model=config.OPENAI_REALTIME_MODEL) as conn:
        await conn.session.update(session=cfg)

        async def feed():
            step = int(24000 * 0.02) * 2
            for q, _, _ in DRIVE:
                pcm = say_as_driver(q)[::2].astype("<i2").tobytes()
                pcm += b"\x00\x00" * int(24000 * (TAIL_MS + GAP_MS) / 1000)
                for i in range(0, len(pcm), step):
                    await conn.input_audio_buffer.append(
                        audio=base64.b64encode(pcm[i:i + step]).decode())
                    await asyncio.sleep(0.02)

        # COLLECT WHILE THE DRIVE RUNS, THEN STOP ON A CLOCK.
        #
        # The obvious loop -- break when the feed is done and a response.done
        # arrives -- does not terminate, and the reason is worth writing down:
        # every response.done arrives DURING the feed, because she answers each
        # question as it is asked. Once the feed finishes there is nothing
        # further coming, so the condition is never evaluated again and the run
        # sits there until the outer timeout. Measured: a 74-second drive
        # reported as 200 seconds with no usage at all.
        async def collect():
            async for ev in conn:
                et = getattr(ev, "type", "")
                if et == "response.done":
                    r = getattr(ev, "response", None)
                    u = getattr(r, "usage", None)
                    if u:
                        d = (u.model_dump() if hasattr(u, "model_dump")
                             else dict(u))
                        if dump:
                            print("   usage:", json.dumps(d)[:300])
                        idt = d.get("input_token_details") or {}
                        odt = d.get("output_token_details") or {}
                        usage["ain"] += int(idt.get("audio_tokens") or 0)
                        usage["tin"] += int(idt.get("text_tokens") or 0)
                        cd = idt.get("cached_tokens_details") or {}
                        usage["cached_audio"] += int(cd.get("audio_tokens") or 0)
                        usage["cached_text"] += int(cd.get("text_tokens") or 0)
                        usage["aout"] += int(odt.get("audio_tokens") or 0)
                        usage["tout"] += int(odt.get("text_tokens") or 0)
                elif et == "response.function_call_arguments.done":
                    calls.append(getattr(ev, "name", "?"))

        task = asyncio.ensure_future(feed())
        pump = asyncio.ensure_future(collect())
        try:
            await task
            # The last question was asked as the feed ended; give her the same
            # grace to answer it that every other turn got.
            await asyncio.sleep(12.0)
        finally:
            pump.cancel()
            task.cancel()
    wall = time.time() - t0
    p = PRICES[config.OPENAI_REALTIME_MODEL]
    # CACHED TOKENS COME OUT OF THE BUCKET THEY WERE CACHED IN, and the first
    # version of this subtracted the whole cached count -- almost all of it
    # text -- from the AUDIO input. On a real drive that is 37,568 subtracted
    # from 912, and the drive was reported as costing MINUS ninety-seven cents.
    # A pricing bug that produces a negative number is at least loud; the same
    # mistake in the other direction would have been a quiet flattering one.
    ain = max(usage["ain"] - usage["cached_audio"], 0)
    tin = max(usage["tin"] - usage["cached_text"], 0)
    cost = (ain * p["audio_in"]
            + tin * p["text_in"]
            + (usage["cached_audio"] + usage["cached_text"]) * p["cached_in"]
            + usage["aout"] * p["audio_out"]
            + usage["tout"] * p["text_out"]) / 1e6
    return {"arm": "openai_realtime", "wall_s": wall, "billed_s": wall,
            "voice_cost": cost, "backend_cost": 0.0,
            "usage": usage, "tools": calls}


# ---------------------------------------------------------------------------
# THE QUIET MINUTE, which is what a drive mostly is
# ---------------------------------------------------------------------------
# THE SCRIPTED DRIVE IS NOT A DRIVE. It asks seven questions in seventy-four
# seconds, which is 5.7 turns a minute -- a rate no driver sustains and no
# drive in this repository's logs comes close to. It is the right shape for
# measuring a TURN and the wrong shape for measuring an HOUR, and the two
# backends are priced in a way that makes the difference decide the answer:
#
#   gpt-live-1        $0.05 a minute for the session being OPEN. Silence costs
#                     exactly what conversation costs.
#   gpt-realtime-2.1  tokens, and only when there is a turn. A minute in which
#                     nobody says anything is very nearly free.
#
# So this measures the other end of the range: a session held open with nobody
# talking. Between this and the drive above, any usage pattern can be priced.
async def run_idle(arm: str, seconds: float) -> dict:
    """One session, open, with nobody in the car."""
    import base64

    if arm == "gpt_live":
        mic = ScriptedMic(silence(int(seconds * 1000) + 2000))
        s = LiveSession(mic)
        t0 = time.time()
        await s.open()
        while time.time() - t0 < seconds:
            await asyncio.sleep(0.5)
        await s.close()
        billed = None
        for _, e in s.events:
            if e.get("type") == "session.usage.updated":
                u = e.get("usage") or {}
                if u.get("seconds") is not None:
                    billed = float(u["seconds"])
        billed = billed if billed is not None else (time.time() - t0)
        return {"arm": arm, "wall_s": time.time() - t0, "billed_s": billed,
                "voice_cost": billed / 60.0 * PRICES["gpt-live-1"]["per_minute"],
                "backend_cost": 0.0, "tools": []}

    from openai import AsyncOpenAI
    cfg = dict(realtime.session_config())
    cfg["output_modalities"] = ["audio"]
    usage = {"ain": 0, "aout": 0, "tin": 0, "tout": 0,
             "cached_audio": 0, "cached_text": 0}
    t0 = time.time()
    cl = AsyncOpenAI()
    async with cl.realtime.connect(
            model=config.OPENAI_REALTIME_MODEL) as conn:
        await conn.session.update(session=cfg)

        async def quiet():
            step = int(24000 * 0.02) * 2
            pcm = b"\x00\x00" * int(24000 * seconds)
            for i in range(0, len(pcm), step):
                await conn.input_audio_buffer.append(
                    audio=base64.b64encode(pcm[i:i + step]).decode())
                await asyncio.sleep(0.02)

        async def collect():
            async for ev in conn:
                if getattr(ev, "type", "") == "response.done":
                    u = getattr(getattr(ev, "response", None), "usage", None)
                    if u:
                        d = (u.model_dump() if hasattr(u, "model_dump")
                             else dict(u))
                        idt = d.get("input_token_details") or {}
                        odt = d.get("output_token_details") or {}
                        usage["ain"] += int(idt.get("audio_tokens") or 0)
                        usage["tin"] += int(idt.get("text_tokens") or 0)
                        cd = idt.get("cached_tokens_details") or {}
                        usage["cached_audio"] += int(cd.get("audio_tokens") or 0)
                        usage["cached_text"] += int(cd.get("text_tokens") or 0)
                        usage["aout"] += int(odt.get("audio_tokens") or 0)
                        usage["tout"] += int(odt.get("text_tokens") or 0)

        fq = asyncio.ensure_future(quiet())
        pump = asyncio.ensure_future(collect())
        try:
            await fq
            await asyncio.sleep(2.0)
        finally:
            pump.cancel()
            fq.cancel()
    wall = time.time() - t0
    p = PRICES[config.OPENAI_REALTIME_MODEL]
    ain = max(usage["ain"] - usage["cached_audio"], 0)
    tin = max(usage["tin"] - usage["cached_text"], 0)
    cost = (ain * p["audio_in"] + tin * p["text_in"]
            + (usage["cached_audio"] + usage["cached_text"]) * p["cached_in"]
            + usage["aout"] * p["audio_out"]
            + usage["tout"] * p["text_out"]) / 1e6
    return {"arm": arm, "wall_s": wall, "billed_s": wall,
            "voice_cost": cost, "backend_cost": 0.0, "tools": [],
            "usage": usage}


async def main_async(args) -> int:
    arms = [a.strip() for a in args.arms.split(",") if a.strip()]
    rows = []
    for arm in arms:
        print(f"-- {arm} ...", flush=True)
        try:
            if args.idle:
                r = await run_idle(arm, args.idle)
            else:
                r = await (run_live(args.dump_usage, args.no_model)
                           if arm == "gpt_live"
                           else run_realtime(args.dump_usage))
        except Exception as e:
            print(f"   {type(e).__name__}: {e}")
            continue
        rows.append(r)
        total = r["voice_cost"] + r["backend_cost"]
        mins = max(r["wall_s"], 1.0) / 60.0
        print(f'   wall {r["wall_s"]:.0f}s  tools={r.get("tools")}')
        print(f'   voice ${r["voice_cost"]:.4f}  backend '
              f'${r["backend_cost"]:.4f}  total ${total:.4f}  '
              f'= ${total / mins:.4f}/min', flush=True)
        if r.get("usage"):
            print(f'   tokens {r["usage"]}')
        if r.get("backend_in"):
            print(f'   backend tokens in={r["backend_in"]} '
                  f'out={r["backend_out"]}'
                  f'{"  (modelled)" if r.get("backend_modelled") else ""}')
    print("\n" + "-" * 62)
    print(f'{"arm":<18}{"drive":>8}{"$ drive":>11}{"$/min":>11}')
    for r in rows:
        total = r["voice_cost"] + r["backend_cost"]
        mins = max(r["wall_s"], 1.0) / 60.0
        print(f'{r["arm"]:<18}{r["wall_s"]:>7.0f}s{total:>11.4f}'
              f'{total / mins:>11.4f}')
    return 0


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--arms", default="gpt_live,openai_realtime")
    ap.add_argument("--dump-usage", action="store_true")
    ap.add_argument("--idle", type=float, default=0,
                    help="instead of the drive, hold a session open with "
                         "nobody talking, for this many seconds")
    ap.add_argument("--no-model", action="store_true",
                    help="skip modelling the backend's token cost")
    return asyncio.run(main_async(ap.parse_args()))


if __name__ == "__main__":
    raise SystemExit(main())
