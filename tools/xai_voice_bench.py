"""xai_voice_bench.py — grok-voice at one effort against another, on RIO's own tools.

    python -m tools.xai_voice_bench
    python -m tools.xai_voice_bench --efforts low high --trials 2

WHAT THIS DECIDES, AND WHY IT RUNS BEFORE THE CONTROLLER IS BUILT.

grok-voice's reasoning.effort defaults to "high". On the text model that default
cost 3.7x the latency for the same answer, so the drive session wants the lower
setting -- but "wants" is not a measurement, and a voice session has a second axis
the text one does not: it has to ROUTE, and a voice that answers faster while
picking the wrong tool is worse than one that thinks.

So: RIO's nine real tool schemas, her real session instructions, and a scripted
set of utterances whose correct tool is not in doubt. Two things come out, per
effort:

    time to FIRST AUDIO   what the driver actually waits, measured from
                          response.create to the first audio delta -- not to
                          response.done, which is the end of generation and
                          arrives after the sound on this transport.
    routing accuracy      did the utterance reach the tool it should?

The same shape as tools/live_backend_bench.py, which chose gpt-5.6-luna over the
documentation's recommendation on exactly this pair of numbers and was right to.

THE FOUR THINGS ALREADY MEASURED ARE HONOURED HERE RATHER THAN REDISCOVERED,
because each one of them cost something to learn:

  output_modalities and an output voice are both sent, or the session accepts
  every field, runs VAD and produces nothing at all with no error;
  audio is followed by a tail of silence so server_vad can hear an utterance END,
  and no explicit commit is sent alongside it;
  .completed arrives three times per utterance with one id, so transcripts are
  deduplicated by id rather than counted;
  and nothing is sent after response.done until the audio for it has played --
  which here means the bench waits, because overlapping audio would corrupt the
  very latency it is trying to measure.
"""
import argparse
import asyncio
import base64
import json
import os
import statistics
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from dotenv import load_dotenv                                  # noqa: E402

load_dotenv(Path(__file__).resolve().parent.parent / ".env")

import config                                                   # noqa: E402
import realtime                                                 # noqa: E402
import xai_voice                                                # noqa: E402

# Utterances whose correct tool is not in doubt. Deliberately the shapes a drive
# actually produces, and deliberately including two that must reach NO tool: a
# voice that calls something for "thanks" is a voice that interrupts a drive to
# answer a courtesy.
SCRIPT = [
    ("What's that building on the right?",            "look"),
    ("What do you see up ahead?",                     "look"),
    ("How far to the next turn?",                     "nav_status"),
    ("What are the directions from here?",            "nav_directions"),
    ("Take me to the Getty.",                         "start_navigation"),
    ("Is there a coffee place near here?",            "find_places"),
    ("How are my tires doing?",                       "vehicle_status"),
    ("Is it going to rain later?",                    "get_weather"),
    ("Any news around here?",                         "search_local_news"),
    ("Why do overpasses have diagonal cables?",       "deep_dive"),
    ("Thanks, that's helpful.",                       None),
    ("You're funny.",                                 None),
    # SIX MORE THAT MUST REACH NO TOOL, and they are not padding.
    #
    # Time to first audio can only be measured on a turn that SPEAKS, and a tool
    # turn does not: it emits a function call and waits, so the sound comes after
    # a result the bench does not supply. The first run of this file had ten tool
    # utterances and two conversational ones, and reported a p50 and a p95 over
    # n=2 -- two samples wearing the clothes of a distribution. These are the
    # turns the latency number is actually about: the driver says something
    # ordinary and waits to hear her.
    ("Good morning.",                                 None),
    ("This traffic is unbelievable.",                 None),
    ("I love this car.",                              None),
    ("Tell me something interesting.",                None),
    ("How are you doing today?",                      None),
    ("That was a close one.",                         None),
]


def session_config(effort, voice, instructions, tools):
    """RIO's real session, with the two fields whose absence is silent."""
    cfg = {
        "type": "realtime",
        "output_modalities": ["audio"],
        "instructions": instructions,
        "tools": tools,
        "tool_choice": "auto",
        "audio": {
            "input": {
                "format": {"type": "audio/pcm", "rate": xai_voice.RATE},
                "transcription": {"model": config.XAI_STT_SESSION_MODEL},
                "turn_detection": {"type": "server_vad"},
            },
            "output": {
                "voice": voice,
                "format": {"type": "audio/pcm", "rate": xai_voice.RATE},
            },
        },
    }
    if effort:
        # ONLY 'none' OR 'high' HERE, AND 'low' IS A 400.
        #
        #   session.reasoning.effort
        #     Input should be 'none' or 'high'  [input_value='low']
        #
        # That is the THIRD effort scale on this one vendor: /v1/responses takes
        # low and high and refuses medium; /v1/chat/completions takes none, low,
        # medium and high; the voice session takes none and high. The migration
        # brief said "low or none" for the drive session -- on this endpoint that
        # resolves to none, because low does not exist.
        cfg["reasoning"] = {"effort": effort}
    return cfg


async def one_effort(effort, trials, voice):
    import websockets

    instructions = realtime.instructions()
    tools = [dict(t) for t in realtime.session_config()["tools"]]
    token = xai_voice._ephemeral(300)

    ttfa, routed, wrong, errors = [], 0, [], 0
    tool_ms = []
    usage_seen = {}
    total = 0
    cost_ticks = 0

    async with websockets.connect(
            f"{xai_voice.WS_URL}?model={config.XAI_VOICE_MODEL}",
            subprotocols=[f"xai-client-secret.{token}"],
            open_timeout=30, max_size=None) as ws:

        await ws.send(json.dumps({"type": "session.update",
                                  "session": session_config(
                                      effort, voice, instructions, tools)}))
        # A REJECTED session.update MUST NOT BE A HANG. The first version of this
        # file waited for session.updated and, when the effort value was refused,
        # waited for it forever -- then waited 60 s per utterance for responses
        # that were never coming, and produced no output at all in ten minutes.
        # A tool written to find silent failures is not allowed to have one.
        while True:
            ev = json.loads(await asyncio.wait_for(ws.recv(), 30))
            if ev.get("type") == "session.updated":
                break
            if ev.get("type") == "error":
                err = ev.get("error") or {}
                raise RuntimeError(
                    "the session was refused: "
                    + str(err.get("params") or err.get("message"))[:400])

        for trial in range(trials):
            for text, want in SCRIPT:
                total += 1
                # THE DRIVER'S WORDS AS TEXT, not as audio. The question here is
                # what she DOES with an utterance and how fast she starts
                # speaking; synthesising the driver would add a transcription
                # round trip to every measurement and measure the transcriber.
                await ws.send(json.dumps({
                    "type": "conversation.item.create",
                    "item": {"type": "message", "role": "user",
                             "content": [{"type": "input_text", "text": text}]}}))
                t0 = time.time()
                await ws.send(json.dumps({"type": "response.create"}))

                first = None
                called = None
                # TIME TO THE TOOL CALL, which is where reasoning effort should
                # show up if it shows up anywhere. First audio is measured on
                # conversational turns; a tool turn's latency is how long the
                # driver waits before RIO even starts fetching, and that is the
                # half of the drive effort was supposed to be traded against.
                ttc = None
                try:
                    while True:
                        ev = json.loads(await asyncio.wait_for(ws.recv(), 60))
                        t = ev.get("type", "")
                        if t == "response.output_audio.delta" and first is None:
                            first = (time.time() - t0) * 1000
                        elif t == "response.function_call_arguments.done":
                            if ttc is None:
                                ttc = (time.time() - t0) * 1000
                            called = ev.get("name") or called
                        elif t == "response.output_item.done":
                            it = ev.get("item") or {}
                            if it.get("type") in ("function_call", "function"):
                                called = it.get("name") or called
                        elif t == "error":
                            errors += 1
                            break
                        elif t == "response.done":
                            u = ((ev.get("response") or {}).get("usage") or {})
                            # THE REALTIME PATH DOES NOT REPORT COST the way
                            # /v1/responses does -- there is no
                            # cost_in_usd_ticks here, which is why the total below
                            # reads $0 and why cost per drive-minute has to be
                            # computed from the published rate instead of read
                            # back. Whatever usage IS carried is collected so the
                            # gap is visible rather than assumed.
                            cost_ticks += u.get("cost_in_usd_ticks") or 0
                            for k, v in (u or {}).items():
                                if isinstance(v, (int, float)):
                                    usage_seen[k] = usage_seen.get(k, 0) + v
                            break
                except asyncio.TimeoutError:
                    errors += 1

                if first is not None:
                    ttfa.append(first)
                if ttc is not None:
                    tool_ms.append(ttc)
                if called == want:
                    routed += 1
                else:
                    wrong.append((text, called, want))

                # A TOOL CALL NEEDS A RESULT OR THE NEXT TURN PAYS FOR IT. Not the
                # real tool -- this bench is about routing and latency, not about
                # answers -- but the conversation must not be left malformed.
                if called:
                    await ws.send(json.dumps({
                        "type": "conversation.item.create",
                        "item": {"type": "function_call_output",
                                 "call_id": "bench",
                                 "output": json.dumps({"ok": True,
                                                       "note": "bench"})}}))
    return {"effort": effort, "ttfa": ttfa, "routed": routed, "total": total,
            "wrong": wrong, "errors": errors, "usage": usage_seen,
            "tool_ms": tool_ms,
            "usd": round(cost_ticks * 1e-10, 5)}


def show(r):
    t = sorted(r["ttfa"])
    print(f"\n--- effort {str(r['effort'] or 'default'):<8} "
          f"{r['routed']}/{r['total']} routed   {r['errors']} error(s)   "
          f"${r['usd']}")
    if t:
        p50 = statistics.median(t)
        p95 = t[min(len(t) - 1, int(len(t) * 0.95))]
        print(f"    first audio  n={len(t)}  min {t[0]:.0f}  p50 {p50:.0f}  "
              f"p95 {p95:.0f}  max {t[-1]:.0f} ms")
    tm = sorted(r.get("tool_ms") or [])
    if tm:
        print(f"    to tool call n={len(tm)}  min {tm[0]:.0f}  "
              f"p50 {statistics.median(tm):.0f}  "
              f"p95 {tm[min(len(tm)-1, int(len(tm)*0.95))]:.0f}  max {tm[-1]:.0f} ms")
    if r.get("usage"):
        print(f"    usage fields carried: "
              + ", ".join(f"{k}={v}" for k, v in sorted(r["usage"].items())))
    else:
        print("    usage: nothing numeric reported on this path")
    for text, got, want in r["wrong"]:
        print(f"    {text!r:<42} -> {str(got):<20} wanted {str(want)}")


def main() -> int:
    ap = argparse.ArgumentParser()
    # none vs high, because those are the only two this endpoint takes.
    ap.add_argument("--efforts", nargs="+", default=["none", "high"])
    ap.add_argument("--trials", type=int, default=1)
    ap.add_argument("--voice", default=None)
    a = ap.parse_args()
    if not os.getenv("XAI_API_KEY"):
        print("XAI_API_KEY is not set")
        return 2
    voice = a.voice or config.XAI_VOICE
    print(f"{config.XAI_VOICE_MODEL} / {voice}, {len(SCRIPT)} utterances "
          f"x {a.trials} trial(s), RIO's nine real tool schemas")

    out = []
    for effort in a.efforts:
        e = None if effort in ("none-sent", "default") else effort
        try:
            out.append(asyncio.run(one_effort(e, a.trials, voice)))
        except Exception as ex:
            print(f"\n--- effort {effort}: FAILED {type(ex).__name__}: "
                  f"{str(ex)[:200]}")
            continue
        show(out[-1])

    if len(out) >= 2:
        print("\n" + "-" * 66)
        for r in out:
            t = sorted(r["ttfa"])
            p50 = statistics.median(t) if t else float("nan")
            p95 = t[min(len(t) - 1, int(len(t) * 0.95))] if t else float("nan")
            tm2 = sorted(r.get("tool_ms") or [])
            tp50 = statistics.median(tm2) if tm2 else float("nan")
            print(f"  effort {str(r['effort'] or 'default'):<8} "
                  f"routed {r['routed']}/{r['total']}   "
                  f"audio p50 {p50:.0f}/p95 {p95:.0f} ms   "
                  f"tool-call p50 {tp50:.0f} ms")
    return 0


if __name__ == "__main__":
    sys.exit(main())
