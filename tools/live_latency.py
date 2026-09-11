"""Speech-end to first audio: gpt-live-1 against gpt-realtime-2.1.

    python -m tools.live_latency
    python -m tools.live_latency --n 5
    python -m tools.live_latency --arms gpt_live

THE ONE NUMBER A DRIVER ACTUALLY FEELS. Everything else in a voice stack --
routing accuracy, token cost, benchmark scores -- is invisible from the
passenger seat. This is not: it is the silence between finishing a question
and hearing the first syllable of the answer, and it is the thing that makes
an assistant feel present or makes it feel like a form being submitted.

MEASURED FROM THE SAME PLACE ON BOTH ARMS, which is the only reason the two
numbers can be compared at all:

  speech end   the wall clock when the LAST SAMPLE OF THE DRIVER'S QUESTION
               was handed to the transport. Not when the detector noticed, not
               when the turn committed -- those are the things under test and
               using either as the origin would hide exactly the difference
               this is looking for.
  first audio  the wall clock when the first frame of HER answer arrives with
               anything in it. On the live arm that is an RTP frame off the
               WebRTC track; on the realtime arm it is the first
               response.output_audio.delta. Both are "the speaker could start
               now".

The two arms differ in transport -- WebRTC for live because the API takes
nothing else, a WebSocket for realtime because that is what it takes -- and
that is a real confound worth naming rather than hiding. WebRTC adds a
negotiation before the session and nothing per-turn; the per-turn path is a
media frame either way.

THE QUESTIONS ARE DELIBERATELY TOOL-FREE. A turn that reaches the camera
measures the camera, and on the live arm it also measures a delegation to a
second model. Both of those are real and neither is this number; the tool
latency is tools/live_backend_bench.py, and the whole-turn budget is
tools/live_tool_turns.py.
"""
import argparse
import asyncio
import base64
import statistics
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from dotenv import load_dotenv                              # noqa: E402

load_dotenv(Path(__file__).resolve().parent.parent / ".env")

import config                                               # noqa: E402
import live                                                 # noqa: E402
import realtime                                             # noqa: E402
from live_harness import (LiveSession, ScriptedMic, RATE,   # noqa: E402
                          say_as_driver, silence)

QUESTIONS = [
    "Hey, what's the traffic looking like up ahead?",
    "How long have you been driving with me?",
    "What's your favourite kind of road?",
]

# After the question, this much quiet, so the turn has something to end on.
# The footnote in config.py's turn-end table is about what happens without it:
# a detector observes silence in the audio it is given and cannot observe an
# absence of audio, so a run that simply stops sending measures nothing.
TAIL_MS = 2500


async def live_turn(question: str) -> float:
    """One question into a gpt-live-1 session. Returns ms, or None."""
    q = say_as_driver(question)
    mic = ScriptedMic(np.concatenate([silence(400), q, silence(TAIL_MS)]))
    end_idx = int(RATE * 0.4) + len(q)
    mic.mark_at("speech_end", end_idx)
    s = LiveSession(mic)
    await s.open()
    t0 = time.time()
    # SHE GREETS. The first sound on the track is "Hey. What's up." before the
    # driver has said anything, so the turn is timed from the first onset that
    # follows the question -- and a turn where she was still mid-greeting when
    # the question ended is thrown away rather than reported as fast.
    while time.time() - t0 < 35:
        await asyncio.sleep(0.05)
        end = mic.marks.get("speech_end")
        if end and s.onset_after(end):
            break
    await s.close()
    end = mic.marks.get("speech_end")
    if not end:
        return None
    onset = s.onset_after(end)
    if not onset:
        return None
    return (onset - end) * 1000.0


async def realtime_turn(question: str) -> float:
    """The same question into gpt-realtime-2.1, the backend this replaces."""
    from openai import AsyncOpenAI

    pcm24 = _as_24k(say_as_driver(question))
    cfg = dict(realtime.session_config())
    cfg["output_modalities"] = ["audio"]
    cfg["tools"] = []
    cfg["tool_choice"] = "none"
    first = None
    speech_end = None
    cl = AsyncOpenAI()
    async with cl.realtime.connect(model=config.OPENAI_REALTIME_MODEL) as conn:
        await conn.session.update(session=cfg)

        async def feed():
            nonlocal speech_end
            step = int(24000 * 0.02) * 2                    # 20 ms of 24 kHz
            payload = pcm24 + b"\x00\x00" * int(24000 * TAIL_MS / 1000)
            speech_bytes = len(pcm24)
            for i in range(0, len(payload), step):
                await conn.input_audio_buffer.append(
                    audio=base64.b64encode(payload[i:i + step]).decode())
                if speech_end is None and i + step >= speech_bytes:
                    speech_end = time.time()
                await asyncio.sleep(0.02)

        task = asyncio.ensure_future(feed())
        t0 = time.time()
        try:
            async for ev in conn:
                if getattr(ev, "type", "") == "response.output_audio.delta":
                    first = time.time()
                    break
                if time.time() - t0 > 30:
                    break
        finally:
            task.cancel()
    if not first or not speech_end:
        return None
    return (first - speech_end) * 1000.0


def _as_24k(a48: np.ndarray) -> bytes:
    """48 kHz mono int16 down to the 24 kHz the realtime socket wants."""
    return a48[::2].astype("<i2").tobytes()


def report(name: str, xs: list):
    xs = [x for x in xs if x]
    if not xs:
        print(f"  {name:<16} no measurements")
        return
    p95 = (statistics.quantiles(xs, n=20)[18] if len(xs) >= 20 else max(xs))
    print(f"  {name:<16} n={len(xs):<3} p50 {statistics.median(xs):7.0f} ms   "
          f"p95 {p95:7.0f} ms   min {min(xs):6.0f}   max {max(xs):6.0f}")


async def main_async(args) -> int:
    arms = [a.strip() for a in args.arms.split(",") if a.strip()]
    out = {}
    for arm in arms:
        xs = []
        for q in QUESTIONS:
            for _ in range(args.n):
                try:
                    ms = await (live_turn(q) if arm == "gpt_live"
                                else realtime_turn(q))
                except Exception as e:
                    print(f"    {arm}: {type(e).__name__}: {e}", flush=True)
                    ms = None
                if ms:
                    print(f"    {arm:<10} {ms:7.0f} ms  {q[:40]!r}", flush=True)
                xs.append(ms)
        out[arm] = xs
    print("\nspeech end -> first audio")
    for arm, xs in out.items():
        report(arm, xs)
    a, b = out.get("gpt_live"), out.get("openai_realtime")
    if a and b:
        a = [x for x in a if x]
        b = [x for x in b if x]
        if a and b:
            d = statistics.median(a) - statistics.median(b)
            print(f"\n  gpt-live-1 is {abs(d):.0f} ms "
                  f"{'slower' if d > 0 else 'faster'} at the median than "
                  f"{config.OPENAI_REALTIME_MODEL}")
    return 0


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=3)
    ap.add_argument("--arms", default="gpt_live,openai_realtime")
    return asyncio.run(main_async(ap.parse_args()))


if __name__ == "__main__":
    raise SystemExit(main())
