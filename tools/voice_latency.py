"""voice_latency.py — how long the driver waits, in each of the three voices.

    python -m tools.voice_latency                    # 5 turns per path
    python -m tools.voice_latency --turns 8
    python -m tools.voice_latency --only elevenlabs_v3

WHAT IS BEING MEASURED, AND WHY THAT AND NOT SOMETHING EASIER
-------------------------------------------------------------
Speech-end to first audio. The instant the driver stops talking, to the instant
the first sample of RIO's reply is produced. That is the silence a person
actually sits in, and it is the only number in this system a driver can feel.

It is deliberately NOT "how fast does the synthesiser return a byte". That
question has a flattering answer for every path here and tells you nothing:
under the ElevenLabs backend the synthesiser cannot start until the model has
written a clause, so a measurement that begins when the text is ready has
already skipped the part that got slower. The clock starts where the driver's
sentence ends, and everything after it — the model deciding to answer, writing
the first clause, the chunker deciding that clause is worth speaking, the
socket, the synthesis — is inside the number.

HOW THE DRIVER IS SIMULATED
---------------------------
A real spoken question, synthesised once and fed into the session's input at
real time, so the server's own voice-activity detector decides when the turn
ended exactly as it does in a car. Dumping the audio in as fast as the socket
takes it would produce a `speech_stopped` at a moment no driver ever produces,
and the whole measurement hangs off that event.

The same question, the same instructions and the same VAD settings across all
three paths, so the only thing varying is the mouth.

THE PATHS
---------
  cedar             the live session speaks for itself. One model, one hop.
  elevenlabs_v2     text mode, phrase-chunked into the multi-context
                    text-to-speech socket. The current conversation path.
  elevenlabs_v3     the same, over the Text-to-Dialogue socket. A different
                    model AND a different transport, which is why it is worth
                    keeping a column rather than a memory of one.
  elevenlabs_flash  text mode, straight to flash over HTTP. What a
                    per-utterance fallback sounds like, and having its number
                    next to the others is what makes "fall back to flash" a
                    decision rather than a hope.

AND THE OTHER QUESTION
----------------------
`--deterministic` measures the OTHER path — the one warnings, turns and health
announcements take, which is an ordinary HTTP stream and not a socket. It is a
different question with a different answer: nothing there is phrase-chunked,
nothing streams into a conversation, and the only number that matters is time
to first byte. It exists to answer "could the deterministic path use the
conversation model too", which is a question about one voice everywhere and is
worth re-asking whenever the conversation model changes.
"""
import argparse
import asyncio
import base64
import os
import statistics
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from dotenv import load_dotenv                              # noqa: E402

load_dotenv(Path(__file__).resolve().parent.parent / ".env")

import config                                               # noqa: E402
import realtime                                             # noqa: E402
import voice_dialogue                                       # noqa: E402

# What the driver says. Short, ordinary, and the kind of thing that gets a
# short answer — a question whose reply is a paragraph would measure the
# model's verbosity as much as anything else.
QUESTION = "Hey, what's the traffic looking like up ahead?"

# 24 kHz mono PCM in both directions: what the realtime session wants on input
# and what the dialogue socket produces on output.
RATE = 24000
FRAME_MS = 20


def driver_audio_for(question: str) -> bytes:
    """One spoken question, as raw PCM, synthesised once and cached on disk.

    Cached because the recording is the CONSTANT in every experiment that uses
    it. Re-synthesising per run puts a different waveform in front of the
    detector each time, which moves the turn-end mark every measurement hangs
    off and quietly makes two runs incomparable.

    Shared with tools/visual_latency.py, which asks its own questions and needs
    them to behave the same way.
    """
    import hashlib

    import httpx

    slug = hashlib.sha1(question.encode()).hexdigest()[:12]
    cache = Path(__file__).resolve().parent / f"_driver_{slug}.pcm"
    if cache.exists() and cache.stat().st_size > 1000:
        return cache.read_bytes()
    r = httpx.post(
        voice_dialogue.FLASH_URL.format(voice=voice_dialogue.voice_id()),
        params={"output_format": f"pcm_{RATE}"},
        headers={"xi-api-key": voice_dialogue.api_key()},
        json={"text": question,
              "model_id": config.ELEVENLABS_DETERMINISTIC_MODEL},
        timeout=60.0)
    r.raise_for_status()
    cache.write_bytes(r.content)
    return r.content


def driver_audio() -> bytes:
    return driver_audio_for(QUESTION)


def _silence(ms: int) -> bytes:
    return b"\x00\x00" * int(RATE * ms / 1000)


async def _feed(conn, pcm: bytes):
    """The driver talking, at the speed a driver talks.

    Real time, frame by frame. The trailing silence is part of the recording
    and not an afterthought: it is what the server's silence window is measured
    against, and without it the turn never ends.
    """
    payload = pcm + _silence(int(config.REALTIME_VAD_SILENCE_MS) + 400)
    step = int(RATE * FRAME_MS / 1000) * 2
    for i in range(0, len(payload), step):
        await conn.input_audio_buffer.append(
            audio=base64.b64encode(payload[i:i + step]).decode("ascii"))
        await asyncio.sleep(FRAME_MS / 1000.0)


def _session(text_out: bool) -> dict:
    """The same session as the car's, with the modality under test."""
    cfg = dict(realtime.session_config())
    cfg["output_modalities"] = ["text"] if text_out else ["audio"]
    # No tools. A turn that reaches for the camera is measuring the camera.
    cfg["tools"] = []
    cfg["tool_choice"] = "none"
    return cfg


# A turn that produces nothing must still END. The realtime stream is an async
# iterator, so a deadline checked inside the loop body is only checked when an
# event arrives — and the interesting failure here is a session that stops
# sending events at all (the account's per-minute token cap arrives as one
# `response.done` and then silence). Without a timeout around the whole thing
# the tool hangs, which is the least useful thing a measurement can do.
TURN_TIMEOUT_S = 45.0


async def _one_turn(path: str) -> dict:
    """One question, one answer, and the millisecond it started arriving."""
    try:
        return await asyncio.wait_for(_run_turn(path), timeout=TURN_TIMEOUT_S)
    except asyncio.TimeoutError:
        return {"path": path, "ms": None, "first_text_ms": None,
                "note": f"no answer inside {TURN_TIMEOUT_S:.0f}s "
                        "(usually the realtime token cap)"}


async def _run_turn(path: str) -> dict:
    from openai import AsyncOpenAI

    text_out = path != "cedar"
    model_for = {
        "elevenlabs_v2": config.ELEVENLABS_CONVERSATION_MODEL,
        "elevenlabs_v3": "eleven_v3_conversational",
        "elevenlabs_flash": config.ELEVENLABS_DETERMINISTIC_MODEL,
    }
    result = {"path": path, "ms": None, "note": None, "first_text_ms": None}
    marks = {"stopped": None, "first_text": None, "first_audio": None}
    sink_done = asyncio.Event()

    async def on_audio(rid, pcm, text):
        if marks["first_audio"] is None and pcm:
            marks["first_audio"] = time.time()
            sink_done.set()

    async def on_event(kind, detail):
        if kind == "fallback" and detail.get("tier") == "flash":
            result["note"] = f"fell back: {detail.get('cause')}"

    session = None
    if text_out:
        # The flash column is the fallback path, exercised by making the
        # dialogue socket's budget unreachable rather than by calling a
        # different function: what is measured is the code that actually runs
        # in a car when v3 is slow, including the moment spent waiting for it.
        model = model_for.get(path, config.ELEVENLABS_CONVERSATION_MODEL)
        # `force_flash` is straight to the fallback model with no dialogue
        # socket in the path at all — the tier-1 destination, timed on its own
        # rather than timed through a failure.
        session = voice_dialogue.DialogueSession(
            on_audio=on_audio, on_event=on_event, model=model,
            force_flash=(path == "elevenlabs_flash"))
        await session.start()

    client = AsyncOpenAI()
    rid = {"id": "turn"}
    try:
        async with client.realtime.connect(
                model=config.OPENAI_REALTIME_MODEL) as conn:
            await conn.session.update(session=_session(text_out))
            feeder = asyncio.create_task(_feed(conn, driver_audio()))
            async for event in conn:
                t = event.type
                if t == "input_audio_buffer.speech_stopped":
                    marks["stopped"] = time.time()
                elif t == "response.created" and session:
                    rid["id"] = event.response.id
                    await session.begin(rid["id"])
                elif t == "response.output_text.delta":
                    if marks["first_text"] is None:
                        marks["first_text"] = time.time()
                    if session:
                        await session.delta(rid["id"], event.delta or "")
                elif t == "response.output_audio.delta":
                    if marks["first_audio"] is None:
                        marks["first_audio"] = time.time()
                    break
                elif t == "response.done":
                    if session:
                        await session.end(rid["id"])
                    if not text_out:
                        break
                elif t == "error":
                    result["note"] = str(getattr(event, "error", "error"))[:120]
                    break
                if marks["first_audio"]:
                    break
            feeder.cancel()
            if text_out and marks["first_audio"] is None:
                try:
                    await asyncio.wait_for(sink_done.wait(), timeout=10)
                except asyncio.TimeoutError:
                    result["note"] = result["note"] or "no audio"
    except Exception as e:
        result["note"] = f"{type(e).__name__}: {str(e)[:120]}"
    finally:
        if session:
            await session.close()

    if marks["stopped"] and marks["first_audio"]:
        result["ms"] = (marks["first_audio"] - marks["stopped"]) * 1000.0
    if marks["stopped"] and marks["first_text"]:
        result["first_text_ms"] = (marks["first_text"] - marks["stopped"]) * 1000.0
    return result


def _pct(values, p):
    if not values:
        return None
    v = sorted(values)
    i = min(len(v) - 1, max(0, int(round((p / 100.0) * (len(v) - 1)))))
    return v[i]


async def run(paths, turns: int) -> dict:
    out = {}
    for path in paths:
        got, notes, texts, synth = [], [], [], []
        print(f"\n  {path}")
        for n in range(turns):
            r = await _one_turn(path)
            if r["ms"]:
                got.append(r["ms"])
                if r["first_text_ms"]:
                    texts.append(r["first_text_ms"])
                    synth.append(r["ms"] - r["first_text_ms"])
                print(f"    turn {n + 1}: {r['ms']:6.0f} ms"
                      + (f"   (model wrote its first word at "
                         f"{r['first_text_ms']:.0f} ms)" if r["first_text_ms"]
                         else "")
                      + (f"   [{r['note']}]" if r["note"] else ""))
            else:
                notes.append(r["note"] or "no measurement")
                print(f"    turn {n + 1}:    ---   {r['note']}")
            # The account is on a per-minute rate limit and these are real
            # sessions. A beat between them costs nothing and keeps a long run
            # from measuring the rate limiter.
            await asyncio.sleep(1.0)
        out[path] = {"samples": [round(x) for x in got],
                     "p50": _pct(got, 50), "p95": _pct(got, 95),
                     "mean": statistics.mean(got) if got else None,
                     "first_text_p50": _pct(texts, 50),
                     # The half of the number this change is actually
                     # responsible for. The model's time-to-first-word varies
                     # by a second between turns and is the same model on every
                     # row here; subtracting it is what makes the three voices
                     # comparable rather than three samples of GPT's mood.
                     "synth_p50": _pct(synth, 50),
                     "synth_p95": _pct(synth, 95),
                     "notes": notes}
    return out


def report(results: dict):
    print("\n" + "=" * 72)
    print("SPEECH-END -> FIRST AUDIO   (the silence the driver sits in)")
    print("=" * 72)
    print(f"  {'path':20} {'n':>3} {'p50':>8} {'p95':>8}"
          f"   {'model 1st word':>15}  {'synthesis p50/p95':>18}")
    for path, r in results.items():
        n = len(r["samples"])
        p50 = f"{r['p50']:.0f} ms" if r["p50"] else "--"
        p95 = f"{r['p95']:.0f} ms" if r["p95"] else "--"
        ft = f"{r['first_text_p50']:.0f} ms" if r["first_text_p50"] else "--"
        sy = (f"{r['synth_p50']:.0f} / {r['synth_p95']:.0f} ms"
              if r["synth_p50"] else "-- (in the model)")
        print(f"  {path:20} {n:>3} {p50:>8} {p95:>8}   {ft:>15}  {sy:>18}")
        if r["notes"]:
            print(f"      {len(r['notes'])} turn(s) produced nothing: "
                  f"{sorted(set(r['notes']))}")
    base = results.get("cedar", {}).get("p50")
    v3 = results.get("elevenlabs_v3", {}).get("p50")
    if base and v3:
        print(f"\n  v3 conversational costs {v3 - base:+.0f} ms against cedar "
              f"at the median.")
    print("\n  The p95 column is mostly the MODEL: a turn where GPT took a "
          "second to write\n  its first word is a second late in every voice. "
          "The synthesis column is the\n  part this change owns.")
    print("\n  A pre-rendered clip is 0 ms and is unaffected by any of this: "
          "the red\n  headway tier and the two tire lines play local files "
          "with no network in\n  the path, in this same voice.")


# ---------------------------------------------------------------------------
# The deterministic path: a table lookup, an HTTP stream, and a first byte
# ---------------------------------------------------------------------------
DETERMINISTIC_LINES = [
    "Take the next left onto Lincoln Boulevard.",
    "Watch your distance.",
    "Your rear left tire is losing pressure.",
]


def report_deterministic(turns: int) -> int:
    """Time to first byte for one warning, per model, on the same voice.

    Not the same measurement as the conversation path and not comparable with
    it. There is no model writing, no chunker and no socket here — the line is
    already known, in full, before anything is called. What is being asked is
    the narrow question the warning path actually cares about: how long after
    the decision to speak does sound start.
    """
    import statistics

    import voice

    print("  the deterministic path — /nav/voice, /headway_voice, "
          "/vehicle/health/voice")
    print(f"  voice: {voice.voice_id()}\n")
    out = {}
    for model in (config.ELEVENLABS_DETERMINISTIC_MODEL,
                  config.ELEVENLABS_CONVERSATION_MODEL,
                  "eleven_turbo_v2_5"):
        got = []
        for i in range(turns):
            line = DETERMINISTIC_LINES[i % len(DETERMINISTIC_LINES)]
            t0 = time.time()
            try:
                stream = voice.synthesize_stream(line, model=model)
                next(iter(stream))
                got.append((time.time() - t0) * 1000)
            except Exception as e:
                print(f"    {model}: {type(e).__name__}: {str(e)[:90]}")
                break
            time.sleep(0.3)
        out[model] = got
        if got:
            print(f"    {model:26} p50 {_pct(got, 50):6.0f} ms   "
                  f"p95 {_pct(got, 95):6.0f} ms   {[round(x) for x in got]}")

    fast = out.get(config.ELEVENLABS_DETERMINISTIC_MODEL) or []
    conv = out.get(config.ELEVENLABS_CONVERSATION_MODEL) or []
    print()
    if fast and conv:
        d = _pct(conv, 50) - _pct(fast, 50)
        print(f"  the conversation model costs {d:+.0f} ms against "
              f"{config.ELEVENLABS_DETERMINISTIC_MODEL} on a warning.")
        print("  What that has to buy: a driver cannot hear the difference "
              "between two\n  readings of \"Take the next left\", and can "
              "certainly hear it arrive late.")
    return 0


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--turns", type=int, default=5)
    ap.add_argument("--only", default=None,
                    help="cedar | elevenlabs_v2 | elevenlabs_v3 | elevenlabs_flash")
    ap.add_argument("--deterministic", action="store_true",
                    help="measure the WARNING path instead: time to first byte "
                         "over HTTP, per model, on the same voice")
    args = ap.parse_args()

    if not os.getenv("OPENAI_API_KEY"):
        print("OPENAI_API_KEY not set (.env)", file=sys.stderr)
        return 2
    if not voice_dialogue.configured():
        print("ELEVENLABS_API_KEY / ELEVENLABS_VOICE_ID not set (.env)",
              file=sys.stderr)
        return 2

    if args.deterministic:
        return report_deterministic(args.turns)
    paths = ([args.only] if args.only
             else ["cedar", "elevenlabs_v2", "elevenlabs_v3",
                   "elevenlabs_flash"])
    print(f"  question : {QUESTION!r}")
    print(f"  voice    : {voice_dialogue.voice_id()}")
    print(f"  models   : {config.OPENAI_REALTIME_MODEL} / "
          f"{config.ELEVENLABS_CONVERSATION_MODEL} / "
          f"{config.ELEVENLABS_DETERMINISTIC_MODEL}")
    print(f"  transport: {type(voice_dialogue.dialect_for('v', config.ELEVENLABS_CONVERSATION_MODEL, 'pcm_24000')).__name__}")
    print(f"  chunker  : >={config.ELEVENLABS_CHUNK_MIN_TOKENS} words at a "
          f"clause boundary, or {config.ELEVENLABS_CHUNK_MAX_WAIT_MS} ms")
    results = asyncio.run(run(paths, args.turns))
    report(results)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
