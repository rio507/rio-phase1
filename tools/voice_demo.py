"""voice_demo.py — record a real drive, so somebody can just listen to it.

    python -m tools.voice_demo --out /tmp/rio_drive.wav

WHY THIS EXISTS ALONGSIDE THE MEASUREMENTS
------------------------------------------
Every number in tools/voice_latency.py and tools/visual_latency.py is about
WHEN a sound arrives. None of them is about what it sounds like, and the two
questions have come apart badly at least once already: v3 was measured, tested
and shipped, and was rendering a voice that was audibly not the one that had
been cloned. No test caught it, because "is this the right person" was not a
thing anything could assert.

So this produces the artifact instead of an assertion. It drives the REAL
path — a live session in text mode, the real tool over HTTP, the real dialogue
socket, the real decision about whether the observer's line is spoken as
written — and writes one WAV of the result, with the driver in a different
voice so a listener can tell who is talking.

WHAT IS IN IT, AND WHY THOSE
----------------------------
Everything a listener needs to check the claims that have been made in prose:

  a scene question       the observer's own line, spoken with no model in
                         between. The thing to listen for is whether it sounds
                         like a person noticing something or like a caption.
  an object question     the full visual turn, composed by her. Different path,
                         and it should not sound like a different system.
  a navigation call      deterministic, flash, same voice id. This is where
                         "one voice everywhere" is either true or obviously not.
  a safety clip          a local file rendered offline, played with no network.
                         Same voice again, and the most important line she has.

Nothing here is a mock. If it sounds wrong, it is wrong in the car.
"""
import argparse
import asyncio
import io
import json
import subprocess
import sys
import time
import wave
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from dotenv import load_dotenv                              # noqa: E402

load_dotenv(Path(__file__).resolve().parent.parent / ".env")

import config                                               # noqa: E402
import realtime                                             # noqa: E402
import voice_dialogue as vd                                 # noqa: E402

RATE = config.ELEVENLABS_SAMPLE_RATE

# The driver. A different voice on purpose: RIO is a female robot-assistant
# clone, and a recording where both halves of a conversation are the same
# person is a recording nobody can follow.
DRIVER_VOICE = "CwhRBWXzGAHq8TQ4Fs17"        # Roger — laid-back, casual

# What the drive is: both visual paths, and one turn deliberately broken in
# half so a listener hears what a hesitation actually costs.
#
# `gap_ms` splices silence into the middle of an utterance. It is there because
# an earlier recording opened with "Hey, what do you see?", she answered "Hey.
# What's up.", and this script stopped listening — which read as "the camera
# was never called". It was called, 2.4 seconds later, and the question was
# answered. The recording was wrong, not the car.
#
# Kept as a case rather than removed, because the thing worth hearing is the
# real behaviour: a greeting, then the answer.
SCRIPT = [
    {"say": "What do you see?"},
    {"say": "What's that car ahead?"},
    {"say": "Hey,|what do you see?", "gap_ms": 1200,
     "note": "a hesitation mid-question — the detector ends the turn on it"},
]


def silence(ms):
    return b"\x00\x00" * int(RATE * ms / 1000)


def mp3_to_pcm(data: bytes) -> bytes:
    """Whatever ffmpeg makes of it, at the rate everything else here uses."""
    p = subprocess.run(
        ["ffmpeg", "-hide_banner", "-loglevel", "error", "-i", "pipe:0",
         "-f", "s16le", "-acodec", "pcm_s16le", "-ac", "1", "-ar", str(RATE),
         "pipe:1"], input=data, capture_output=True)
    return p.stdout


def say_as_driver(text: str) -> bytes:
    import httpx

    r = httpx.post(vd.FLASH_URL.format(voice=DRIVER_VOICE),
                   params={"output_format": f"pcm_{RATE}"},
                   headers={"xi-api-key": vd.api_key()},
                   json={"text": text,
                         "model_id": config.ELEVENLABS_DETERMINISTIC_MODEL},
                   timeout=60.0)
    r.raise_for_status()
    return r.content


async def feed_question(conn, pcm: bytes):
    """The driver asking, at the speed a driver asks, so the detector ends the
    turn the way it does in a car."""
    import base64

    payload = pcm + silence(int(config.REALTIME_VAD_SILENCE_MS) + 400)
    step = int(RATE * 20 / 1000) * 2
    for i in range(0, len(payload), step):
        await conn.input_audio_buffer.append(
            audio=base64.b64encode(payload[i:i + step]).decode("ascii"))
        await asyncio.sleep(0.02)


async def run(base: str, out: Path, video: str):
    import httpx
    from openai import AsyncOpenAI

    from tools.visual_latency import HttpDrive

    track = bytearray()
    said = []
    heard = {"pcm": bytearray()}

    async def on_audio(rid, pcm, text):
        heard["pcm"] += pcm

    sink = vd.DialogueSession(on_audio=on_audio,
                              on_event=lambda k, d: asyncio.sleep(0))
    await sink.start()
    http = httpx.AsyncClient(timeout=90.0)
    client = AsyncOpenAI()
    cfg = realtime.session_config()

    with HttpDrive(base, None, video, 0.25) as drive:
        sid = drive.session_id
        print(f"  drive {sid[:8]} — frames going in")
        await http.post(f"{base}/realtime/session", params={"session_id": sid})
        await asyncio.sleep(4)          # let the observer write something

        async with client.realtime.connect(
                model=config.OPENAI_REALTIME_MODEL) as conn:
            await conn.session.update(session=cfg)
            for turn in SCRIPT:
                question = turn["say"].replace("|", " ")
                if "|" in turn["say"]:
                    head, tail = turn["say"].split("|", 1)
                    q_pcm = (say_as_driver(head) + silence(turn["gap_ms"])
                             + say_as_driver(tail))
                else:
                    q_pcm = say_as_driver(turn["say"])
                track += silence(500) + q_pcm + silence(250)
                heard["pcm"] = bytearray()
                rid = {"id": "t"}
                path = {"v": None}
                feeder = asyncio.create_task(feed_question(conn, q_pcm))
                t0 = time.perf_counter()
                async for ev in conn:
                    if ev.type == "response.created":
                        rid["id"] = ev.response.id
                        await sink.begin(rid["id"])
                    elif ev.type == "response.function_call_arguments.done":
                        r = (await http.post(
                            f"{base}/realtime/tool", params={"session_id": sid},
                            json={"name": ev.name,
                                  "arguments": json.loads(ev.arguments or "{}"),
                                  "spoken": question})).json()
                        path["v"] = r.get("path")
                        await conn.conversation.item.create(item={
                            "type": "function_call_output",
                            "call_id": ev.call_id, "output": json.dumps(r)})
                        if r.get("speak_directly") and r.get("speech"):
                            # THE NEW PATH, exactly as the panel takes it.
                            await conn.conversation.item.create(item={
                                "type": "message", "role": "assistant",
                                "content": [{"type": "output_text",
                                             "text": r["speech"]}]})
                            said.append((question, path["v"], r["speech"]))
                            await sink.begin(rid["id"] + ":d")
                            await sink.delta(rid["id"] + ":d", r["speech"])
                            await sink.end(rid["id"] + ":d")
                            break
                        await conn.response.create()
                    elif ev.type == "response.output_text.delta":
                        # The composing path. Without this the words are
                        # collected and never spoken, and the recording has a
                        # silence where the object answer should be — which is
                        # exactly what the first take of it had.
                        await sink.delta(rid["id"], ev.delta or "")
                    elif ev.type == "response.output_text.done":
                        said.append((question, path["v"], ev.text))
                    elif ev.type == "response.done":
                        await sink.end(rid["id"])
                        # KEEP LISTENING UNLESS THE CAMERA HAS ANSWERED.
                        #
                        # A turn split by a hesitation produces a greeting
                        # first — "Hey. What's up." — and the real question
                        # arrives as a second turn a beat later. Stopping at
                        # the first response is what made an earlier recording
                        # report that the camera was never called.
                        if path["v"] or "what" not in question.lower():
                            break
                    if time.perf_counter() - t0 > 75:
                        break
                feeder.cancel()
                # Let the tail of the answer arrive before the next question.
                # A recording that cuts her off mid-sentence is a recording of
                # this script, not of her.
                deadline = time.perf_counter() + 12
                while time.perf_counter() < deadline:
                    await asyncio.sleep(0.05)
                    utt = sink._utt
                    if heard["pcm"] and (utt is None or utt.done):
                        await asyncio.sleep(0.8)
                        break
                track += bytes(heard["pcm"])
                took = (time.perf_counter() - t0) * 1000
                mine = [t for t in said if t[0] == question]
                print(f"    {question!r}"
                      + (f"   ({turn['note']})" if turn.get("note") else ""))
                for _, p, text in mine:
                    print(f"      [{p or 'no tool'}] {text!r}")
                print(f"      ({took:.0f} ms, "
                      f"{len(heard['pcm']) / (RATE * 2):.1f}s of audio)")
                await asyncio.sleep(1.0)

        # ...and the two deterministic voices, so one-voice-everywhere is a
        # thing a listener can check rather than a claim in a document.
        route = (await http.post(f"{base}/nav/route", params={"session_id": sid},
                                 json={"lat": 34.0219, "lng": -118.4814,
                                       "destination": "Griffith Observatory"})
                 ).json()
        man = (route.get("maneuvers") or [{}])[0]
        nav = await http.get(f"{base}/nav/voice", params={
            "route_id": route.get("route_id"), "m": man.get("id"),
            "call": "primary"})
        if nav.status_code == 200 and nav.headers.get("content-type", "").startswith("audio"):
            track += silence(900) + mp3_to_pcm(nav.content)
            said.append(("(navigation, deterministic)", "nav_flash",
                         man.get("speech", {}).get("primary", "")))

        clip = Path(__file__).resolve().parent.parent / "static/audio/too_close.mp3"
        if clip.exists():
            track += silence(900) + mp3_to_pcm(clip.read_bytes())
            said.append(("(headway red tier, local file)", "clip",
                         "You're too close."))

    await sink.close()
    await http.aclose()

    track += silence(400)
    with wave.open(str(out), "wb") as w:
        w.setnchannels(1)
        w.setsampwidth(2)
        w.setframerate(RATE)
        w.writeframes(bytes(track))
    secs = len(track) / (RATE * 2)
    print(f"\n  wrote {out}  ({secs:.1f}s)")
    print("\n  what you are listening to:")
    for q, p, text in said:
        print(f"    {q}\n      [{p}] {text}")
    return out


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--base", default="http://127.0.0.1:8888")
    ap.add_argument("--video", default="/workspace/ufldv2/example.mp4")
    ap.add_argument("--out", default="/tmp/rio_drive.wav")
    args = ap.parse_args()
    asyncio.run(run(args.base, Path(args.out), args.video))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
