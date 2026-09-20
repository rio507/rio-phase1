"""xai_interruptible_probe.py — does interruptible:false actually refuse a barge-in?

    python -m tools.xai_interruptible_probe
    python -m tools.xai_interruptible_probe --audio static/audio/eve/back_off.mp3

THE QUESTION, AND WHY IT IS NOT A DOCUMENTATION QUESTION.

force_message takes interruptible:false, documented to drop caller audio while the
line plays. If it works, RIO gets something she has never had: a red-tier warning
that cannot be talked over. If it does not, and we ship the field believing it
does, the red tier's behaviour is a belief rather than a mechanism -- and the day
it matters is the day somebody is trying to say something while the car is telling
them to back off.

WHY IT WAS 'NOT YET MEASURED' UNTIL NOW. Testing it needs a barge-in to refuse,
which reads as needing two speakers in a room. It does not: it needs AUDIO ARRIVING
WHILE SHE TALKS, and audio is a file. A clip goes into the input buffer during
playback and the question becomes whether the session hears it.

    control    force_message with no interruptible field, audio injected during
               playback  ->  does a transcript arrive?
    treatment  the same line with interruptible:false, same audio, same timing
               ->  does a transcript arrive?

A DIFFERENCE IS THE FEATURE. THE SAME ANSWER TWICE MEANS THE FIELD DOES NOTHING
HERE, whatever the documentation says, and the red tier cannot be built on it.

WHAT THIS CANNOT TELL US, said plainly: whether a REAL driver's voice in a cabin
is dropped. The clip is Eve's own voice through a loopback with no room, no
distance and no engine, and server_vad's behaviour on that is not identical to its
behaviour on a person. What it can settle is whether the field changes anything at
all, which is the part that decides whether the red tier may depend on it.

The four already-measured facts are honoured: output_modalities and an output voice
are both sent, no explicit commit is ever sent, .completed is deduplicated by id,
and audio is followed by a silence tail so server_vad can hear an utterance end.
"""
import argparse
import asyncio
import base64
import json
import os
import subprocess
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
sys.path.insert(0, str(Path(__file__).resolve().parent))

import config                                                     # noqa: E402
import xai_voice                                                  # noqa: E402

RATE = 24000
FRAME_MS = 40
FRAME_BYTES = RATE * 2 * FRAME_MS // 1000
LINE = ("Back off now. You are far too close to the car in front and I need you "
        "to ease off the accelerator until there is a full two seconds of space.")


def pcm16(path: Path) -> bytes:
    """Whatever the file is, as mono PCM16 at RATE."""
    out = subprocess.run(
        ["ffmpeg", "-hide_banner", "-loglevel", "error", "-i", str(path),
         "-ac", "1", "-ar", str(RATE), "-f", "s16le", "-"],
        capture_output=True, check=True)
    return out.stdout


async def run(audio: bytes, interruptible: bool, quiet: bool) -> dict:
    import websockets

    key = os.environ.get("XAI_API_KEY")
    if not key:
        raise SystemExit("XAI_API_KEY is not set")

    got = {"audio_ms": 0.0, "transcript": None, "speech_started": False,
           "committed": False, "forced_transcript": None, "errors": [],
           "responses": 0, "injected_frames": 0}
    seen_items = set()

    async with websockets.connect(
            f"{xai_voice.WS_URL}?model={config.XAI_VOICE_MODEL}",
            additional_headers={"Authorization": f"Bearer {key}"},
            max_size=1 << 24) as ws:

        async def send(obj):
            await ws.send(json.dumps(obj))

        # THE POLICY, VERBATIM FROM THE SERVER. Both of the fields whose absence
        # is silent are in it.
        await send({"type": "session.update",
                    "session": xai_voice.session_policy()})

        injected = False
        first_audio_at = None
        t0 = time.time()
        deadline = t0 + 45

        async def inject():
            """The barge-in: a clip into the input buffer, then the silence tail
            so server_vad can hear the end of it. No explicit commit -- measured,
            one suppresses the transcript entirely under server_vad, which would
            fake exactly the result this probe is looking for."""
            for i in range(0, len(audio) - FRAME_BYTES, FRAME_BYTES):
                await send({"type": "input_audio_buffer.append",
                            "audio": base64.b64encode(
                                audio[i:i + FRAME_BYTES]).decode()})
                got["injected_frames"] += 1
                await asyncio.sleep(FRAME_MS / 1000)
            tail = b"\x00" * FRAME_BYTES
            for _ in range(int(config.XAI_SILENCE_TAIL_MS // FRAME_MS)):
                await send({"type": "input_audio_buffer.append",
                            "audio": base64.b64encode(tail).decode()})
                await asyncio.sleep(FRAME_MS / 1000)

        pending = None
        while time.time() < deadline:
            try:
                raw = await asyncio.wait_for(ws.recv(), timeout=2.0)
            except asyncio.TimeoutError:
                if got["audio_ms"] and injected and pending and pending.done():
                    break
                continue
            ev = json.loads(raw)
            t = ev.get("type")

            if t == "session.updated" and not injected:
                # THE LINE, SAID BY THE SESSION AND NOT BY A MODEL. The
                # content-array form: the `text` form produces "Hey. What's up."
                item = xai_voice.force_message_item(
                    LINE, interruptible=interruptible)
                await send({"type": "conversation.item.create", "item": item})

            elif t == "error":
                got["errors"].append(str(ev)[:300])

            elif t == "response.created":
                got["responses"] += 1

            elif t == "response.output_audio.delta" and ev.get("delta"):
                n = len(base64.b64decode(ev["delta"]))
                got["audio_ms"] += n / 2 / RATE * 1000
                if first_audio_at is None:
                    first_audio_at = time.time()
                    # SHE IS TALKING. Interrupt her.
                    injected = True
                    pending = asyncio.create_task(inject())

            elif t == "response.output_audio_transcript.done":
                got["forced_transcript"] = ev.get("transcript")

            elif t == "input_audio_buffer.speech_started":
                got["speech_started"] = True
            elif t == "input_audio_buffer.committed":
                got["committed"] = True
            elif t == "conversation.item.input_audio_transcription.completed":
                iid = ev.get("item_id")
                if iid in seen_items:
                    continue                      # the second and third of three
                seen_items.add(iid)
                got["transcript"] = ev.get("transcript")

            if not quiet and t not in ("response.output_audio.delta",):
                print(f"    {t}")

        if pending and not pending.done():
            pending.cancel()
    return got


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--audio", default="static/audio/eve/back_off.mp3")
    ap.add_argument("--verbose", action="store_true")
    a = ap.parse_args()

    path = Path(a.audio)
    if not path.exists():
        raise SystemExit(f"no such audio: {path}")
    audio = pcm16(path)
    print(f"{config.XAI_VOICE_MODEL} / {config.XAI_VOICE}, "
          f"{len(audio) / 2 / RATE:.1f}s of injected audio from {path.name}")

    out = {}
    for label, interruptible in (("control (no field)", True),
                                 ("interruptible:false", False)):
        print(f"\n--- {label}")
        got = asyncio.run(run(audio, interruptible, not a.verbose))
        out[label] = got
        print(f"    her audio        {got['audio_ms']:.0f} ms")
        print(f"    her words        {(got['forced_transcript'] or '')[:70]!r}")
        print(f"    frames injected  {got['injected_frames']}")
        print(f"    speech_started   {got['speech_started']}")
        print(f"    committed        {got['committed']}")
        print(f"    HEARD            {(got['transcript'] or '(nothing)')[:70]!r}")
        for e in got["errors"]:
            print(f"    ERROR            {e}")

    c = out["control (no field)"]
    x = out["interruptible:false"]
    print("\n" + "-" * 66)
    if c["transcript"] and not x["transcript"]:
        print("  interruptible:false DROPS caller audio during playback. The "
              "control was heard and this was not.")
        print("  -> usable for the red tier and the imminent turn call, and for "
              "nothing else: commit 0bdec46.")
    elif c["transcript"] and x["transcript"]:
        print("  NO DIFFERENCE. Both runs were heard, so the field changes "
              "nothing on this wire whatever the docs say.")
        print("  -> the red tier must not be built on it. Refusing a barge-in "
              "stays the controller's job.")
    elif not c["transcript"]:
        print("  INCONCLUSIVE: the CONTROL was not heard either, so this says "
              "nothing about the field.")
        print("  -> injected audio during playback is not reaching VAD at all. "
              "Check the clip and the tail before reading anything into it.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
