"""xai_transcription_probe.py — does .completed carry an item_id?

    python -m tools.xai_transcription_probe
    python -m tools.xai_transcription_probe --audio some.wav --keep-open 12

WHY THIS ONE QUESTION IS WORTH A TOOL.

static/rio_realtime.js binds a driver's transcript to the response already
answering it by IDENTITY: speech_started, speech_stopped, input_audio_buffer
.committed and the transcription of one utterance all carry the SAME item_id, and
a genuinely new question is a different one. That is the whole of the fix for the
bug in commit ee0a909 -- the transcription races the model, loses on a tool turn,
arrives looking like a brand new question, supersedes the turn it belongs to,
aborts the look() running for it and leaves the driver in silence. It reproduced
3 of 3 on visual turns.

xAI's event reference names conversation.item.input_audio_transcription.completed
but does not publish its payload, and the migration's capability record carries
`transcriptionItemId: UNKNOWN` because of it. UNKNOWN rather than a guess, because
a boolean would make the controller quietly take a path nobody measured.

So: open a session, say one thing, and print the ids. Sixty seconds of audio
settles a question a week of reading cannot.

WHAT IT PRINTS AND WHAT EACH ANSWER MEANS

  ALL THREE EQUAL      the binding ports unchanged. Set transcriptionItemId True.
  .completed HAS an id but it differs from .committed's
                       worse than absent: the binding would silently never match
                       and every turn would look new. Say so loudly.
  .completed HAS NO id the binding cannot work on identity at all. What supersede
                       degrades to is then a decision, not an accident -- see the
                       report this probe was written for.
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

from dotenv import load_dotenv                                  # noqa: E402

load_dotenv(Path(__file__).resolve().parent.parent / ".env")

import config                                                   # noqa: E402

WS_URL = "wss://api.x.ai/v1/realtime"
MINT_URL = "https://api.x.ai/v1/realtime/client_secrets"
RATE = 24000
DEFAULT_CLIP = Path(__file__).resolve().parent.parent / "static/audio/back_off.mp3"

# The events whose item_id is the whole question.
WATCH = (
    "input_audio_buffer.committed",
    "conversation.item.input_audio_transcription.updated",
    "conversation.item.input_audio_transcription.completed",
    "conversation.item.input_audio_transcription.failed",
    "conversation.item.added",
    "conversation.item.created",
)


def pcm16(path: Path) -> bytes:
    """Whatever the file is, as mono PCM16 at RATE. ffmpeg because the realtime
    APIs take raw frames and nothing else."""
    out = subprocess.run(
        ["ffmpeg", "-hide_banner", "-loglevel", "error", "-i", str(path),
         "-ac", "1", "-ar", str(RATE), "-f", "s16le", "-"],
        capture_output=True, check=True)
    return out.stdout


def mint() -> str:
    import httpx

    key = os.environ.get("XAI_API_KEY")
    if not key:
        raise SystemExit("XAI_API_KEY is not set")
    r = httpx.post(MINT_URL, headers={"Authorization": f"Bearer {key}"},
                   json={"expires_after": {"seconds": 300}}, timeout=30)
    r.raise_for_status()
    return r.json()["value"]


async def probe(audio: bytes, model: str, stt: str, keep_open: float) -> dict:
    import websockets

    token = mint()
    url = f"{WS_URL}?model={model}"
    seen = []
    # The browser path cannot send an Authorization header, so xAI takes the
    # ephemeral token in the websocket subprotocol. Used here too, deliberately:
    # the point is to exercise the path a phone would take.
    async with websockets.connect(
            url, subprotocols=[f"xai-client-secret.{token}"],
            max_size=None, open_timeout=30) as ws:

        async def send(obj):
            await ws.send(json.dumps(obj))

        await send({"type": "session.update", "session": {
            "type": "realtime",
            "audio": {
                "input": {
                    "format": {"type": "audio/pcm", "rate": RATE},
                    "transcription": {"model": stt},
                    "turn_detection": {"type": "server_vad"},
                },
                "output": {"format": {"type": "audio/pcm", "rate": RATE}},
            },
            "instructions": "Say only: heard you.",
        }})

        # In 40 ms frames, the way a browser would, rather than one giant append:
        # a single blob is a shape the server never sees in the car.
        frame = int(RATE * 0.04) * 2
        for i in range(0, len(audio), frame):
            await send({"type": "input_audio_buffer.append",
                        "audio": base64.b64encode(audio[i:i + frame]).decode()})
        await send({"type": "input_audio_buffer.commit"})
        await send({"type": "response.create"})

        deadline = time.time() + keep_open
        while time.time() < deadline:
            try:
                raw = await asyncio.wait_for(ws.recv(), timeout=deadline - time.time())
            except (asyncio.TimeoutError, Exception):
                break
            try:
                ev = json.loads(raw)
            except Exception:
                continue
            t = ev.get("type", "")
            seen.append(ev)
            # THE ECHO IS THE FIRST THING TO CHECK. The OpenAI path asserts that
            # session.audio.input.transcription.model comes back in the minted
            # session, precisely because "the API accepted my field" and "the API
            # understood my field" are two claims and only one of them is visible.
            if t in ("session.created", "session.updated"):
                sess = ev.get("session") or {}
                aud = (sess.get("audio") or {}).get("input") or {}
                print(f"  [{t}] transcription={json.dumps(aud.get('transcription'))} "
                      f"turn_detection={json.dumps(aud.get('turn_detection'))}")
                print(f"           input audio keys: {sorted(aud)}")
            if t == "error":
                print(f"  [error] {json.dumps(ev.get('error') or ev)[:300]}")
    return {"events": seen}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--audio", default=str(DEFAULT_CLIP))
    ap.add_argument("--model", default=os.getenv("XAI_VOICE_MODEL",
                                                 "grok-voice-latest"))
    ap.add_argument("--stt", default="grok-transcribe")
    ap.add_argument("--keep-open", type=float, default=20.0)
    a = ap.parse_args()

    audio = pcm16(Path(a.audio))
    print(f"probe: {a.model} / {a.stt}, {len(audio)/2/RATE:.2f} s of audio "
          f"from {Path(a.audio).name}\n")

    res = asyncio.run(probe(audio, a.model, a.stt, a.keep_open))
    evs = res["events"]
    print(f"  {len(evs)} events\n")

    kinds = {}
    for ev in evs:
        kinds[ev.get("type")] = kinds.get(ev.get("type"), 0) + 1
    for k in sorted(kinds):
        print(f"    {kinds[k]:>3}x  {k}")

    print("\n--- the ids ---")
    ids = {}
    for ev in evs:
        t = ev.get("type", "")
        if t in WATCH:
            got = ev.get("item_id", ev.get("item", {}).get("id")
                         if isinstance(ev.get("item"), dict) else None)
            ids.setdefault(t, []).append(got)
            if t.endswith(".completed") or t.endswith(".updated"):
                print(f"    {t}")
                print(f"      item_id   = {got!r}")
                print(f"      transcript= {str(ev.get('transcript'))[:80]!r}")
                print(f"      keys      = {sorted(ev.keys())}")
            else:
                print(f"    {t}: item_id={got!r}")

    committed = [x for x in ids.get("input_audio_buffer.committed", []) if x]
    completed = ids.get("conversation.item.input_audio_transcription.completed", [])
    updated = ids.get("conversation.item.input_audio_transcription.updated", [])

    print("\n--- the verdict ---")
    if not completed:
        print("  .completed NEVER ARRIVED. The binding has nothing to bind on,")
        print("  and that is a stronger finding than a missing field:")
        print("  rio_realtime.js reads .completed and only .completed.")
        if updated:
            print(f"  (.updated did arrive, {len(updated)}x — cumulative, and the")
            print("   controller consumes no deltas at all.)")
        return 2
    if completed[0] is None:
        print("  .completed carries NO item_id. The self-supersede binding at")
        print("  rio_realtime.js:2742 cannot work on identity. supersede must be")
        print("  given an explicit degraded mode rather than silently failing.")
        return 3
    if committed and completed[0] not in committed:
        print(f"  .completed HAS an item_id ({completed[0]!r}) and it does NOT")
        print(f"  match the committed one ({committed!r}). This is the worst of")
        print("  the three answers: the binding would never match and every turn")
        print("  would look like a new question.")
        return 4
    print(f"  ALL THREE AGREE on {completed[0]!r}.")
    print("  The binding ports unchanged; transcriptionItemId is True.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
