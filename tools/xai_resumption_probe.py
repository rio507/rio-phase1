"""xai_resumption_probe.py — does a dropped session come back knowing anything?

    python -m tools.xai_resumption_probe

WHAT FAILS TODAY, AND WHY THIS IS THE ONE EXTENSION WORTH WANTING.

On a real drop -- a tunnel, a handover, a phone that slept -- the conversation is
gone. The page reconnects and RIO has no memory of the last twenty minutes: not
the destination, not the question she was halfway through answering, not that the
driver already asked about the tyre pressure. Our own resume mechanism is a
different thing and a smaller one: it carries what the driver HEARD of a single
interrupted answer, so she can finish that sentence. It has never carried a
conversation.

xAI documents resumption.enabled and a server-assigned conversation_id to hand
back on reconnect. MEASURED, AND ONLY ONE OF THREE SPELLINGS WORKS -- see the
`spellings` list below and the result in static/rio_provider.js. So:

    1. open a session, tell it something only this conversation could know
    2. read the conversation_id it assigns, if it assigns one
    3. drop the socket the way a tunnel does -- no goodbye
    4. reconnect with the id and ask about the thing from step 1

A CORRECT ANSWER IN STEP 4 IS THE FEATURE. An empty or refused reconnect means the
capability record's `resumption: 'conversation_id'` is documentation rather than
behaviour, and the honest place for that is UNKNOWN or false.

AND IT SAYS WHAT DEGRADES ON A DROP, which is a report item and currently an
analysis rather than a measurement: a WebSocket close is final (there is no
`disconnected` state between open and gone, so REALTIME_PEER_DISCONNECT_GRACE_MS
has nothing to time on this wire), and the controller is told at once.

The four measured facts are honoured: output_modalities and an output voice are
sent, no explicit commit, .completed deduplicated by id, and a silence tail after
injected audio so server_vad can hear the end of the utterance.
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

# Something no model could answer from general knowledge, and short enough to say
# in one breath. It is the DRIVER speaking, so it goes in as audio.
SECRET_LINE = "Remember this: my parking space number is forty-one."
ASK_LINE = "What did I say my parking space number was?"


def say(text: str) -> bytes:
    """The driver's lines, as PCM16 at RATE.

    Rendered through xai_voice.render_speech, which is Eve reading them. It has to
    be a VOICE and not a tone -- server_vad decides what is speech -- and there is
    no local synthesiser on this machine and no /v1/audio/speech on this team (403),
    so the only voice available is hers. tools/xai_interruptible_probe.py proved the
    loop works: a clip of Eve injected into the input buffer came back correctly
    transcribed.

    WHAT THIS COSTS IN HONESTY, said plainly: the driver is a synthesised voice with
    no room, no distance and no engine behind it. That is fine for "does the
    conversation survive a reconnect", which is a question about state, and it is
    not evidence about VAD on a person.
    """
    r = xai_voice.render_speech(text)
    if not r.get("ok"):
        raise SystemExit(f"could not render the driver's line: {r.get('note')}")
    conv = subprocess.run(
        ["ffmpeg", "-hide_banner", "-loglevel", "error", "-i", "-",
         "-ac", "1", "-ar", str(RATE), "-f", "s16le", "-"],
        input=r["wav"], capture_output=True, check=True)
    return conv.stdout


async def turn(ws, audio: bytes, want_audio: float = 25.0) -> dict:
    """One driver utterance and whatever comes back."""
    got = {"transcript": None, "her_words": "", "audio_ms": 0.0, "errors": []}
    seen = set()
    for i in range(0, len(audio) - FRAME_BYTES, FRAME_BYTES):
        await ws.send(json.dumps({
            "type": "input_audio_buffer.append",
            "audio": base64.b64encode(audio[i:i + FRAME_BYTES]).decode()}))
        await asyncio.sleep(FRAME_MS / 1000 / 4)     # faster than real time
    tail = b"\x00" * FRAME_BYTES
    for _ in range(int(config.XAI_SILENCE_TAIL_MS // FRAME_MS) + 4):
        await ws.send(json.dumps({
            "type": "input_audio_buffer.append",
            "audio": base64.b64encode(tail).decode()}))
        await asyncio.sleep(FRAME_MS / 1000 / 4)

    deadline = time.time() + want_audio
    done = False
    while time.time() < deadline and not done:
        try:
            raw = await asyncio.wait_for(ws.recv(), timeout=3.0)
        except asyncio.TimeoutError:
            if got["audio_ms"]:
                break
            continue
        ev = json.loads(raw)
        t = ev.get("type")
        if t == "error":
            got["errors"].append(str(ev)[:300])
        elif t == "response.output_audio.delta" and ev.get("delta"):
            got["audio_ms"] += len(base64.b64decode(ev["delta"])) / 2 / RATE * 1000
        elif t == "response.output_audio_transcript.done":
            got["her_words"] = ev.get("transcript") or ""
        elif t == "conversation.item.input_audio_transcription.completed":
            if ev.get("item_id") in seen:
                continue
            seen.add(ev.get("item_id"))
            got["transcript"] = ev.get("transcript")
        elif t == "response.done":
            done = bool(got["audio_ms"])
    return got


async def connect(key: str, session_extra: dict = None,
                  query_conversation_id: str = None):
    import websockets

    url = f"{xai_voice.WS_URL}?model={config.XAI_VOICE_MODEL}"
    if query_conversation_id:
        url += f"&conversation_id={query_conversation_id}"
    ws = await websockets.connect(
        url,
        additional_headers={"Authorization": f"Bearer {key}"},
        max_size=1 << 24)
    pol = xai_voice.session_policy()
    if session_extra:
        pol = {**pol, **session_extra}
    await ws.send(json.dumps({"type": "session.update", "session": pol}))
    ids = {"session_id": None, "conversation_id": None, "errors": [],
           "events": [], "where": None}
    deadline = time.time() + 15
    while time.time() < deadline:
        raw = await asyncio.wait_for(ws.recv(), timeout=5.0)
        ev = json.loads(raw)
        ids["events"].append(ev.get("type"))
        if ev.get("type") == "error":
            ids["errors"].append(str(ev)[:300])
        # ANY conversation id, ANYWHERE in ANY event, because "no id was given"
        # is a claim about the whole handshake and not about one field of one
        # event. Scanned rather than looked up.
        found = _find_conversation_id(ev)
        if found and not ids["conversation_id"]:
            ids["conversation_id"], ids["where"] = found
        if ev.get("type") in ("session.created", "session.updated"):
            s = ev.get("session") or {}
            ids["session_id"] = s.get("id") or ev.get("session_id") \
                or ids["session_id"]
            if ev.get("type") == "session.updated":
                break
    return ws, ids


def _find_conversation_id(obj, path=""):
    """(id, where) for the first key that looks like a conversation id."""
    if isinstance(obj, dict):
        for k, v in obj.items():
            here = f"{path}.{k}" if path else k
            if "conversation" in k.lower():
                if isinstance(v, str) and v:
                    return v, here
                if isinstance(v, dict) and v.get("id"):
                    return v["id"], here + ".id"
            hit = _find_conversation_id(v, here)
            if hit:
                return hit
    elif isinstance(obj, list):
        for i, v in enumerate(obj):
            hit = _find_conversation_id(v, f"{path}[{i}]")
            if hit:
                return hit
    return None


async def main_async(verbose: bool, secret: bytes, ask: bytes) -> int:
    key = os.environ.get("XAI_API_KEY")
    if not key:
        raise SystemExit("XAI_API_KEY is not set")
    print(f"{config.XAI_VOICE_MODEL} / {config.XAI_VOICE}")

    print("\n--- first session: tell her something")
    ws, ids = await connect(key, {"resumption": {"enabled": True}})
    print(f"    resumption accepted : {not ids['errors']}")
    for e in ids["errors"]:
        print(f"    ERROR               {e}")
    print(f"    session id          : {ids['session_id']}")
    print(f"    handshake events    : {', '.join(x for x in ids['events'] if x)}")
    print(f"    conversation id     : {ids['conversation_id'] or '(none given)'}"
          f"{' at ' + ids['where'] if ids['where'] else ''}")
    a = await turn(ws, secret)
    print(f"    she heard           : {(a['transcript'] or '')[:60]!r}")
    print(f"    she said            : {a['her_words'][:60]!r}")

    # A CONTROL, ON THE SAME SOCKET. If she cannot answer this without a
    # reconnect, the question below is not about resumption at all.
    b = await turn(ws, ask)
    print(f"    control (same sock) : {b['her_words'][:70]!r}")
    same_socket_ok = "41" in b["her_words"] or "forty-one" in b["her_words"].lower()
    print(f"    remembers in-session: {same_socket_ok}")

    print("\n--- the tunnel: drop the socket with no goodbye")
    # ABORTED, not closed. A close frame is a goodbye and 1006 cannot be SENT --
    # it is what the other end reports when the connection simply stopped, which
    # is what a tunnel does and what this has to imitate.
    try:
        ws.transport.abort()
    except Exception:
        await ws.close()

    # THREE SPELLINGS, BECAUSE ONE PROVES NOTHING. force_message is the lesson:
    # the same extension worked perfectly in the content-array form and produced
    # "Hey. What's up." in the `text` form, with no error either way. "It does not
    # work" is a claim about an API and it has to be tried more than one way
    # before it is worth writing down.
    cid = ids["conversation_id"]
    spellings = [
        ("session.conversation_id", {"resumption": {"enabled": True},
                                     "conversation_id": cid}, None),
        ("resumption.conversation_id",
         {"resumption": {"enabled": True, "conversation_id": cid}}, None),
        ("?conversation_id= on the URL", {"resumption": {"enabled": True}}, cid),
    ]
    resumed = False
    said = {}
    for label, extra, query_cid in spellings:
        if not cid:
            break
        print(f"\n--- reconnect, {label}")
        try:
            ws2, ids2 = await connect(key, extra, query_cid)
        except Exception as e:
            print(f"    REFUSED             {type(e).__name__}: {str(e)[:90]}")
            continue
        print(f"    accepted            : {not ids2['errors']}")
        for e in ids2["errors"]:
            print(f"    ERROR               {e}")
        new_cid = ids2["conversation_id"]
        print(f"    conversation id     : {new_cid}"
              f"{'  (SAME as before)' if new_cid == cid else '  (a NEW one)'}")
        c = await turn(ws2, ask)
        print(f"    she said            : {c['her_words'][:70]!r}")
        said[label] = c["her_words"]
        try:
            await ws2.close()
        except Exception:
            pass
        if "41" in c["her_words"] or "forty-one" in c["her_words"].lower():
            resumed = True
            print("    REMEMBERED")
            break

    c = {"her_words": " | ".join(said.values())}
    print("\n" + "-" * 66)
    if not ids["conversation_id"]:
        print("  NO conversation_id was ever given, so there is nothing to hand "
              "back.")
        print("  -> the capability record should say UNKNOWN or false, not "
              "'conversation_id'. A drop loses the conversation, as it does "
              "today.")
    elif resumed:
        print("  RESUMED: a new socket with the old conversation_id remembered "
              "the drive.")
        print("  -> worth having, and it OVERLAPS our own resume (which carries "
              "what was HEARD, not what was generated). Two resume mechanisms "
              "that disagree about what she said is worse than one — decide "
              "which owns an interrupted answer before wiring it.")
    else:
        print("  NOT RESUMED: the id was accepted and the memory did not come "
              "back.")
        print("  -> a drop still loses the conversation. The capability record "
              "must not promise otherwise.")
    if not same_socket_ok:
        print("  NOTE: she did not hold the fact within ONE socket either, so "
              "read the above with care — the probe may be asking too much of "
              "a three-turn session rather than measuring resumption.")
    return 0


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--verbose", action="store_true")
    a = ap.parse_args()
    # RENDERED OUTSIDE THE LOOP, because render_speech calls asyncio.run() and
    # asyncio.run() inside a running loop raises -- which is the same trap
    # tools/xai_drive_harness.py hit, where every turn failed and the harness
    # then reported no silent turns because a `continue` skipped the check.
    return asyncio.run(main_async(a.verbose, say(SECRET_LINE), say(ASK_LINE)))


if __name__ == "__main__":
    sys.exit(main())
