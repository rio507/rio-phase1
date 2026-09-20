"""xai_voice.py — grok-voice, server-side, for rendering clips and nothing else.

WHAT THIS IS AND IS NOT ON THE PATH OF. Exactly as realtime.render_speech: a
short-lived session that speaks one line and hands back the audio, so the
critical clips can be PRE-RENDERED in the same voice the live session would use.
The car never calls this. In the car the browser holds its own session.

It exists as its own module rather than as a branch inside realtime.py for the
reason realtime.py is asserted to contain no model id at all, prose included:
that file is the OpenAI realtime client and keeping a second vendor's websocket
framing out of it is the same discipline static/rio_provider.js applies in the
browser.

FOUR THINGS THIS SESSION WILL DO IF YOU CONFIGURE IT WRONG, all of them quietly,
and all of them cost me a wrong conclusion before they were understood:

  no output_modalities        the session accepts every field, echoes the config
  no output voice             back, runs VAD, commits audio -- and returns
                              NOTHING. Not an error. Silence.
  an explicit commit with     server_vad decides where an utterance ENDS and has
  server_vad on              to hear the end. With no tail of silence the turn
                              stays open and no transcript ever arrives.
  ?model= in the URL          is IGNORED. A deliberately invalid model name
                              connects and behaves identically, so the query
                              parameter proves nothing about what answered. The
                              session's own session.updated payload is the only
                              honest record, and it names
                              grok-voice-think-fast-2.0 for grok-voice-latest.

So the session config below is spelled out in full and the resolved model is read
back off the wire rather than assumed.
"""
import base64
import io
import json
import os
import time
import wave

import config

WS_URL = "wss://api.x.ai/v1/realtime"
MINT_URL = "https://api.x.ai/v1/realtime/client_secrets"
STT_URL = "https://api.x.ai/v1/stt"

# PCM16 mono. 24 kHz is what the session is asked for and what it returns; the
# WAV header written below has to agree with it or every clip plays at the wrong
# pitch, which is the kind of bug that sounds like a bad voice.
RATE = 24000


def _ephemeral(timeout_s: float) -> str:
    """A short-lived secret for one render.

    The account key would work here -- this is a server -- but the mint is the
    path a browser must use, and exercising it means the credential half of the
    migration is proven by every clip run rather than by a separate test.
    """
    import httpx

    key = os.environ.get("XAI_API_KEY")
    if not key:
        raise RuntimeError("XAI_API_KEY is not set")
    r = httpx.post(MINT_URL, headers={"Authorization": f"Bearer {key}"},
                   json={"expires_after": {"seconds": max(60, int(timeout_s) * 4)}},
                   timeout=30)
    r.raise_for_status()
    return r.json()["value"]


def session_config(voice: str) -> dict:
    """Every field spelled out. See the module docstring for what each omission
    costs, which is silence rather than an error in three cases out of four."""
    return {
        "type": "realtime",
        # WITHOUT THIS THE SESSION RETURNS NOTHING AT ALL.
        "output_modalities": ["audio"],
        "audio": {
            "input": {
                "format": {"type": "audio/pcm", "rate": RATE},
                "transcription": {"model": config.XAI_STT_MODEL},
                "turn_detection": {"type": "server_vad"},
            },
            "output": {
                "voice": voice,
                "format": {"type": "audio/pcm", "rate": RATE},
            },
        },
    }


async def _speak(line: str, voice: str, timeout_s: float) -> dict:
    import websockets

    token = _ephemeral(timeout_s)
    pcm = bytearray()
    transcript = ""
    resolved = None
    first_audio_ms = None
    t0 = time.time()

    async with websockets.connect(
            f"{WS_URL}?model={config.XAI_VOICE_MODEL}",
            subprotocols=[f"xai-client-secret.{token}"],
            open_timeout=min(30, timeout_s), max_size=None) as ws:

        await ws.send(json.dumps({"type": "session.update",
                                  "session": session_config(voice)}))
        # Wait for the echo before asking for anything: it is where the resolved
        # model name and the accepted voice are readable, and it is the only
        # place either is stated.
        while True:
            ev = json.loads(await _recv(ws, timeout_s))
            if ev.get("type") == "session.updated":
                resolved = (ev.get("session") or {}).get("model")
                break
            if ev.get("type") == "error":
                return {"ok": False, "note": _err(ev)}

        # THE WORDS ARE THE REQUEST. response.create carrying the line is the
        # dictation path -- the same mechanism the live session uses for a
        # warning, so a clip rendered here is produced the way a dictated line
        # would be.
        t0 = time.time()
        await ws.send(json.dumps({"type": "response.create", "response": {
            "instructions": f"Say exactly this and nothing else: {line}"}}))

        while True:
            ev = json.loads(await _recv(ws, timeout_s))
            t = ev.get("type", "")
            if t == "response.output_audio.delta":
                if first_audio_ms is None:
                    first_audio_ms = round((time.time() - t0) * 1000, 1)
                pcm += base64.b64decode(ev.get("delta") or b"")
            elif t == "response.output_audio_transcript.done":
                transcript = ev.get("transcript") or transcript
            elif t == "response.output_audio_transcript.delta":
                if not transcript:
                    pass          # the .done carries the whole of it
            elif t == "error":
                return {"ok": False, "note": _err(ev)}
            elif t == "response.done":
                break

    if not pcm:
        return {"ok": False, "note": "no audio"}

    buf = io.BytesIO()
    with wave.open(buf, "wb") as w:
        w.setnchannels(1)
        w.setsampwidth(2)
        w.setframerate(RATE)
        w.writeframes(bytes(pcm))
    return {"ok": True, "wav": buf.getvalue(), "transcript": transcript,
            "ms": round((time.time() - t0) * 1000, 1),
            "first_audio_ms": first_audio_ms,
            "model": resolved, "voice": voice,
            "seconds": round(len(pcm) / 2 / RATE, 2)}


async def _recv(ws, timeout_s):
    import asyncio

    return await asyncio.wait_for(ws.recv(), timeout=timeout_s)


def _err(ev) -> str:
    e = ev.get("error") or ev
    if isinstance(e, dict):
        return str(e.get("message") or e.get("type") or e)[:200]
    return str(e)[:200]


def render_speech(text: str, voice: str = None, timeout_s: float = 45.0) -> dict:
    """Speak one line and return the audio.

    Returns {ok, wav, transcript, ms, model, voice, seconds} -- the same shape
    realtime.render_speech returns, so tools/render_alerts.py can treat the two
    interchangeably. `transcript` is the model's own account of what it said,
    which is the FIRST half of the verbatim check and on its own is not enough:
    see the comment above ClipUnverified in render_alerts, where a voice's own
    transcript passed a clip whose entire instruction had been replaced.
    """
    import asyncio

    line = (text or "").strip()
    if not line:
        return {"ok": False, "note": "no text"}
    try:
        return asyncio.run(_speak(line, voice or config.XAI_VOICE, timeout_s))
    except Exception as e:
        return {"ok": False, "note": f"{type(e).__name__}: {str(e)[:120]}"}


def transcribe(audio_bytes: bytes, filename: str = "clip.mp3") -> str:
    """What a finished clip actually says, according to a DIFFERENT model.

    /v1/stt with XAI_STT_MODEL, which is not the model that spoke -- and that
    independence is the entire point. The verifier exists because a voice's own
    transcript once agreed with a clip in which "Pull over when it's safe" had
    become "Hey, Ava, when it's safe": the instruction the warning exists to give,
    replaced by a greeting, and self-reported as correct.

    Returns "" when transcription is unavailable, which downgrades the check
    rather than failing the render -- the caller decides what an unverifiable
    clip is worth, and in render_alerts the answer is "it does not ship".
    """
    import httpx

    key = os.environ.get("XAI_API_KEY")
    if not key:
        return ""
    try:
        r = httpx.post(
            STT_URL, headers={"Authorization": f"Bearer {key}"},
            files={"file": (filename, audio_bytes)},
            data={"model": config.XAI_STT_MODEL}, timeout=60)
        r.raise_for_status()
        return r.json().get("text") or ""
    except Exception as e:
        print(f"      (could not verify: {type(e).__name__}) ", end="")
        return ""


# ---------------------------------------------------------------------------
# THE BROWSER'S CREDENTIAL, AND THE SESSION POLICY THAT TRAVELS WITH IT
# ---------------------------------------------------------------------------

def session_policy() -> dict:
    """The drive session's configuration, assembled server-side.

    THE SAME DISCIPLINE realtime.mint_client_secret FOLLOWS, and for the same
    reason: every number the browser applies is decided in config.py and carried
    here, so the page holds no second copy of a policy to drift from the one the
    tests check. The browser builds no session config of its own.

    THE FOUR FIELDS WHOSE ABSENCE IS SILENT are all present and all commented,
    because omitting any of them produces a session that accepts everything, runs
    VAD, and never speaks -- with no error event to notice.
    """
    import realtime

    cfg = realtime.session_config()
    pol = {
        "type": "realtime",
        # 1. WITHOUT THIS THE SESSION PRODUCES NOTHING AT ALL.
        "output_modalities": ["audio"],
        "instructions": cfg["instructions"],
        "tools": [dict(t) for t in cfg["tools"]],
        "tool_choice": "auto",
        "audio": {
            "input": {
                "format": {"type": "audio/pcm", "rate": RATE},
                "transcription": {"model": config.XAI_STT_SESSION_MODEL},
                "turn_detection": {"type": "server_vad"},
            },
            "output": {
                # 2. ...NOR WITHOUT THIS ONE.
                "voice": config.XAI_VOICE,
                "format": {"type": "audio/pcm", "rate": RATE},
            },
        },
    }
    if config.XAI_VOICE_EFFORT:
        # 3. 'none' or 'high' ONLY on this endpoint -- 'low' is a validation
        #    error. Measured: effort high has a tool-call p95 of 2,270 ms against
        #    521, so the drive session takes none. See config.XAI_VOICE_EFFORT.
        pol["reasoning"] = {"effort": config.XAI_VOICE_EFFORT}
    return pol


def mint_client_secret() -> dict:
    """An ephemeral secret for the browser, plus everything it needs to connect.

    Deliberately the same shape realtime.mint_client_secret returns, so
    /realtime/session can dispatch on the backend and the page's connect path does
    not branch on a vendor name.

    4. THE EXPLICIT COMMIT AND THE SILENCE TAIL are the browser's job and are
       carried as policy rather than left to it: server_vad owns the end of an
       utterance, so the page must NOT send input_audio_buffer.commit (the
       provider's outbound map drops it) and MUST keep feeding frames through the
       gap after speech, or the turn never closes. silence_tail_ms is how much.
    """
    pol = session_policy()
    secret = _ephemeral(300)
    return {
        "client_secret": secret,
        "ws_url": f"{WS_URL}?model={config.XAI_VOICE_MODEL}",
        # The subprotocol the token rides in, because a browser cannot set an
        # Authorization header on a WebSocket.
        "ws_subprotocol": f"xai-client-secret.{secret}",
        "model": config.XAI_VOICE_MODEL,
        "voice": config.XAI_VOICE,
        "live_voice": config.XAI_VOICE,
        "voice_backend": "xai_voice",
        "session": pol,
        "tool": "deep_dive",
        "tools": [t["name"] for t in pol["tools"]],
        "tool_schemas": [dict(t) for t in pol["tools"]],
        "sample_rate": RATE,
        # How long a tail of silence to keep sending after the driver stops, so
        # server_vad can hear the end of the utterance. 400 ms is ten 40 ms
        # frames; measured, audio that stops dead on the last syllable leaves the
        # turn open and no transcript ever follows.
        "silence_tail_ms": int(config.XAI_SILENCE_TAIL_MS),
        # .completed arrives three times per utterance with one id. The page
        # deduplicates BY ID rather than counting, and is told the number so a
        # drive log can notice if it ever changes.
        "transcript_repeats": 3,
        "expires_at": None,
    }


# ---------------------------------------------------------------------------
# A DETERMINISTIC LINE, WITHOUT ASKING A MODEL TO SAY IT
# ---------------------------------------------------------------------------
# force_message is xAI's own extension: a conversation.item.create whose item
# type synthesises the words with no model in the path. Measured, and the
# measurement matters more than the feature:
#
#   item = {"type": "force_message",
#           "content": [{"type": "input_text", "text": LINE}]}
#       -> 1.43 s of audio, transcript "Back off — now." EXACT, and no
#          response.create needed. This is the form to use.
#
#   item = {"type": "force_message", "text": LINE}  + response.create
#       -> 1.43 s of audio saying "Hey. What's up."
#
# READ THE SECOND ONE AGAIN. It produces AUDIO, of a plausible length, with no
# error -- and the words are a greeting instead of a warning. A caller that
# checked "did it make a sound" would ship it. That is the same failure as the
# clip quoted above ClipUnverified in tools/render_alerts.py, where "Pull over
# when it's safe" came out as "Hey, Ava, when it's safe" and the voice's own
# transcript said it was fine. Same shape, different mechanism, and this time the
# cause is one wrong field name.
#
#   item = {"type": "force_message", "role": "assistant", "text": LINE}
#       -> nothing at all. No audio, no error.
#
# So: the content-array form, and the transcript checked against the words even
# here. "Exact by construction" is a property of ONE spelling of this event.
#
# NOT YET MEASURED: interruptible:false. The field is documented to drop caller
# audio during playback, which is a policy change rather than a mechanism one --
# commit 0bdec46 ("Eleven refused barge-ins were eleven questions being
# answered") is the argument for using it only on the red tier and the imminent
# turn call. Testing it needs a barge-in to refuse, which needs two speakers.

def force_message_item(line: str, interruptible: bool = True) -> dict:
    """The conversation.item.create item that says `line` verbatim.

    The content-array form, because the `text` form produces a greeting -- see
    above. Callers should still verify the transcript: one wrong field name is
    the difference between a warning and "Hey. What's up."
    """
    item = {"type": "force_message",
            "content": [{"type": "input_text", "text": line}]}
    if not interruptible:
        # Documented to DROP caller audio while it plays, so a driver speaking
        # over it is not heard at all rather than merely ignored. Red tier and the
        # imminent turn call only.
        item["interruptible"] = False
    return item
