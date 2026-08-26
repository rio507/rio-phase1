"""Speech synthesis — the fallback voice, and the only one this process owns.

RIO's active voice is the live session (config.VOICE_BACKEND = "realtime"): her
own replies come out of it, and deterministic lines are DICTATED to it word for
word, so that a warning and a conversation sound like the same person.

This file is what speaks when that is not possible. No session open and none
can be started, dictation not starting inside its budget, the model refusing —
all of them end here, and none of them is exotic in a car that drives through
tunnels. The code is kept complete and off the active path, which is what makes
it a fallback rather than a deletion.

It is reached through the server's TTS endpoints, which are therefore the
fallback path by definition: the browser only calls them when the live voice
could not say the line.
"""
import os

from elevenlabs.client import ElevenLabs

import config

eleven = ElevenLabs(api_key=os.getenv("ELEVENLABS_API_KEY"))


class VoiceUnavailable(RuntimeError):
    """This process cannot synthesise with the backend it was asked for.

    Raised rather than silently substituting: a caller that asked for a voice
    it did not get should know, and the one thing worse than the wrong voice is
    the wrong voice arriving unannounced.
    """


def synthesize_stream(text: str, backend: str = None):
    """Stream MP3 for one line.

    `backend` defaults to the FALLBACK backend rather than to the active one,
    because every caller of this function is the fallback path — the active
    path never gets here, it dictates to the live session in the browser.
    """
    backend = backend or config.VOICE_FALLBACK_BACKEND
    if backend != "elevenlabs":
        raise VoiceUnavailable(
            f"no local synthesiser for backend {backend!r}; RIO's active voice "
            "is the live session and this process cannot produce it")

    voice_id = os.getenv("ELEVENLABS_VOICE_ID")
    if not voice_id:
        raise RuntimeError("Missing ELEVENLABS_VOICE_ID")

    stream = eleven.text_to_speech.stream(
        text=text,
        voice_id=voice_id,
        model_id=config.ELEVENLABS_MODEL,
        output_format="mp3_44100_128",
    )

    for chunk in stream:
        yield chunk
