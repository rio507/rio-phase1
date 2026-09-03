"""Speech synthesis for everything RIO does not compose herself.

WHAT COMES THROUGH HERE
-----------------------
Turn instructions, vehicle-health announcements, the calm headway tier: lines
written by policy code, word for word, addressed by an id rather than sent as
text. The server's TTS endpoints (/nav/voice, /headway_voice,
/vehicle/health/voice) are the only callers, and each of them looks its line up
in a table before it gets here — which is what keeps a deterministic warning
channel from quietly becoming a text-to-speech endpoint.

ONE VOICE, TWO MODELS
---------------------
The voice id is the same one RIO converses in. That is the whole of "one voice
everywhere", and it is now true by configuration rather than by pre-rendering:
conversation, warnings, turns and the offline clips all name
config.ELEVENLABS_VOICE_ID.

The MODEL is not the same, and should not be. Conversation runs on v3
conversational, which has the prosody and the audio tags. Everything here runs
on flash, which has neither and is roughly three times faster to first byte —
and a turn instruction six seconds from a junction is a line where speed is the
only property that matters. A driver cannot hear the difference between two
readings of "Take the next left"; they can certainly hear it arrive late.

NO TAGS, EVER
-------------
Every line is stripped of audio tags before synthesis, unconditionally. Nothing
upstream is supposed to produce one — these lines come from tables, not models
— which is exactly why the strip is here rather than trusted to be unnecessary:
the cost is a regex, and the failure it prevents is a synthesiser reading the
word "sighs" out loud at a junction.
"""
import os

from elevenlabs.client import ElevenLabs

import config
import voice_tags


class VoiceUnavailable(RuntimeError):
    """This process cannot synthesise with the backend it was asked for.

    Raised rather than silently substituting: a caller that asked for a voice
    it did not get should know, and the one thing worse than the wrong voice is
    the wrong voice arriving unannounced.
    """


def api_key() -> str:
    """The key, with the whitespace taken off.

    A key pasted out of a browser arrives with a non-breaking space in front of
    it. Every call then fails with "Invalid API key", which reads as a wrong
    key and is not one, and nothing in the message mentions whitespace.
    """
    return (os.getenv("ELEVENLABS_API_KEY") or "").strip()


def voice_id() -> str:
    """The one voice. Config first, environment as the fallback."""
    return (config.ELEVENLABS_VOICE_ID
            or (os.getenv("ELEVENLABS_VOICE_ID") or "").strip())


_client = None


def client() -> ElevenLabs:
    """Built on first use, not at import.

    It used to be constructed at module scope, which meant a missing or
    malformed key turned every importer of this file — including the test
    suite and the clip renderer — into an import error rather than a synthesis
    error. Those are different failures and deserve different messages.
    """
    global _client
    if _client is None:
        _client = ElevenLabs(api_key=api_key())
    return _client


def synthesize_stream(text: str, backend: str = None, model: str = None):
    """Stream MP3 for one deterministic line.

    MP3 rather than the PCM the conversation path uses, and deliberately: these
    arrive as an ordinary HTTP response into an <audio> element that plays them
    directly, with no scheduling and nothing to fade. The conversation path
    needs raw samples because it has to be able to stop RIO in 30 ms; a warning
    is the thing doing the stopping.
    """
    backend = backend or config.VOICE_FALLBACK_BACKEND
    if backend != "elevenlabs":
        raise VoiceUnavailable(
            f"no local synthesiser for backend {backend!r}; the only voice "
            "this process can produce on its own is ElevenLabs")

    vid = voice_id()
    if not vid:
        raise RuntimeError("Missing ELEVENLABS_VOICE_ID")
    if not api_key():
        raise RuntimeError("Missing ELEVENLABS_API_KEY")

    line = voice_tags.strip(text)
    if not line.strip():
        raise VoiceUnavailable("nothing to say once the line was cleaned")

    stream = client().text_to_speech.stream(
        text=line,
        voice_id=vid,
        model_id=model or config.ELEVENLABS_DETERMINISTIC_MODEL,
        output_format="mp3_44100_128",
    )

    for chunk in stream:
        yield chunk
