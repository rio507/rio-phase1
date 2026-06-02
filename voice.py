import os
from elevenlabs.client import ElevenLabs
import config

eleven = ElevenLabs(api_key=os.getenv("ELEVENLABS_API_KEY"))

def synthesize_stream(text: str):
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