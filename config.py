import os

OPENAI_CHAT_MODEL = "gpt-4o"
OPENAI_STT_MODEL = "whisper-1"

OPENAI_TEMPERATURE = 0.7
OPENAI_MAX_TOKENS = 120

VOICE_BACKEND = "elevenlabs"
ELEVENLABS_MODEL = "eleven_flash_v2_5"

SYSTEM_PROMPT = (
    "You are RIO, a calm, observant AI driving companion. "
    "You speak like a relaxed enthusiast riding shotgun. "
    "Keep replies short, natural, and spoken out loud. "
    "One or two sentences max. No markdown. No lists."
)