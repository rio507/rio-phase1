import os

# RIO prompts now sourced from rio_prompts.py (compiled from behavior bible v1)
from rio_prompts import RIO_SYSTEM_PROMPT

OPENAI_CHAT_MODEL = "gpt-5.5"
OPENAI_STT_MODEL = "whisper-1"

OPENAI_TEMPERATURE = 1
OPENAI_MAX_TOKENS = 120

VOICE_BACKEND = "elevenlabs"
ELEVENLABS_MODEL = "eleven_flash_v2_5"

SYSTEM_PROMPT = RIO_SYSTEM_PROMPT

VISION_ENABLED = True
