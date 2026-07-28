import os

# RIO prompts now sourced from rio_prompts.py (compiled from behavior bible v1)
from rio_prompts import RIO_SYSTEM_PROMPT

OPENAI_CHAT_MODEL = "gpt-5.5"
OPENAI_STT_MODEL = "whisper-1"

OPENAI_TEMPERATURE = 1
# gpt-5.5 is a reasoning model: max_completion_tokens covers reasoning AND output.
# At 120 the reasoning pass could eat the whole budget and RIO returned an empty
# reply (finish_reason=length, 120/120 reasoning tokens). 300 leaves headroom.
OPENAI_MAX_TOKENS = 300

# Keep the thinking pass short — RIO needs fast, terse replies in a moving car.
# gpt-5.5 does not accept "minimal"; valid values are none/low/medium/high/xhigh.
# "none" spends zero reasoning tokens, so the whole budget is available for the
# spoken reply and the empty-reply failure cannot recur. "low" if RIO needs more.
OPENAI_REASONING_EFFORT = "none"

VOICE_BACKEND = "elevenlabs"
ELEVENLABS_MODEL = "eleven_flash_v2_5"

SYSTEM_PROMPT = RIO_SYSTEM_PROMPT

VISION_ENABLED = True
