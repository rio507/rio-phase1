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


# ---------------------------------------------------------------------------
# Visual conversation (docs/visual_qa.md)
# ---------------------------------------------------------------------------
VISUAL_QA_ENABLED = True

# --- frame ring buffer ------------------------------------------------------
# Six seconds at the 4 fps headway cadence. The point of retaining ANY history
# is that the newest frame is often not the best one: the driver asks about a
# car a beat after seeing it, and the frame where it was biggest and sharpest
# has usually already gone past. MAX_FRAMES is a hard ceiling so a fast client
# cannot grow the buffer without bound.
RING_SECONDS = 6.0
RING_MAX_FRAMES = 32

# Raw frames are NEVER written to disk unless this is turned on. The ring is
# RAM-only and dies with the session; nothing in the normal path leaves a
# picture of the road behind. Turning this on writes the selected frame and
# crop of each visual answer under training_data/visual/<session_id>/, which is
# useful for reviewing a drive and is a privacy decision the operator makes
# deliberately, not a default.
RING_PERSIST = False

# --- frame selection --------------------------------------------------------
# How far back a "best frame" may be pulled. Beyond this the scene has moved on
# and answering from it would be answering about a different moment.
FRAME_MAX_AGE_S = 4.0
# Sharpness is measured lazily, only on the shortlist, because it costs a JPEG
# decode per frame (~5 ms). This caps how many get decoded per question.
FRAME_SHORTLIST = 6

# --- crops ------------------------------------------------------------------
# Context around the object, as a fraction of the box. The spec asks for "a
# high-res crop with surrounding context": a vehicle cut exactly at its own
# edges loses the road, the lane and the vehicles beside it, which is most of
# what makes a shape readable as a particular car.
CROP_PAD_FRAC = 0.6
# Crops are upscaled to at least CROP_MIN_PX on the long side. This is not
# cosmetic and it is not "adding detail" -- it is about how many image tokens
# the object gets at the far end. A high-detail image is tiled at ~512 px, so a
# 400 px crop is one tile and the vehicle inside it lands on a fraction of one;
# doubling it puts the same pixels across four tiles and the encoder spends
# proportionally more of its attention on the car.
#
# MEASURED, on the white saloon in the test clip (398x246 native crop, true
# object 245x111 px):
#     native      -> "a Toyota Camry"        wrong
#     upscaled x2 -> "a Lexus LS 460"        correct
#     upscaled x3 -> "a Lexus LS 460"        correct
# The pixels are identical in all three. Only the tiling changed.
#
# There is a real limit past which this stops being true: when the object was
# genuinely tiny in the source frame, no amount of interpolation puts a badge
# back, and a model shown a smooth 768 px image of a 30 px car will read detail
# that was never there. So the TRUE object size travels with the crop and
# anything under CROP_DETAIL_LIMIT_PX is flagged to the model as detail-limited.
CROP_MIN_PX = 768
CROP_MAX_PX = 1024
CROP_DETAIL_LIMIT_PX = 96

# --- Qwen enrichment --------------------------------------------------------
# Attribute enrichment is on demand, never on the 4 fps path: one Qwen call per
# crop, capped, cached per track for this long. It runs when a question turns on
# an attribute ("the silver one") or when the referent needs describing, and NOT
# for plain scene questions -- GPT-5.5 is looking at the same frame and reads
# colour off it directly, so paying an 8B decode for that would be latency spent
# on nothing.
ENRICH_ENABLED = True
ENRICH_MAX_OBJECTS = 3
ENRICH_TTL_S = 20.0
ENRICH_MAX_NEW_TOKENS = 48

# --- the multimodal turn ----------------------------------------------------
OPENAI_VISUAL_MODEL = OPENAI_CHAT_MODEL
# Roomier than the 300 the voice path uses: a visual answer is two or three
# sentences rather than one, and the same finish_reason=length failure that
# forced 300 up from 120 applies here with a longer reply to fit. Observed at
# 300 with reasoning_effort="low": the thinking pass consumed the whole budget
# and the reply came back empty. The margin is deliberate — an over-long reply
# gets truncated, an under-budgeted one is silence.
OPENAI_VISUAL_MAX_TOKENS = 700
# "low" rather than the voice path's "none". Reading a shape off a photograph
# and saying honestly how sure you are is the one thing in this product that
# actually benefits from a thinking pass.
OPENAI_VISUAL_REASONING_EFFORT = "low"
# Full frame at "auto", crop at "high": the crop is the image the answer turns
# on, and it is small.
VISUAL_FRAME_DETAIL = "auto"
VISUAL_CROP_DETAIL = "high"

# How many prior conversation turns ride along for follow-ups.
VISUAL_HISTORY_TURNS = 6
# An active referent older than this is stale: the driver has moved on, and
# "what year is it" should not silently attach to a car from two minutes ago.
REFERENT_TTL_S = 90.0
