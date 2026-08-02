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
# Long side of the object IN THE ORIGINAL FRAME, below which fine detail is not
# really present. Raised from 96 after looking at what the numbers correspond
# to: the saloon that GPT-5.5 identifies correctly and hedges the year on sits
# at 245 px, while a car 70 m back comes in around 120 px and carries no
# readable badge at all. 96 was letting the second case through unflagged.
CROP_DETAIL_LIMIT_PX = 160

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


# ---------------------------------------------------------------------------
# Phase B — clarification, lost objects, comparisons, reading text
# ---------------------------------------------------------------------------

# How long RIO waits for an answer to "which one?". Longer than a referent's
# idle life is wrong (the driver has moved on) and shorter than a few seconds is
# wrong too (they were driving). A pending question that expires simply lapses:
# the next utterance is treated as a fresh one, never as an answer to something
# RIO asked a minute ago.
CLARIFY_TTL_S = 45.0

# Candidates offered in a clarifying question. Two is the natural shape of the
# question ("the black one, or the white one?"); three is the most a driver can
# hold while driving, and past that the honest move is to describe the group.
CLARIFY_MAX_CANDIDATES = 3
CLARIFY_MAX_TOKENS = 300

# A comparison needs exactly two objects. More than that is not a comparison,
# it is a survey, and the answer stops being useful at a glance.
COMPARE_MAX_OBJECTS = 2

# Reading text needs resolution above all else, so the full frame goes at high
# detail regardless of what a scene question would use. Nothing in the
# detector's vocabulary is a sign (COCO gives us person/bicycle/car/motorcycle/
# bus/truck), so there is usually no tracked object to crop and the frame is
# all there is — see docs/visual_qa.md §12.
READ_TEXT_FRAME_DETAIL = "high"


# ---------------------------------------------------------------------------
# Vehicle health — tires (phase 1)
# ---------------------------------------------------------------------------
# Every threshold the Vehicle Health column reacts to lives here and nowhere
# else. tires.py reads them, the browser never sees them: a number duplicated in
# JavaScript is a number that will disagree with this file the first time
# somebody tunes it.

# Which TireHealthProvider is live. "mock" until there is hardware to point at;
# the Bluetooth / RF receiver / ESP32 / RIO Connect providers land here later
# and nothing above this line changes when they do.
TIRE_PROVIDER = "mock"

# Placard pressures, cold, per corner. Front and rear differ on most cars and
# the panel is worthless if it compares every tire to the same number: a
# correctly-inflated rear would read 2 PSI low all day and the driver would
# learn to ignore the warning, which is the only real failure mode this feature
# has.
TIRE_TARGET_PSI = {"FL": 35.0, "FR": 35.0, "RL": 33.0, "RR": 33.0}

# How far under target before it is worth saying something. 3.0 PSI is roughly
# where handling and wear start to move and comfortably outside the ~1.5 PSI of
# swing a tire sees between a cold morning and a hot motorway hour — tighter
# than this and the panel cries wolf every sunrise.
TIRE_PRESSURE_WARN_DELTA = 3.0
# ~20% under a 33-35 PSI placard: the point at which the sidewall is carrying
# load it was not designed to carry and heat starts to build faster than the
# tire can shed it. This is a "stop driving on it" number, not a "top it up"
# number, and it is coloured accordingly.
TIRE_PRESSURE_CRITICAL_DELTA = 6.0
# Over-inflation gets a wider band than under-inflation because it is genuinely
# less dangerous and because a tire that has been sitting in the sun legitimately
# reads high.
TIRE_PRESSURE_HIGH_DELTA = 5.0

# Running temperature. A tire on a warm day at speed sits around 100-120°F;
# 150 means something is wrong with the pressure, the alignment or the brake
# behind it, and 180 is where the rubber-to-belt bond starts to suffer.
TIRE_TEMP_WARN_F = 150.0
TIRE_TEMP_CRITICAL_F = 180.0

# Loss over 24 hours that counts as a leak rather than weather. A sealed tire
# loses ~1 PSI a month to permeation and about 1 PSI per 10°F of ambient swing,
# so anything past 1.5 PSI/day is air leaving through something. This is the one
# threshold that can flag a puncture while the pressure is still in band.
TIRE_TREND_LEAK_PSI_24H = -1.5

# Sensor battery. TPMS cells are 5-10 year lithium units that fall off a cliff
# rather than fading, so this is "book the replacement", not "urgent".
TIRE_BATTERY_LOW_PCT = 20.0

# A reading older than this is not a reading. Direct TPMS sensors report every
# 30-60 s while rolling and go to sleep when parked, so this has to be long
# enough to survive a set of traffic lights and short enough that a receiver
# that died ten minutes ago is not still being believed.
TIRE_STALE_AFTER_S = 180.0

# How often the dashboard asks. Sent to the browser in the /vehicle/tires
# payload rather than written into the JavaScript, so this is the only place it
# exists. Well under TIRE_STALE_AFTER_S so a tire goes stale on screen within
# one poll of going stale in fact.
TIRE_POLL_MS = 5000

# Which mock scenario a fresh process starts in. Dev only — a real provider has
# exactly one scenario, which is whatever the tires are actually doing.
TIRE_DEFAULT_SCENARIO = "all_normal"
