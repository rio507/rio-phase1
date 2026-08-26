import os

# RIO prompts now sourced from rio_prompts.py (compiled from behavior bible v1)
from rio_prompts import RIO_SYSTEM_PROMPT

# ---------------------------------------------------------------------------
# Models
# ---------------------------------------------------------------------------
# Every model id RIO uses is named here and overridable from the environment,
# so swapping one is a config change and not a search through the codebase.
#
#   REALTIME    RIO herself, live: her ears, her brain and her voice in one
#               speech-to-speech session. The driver talks, she talks back, and
#               either can interrupt the other.
#   REASONING   the deeper, slower one, reached only as a TOOL the realtime
#               model calls when a question needs research or careful work.
#               Never a second voice — RIO speaks its result in her own.
#   CHAT        the text conversation path (/talk, /ask), which is what answers
#               when the live session is not running, and what the visual
#               question path is built on.
#   STT         Whisper. Kept deliberately: every transcript consumer outside
#               the live loop — the session log, /last_talk, the router, the
#               visual pipeline — reads what Whisper produced, and the live
#               session is configured to transcribe with the same model so a
#               drive's transcripts come from one place.
OPENAI_REALTIME_MODEL = os.getenv("OPENAI_REALTIME_MODEL", "gpt-realtime-2.1")
OPENAI_REASONING_MODEL = os.getenv("OPENAI_REASONING_MODEL", "gpt-5.6-sol")
OPENAI_CHAT_MODEL = "gpt-5.5"
OPENAI_STT_MODEL = "whisper-1"

# cedar or marin. Config, not code: it is the single most noticeable thing
# about RIO and the one most likely to be argued about.
OPENAI_REALTIME_VOICE = os.getenv("OPENAI_REALTIME_VOICE", "cedar")

# --- deterministic speech through the live voice ---------------------------
# Warnings, health announcements and turn instructions dictated to the live
# session instead of synthesised separately. Off returns every one of them to
# ElevenLabs, which is a supported configuration and not a degraded one.
REALTIME_SPEECH_ENABLED = True

# How long a dictated line may take to START before RIO gives up on the live
# session and says it the other way.
#
# MEASURED on this stack, across two sessions: dictation reaches first audio in
# 390-585 ms, ElevenLabs in 145-172 ms, a pre-rendered clip in 0. So the budget
# sits above the observed worst case with room, and below a second: a false bail
# costs a line in the fallback voice, while a budget that is too generous costs
# the driver the delay AND the fallback on top of it.
#
# The lines where this would actually matter are not dictated at all — the red
# tier and the tire fast path play local clips.
REALTIME_SPEAK_TIMEOUT_MS = 900

# Channels dictated to the live voice. Per-channel because they are not the
# same kind of speech: a turn instruction six seconds out and a gap warning
# that is already late have completely different tolerance for a few hundred
# milliseconds.
#
# The most time-critical lines are not in here at all and never will be: the
# red-tier headway warnings and the two tire fast-path lines play PRE-RENDERED
# CLIPS from static/audio/, with no network in the path (docs/warning_logic_v2).
# Those clips are rendered in the live voice, which is what makes "one voice
# everywhere" true rather than approximately true.
REALTIME_SPEECH_CHANNELS = {"nav": True, "health": True, "headway": True}

# The live session is the conversation path when it is available. Turning this
# off returns RIO to hold-to-talk through Whisper and ElevenLabs, which is not
# a degraded mode so much as the previous one — every other voice on the page
# (headway, health, navigation) is unaffected either way, because none of them
# has ever gone near a conversation model.
REALTIME_ENABLED = True

# How long RIO waits for the reasoning model before carrying on without it.
# Sol answers a plain question in ~4 s and a web-search one in ~6 s; past this
# the driver has been listening to silence for too long, and an answer she
# gives from what she already knows is better than a better answer that arrives
# after the exit.
REALTIME_TOOL_TIMEOUT_S = 25.0
# Let the reasoning model search when the question needs current information.
REALTIME_WEB_SEARCH = True
# A spoken answer is not a document. This bounds how long RIO can talk for.
REALTIME_TOOL_MAX_OUTPUT_TOKENS = 3000

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

# ElevenLabs: RIO's fallback voice, and nothing else now.
#
# One voice everywhere was the goal, and two voices is what you get if the
# warnings keep their own synthesiser: conversation in one, alerts in another,
# with the alerts — the lines that matter most — sounding like a different
# system. So deterministic speech is DICTATED to the live session, word for
# word, and the pre-rendered clips are rendered in the same voice.
#
# ElevenLabs stays wired up, complete, and off the active path. It is what
# speaks when the live session is not there: no session open and one cannot be
# started, the dictation timing out, the model refusing. That is not a
# hypothetical — a car drives through tunnels — and it is the reason this is a
# fallback rather than a deletion.
# RIO's ACTIVE voice: the live session. Deterministic lines are dictated to it
# and the pre-rendered clips are rendered in it.
VOICE_BACKEND = "realtime"
# ...and the voice that speaks when the live one cannot. Every server-side TTS
# endpoint (/nav/voice, /headway_voice, /vehicle/health/voice) is now, by
# definition, the fallback path: the browser only reaches for it when dictation
# was not possible or did not start in time.
VOICE_FALLBACK_BACKEND = "elevenlabs"
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

# Loss over 24 hours that is not a leak any more. TIRE_TREND_LEAK_PSI_24H catches
# a puncture on the day it happens; this catches one that will not last the
# journey — 6 PSI a day is a tire that will be flat before tomorrow morning, and
# it is the difference between "check it soon" and RIO saying something out loud.
TIRE_RAPID_LOSS_PSI_24H = -6.0

# Below this the tire is not under-inflated, it is failing. A 15 PSI tire at road
# speed is running on its sidewall and building heat faster than it can shed it;
# the distinction from TIRE_PRESSURE_CRITICAL_DELTA exists so the words RIO uses
# can be "pull over" rather than "worth stopping".
TIRE_BLOWOUT_PSI = 15.0

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


# ---------------------------------------------------------------------------
# Vehicle telemetry — the Holley sensor set and everything judged against it
# ---------------------------------------------------------------------------
# Same rule as the tire block above and for the same reason: every band, every
# window and every glyph threshold the Vehicle Health column reacts to lives
# here. telemetry.py reads them and hands the browser finished strings. There is
# not one number below that also exists in static/rio_vehicle.js.

# Which TelemetryProvider is live. "mock" until there is an ECU to talk to; the
# real one is a Holley serial/CAN reader and lands beside MockHolleyProvider
# without anything above this line changing.
TELEMETRY_PROVIDER = "mock"
TELEMETRY_DEFAULT_SCENARIO = "normal_idle"

# How often the dashboard asks for telemetry. 1 s, an order of magnitude faster
# than the tire poll: these are engine channels and a coolant needle that steps
# once every five seconds does not read as live. Insights move far more slowly
# and get their own, lazier cadence — an event log that repaints at 1 Hz is a
# log nobody can read a line of.
TELEMETRY_POLL_MS = 1000
INSIGHTS_POLL_MS = 15000

# Above this the engine is turning under its own power. This is the gate on
# every band that is only meaningful on a running engine: oil pressure is 0 PSI
# at key-on and that is correct, not critical, and a panel that shouts about it
# is a panel the driver switches off. 400 rpm sits above a healthy crank
# (~200-300) and well under any idle.
TELEMETRY_ENGINE_RUNNING_RPM = 400.0

# --- trend ------------------------------------------------------------------
# Trend is the slope of a least-squares fit over this window, not a comparison
# with the previous sample. Same lesson as the headway warnings: one-sample
# differencing on a noisy channel produces an arrow that flickers up and down
# every poll and means nothing. 20 s at a 1 s poll is 20 points — long enough
# that sensor wander averages out, short enough that a coolant temperature
# genuinely climbing shows an up arrow within half a minute.
TELEMETRY_TREND_WINDOW_S = 20.0
# Below this many samples in the window there is no fit worth doing and the row
# shows a flat dash rather than guessing.
TELEMETRY_TREND_MIN_SAMPLES = 5
# How much a channel must move ACROSS THE WHOLE WINDOW before the arrow leaves
# stable, in that channel's own units. Set per sensor because 0.1 is nothing to
# a coolant temperature and a great deal to a battery voltage. Anything absent
# here never shows a direction, only stable.
TELEMETRY_TREND_DELTA = {
    "battery_voltage": 0.12,
    "rpm": 90.0,
    "coolant_temp": 1.5,
    "intake_air_temp": 1.5,
    "map_kpa": 4.0,
    "maf_gs": 3.0,
    "throttle_pct": 4.0,
    "engine_load": 5.0,
    # Tighter than the other percentages on purpose. A long-term fuel trim
    # that has moved 1.5% in twenty seconds is not drifting, it is being
    # driven somewhere — and the arrow is the only place on the row that
    # says which.
    "stft_b1": 2.5,
    "ltft_b1": 1.5,
    "afr_target": 0.25,
    "afr_wideband": 0.25,
    "fuel_pressure": 1.2,
    "oil_pressure": 2.5,
    "oil_temp": 1.5,
    "vehicle_speed": 3.0,
    "tire_pressure": 0.35,
    "tire_temp": 3.0,
}

# --- bands ------------------------------------------------------------------
# Four optional edges per channel. None means "this channel has no limit in that
# direction", which is the honest answer for most of them: there is no such
# thing as too little intake air temperature.
#
# `running` marks a band that is only applied when the engine is turning. Oil
# and fuel pressure and both AFR channels read zero or garbage on a stopped
# engine, and judging them then would fill the panel with faults that are just
# the key being off.
TELEMETRY_BANDS = {
    # Charging system. 13.2 is the floor of a working alternator at idle and
    # 12.4 is a battery being drained rather than charged; 15.2 is an
    # overcharge that boils electrolyte and kills electronics.
    "battery_voltage": {"crit_low": 12.4, "warn_low": 13.2,
                        "warn_high": 15.0, "crit_high": 15.2, "running": True},
    # A small-block's redline. Warn a little under it because the number that
    # matters to the driver is the one before the valves float.
    "rpm":             {"warn_high": 6000.0, "crit_high": 6400.0},
    # 225°F is where a 50/50 mix under a 16 psi cap is still fine but has no
    # margin left; 240 is where it boils and the head gasket is next.
    "coolant_temp":    {"warn_high": 225.0, "crit_high": 240.0},
    # Charge air. 160°F is heat-soak that costs real power and pulls timing;
    # 200 means the intake is cooking on a stopped car.
    "intake_air_temp": {"warn_high": 160.0, "crit_high": 200.0},
    "map_kpa":         {},
    "maf_gs":          {},
    "throttle_pct":    {},
    # Load is context, not a fault. An engine at 95% load is working hard,
    # which is what engines are for.
    "engine_load":     {},
    # Fuel trim is the one channel on this panel where the SIGN carries as
    # much as the magnitude, and the band is symmetric because both
    # directions are real faults: positive is the ECU adding fuel to cover
    # air it did not meter, negative is it pulling fuel back from a leaking
    # injector or a failing MAF. +/-10% is the number a technician reaches
    # for; +/-25% is where the ECU runs out of authority and sets a code of
    # its own, which is the moment RIO stops being early and starts merely
    # agreeing with the dashboard.
    "stft_b1":         {"crit_low": -25.0, "warn_low": -10.0,
                        "warn_high": 10.0, "crit_high": 25.0, "running": True},
    "ltft_b1":         {"crit_low": -25.0, "warn_low": -10.0,
                        "warn_high": 10.0, "crit_high": 25.0, "running": True},
    # Lean is what kills pistons; rich only wastes fuel. Hence the asymmetry:
    # a tenth of a point lean of 15.2 gets attention, and it takes 11.0 the
    # other way to say anything at all.
    "afr_wideband":    {"crit_low": 10.5, "warn_low": 11.0,
                        "warn_high": 15.2, "crit_high": 16.0, "running": True},
    "afr_target":      {},
    # A Terminator X Stealth returnless system holds ~58 psi. Under 48 the
    # injectors stop flowing what the table thinks they flow; under 40 it is
    # leaning out under load, which is the dangerous direction.
    "fuel_pressure":   {"crit_low": 40.0, "warn_low": 48.0,
                        "warn_high": 72.0, "running": True},
    # The old rule is 10 psi per 1000 rpm. This panel does not know the rpm
    # when it judges the pressure, so these are the idle-safe absolutes: under
    # 20 psi hot is a bearing problem, under 12 is an engine about to stop
    # being an engine. Anything subtler than that — pressure a few psi below
    # where it normally sits at this rpm — is the insight engine's job, and it
    # catches it weeks earlier than a band ever could.
    "oil_pressure":    {"crit_low": 12.0, "warn_low": 20.0,
                        "warn_high": 85.0, "running": True},
    # Oil stops being oil somewhere past 260°F.
    "oil_temp":        {"warn_high": 250.0, "crit_high": 275.0},
    "vehicle_speed":   {},
}

# --- contextual modes -------------------------------------------------------
# Not every state is a judgement. An engine at 900 rpm is not "normal", it is
# idling, and a car at 0 mph is stopped — the spec asks for exactly those words
# and they carry no severity at all. Each entry is (label, low, high); either
# edge may be None. First match wins, and a mode only applies when the channel
# is otherwise NORMAL: an actual fault always outranks a description.
TELEMETRY_MODES = {
    "rpm":           [("CRANKING", 1.0, 400.0), ("IDLE", 400.0, 1100.0)],
    "throttle_pct":  [("IDLE", None, 3.0)],
    "vehicle_speed": [("STOPPED", None, 0.5)],
    # Below the thermostat the engine has not finished warming up. Saying
    # NORMAL there would be true and useless; saying WARMING is what a driver
    # actually wants to know in the first two minutes.
    "coolant_temp":  [("WARMING", None, 170.0)],
    "oil_temp":      [("WARMING", None, 160.0)],
    # The alternator is charging. Blue, not amber — this is the healthy state,
    # it is just worth naming.
    "battery_voltage": [("CHARGING", 13.6, 15.0)],
}

# A reading older than this is not a reading. Far tighter than the tire
# equivalent because an ECU that has stopped answering has stopped answering
# now, not in three minutes.
TELEMETRY_STALE_AFTER_S = 6.0


# ---------------------------------------------------------------------------
# Vehicle insights — the predictive layer
# ---------------------------------------------------------------------------
# Display only, and deliberately so. Nothing in insights.py touches the speech
# arbiter and nothing raises a warning: a predictive observation is a line in a
# log the driver reads when parked, not something that interrupts them. The
# firewall that keeps the tire column silent applies here unchanged.

INSIGHTS_ENABLED = True

# Where the history that makes "yesterday" a real word lives. Beside the session
# logs, because it is the same kind of thing: a record of drives that outlives
# the process.
INSIGHTS_DIR = "/workspace/rio-phase1/training_data/vehicle"

# How much log to keep, and how much of it to send. The file is trimmed to the
# first number on write; the panel gets the second.
INSIGHTS_MAX_ENTRIES = 400
INSIGHTS_FEED_LIMIT = 40

# How long a channel must be sampled before that day counts as a day. At the
# 1 Hz telemetry cadence this is five minutes of engine running.
#
# It is deliberately not a token number. A day that qualifies on twenty seconds
# of data is a day of a cold engine cranking, and it lands in the baseline
# weighing the same as an hour on the motorway — every deviation and drift
# figure downstream inherits that lie. Measured: with this at 20, two minutes of
# the warm-up scenario was enough to produce "Coolant Temp is consistently 32°F
# below your normal baseline", which was arithmetically true and completely
# false about the car.
INSIGHTS_MIN_SAMPLES_PER_DAY = 300

# Deviation: how far today's running mean has to sit from the historical
# baseline before it is worth a sentence, per channel and in that channel's
# units. These are "a mechanic would notice" numbers, not "a sensor moved" ones.
INSIGHTS_DEVIATION_DELTA = {
    "battery_voltage": 0.35,
    "coolant_temp": 6.0,
    "oil_pressure": 6.0,
    "oil_temp": 10.0,
    "fuel_pressure": 4.0,
    "intake_air_temp": 15.0,
    "afr_wideband": 0.4,
    "ltft_b1": 3.0,
}

# Drift: slope across the daily baselines, expressed as total change over the
# window. This is the detector the whole feature exists for — it is what says
# a battery has been sliding for three weeks while every gauge still reads
# normal, which is the difference between a booking and a tow.
INSIGHTS_DRIFT_WINDOW_DAYS = 28
INSIGHTS_DRIFT_MIN_DAYS = 6
INSIGHTS_DRIFT_DELTA = {
    "battery_voltage": 0.30,
    "coolant_temp": 5.0,
    "oil_pressure": 5.0,
    "fuel_pressure": 3.5,
    # The drift detector's best channel. A long-term trim that has climbed
    # 2.5% across four weeks is a vacuum leak developing, and it passes
    # every band in TELEMETRY_BANDS on every single one of those days.
    "ltft_b1": 2.5,
}

# The same observation must not be able to fill the log. One per kind per
# channel per hour; the nominal "everything is fine" heartbeat far less often
# than that, since it is the least interesting line in the file.
INSIGHTS_COOLDOWN_S = 3600.0
INSIGHTS_NOMINAL_COOLDOWN_S = 10800.0

# Seed a demo history on an empty install: five days of plausible entries and
# four weeks of daily baselines, every one of them flagged `seeded` in the
# payload and labelled as such on screen. A demo that silently fabricates
# history it presents as measured is the one thing this layer must never do.
INSIGHTS_SEED_DEMO = True


# ---------------------------------------------------------------------------
# Vehicle health — the conversation layer and the announcement channel
# ---------------------------------------------------------------------------
# What lives here and what deliberately does not.
#
# HERE: the cadences and the plumbing — how often the browser asks whether RIO
# has something to say, and how much of the health picture a conversation turn
# is allowed to cost.
#
# NOT HERE: the severity threshold that makes RIO speak, the cooldowns, and the
# words. Those are in vehicle_health_policy.py's PROVISIONAL block, for the same
# reason headway/live_policy.py keeps its own: that module imports NOTHING, and a
# `import config` in it would be a hole in the firewall the whole design rests
# on. Everything in it is a module constant and is tuned there.

VEHICLE_HEALTH_ENABLED = True

# How often the browser asks the server whether a critical announcement is due.
# Sent down in the payload rather than written into the JavaScript, like every
# other cadence in this codebase. This is also the policy's tick rate: the
# server decides, the client only speaks, so nothing happens between polls.
#
# 3 s, not 1 s: the tire poll is already 5 s and a pressure that crosses a
# threshold is not a millisecond-critical event the way a closing gap is. The
# thing that must be fast is the arbiter cutting in once the decision is made,
# and that is client-side and immediate.
HEALTH_POLL_MS = 3000

# Speed above which a tire sensor going quiet stops being a maintenance note and
# becomes something RIO says out loud. A sensor that drops out in the driveway is
# a dead battery; one that drops out at 60 mph is a corner of the car nobody can
# see any more, on the one channel where the failure mode is a blowout.
HEALTH_DRIVING_MPH = 5.0

# How many issues the full health context may carry into a conversation turn.
# A driver who asks "is anything wrong" wants the answer, not a fault log, and
# the issue list is already sorted worst-first — so the cut falls on the ones
# that were never going to be mentioned.
HEALTH_MAX_ISSUES = 6


# ---------------------------------------------------------------------------
# TPMS radio behaviour — what the mock has to imitate to be worth testing against
# ---------------------------------------------------------------------------
# The tire mock used to hand back four perfect readings on every call, stamped
# with the instant it was asked. Nothing that consumes it noticed, because
# nothing consumed it but a dashboard that repaints every five seconds.
#
# A diagnostic monitor notices immediately. Every enabling condition it has --
# "enough valid samples", "comparable thermal state", "a report actually
# arrived" -- is meaningless against a stream that answers instantly, always,
# with a fresh number. A monitor tuned on that stream would confirm faults in
# four polls and then never fire on real hardware, which reports twice a minute,
# sleeps in the driveway, and lies for the first second after it wakes up.
#
# So these describe a direct TPMS sensor as one actually behaves. They are what
# the mock is measured against, not what it is convenient for the mock to do.

# How often a rolling sensor transmits. Real direct TPMS sensors send every
# 30-60 s while the wheel is turning, and the spread matters: four sensors that
# reported in lockstep would let a monitor compare four corners at the same
# instant, which is exactly the luxury real hardware does not give you.
TIRE_REPORT_INTERVAL_S = 45.0
TIRE_REPORT_JITTER_S = 12.0

# Sensors sleep when the wheel stops, to make a 5-10 year battery last. This is
# the single most important behaviour in this block: it means "no reading" is
# the NORMAL state of a parked car, and any monitor that treats silence as a
# fault will scream every night. Motion is what wakes them.
TIRE_SLEEP_AFTER_PARKED_S = 900.0

# The first reports after a wake-up are junk. The sensor has just powered its
# radio and its ADC from cold, and the first one or two frames carry pressures
# that can be tens of PSI out. Every real receiver discards them; ours has to
# know they exist in order to discard them.
TIRE_WAKE_JUNK_REPORTS = 2

# Fraction of transmissions that simply do not arrive. 433 MHz through a wheel
# arch, a steel rim and a moving car: 3% is a good receiver on a good day, and a
# monitor that requires N consecutive reports without allowing for this will
# never reach READY.
TIRE_PACKET_LOSS_FRAC = 0.03

# A direct TPMS sensor watches its own pressure between transmissions and
# switches to a fast mode when it moves quickly. This is why a blowout is
# detectable at all: at the nominal 45 s interval the tire would be flat before
# the second report. A monitor tuned against a mock without this would be tuned
# against a limitation the hardware does not have.
TIRE_FAST_MODE_PSI = 3.0
TIRE_FAST_MODE_INTERVAL_S = 6.0

# A sensor whose cell is dying does not go quiet, it goes erratic -- long gaps,
# then a burst, with values that wander. Distinguishing that from a tire that is
# actually losing air is the plausibility monitor's whole job.
TIRE_DYING_BATTERY_PCT = 4.0
TIRE_DYING_LOSS_FRAC = 0.45
TIRE_DYING_NOISE_PSI = 2.5

# An impossible step between consecutive reports from one sensor. A tire cannot
# gain or lose this much in a minute except by being inflated or destroyed, and
# both of those have their own monitors -- so a step this large in ONE report,
# with no supporting evidence, is a bad packet and is rejected as a measurement.
TIRE_IMPLAUSIBLE_STEP_PSI = 8.0
# Outside this, the number is not a tire pressure at all.
TIRE_PLAUSIBLE_RANGE_PSI = (2.0, 70.0)


# ---------------------------------------------------------------------------
# Tire diagnostic monitors (tire_diag/) — OBD-inspired, not OBD-II
# ---------------------------------------------------------------------------
# RIO Tire Health is not an OBD-II system and emits no SAE powertrain codes.
# What is borrowed is the discipline: a monitor runs only under valid
# conditions, one bad reading makes a PENDING fault and never a confirmed one,
# and a problem is repaired only after passing verification.
#
# These are the knobs that decide how much evidence is enough. They are
# PROVISIONAL: nothing here has seen a real drive, which is exactly why every
# monitor ships in shadow mode. Read the shadow logs, then tune these, then
# consider letting one speak.

TIRE_DIAG_ENABLED = True

# Where the diagnostic record lives, beside the insight baselines. Diagnostic
# history has to survive a restart -- an issue that vanishes when the process
# does is not a diagnostic system, it is a status light.
TIRE_DIAG_DIR = "/workspace/rio-phase1/training_data/vehicle"
# Append-only. Trimmed only by age, never by "the problem went away".
TIRE_DIAG_MAX_EVENTS = 4000
# A resolved issue stays queryable this long, so "has this happened before on
# this tire" has an answer. Recurrence is the whole reason to keep it.
TIRE_DIAG_RESOLVED_RETAIN_DAYS = 180.0

# --- what counts as a sample -----------------------------------------------
# A report older than this is not evidence of anything current. Shorter than
# TIRE_STALE_AFTER_S because a monitor needs a stricter bar than a display: the
# panel showing a two-minute-old number is fine, a leak monitor fitting a trend
# through one is not.
TIRE_DIAG_SAMPLE_MAX_AGE_S = 150.0
# How long a corner may go without a report, while moving, before the
# connectivity monitor starts counting misses. Three missed transmissions at the
# nominal interval -- enough to ride out the packet loss the radio really has.
TIRE_DIAG_MISSED_REPORT_S = 150.0

# --- run pacing ------------------------------------------------------------
# A monitor run needs NEW evidence. Two runs off the same sample are one run
# that was counted twice, and confirmation counts would then be a measure of
# poll rate rather than of persistence.
TIRE_DIAG_MIN_RUN_SPACING_S = 20.0

# --- thermal comparability -------------------------------------------------
# Tire pressure moves about 1 PSI per 10°F. Comparing a warm motorway reading
# with yesterday's cold parked one produces a 4 PSI "loss" that is entirely
# thermal, and a leak monitor that does it will find a leak in every tire on
# every car on the first cold night of the year. Two samples are comparable when
# their temperatures are within this.
TIRE_DIAG_COMPARABLE_TEMP_F = 12.0

# --- slow leak -------------------------------------------------------------
# Decline that counts as evidence, across thermally comparable samples.
TIRE_DIAG_LEAK_PSI = 1.2
# ...and how much more than the peers it has to be. This is what separates a
# leak from weather: four tires down 4 PSI on a cold morning is the air outside,
# one tire down 4 PSI while its peers held is the air inside.
TIRE_DIAG_LEAK_PEER_MARGIN_PSI = 0.9
# Over at least this long. A leak is a rate, and a rate needs a baseline.
TIRE_DIAG_LEAK_WINDOW_S = 1800.0
TIRE_DIAG_LEAK_MIN_SAMPLES = 4

# --- asymmetric loss -------------------------------------------------------
# One corner against its axle peer, which shares load, road and weather. A
# difference this large between them is about the tire, not the day.
TIRE_DIAG_ASYM_MARGIN_PSI = 1.5
TIRE_DIAG_ASYM_WINDOW_S = 1200.0
TIRE_DIAG_ASYM_MIN_SAMPLES = 3

# --- critical pressure -----------------------------------------------------
# An absolute floor, independent of the placard target. Below this the sidewall
# is carrying load it was not built for whatever the target says, and a car with
# a 28 PSI placard is in as much trouble at 15 PSI as one with 35.
TIRE_DIAG_CRITICAL_FLOOR_PSI = 18.0
# Falling, for the urgent path: this much down between consecutive validated
# reports. A tire that is critically low AND still going is a different problem
# from one that has been low since Tuesday.
TIRE_DIAG_FALLING_PSI = 0.8

# --- inflation -------------------------------------------------------------
# A step up this large is somebody with an airline, not a tire warming up.
# Recognising it matters as much as recognising a leak: it is how a pressure
# issue gets verified as repaired rather than quietly healed for the wrong
# reason.
TIRE_DIAG_INFLATION_STEP_PSI = 2.0

# --- plausibility ----------------------------------------------------------
# How many implausible samples in the window before the SENSOR is the suspect
# rather than the packet. One malformed frame is a radio; four is hardware.
TIRE_DIAG_IMPLAUSIBLE_COUNT = 4
TIRE_DIAG_IMPLAUSIBLE_WINDOW_S = 900.0

# --- confirmation and healing ----------------------------------------------
# Qualifying monitor runs before a CANDIDATE becomes ACTIVE. Per monitor,
# because the consequence of being wrong is not the same for a slow leak as for
# a flat tire.
TIRE_DIAG_CONFIRM_RUNS = {
    "tire.low_pressure": 2,
    "tire.critical_low_pressure": 2,
    "tire.slow_leak": 3,
    "tire.asymmetric_loss": 2,
    "tpms.sensor_connectivity": 2,
    "tpms.sensor_plausibility": 2,
    "tire.sensor_loss_during_decline": 1,   # one-trip; the gates are elsewhere
    "tire.inflation_event": 1,              # an observation, not a fault
    "tpms.receiver_health": 2,
}

# Drive cycles required on top of the run count. Mostly zero, deliberately: OBD
# waits for drive cycles because an emissions fault is never urgent, and a
# critically low tire that waited through three drives to be mentioned would be
# a design failure. Only the slow leak uses one, because a leak measured within
# a single drive is mostly measuring the drive.
TIRE_DIAG_CONFIRM_CYCLES = {
    "tire.slow_leak": 1,
}

# Passing runs before an ACTIVE issue is RESOLVED. Always more than one: a
# single good sample is how a warm tire on a motorway "fixes" a leak.
TIRE_DIAG_HEAL_RUNS = {
    "tire.low_pressure": 2,
    "tire.critical_low_pressure": 3,
    "tire.slow_leak": 2,
    "tire.asymmetric_loss": 2,
    "tpms.sensor_connectivity": 2,
    "tpms.sensor_plausibility": 3,
    "tire.sensor_loss_during_decline": 2,
    "tpms.receiver_health": 1,
}
# ...and how long the good behaviour has to hold.
TIRE_DIAG_HEAL_STABLE_S = {
    "tire.low_pressure": 300.0,
    "tire.critical_low_pressure": 600.0,
    "tire.slow_leak": 1800.0,
    "tire.asymmetric_loss": 900.0,
    "tpms.sensor_connectivity": 300.0,
    "tpms.sensor_plausibility": 600.0,
    "tire.sensor_loss_during_decline": 600.0,
    "tpms.receiver_health": 60.0,
}
# Recovery has to clear the threshold by this much before it counts as recovery
# at all, so a pressure hovering on the line cannot heal and re-fail forever.
TIRE_DIAG_HEAL_HYSTERESIS_PSI = 1.0

# --- drive cycles ----------------------------------------------------------
# A drive cycle starts when the car has been parked long enough for the previous
# one to have ended and then moves. These are built on the existing session
# infrastructure -- sessions.py already knows when a drive starts and ends --
# and these two thresholds only cover the case where nobody told us.
TIRE_DIAG_DRIVE_START_MPH = 5.0
TIRE_DIAG_DRIVE_END_PARKED_S = 300.0

# --- shadow mode -----------------------------------------------------------
# The master switch, over the per-code `speak` flags in tire_diag/codes.py.
# While this is True nothing these monitors find is ever spoken, whatever any
# individual code says -- the announcement RIO would have made is written to the
# shadow log instead. The urgent fast path is the documented exception.
TIRE_DIAG_SHADOW_MODE = True


# ---------------------------------------------------------------------------
# Vehicle data layer — canonical ingestion, gateways, and the engine domain
# ---------------------------------------------------------------------------
# The cloud side of the OBD-II / Holley work. Same rule as every block above:
# every threshold, cadence and limit the vehicle data layer reacts to lives
# here, and the bridge that will one day run in the car reads its own copy from
# its own config file rather than importing this one.
#
# NOT HERE, deliberately: anything the diagnostic framework in diag/ needs. That
# package imports no config at all — two domains cannot share one module-level
# constant, and a framework that reached for config would have to know which
# domain was asking. Domains pass their tunables in.

# Where the vehicle data layer keeps what it knows. Beside the insight baselines
# and the tire diagnostic record, because it is the same kind of thing: a record
# of drives that outlives the process.
VEHICLE_DIAG_DIR = "/workspace/rio-phase1/training_data/vehicle"

# The single vehicle this prototype watches. Every stateful thing in this
# codebase already assumes one driver and one car — the announcement policy, the
# diagnostic engines, nav's route registry, _last_talk — and this constant makes
# that assumption something you can read rather than something you discover.
# The API accepts a vehicle_id on every route so the contract is already
# multi-vehicle; the STATE behind it is not, and pretending otherwise would be
# the more expensive lie.
VEHICLE_ID = "vehicle_prototype_01"

# --- gateway registration and authentication --------------------------------
# The bootstrap key that admits a new gateway. From the environment, never from
# the source tree. UNSET MEANS REGISTRATION IS REFUSED: an unconfigured
# deployment that accepts any device is worse than one that accepts none,
# because the first failure is silent and the second is immediate.
VEHICLE_GATEWAY_REGISTRATION_KEY = os.getenv("RIO_GATEWAY_REGISTRATION_KEY", "")

# How long since a heartbeat before the cloud stops calling the link connected.
# Three missed beats at the 10 s active cadence in the bridge spec.
VEHICLE_GATEWAY_STALE_S = 35.0

# --- ingestion --------------------------------------------------------------
# Batches per minute per gateway, and how many may arrive at once. Sized in
# BATCHES rather than events on purpose: a bridge uploading a backlog after a
# tunnel sends few large batches, and a per-event limit would throttle exactly
# the recovery behaviour the outbox exists to produce.
VEHICLE_INGEST_RATE_PER_MIN = 240.0
VEHICLE_INGEST_BURST = 60.0

# Hard ceilings on one batch. A payload larger than this is refused whole rather
# than half-processed — a partially accepted batch is the one shape the outbox's
# retry logic cannot reason about.
VEHICLE_INGEST_MAX_EVENTS = 2000
VEHICLE_INGEST_MAX_BYTES = 4 * 1024 * 1024

# How many event ids are remembered for deduplication. At ten signals and a few
# hertz this is roughly the last hour of a drive, which comfortably covers a
# bridge retrying a batch it never saw acknowledged.
VEHICLE_INGEST_DEDUP_MAX = 20000

# How long an ingested reading stays current before the panel calls it stale.
# Matches TELEMETRY_STALE_AFTER_S: a channel arriving over a network is judged
# by the same clock as one arriving from a mock, or the two sources would
# disagree about what "live" means.
VEHICLE_INGEST_STALE_AFTER_S = TELEMETRY_STALE_AFTER_S

# The rolling window of raw canonical events kept in memory, for the early-fault
# snapshot that has to reach BACKWARDS from the moment a code appears. The trend
# ring in telemetry.py is 20 s and is cleared whenever a scenario changes, so it
# cannot answer "what was happening a minute before this code was set" — this
# can. Three minutes at ~12 channels and 1 Hz is a few thousand small dicts.
VEHICLE_EVENT_RING_S = 180.0
VEHICLE_EVENT_RING_MAX = 8000

# --- source selection -------------------------------------------------------
# Which producer the telemetry pipeline is listening to. The interpretation
# pipeline is identical for every one of them — see vehicle/__init__.py — and
# switching does not restart anything.
#
#   mock_holley   the in-process Holley mock, read directly. The original path.
#   simulation    the same physics, pushed through the canonical ingestion API.
#   live_obd      a bridge on a CAN OBD-II vehicle.
#   live_holley   a bridge listening passively to a Holley bus.
#   replay        a recorded canonical log, played back.
VEHICLE_SOURCE_DEFAULT = "mock_holley"

# --- powertrain diagnostic monitors ----------------------------------------
# The engine-domain equivalent of the TIRE_DIAG_* block above, and shadowed for
# the same reason with one difference that matters: the tire monitors have
# shadow logs from real drives behind them, and these have never seen a vehicle
# at all. That is why clearance is per domain now.
VEHICLE_DIAG_ENABLED = True
VEHICLE_DIAG_SHADOW_MODE = True

# --- diagnostic trouble codes ----------------------------------------------
# The append-only DTC record. Same shape and same reasoning as
# TIRE_DIAG_MAX_EVENTS: trimmed only by age, never because a code went away.
VEHICLE_DIAG_MAX_EVENTS = 4000

# The early-fault snapshot (§16.7). Sixty seconds either side of the moment a
# pending code first appears — the half BEFORE is the half no code reader can
# ever give you, and it is the reason vehicle/providers/ingested.py keeps a ring
# at all.
#
# The "after" half is captured later, when enough time has passed. A snapshot
# that waited for it before storing anything would lose the "before" half to a
# process restart in the intervening minute, which is exactly when it matters.
VEHICLE_DTC_SNAPSHOT_BEFORE_S = 60.0
VEHICLE_DTC_SNAPSHOT_AFTER_S = 60.0
VEHICLE_DTC_SNAPSHOT_MAX = 200

# Scan cadences (§16.4). Bounded and sequential — §13's bus etiquette is not
# optional, and a scheduler that asked for everything at once would be the
# fastest way to make a vehicle's own diagnostics unreliable.
VEHICLE_DTC_MIL_POLL_S = 30.0       # Mode 01 PID 01: lamp state and code count
VEHICLE_DTC_PENDING_POLL_S = 120.0  # Mode 07
VEHICLE_DTC_STORED_POLL_S = 300.0   # Mode 03
VEHICLE_DTC_PERMANENT_POLL_S = 0.0  # Mode 0A: drive start, report, drive end only


# ---------------------------------------------------------------------------
# Powertrain diagnostic monitors (powertrain_diag/) — instances of diag/
# ---------------------------------------------------------------------------
# The engine-domain equivalent of the TIRE_DIAG_* block, and it is short for a
# reason: the lifecycle, the healing, the freeze frames and the shadow machinery
# are all inherited from diag/, so what is left here is genuinely only "how much
# evidence is enough" for nine engine monitors.
#
# PROVISIONAL, and more so than the tire block. Those numbers have shadow logs
# from real drives behind them. These have never seen a vehicle at all — which
# is exactly why shadow clearance became per-domain.
#
# WHERE A LIMIT IS NOT HERE. The coolant ceiling, the charging floor and the
# fuel-trim limit are NOT repeated in this block: the monitors read them from
# TELEMETRY_BANDS, which is where the panel reads them. A monitor that held its
# own copy would disagree with the row above it the first time somebody tuned
# one of them, and a driver looking at an amber coolant row while RIO says
# nothing is the exact failure this whole convention exists to prevent.

# How long a channel's reading stays evidence. Far tighter than the tire
# equivalent, because an ECU that has stopped answering has stopped answering
# now, not in three minutes — the same reasoning as TELEMETRY_STALE_AFTER_S.
POWERTRAIN_SAMPLE_MAX_AGE_S = 30.0

# A monitor run needs new evidence, and engine channels arrive far faster than
# TPMS reports do.
POWERTRAIN_MIN_RUN_SPACING_S = 5.0

# --- coolant ---------------------------------------------------------------
# How long above the fixed ceiling before it is a finding rather than a spike.
# A momentary reading past the limit on one sample is a sensor; twenty seconds
# of it is an engine.
POWERTRAIN_COOLANT_LIMIT_HOLD_S = 20.0
# Rate of rise. A healthy engine warming up climbs fast and then stops; one that
# has lost coolant climbs at this rate and keeps going, and the difference is
# visible a long way before any ceiling.
POWERTRAIN_COOLANT_RISE_F_PER_MIN = 7.0
POWERTRAIN_COOLANT_RISE_WINDOW_S = 120.0
POWERTRAIN_COOLANT_RISE_MIN_SAMPLES = 8
# Contextual: how far above THIS car's own conditioned baseline counts. Smaller
# than any fixed band, because the whole point is to notice while everything
# still passes.
# ...measured over this window rather than the whole ring. A conditioned mean
# is a claim about how the car is running NOW; averaging in everything still
# in memory would smear a change across the moment it happened and delay the
# finding by exactly as long as the ring is deep.
POWERTRAIN_COOLANT_CONTEXT_WINDOW_S = 120.0
POWERTRAIN_COOLANT_CONTEXT_DELTA_F = 8.0
POWERTRAIN_COOLANT_CONTEXT_MIN_DAYS = 3

# --- charging --------------------------------------------------------------
# How long below the charging floor, engine running, before it is a finding.
POWERTRAIN_CHARGING_HOLD_S = 30.0
# Cranking voltage: the absolute floor, and how much decline across the recorded
# start history counts as a trend. The second is the interesting one — a battery
# losing capacity holds its running voltage perfectly and drops a little further
# every time the starter loads it.
POWERTRAIN_START_V_FLOOR = 9.0
POWERTRAIN_START_V_DECLINE_V = 0.6
POWERTRAIN_START_EVENTS_MIN = 4
POWERTRAIN_START_EVENTS_KEEP = 40

# --- fuel trim -------------------------------------------------------------
# How long a long-term trim has to sit past its band, warm and in closed loop,
# before it is a finding. Long, deliberately: LTFT moves slowly by design and a
# short window would just be measuring the drive.
POWERTRAIN_LTFT_HOLD_S = 120.0
POWERTRAIN_LTFT_MIN_SAMPLES = 10
# Below this coolant temperature the engine is not warm and its trims mean
# nothing yet.
POWERTRAIN_WARM_COOLANT_F = 170.0

# --- signal integrity ------------------------------------------------------
# A channel that has not moved AT ALL for this long, on an engine that is
# running, is stuck. The nastiest sensor failure there is: every plausibility
# check passes and the number is in band.
POWERTRAIN_FROZEN_S = 90.0
POWERTRAIN_FROZEN_MIN_SAMPLES = 12
# A step larger than this fraction of the channel's own plausible range, between
# consecutive samples, is a discontinuity no physical process produces.
POWERTRAIN_DISCONTINUITY_FRAC = 0.35

# --- connection ------------------------------------------------------------
# How long with no usable engine data at all before the link is the finding
# rather than the engine.
POWERTRAIN_NO_DATA_S = 30.0
# Outbox depth a bridge reports before it is worth saying something.
POWERTRAIN_OUTBOX_WARN = 500

# --- confirmation and healing ----------------------------------------------
POWERTRAIN_CONFIRM_RUNS = {
    "engine.new_dtc": 1,                 # the ECU already confirmed it
    "engine.coolant_hard_limit": 2,
    "engine.coolant_rate_of_rise": 2,
    "engine.coolant_contextual": 3,
    "engine.charging_voltage": 3,
    "engine.start_voltage_trend": 2,
    "engine.fuel_trim_long_term": 3,
    "engine.signal_integrity": 3,
    "engine.connection": 2,
}
POWERTRAIN_CONFIRM_CYCLES = {
    # Both of these are claims about how this car normally behaves, and a claim
    # like that measured inside a single drive is mostly measuring the drive.
    "engine.coolant_contextual": 1,
    "engine.fuel_trim_long_term": 1,
}
POWERTRAIN_HEAL_RUNS = {
    "engine.new_dtc": 2,
    "engine.coolant_hard_limit": 3,
    "engine.coolant_rate_of_rise": 3,
    "engine.coolant_contextual": 3,
    "engine.charging_voltage": 3,
    "engine.start_voltage_trend": 2,
    "engine.fuel_trim_long_term": 3,
    "engine.signal_integrity": 3,
    "engine.connection": 2,
}
POWERTRAIN_HEAL_STABLE_S = {
    "engine.new_dtc": 300.0,
    "engine.coolant_hard_limit": 300.0,
    "engine.coolant_rate_of_rise": 300.0,
    "engine.coolant_contextual": 1800.0,
    "engine.charging_voltage": 600.0,
    "engine.start_voltage_trend": 1800.0,
    "engine.fuel_trim_long_term": 1800.0,
    "engine.signal_integrity": 300.0,
    "engine.connection": 120.0,
}


# ---------------------------------------------------------------------------
# Contextual navigation (docs/navigation_v1.md)
# ---------------------------------------------------------------------------
# Every threshold navigation has is here. None of them is a magic number buried
# in the tracker, because every one of them is a claim about driving that will
# be wrong somewhere and has to be tunable when it is.
#
# The whole of navigation's timing is expressed in SECONDS TO THE MANEUVER, not
# metres. 200 m of downtown and 200 m of arterial are the same distance and
# completely different warnings; seconds are what a driver needs to act. The
# distance clamps below exist only to stop the seconds producing something
# absurd at the extremes.

NAV_ENABLED = True

# --- GPS health -------------------------------------------------------------
# GPS health and off-route are SEPARATE questions (§6). A stale fix means we do
# not know where the car is; it does not mean the car left the route, and it
# must never cause a reroute.
NAV_GPS_STALE_TIMEOUT_S = 5.0       # no fix for this long -> GPS_STALE
NAV_GPS_ACCURACY_LIMIT_M = 30.0     # worse than this -> GPS_DEGRADED
NAV_GPS_DEGRADED_BIAS_S = 2.0       # degraded near a maneuver: speak this much
                                    # EARLIER, never later (§6)

# --- off route --------------------------------------------------------------
# Distance from the polyline plus persistence, and nothing else (§7). No road
# network matching, no lane inference, no probabilistic road inference: those
# are the things that make an off-route detector confidently wrong.
NAV_OFF_ROUTE_DISTANCE_M = 45.0
NAV_OFF_ROUTE_PERSISTENCE = 3       # consecutive fixes beyond the distance
NAV_REROUTE_DEBOUNCE_S = 12.0       # floor between reroutes — anti-flap
NAV_REROUTE_MAX_PER_JOURNEY = 12    # hard stop on a reroute loop

# --- route progress ---------------------------------------------------------
# Progress is monotonic under noise: a fix that projects behind where we have
# already been is jitter, not a reversal, unless it is this far back. A genuine
# wrong turn shows up as off-route, not as rewind.
NAV_PROGRESS_REWIND_TOLERANCE_M = 30.0
NAV_MANEUVER_PASSED_EPS_M = 8.0     # this far past the point and it is behind you
NAV_ARRIVE_RADIUS_M = 25.0
NAV_PROJECTION_BACK_M = 80.0        # projection search window, behind
NAV_PROJECTION_FWD_M = 400.0        # ...and ahead

# --- heading fallback (§8) --------------------------------------------------
# Browser Geolocation on iOS frequently reports heading: null and speed: null.
# Heading is then derived from consecutive fixes — but only when the derivation
# means anything: fresh samples, real displacement, usable accuracy, actually
# moving.
NAV_HEADING_MIN_DISPLACEMENT_M = 8.0
NAV_HEADING_MAX_SAMPLE_AGE_S = 3.0
NAV_HEADING_MIN_SPEED_MS = 1.5
NAV_STATIONARY_SPEED_MS = 0.7

# --- speech windows (§12) ---------------------------------------------------
# Three OPPORTUNITIES, not three mandatory calls. RIO is a passenger who tells
# you about the turn, not a GPS that counts down to it.
NAV_EARLY_GUIDANCE_S = 25.0         # "Right turn coming up."      (optional)
NAV_ANCHOR_ACQUISITION_S = 11.0     # start looking for the landmark
NAV_CONTEXT_CALL_S = 6.0            # "Turn right by the Shell station."
NAV_NEAR_TURN_S = 2.5               # "Right here."                (only if needed)
# Clamps, so that a crawl does not announce a turn 4 m ahead and a fast road
# does not announce one from a kilometre out.
NAV_MIN_CALL_DISTANCE_M = 20.0
NAV_MAX_CALL_DISTANCE_M = 400.0
NAV_EARLY_MAX_DISTANCE_M = 900.0
# Below this, time-to-maneuver stops meaning anything: at 0.2 m/s every
# maneuver is hours away and nothing is ever said, including the turn being
# crept towards in traffic. A floor for the arithmetic, not a claimed speed.
NAV_SPEED_FLOOR_MS = 3.0
NAV_SPEED_NOMINAL_MS = 11.0         # only when there is no speed at all
NAV_DUPLICATE_INSTRUCTION_COOLDOWN_S = 8.0
# A speech candidate is true only inside a window. These are the windows.
NAV_SPEECH_TTL_S = {"early": 8.0, "primary": 5.0, "imminent": 2.5, "arrival": 8.0}

# --- landmark candidates (V1.1) ---------------------------------------------
# Fetched ONCE per route generation, one pass over the maneuvers at route load,
# cached for that generation's lifetime, refreshed only on reroute. Never
# per-frame, never on an interval — that is the difference between a place
# lookup and a place subscription, and only one of them is affordable.
NAV_LANDMARKS_ENABLED = True
NAV_LANDMARK_SEARCH_RADIUS_M = 90.0
NAV_LANDMARK_MAX_LOOKUPS_PER_ROUTE = 12     # hard budget cap (addendum)
NAV_LANDMARK_MAX_CANDIDATES_PER_MANEUVER = 4
NAV_LANDMARK_MAX_DISTANCE_M = 80.0          # further than this from the maneuver
                                            # and it is not "by" the turn
# Allowed anchor classes (§21). Branded fuel and major chain signage only:
# things with a large, standardised, permanently-lit sign that a driver reads
# without looking for it. Everything else is out of scope until this is
# reliable.
NAV_ANCHOR_TYPES = ("gas_station", "coffee_shop", "fast_food_restaurant",
                    "pharmacy", "convenience_store")
NAV_ANCHOR_BRANDS = {
    # brand key -> (spoken form, anchor class, salience 0-1)
    "shell":        ("the Shell station", "gas_station", 1.0),
    "chevron":      ("the Chevron station", "gas_station", 1.0),
    "mobil":        ("the Mobil station", "gas_station", 0.95),
    "exxon":        ("the Exxon station", "gas_station", 0.95),
    "76":           ("the 76 station", "gas_station", 0.9),
    "arco":         ("the Arco station", "gas_station", 0.9),
    "valero":       ("the Valero station", "gas_station", 0.85),
    "bp":           ("the BP station", "gas_station", 0.85),
    "starbucks":    ("the Starbucks", "coffee_shop", 0.9),
    "mcdonald's":   ("the McDonald's", "fast_food_restaurant", 0.95),
    "mcdonalds":    ("the McDonald's", "fast_food_restaurant", 0.95),
    "burger king":  ("the Burger King", "fast_food_restaurant", 0.85),
    "taco bell":    ("the Taco Bell", "fast_food_restaurant", 0.85),
    "cvs":          ("the CVS", "pharmacy", 0.85),
    "walgreens":    ("the Walgreens", "pharmacy", 0.85),
    "7-eleven":     ("the 7-Eleven", "convenience_store", 0.8),
}

# --- landmark relation, from MAP DATA (§16 + addendum) ----------------------
# turn_relation_to_anchor describes where the TURN is relative to the LANDMARK,
# and it is computed from coordinates — never estimated by the camera.
#   |along delta| <= NEAR band          -> NEAR        "turn left by the Shell"
#   landmark before the turn            -> JUST_AFTER  "turn right just after..."
#   landmark past the turn              -> JUST_BEFORE "turn left just before..."
# NEAR is the default because it needs the least spatial certainty. The other
# two are claims about ORDER, and a wrong one sends a driver through the
# junction, so they demand a much wider margin before they are allowed.
NAV_RELATION_NEAR_BAND_M = 22.0
NAV_RELATION_ORDERED_MIN_M = 30.0     # below this margin, degrade to NEAR
NAV_RELATION_ORDERED_MAX_M = 75.0     # beyond this the landmark is not "just" anything
NAV_RELATION_MAX_LATERAL_M = 45.0     # off-route offset that still counts as roadside

# How useful a class of landmark is to SAY, as distinct from how visible it is.
# A fuel brand is the most useful thing on a corner: enormous, lit, standardised
# nationally, and drivers already navigate by them. A convenience store is real
# but weaker — smaller sign, more of them, more easily confused with the next
# one along. Used only to rank candidates that have already passed every gate.
NAV_ANCHOR_TYPE_USEFULNESS = {
    "gas_station": 1.0,
    "pharmacy": 0.9,
    "coffee_shop": 0.85,
    "fast_food_restaurant": 0.85,
    "convenience_store": 0.75,
}

# --- anchor validation gates (§18) ------------------------------------------
# Hard gates, all of them, before anything is ranked. Frequent rejection is the
# design working: the fallback is ordinary navigation, which is fine, and a
# confidently wrong landmark is worse than no landmark.
NAV_ANCHOR_MIN_IDENTITY_CONFIDENCE = 0.75
NAV_ANCHOR_MIN_VISIBILITY_CONFIDENCE = 0.6
NAV_ANCHOR_MIN_RELATION_CONFIDENCE = 0.6
NAV_ANCHOR_ORDERED_MIN_RELATION_CONFIDENCE = 0.8   # JUST_BEFORE / JUST_AFTER
NAV_ANCHOR_MIN_TRACKING_DURATION_S = 1.2
NAV_ANCHOR_MIN_OBSERVATIONS = 2
NAV_ANCHOR_MAX_AGE_S = 3.0            # an observation older than this is history
NAV_ANCHOR_VALID_FOR_S = 6.0          # a VerifiedAnchor's own shelf life
NAV_ANCHOR_MAX_PER_MANEUVER = 1       # one anchor, ever (§19)

# --- visual verification (V1.1) ---------------------------------------------
# The camera answers ONE question: is this expected landmark clearly visible?
# It does not locate the turn, it does not compute intersection coordinates and
# it does not get a vote on the route.
NAV_VISION_ENABLED = True
NAV_VERIFY_MAX_FRAMES = 3             # observations per verification pass
NAV_VERIFY_FRAME_MAX_AGE_S = 4.0
NAV_VERIFY_MIN_SPACING_S = 0.4        # two reads of the same instant are one observation
NAV_VERIFY_DEPTH_ENABLED = True
NAV_VERIFY_DEPTH_MAX_M = 90.0         # a "landmark" reported 200 m out is not
                                      # the one 40 m from the maneuver
NAV_VERIFY_DEPTH_MIN_M = 3.0
# Depth Anything reports its own confidence per ROI (spread, valid pixels,
# range). Below this the reading is a number without a meaning — a box
# straddling a sign and the sky behind it — and the consistency check abstains
# rather than acting on it.
NAV_VERIFY_DEPTH_MIN_CONF = 0.35
