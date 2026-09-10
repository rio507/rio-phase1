import os
from pathlib import Path

from dotenv import load_dotenv

# Read here rather than only in app.py. Everything in this file that comes from
# the environment is read AT IMPORT, so a tool that imports config without
# having loaded .env first got an empty voice id and a default model — which
# looks exactly like a misconfiguration and is not one. Idempotent: app.py
# still calls it, and the first call wins.
load_dotenv(Path(__file__).resolve().parent / ".env")

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
#   STT         What the driver said, in writing. ONE model for the whole
#               system: the live session transcribes the cabin with it, and so
#               do the consumers outside the live loop — the session log,
#               /last_talk, the router, the visual pipeline, the clip
#               verifier. Two transcribers would make two records that
#               disagree, and the disagreement would only ever show up in a
#               drive nobody could reproduce.
OPENAI_REALTIME_MODEL = os.getenv("OPENAI_REALTIME_MODEL", "gpt-realtime-2.1")
OPENAI_REASONING_MODEL = os.getenv("OPENAI_REASONING_MODEL", "gpt-5.6-sol")
OPENAI_CHAT_MODEL = "gpt-5.5"
# gpt-transcribe, the model the realtime playground calls "User transcript
# model". Whisper is still accepted by both APIs and is one env var away.
#
# CHECKED ON THIS ACCOUNT, both halves, because "the realtime session takes it"
# and "the REST endpoint takes it" are two claims and only one of them was in
# doubt. The realtime mint echoes it back in
# session.audio.input.transcription.model; /v1/audio/transcriptions returns the
# same words Whisper returned for the same clip, in a third of the time
# (0.99 s against 3.36 s on a five-second file).
#
# ONE INCOMPATIBILITY, and it costs nothing here: response_format
# "verbose_json" is refused by every gpt-*-transcribe model ("Use 'json' or
# 'text' instead"). Nothing in this system asks for it — every caller takes the
# default json and reads `.text` — so this is a drop-in. If a caller ever needs
# segment timings back, that caller needs whisper-1 by name and this constant
# is the wrong thing to change.
OPENAI_STT_MODEL = os.getenv("OPENAI_STT_MODEL", "gpt-transcribe")

# cedar or marin. Config, not code: it is the single most noticeable thing
# about RIO and the one most likely to be argued about.
#
# THE DEFAULT IS THE VOICE, not a fallback to be corrected by .env. Everything
# that speaks names this constant — the live session, the dictated warnings,
# the pre-rendered clips, the mid-drive fallback — so a default that disagreed
# with the running config would be a second voice waiting for the day somebody
# starts a process without the environment.
OPENAI_REALTIME_VOICE = os.getenv("OPENAI_REALTIME_VOICE", "marin")

# --- deterministic speech through the live voice ---------------------------
# Warnings, health announcements and turn instructions dictated to the live
# session instead of synthesised separately. Off returns every one of them to
# ElevenLabs, which is a supported configuration and not a degraded one.
REALTIME_SPEECH_ENABLED = True

# How long a dictated line may take to START before RIO gives up on the live
# session and says it the other way.
#
# ONE NUMBER WAS THE WRONG SHAPE FOR THIS, and the drive that showed it is the
# navigating one: of eleven deterministic lines, five reached the speaker in
# marin and six timed out into the ElevenLabs fallback. Every miss was a
# `timeout`, not a busy mouth. So "one voice everywhere" was running at about
# half, and the cause was a budget derived from the single most urgent line
# being applied to every line.
#
# MEASURED, from `response.created` to the first audio, over a tools drive and
# a navigating drive (n=27):
#
#     min 418 ms   p50 ~700 ms   p90 ~1140 ms   max 1277 ms
#
# The old 900 ms came from an IDLE session, where the same injection lands in
# 390-585 ms. A drive is not idle: the realtime API serialises responses, so a
# turn call asked for while she is finishing a sentence — or while the previous
# turn call is still being spoken — waits behind it.
#
# WHAT THE TRADE ACTUALLY IS, because it is not "voice versus speed" and
# reading it that way gets the sign wrong on the one line that matters. With a
# budget B: a line that starts in time is heard at up to B, in her voice; a
# line that misses is heard at about B + 180 ms (the fallback's measured first
# byte), in the other one. So a LOOSER budget buys vocal consistency and
# lengthens the WORST CASE. A tighter budget bounds the worst case and spends
# the voice.
#
# Which of those two a channel wants depends entirely on what it is saying, and
# that is why the numbers below differ:
#
#   nav / imminent    "Left here." The backup call, AT the turn. This is the
#                     one line where the worst case is the whole point — late
#                     enough and the driver has passed it — so it keeps the
#                     tight budget and spends the voice to bound the delay.
#   nav / primary     "Take the next left onto 16th St." The instruction, and
#                     it is issued with room in front of the maneuver.
#   nav / early       "Right turn coming up onto Cloverfield Blvd." Seconds
#                     out. Nothing about this is time-critical.
#   nav / arrival     Ditto.
#   health            An announcement about the car. The genuinely urgent ones
#                     are not dictated at all: the two tire fast-path lines
#                     play pre-rendered clips with no network in the path.
#   headway           Only the CALM tier reaches dictation — coaching, not an
#                     alert; the red tier plays clips. Kept the shortest of the
#                     loosened budgets because its arbiter item carries a
#                     2500 ms TTL: a gap measured three seconds ago is not a
#                     gap, and B + fallback has to fit inside that.
#
# THE LINES WHERE THIS WOULD MATTER MOST ARE STILL NOT DICTATED AT ALL. The red
# headway tier and the tire fast path play local files, and no number here can
# make them late.
REALTIME_SPEAK_TIMEOUT_MS = 900

# Per channel, and for navigation per CALL TYPE, because "a turn call" is four
# different sentences with four different deadlines. `_default` covers a call
# type this table has not been taught about, so a new one is loose rather than
# accidentally urgent.
REALTIME_SPEAK_TIMEOUT_MS_BY_CHANNEL = {
    "nav": {
        "junction": 900,
        "near": 1500,
        "far": 2000,
        "far_mid": 2000,
        "depart": 2500,
        "arrival": 2000,
        "arrived": 2000,
        "_default": 1500,
    },
    "health": {"_default": 2000},
    "headway": {"_default": 1200},
}


def speak_timeout_ms(channel: str = None, call_type: str = None) -> int:
    """How long THIS line may take to start before RIO says it the other way.

    Falls back to REALTIME_SPEAK_TIMEOUT_MS for a channel the table does not
    name, which keeps the old behaviour for anything new rather than giving it
    the loosest budget in the file by accident.
    """
    by_call = REALTIME_SPEAK_TIMEOUT_MS_BY_CHANNEL.get(channel)
    if not by_call:
        return int(REALTIME_SPEAK_TIMEOUT_MS)
    return int(by_call.get(call_type or "_default",
                           by_call.get("_default", REALTIME_SPEAK_TIMEOUT_MS)))

# ...AND THE SAME QUESTION FOR A LINE THAT IS NOT A WARNING.
#
# A vetted scene answer is injected the same way a warning is, and the budget
# above is wrong for it — not because the mechanism differs but because the
# ARGUMENT does. 900 ms is derived from "a warning that arrives late has
# stopped being a warning", and a passenger answering "what's out there" a
# second and a half later has not stopped answering.
#
# MEASURED, and this is why it is a separate number rather than the same one:
# a warning dictated into an idle session reaches first audio in 390-585 ms,
# but a scene answer is injected immediately after a tool call, and on a real
# drive one took longer than 900 ms to make a sound. The budget fired, the line
# was cancelled and counted as a failure, an ordinary response was asked for as
# the recovery — and the audio then arrived anyway. She said the right thing,
# and everything watching her said it had gone wrong.
#
# The thing on the other side of this trade is not silence, it is the composed
# answer, which costs a further round trip and about 450 ms of model. So the
# budget sits well above the observed case and still below the point where
# giving up would have been the faster route to a sentence.
REALTIME_DIRECT_SPEECH_TIMEOUT_MS = 2500

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
# The camera, the route, the car and the places search: all of them answer in
# under a second or fail. Past this the driver has been listening to silence
# for too long, and an answer she gives from what she already knows is better
# than a better answer that arrives after the exit.
#
# deep_dive is NOT bound by this any more — see DEEP_ANSWER_TIMEOUT_S. It was,
# and the note here used to say a web-search answer took about six seconds,
# which measurement did not support: a news question with three searches took
# 25.4 s and would have been aborted one tenth of a second before it arrived.
REALTIME_TOOL_TIMEOUT_S = 25.0
# Let the reasoning model search when the question needs current information.
REALTIME_WEB_SEARCH = True
# A spoken answer is not a document. This bounds how long RIO can talk for.
REALTIME_TOOL_MAX_OUTPUT_TOKENS = 3000

# --- two tiers of answer ----------------------------------------------------
# A driver asking "what's that building?" wants a sentence, now. They do not
# want RIO to go away and think, and the thing that made her go away and think
# was that deep_dive was reachable from a question the camera had already
# answered. So the first answer to anything visual is the camera's, and the
# reasoning model is unlocked only by being asked for.
#
# The window: a visual question inside this many seconds of a look is a FIRST
# look, and deep_dive is refused for it. Sized as "the same exchange" rather
# than "recently" -- a minute later, the driver asking about a building is
# asking a fresh question, not still waiting on the last one.
DEPTH_COLD_S = 60.0

# Once the driver has asked for more, they can keep asking: a chain of
# follow-ups about the same thing is one conversation, not a series of cold
# questions to be refused one at a time.
DEPTH_WINDOW_S = 120.0

# What a spoken deep answer may run to. The reasoning model will happily write
# an essay, and an essay read aloud in a car is a monologue nobody can
# interrupt politely. Three or four sentences, then an offer to go on.
DEEP_ANSWER_MAX_TOKENS = 320

# ...AND WHAT IT MAY SPEND THINKING BEFORE IT WRITES THEM, because the API
# takes ONE number for both and this is the second time that has cost an
# answer. OPENAI_MAX_TOKENS below carries the same lesson from the chat path:
# a budget that covers reasoning AND output, set to the length of the reply,
# is a budget the reasoning pass can spend in full before saying anything.
#
# On deep_dive it is worse, because web_search reasons BETWEEN searches.
# "What happened in the Pacific Palisades fire?" spent 320 of 320 tokens on
# reasoning across two searches, came back `incomplete` with
# reason=max_output_tokens and no text at all, and RIO told the driver she
# could not look it up right now. Measured, the same question needs 168-530
# reasoning tokens depending on effort, and answers in 9-13 seconds.
#
# So the answer keeps its own ceiling and the thinking gets its own, and what
# goes to the API is the sum. 1,200 is roughly twice the worst case measured,
# which is the point: this number exists to be generous, and the one that
# keeps an answer short is the one above it.
DEEP_REASONING_MAX_TOKENS = 1200

# ...AND HOW LONG THE DRIVER WAITS FOR IT. Measured against the live API on
# news questions, which are the slow ones because each search is a round trip
# the model then reasons about:
#
#   no search at all ("why did carmakers switch to EPS")     2.1 s
#   one search, one fact ("who won the last World Cup")      4.2 s
#   two searches ("what changed in California EV rebates")  12.9 s
#   six searches ("latest on the fire recovery")            19.6 s
#   three searches ("what happened in the Palisades fire")  25.4 s
#
# The old ceiling was the 25 s every other tool shares, and the last row is
# what that costs: an answer that existed, was paid for, and was thrown away
# a tenth of a second before it arrived. This is roughly twice the worst case
# measured, and it is a ceiling rather than a target — the holding line in the
# instructions is what makes the wait tolerable, not this number.
DEEP_ANSWER_TIMEOUT_S = 45.0

# --- brevity, out loud ------------------------------------------------------
# The ceiling on any single spoken response, enforced at the API rather than
# asked for in the prompt: instructions are guidance and this is a limit. ~300
# tokens is roughly 35 seconds of speech, which is already long for a car and
# is meant to be unreachable in ordinary conversation rather than typical.
#
# RAISED FROM 200 BECAUSE IT WAS BEING REACHED. A question about the car is the
# one that runs long -- "how are my tires" is answered from a structure with
# several issues in it, and the honest answer names more than one -- and those
# came back at 156 and 184 output tokens in a straight measurement, with a live
# drive hitting the cap and stopping her mid-sentence. A ceiling that ordinary
# answers touch is not a ceiling, it is a length, and the driver hears it as
# her trailing off rather than as a limit doing its job.
#
# It costs the token budget, and that is accounted for rather than ignored:
# tools/realtime_selftest.py run_session_cost adds the cap to the per-response
# input floor and asserts that three tool turns a minute still fit inside the
# account's 40,000 even if every single answer runs the whole way to it.
#
# AND THE UNITS ARE THE MODALITY'S, which is the second time that has cost an
# answer -- REALTIME_LOOK_ANSWER_* below carries the same lesson from the same
# afternoon. `max_output_tokens` counts what the session PRODUCES. Under text
# mode that is words. Under speech to speech it is SOUND, and audio tokens
# dominate: measured over a real drive, audio costs a flat 20 tokens for every
# second of speech, and the transcript that comes with it costs more on top.
#
# So 300 is about 1,200 characters written and about 160-260 characters spoken,
# and on the six-turn audio drive four of six answers reached it and stopped
# mid-sentence:
#
#     "...so this is coming from the tire data that is"
#
# which is the driver hearing her trail off, and is precisely what the
# paragraph above says a ceiling must not do. Same fault as the 200 -> 300
# raise, arrived at by changing the modality rather than the number.
#
# 1,200 UNDER AUDIO, AND THE UNIT THAT PICKED IT IS SECONDS.
#
# Tokens per character was the wrong way to reason about this -- it varies from
# 1.25 to 3.6 depending mostly on how much fixed per-response overhead a short
# answer is carrying. Measured against the ACTUAL AUDIO instead (decoded from
# the deltas of a real drive, n=10), the picture is flat and obvious:
#
#     audio output is exactly 20 tokens per second of speech
#     total output, on answers long enough to matter, is 25-36 tokens/second
#
# So the ceiling in the only unit that survives a change of modality:
#
#     300 text tokens   ~35 seconds   (the text-mode number, and its intent)
#   1,200 audio tokens  ~34 seconds   (the same length, in the other currency)
#
# That is the whole derivation. The number changed because the units did; what
# a driver is allowed to sit through did not.
#
# Bounded from above as well, so it is not merely large enough:
# run_session_cost asserts three tool turns a minute fit at
# (floor + cap) * 2 * 3 <= 40,000. With today's floor that caps the cap at
# about 1,830, and 1,200 sits comfortably inside it rather than against it --
# 37,194 of 40,000 in the pessimistic reading where every answer runs the whole
# way to the ceiling.
REALTIME_MAX_RESPONSE_TEXT_TOKENS = 300
REALTIME_MAX_RESPONSE_AUDIO_TOKENS = 1200


def max_response_tokens() -> int:
    """The ceiling on one answer, in the units the live session is billed in.

    A function for the same reason look_answer_max_tokens() is: VOICE_BACKEND
    is decided further down this file, and a constant that had to be kept in
    step with the backend by hand is the shape of the bug it replaces.
    """
    return (REALTIME_MAX_RESPONSE_TEXT_TOKENS if VOICE_BACKEND == "elevenlabs"
            else REALTIME_MAX_RESPONSE_AUDIO_TOKENS)


# The old name. Kept because it is what the cut-off classification and two
# suites talk about, and because it is the number under the backend the
# comment above was originally written for.
REALTIME_MAX_RESPONSE_TOKENS = REALTIME_MAX_RESPONSE_TEXT_TOKENS

# What a spoken answer to "what do you see" may run to.
#
# MEASURED: on the current path those answers came back at 23 words at the
# median and 44 at p95, against instructions that ask for ONE short sentence.
# Instructions are guidance; this is a limit, which is the same argument
# REALTIME_MAX_RESPONSE_TOKENS already makes for answers in general.
#
# It matters more here than it did with the old voice. The whole answer is
# synthesised, so its length is time the driver spends listening to a caption
# being elaborated — and a scene answer is the one kind that has nothing to
# elaborate: the road looks how it looks, and the useful version is the
# sentence a passenger would say without being asked twice.
#
# TWO NUMBERS, BECAUSE A TOKEN IS NOT ONE THING.
# ----------------------------------------------
# The budget wanted here is "two short sentences", and how many tokens that
# costs depends entirely on what the session is producing.
#
# Under a TEXT-mode session the output is words, and 60 of them is two short
# sentences. That number was measured under that backend and is still right
# for it.
#
# Under a SPEECH-TO-SPEECH session the output is SOUND, and `max_output_tokens`
# counts audio tokens — which dominate. Measured over a real audio drive
# (tools/live_tool_turns, six turns): a spoken answer costs about 1.16 audio
# tokens per character, plus roughly 0.62 text tokens for each of those for the
# transcript that comes with it. So 60 tokens is not two sentences under audio,
# it is TWENTY-SIX CHARACTERS — and the drive that proved it has her answering
# "what kind of car is in front of us" with:
#
#     "Looks like a BMW 5 Series,"
#
# cut there, mid-clause, and filed as max_output_tokens. Which is precisely the
# failure REALTIME_MAX_RESPONSE_TOKENS was raised to 300 to fix, on a different
# constant, arrived at by flipping the backend rather than by editing anything.
#
# 240 is two short sentences of SPEECH by the same measurement (~100 characters
# → ~116 audio + ~72 text, with room). Still a real ceiling — an ordinary
# one-sentence scene answer measured 90-150 output tokens on that drive — and
# still far below the session's own cap.
#
# Named as two constants and a function rather than one number, because the
# number is wrong for one of the two backends whichever value it holds, and a
# single constant is how it was wrong in the first place.
REALTIME_LOOK_ANSWER_TEXT_TOKENS = 60
# 240 -> 360, and the measurement that moved it. On the acceptance pass of
# 2026-09-09 a normal two-sentence visual answer --
#
#     "That white sedan ahead looks like a BMW 5 Series, probably an E60 from
#      the mid-2000s. Clean shape, very BMW about it."
#
# -- came back status=incomplete, reason max_output_tokens, and was filed as a
# token_cap cutoff. 117 characters. The 240 above was sized at ~100 characters
# by the arithmetic in the note over it, so an answer barely longer than the
# sizing case hit the ceiling.
#
# AND IT IS NOT A CHARACTER COUNT, which is why the new number has real room
# rather than another 20%: a 131-character answer on the very next run did NOT
# truncate. Audio tokens are billed on the SPEECH, and how long a sentence
# takes to say varies with what is in it -- so characters predict the cost
# loosely and cannot be tuned against tightly. 360 is ~190 characters at the
# measured 1.879 tokens/char, which covers two sentences plus the one short
# "want to know more about it?" clause the look() rules explicitly invite, with
# margin for the same sentence being said slower.
#
# Still a real ceiling, and still far below the session's own cap: the point of
# it is that a camera answer is one or two sentences and not an essay, and that
# is enforced by the rules in the look() result, not by cutting her off
# mid-clause. A ceiling should be the thing that never happens.
REALTIME_LOOK_ANSWER_AUDIO_TOKENS = 360


def look_answer_max_tokens() -> int:
    """The scene-answer ceiling, in the units the live session is billed in.

    A function and not a constant: VOICE_BACKEND is decided further down this
    file, so anything evaluated here would read it before it exists — and a
    constant that had to be kept in step with the backend by hand is the shape
    of the bug this replaces.
    """
    return (REALTIME_LOOK_ANSWER_TEXT_TOKENS if VOICE_BACKEND == "elevenlabs"
            else REALTIME_LOOK_ANSWER_AUDIO_TOKENS)


# --- when RIO should stop talking, and when she should not ------------------
# The complaint these exist for: her answers cut out mid-sentence and the
# driver had to ask again. Three of the four causes were one cause — the voice
# activity detector firing on RIO's own voice coming back through the cabin, or
# on a cough, a door, a wiper — and the answer being thrown away for it.
#
# HOW THE DRIVER'S TURN ENDS, which is a different question from how RIO's
# does and is answered by a different mechanism. See REALTIME_BARGE_* below
# for the second one; nothing here touches it.
#
# "semantic_vad" ends the turn on whether the sentence SOUNDS FINISHED, and
# "server_vad" ends it on a fixed stretch of quiet. That difference is the
# whole point, because a silence timer is a bet that no driver pauses for
# longer than it, and drivers do: "take me to... um... the Getty".
#
# MEASURED, on the real API, thirty complete questions and eight with a
# hesitation spliced into the middle (tools/turn_end_bench.py):
#
#   config              p50     p95   ended   sentences cut in half
#   server_vad 400     528m    568m   30/30      6/8
#   server_vad 500     631m    669m   30/30      4/8
#   server_vad 600     728m    769m   30/30      4/8
#   server_vad 700     829m    869m   30/30      3/8   <- what this was
#   semantic_vad low   762m    942m   26/30*     0/8
#   semantic_vad med   723m   1678m   29/30*     0/8
#   semantic_vad high  705m   1308m   50/50      0/40  <- what this WAS
#
# AND RE-MEASURED ON 2026-09-09, both rows on the same bench in the same hour,
# because the 3/8 above is what the choice below rested on and it did not
# survive being checked:
#
#   config              p50     p95     max    clean   sentences cut in half
#   server_vad 700     812m    848m    848m   10/10      0/8   <- what this IS
#   semantic_vad high  781m   2167m   2167m   10/10      0/8
#
#   ...and by gap length, which is where a timer is supposed to fail:
#   config             350ms   500ms   700ms   900ms
#   server_vad 700       0/2     0/2     0/2     0/2
#   semantic_vad high    0/2     0/2     0/2     0/2
#
# server_vad cut NOTHING. Not at 350 ms, not at 900 ms. The 3/8 in the row
# above is from before the bench's TAIL_MS bug was fixed -- the same bug the
# two starred rows carry a footnote for -- and it was never re-measured for
# this row, so the argument for semantic_vad was resting on a number the fix
# had already invalidated. Corrected here rather than quietly overwritten,
# because the reasoning below was written against it.
#
# AND ON A LIVE DRIVE, three runs of the seven-question script per arm
# (tools/live_tool_turns.py), turn-end measured to the commit:
#
#   arm                          first turns       later p50   p95    max  cuts
#   semantic_vad high      704, 2321, 2306             546   2321   2330     0
#   server_vad 700           619,  558,  603            620    671    685     0
#   semantic + 1s backstop  1112, 1091,  423             510   1112   1124     0
#
# The p95 is the whole story: 671 ms against 2321. semantic_vad's median is
# better by 74 ms and its tail is worse by 1.6 SECONDS, and a driver feels the
# tail. It also has no first-turn penalty -- 558 to 619 ms where semantic_vad
# spent 704 to 2321 on the same question.
#
# So the trade this file was built around is not there. It was real when it was
# measured and it is not real now, and the honest response to that is to switch
# and say why.
#
# * READ THOSE TWO STARS AS UNKNOWN, NOT AS BAD. Those runs fed a fixed tail
#   of silence after each question and then stopped sending; a detector
#   observes silence in the audio it is given and cannot observe an absence of
#   audio, so a decision slower than the tail had nothing left to decide on
#   and the turn simply never ended. That is the bench, not the detector, and
#   it is fixed (tools/turn_end_bench.py TAIL_MS, and the microphone in
#   tools/live_tool_turns.py, which now keeps running the way a cabin does).
#   The chosen row was re-measured after the fix and did not need the excuse:
#   50 questions, 50 turns ended, 40 hesitations and none cut.
#
# Read the 600 ms row against the last one: the same median wait, and four
# hesitations cut instead of none. That is the whole argument for the switch —
# not that semantic_vad is dramatically quicker, but that every server_vad
# setting fast enough to be worth having cuts more sentences in half than the
# one it replaces.
#
# The 700 ms silence window was buying protection it did not deliver: at that
# setting three of eight hesitations still cut the sentence in half, and the
# driver paid 829 ms on every single turn for it. semantic_vad cut none of
# them and is faster at the median.
#
# WHAT IT COSTS, stated because it is real: the tail is worse and less
# predictable. server_vad is a timer and lands within 100 ms of the same
# number every time; semantic_vad is a judgement, and the same sentence came
# back at 647 ms and at 1,090 ms on different runs. The median is better than
# what it replaces and the p95 is worse, and a driver notices both.
#
# "high" over "medium" on the tail, which is the only place they differ that
# matters: medium's p95 was 1,678 ms against high's 1,308, and the medians are
# within the jitter of each other. "high" is also the row that was
# re-measured at scale after the bench was fixed, so it is the one whose
# numbers above are not carrying an asterisk.
#
# WHAT THE GUARANTEE ACTUALLY IS, stated exactly rather than as "no false
# cuts". A silence timer cuts on the GAP and does not care what was said, so
# it splits "take me to... um... the Getty" whatever the words are. This cuts
# on the WORDS: it holds a half left dangling on a preposition or an auxiliary
# — "take me to...", "navigate to...", "what do you..." — which is what a
# hesitating driver leaves, measured at 0 cuts in 40 over gaps from 350 to
# 900 ms. It ends a half that was already a question: "What's that... car
# ahead?" gets ended after "What's that", and so does "Can you find me... a
# coffee?" after "Can you find me". That is the detector working. The
# difference in what it costs is the point — a turn ended there gets a
# sensible answer to a real question, where a timer ending after "take me to"
# answers a fragment.
#
# NO THRESHOLD UNDER semantic_vad, and that turned out not to matter. The 0.62
# below was raised from 0.5 to stop RIO's own voice returning through the
# cabin from reading as speech -- and measured, it never did that job: her
# voice at a tenth of full scale fires the detector at 0.62, at 0.5, and under
# semantic_vad, identically. What actually protects an answer from the cabin
# is the browser's sustain gate, exactly as REALTIME_BARGE_SUSTAIN_MS says.
# The server_vad numbers below are kept because server_vad is one env var
# away and they are what it should be set to.
# server_vad since 2026-09-09, on the re-measurement above: same cut rate as
# semantic_vad (zero, at every gap width tested) and a p95 3.5x better. Set
# RIO_TURN_DETECTION=semantic_vad to go back, and if you do, consider turning
# REALTIME_TURN_BACKSTOP_MS on with it -- it caps semantic's tail at 1124 ms
# where it otherwise runs to 2330.
REALTIME_TURN_DETECTION = os.getenv("RIO_TURN_DETECTION", "server_vad")

# How ready semantic_vad is to call a sentence finished: low, medium, high or
# auto. Higher is quicker to end the turn and quicker to be wrong about it.
REALTIME_SEMANTIC_EAGERNESS = os.getenv("RIO_SEMANTIC_EAGERNESS", "high")

# ...and the server_vad settings, for when it is selected. Defaults (threshold
# 0.5, 300 ms prefix, 500 ms silence) are tuned for a quiet room with a
# headset; a car is neither.
REALTIME_VAD_THRESHOLD = 0.62
REALTIME_VAD_PREFIX_MS = 300
REALTIME_VAD_SILENCE_MS = 700

# ---------------------------------------------------------------------------
# THE BACKSTOP: a client-side floor under semantic_vad's tail
# ---------------------------------------------------------------------------
# semantic_vad is a JUDGEMENT and its tail is unpredictable -- median 705 ms,
# p95 1308 ms, and measured on live drives the FIRST turn of a session lands
# between 829 and 2482 ms while every later turn sits near 500. server_vad is a
# timer: it lands within 100 ms of the same number every time, and it cuts
# sentences a hesitating driver leaves dangling.
#
# This is the third option, and the idea is to take the good half of each: keep
# semantic_vad deciding, and put a TIMER UNDER IT that only ever fires when the
# judgement has taken too long. The driver's microphone has been quiet this
# long and the detector still has not called the turn over, so the browser
# commits it.
#
# 0 DISABLES IT, which is the shipped default until the measurement says
# otherwise. A backstop that fires on an ordinary turn is server_vad with extra
# steps -- it has to sit far enough out that it is invisible on the turns
# semantic_vad handles well and only catches the tail.
#
# IT NEEDS A MICROPHONE LEVEL, and that is the honest limit on it: the browser
# has one (connect() passes `levels`), and a caller that provides no meter gets
# no backstop rather than a guess. Silence cannot be inferred from the server's
# own detector, because the server's own detector being slow is the thing being
# backstopped.
REALTIME_TURN_BACKSTOP_MS = int(os.getenv("RIO_TURN_BACKSTOP_MS", "0"))
# What counts as the driver having stopped, in dBFS on the microphone. The
# barge gate's echo floor is the same kind of number and this sits with it: a
# cabin at rest is well under this, and speech is well over.
REALTIME_TURN_BACKSTOP_MIC_DB = -45.0

# INTERRUPTION IS THE CLIENT'S DECISION, NOT THE SERVER'S — see
# realtime.session_config. The browser mutes RIO the instant the detector fires
# and only cancels her generation if the speech is still going after this long.
# Anything shorter than this was a noise, and a noise must not cost an answer.
#
# 300 ms is about the shortest real word. Below it the gate lets noise through;
# far above it the driver hears a gap before she stops generating (she is
# already silent — the mute is immediate), and the audio that gets discarded
# grows.
REALTIME_BARGE_SUSTAIN_MS = 300

# ---------------------------------------------------------------------------
# ...AND THE SAME GATE ON A PHONE, WHICH IS A DIFFERENT ROOM
# ---------------------------------------------------------------------------
# A laptop puts RIO's voice a foot from a microphone that is behind an echo
# canceller with a reference signal for everything it plays. A phone on a mount
# puts her voice out of a loudspeaker eight inches from the microphone, at
# driving volume, and — this is the part that is not obvious — only SOME of
# what she says goes out through a renderer the canceller can subtract.
#
# The audit is `node tools/echo_barge_probe.js`: the live session's own voice
# arrives as a WebRTC track and is cancelled; every deterministic line that
# falls back to the synthesiser, every pre-rendered clip, and the whole
# ElevenLabs path are ordinary media playback, which the canceller has no
# reference for. Those come back into the microphone at full level, the turn
# detector upstream calls them speech, and RIO interrupts herself. Measured
# with the shipped 300 ms gate and nobody in the car: five answers, five
# false barge-ins.
#
# Three separate changes, because echo is separable from speech in three
# different ways and no one of them is enough on its own:
#
#   SUSTAIN. Echo tracks her voice, so it stops when she pauses — which real
#   speech does not do on her schedule. 600 ms is longer than the gaps between
#   her own words and shorter than any interruption a driver actually makes:
#   somebody who means to cut in says at least a word, and a word is 300-600 ms
#   before the space after it.
#
#   ONSET GUARD. The detector fires hardest at the START of an utterance --
#   the canceller has not converged, the level jumps, and the first syllable is
#   the loudest thing in the cabin. The first 400 ms of any RIO utterance is
#   therefore the least trustworthy evidence of a driver there is. NOT
#   discarded: deferred. If the speech is still going when the guard expires it
#   is treated as a barge-in from that moment, so a driver who talks over her
#   opening word still stops her — a fifth of a second later than before.
#
#   LEVEL MARGIN. The one test that separates echo from a person on physics
#   rather than on timing: echo cannot be louder than what produced it. The
#   page measures the microphone and the audio it is rendering, and requires
#   the microphone to beat the output by this margin before a cut-off is
#   allowed to cost an answer. 6 dB is a factor of two in amplitude — comfortably
#   above what leaks back through a loudspeaker and comfortably below a driver
#   speaking up to be heard over her.
#
# ALL THREE ARE ZERO/UNCHANGED ON DESKTOP. The desk case works, is tested, and
# is not what broke; a phone is the exception and pays for itself.
REALTIME_BARGE_SUSTAIN_MS_TOUCH = 600

# The opening of her utterance, during which the detector is not believed on
# its own. Desktop keeps none: nothing there produces an onset transient the
# canceller cannot handle.
REALTIME_BARGE_ONSET_GUARD_MS = 0
REALTIME_BARGE_ONSET_GUARD_MS_TOUCH = 400

# How far the microphone has to beat the rendered output, in dB, before speech
# during her turn is allowed to cancel an answer. 0 disables the test, which is
# the desktop setting: there, the canceller has already done this job.
REALTIME_BARGE_ECHO_MARGIN_DB = 0
REALTIME_BARGE_ECHO_MARGIN_DB_TOUCH = 6

# Below this the output is not loud enough to be echoing anything, so the level
# test is skipped and the gate behaves exactly as it does on a desk. dBFS.
REALTIME_BARGE_ECHO_FLOOR_DB = -50

# After a cancel, how long to wait for a transcript before concluding there was
# never anyone there. A real interruption produces one: the words are already
# in the input buffer and transcription follows within a beat. Silence past
# this means the detector fired on nothing, and the answer it cost gets
# finished. Generous, because a slow transcription that resumed over the top of
# a driver who really was talking is the one failure worse than the original.
REALTIME_BARGE_CONFIRM_MS = 1500

# How many times one answer may be resumed. One: an answer that is cut off,
# resumed, and cut off again is in an argument with the cabin, and repeating
# "as I was saying" is worse than stopping.
REALTIME_MAX_RESUMES = 1

# ---------------------------------------------------------------------------
# A CANCEL IS A SUPERSEDE ONLY WHEN A REAL NEW QUESTION EXISTS
# ---------------------------------------------------------------------------
# THE BUG THIS CLOSES, from a live iPhone test on 2026-09-09. Two sessions,
# 36 cut-offs between them, and RIO never finished a sentence:
#
#     turn_superseded   15   of which TEN were by="Hello." or "What's up?"
#     cutoff other      12   every one reason="cleared"
#     cutoff barge_in    1   a genuine interruption
#     echo_suppressed    1   the gate working
#     blips_absorbed     1   the gate working
#
# "Hey. What's up." is RIO's own opening line (rio_prompts.py). Her voice came
# out of the iPhone speaker, back into the microphone, through the input
# transcriber as "Hello." / "What's up?", and the newest-wins supersede treated
# each one as a new driver question and cancelled the response that was
# producing the audio. A self-sustaining loop: she could never get past her own
# greeting. The said_chars at the moment of cancellation were 4, 9, 4, 4 -- she
# was four characters in.
#
# The twelve `cleared` cut-offs were the same bug downstream: supersedeTurn
# calls RIO.speech.clear('convo'), the arbiter drops the speaking item with
# reason 'cleared', and that is reported as a cut-off. One cause, 27 events.
#
# AND THE GATES THAT SHOULD HAVE STOPPED IT WERE ALREADY THERE AND WORKING.
# bargeIn() returns without creating a pendingBarge in three cases -- during a
# dictation, inside the onset guard, and when the level test says the
# microphone is quieter than the loudspeaker. All three mean "this is her, do
# not cancel". The supersede never asked. It fired on any non-empty transcript.
#
# So the rule, and it is the same one barge-in has always had: WHILE SHE IS
# SPEAKING, a transcript may only supersede if the barge gate CONFIRMED it --
# sustained past the guard and past the level test. While she is silent there
# is nothing of hers in the room and a transcript is a question.

# How long after her audio stops the room may still contain it. Her voice
# reaches the microphone through a speaker a few centimetres away, so this is
# generous rather than tight; what it buys is that a transcript arriving in the
# tail of her own sentence is still judged as an echo rather than as a question.
REALTIME_ECHO_TAIL_MS = 600

# The second net: does the transcript look like something she just said?
#
# Deliberately a SECOND net and not the first one, because it cannot be relied
# on. Her audio is transcribed twice by two different models -- once as her
# output transcript, once as microphone input -- and they disagree: she said
# "Hey" and the input transcriber wrote "Hello". A text test would not have
# caught the very case that motivated it. It catches the verbatim ones
# ("What's up?" against her own "What's up."), the structural gate above
# catches the rest, and neither is sufficient alone.
REALTIME_ECHO_TEXT_WINDOW_S = 15.0
# Fraction of the incoming transcript's words that must also appear in what she
# has recently said for it to be called an echo. 0.8 is "almost all of it":
# a driver's question that happens to reuse two of her words is not an echo.
REALTIME_ECHO_TEXT_OVERLAP = 0.8
# ...and below this many words, only an exact containment counts. "Yes" and
# "no" share every word with almost anything.
REALTIME_ECHO_TEXT_MIN_WORDS = 2

# ---------------------------------------------------------------------------
# Newest wins: superseding a turn (item 4 of the first real-drive punch list)
# ---------------------------------------------------------------------------
# WHY THIS IS URGENT AND NOT TIDINESS. On the first real drive the four `look`
# tool calls took 40.1 s, 12.9 s, 48.5 s and 27.1 s -- the visual path went out
# to Qwen and one /perceive on that drive took 52.6 seconds. A driver does not
# wait 40 seconds. They ask again. And until now the first question's tool call
# went on running, came back, and asked the model to speak about it.

# THE COMMANDS THAT PREEMPT, and they are matched as WHOLE utterances.
#
# Anchored on purpose. "stop" inside "don't stop at the next light" is not a
# command and must never be treated as one, so a command is an utterance that
# is essentially nothing but the command. Two kinds:
#
#   nav       acted on in the page, immediately, without waiting for a model
#             turn. These phrases have no other meaning in a car, and the
#             alternative -- waiting for a round trip behind a 40-second tool
#             call -- is the thing this item exists to fix.
#   silence   stop talking. Answered by stopping, and by NOT asking for a
#             response: replying to "be quiet" with speech is the wrong shape
#             of obedience.
REALTIME_DRIVER_COMMANDS = {
    "stop_navigation": [
        r"stop( the)? nav(igation)?", r"cancel( the)? nav(igation)?",
        r"end( the)? nav(igation)?", r"stop( the)? route", r"cancel( the)? route",
        r"stop navigating", r"stop guiding me",
    ],
    "reroute": [
        r"re-?route", r"re-?calculate", r"find (me )?another way",
        r"different route", r"new route",
    ],
    "silence": [
        r"stop", r"stop talking", r"be quiet", r"quiet", r"shut up",
        r"never ?mind", r"forget it", r"cancel that",
    ],
}
# Anything longer than this is a sentence, not a command, whatever it contains.
REALTIME_COMMAND_MAX_WORDS = 5

# --- coalescing rapid-fire fragments ---------------------------------------
# Semantic VAD splits an utterance the moment the driver breathes, and a driver
# saying "is there a petrol station... near the next exit" is asking one
# question. Superseding the first half with the second is technically correct
# and practically the same bug in the other direction: it throws away the tool
# call for the question being asked.
#
# So two fragments inside this window, WHERE THE TRANSCRIPT SHOWS CONTINUATION,
# are one turn: no supersede, no abort, and the model answers both because both
# are already in the conversation.
REALTIME_COALESCE_GAP_MS = 1500
# The second fragment continues the first if it opens with one of these...
REALTIME_COALESCE_OPENERS = (
    "and", "or", "but", "then", "also", "plus", "so", "um", "uh", "er",
    "like", "actually", "i mean", "near", "next to", "on", "in", "at", "to",
    "for", "with", "by", "from", "about", "around", "over", "under", "after",
    "before", "because", "which", "that", "who", "just",
)
# ...or the first ends on one of these, which is a sentence that has not
# finished being a sentence.
REALTIME_COALESCE_TRAILERS = (
    "and", "or", "but", "the", "a", "an", "of", "to", "for", "with", "at",
    "on", "in", "is", "are", "was", "were", "near", "by", "from", "that",
    "like", "about", "some", "any", "my", "your", "it's", "there's",
)

# ---------------------------------------------------------------------------
# ROAD NOISE, AND THE ONE ANSWER IT GETS
# ---------------------------------------------------------------------------
# The coalescing above joins two fragments the TRANSCRIPT shows are one
# sentence -- a dangling "and", an opener. That is the right test for a driver
# taking a breath mid-question and it is no test at all for road noise, which
# arrives as several complete little sentences with nothing joining them.
# Session 738fbb82, t=180.97 to t=180.98 -- ten milliseconds:
#
#     "Hello."   "Hello."   "Thanks for your help."   "Got it."
#
# Twenty of those over one drive, from a phone on a mount at 11 m/s with the
# windows down. The barge gate refused every one as a phantom so none of them
# superseded -- but turn_detection.create_response is on, so the SERVER had
# already created a response for each committed utterance, and each was an
# answer to nothing stacked behind whatever she was saying.
#
# So fragments inside this window are ONE utterance whatever the words are.
REALTIME_NOISE_COALESCE_MS = 2000
# Above this many words a transcript carries a request in it and is a turn,
# whatever it is made of. Four, because "Thanks for your help." is four and was
# one of the fragments on the drive -- and "Turn left onto Ocean" is four too,
# which is why the ceiling alone decides nothing: every word has to be in the
# no-request list below as well.
REALTIME_NOISE_MAX_WORDS = 4
# One "Didn't catch that." per window, and one per this -- a rough road is a
# rough road for minutes at a time, and a car that apologises every two seconds
# is broken in a new way.
REALTIME_NOISE_REPLY_COOLDOWN_MS = 12000
REALTIME_NOISE_REPLY = "Didn't catch that."
# The words that carry no request. A transcript is the road only if it is short
# AND made of nothing else -- a length test alone swallows "What's that?",
# which is three words and the most common real question in the car.
REALTIME_NOISE_TOKENS = (
    "uh", "um", "er", "erm", "mm", "mmm", "hmm", "huh", "ah", "oh", "eh",
    "hey", "hello", "hi", "yeah", "yep", "yes", "no", "nope", "ok", "okay",
    "right", "sure", "thanks", "thank", "you", "got", "it", "wow", "well",
    "so", "like", "the", "a", "and", "for", "your", "help", "good", "nice",
    "cool", "alright", "sorry", "please", "there",
)

# ---------------------------------------------------------------------------
# THE FRAME SOCKET'S IDLE CLOSE
# ---------------------------------------------------------------------------
# On 2026-09-09 the page stopped sending frames 40 s into a drive and this
# socket stayed open until t=400.4 -- six minutes of a server holding a
# session, a worker task and a frame slot for a client that had gone quiet, and
# six minutes in which neither end said anything.
#
# Closing is the right action rather than a tidy one: the browser's `onclose`
# is what starts a reconnect, so a close here is how a page suspended by iOS --
# which cannot run its own timers to notice anything -- gets a working
# transport back the moment it wakes.
#
# Well clear of a ping interval (5 s) and of any plausible pause between
# frames; a client that is alive at all sends something inside this.
HEADWAY_WS_IDLE_CLOSE_S = 30.0

# How long a tool call may keep running for a turn that has been superseded.
# Zero is the honest number on the client -- the AbortController fires at once
# -- and this is the SERVER's grace: /realtime/tool watches for the client
# going away and stops waiting on work nobody is going to hear about.
REALTIME_TOOL_DISCONNECT_POLL_S = 0.25

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

# ---------------------------------------------------------------------------
# WHOSE VOICE
# ---------------------------------------------------------------------------
# Two backends, one at a time, chosen by VOICE_BACKEND and read at startup:
#
#   "openai_realtime"  RIO's voice is the live session's own. Speech to speech:
#                      one model hears the driver, thinks, and speaks, and the
#                      words never exist as text on the way to the speaker.
#                      THIS IS THE SHIPPED PATH. Deterministic lines — a turn,
#                      a health announcement, a headway warning — are DICTATED
#                      into that same session word for word, which is what
#                      makes one voice everywhere a mechanism rather than a
#                      coincidence of two configs agreeing.
#   "elevenlabs"       RIO's voice is ElevenLabs. The live session still hears
#                      the driver and still does the thinking, but it is put in
#                      TEXT mode and its words are synthesised — streamed into
#                      a Text-to-Dialogue socket phrase by phrase, so the first
#                      audio starts before the model has finished the sentence.
#                      Complete and DORMANT: nothing on the active path reaches
#                      it, and it is one env var from being the voice again.
#
# The choice is config plus a restart, not a runtime toggle, and deliberately:
# the two paths differ in what the live session is even asked to PRODUCE, and a
# session that changed its mind about that halfway through a drive would be a
# third path nobody had tested. The per-utterance fallbacks below are what
# handles trouble inside a drive; this decides which mouth a drive starts with.
#
# DORMANT IS NOT DELETED, and the distinction is the whole reason the losing
# branch is kept compiling. Under this backend the ElevenLabs endpoints are
# still the SECOND tier of the deterministic fallback chain (see rio_speak.js):
# dictation first, the synthesiser if the session will not start the line in
# time, a pre-rendered clip if that fails too. A warning that arrives in a
# slightly different voice is a much smaller thing than a warning that does
# not arrive, so "nothing on the active path" means exactly that and not
# "nothing anywhere".
VOICE_BACKEND = os.getenv("VOICE_BACKEND", "openai_realtime")
# "realtime" was what this setting called the live voice before ElevenLabs came
# back. Old .env files and old shell exports still say it, and a car that comes
# up mute because a value was renamed is a bad trade for a tidier vocabulary.
if VOICE_BACKEND == "realtime":
    VOICE_BACKEND = "openai_realtime"

# The one thing that does not vary WHEN ELEVENLABS IS THE VOICE. Every
# ElevenLabs path below — the dialogue socket, the deterministic lines, the
# pre-rendered clips, both fallback tiers — names THIS voice. Under
# openai_realtime the same job is done by OPENAI_REALTIME_VOICE, and by the
# same argument: one constant that everything which speaks reads.
ELEVENLABS_VOICE_ID = (os.getenv("ELEVENLABS_VOICE_ID") or "").strip()

# RIO's conversation, when ElevenLabs is her voice.
#
# eleven_multilingual_v2, because this is a PROFESSIONAL VOICE CLONE and v3
# does not reproduce it. v3 has the audio tags and the prosody that carries a
# shrug, and none of that is worth a voice that is recognisably not the one
# that was cloned — the whole argument for one voice everywhere is that the
# driver hears one person, and a conversational model that renders her as
# somebody else breaks it more thoroughly than two models ever could.
#
# THE MODEL DECIDES THE TRANSPORT, which is why this is not a one-word change.
# The Text-to-Dialogue socket takes eleven_v3 models and nothing else; every
# other model streams over the text-to-speech socket. voice_dialogue.py speaks
# both dialects and picks by the name here, so switching back to v3 is still
# this one setting.
ELEVENLABS_CONVERSATION_MODEL = os.getenv(
    "ELEVENLABS_MODEL_CONVERSATION",
    os.getenv("ELEVENLABS_MODEL", "eleven_multilingual_v2"))

# Which socket that model needs. Named as a question about the model rather
# than as a second setting, so the two can never disagree.
def uses_dialogue_socket(model: str = None) -> bool:
    """Is this a Text-to-Dialogue model? Only the v3 family is."""
    return str(model or ELEVENLABS_CONVERSATION_MODEL).startswith("eleven_v3")


# The old name for the setting above, kept because tools and docs use it.
ELEVENLABS_DIALOGUE_MODEL = ELEVENLABS_CONVERSATION_MODEL

# Everything deterministic, and every fallback. Flash is the fastest thing
# ElevenLabs has to first byte, and a warning is exactly the line where that is
# the only property worth optimising: a turn instruction does not need prosody,
# it needs to arrive. Same voice id, so the driver hears one person.
ELEVENLABS_DETERMINISTIC_MODEL = "eleven_flash_v2_5"

# Back-compatible alias. voice.synthesize_stream() and the tests reached for
# this name when there was only one ElevenLabs model in the system.
ELEVENLABS_MODEL = ELEVENLABS_DETERMINISTIC_MODEL

# PCM rather than MP3, and this is not a preference. The browser schedules
# RIO's audio itself so it can fade her out in 30 ms when a warning arrives,
# and scheduling means decoding: a raw 24 kHz frame is decodable the instant it
# lands, where an MP3 chunk out of the middle of a stream is not decodable at
# all without the frames around it. 24 kHz mono is 48 KB/s over a loopback
# socket, which costs nothing and buys the cancel path.
ELEVENLABS_OUTPUT_FORMAT = "pcm_24000"
ELEVENLABS_SAMPLE_RATE = 24000

# --- when a phrase is worth speaking -----------------------------------------
# The whole point of text mode is that RIO starts talking before she has
# finished thinking. That only works if something decides, mid-stream, that
# enough words have arrived to be worth synthesising — too eager and she speaks
# in fragments with a seam in the prosody at every one; too patient and the
# text path is slower than the audio path it replaced.
#
# A phrase is sent when it reaches a clause boundary AND has at least this many
# words behind it...
ELEVENLABS_CHUNK_MIN_TOKENS = 5
# ...or when this long has passed since the first unsent word arrived, whatever
# the model is doing. The second rule is what covers an answer that opens with
# a long subordinate clause, and what stops "Yeah." waiting forever for a
# boundary that has already been and gone.
ELEVENLABS_CHUNK_MAX_WAIT_MS = 250

# ...and a lower bar for the FIRST phrase of an answer, because it is the only
# one anybody is waiting on.
#
# Every later phrase is spoken over audio that is already playing, so its
# threshold is about prosody: a seam in the middle of a sentence is audible and
# a longer piece reads better. The first one has nothing playing behind it, and
# every millisecond it waits is silence the driver is sitting in.
#
# MEASURED at 74 ms p50 on a visual answer under the ordinary rule, which is
# small — this is the last stage of the budget that belongs to this system
# rather than to a model, and it is worth having at its floor rather than
# nearly there. Not lower than three words: two words is not enough for the
# synthesiser to pitch a sentence, and a first chunk that arrives fast and
# sounds wrong is a worse trade than 40 ms.
ELEVENLABS_FIRST_CHUNK_MIN_TOKENS = 3
ELEVENLABS_FIRST_CHUNK_MAX_WAIT_MS = 120

# The socket closes itself after 20 seconds without a client message. A car
# spends most of a drive inside that window, so a keep-alive goes out at half
# the interval — comfortably inside it, and cheap: the message performs no
# synthesis and costs no characters.
ELEVENLABS_KEEPALIVE_MS = 9000
ELEVENLABS_IDLE_TIMEOUT_MS = 20000

# --- the two fallbacks, and what separates them ------------------------------
# TIER 1, per utterance: v3 conversational was slow or errored on THIS line.
# The line is re-synthesised on flash, same voice, and the drive carries on with
# the dialogue socket for the next one. A conversational reply that arrives a
# beat late in a slightly flatter reading is a much smaller thing than a reply
# that does not arrive.
ELEVENLABS_FIRST_BYTE_BUDGET_MS = 1500

# ...and one condition that is neither a slow line nor a dead service.
#
# MEASURED on this account (Starter): the workspace may hold 21 concurrent
# DIALOGUE sessions, which is a separate and much larger pool than the standard
# concurrency limit (3 on multilingual v2, 6 on flash). The seat is taken
# lazily — a connection alone costs nothing, and 22 idle sockets are all
# accepted — so the refusal does not arrive at connect time. It arrives the
# first time a session tries to SYNTHESISE, as a 1008 close carrying
# `too_many_concurrent_requests`.
#
# Which means a car can open its socket cleanly at the start of a drive and be
# cut off the moment RIO first speaks. That looks exactly like a dropped
# socket, and the reconnect that is right for a dropped socket is wrong here:
# the new connection is accepted, the next utterance is refused again, and the
# drive spends itself reconnecting once per sentence into a pool that is full.
#
# So a capacity refusal parks the dialogue socket for this long and runs on
# flash meanwhile — same voice, faster, and no churn. Sixty seconds because the
# thing that frees a seat is another car finishing a drive, which is minutes
# away, not milliseconds.
ELEVENLABS_CAPACITY_BACKOFF_S = 60.0

# TIER 2, per session: ElevenLabs is not answering at all — the socket will not
# open and flash will not stream either. Then RIO's voice goes back to the live
# session's own, mid-drive, by switching that session to audio output. Sticky
# for the rest of the drive: a voice that flickers between two people because
# the network is flickering is worse than either voice.
#
# How many consecutive per-utterance failures before concluding it is the
# service and not the line. Two, because one is an utterance and three is a
# conversation the driver has already noticed.
ELEVENLABS_FAILURES_BEFORE_CEDAR = 2

# --- audio tags (v3) ---------------------------------------------------------
# v3 can be told how to say something: [laughs], [sighs], [whispers]. Used
# sparingly this is the difference between a voice and a reading. Used the way
# a model will use it if you let it, it is exactly the "performative" register
# the bible bans in as many words (§ "NOT loud, polished, corporate, robotic,
# or performative").
#
# So the tags are allowed in ONE place — conversation — from a short list, one
# per utterance, and a tag is never the whole utterance. Everywhere else they
# are stripped before synthesis, because the alternative is a navigation
# instruction in which the model has written the word "sighs" and the driver
# hears it read out at a junction.
#
# The list is short on purpose. Every tag on it is something a friend in the
# passenger seat actually does; nothing on it is a performance.
# UNREACHABLE UNDER openai_realtime, and left switched on rather than switched
# off, because the thing that makes them unreachable is not this flag. A
# speech-to-speech session emits sound, not text: there is no string between
# the model and the speaker for a bracket to be written into, and nothing on
# that path could read one if there were. So the gate is upstream of the
# feature — realtime.instructions() adds the tag paragraph ONLY under
# elevenlabs, and a model never told about a mechanism does not reach for it.
# The regression pass checks the transcripts anyway, because "cannot happen"
# and "did not happen" are different claims and only one of them is evidence.
AUDIO_TAGS_ENABLED = True
AUDIO_TAGS_ALLOWED = ("laughs", "sighs", "whispers", "exhales")
AUDIO_TAGS_MAX_PER_UTTERANCE = 1

# --- the old shape, still true ----------------------------------------------
# The server's TTS endpoints (/nav/voice, /headway_voice,
# /vehicle/health/voice) synthesise with this. Which tier they are depends on
# whose voice RIO has, and that is the only thing that changes:
#
#   openai_realtime   the FALLBACK. Nav, health and the calm headway tier are
#                     dictated into the live session, and these endpoints are
#                     what the browser reaches for when a line will not start
#                     in REALTIME_SPEAK_TIMEOUT_MS. A different voice for one
#                     late warning, which is the right trade for that warning
#                     arriving at all.
#   elevenlabs        THE path for everything deterministic — nav, health and
#                     the calm headway tier all come out of flash on the same
#                     voice id RIO converses in, which is what made one voice
#                     everywhere true under that backend without dictating
#                     anything.
VOICE_FALLBACK_BACKEND = "elevenlabs"

SYSTEM_PROMPT = RIO_SYSTEM_PROMPT

VISION_ENABLED = True

# ---------------------------------------------------------------------------
# Place search (places.py) — what is actually around the car
# ---------------------------------------------------------------------------
# RIO answers "what's good round here" from Google Places, never from the
# model's memory of restaurants. See places.py's header for why that is not a
# preference.
PLACES_ENABLED = True

# Results per question. Five is what a driver can hold in their head; RIO reads
# the best two or three of them and offers the rest. It is also the cap on the
# billed request, since maxResultCount is sent.
PLACES_MAX_RESULTS = 5

# The bias circle for "near me". Wide enough that a quiet suburb still returns
# somewhere to eat, tight enough that "near me" does not mean the next city.
PLACES_BIAS_RADIUS_M = 8000.0

# A GPS fix older than this is not where the car is. Refused rather than used:
# "near me" answered from a ten-minute-old position is wrong in the one way the
# driver cannot detect. RIO asks for an area instead.
PLACES_FIX_MAX_AGE_S = 600.0

# The drive-time ESTIMATE (places.drive_minutes). Not a routed time — that
# would be a Routes call per result, five billed requests to decorate one
# sentence. 9 m/s is ~32 km/h, an urban average with lights; 1.35 is the usual
# straight-line-to-road detour factor. Both are only ever spoken as "about".
PLACES_DRIVE_SPEED_MS = 9.0
PLACES_DETOUR_FACTOR = 1.35

PLACES_TIMEOUT_S = 8.0

# How long "the second one" still means something. Long enough for a couple of
# exchanges about the list, short enough that open-now cannot have flipped
# unnoticed.
PLACES_CACHE_TTL_S = 180.0


# --- replay presentation buffer ---------------------------------------------
# Seconds the ANALYSIS stream runs ahead of the picture during an uploaded-clip
# headway run. It is the fix for a rendering fault, not a detection one: a box
# is computed from the frame at time T and cannot exist until T + latency, so
# drawn on arrival it lands on a frame the road has already moved past. On a
# file there is no reason to accept that -- the future of the clip is sitting
# on disk. The dashboard therefore analyses from a hidden video element held
# this far AHEAD of the visible one, so by the time the viewer reaches frame T
# its result is already in hand and the box lands on the pixels it was computed
# from.
#
# Sized from the measured end-to-end latency, p95 rather than p50: the buffer
# has to cover the slow frames, since the fast ones simply wait. Measured over
# HTTP on this pod at 1280x720: p50 41 ms, p95 173 ms, worst 246 ms. 250 ms
# covers that with headroom and is still under the ~300 ms at which a delayed
# start becomes noticeable when the clip is scrubbed.
#
# The cost is start-up: nothing can be drawn over the first HEADWAY_REPLAY_LEAD_S
# of the clip, because no frame that old has been analysed yet.
#
# Injected into the page at "/" (like the Maps key) so the browser, the harness
# and this file cannot disagree about it. Live camera mode ignores it entirely
# -- reality cannot be buffered, and extrapolation covers that case instead.
HEADWAY_REPLAY_LEAD_S = 0.25

# One frame at 24 fps. The alignment target: a box must be drawn on a frame
# within this of the one it was computed from, which is the point at which the
# error is smaller than the interval between frames and cannot be seen.
HEADWAY_ALIGN_TOLERANCE_S = 1.0 / 24.0


# ---------------------------------------------------------------------------
# Live frame transport (item 1 of the first real-drive punch list)
# ---------------------------------------------------------------------------
# WHAT THE DRIVE LOG SAID, and it is the reason every number below exists.
# Session 06af3214 (iPhone, 2026-09-08, 607.8 s, 1949 frames):
#
#   inter-frame arrival    p50 258 ms  p90 406 ms  p99 1503 ms   (3.88 fps p50)
#   server processing      p50  23.6 ms  p90 33.5 ms
#     decode 3.0 / lanes 3.0 / depth 6.8 / detect 9.1 / membership 1.2 / filter 0.3
#   frames carrying a capture timestamp:   0 of 1949
#
# So the server was never the bottleneck -- it spent 24 ms on a frame that
# arrived every 258 ms -- and the 250 ms floor in the page was the whole of the
# frame rate. And the last line is the one that mattered most: with no
# `frame_t` on any frame, the server could measure when a frame ARRIVED and
# never how old it was, so "frame age at detection" -- the only number that
# says whether a warning is about the road the car is on now -- was not a
# number anybody had.
#
# Both are fixed here: the phone stamps every frame, and the transport stops
# being one HTTP request per picture.

# The WebSocket push. Falls back to the POST path (which is unchanged and still
# the desk-testing path) when the socket will not open or drops.
HEADWAY_WS_ENABLED = True

# Adaptive capture rate, in frames per second. The floor is what the link is
# allowed to drag the loop down to before frames start being dropped instead;
# the ceiling is where extra frames stop buying anything the filter can use.
HEADWAY_WS_MIN_FPS = 8.0
HEADWAY_WS_MAX_FPS = 15.0
HEADWAY_WS_START_FPS = 10.0

# DROP, NEVER QUEUE. The single rule this transport exists to enforce.
#
# A queued frame is a frame that will be measured against a dt that has already
# passed, and a warning computed from it is a warning about a road the car has
# left. So there is exactly one frame in flight at a time and exactly one frame
# waiting on the server, and a new capture that finds either slot occupied
# REPLACES what is there rather than lining up behind it.
#
# This is the same argument the POST path already made with its non-blocking
# per-session lock (see app.headway_frame_endpoint); the difference is that on
# a socket the newest frame can evict the waiting one instead of being thrown
# away itself, so the frame that gets processed is always the freshest one.
# How many frames may be ON THE WIRE at once, and it is a PIPELINE DEPTH, not
# a queue length -- the distinction is what keeps "never let frame age grow"
# true at 8-15 fps.
#
# With one frame in flight the achievable rate is 1/(serialise + round trip),
# and on the link the first drive was measured over -- 2.0 Mbit/s up, 60 ms RTT
# -- that is about 4 fps whatever the capture loop asks for. Measured: frame
# age held at 161 ms p50 and the stream ran at 3.85 fps with 182 captures
# skipped. Frame age was excellent and the frame rate was the old one.
#
# So the depth is allowed to grow, and it is FRAME AGE that decides when. The
# client raises it only while measured age sits well under target, and drops it
# straight back to 1 the moment age crosses the ceiling. The server's slot is
# unaffected: exactly one frame ever waits there, and a newer one evicts it.
# Nothing accumulates on either side, which is the property that matters --
# what grows is how much of the radio is kept busy, not how far behind the
# picture is.
HEADWAY_WS_MAX_INFLIGHT = 3
# Bytes still unsent on the socket above which the next capture is skipped
# outright. Roughly two frames at the byte budget below: past that the radio is
# behind and adding a picture to the back of it only makes the next one older.
HEADWAY_WS_BUFFER_LIMIT_BYTES = 120_000

# Frame age (capture -> detection complete) the rate controller aims to keep
# under. Above the ceiling it sheds frame rate and then quality; below the
# floor it takes the rate back up.
HEADWAY_WS_TARGET_AGE_MS = 220.0
HEADWAY_WS_MAX_AGE_MS = 400.0

# --- JPEG, tuned for a mobile uplink ----------------------------------------
# The old path encoded at the camera's own resolution and a fixed q0.8, which
# on this phone produced 71-90 KB per frame (measured off the /perceive events
# of the same feed). At 4 fps that is ~2.6 Mbit/s of uplink for a picture the
# detector runs on at a few hundred pixels a side, and at 15 fps it would be
# ~10 Mbit/s -- more than an LTE uplink has. Downscaling first is what makes
# the higher frame rate affordable at all.
#
# 640 on the long side is above every consumer of these frames: the detector
# letterboxes to 560, the depth model to 518, the lane model to 800x320. So
# this costs nothing that is measured and saves most of the bytes.
HEADWAY_WS_MAX_SIDE_PX = 640
HEADWAY_WS_JPEG_QUALITY = 0.62
HEADWAY_WS_JPEG_QUALITY_MIN = 0.40
HEADWAY_WS_JPEG_QUALITY_MAX = 0.78
# The byte budget one frame is allowed. Quality walks between the bounds above
# to hold this; at 12 fps it is ~2.3 Mbit/s, which is what a mobile uplink can
# actually carry while a voice session is open on the same radio.
HEADWAY_WS_TARGET_BYTES = 24_000

# How long the socket may go without a frame before the server lets the session
# go. Longer than any single stall the drive log showed (max 2.0 s) and shorter
# than the session reaper, which is about a drive ending rather than a link
# hiccupping.
HEADWAY_WS_IDLE_TIMEOUT_S = 30.0


# ---------------------------------------------------------------------------
# Live headway policy — every tunable the live drive loop reads
# (docs/live_headway_v3.md; item 2 of the first real-drive punch list)
# ---------------------------------------------------------------------------
# PROVISIONAL prototype values, and the warning_logic_v2.md header still
# applies verbatim: "These are engineering starting points for shadow-mode
# tuning -- NOT validated safety thresholds. Do not represent them as such
# anywhere, ever."
#
# They lived in headway/live_policy.py's own PROVISIONAL block until the first
# real drive, and they are here now for the reason the punch list gives: a
# threshold that decides whether a car speaks is a threshold that has to be
# readable, diffable and settable in one place, next to the navigation timing
# and the transport tuning that are already here.
#
# headway/live_policy.py imports these and holds no numbers of its own.

# --- band entry, in TIME HEADWAY -------------------------------------------
# gap / own speed. The two-second rule, which is the rule because it is the one
# quantity that means the same thing at 15 mph and at 65: 20 m behind a car is
# a comfortable gap in town and a second and a bit on a freeway.
HEADWAY_TAU_GETTING_UNSAFE_S = 3.0
HEADWAY_TAU_UNSAFE_S = 2.0
# Leaving a worse band needs tau > entry + this. Exit only; entry is the bare
# threshold.
HEADWAY_TAU_HYSTERESIS_S = 0.2

# --- the speed floor --------------------------------------------------------
# Below this, tau = d/v explodes and a 4 m gap in a car park reads as "getting
# unsafe". Creeping in traffic is not danger, and this is the line that says so.
# 5 m/s is 11 mph.
HEADWAY_MIN_COACH_SPEED_MS = 5.0

# --- what the detector costs, and how that is checked -----------------------
#
# THE OLD CHECK WAS A COIN FLIP. headway/live_selftest.py asserted that
# detection ran in "single-digit milliseconds" -- p95 wall time under 10 ms --
# and on this box the same code reads 9.99, 10.39, 11.55 or 15.25 depending on
# what else is running. That is not a bar with headroom; it is a round number
# sitting in the middle of the measurement's own distribution, and it went red
# on load and green on luck.
#
# WHAT THE DETECTOR IS ACTUALLY DOING, which is the thing nobody had measured.
# Profiled over ten forwards (RTX 4090, torch 2.11/cu128, RF-DETR nano fp16,
# 384x384 input, 2026-09-08):
#
#     GPU work            1.63 ms per forward
#     CPU work           11.0  ms per forward
#     elapsed             9.1  ms per forward
#     addmm launches      104  per forward, ~5.9 us of GPU each
#
# It is LAUNCH BOUND. The GPU does a millisecond and a half of arithmetic in a
# few hundred tiny kernels and spends the rest of the time waiting for the CPU
# to submit the next one. Three consequences, all measured rather than reasoned:
#
#   * making the arithmetic heavier costs NOTHING. fp32 instead of fp16: 9.08
#     against 9.40 ms. Input resolution x1.5, which is 2.25x the pixels: 9.05
#     against 9.10 ms. Neither is a rounding error in the check's favour --
#     they are genuinely free here.
#   * making the model BIGGER costs in proportion, because a bigger model is
#     more launches. Running the same forward twice: 18.5 ms against 9.4.
#   * the variance is CPU scheduling, not GPU contention. Measured against the
#     live server driving 14 fps of frames through detect, depth and lanes on
#     the same card: floor 9.28 -> 8.29, p50 9.78 -> 9.74, p95 10.35 -> 10.31,
#     and only the max moved, 13.4 -> 15.9.
#
# So the check asserts the FLOOR of the model pass's GPU-event time. A floor is
# what the hardware can do; load can only ever push a sample above it, never
# below. And the floor moves for exactly the regression this check is for -- a
# heavier model -- while precision and model identity are asserted DIRECTLY by
# the two checks next to it (the fp16 dtype and the parameter count), which is
# where they belong and where a timer would have been lying about them.
#
# A NUMBER WORTH COMING BACK TO: 1.63 ms of work taking 9.1 ms is roughly a 5x
# speed-up sitting there, in CUDA graphs or torch.compile(mode="reduce-
# overhead"), if detection ever needs to be cheaper than it is. It does not
# today -- see the frame budget below.

# 1.5x the measured floor of 9.1-9.8 ms. Above every launch-count regression
# and far above the scheduling jitter of a busy machine.
#
# Calibrated to THIS pod's CPU, which is what the floor depends on. A slower
# host would need this re-measured, not raised: re-run
# `python -m headway.live_selftest` on a quiet machine and read the floor it
# prints.
HEADWAY_DETECT_KERNEL_BUDGET_MS = 15.0

# ...and the property the check is FOR, which is not a round number at all:
# detection has to be cheap enough to run on every frame. At the transport's
# ceiling of HEADWAY_WS_MAX_FPS a frame is 67 ms, and detection shares it with
# depth, lanes, the JPEG decode, membership and the filter -- 14 ms of them on
# the first real drive. Half a frame is the bar, generous on purpose: this one
# IS asserted on the tail, so it has to survive a bad minute.
# For contrast, the Qwen enumeration this replaced took 600-1500 ms.
HEADWAY_DETECT_FRAME_BUDGET_FRAC = 0.5

# --- making the detector cheap enough for a Jetson --------------------------
#
# JETSON PREP, NOT A POD NEED. On this box detection is 11.8 ms of a 67 ms
# frame and nothing is waiting on it. On an Orin the same model has perhaps a
# fifth of the launch throughput and a CPU that is doing the camera and the
# radio as well -- and the profile above says this model's cost is almost
# entirely launch overhead, which is exactly the cost that gets worse on a
# smaller CPU. So the work happens here, on hardware where it can be measured
# against a known-good answer, rather than on the target where a regression
# would look like the target being slow.
#
# AND ONLY THE DETECTOR. The same question was asked of depth and lanes and
# answered no, by measurement (`tools/accel_verify.py --profile-all`):
#
#     detect   elapsed 10.30 ms   GPU 3.25 ms   2940 aten calls   3.2x  LAUNCH BOUND
#     depth    elapsed  7.47 ms   GPU 9.85 ms   1477 aten calls   0.8x  gpu bound
#     lanes    elapsed  2.78 ms   GPU 2.70 ms    403 aten calls   1.0x  gpu bound
#
# CUDA graphs remove launch overhead and nothing else. Depth and lanes have
# essentially none to remove -- their GPUs are already saturated and
# overlapping their own kernels -- so compiling them would buy nothing and
# would still be paid for with a numeric tolerance against the eager path.
#
# WHAT IT IS. torch.compile(mode="reduce-overhead") -- Inductor codegen plus
# CUDA graphs. Measured on an RTX 4090, RF-DETR nano fp16, 384x384:
#
#     eager           floor 9.6 ms
#     compiled        floor 3.1 ms      3.15x
#     compile cost    ~7 s, once, at warm-up
#
# WHAT WAS TRIED AND REJECTED:
#
#   a raw torch.cuda.CUDAGraph capture would be bit-for-bit identical, and it
#   cannot be captured: rfdetr's transformer builds `spatial_shapes` with
#   torch.as_tensor(<python list>, device=cuda) on every forward, and an
#   unpinned host-to-device copy is not capturable. Making it capturable means
#   patching a vendored library's internals in the path that decides when to
#   warn a driver, and that is a worse trade than a measured tolerance.
#
#   torch.compile(backend="cudagraphs") IS bit-for-bit identical -- and worth
#   1.09x, because dynamo graph-breaks on those same calls and the graph ends
#   up in fragments.
#
#   forcing fp32 accumulation (allow_fp16_reduced_precision_reduction=False)
#   does not close the numeric gap: 4.219 against 4.213. The difference is
#   Inductor's kernels and reduction orders, not GEMM accumulation.
#
# SO IT IS A TOLERANCE, AND THE TOLERANCE IS ON THE DETECTIONS. The compiled
# model's raw tensors differ by up to 4.21 on pred_logits -- 300 queries x 91
# classes of mostly junk at large negative logits, where that is nothing. What
# is checked instead is what the rest of the system actually receives: the
# gated detections, after the confidence gate, the size floor, the
# ego-structure gate and duplicate suppression. Same count, same labels, and
# boxes and scores inside the two numbers below.
#
# "auto" compiles at warm-up, runs that comparison on real frames, and adopts
# the compiled model ONLY if it passes -- so the default is on only where the
# identity check passes, on the machine it is actually running on. Anything
# that throws, at any point, leaves the eager model in place.
#   auto | on | off        ("on" skips the check; for a bench, not for a drive)
HEADWAY_DETECT_ACCEL = "auto"

# THE TOLERANCE, AND THE UNITS IT IS IN. Two of them, because two different
# things are being bounded, and the first attempt at this got both wrong.
#
# Attempt one was 0.25 absolute pixels. It failed on a 215x102 px car in a
# 1282-wide frame, by 0.626 px -- and absolute pixels turned out to be the
# wrong unit entirely: the model emits NORMALISED coordinates and the
# postprocess multiplies by the frame size, so a pixel bound charges a large
# frame for arithmetic it did not do.
#
# Measured over 411 detections on 180 real-road frames, three clips, four frame
# sizes (tools/accel_verify.py):
#
#                                     p50       p95       p99       max
#     absolute px                  0.0195    0.0782    0.4688    0.7812
#     fraction of frame width      3.1e-5    9.2e-5    5.2e-4    9.8e-4
#     fraction of the box's
#       SMALLER dimension          5.4e-4    2.2e-3    8.7e-3    2.3e-2
#
# 1. OF FRAME WIDTH is where the error lives, so it is the sensitive detector
#    of DRIFT -- a compiled model that started diverging would move this first.
#    3e-3 is about three times the largest disagreement seen. (An earlier
#    1e-3 sat at 98% of budget on the very corpus it was set from, which is
#    not a tolerance, it is a flake with a decimal point.)
HEADWAY_ACCEL_MAX_BOX_DELTA_FRAC = 0.003

# 2. OF THE BOX ITSELF is what bounds BEHAVIOUR, and it is a different question.
#    The worst case measured was 0.47 px on a 20x101 box -- 2.3% of its width --
#    and it is narrow boxes, not big frames, where a sub-pixel shift means
#    something. Every downstream consumer of a box reads a fraction of it:
#    membership is the fraction of the bottom edge inside the lane polygon,
#    plausibility divides by the pixel height, the depth ROI is a median over
#    its interior. Bounding the shift to 5% of the box's smaller side bounds
#    all of them to 5%, which is a third of the 0.15 hysteresis band between
#    MEMBER_ENTER_FRAC and MEMBER_EXIT_FRAC -- so no membership decision can
#    turn on it. Roughly twice what was observed.
HEADWAY_ACCEL_MAX_BOX_DELTA_OF_BOX = 0.05

# Scores: observed max disagreement 0.008, so this is ~2.5x. It does not need
# to be tight, because the thing it might otherwise let through -- a detection
# crossing a class gate and appearing or vanishing -- is caught by the count
# check below, which has no tolerance at all.
HEADWAY_ACCEL_MAX_SCORE_DELTA = 0.02
#
# AND THE DISCRETE OUTCOMES HAVE NO TOLERANCE. A detection appearing or
# vanishing, or the `confirmed` flag flipping -- which is the size floor, the
# thing that decides whether a range may be claimed for a box at all -- fail
# the check outright, because no distance in pixels describes either of them.
# Measured: zero of 411 detections flipped `confirmed`.
#
# THE ONE LABEL EXCEPTION, and it was measured rather than assumed. Two of the
# 411 disagreed: `bus` 0.3797 against `truck` 0.3795, and `bus` 0.3794 against
# `truck` 0.3789. detect.EXCLUSIVE_LABELS already exists because car, truck,
# bus and motorcycle are competing readings of ONE object, and duplicate
# suppression keeps whichever scored higher. So the eager model was asked again
# with the frame nudged invisibly -- one grey level up, one down, re-encoded at
# JPEG q95 and q92, shifted a single pixel:
#
#     EAGER ITSELF ANSWERED DIFFERENTLY ON FIVE OF SIX PERTURBATIONS.
#
# Requiring a compiled model to reproduce a tie-break the eager model loses to
# a one-grey-level change is requiring bit-identity. So an exclusive-label
# disagreement inside the score tolerance is allowed and counted; every other
# label disagreement fails.

# How many real frames the runtime check uses before adopting the compiled
# model. Small on purpose: it runs at warm-up, on the critical path to a drive
# being able to start, and tools/accel_verify.py is where the broad sweep
# lives. Eight frames at three sizes is enough to catch a compile that has
# gone wrong, which is what this is for -- the broad question was answered
# offline.
HEADWAY_ACCEL_VERIFY_FRAMES = 8

# --- the car's own bodywork, which is not a car -----------------------------
#
# THE UPSTREAM FIX FOR THE FLOOR BELOW. HEADWAY_TAU_IMPLAUSIBLE_S stops RIO
# SPEAKING about a lead that cannot be there; these stop the lead existing.
#
# Session 06af3214, 584 frames of it: a box at [0.6, 308, 640, 478] on a
# 640x480 frame -- the full width of the picture, flush with the bottom edge,
# and its TOP edge 64% of the way down the image. Ranged at 3.7 m and held as
# the lead at 20 m/s. It is the phone's view of the car's own bonnet and
# dashboard, and it moved by two pixels in ten minutes of driving.
#
# detect.py already refused one shape of this -- the thin full-width strip a
# windscreen dashcam sees, aspect 21.6. A phone sits higher and closer, so it
# sees a much TALLER slice of the same bodywork, aspect 3.8, and sailed
# through. Three independent gates now, in three layers, because each says
# something different and any one of them can be wrong on a given frame:
#
#   shape      (detect.py)        a box spanning the whole frame whose top
#                                 never rises above the horizon is not a
#                                 vehicle: anything that wide is ~2 m away, and
#                                 at 2 m a vehicle's roof is far above it.
#   arithmetic (plausibility.py)  a box that wide CANNOT be as far away as the
#                                 depth model says. The mirror of the height
#                                 bound already there.
#   motion     (membership.py)    the car drove twenty metres and this object
#                                 did not move three pixels. Only something
#                                 bolted to the camera can do that.

# --- shape: the ego structure gate (headway/detect.py) ----------------------
# How much of the frame width a box must span before the "top below the
# horizon" argument is allowed to apply at all. Deliberately near-total: the
# geometry is overwhelming at 0.95 and merely suggestive at 0.85, and the
# existing wide-and-flat rule already covers the rest.
HEADWAY_EGO_SLAB_WIDTH_FRAC = 0.95
# ...and how far above the pinhole horizon the box's top edge may rise and
# still count as bodywork. Zero: the test is "does not rise above the horizon
# at all", which is what makes it a statement about geometry rather than a
# fitted threshold.
HEADWAY_EGO_SLAB_HORIZON_SLACK_FRAC = 0.0

# --- arithmetic: the width bound (headway/plausibility.py) ------------------
# Only applied to a box spanning this much of the frame. A box width is a poor
# range estimator in general -- the boxes are loose, HFOV_DEG is uncalibrated,
# and a car at 50 m would be falsely vetoed. At near-full width it says
# something no calibration error can explain: 640 px of car is 1.6 m away, and
# no amount of slack makes that 3.7.
HEADWAY_WIDE_BOX_FRAC = 0.95

# --- motion: the static structure veto (headway/membership.py) -------------
# How far the CAR must travel before immobility means anything. Twenty metres,
# because the claim being made is physical: over twenty metres of road, every
# real object changes its size or its place in the picture, and only something
# rigidly attached to the camera does not.
HEADWAY_STATIC_TRAVEL_M = 20.0
# How much the box may drift over that distance -- as a fraction of frame
# width, so it is resolution-independent. Measured on the drive: every genuine
# road user drifted at least 0.0041 of frame width over twenty metres, and
# every bonnet frame drifted at most 0.0044 with a median of 0.0028. 0.004 sits
# between them, and the bottom-edge requirement below is what makes the margin
# safe rather than lucky.
HEADWAY_STATIC_DRIFT_FRAC = 0.004
# ...and it must be touching the bottom of the frame. This is what keeps the
# rule pointed at the thing it was written for. A lead vehicle at a locked gap
# on a straight road is the one real object that can hold still in the picture,
# and it is never clipped by the bottom edge unless it is close enough to touch.
HEADWAY_STATIC_REQUIRE_BOTTOM_EDGE = True

# --- the physical floor, and it is a MEASUREMENT veto -----------------------
# A time headway below this, at a speed above the floor above, is not a
# following distance. It is a claim that the car has been a fraction of a
# second behind a vehicle at road speed, and has been for as long as the claim
# has stood.
#
# THE FIRST REAL DRIVE IS THE ARGUMENT. Session 06af3214: gap p50 3.9 m,
# tau p50 0.28 s, 706 frames (36% of the drive) reporting a lead under 6 m
# while doing over 10 m/s, and FIFTEEN of the twenty-two warnings RIO actually
# spoke were about a gap of 3.3-4.7 m at 6-24 m/s. 3.7 m at 20 m/s is 0.19 s.
# Nobody drives at 0.19 s for twenty seconds; the reading was wrong, and it was
# wrong for over a third of the drive.
#
# The root cause is upstream of this file and is recorded here because this
# gate is a symptom filter and should be read as one: the phone was in
# PORTRAIT, so the lower half of every frame is dashboard and hood, and the
# offending lead boxes span the full frame width down to the bottom edge
# (e.g. [0.5, 244.2, 480.0, 639.5] on a 480x640 frame). headway/plausibility.py
# deliberately does not apply its NEAR bound to a vertically truncated box --
# for a good reason, because a genuinely close car is truncated too -- so the
# one check that would have caught it was switched off exactly where it was
# needed. Fixing that belongs to the detector and membership layer.
#
# What this floor does is refuse to SPEAK on such a reading. 0.35 s is below
# human reaction time: a gap that small at road speed is either a collision
# already in progress, in which case a sentence is not the intervention, or a
# measurement that has been falsified by the car not having crashed.
HEADWAY_TAU_IMPLAUSIBLE_S = 0.35

# --- TTC, the urgent trigger ------------------------------------------------
# v2 §1/§9: time to contact under this WITH the gap collapsing is urgent from
# any band, cooldown or not. It was computed on every frame of the first real
# drive, logged on every frame, and never reached the policy at all -- tick()
# was not given it. 41 frames of that drive had TTC under 2.5 s.
HEADWAY_TTC_URGENT_S = 2.5

# --- speed sources: OBD > GPS > coasted GPS > none --------------------------
# Only a source that is a CAR may outrank the phone. The mock and the
# simulation are development paths that happily report 0 mph while parked at a
# desk, and letting one of those beat a live GPS fix would silence every
# warning on a real drive -- the exact opposite of what the priority is for.
HEADWAY_OBD_SPEED_SOURCES = ("live_obd", "live_holley", "replay")
# An OBD reading older than this is not a speed.
HEADWAY_OBD_SPEED_MAX_AGE_S = 1.5
# ...and one this far from the GPS fix is a disagreement rather than a better
# number. Both are then distrusted: the drive continues DEGRADED on the GPS
# value with widened margins, rather than on a bus reading nobody can check.
HEADWAY_SPEED_DISAGREEMENT_MS = 8.0

# A GPS fix older than this is not a speed either...
HEADWAY_V_HOST_STALE_S = 2.0
# ...but it is not nothing. Between STALE and this, the last speed is COASTED
# and the drive continues with widened margins -- see the bias below. A car's
# speed does not change much in eight seconds, and going silent because the sky
# went quiet under a bridge is how a driver learns to ignore the system.
HEADWAY_V_HOST_COAST_S = 8.0

# --- degraded: WIDER, NEVER QUIETER -----------------------------------------
# Added to both band thresholds while the speed is coasted or the sources
# disagree. A speed we are less sure of is a reason to give the driver more
# room, not less -- the same argument navigation already makes with
# NAV_GPS_DEGRADED_BIAS_S, and the same direction.
HEADWAY_DEGRADED_TAU_BIAS_S = 0.75
# The confidence floor is relaxed by this much while degraded, for the same
# reason: the alternative to a slightly-less-certain warning is no warning.
HEADWAY_DEGRADED_CONF_RELIEF = 0.05

# --- coasting a lost lead, scaled by speed ----------------------------------
# The old rule was a flat 1.0 s, which is 30 m of road at 30 m/s and 5 m at
# 5 m/s -- the same number meaning two completely different things. It is a
# DISTANCE budget now, converted to seconds against the speed of the moment,
# and clamped so it never becomes absurd at either end.
HEADWAY_COAST_DISTANCE_M = 18.0
HEADWAY_COAST_MIN_S = 0.6
HEADWAY_COAST_MAX_S = 2.5

# --- temporal confirmation --------------------------------------------------
HEADWAY_CONFIRM_S = 0.5
HEADWAY_CONFIRM_TOLERANCE_S = 0.1
HEADWAY_MIN_CONFIRM_FRAMES = 2
HEADWAY_CONFIRM_DOWN_S = 1.0

# --- voice cooldowns --------------------------------------------------------
HEADWAY_COOLDOWN_CALM_S = 30.0
HEADWAY_COOLDOWN_UNSAFE_S = 15.0
HEADWAY_GENUINE_CLEAR_S = 10.0
HEADWAY_ESCALATE_AFTER_S = 5.0

# --- deferred band entries --------------------------------------------------
HEADWAY_PENDING_ENTRY_MAX_S = 3.0

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

# --- the running observation (observer.py) ----------------------------------
# "What do you see?" was slow, and the measurement said why: the multimodal
# answer is a remote call to a reasoning model — ~1.1 s to the first word, ~2.0
# s in full, and 99% of the wait at p50. That price is right for "what colour
# is the car on the left" and wrong for the question a driver asks most, which
# is the same question about the same road every time.
#
# So during a live conversation the scene is described BEFORE it is asked
# about: Qwen, resident and local, over the newest frame in the ring, about
# once a second. A scene question is then answered out of that cache instead of
# out of a network call.
OBSERVER_ENABLED = True

# ~1 Hz. Measured cost is ~0.4 s of GPU per observation, on the same card the
# 4 fps headway loop is using, so this is a real share of it — and the reason
# the loop runs only while a conversation is open.
OBSERVER_PERIOD_S = 1.0

# How old a cached description may be and still be spoken as current. At 60
# km/h a second is 17 metres, which is the same road; ten seconds is not. Past
# this the fast path declines and the full path looks at the road NOW — the
# refusal is the honesty, not a fallback that got unlucky.
OBSERVER_FRESH_S = 2.0
# ...and how old the newest frame in the ring may be before the observer stops
# spending a forward pass on it at all. Same number, and it is a separate name
# because they answer different questions: FRESH_S is "may this sentence be
# SERVED", MAX_FRAME_AGE_S is "is this picture worth DESCRIBING". On a stalled
# feed the first one was already right and the second one did not exist, so a
# GPU pass went on a road the car had left minutes earlier.
OBSERVER_MAX_FRAME_AGE_S = 2.0

# The observer stops when nobody has asked to see anything for this long. It
# costs GPU that detection, depth and lanes are also asking for.
OBSERVER_IDLE_S = 90.0

# Long edge fed to Qwen for an observation. Measured on this GPU: full frame
# 432 ms, 768 px 377 ms, 512 px 360 ms — with the same sentence coming back.
OBSERVER_MAX_SIDE_PX = 512

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
# --- which model writes a visual answer -------------------------------------
# "openai" sends the crop to a reasoning model. "qwen" answers from the crop
# locally, in a few hundred milliseconds. "auto" picks local for object
# questions and remote for scene ones.
#
# MEASURED, on the same crops, same questions, same ring — five object
# questions, each arm fed fresh frames so none of them was answering from an
# aged-out buffer:
#
#     openai (effort low)   p50 2800 ms   median 32 words
#     openai (effort none)  p50 2897 ms   median 25 words
#     qwen                  p50 1483 ms   median  3 words
#
# Qwen is 1.3 s faster and materially worse, in the way that matters most:
# asked "what kind of car is that" it says "A white sedan." — which is the
# question restated, not answered — where the remote model says "a Toyota
# Camry, likely early-2010s; can't pin the year from this angle". Identifying
# a car IS the object question, so the hop that does it is not redundant.
#
# Turning the thinking pass off does not buy the latency back either (2897 vs
# 2800 ms), so there is no cheap version of the remote answer to prefer.
#
# The local path stays implemented, tested and one env var away
# (RIO_VISUAL_ANSWER_MODEL=qwen) for a deployment that would rather have "a
# white sedan" in 1.5 s than the make in 2.8 s. This one would not.
VISUAL_ANSWER_MODEL = os.getenv("RIO_VISUAL_ANSWER_MODEL", "openai")

# One or two sentences from an 8B model. Long enough for "a white Lexus saloon,
# a couple of cars ahead in the next lane over"; short enough that a rambling
# answer is cut rather than spoken.
VISUAL_QWEN_MAX_TOKENS = 96

# The crop, and the frame it came out of. More than two images and an 8B model
# starts answering about the wrong one.
VISUAL_QWEN_MAX_IMAGES = 2

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
# --- when a fix stops arriving, and what the tracker does about it ----------
#
# THE FIRST REAL DRIVE IS THE ARGUMENT FOR ALL THREE OF THESE. On session
# a2da65cd the browser's single watchPosition delivered a fix at nav clock
# 49.06 and the next one at 261.53 -- 212 SECONDS with no position update --
# while the page went on posting headway frames at 3.9 per second throughout,
# so it was demonstrably alive. On 7e76d316 fixes arrived roughly every six
# seconds, tripping the five-second staleness timer three times in nineteen.
# Not one turn was called on either drive, and no speech was ever queued and
# dropped: NAV_SPEECH_EXPIRED and NAV_SPEECH_INVALIDATED are zero across every
# drive in the log. Nothing reached the arbiter, because the speech planner is
# driven by NAV_PROGRESS and the tracker emits NAV_PROGRESS only when a fix
# arrives.

# How long the BROWSER waits for a fix before tearing down the geolocation
# watch and starting a new one. Shorter than the tracker's staleness timeout so
# the recovery is already under way by the time the tracker notices.
#
# A watch that has gone quiet on iOS does not report an error and does not
# recover on its own; nothing in the old page watched for it, and its error
# callback neither logged nor re-armed.
NAV_GPS_WATCHDOG_S = 4.0
# ...and how many times, before this page accepts that the radio is not coming
# back and stops churning the battery.
NAV_GPS_WATCHDOG_MAX_REARMS = 8
# On the third re-arm and after, high accuracy is dropped. A coarse fix every
# second is worth more to a route tracker than a precise one that never comes,
# and enableHighAccuracy is the setting most often implicated when an iOS watch
# stalls.
NAV_GPS_WATCHDOG_COARSE_AFTER = 2

# How long the TRACKER dead-reckons along the route after fixes stop.
#
# The route is already in hand and following it needs no help -- that was
# always the argument for tracking client-side, and it was only ever half
# implemented: the tracker kept its state but stopped SAYING anything. Inside
# this window it advances along the polyline from the last fix at the last
# known speed and goes on making calls, marked coasted, with every margin
# widened. It does NOT pass maneuvers or arrive on invented progress -- a
# junction is passed when a real fix says so, because coasting past a turn the
# driver never took is worse than a late call.
NAV_GPS_COAST_MAX_S = 20.0

# ---------------------------------------------------------------------------
# THE ANNOUNCEMENT LADDER -- Google Maps' distances, because drivers have them
# ---------------------------------------------------------------------------
# What was here before was a set of three calls timed in SECONDS to the turn
# with metre floors underneath, and on the drive of 2026-09-09 (session
# 738fbb82) it produced this, for one maneuver, in this order:
#
#     t=76.6   "Take the next right onto 14th St."   31.1 m   the instruction
#     t=102.3  "Coming up on a right onto 14th St."  28.2 m   the PREPARATION
#
# The preparation line arrived after the instruction and closer to the
# junction, because at 0.12 m/s both floors were already crossed. That is not a
# tuning miss; a speed-scaled floor is the wrong shape for this problem. Every
# navigation system a driver has used announces at FIXED DISTANCES on a ladder
# chosen by road class, and the value of doing the same is not correctness --
# it is that the driver already knows what each beat means.
#
# So: distances, by road class, from the maneuver. No seconds anywhere.
#
#   SURFACE   half a mile, then 150 m, then the junction
#   HIGHWAY   two miles, then one mile, then a quarter mile, then the gore
#
# JUNCTION is a floor rather than an announcement: it is where the two-word
# confirmation goes IF the near call is far enough behind to make it worth
# having (NAV_JUNCTION_MIN_GAP_S). 150 m on a freeway rather than 35 is the
# same five seconds at 30 m/s that 35 m is at 7.
NAV_TIER_DISTANCES_M = {
    "SURFACE": {"far": 804.7, "near": 150.0, "junction": 35.0},
    "HIGHWAY": {"far": 3218.7, "far_mid": 1609.3, "near": 402.3, "junction": 150.0},
}

# THE RULE THAT KEEPS THE JUNCTION CALL FROM BEING NOISE.
#
# "Turn right onto Ocean Ave." at 150 m and "Turn right." at 35 m are two
# instructions about the same turn, and whether the second one is useful or is
# RIO talking over herself depends entirely on how long ago the first one was.
# In town at 11 m/s the gap is ten seconds and the confirmation is exactly what
# a driver wants at the mouth of the junction. On a freeway at 30 m/s the same
# two distances are four seconds apart and the second one is an interruption.
#
# Ten seconds, which is Google's own behaviour and is long enough that the
# driver has had time to look for the street sign and not find it.
NAV_JUNCTION_MIN_GAP_S = 10.0

# A full instruction takes about two seconds to say. Inside this much lead the
# near call is skipped and the junction gets the two-word line instead -- never
# a longer sentence begun too late to finish before the driver has to act.
NAV_NEAR_MIN_LEAD_S = 4.0

# HOW CLOSE THE NEXT MANEUVER HAS TO BE TO RIDE ON THIS ONE'S NEAR CALL.
# "Turn right onto Ocean, then turn left onto 2nd." Two junctions 120 m apart
# are one move to a driver, and announcing them as two separate events means
# the second call lands while the car is still in the first turn.
NAV_CHAIN_WINDOW_M = 200.0

# Where "Your destination is on the right." goes. The final "You have arrived."
# is fired by the tracker's own arrival, not by a distance.
NAV_ARRIVAL_CALL_M = 150.0

# Imperial or metric. The ladder above is in metres either way -- this decides
# only how the FAR call's distance is SAID. See navigation/distance.py.
NAV_UNITS = "imperial"

# WHEN THE CAMERA IS ASKED ABOUT THE LANDMARK, and it is a distance now like
# everything else on this path. It used to be 11 seconds to the turn, which was
# tuned against a primary call at 6 seconds -- with the near call moved to a
# fixed 150 m, 11 seconds at a crawl is INSIDE it, and the anchor would arrive
# after the sentence it was for had already been spoken.
#
# So: this far BEFORE the near call. 150 m of lead on a surface street puts the
# question at 300 m out, which is about where a forecourt sign becomes
# readable, and on a freeway it scales with the ladder.
NAV_ANCHOR_ACQUISITION_LEAD_M = 150.0
# Clamps, so that a crawl does not announce a turn 4 m ahead and a fast road
# does not announce one from further out than its own ladder allows.
NAV_MIN_CALL_DISTANCE_M = 20.0
NAV_MAX_CALL_DISTANCE_M = 500.0
NAV_FAR_MAX_DISTANCE_M = 4000.0

# ---------------------------------------------------------------------------
# ...AND THE TICK THE CALL ACTUALLY FIRES ON
# ---------------------------------------------------------------------------
# A threshold is checked on a tick, so a call fires on the first tick AFTER the
# car crosses it, never at it. rio_nav.js has said so since it was written --
# "a one-second granularity puts up to a second of error into 'Left here'" --
# and the acceptance pass of 2026-09-09 measured what that costs. Griffith
# Observatory at 25 mph, against a 35 m floor:
#
#     'Left here.'    29.4 m   2.6 s out
#     'Right here.'   24.8 m   2.2 s out
#     'Left here.'    24.0 m   2.2 s out
#
# 6 to 11 metres late, which at 11.2 m/s is one to two ticks. "Left here" is
# the one line in the system whose entire worth is WHEN it arrives: late, it is
# a confirmation of a turn the driver is already in.
#
# So the threshold is LED by what the car covers between the tick that could
# fire it and the next one. The call then fires on the tick BEFORE the
# crossing, landing at or just before the floor rather than after it. Speed
# times these two, added to the distance floor and to the time floor -- the
# time term overshoots for the same reason and needs the same lead.
#
# HALF A SECOND, because that is the interval rio_nav.js ticks the tracker at
# and the granularity every distance term is subject to. Stated here rather
# than left implicit in a setInterval, because it is now load-bearing for a
# call's timing and a change to one without the other reintroduces the lag.
NAV_PROGRESS_TICK_S = 0.5
# ...and the time between the planner deciding and a listener hearing the front
# of the clip. Measured at 0-1 ms across every junction call of that pass --
# the clips are pre-rendered and preloaded, which is what makes it a rounding
# error rather than a term. It is named and added anyway: it is real, it is not
# guaranteed to stay at 1 ms, and a lead that silently omits it is a lead that
# is wrong by however much it grows to.
NAV_CLIP_START_LATENCY_S = 0.05
# ...and a fixed metre or two on top, for the part of the overshoot that is not
# travel at all.
#
# The speed term above is exact when the car is moving: at 21 m/s the junction
# call now goes out at 60 m, led by the time term, and at 11 m/s the lead is
# 6 m of real travel. It collapses to nothing at a crawl -- 1.23 m/s is 0.7 m
# per tick -- and yet the replay still called two junctions at 33.7 m and
# 32.9 m against a 35 m floor while crawling.
#
# That residue is not the tick. Fixes arrive at 1 Hz and the tracker is ticked
# at 2 Hz, so half the progress events are dead-reckoned and the next real fix
# CORRECTS them; the projection can step a metre or two when it lands. A lead
# measured in travel cannot absorb a correction that is not travel.
#
# Two metres, which is under a second even at a crawl and is invisible at
# speed next to the terms above.
NAV_JUNCTION_LEAD_MARGIN_M = 2.0
# Below this, time-to-maneuver stops meaning anything: at 0.2 m/s every
# maneuver is hours away and nothing is ever said, including the turn being
# crept towards in traffic. A floor for the arithmetic, not a claimed speed.
NAV_SPEED_FLOOR_MS = 3.0
NAV_SPEED_NOMINAL_MS = 11.0         # only when there is no speed at all
NAV_DUPLICATE_INSTRUCTION_COOLDOWN_S = 8.0
# A speech candidate is true only inside a window. These are the windows.
NAV_SPEECH_TTL_S = {"depart": 12.0, "far": 10.0, "far_mid": 10.0, "near": 6.0,
                    "junction": 2.5, "arrival": 8.0, "arrived": 8.0}

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
# A VerifiedAnchor'S OWN SHELF LIFE, IN ROAD RATHER THAN IN SECONDS.
#
# Six seconds was the right number when the camera was asked at 11 s to the
# turn and the instruction went out at 6. With acquisition at 300 m and the
# near call at 150 m, six seconds is between eight and eighteen metres of
# travel and the anchor is always stale by the time the sentence it was
# acquired FOR is due -- so every contextual line silently became the canonical
# one, which is a fallback that looks exactly like the feature working.
#
# What actually goes stale is not the clock, it is the distance: "the Shell is
# at this junction" stays true while the junction is still ahead. 400 m covers
# acquisition through the near call with room, and still refuses an anchor
# carried over from a maneuver ago.
NAV_ANCHOR_VALID_FOR_M = 400.0
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


# ===========================================================================
# TEACHER PANEL — two AV foundation models watching the same road, in shadow
# ===========================================================================
# docs/teacher_panel.md is the design. The short version, because every knob
# below only makes sense against it:
#
# Alpamayo 1.5 and Cosmos-Reason2 are NOT part of RIO. They do not warn, they
# do not speak, they do not answer a question, and nothing they produce reaches
# the arbiter, the speech path, look(), or the observer cache. Qwen3-VL-8B
# remains the observer and the only model whose sentences RIO may say. These
# two are READ, RECORDED and DRAWN, so that a drive can be reviewed against
# what two purpose-built driving models saw at the same instant -- and so that
# the recording is a training corpus later.
#
# They also do not live in this process. Each runs in its own environment,
# behind a loopback HTTP service with a bounded queue, so a teacher that hangs,
# leaks or dies takes nothing with it. See tools/teacher_firewall_selftest.py,
# which asserts all of the above from the AST rather than from this comment.
TEACHERS_ENABLED = True

# Where the two services listen. Loopback only -- these are not on the network,
# they have no authentication, and they hand out model weights' opinions about
# the road in front of a real car.
TEACHER_ALPAMAYO_URL = "http://127.0.0.1:8801"
TEACHER_COSMOS_URL = "http://127.0.0.1:8802"

# --- cadence ---------------------------------------------------------------
# A keyframe is the unit of everything here: one instant, one 4-frame window,
# one ego history, two readings, one corpus row.
#
# TWO WAYS ONE HAPPENS. An EVENT -- a band change, a nav maneuver, an IMU jolt,
# a driver question -- is the interesting instant and is why this exists at all.
# The FLOOR is what stops a quiet motorway producing nothing to compare: a
# keyframe every so often whether or not anything happened.
TEACHER_KEYFRAME_FLOOR_S = 2.0
# ...and a ceiling on how close together two keyframes may be however many
# events fire. Without it a burst of band flapping at a junction would queue
# faster than either model can answer, and every one of them would be dropped
# stale a second later -- which is correct, and is also a lot of work done to
# reach the same place as not asking.
TEACHER_KEYFRAME_MIN_GAP_S = 1.0

# --- freshness -------------------------------------------------------------
# THE GATE THAT MAKES THIS SAFE TO LOOK AT. A reading is a claim about a
# picture; a picture more than this old is a claim about a road the car has
# left. Enforced twice -- when the keyframe is built (is t0 current?) and when
# the job is picked up by the worker (is it still current?) -- because the
# queue is where the second of delay actually accumulates.
TEACHER_FRAME_MAX_AGE_S = 1.0
# How long a reading stays on the dashboard before the card says so. Longer
# than the frame gate on purpose: a 3 s old reading is still worth SEEING next
# to the other model's, it is just not worth believing about the road NOW.
TEACHER_READING_FRESH_S = 4.0

# --- the shared input ------------------------------------------------------
# Four frames at t0-0.3, t0-0.2, t0-0.1, t0 -- the window Alpamayo's own loader
# builds (num_frames=4, time_step=0.1) and the window Cosmos is given too, so
# the two readings are of the same instant and are directly comparable.
TEACHER_WINDOW_FRAMES = 4
TEACHER_WINDOW_STEP_S = 0.1
# How far a real frame may sit from its nominal slot before the window stops
# being the window the model expects.
#
# DERIVED, NOT TUNED. Past half a step (50 ms) the nearest frame to one slot is
# nearer to its neighbour, so the "window" is no longer four evenly spaced
# instants -- it is some frames counted twice and some skipped. 60 ms allows a
# little jitter around that boundary and nothing more.
#
# What it does NOT do is refuse the keyframe. At the socket's 8-15 fps the
# spacing is 65-125 ms and the worst slot error comes out at ~50 ms, so the
# window is exact; at the POST fallback's 4 fps it comes out at 100 ms and the
# window is marked `exact: false` in the row. Both are sent -- an inexact
# window is still a real 0.3 s of road and still worth two readings. What must
# never happen is one quietly meaning the other.
TEACHER_WINDOW_SLOT_TOLERANCE_S = 0.06

# Ego-motion history: 16 poses at 10 Hz ending at t0, in the ego frame at t0 --
# again the shape Alpamayo's loader produces (num_history_steps=16).
TEACHER_EGO_HISTORY_STEPS = 16
TEACHER_EGO_HISTORY_STEP_S = 0.1
# An ego sample older than this is not history, it is a different drive. The
# client posts at ~1 Hz in batches of ~10 samples, so two missed posts is the
# limit before the history is reconstructed from speed alone and marked as
# such.
TEACHER_EGO_SAMPLE_MAX_AGE_S = 3.0
TEACHER_EGO_RING = 256                # ~25 s of 10 Hz samples

# --- the services ----------------------------------------------------------
# ONE JOB IN FLIGHT, ONE WAITING, NEWEST WINS. The same rule the frame
# transport runs on and for the same reason: a queued keyframe is measured
# against a road that has already gone past. Two rather than one so a model
# that is mid-generation has something to start on the instant it finishes.
TEACHER_QUEUE_DEPTH = 2
TEACHER_HTTP_TIMEOUT_S = 90.0
# A service that has failed this many times in a row is left alone until the
# backoff expires. Neither model is load-bearing, and hammering a loopback
# port that is not listening is noise in the log and nothing else.
TEACHER_FAIL_BACKOFF_S = 15.0
TEACHER_FAIL_STREAK = 3

# --- recording -------------------------------------------------------------
# Per drive session, on the persistent volume, schema'd so it is usable as a
# training corpus later rather than as a debug dump now. See teachers/corpus.py
# and TEACHER_CORPUS_SCHEMA in teachers/schema.py.
TEACHER_CORPUS_ENABLED = True
TEACHER_CORPUS_DIR = "/workspace/rio-phase1/training_data/teachers"
# The four JPEGs are what makes a row trainable, and they are also the entire
# size of it: ~24 kB each, ~100 kB a keyframe, ~180 MB an hour at the 2 s
# floor. Kept, because a corpus without the pictures is a corpus of opinions.
TEACHER_CORPUS_KEEP_FRAMES = True

# --- what is asked ---------------------------------------------------------
# ONE PROMPT SET, BOTH MODELS, WORD FOR WORD. The whole point of the panel is
# that the two columns are answers to the same question; a prompt tuned for
# one of them would make the comparison a comparison of prompts.
# TWO OF THESE ARE NOT THE OBVIOUS WORDING, AND THAT IS THE POINT.
#
# "Describe the scene." is the natural question and it is the one NVIDIA's own
# VQA notebook uses. Asked of Alpamayo on a real road it returns, every time,
# a string out of a TRAINING LABEL SET rather than a description:
#
#     "A vehicle controls loss. The vehicle driver is in distracted driving."
#     "A vehicle changes lanes with the same direction to ego-car/another
#      vehicle. Vehicles do not notice the coming vehicles when turning or
#      changling lanes"
#     "Lead vehicle stops. The vehicle driver is in distracted driving."
#
# Eight of ten keyframes on the acceptance clip returned the first one
# VERBATIM, and the giveaways are in the characters: NON-BREAKING SPACES
# between the words, and "changling" -- a typo carried out of a label
# spreadsheet. It is scenario-category recall, not perception, and it happens
# on every sample, seeded or not.
#
# This project has met that failure before from the other side: the observer's
# prompt ended with four example sentences and Qwen returned the first one
# whatever the frame was (rio_prompts.is_prompt_example, vision.parroted).
# Same shape, different memory -- there the prompt's examples, here the
# training set's labels.
#
# Rewording fixes it completely. Asked "Describe the road, the traffic and the
# weather", the SAME model on the SAME four frames answers:
#
#     "The scene shows a clear day with good visibility... The road appears to
#      be a multi-lane highway with a concrete divider..."
#
# So the wording below is chosen to elicit perception from BOTH models rather
# than to be the tidiest phrasing, and it is still ONE prompt set asked of both
# word for word -- which is the property that matters. See teachers/canned.py,
# which flags a reading that looks recited so this cannot come back silently,
# and docs/teacher_panel.md §6c for the A/B.
TEACHER_PROMPTS = {
    # NOT "Describe the scene." -- see above.
    "scene": "Describe the road, the traffic and the weather.",
    "critical_actor": (
        "Which single road user is the most safety-relevant to the ego "
        "vehicle right now, and why?"
    ),
    # NOT "What should the ego vehicle pay attention to in the next 2 seconds?"
    # -- that one returns "The vehicle should pay attention to vehicles
    # changing lanes" on nine keyframes out of ten. Asking for the hazard AND
    # its position gets a real answer from both models, and gives the actor
    # association something to work with.
    "attention": (
        "What is the main hazard the ego vehicle should watch in the next two "
        "seconds, and where is it?"
    ),
}
# COSMOS'S OWN ROW, AND THE WORDING IT TOOK TO GET IT.
#
# Alpamayo's unique column is derived arithmetic (teachers/decision.py, from
# its predicted trajectory). Cosmos's has to be asked for, and the obvious
# phrasings do not get it.
#
# "What is moving in this scene, what is about to happen next, and is that
# physically plausible? Think step by step." -- the shipped wording -- returns
# FIRST-PERSON DRIVING NARRATION, because that is what the model was
# post-trained to produce:
#
#     "I am driving on the first lane of a 5-lane highway. I maintain a
#      constant speed and continue to drive straight on the same lane. I keep
#      an eye on the surroundings for any sudden movements..."
#
# True, and useless: it is the ego vehicle's intentions, which RIO already
# knows, instead of the other road users' motion, which is the one thing
# Cosmos can say that nothing else here can. It also ran to 28 s, filling its
# budget -- and once produced 88 consecutive copies of one sentence.
#
# Three things fixed it, and all three are load-bearing:
#   "in the third person" and an explicit "Do not use I or my" -- 0 first-person
#      references across three runs, against a majority before.
#   naming the FIELDS wanted per actor (position relative to ego, direction,
#      rough speed, next two seconds) -- so it enumerates road users instead of
#      narrating a drive.
#   a line beginning "Plausibility:" -- which gives the parser something
#      deterministic to split on rather than hunting for a verdict in prose.
#
# Measured on the clean clip: 1.6-6.2 s, no sentence repeated, and the
# Plausibility line present on every run. See docs/teacher_panel.md.
TEACHER_COSMOS_PHYSICAL_PROMPT = (
    "Describe, in the third person, each road user other than the ego "
    "vehicle: where it is relative to the ego vehicle, which way it is moving "
    "and roughly how fast, and what it will most likely do in the next two "
    "seconds. Do not use \"I\" or \"my\". Finish with a line beginning "
    "\"Plausibility:\" stating whether anything you have described could not "
    "physically happen."
)

# The marker the answer is split on. One string, in one place, because the
# prompt above asks for it by name and the parser looks for it by name.
TEACHER_PLAUSIBILITY_MARKER = "Plausibility:"

# Generation. Same numbers NVIDIA's own examples use, so a reading here is
# comparable with a reading from the model card.
TEACHER_TOP_P = 0.98
TEACHER_TEMPERATURE = 0.6
TEACHER_MAX_NEW_TOKENS = 256
# COSMOS'S REASONING BUDGET, AND WHAT IT COSTS -- measured, on a real road clip
# at 1282x684, not guessed:
#
#   physical reasoning   28.4 s   4275 characters (the full 1024-token budget)
#   scene                 2.2 s    399
#   critical actor        1.0 s    149
#   attention             0.9 s    152
#
# So Cosmos is a ~32 s teacher, and one question is 88% of it. That is not
# waste -- the 4275 characters ARE the physical-reasoning trace this panel
# exists to collect, and truncating them to fit a 2 s floor would be spending
# the GPU to produce a worse version of the one output nobody else can give us.
#
# The consequence, stated plainly because it is a design choice and not a bug:
# at the 2 s floor Cosmos answers roughly one keyframe in fifteen and the rest
# are evicted (newest wins, so it is always working on the most recent road).
# Every keyframe it DOES answer is also one Alpamayo answered, because Alpamayo
# is faster and gets the same jobs -- so every Cosmos reading is a comparison,
# there are just fewer of them. Over a twenty-minute drive that is ~40
# side-by-side readings, which is plenty for the purpose.
#
# Lower this and you get more readings and shorter traces. That is the trade,
# and it is an operator's to make rather than one to make silently here.
TEACHER_COSMOS_MAX_NEW_TOKENS = 1024
TEACHER_TRAJ_SAMPLES = 1               # display-only; 1 keeps VRAM at ~24 GB
