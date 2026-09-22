"""eyeread.py — the eye, asked NVIDIA's question about a stretch of road.

WHAT CHANGED, IN ONE PARAGRAPH
------------------------------
The eye used to be handed one still and a question we wrote, and it answered
with space it had guessed at: a pickup truck that was not in the frame, a rule
about overtaking on the median that nobody asked for, "car 20 km/h" on a
picture that cannot show a speed. It is now handed six seconds of video and
every number this car actually measured over those same six seconds, and asked
NVIDIA's own autonomous-vehicle question. It may reason about those numbers.
It may not produce numbers of its own.

WHY NVIDIA'S PROMPT AND NOT OURS
--------------------------------
Because ours kept being the fault. Two of them, both measured:

  a FIELD TEMPLATE (ROAD/TRAFFIC/RISK, 23c3dcd) taught a 2B model to complete
  a form, and a model completing a form completes the EXAMPLE in the form --
  a hundred and fifty readings with "vehicles that matter" in the traffic
  field, and the prompt's own worked example returned verbatim on frames that
  had none of it in view.

  a LEADING QUESTION ("describe the road ahead and any other road users")
  presupposes a road and other road users. Asked that about a blank frame it
  described a road flanked by green grass verges, because the question had
  already told it there was one.

prompts/av_cot.yaml is NVIDIA's, it is one sentence, it names no fields, and
it presupposes only what the camera is mounted to. The two additions below are
the two things the spec asks for, written so that "nothing" is a complete
answer to either.

WHAT THIS IS NOT ALLOWED TO DO
------------------------------
Trigger speech, change a band, or reach the warning path. The deterministic
headway loop is unchanged and this reads nothing into it. A reading here is
evidence on a card and, when it is fresh, evidence RIO may be handed. That is
the whole of its authority, and the corroboration counts this module produces
are the thing that has to look good before anybody argues for more.
"""
import re
import threading
import time
from pathlib import Path

import config
import grounding

# ---------------------------------------------------------------------------
# The prompt.
# ---------------------------------------------------------------------------
# NVIDIA's, verbatim, from prompts/av_cot.yaml. Kept as its own constant so
# that a diff against their repo is one line and so nobody has to work out
# later which half of the prompt we wrote.
AV_COT = ("The video depicts the observation from the vehicle's camera. You "
          "need to think step by step and identify the objects in the scene "
          "that are critical for safe navigation.")

# THE SECOND ASK, AND THE PERMISSION TO SAY NOTHING.
#
# Both questions the spec asks for have to be answerable with "nothing", and a
# model will not answer "nothing" unless it is told that is an answer -- the
# blank-frame fault was exactly a model that had no permitted way to say the
# picture was empty. This says it twice, once per question, because a single
# general permission at the end was read as applying only to the last one.
ASKS = ("Answer two things: what in this scene matters and why, and what the "
        "driver should be aware of.\n\n"
        "If nothing in the scene is critical for safe navigation, say that — "
        "it is a complete answer. If there is nothing the driver needs to be "
        "aware of, say that. Neither is a failure to answer.")

# THE NUMBERS RULE, which is the whole point of the grounding block.
#
# Phrased as what it KNOWS rather than as a formatting ban, because a ban on
# digits produced readings that spelled numbers out in words. "You do not know
# it" is the true statement and it is the one that generalises.
NUMBERS_RULE = (
    "Measured state from this vehicle's own sensors follows, covering the same "
    "seconds as the video. You may reason about those numbers and repeat them. "
    "Any distance, speed or time to contact that is not among them is something "
    "you do not know: do not estimate one. When something in the video matters, "
    "say which measured track it is, by its track number.")

# Verbatim from the model card, and already in vision.py for the still path.
# Repeated rather than imported so this file can be read as one prompt.
COSMOS_FORMAT = ("Answer the question using the following format:\n\n<think>\n"
                 "Your reasoning.\n</think>\n\nWrite your final answer "
                 "immediately after the </think> tag.")

SYSTEM = "You are a helpful assistant."


# ---------------------------------------------------------------------------
# THE THREE ARMS.
# ---------------------------------------------------------------------------
# The question is the variable; everything around it is held constant, so a
# difference between two arms is a difference in the question and not in the
# scaffolding. What is held constant: the system prompt, the numbers rule, the
# measured block, the judgement tail and NVIDIA's answer-format instruction.
#
# WHAT NVIDIA'S EMBODIED PROMPT ACTUALLY IS, because it decides arm B and it
# is not what "embodied agent" suggests:
#
#     prompts/embodied_reasoning.yaml, in full:
#         What can be the next immediate action?
#
# There is NO PERSONA IN IT. It does not tell the model it is a robot, a car,
# or anything else. The embodiment is in the TASK -- predict the next step
# from this vantage point -- not in a character the model is asked to play.
# That matters for the gate: the blank-frame fault came from a question that
# presupposed a road ("describe the road ahead and any other road users"), and
# NVIDIA's embodied prompt presupposes nothing at all. It is the safest of the
# three on that axis by construction, and the most dangerous on another -- see
# the actuation note on arm C.
ARM_A = AV_COT + "\n\n" + ASKS

# NVIDIA'S, VERBATIM, NOT ONE WORD ADDED. Including no actuation clause: the
# spec forbids control recommendations, and the honest way to find out whether
# NVIDIA's shipped prompt produces them is to ASK IT AS SHIPPED and count,
# not to muzzle it in the prompt and then report that it stayed quiet. The
# muzzle exists as a counted guard (`control_decision`) applied to every arm
# equally, which is the measurement the spec actually wants.
ARM_B = "What can be the next immediate action?"

# ARM C: B, ADAPTED. THE DIFF, AND WHY EACH LINE OF IT EXISTS.
#
#   + "The video depicts the observation from the vehicle's camera."
#       Verbatim from NVIDIA's own prompts/av_cot.yaml. Not written here --
#       arm B alone never says what the camera is attached to, so "the next
#       immediate action" is as likely to be read as a robot arm's as a
#       vehicle's. This is the minimum context that makes the question about
#       a forward-facing vehicle, and it is still NVIDIA's sentence.
#
#   ~ "action"  ->  "development that this vehicle would need to be ready for"
#       THE ONE CHANGE THE SPEC REQUIRES. "Action" asks what to DO; RIO does
#       not actuate, and an answer phrased as a control decision is out of
#       scope however good it is. "Development to be ready for" keeps the
#       forward-looking, next-step quality that makes the prompt embodied --
#       it is still about what happens next -- while moving the subject from
#       the vehicle's controls to the road's behaviour.
#
#   + "Do not recommend a steering, throttle or braking action."
#       Belt and braces on the same rule, stated as a prohibition because the
#       rewrite above is a hint and this is a requirement. Counted either way.
#
#   + the permission to say nothing
#       Carried over from arm A for one reason: the blank frame. A question
#       about what happens next, asked of a dead camera, has to have "nothing
#       is happening and I cannot see" available as an answer or the model
#       will manufacture a development. This is the clause that decides the
#       gate, and leaving it out of C would be testing a prompt nobody would
#       ship.
ARM_C = ("The video depicts the observation from the vehicle's camera. "
         "What can be the next immediate development that this vehicle would "
         "need to be ready for?\n\n"
         "Do not recommend a steering, throttle or braking action — this "
         "system advises, it does not drive.\n\n"
         "If nothing in the video needs to be prepared for, say that — it is "
         "a complete answer.")

# ---------------------------------------------------------------------------
# ARM D: the RIO Road Hazard Reasoning Engine prompt, as written.
# ---------------------------------------------------------------------------
# HELD IN A FILE, NOT A STRING LITERAL. It is four thousand tokens of
# specification and it will be edited by people who are not editing Python;
# a .txt diffs cleanly, and keeping it out of here stops this module becoming
# mostly prompt.
#
# IT GOES IN THE SYSTEM SLOT, because it says "SYSTEM ROLE" on its first line
# and because that is what it is -- a standing description of the layer's job,
# not a question about this video. NVIDIA's own shape is a system prompt plus
# a user turn carrying the task, so this substitutes for "You are a helpful
# assistant." and the user turn still carries the measured state, the
# question and the answer format.
#
# WHAT IT COSTS, MEASURED WITH THE MODEL'S OWN TOKENIZER: 4340 tokens. The
# video window is ~2636 with its grounding block, so a grounded arm-D reading
# prefills ~6976 tokens before it generates a word -- against NVIDIA's
# recommended max_model_len of 8192 for this checkpoint. It fits. It fits
# with about a thousand tokens to spare, and the reading budget of 1536 does
# not fit inside that spare thousand.
_HAZARD_PROMPT_PATH = Path(__file__).resolve().parent / "prompts" / "rio_hazard_engine.txt"

try:
    ARM_D = _HAZARD_PROMPT_PATH.read_text().strip()
except OSError:
    ARM_D = ""

# THE TWO WORKED EXAMPLES AT THE END OF IT, pulled out so they can be watched
# for coming back. READ FROM THE PROMPT rather than copied here, so editing
# the prompt cannot leave this detector watching for examples that are no
# longer in it.
#
# THEY WERE REPLACED, AND WHY. The originals were a pedestrian behind a parked
# SUV and brake lights compressing ahead on a freeway -- both generic daylight
# road scenes that could plausibly be true of our own footage, which makes a
# copy indistinguishable from a correct reading. They were copied: on the
# FROZEN window, where nothing moves at all, arm D reported "multiple brake
# lights activating (Tracks 48 & 49)"; on the suburban window it reported a
# "parked SUV" occluding a driveway and "potential emergence of
# pedestrians/children" on a street holding nine tracks and no pedestrian.
#
# The replacements teach the same two lessons -- occlusion, and looking past
# the lead -- in scenes that share no vocabulary with sunny multi-lane
# California highway: a gritter stopped on a snowbound alpine road at night,
# and a tractor ahead of a level crossing in fog at dusk. If either comes back
# now it is unmistakable rather than plausible, which is the whole point of
# the substitution.
ARM_D_EXAMPLES = []
try:
    import rubric as _rubric_for_examples
    ARM_D_EXAMPLES = list(_rubric_for_examples.WORKED_EXAMPLES)
except Exception:
    pass


# ---------------------------------------------------------------------------
# ARM E: the same specification, distilled to what a 2B can hold.
# ---------------------------------------------------------------------------
# WHAT WAS KEPT, and it is exactly the three things the brief names:
#
#   THE BOUNDARY RULE    "not the collision-warning system ... separate
#                        deterministic systems ... they decide" -- the single
#                        most important sentence in the whole document,
#                        because it is what stops the layer competing with
#                        the headway loop.
#   THE SILENCE DEFAULT  "silent by default", plus the false-positive control
#                        list compressed from ten bullets to one sentence
#                        naming the same categories.
#   THE LANGUAGE RULES   the hedging vocabulary verbatim ("appears",
#                        "possible", "likely", "developing", "partially
#                        obscured", "visibility limited"), the ban on
#                        dramatic language, and the ban on inventing
#                        certainty.
#
# WHAT WAS DROPPED, AND WHY EACH DROP IS THE POINT RATHER THAN A COMPROMISE:
#
#   THE TWENTY-THREE EDGE-CASE SECTIONS. Every one is a list of hazards to
#   watch for -- deer, a ball, a stroller, flaggers, steel plates, standing
#   water, school buses. That vocabulary is the specific risk this arm exists
#   to isolate: a 2B model handed a list of plausible hazards has a list of
#   plausible hazards to report. Arm D keeps them and arm E does not, and the
#   borrowing count is the difference between the two.
#
#   THE TWO WORKED EXAMPLES. 23c3dcd. Nothing else to say about it.
#
#   THE OUTPUT SCHEMA. Measured, in this file's JUDGEMENT note: a schema
#   anywhere in the prompt returns the schema and no reading. It is enforced
#   by guided decoding on the label pass instead, which is the only place a
#   schema belongs.
#
#   THE ROAD-CONTEXT AND ROAD-USER TAXONOMIES. Twenty-three road types and
#   seventeen road-user kinds are a vocabulary to choose from, and the
#   detector already says which of six classes are actually present.
ARM_E = (
    "You are the visual reasoning layer for a driving co-pilot. You do not "
    "drive the vehicle and you are not the collision-warning system: "
    "separate deterministic systems handle tracking, depth, time to "
    "collision and following distance, and they decide whether a warning is "
    "given. Your job is to give them semantic context.\n\n"
    "Be silent by default. Normal driving is not a stream of warnings. Do "
    "not elevate vehicles travelling normally in adjacent lanes, pedestrians "
    "safely on sidewalks, ordinary curves, ordinary braking, parked vehicles "
    "showing no activity, or distant objects with no plausible interaction. "
    "Something matters only when motion, geometry, behaviour, visibility or "
    "context creates a plausible conflict.\n\n"
    "Use precise, neutral language: appears, possible, likely, developing, "
    "partially obscured, visibility limited. Avoid dramatic language. Never "
    "invent certainty where the visual evidence is weak, and do not claim an "
    "unseen person or object exists — say what is restricting visibility "
    "instead.\n\n"
    "Say what in this scene is becoming important, and why. If nothing is, "
    "say that — it is a complete answer.")


ARMS = {"A": ARM_A, "B": ARM_B, "C": ARM_C, "D": ARM_D, "E": ARM_E}

# WHICH ARMS PUT THEIR TEXT IN THE SYSTEM SLOT. D declares itself a SYSTEM
# ROLE and is one; everything else is a question and belongs in the user turn
# behind NVIDIA's own "You are a helpful assistant."
ARM_SYSTEM = {"D": ARM_D}

# WHAT THE USER TURN ASKS WHEN THE ARM IS A STANDING ROLE RATHER THAN A
# QUESTION. D and E describe a job; they never actually ask anything about
# THIS video, and a model given a role and no question answers whatever it
# likes. NVIDIA's av_cot line is the question, so the role arms get it.
ARM_QUESTION_FALLBACK = AV_COT

# A PROMPT'S OWN WORKED EXAMPLES, so they can be watched for coming back.
# Empty for A, B, C and E: none of them contains one, which is itself the
# point -- an arm with no example cannot have its example copied, and D's two
# are generic road scenes that could match our footage.
ARM_EXAMPLES = {"A": [], "B": [], "C": [], "D": ARM_D_EXAMPLES, "E": []}
ARM_NAMES = {
    "A": "plain question (av_cot + two asks)",
    "B": "NVIDIA embodied_reasoning, verbatim",
    "C": "embodied_reasoning adapted to a forward-facing vehicle",
    "D": "RIO Road Hazard Reasoning Engine, full",
    "E": "RIO Road Hazard Reasoning Engine, distilled",
}
DEFAULT_ARM = "A"


# ---------------------------------------------------------------------------
# THE JUDGEMENT TAIL.
# ---------------------------------------------------------------------------
# A SCHEMA IS NOT THE TEMPLATE THAT BROKE THIS BEFORE, and the difference is
# worth being precise about, because "no field template" is a rule this repo
# earned the hard way. What broke at 23c3dcd was a template WITH A WORKED
# EXAMPLE IN IT -- a filled-in reading of a road that was not in front of the
# car -- and a 2B model completing a form completes the example. There is no
# example here. There is a key list and an enum, which is content-free: there
# is no road in "none|watch|advise|urgent" to hand back.
#
# ASKING FOR JSON IS NVIDIA'S OWN IDIOM, not an invention of ours. Three of
# their shipped prompts do it: robot_cot.yaml ('{"point_2d": [x, y], "label":
# ...}'), describe_anything.yaml ("subject_id", "category", "caption") and
# temporal_localization.yaml ("start", "end", "caption"). mvp_bench.yaml uses
# a bare enum ((A) Possible / (B) Impossible). So both halves of this are
# shapes these weights were demonstrated on.
#
# IDENTICAL ACROSS ALL THREE ARMS. The arms differ in the question and in
# nothing else, or the comparison measures the scaffolding.
# THE JUDGEMENT IS A SECOND PASS, AND IT HAS TO BE. MEASURED:
#
#   schema at the END of the prompt      prose 0 chars, 0 chars
#   schema in the MIDDLE of the prompt   prose 0 chars, 0 chars
#   no schema at all                     prose 808 chars, 246 chars
#
# Four readings with a JSON schema anywhere in the prompt returned the object
# alone and answered nothing; two without it answered normally. Placement is
# not the variable -- the presence of a schema is. A 2B model given a form
# and a question fills in the form, which is the 23c3dcd fault exactly, and
# no amount of "the prose answer is what is wanted" in front of it helped.
#
# So the question and the label are two generates. The cost is a second
# prefill of the same window; the alternative is a judgement bought by
# destroying the reading it is supposed to be a judgement OF.
#
# THE SECOND PASS SEES THE VIDEO AGAIN, not just the prose. A risk level
# derived from the model's own words would be a judgement about a paragraph;
# this is meant to be a judgement about a road. It also gets the measured
# block and its own first answer, so the label is consistent with the
# reading rather than a second unrelated opinion.
# THE PROSE IS THE ANSWER; THE JSON IS A LABEL ON IT.
#
# The first version of this tail said only "after your answer, give a JSON
# object". Measured on the suburban window, all six readings came back as the
# JSON object ALONE, fenced, with no prose at all -- a 2B model handed a
# schema at the end of a long prompt answers the schema and drops the
# question. That is the 23c3dcd form-completion fault wearing a different
# hat, and the fix is the same shape: make the thing we actually want the
# thing the prompt asks for first and most plainly, and demote the schema to
# an afterthought about it.
JUDGEMENT = (
    "Answer the question above in your own words first. That prose answer is "
    "what is wanted; the object below is only a label on it and must not "
    "replace it.\n\n"
    "Then, on a new line after your answer, give a JSON object and write "
    "nothing after it:\n"
    '{"risk": ..., "should_speak": ..., "why": ..., "about": [...]}\n'
    '"risk" is exactly one of: none, watch, advise, urgent.\n'
    '"should_speak" is true or false: whether speaking a warning aloud '
    'to the driver is justified right now.\n'
    '"why" is one short sentence, in your own words, for that '
    'decision.\n'
    '"about" lists the track numbers the judgement concerns, and is '
    'empty if it concerns none.')

# The scale, in order. `none` and `watch` are silent by construction; the
# judgement may still say should_speak on them and that disagreement is itself
# worth counting, so nothing here enforces the pairing.
RISK_LEVELS = ("none", "watch", "advise", "urgent")

# WHERE THIS SCALE CAME FROM, AND WHERE IT DOES NOT COME FROM.
#
# It is the spec's, given as "the same scale as the post-training target".
# It does not exist anywhere else in this repository -- not in the training
# data under training_data/, not in teachers/, and not in headway/, whose
# bands are COMFORTABLE/NORMAL/GETTING_UNSAFE/UNSAFE/CRITICAL plus an URGENT
# override. Checked before adopting it, and said here so that nobody later
# reads these four words as a scheme this code already labelled against. The
# mapping to the bands used for shadow scoring is BAND_SPEAKS below, and it
# is deliberately the only bridge between the two vocabularies.
def build_prompt(grounding_text: str = "", arm: str = DEFAULT_ARM,
                 judgement: bool = False) -> str:
    """The user turn for the READING pass, in NVIDIA's order. -> str.

    `judgement` defaults to False: the label is its own pass now, and putting
    the schema back in here is what the measurement above says not to do. The
    flag survives so the A/B that produced that measurement can be re-run.
    """
    body = ARMS.get(arm, ARM_A)
    if arm in ARM_SYSTEM:
        # The arm's text is the system prompt; the user turn needs a question.
        body = ARM_QUESTION_FALLBACK
    parts = [body]
    if grounding_text:
        parts.append(NUMBERS_RULE)
        parts.append(grounding_text)
    if judgement:
        parts.append(JUDGEMENT)
    parts.append(COSMOS_FORMAT)
    return "\n\n".join(parts)


def build_judgement_prompt(reading: str, grounding_text: str = "") -> str:
    """The user turn for the LABEL pass. -> str."""
    parts = ["The video depicts the observation from the vehicle's camera. "
             "You have already described it as follows.",
             "Your description:\n" + (reading or "").strip()]
    if grounding_text:
        parts.append(NUMBERS_RULE)
        parts.append(grounding_text)
    parts.append(JUDGEMENT)
    parts.append(COSMOS_FORMAT)
    return "\n\n".join(parts)


def instructions_only(grounding_text: str = "", arm: str = DEFAULT_ARM):
    """The prompt MINUS the measured block, for the echo guard.

    THIS DISTINCTION IS NOT PEDANTRY. The echo guard refuses a reading that
    hands the prompt's own words back as an answer, and it was written when
    every word of the prompt was an instruction. Half the prompt is now
    MEASUREMENTS, and repeating those is not an echo -- it is the thing the
    model was explicitly told it could do. Comparing against the whole prompt
    would refuse every correctly grounded reading, which is the guard firing
    on exactly the behaviour it was supposed to encourage.
    """
    body = ARM_QUESTION_FALLBACK if arm in ARM_SYSTEM else ARMS.get(arm, ARM_A)
    return "\n\n".join([body, NUMBERS_RULE, JUDGEMENT, COSMOS_FORMAT])


def system_for(arm: str) -> str:
    """The system turn for this arm. -> str."""
    return ARM_SYSTEM.get(arm) or SYSTEM


# ---------------------------------------------------------------------------
# NVIDIA separates the trace from the answer. So do we, the same way.
# ---------------------------------------------------------------------------
# vLLM does this with `--reasoning-parser qwen3` and returns the trace as its
# own field on the message, beside `content`. MEASURED ON THIS POD, vllm 0.29:
# the field is `message.reasoning` and `usage.completion_tokens_details.
# reasoning_tokens` counts it; NVIDIA's own sample log, taken on an older
# build, shows `reasoning_content` as well. Anything reading that response has
# to accept both names.
#
# We are not on vLLM in the live server -- see the runtime note at the bottom
# of this file -- so this is the same split done locally, and deliberately the
# same OUTPUT SHAPE. The card renders `reasoning` and `answer` exactly where
# their CLI prints `Reasoning:` and `Assistant:`, so moving the live path onto
# vLLM later changes the producer and not the card.
#
# THE OLD BEHAVIOUR WAS TO THROW THE TRACE AWAY AND COUNT IT AS A FAULT. That
# made sense when the prompt asked for no trace. The prompt now asks for one,
# NVIDIA's format asks for one, and the trace is the most inspectable thing a
# reasoning model produces -- it is where a fabrication is visible BEFORE it
# reaches the answer. Discarding it was throwing away the evidence.
_THINK_BLOCK = re.compile(r"<think>(.*?)</think>", re.S)
_THINK_OPEN = re.compile(r"<think>(.*)$", re.S)


def split_reasoning(raw: str):
    """-> (reasoning, answer, unterminated).

    An unterminated <think> still yields its partial reasoning, because that is
    worth showing on a card marked as cut off -- but the answer is empty, and
    the caller refuses the reading. A trace that filled the budget means the
    model never got to the answer, and there is nothing after it to publish.
    """
    t = (raw or "").strip()
    if "<think>" not in t:
        return "", t, False
    m = _THINK_BLOCK.search(t)
    if m:
        reasoning = m.group(1).strip()
        answer = _THINK_BLOCK.sub("", t).strip()
        if "<think>" in answer:
            mo = _THINK_OPEN.search(answer)
            return reasoning, _THINK_OPEN.sub("", answer).strip(), bool(mo)
        return reasoning, answer, False
    mo = _THINK_OPEN.search(t)
    return (mo.group(1).strip() if mo else ""), "", True


# ---------------------------------------------------------------------------
# The numbers rule, enforced.
# ---------------------------------------------------------------------------
# A number with a unit attached to it. Deliberately not every digit: "three
# lanes" and "track 7" are not measurements and refusing them would be a guard
# that fires on the model doing what it was told. What is being looked for is
# a QUANTITY OF SPACE, TIME OR SPEED, which is the class of thing a camera
# cannot show and geometry can.
_NUM_UNIT = re.compile(
    r"(\d+(?:\.\d+)?)\s*"
    r"(m/s|mps|km/?h|kph|mph|metres|meters|metre|meter|feet|ft|seconds|second|secs|sec|m|s)\b",
    re.I)


def check_numbers(answer: str, supplied: list):
    """Every measurement in the answer, judged against what we gave it.

    -> (clean_answer, sourced, invented)

    `sourced` is a restatement of a number we supplied and is LEFT IN the text:
    it is the model using its grounding, which is the behaviour being asked
    for, and stripping it would leave a reading that is vaguer than the truth.
    It is marked so the card can say the number came from geometry rather than
    from the model, which is the distinction a driver actually needs.

    `invented` is stripped, exactly as an invented measurement is stripped
    today -- the existing rule, applied to a model that now has no excuse.
    """
    sourced, invented, out, last = [], [], [], 0
    for m in _NUM_UNIT.finditer(answer or ""):
        value, unit = float(m.group(1)), m.group(2)
        hit = grounding.matches_supplied(value, unit, supplied or [])
        if hit:
            sourced.append({"text": m.group(0), "value": value, "unit": unit,
                            "from": hit["what"], "supplied": hit["value"]})
            continue
        invented.append({"text": m.group(0), "value": value, "unit": unit})
        out.append(answer[last:m.start()])
        last = m.end()
    if not invented:
        return (answer or ""), sourced, invented
    out.append(answer[last:])
    # Leave a mark rather than a hole. A sentence that reads "the lead is
    # ahead" where it said "the lead is 40 m ahead" has quietly become a
    # different, more confident claim; "[unmeasured]" keeps the reader aware
    # that something was removed from this sentence.
    clean = "[unmeasured]".join(s.strip() for s in out)
    clean = re.sub(r"\s{2,}", " ", clean).strip()
    return clean, sourced, invented


# ---------------------------------------------------------------------------
# Corroboration: did the thing it named actually get measured?
# ---------------------------------------------------------------------------
# The detector's six classes, and the words a language model uses for them.
# Built from headway.detect.COCO_TO_LABEL rather than invented here, so a
# seventh class added to the detector is a KeyError in the test rather than a
# hazard silently counted as unverified forever.
_SYNONYMS = {
    "car": {"car", "sedan", "hatchback", "suv", "estate", "wagon", "coupe",
            "vehicle", "automobile", "saloon", "crossover"},
    "truck": {"truck", "lorry", "pickup", "semi", "trailer", "hgv", "van",
              "tractor-trailer", "rig"},
    "bus": {"bus", "coach", "minibus"},
    "motorcycle": {"motorcycle", "motorbike", "bike", "scooter", "moped"},
    "pedestrian": {"pedestrian", "person", "walker", "man", "woman", "child",
                   "people"},
    "cyclist": {"cyclist", "bicycle", "bike", "rider"},
}

# Things a reading may name that are not road users and that the detector
# does not track. Naming one is not a fabrication -- the model can see paint
# and signs perfectly well -- but it is also not corroborated, and the count
# has to distinguish "named a vehicle that is not there" from "mentioned the
# hard shoulder". Only the first is a fabrication.
_NON_TRACKED = {"lane", "lanes", "line", "lines", "marking", "markings",
                "sign", "signs", "barrier", "guardrail", "median", "divider",
                "curve", "bend", "shoulder", "verge", "exit", "ramp",
                "surface", "asphalt", "road", "highway", "sun", "glare",
                "shadow", "tree", "trees", "hill", "hills", "sky", "bridge"}

# A CITATION, IN EVERY SHAPE THE MODEL ACTUALLY WRITES ONE.
#
# The first version of this was `track\s*#?\s*(\d+)` and it scored zero cites
# on a reading that opened "Track IDs 9, 11, 16, 18, 21, and 22 are critical"
# -- because "ID" sits between the word and the number, and because six ids
# follow one "Track". A corroboration count that misses the citations is worse
# than no count: it reports the model ignoring its grounding at the exact
# moment the model is leaning on it hardest.
#
# So: the word, optionally "id"/"ids"/"#", then a RUN of numbers joined by
# commas, slashes, "and" or "&". Bounded to a short run so a sentence that
# happens to continue into unrelated digits does not swallow them.
# Words that claim a direction of travel, and the direction each claims. Read
# in a window of text around a citation, because that is where a claim about a
# track lives -- "Track 18: Truck slowing down on the far left".
_CLOSING_WORDS = re.compile(
    r"\b(closing|slowing|slows|decelerat\w*|braking|brakes|approach\w*|"
    r"nearing|coming closer|gaining|cutting in)\b", re.I)
_OPENING_WORDS = re.compile(
    r"\b(opening|pulling away|moving away|accelerat\w*|receding|"
    r"drawing away|speeding up)\b", re.I)
# How much text around a citation counts as being about it. One clause, give
# or take: far enough to catch "Track 18: Truck slowing down", short enough
# that the next sentence about a different vehicle is not read as this one's.
_CLAIM_SPAN = 90


def _claim_near(text, tid):
    """Does the text near this citation claim it is closing or opening? -> str|None."""
    for m in re.finditer(rf"\btracks?\s*(?:ids?\s*)?#?\s*{tid}\b", text, re.I):
        seg = text[m.start():m.start() + _CLAIM_SPAN]
        if _CLOSING_WORDS.search(seg):
            return "closing"
        if _OPENING_WORDS.search(seg):
            return "opening"
    return None


_TRACK_CITE = re.compile(
    r"\btracks?\s*(?:ids?\s*)?#?\s*(\d+(?:\s*(?:,\s*and|,|/|&|and)\s*\d+){0,11})",
    re.I)
_CITE_NUMS = re.compile(r"\d+")


# ---------------------------------------------------------------------------
# The judgement: risk, should_speak, why, about.
# ---------------------------------------------------------------------------
# PARSED OUT OF THE ANSWER AND SCORED SEPARATELY FROM IT, because they fail
# differently. A reading can be good prose with a nonsense risk level, or a
# defensible risk level attached to a hallucinated object, and a card that
# showed one number for both would hide whichever was worse.
#
# The prose is what the numbers rule and the corroboration run on; the JSON is
# what the shadow scoring runs on. Splitting them here is what keeps the
# judgement's own track ids out of the prose citation count and vice versa --
# they are merged deliberately in `corroborate`, not by accident here.
_JSON_TAIL = re.compile(r"\{[^{}]*\}", re.S)


def split_judgement(answer: str):
    """-> (prose, judgement|None, problem|None).

    `problem` is "missing" when no JSON came back at all and "malformed" when
    something JSON-shaped did but does not carry the four keys with usable
    values. The two are counted apart: a model that never emits the object is
    a prompt failure, and one that emits a broken object is a model failure,
    and the fix is different.
    """
    import json as _json
    text = (answer or "").strip()
    matches = list(_JSON_TAIL.finditer(text))
    if not matches:
        return text, None, "missing"
    # The LAST object, because the prompt asks for it last and a reading that
    # mentions a brace earlier should not shadow the real one.
    m = matches[-1]
    prose = (text[:m.start()] + " " + text[m.end():])
    # THE MODEL FENCES ITS JSON, and NVIDIA's own temporal_localization sample
    # shows it doing exactly that ("```json ... ```"). Removing the object
    # from inside the fence leaves the fence behind, and "```json ```" is not
    # prose -- it read as an eleven-character answer and tripped the
    # bare-list guard, which reported a formatting artefact as a model fault.
    prose = re.sub(r"```[a-zA-Z]*", " ", prose)
    prose = re.sub(r"\s{2,}", " ", prose).strip()
    try:
        raw = _json.loads(m.group(0))
    except Exception:
        return prose, None, "malformed"
    if not isinstance(raw, dict):
        return prose, None, "malformed"

    risk = str(raw.get("risk", "")).strip().lower()
    if risk not in RISK_LEVELS:
        return prose, None, "malformed"

    # should_speak: accept the JSON literal and the two strings a model
    # reaches for when it forgets which it is writing. Anything else is
    # malformed rather than quietly falsy -- a judgement whose speak decision
    # we had to guess at is not a judgement worth scoring.
    sp = raw.get("should_speak")
    if isinstance(sp, bool):
        speak = sp
    elif isinstance(sp, str) and sp.strip().lower() in ("true", "false", "yes", "no"):
        speak = sp.strip().lower() in ("true", "yes")
    else:
        return prose, None, "malformed"

    about = []
    for v in (raw.get("about") or []):
        try:
            about.append(int(v))
        except (TypeError, ValueError):
            # A non-numeric entry ("the white sedan") is not a track id. Kept
            # out of `about` and left in the raw record, so the card can show
            # what it actually said.
            pass
    # BOTH VOCABULARIES ON EVERY JUDGEMENT, whichever backend produced it.
    # The four-level scale is the post-training target's; the five-level one
    # is the hazard engine's, and is what rubric.grade scores against. A
    # record carrying only one of them would be ungradeable or unlabelable
    # depending on which, and the mapping lives in exactly one place.
    import rubric as _rubric
    return prose, {
        "risk": risk,
        "priority": _rubric.to_engine(risk),
        "should_speak": bool(speak),
        "why": str(raw.get("why") or "").strip(),
        "about": about,
        "about_raw": raw.get("about"),
        "occluded": bool(raw.get("occluded")),
        "source": "prompt",
    }, None


# CONTROL DECISIONS, WHICH THIS SYSTEM MAY NOT MAKE.
#
# RIO advises; it does not steer, brake or accelerate. Counted on EVERY arm
# rather than forbidden in every prompt, and that asymmetry is the experiment:
# arm B is NVIDIA's "What can be the next immediate action?" as shipped, which
# is a question about what to DO, and muzzling it in the prompt would hide the
# one thing worth knowing about running it here.
#
# A RECOMMENDATION, NOT A MENTION. "The lead may brake" is an observation
# about another vehicle and is fine; "prepare to brake" is a braking
# recommendation and is not. So a control verb only counts when a directive
# marker is near it.
_CONTROL_VERB = re.compile(
    r"\b(brake|braking|accelerat\w*|decelerat\w*|steer\w*|swerv\w*|"
    r"slow down|speed up|change lanes?|lane[- ]change|merge|overtake|pull "
    r"over|stop the (?:car|vehicle)|apply the brakes|turn the wheel|"
    r"maintain speed|reduce speed|increase speed)\b", re.I)
_DIRECTIVE = re.compile(
    r"\b(should|must|need to|needs to|ought to|recommend\w*|advis\w*|"
    r"prepare to|be ready to|be prepared to|action:|immediately|"
    r"driver (?:should|must)|it is advisable)\b", re.I)
_CONTROL_SPAN = 70


# AN IMPERATIVE IS THE MOST DIRECT CONTROL DECISION THERE IS, and the first
# version of this missed every one of them. It required a directive marker
# ("should", "must", "prepare to") near the verb, so "The driver should
# prepare to brake" was caught and "Accelerate cautiously to match the flow of
# traffic" -- an instruction with no hedge at all -- scored zero. Measured on
# the stored runs: "Reduce speed to maintain a safe buffer", "Continue waiting
# at the STOP sign", "Accelerate smoothly into the left turn", none counted.
#
# A control verb opening a sentence, a clause, a bullet or a numbered item,
# with no subject in front of it, is an imperative.
_IMPERATIVE = re.compile(
    r"(?:^|[.!?;:]\s+|\n\s*|^\s*[-*\u2022]\s*|\d+\.\s+)"
    r"(brake|accelerate|decelerate|steer|swerve|slow down|speed up|"
    r"change lanes?|merge|overtake|pull over|stop|continue|proceed|"
    r"maintain|reduce speed|increase speed|keep|avoid|remain|stay)\b",
    re.I | re.M)


def control_decisions(text: str):
    """Phrases recommending a control action. -> [str].

    Two shapes, because a control decision arrives as either:
      a HEDGED recommendation  "the driver should prepare to brake"
      a BARE imperative        "Accelerate cautiously to match the flow"
    A mention is neither: "the lead vehicle may brake suddenly" is an
    observation about somebody else's car and is left alone.
    """
    out = []
    body = text or ""
    for m in _CONTROL_VERB.finditer(body):
        lo = max(0, m.start() - _CONTROL_SPAN)
        if _DIRECTIVE.search(body[lo:m.end() + 20]):
            out.append(m.group(0).lower())
    for m in _IMPERATIVE.finditer(body):
        out.append(m.group(1).lower())
    return sorted(set(out))


def judge_faults(judgement, state, corr):
    """What is wrong with this judgement that its words do not show. -> dict.

    Two checks, both from the spec, both measured against geometry:

      risk_unverified    a risk above `none` has to be ABOUT something we
                         measured. A judgement that says "advise" and names
                         no track, or names one we are not holding, is a
                         mood rather than a finding.

      urgent_contradicted  `urgent` means something is closing now. If every
                         track the judgement names was measured OPENING, and
                         the headway filter has no closing gap either, the
                         measured range rate says the opposite of the
                         judgement. That is the same class of fault as a
                         reading calling an opening track "slowing", and it
                         is counted in the same place.
    """
    out = {"risk_unverified": False, "urgent_contradicted": False,
           "reasons": []}
    if not judgement:
        return out
    risk = judgement.get("risk")
    ids = set(judgement.get("about") or [])
    held = grounding.track_ids(state or {})
    known = ids & held

    if risk and risk != "none":
        if not known:
            out["risk_unverified"] = True
            out["reasons"].append(
                f"risk '{risk}' names {sorted(ids) if ids else 'no track'}, "
                f"and none of that is a track in this window")

    if risk == "urgent":
        tracks = {t["id"]: t for t in (state or {}).get("tracks") or []}
        closing = []
        for tid in known:
            rr = (tracks.get(tid) or {}).get("range_rate_ms")
            if rr is not None and rr < -0.5:
                closing.append(tid)
        hw = (state or {}).get("headway") or {}
        gap_closing = False
        if hw.get("gap_m") is not None and hw.get("gap_first_m") is not None:
            gap_closing = float(hw["gap_m"]) < float(hw["gap_first_m"]) - 1.0
        if not closing and not gap_closing:
            out["urgent_contradicted"] = True
            out["reasons"].append(
                "urgent, and nothing it names was measured closing "
                "(nor was the lead gap)")
    return out


def corroborate(answer: str, state: dict, judgement: dict = None):
    """Which road users the reading names, and whether we measured them.

    -> dict with `cited`, `uncited`, `fabricated`, `missed`, `verdict`.

    THE ASYMMETRY IS DELIBERATE, and it is the same one reconcile.py already
    uses: the detector wins on EXISTENCE. If the model names a bus and no
    track in the window is a bus, that is a fabrication, because a bus is a
    large object at close range and a detector that missed one has a bigger
    problem than this card. If the model does NOT name a track the headway
    loop was holding, that is a MISS and is counted separately -- it may be a
    correct judgement that the track did not matter, which is exactly what the
    model is being asked for, and calling it an error would punish the model
    for doing its job.
    """
    text = (answer or "").lower()
    ids = grounding.track_ids(state)
    labels = grounding.track_labels(state)
    tracks = {t["id"]: t for t in (state.get("tracks") or [])}

    cited, bad_cites = [], []
    for m in _TRACK_CITE.finditer(text):
        for num in _CITE_NUMS.findall(m.group(1)):
            tid = int(num)
            (cited if tid in ids else bad_cites).append(tid)
    # THE JUDGEMENT'S `about` IS A CITATION TOO. It is the most load-bearing
    # one in the record -- it is what makes a risk level checkable -- and
    # leaving it out would score a judgement that named exactly the right
    # track as having cited nothing.
    for tid in ((judgement or {}).get("about") or []):
        (cited if tid in ids else bad_cites).append(tid)

    # Which detector classes are actually present in the window, expanded to
    # the words a model would use for them.
    present_words = set()
    for lab in labels:
        present_words |= _SYNONYMS.get(lab, {lab})

    named, fabricated = [], []
    for lab, words in _SYNONYMS.items():
        for w in words:
            if re.search(rf"\b{re.escape(w)}s?\b", text):
                if w in present_words:
                    named.append(w)
                else:
                    # A word for a road user, and nothing of that kind was
                    # measured anywhere in six seconds of video.
                    fabricated.append(w)
                break

    # What the loop was holding that the reading never mentions. Only tracks
    # worth mentioning: confirmed, ranged, and either in the ego lane or the
    # lead. A detector box on a car three lanes over at 90 m is not something
    # a reading is wrong to leave out.
    missed = []
    for t in tracks.values():
        if not t.get("confirmed"):
            continue
        if not (t.get("in_lane") or t.get("is_lead") or t.get("vulnerable")):
            continue
        if t["id"] in cited:
            continue
        lab = (t.get("label") or "").lower()
        if any(w in named for w in _SYNONYMS.get(lab, {lab})):
            continue
        missed.append({"id": t["id"], "label": t.get("label"),
                       "range_m": t.get("last_range_m"),
                       "in_lane": t.get("in_lane"), "is_lead": t.get("is_lead")})

    # THE CITE IS NOT THE CHECK.
    #
    # A reading that says "Track 18: truck SLOWING DOWN, may merge into the
    # ego's lane" cites a real track and contradicts the measurement attached
    # to it -- track 18 was measured OPENING at 9.8 m/s, moving away. Scoring
    # that as corroborated because the number existed is the corroboration
    # count lying in the most flattering possible direction. So the direction
    # of travel is checked too, on the tracks the reading actually cited.
    contradicts = []
    for tid in set(cited):
        t = tracks.get(tid) or {}
        rr = t.get("range_rate_ms")
        if rr is None or abs(rr) < 0.5:
            continue          # too slow to have a direction worth contesting
        said = _claim_near(text, tid)
        if said and ((said == "closing" and rr > 0) or
                     (said == "opening" and rr < 0)):
            contradicts.append({
                "id": tid, "said": said,
                "measured": ("opening" if rr > 0 else "closing"),
                "range_rate_ms": rr})

    verdict = "corroborated"
    if fabricated or bad_cites or contradicts:
        verdict = "unverified"
    elif not cited and named:
        # It named road users and pointed at none of our tracks. Not a
        # fabrication -- the things it named are out there -- but nothing in
        # the reading is tied to a measurement, so it cannot be checked.
        verdict = "uncited"
    return {
        "cited": sorted(set(cited)),
        "bad_cites": sorted(set(bad_cites)),
        "contradicts": contradicts,
        "named": sorted(set(named)),
        "fabricated": sorted(set(fabricated)),
        "missed": missed,
        "verdict": verdict,
    }


# ---------------------------------------------------------------------------
# PROMPT BORROWING: hazard words that are in the prompt and not in the frame.
# ---------------------------------------------------------------------------
# THE RISK A RICH PROMPT ACTUALLY CARRIES. A prompt that lists what to watch
# for -- deer, a ball, a stroller, standing water, steel plates -- hands the
# model a vocabulary of plausible hazards, and 23c3dcd is the measurement
# that a 2B model will hand a prompt's own contents back as an observation.
# The richer the list, the more convincing the fabrication.
#
# THE LIST IS DERIVED FROM THE PROMPT, NEVER WRITTEN HERE. A hand-maintained
# list of forbidden words would go stale the first time a prompt changed, and
# would silently stop measuring the arm it was written for. `prompt_nouns`
# reads whichever prompt is being scored and extracts its own object
# vocabulary, so a new arm is covered the moment it exists.
#
# WHAT COUNTS AS BORROWED, precisely: a physical-object word that appears in
# the PROMPT, appears in the READING, and is corroborated by NOTHING -- not a
# detector track, not a class the detector holds. The detector knows six
# classes, so anything outside them can never be corroborated; an unfamiliar
# hazard noun in a reading is therefore always unverifiable, and one that the
# prompt supplied is the case worth counting apart.

# Words that are grammar or instruction, never an object in the road. Used to
# keep the derived vocabulary down to things that could be SEEN.
_NOT_OBJECTS = frozenset("""
a an and are as at be been being but by can cannot did do does for from had
has have how if in into is it its may might must no not of on one only or
other our out over said say should so some such than that the their them then
there these they this those through to too under until up upon was were what
when where whether which while who why will with within would you your yours
answer answers question questions describe description detail details give
given gives identify identified list lists name named names note noted report
reports respond response return returns say saying said state states tell
video videos image images frame frames scene scenes camera cameras view views
observation observations reasoning reason reasons think thinking thought
critical criticality important importance matter matters matters risk risks
hazard hazards safe safety safely navigation navigate driver drivers driving
drive vehicle vehicles ego object objects thing things next immediate action
actions development developments ready prepare prepared aware awareness
nothing none complete completely failure fail example examples format json
object true false empty short sentence sentences words own new line lines
number numbers track tracks measured measurement measurements sensor sensors
speed distance time contact estimate estimates level levels priority
priorities output outputs schema field fields key keys value values string
strings boolean list array rule rules must never always all any each every
first second third then after before above below following follows
""".split())

# Multi-word hazards are what a rich prompt is made of ("standing water",
# "steel plates", "school bus"), and a bare word list would score "water" and
# "plates" separately and match neither. Picked up as phrases from the prompt.
_PHRASE_RE = re.compile(r"\b([a-z]{3,})\s+([a-z]{3,})\b")

# A word that could be a thing in a road scene. Deliberately broad -- it is a
# filter on the PROMPT's vocabulary, not a hazard list of its own, so a false
# positive here costs a word being watched that did not need watching.
_OBJECTISH = re.compile(
    r"(?:er|or|ers|ors|le|les|ck|cks|ge|ges|ne|nes|te|tes|ll|lls|te|"
    r"ent|ents|ing|ings|al|als|an|ans|on|ons|ar|ars|ot|ots|ay|ays|"
    r"ate|ates|el|els|il|ils|ow|ows|ee|ees|og|ogs|at|ats|ap|aps)$")


def prompt_nouns(prompt_text: str) -> set:
    """Object-ish words and phrases this prompt supplies. -> set of str.

    Read off the prompt, so it cannot go stale when the prompt changes.
    """
    text = (prompt_text or "").lower()
    words = set(re.findall(r"\b[a-z][a-z-]{2,}\b", text))
    singles = {w for w in words
               if w not in _NOT_OBJECTS and _OBJECTISH.search(w)}
    phrases = set()
    for a, b in _PHRASE_RE.findall(text):
        if a in _NOT_OBJECTS and b in _NOT_OBJECTS:
            continue
        if b in singles or a in singles:
            phrases.add(f"{a} {b}")
    return singles | phrases


def borrowed_terms(answer: str, prompt_text: str, state: dict) -> list:
    """Hazard words the prompt supplied and the frame does not support.

    -> [{"term", "phrase"}], most specific first.

    A term is NOT borrowed when the detector corroborates it: if the prompt
    says "truck" and a truck is tracked, the reading naming one is the system
    working. Only the uncorroborated ones count, which is why this takes the
    measured state and not just two strings.
    """
    text = (answer or "").lower()
    supplied_words = set()
    for lab in grounding.track_labels(state or {}):
        supplied_words |= _SYNONYMS.get(lab, {lab})
    out = []
    nouns = prompt_nouns(prompt_text)
    # Longest first, so "standing water" is reported instead of "water".
    for term in sorted(nouns, key=lambda t: (-len(t), t)):
        if term in supplied_words:
            continue
        if any(term in seen["term"] for seen in out):
            continue          # already covered by a longer phrase
        if re.search(rf"\b{re.escape(term)}s?\b", text):
            out.append({"term": term, "phrase": " " in term})
    return out


# HOW MUCH OF AN EXAMPLE HAS TO COME BACK BEFORE IT IS AN ECHO.
#
# THE FIRST VERSION OF THIS MEASURED ZERO AND WAS WRONG. It required a shared
# four-word run, and a model does not copy a worked example that way -- it
# paraphrases it. Arm D's reading of a FROZEN frame, where nothing moves at
# all, reported "multiple brake lights activating (Tracks 48 & 49)": that is
# the prompt's second worked example ("Multiple brake lights. Reduced spacing
# farther ahead. Lead vehicle may begin decelerating"), re-emitted with our
# own track ids stapled on, and it shared 50% of the example's content words
# and zero four-grams. A detector that reports 0 on that is not measuring
# what it claims to.
#
# So: a shared THREE-word run, or 40% of the example's distinctive words.
# Both thresholds are arbitrary and both are stated; the fraction is what
# catches a paraphrase and the n-gram is what catches a quotation.
_ECHO_NGRAM = 3
_ECHO_OVERLAP = 0.40


def worked_example_echo(answer: str, examples) -> list:
    """Which of a prompt's worked examples came back. -> [int].

    A WORKED EXAMPLE GETS COPIED -- that is 23c3dcd, and it is the reason
    this exists rather than a note saying to be careful.
    """
    def words(t):
        return re.findall(r"[a-z]{4,}", (t or "").lower())

    def grams(w, n):
        return {tuple(w[i:i + n]) for i in range(len(w) - n + 1)}

    hits = []
    a_w = words(answer)
    a_g = grams(a_w, _ECHO_NGRAM)
    a_set = set(a_w)
    for i, ex in enumerate(examples or []):
        e_w = words(ex)
        if not e_w:
            continue
        shared = len(a_set & set(e_w)) / len(set(e_w))
        if (a_g & grams(e_w, _ECHO_NGRAM)) or shared >= _ECHO_OVERLAP:
            hits.append(i)
    return hits


# ---------------------------------------------------------------------------
# Counters. The same discipline vision._flags keeps, for the video path.
# ---------------------------------------------------------------------------
# NOT SHARED WITH vision._flags, and that is the point. Those count a still
# being read with a prompt that forbade a reasoning trace; these count a video
# being read with a prompt that asks for one. Adding them together would give
# a think_trace rate that means nothing, which is the fault the still path's
# own counter currently has -- see the note on `think_trace` in vision.py.
_lock = threading.Lock()
_flags = {
    "total": 0,
    # the trace filled the budget and there was no answer after it
    "think_unterminated": 0,
    # the answer hit the token cap mid-sentence
    "truncated": 0,
    # a measurement we did not supply, stripped
    "invented_units": 0,
    # ...and how many readings contained at least one
    "readings_with_invented": 0,
    # a number that WAS ours, restated. Not a fault -- the behaviour being
    # asked for -- counted so "grounding changed nothing" is answerable.
    "sourced_numbers": 0,
    # named a class of road user the detector held none of, anywhere in the
    # window
    "fabricated_object": 0,
    # cited a track number that does not exist
    "bad_track_cite": 0,
    # cited a real track and then said the opposite of what was measured about
    # it. The fault a citation count alone cannot see, and the one that most
    # deserves to be watched: it is the model using its grounding as decoration.
    "contradicts_measurement": 0,
    # a confirmed in-lane or lead track the reading never mentions
    "missed_track": 0,
    # the reading refused outright, by any guard
    "refused": 0,
    # the window was structurally empty and the answer described a road anyway
    "blank_window_fabrication": 0,
    # the answer was a bare list of track numbers and no reasoning at all
    "degenerate_list": 0,
    # a hazard word the PROMPT supplied, in the reading, corroborated by
    # nothing. The specific risk a richly-worded prompt carries.
    "prompt_borrowing": 0,
    # a prompt's own worked example, coming back as an observation. 23c3dcd.
    "worked_example_echo": 0,
    # the reading elevated something to CAUTION or above
    "elevated": 0,
    # ...and it was something the hazard engine's own false-positive control
    # list says must NOT be elevated, with a measurement to prove it
    "fp_violation": 0,
    # which backend produced the structured label
    "judged_guided": 0,
    "judged_prompt": 0,
    # the model returned the judgement object and NO PROSE -- it answered the
    # schema instead of the question
    "no_prose": 0,
    # --- the judgement -----------------------------------------------------
    # no JSON object came back at all. A PROMPT failure: the model was asked
    # for one and did not produce one.
    "judgement_missing": 0,
    # one came back and could not be used -- bad risk word, missing key, a
    # should_speak that is neither true nor false. A MODEL failure, counted
    # apart because the fix is a different one.
    "judgement_malformed": 0,
    # a risk above `none` that names no track we are holding
    "risk_unverified": 0,
    # `urgent`, with nothing it names measured closing and no closing lead gap
    "urgent_contradicted": 0,
    # told the driver to steer, brake or accelerate. RIO advises; it does not
    # drive. Counted on every arm, forbidden in none of them -- see the note
    # on control_decisions.
    "control_decision": 0,
    # --- the four shadow counts. Nothing is acted on; these are the numbers
    # that decide whether Cosmos is ever given the mouth. -------------------
    "agree_speak": 0,
    "false_alarm": 0,
    "miss": 0,
    "agree_quiet": 0,
    # a reading that was refused outright while the loop WAS firing. Not one
    # of the four -- a refusal is not a judgement -- but operationally the
    # same silence as a miss, so it is never left out of the report.
    "refused_while_loop_spoke": 0,
    # readings that named nothing and cited nothing: the honest "nothing here"
    "said_nothing_of_note": 0,
    # the reading told the driver to DO something rather than what to be aware
    # of. COUNTED, NOT REFUSED, and the difference matters: the still path
    # refuses an advisory outright because an instrument giving instructions
    # is the fault (rio_prompts.sensor_faults). This path is explicitly asked
    # "what should the driver be aware of", so an answer that shades into
    # advice is the question's doing, not the model's -- refusing it would be
    # refusing the reading for answering what it was asked. The count is here
    # because the line between awareness and instruction is exactly what has
    # to be watched before anybody argues this should trigger speech.
    "advisory_tone": 0,
}


# PER ARM AS WELL AS IN TOTAL. The whole point of running three questions is
# to compare them, and a single pooled counter would average the arm that
# invents things together with the arm that does not.
_arm_flags = {}


def _bump_arm(arm, key, n=1):
    with _lock:
        d = _arm_flags.setdefault(arm, {})
        d[key] = d.get(key, 0) + n


def arm_token_budget(grounding_text: str = "") -> dict:
    """What each arm's prompt costs, and what it leaves to read with. -> dict.

    Reported in TOKENS, measured with the model's own tokenizer, because a
    prompt's length is only meaningful against the budget it shares. The
    reading budget is already truncating -- 1 reading in 7 at 512 on the
    still path -- so an arm that spends three hundred more tokens on its
    question is taking them from the answer.
    """
    try:
        import vision
        vision._ensure_loaded()
        tok = vision._processor.tokenizer
    except Exception as e:
        return {"error": f"{type(e).__name__}: {e}"}
    out = {}
    budget = int(getattr(config, "EYE_WINDOW_MAX_TOKENS", 1536))
    for arm in ARMS:
        # SYSTEM AND USER TOGETHER. Arm D's text lives in the system slot, so
        # counting only the user turn reported it as the CHEAPEST arm in the
        # set at 58 tokens. What costs is what is prefilled, wherever it sits.
        user = build_prompt(grounding_text, arm=arm)
        sysp = system_for(arm)
        n_sys, n_user = len(tok.encode(sysp)), len(tok.encode(user))
        out[arm] = {
            "name": ARM_NAMES.get(arm, arm),
            "system_tokens": n_sys,
            "user_tokens": n_user,
            "text_tokens": n_sys + n_user,
            "generate_budget": budget,
        }
    return out


def arm_flags() -> dict:
    """Counters split by which question was asked. -> {arm: {...}}."""
    with _lock:
        out = {a: dict(d) for a, d in _arm_flags.items()}
    for a, d in out.items():
        total = d.get("total") or 0
        d["name"] = ARM_NAMES.get(a, a)
        d["rate"] = {k: (round(v / total, 4) if total else None)
                     for k, v in d.items() if isinstance(v, int) and k != "total"}
    return out


def flags() -> dict:
    with _lock:
        total = _flags["total"] or 0
        out = dict(_flags)
    out["model"] = config.local_vision_label()
    out["rate"] = {k: (round(v / total, 4) if total else None)
                   for k, v in out.items() if isinstance(v, int) and k != "total"}
    return out


def reset_flags() -> None:
    with _lock:
        for k in _flags:
            _flags[k] = 0
        _arm_flags.clear()


def _bump(key, n=1):
    with _lock:
        _flags[key] = _flags.get(key, 0) + n


# ---------------------------------------------------------------------------
# The reading.
# ---------------------------------------------------------------------------
# THE READING SAID IT COULD NOT SEE, WHICH IS THE CORRECT ANSWER.
#
# Without this the blank-window guard refuses the right answer. Measured: on a
# black window the reading opened "The video depicts a completely black screen
# with no visible objects" and went on to mention that lane geometry was
# "present but unobservable" -- correct, careful, and refused, because the word
# "lane" appeared in it. A guard that fires on the sentence describing its own
# blindness is not a guard, it is a word filter.
# A FIXED PHRASE LIST WAS NOT ENOUGH, TWICE.
#
# First version missed "lane geometry is present but unobservable". Second
# missed "the feed is compromised and lacking environmental data ... lane
# boundaries, obstacles, or surrounding vehicles are not measurable or
# visible" -- because the phrase there is "not measurable OR visible", and a
# list of exact phrases can only ever catch the wordings somebody thought of.
# Both were correct readings of a dead camera, refused for containing the word
# "lane".
#
# So the second half is a NEGATION NEAR A VISIBILITY WORD, which is what all
# of these actually are, rather than another handful of phrases to be caught
# out by a third wording.
_SAYS_UNUSABLE = re.compile(
    r"\b(black screen|blank|nothing visible|cannot see|can't see|unobservable|"
    r"unusable|no view|no image|obscured|entirely dark|completely dark|"
    r"devoid of|compromised feed|feed (?:is |being )?compromised|"
    r"no (?:visual|environmental|actionable) (?:data|information|context)|"
    r"lacking environmental)\b"
    r"|\b(?:no|not|cannot|can't|nothing|none)\b[^.]{0,60}?"
    r"\b(visible|discernible|observable|measurable|perceiv\w+|detectable)\b",
    re.I)

# AN ANSWER THAT IS ONLY A LIST OF TRACK NUMBERS IS NOT AN ANSWER.
#
# Measured, and it is the fault the corroboration count is least able to see.
# Three of nine grounded readings collapsed to this:
#
#     "track 28, track 29, track 30, track 31, track 33"
#     "Track 34, Track 35"
#     "track 56, track 54, track 50, ... track 56, track 54"   <- and looping
#
# Every one scored `corroborated` with the highest citation counts in the run,
# because every id was real. The citation metric was rewarding the most
# useless possible output -- a model told to cite tracks, doing only that, and
# answering neither of the two questions it was asked. The third one is also a
# decoding loop, repeating a pair it had already emitted.
#
# So: strip the citations and the punctuation, and see whether anything is
# left that could be a reason. Refused rather than counted, because unlike an
# advisory tone this is not a partial answer to the question -- it is the
# question left unanswered in a shape that passes the checks.
_LIST_MIN_WORDS = 6


def _is_bare_list(text):
    """Is this answer nothing but track numbers? -> bool.

    IT MUST ACTUALLY BE A LIST OF TRACK NUMBERS. The first version asked only
    whether the residue was short, and so refused every terse answer -- which
    on the distilled arm meant refusing the exact behaviour that arm exists to
    produce. Measured: "Nothing special to report.", "No specific warnings
    generated." and "Regular driving, monitoring surrounding vehicles" were
    all thrown away as degenerate lists. Three of that arm's six refusals on
    the 8B were its silence default working.

    A short answer is a short answer. This guard is for the decoding loop --
    "track 69, track 66, track 68, ..." for fifteen hundred tokens -- so it
    now requires that stripping the citations actually REMOVED something
    substantial before it will call the residue too thin.
    """
    raw = text or ""
    stripped = _TRACK_CITE.sub(" ", raw)
    if len(stripped) > 0.6 * len(raw):
        # Citations were not most of the answer, so whatever is left is the
        # answer rather than the residue of one.
        return False
    words = re.findall(r"[A-Za-z]{2,}", stripped)
    return len(words) < _LIST_MIN_WORDS


_ROAD_WORDS = re.compile(
    r"\b(road|highway|lane|lanes|traffic|vehicle|vehicles|car|cars|street|"
    r"motorway|freeway|intersection|junction)\b", re.I)


def _note_refusal(rec, state, arm):
    """Book a refused reading against what the loop was doing. -> None.

    A REFUSAL IS NOT A JUDGEMENT AND IS NOT SCORED AS ONE. The four shadow
    counts are about what Cosmos decided; a reading that was thrown away
    decided nothing, and folding it into `agree_quiet` would let a model that
    breaks on every hard frame post a perfect quiet-agreement rate.

    It is not nothing, either. OPERATIONALLY a refusal while the headway loop
    was firing is a frame on which Cosmos would have said nothing and a
    warning was warranted -- the same outcome as a miss, from a different
    cause. Counted under its own name so the report can state both without
    either hiding inside the other.
    """
    loop_spoke = bool((state or {}).get("loop", {}).get("spoke"))
    rec["shadow"] = {"outcome": "refused", "cosmos_speak": None,
                     "cosmos_risk": None, "loop_spoke": loop_spoke,
                     "loop_reasons": (state or {}).get("loop", {}).get("reasons") or [],
                     "loop_bands": (state or {}).get("loop", {}).get("bands") or []}
    if loop_spoke:
        _bump("refused_while_loop_spoke")
        _bump_arm(arm, "refused_while_loop_spoke")


def read_window(window, state=None, grounded=True, max_new_tokens=None,
                arm=DEFAULT_ARM, judgement=True):
    """One grounded reading of one window. -> record dict.

    The record is the card's, the log's and RIO's, and it is one shape for all
    three. Keys:

        reasoning   the <think> trace, kept (NVIDIA shows this; so do we)
        answer      what the model said after it, after the numbers rule ran
        answer_raw  before the numbers rule, so a strip is inspectable
        sourced     measurements it restated, each with what it restated
        invented    measurements it made up, stripped
        corroboration  which tracks it cited, named, fabricated, missed
        window      span, end age, frame ids, how much of it the tower saw
        state       the measured block it was given, verbatim
        refused     None, or why this reading may not be published
        judgement   risk / should_speak / why / about, parsed from the JSON
        shadow      what the judgement would have done against what the
                    deterministic loop actually did
        arm         which question was asked
    """
    import vision
    import eyewindow

    if state is None:
        state = grounding.window_state(window) if grounded else {}
    g_text = grounding.render(state) if grounded else ""
    user = build_prompt(g_text, arm=arm, judgement=False)
    _bump("total")
    _bump_arm(arm, "total")

    raw, info = vision.generate_video(
        window.frames, window.metadata(), system_for(arm), user,
        processor_kwargs=eyewindow.processor_kwargs(),
        max_new_tokens=(max_new_tokens or config.EYE_WINDOW_MAX_TOKENS),
    )
    rec = {
        "model": info.get("model") or config.local_vision_label(),
        "grounded": bool(grounded),
        "arm": arm,
        "arm_name": ARM_NAMES.get(arm, arm),
        "judgement": None,
        "judgement_problem": None,
        "judge_faults": {},
        "control_decision": [],
        "shadow": {},
        "prompt": user,
        "system": SYSTEM,
        "window": window.to_meta(),
        "state": state,
        "timing": info,
        "at": time.time(),
        "refused": None,
        "reasoning": "", "answer": "", "answer_raw": "",
        "sourced": [], "invented": [], "corroboration": {},
        "truncated": bool(info.get("truncated")),
    }
    if info.get("skipped"):
        rec["refused"] = info["skipped"]
        _bump("refused")
        return rec
    rec["window"]["grid_thw"] = info.get("grid_thw")

    reasoning, answer, unterminated = split_reasoning(raw)
    rec["reasoning"] = reasoning
    rec["answer_raw"] = answer
    if unterminated or not answer.strip():
        # The trace ate the budget. There is no answer, and the partial trace
        # is deliberation rather than perception -- shown on the card as a cut
        # reading, never served as one.
        rec["refused"] = "reasoning_unterminated" if unterminated else "no_answer"
        _bump("think_unterminated" if unterminated else "refused")
        _bump_arm(arm, "think_unterminated" if unterminated else "refused")
        if unterminated:
            _bump("refused"); _bump_arm(arm, "refused")
        _note_refusal(rec, state, arm)
        return rec
    if rec["truncated"]:
        _bump("truncated")

    # A JSON OBJECT MAY STILL ARRIVE UNASKED, and if it does it is not prose.
    # The reading pass carries no schema now, but the model has seen enough
    # of them to volunteer one, and leaving it in the text would feed braces
    # to the numbers rule and the corroboration.
    prose, volunteered, _ = split_judgement(answer)
    answer = prose if prose else answer

    supplied = grounding.supplied_numbers(state) if grounded else []
    clean, sourced, invented = check_numbers(answer, supplied)
    rec["answer"] = clean
    rec["sourced"] = sourced
    rec["invented"] = invented
    if sourced:
        _bump("sourced_numbers", len(sourced))
    if invented:
        _bump("invented_units", len(invented))
        _bump("readings_with_invented")

    if not clean.strip():
        # THE JSON WAS THE WHOLE ANSWER. Not a bare list and not a decoding
        # loop: the model skipped the question and filled in the form. Its
        # own counter, because the fix is a prompt change and the fix for a
        # bare list is not.
        rec["refused"] = "no_prose"
        _bump("no_prose"); _bump_arm(arm, "no_prose")
        _bump("refused"); _bump_arm(arm, "refused")
        _note_refusal(rec, state, arm)
        return rec
    if _is_bare_list(clean):
        rec["refused"] = "degenerate_list"
        _bump("degenerate_list"); _bump_arm(arm, "degenerate_list")
        _bump("refused"); _bump_arm(arm, "refused")
        rec["corroboration"] = corroborate(clean, state, volunteered) if state else {}
        _note_refusal(rec, state, arm)
        return rec

    import rio_prompts as _rp
    if _rp.sensor_faults(clean):
        _bump("advisory_tone")
        rec["advisory"] = _rp.sensor_faults(clean)

    # HAZARD WORDS THE PROMPT SUPPLIED AND THE FRAME DOES NOT SUPPORT.
    # Scored against the ARM'S OWN QUESTION, not the whole prompt: the
    # measured block legitimately supplies "truck" and "car", and counting a
    # reading for repeating the tracks it was given would be the borrowing
    # check firing on the grounding working.
    borrowed = borrowed_terms(clean, ARMS.get(arm, ""), state)
    rec["borrowed"] = borrowed
    if borrowed:
        _bump("prompt_borrowing", len(borrowed))
        _bump_arm(arm, "prompt_borrowing", len(borrowed))

    ex_hits = worked_example_echo(clean, ARM_EXAMPLES.get(arm) or [])
    rec["example_echo"] = ex_hits
    if ex_hits:
        _bump("worked_example_echo", len(ex_hits))
        _bump_arm(arm, "worked_example_echo", len(ex_hits))

    ctrl = control_decisions(clean)
    rec["control_decision"] = ctrl
    if ctrl:
        _bump("control_decision", len(ctrl))
        _bump_arm(arm, "control_decision", len(ctrl))

    corr = corroborate(clean, state) if state else {}
    rec["corroboration"] = corr
    if corr:
        if corr.get("fabricated"):
            _bump("fabricated_object", len(corr["fabricated"]))
        if corr.get("bad_cites"):
            _bump("bad_track_cite", len(corr["bad_cites"]))
        if corr.get("contradicts"):
            _bump("contradicts_measurement", len(corr["contradicts"]))
        if corr.get("missed"):
            _bump("missed_track", len(corr["missed"]))
        if not corr.get("named") and not corr.get("cited"):
            _bump("said_nothing_of_note")

    # --- THE LABEL PASS. Grammar-constrained if a sidecar is up. ----------
    judged, jproblem = None, None
    backend = str(getattr(config, "EYE_JUDGE_BACKEND", "guided")).lower()
    if judgement and backend == "guided":
        import eyejudge
        judged, j_info = eyejudge.judge(clean, g_text)
        rec["judgement_timing"] = j_info
        rec["judgement_backend"] = "guided" if judged else "guided:failed"
        if judged is None:
            # The sidecar is down, slow or refused. Fall through to the
            # prompt backend rather than record no judgement at all -- and
            # KEEP THE REASON. Without it the record says only "prompt", and
            # "the sidecar is not running" and "the sidecar returned an
            # error" become the same line in the log, which is how a
            # degraded label backend stays invisible for a whole drive.
            rec["judgement_fallback_reason"] = (j_info or {}).get("error") or "unknown"
            backend = "prompt"
        else:
            _bump("judged_guided"); _bump_arm(arm, "judged_guided")
    if judgement and judged is None and backend == "prompt":
        rec["judgement_backend"] = ("prompt:after_guided_failed"
                                    if rec.get("judgement_fallback_reason")
                                    else "prompt")
        j_user = build_judgement_prompt(clean, g_text)
        j_raw, j_info = vision.generate_video(
            window.frames, window.metadata(), SYSTEM, j_user,
            processor_kwargs=eyewindow.processor_kwargs(),
            max_new_tokens=int(getattr(config, "EYE_JUDGEMENT_MAX_TOKENS", 512)),
        )
        rec["judgement_timing"] = j_info
        _bump("judged_prompt"); _bump_arm(arm, "judged_prompt")
        _, j_answer, j_unterm = split_reasoning(j_raw)
        _, judged, jproblem = split_judgement(j_answer)
        if judged is None and volunteered is not None:
            # The label pass failed and the reading pass had already offered
            # one. Taking it is better than recording no judgement at all,
            # and it is marked so nobody reads it as the label pass working.
            judged, jproblem = volunteered, None
            rec["judgement_from"] = "reading_pass"
        rec["judgement_raw"] = j_answer
    rec["judgement"] = judged
    rec["judgement_problem"] = jproblem
    if jproblem == "missing":
        _bump("judgement_missing"); _bump_arm(arm, "judgement_missing")
    elif jproblem == "malformed":
        _bump("judgement_malformed"); _bump_arm(arm, "judgement_malformed")
    if judged is not None:
        # The judgement's track ids are citations too -- recompute so the
        # corroboration on the record includes them.
        corr = corroborate(clean, state, judged) if state else corr
        rec["corroboration"] = corr

    # --- THE ENGINE'S OWN RUBRIC, applied to every arm. -------------------
    # The false-positive control list and the priority levels come out of the
    # hazard-engine prompt, and the brief is explicit that they grade the
    # WHOLE comparison rather than only the arm they were written for. An arm
    # that never sees those rules is still judged by them.
    import rubric as _rubric
    rec["rubric"] = _rubric.grade(clean, judged, state)
    if rec["rubric"].get("n_violations"):
        _bump("fp_violation", rec["rubric"]["n_violations"])
        _bump_arm(arm, "fp_violation", rec["rubric"]["n_violations"])
    if rec["rubric"].get("elevated"):
        _bump("elevated"); _bump_arm(arm, "elevated")

    # --- the judgement's own faults ----------------------------------------
    jf = judge_faults(judged, state, corr)
    rec["judge_faults"] = jf
    if jf.get("risk_unverified"):
        _bump("risk_unverified"); _bump_arm(arm, "risk_unverified")
    if jf.get("urgent_contradicted"):
        _bump("urgent_contradicted"); _bump_arm(arm, "urgent_contradicted")
        _bump("contradicts_measurement")

    # --- SHADOW SCORING. The four counts, and nothing acted on. ------------
    if judged is not None and state:
        loop_spoke = bool((state.get("loop") or {}).get("spoke"))
        said_speak = bool(judged.get("should_speak"))
        if said_speak and loop_spoke:
            outcome = "agree_speak"
        elif said_speak and not loop_spoke:
            outcome = "false_alarm"
        elif not said_speak and loop_spoke:
            outcome = "miss"
        else:
            outcome = "agree_quiet"
        rec["shadow"] = {
            "outcome": outcome,
            "cosmos_speak": said_speak,
            "cosmos_risk": judged.get("risk"),
            "loop_spoke": loop_spoke,
            "loop_reasons": (state.get("loop") or {}).get("reasons") or [],
            "loop_bands": (state.get("loop") or {}).get("bands") or [],
        }
        _bump(outcome); _bump_arm(arm, outcome)

    # THE BLANK WINDOW. Not refused before the pass -- a static or black
    # stretch of road is a thing the reading has to be able to SAY, and a
    # guard that refuses it first makes that untestable. Judged after, on the
    # words: a window with no structure in it that produced a description of a
    # road is a fabrication, and it is the one this repo has caught most often.
    if (state.get("window", {}).get("blank") and _ROAD_WORDS.search(clean)
            and not _SAYS_UNUSABLE.search(clean)):
        rec["refused"] = "blank_window_fabrication"
        _bump("blank_window_fabrication")
        _bump("refused")
    return rec


# ---------------------------------------------------------------------------
# THE RUNTIME QUESTION, MEASURED RATHER THAN ARGUED.
# ---------------------------------------------------------------------------
# NVIDIA recommends vLLM for Cosmos-Reason2 and it is faster. On this pod,
# same six-second window, same prompt, same sampling, five runs each:
#
#     transformers (this path)   p50 4478 ms   3180-5298 ms    5.4 GiB
#     vLLM 0.29                  p50 2100 ms   1207-6362 ms   ~8 GiB used,
#                                                             24 GiB reserved
#
# Roughly twice as fast, and the reasoning split comes from the server instead
# of from a regex. Three things argue against moving the live path onto it now:
#
#   1  IT IS A SECOND PROCESS WITH ITS OWN CARD RESERVATION. vLLM preallocates
#      its KV cache from --gpu-memory-utilization and holds it; at the 0.25
#      it needed to start here that is 24 GiB standing, against 5.4 GiB for
#      the resident model. The detector, depth and the lane net share this
#      card with the eye on every frame of a drive, and a reservation is not
#      a peak -- it does not give any of it back.
#
#   2  IT DOES NOT TAKE FRAMES, IT TAKES A FILE. The window lives in RAM as
#      the exact JPEGs the detector measured, with raw_sha proving it. Feeding
#      vLLM means writing an mp4 and handing over a path, which re-encodes the
#      pixels the whole provenance chain is about (framebuf.verify_raw) and
#      puts frames of the road on disk, against the retention posture in
#      framebuf's header. Measured cost of that round trip, unprompted: the
#      same window arrived as 7026 prompt tokens through vLLM against 2632
#      here, because the file path bypasses our total_pixels budget.
#
#   3  IT IS A DIFFERENT TORCH. vllm 0.29 pulls torch 2.13/cu130; this venv is
#      torch 2.11/cu128 with rfdetr, Depth Anything and the lane net built
#      against it. vLLM has to live in its own environment, which makes the
#      eye a service with a socket between it and the frames.
#
# So: worth it when the eye is the bottleneck, and it is not. The reading is
# taken every eight seconds and consumed by a card. Revisit if the eye ever
# gains a consumer that has to be current -- at which point 2.1 s against
# 4.5 s is the difference between a reading about now and a reading about
# then, and points 1-3 become costs worth paying rather than reasons not to.
