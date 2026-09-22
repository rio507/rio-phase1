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


def build_prompt(grounding_text: str = "") -> str:
    """The user turn, in NVIDIA's order. -> str."""
    parts = [AV_COT, ASKS]
    if grounding_text:
        parts.append(NUMBERS_RULE)
        parts.append(grounding_text)
    parts.append(COSMOS_FORMAT)
    return "\n\n".join(parts)


def instructions_only(grounding_text: str = "") -> str:
    """The prompt MINUS the measured block, for the echo guard.

    THIS DISTINCTION IS NOT PEDANTRY. The echo guard refuses a reading that
    hands the prompt's own words back as an answer, and it was written when
    every word of the prompt was an instruction. Half the prompt is now
    MEASUREMENTS, and repeating those is not an echo -- it is the thing the
    model was explicitly told it could do. Comparing against the whole prompt
    would refuse every correctly grounded reading, which is the guard firing
    on exactly the behaviour it was supposed to encourage.
    """
    return "\n\n".join([AV_COT, ASKS, NUMBERS_RULE, COSMOS_FORMAT])


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


def corroborate(answer: str, state: dict):
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
    """Is this answer nothing but track numbers? -> bool."""
    stripped = _TRACK_CITE.sub(" ", text or "")
    words = re.findall(r"[A-Za-z]{2,}", stripped)
    return len(words) < _LIST_MIN_WORDS


_ROAD_WORDS = re.compile(
    r"\b(road|highway|lane|lanes|traffic|vehicle|vehicles|car|cars|street|"
    r"motorway|freeway|intersection|junction)\b", re.I)


def read_window(window, state=None, grounded=True, max_new_tokens=None):
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
    """
    import vision
    import eyewindow

    if state is None:
        state = grounding.window_state(window) if grounded else {}
    g_text = grounding.render(state) if grounded else ""
    user = build_prompt(g_text)
    _bump("total")

    raw, info = vision.generate_video(
        window.frames, window.metadata(), SYSTEM, user,
        processor_kwargs=eyewindow.processor_kwargs(),
        max_new_tokens=(max_new_tokens or config.EYE_WINDOW_MAX_TOKENS),
    )
    rec = {
        "model": info.get("model") or config.local_vision_label(),
        "grounded": bool(grounded),
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
        if unterminated:
            _bump("refused")
        return rec
    if rec["truncated"]:
        _bump("truncated")

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

    if _is_bare_list(clean):
        rec["refused"] = "degenerate_list"
        _bump("degenerate_list")
        _bump("refused")
        rec["corroboration"] = corroborate(clean, state) if state else {}
        return rec

    import rio_prompts as _rp
    if _rp.sensor_faults(clean):
        _bump("advisory_tone")
        rec["advisory"] = _rp.sensor_faults(clean)

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
