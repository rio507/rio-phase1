"""Request router — is this a question about what we can see, and about what?

Design ref: docs/visual_qa.md §6.

The router is the gate on an expensive path. Getting it wrong is asymmetric:

  A visual question routed to the plain conversation path is answered from a
  stale one-line caption, which is exactly the robotic behaviour this whole
  pipeline exists to remove.

  A non-visual question routed to the visual path costs a frame, a crop and a
  multimodal call, and RIO answers "hey" while staring at a photograph of the
  road.

So it runs rules first and asks a model only when the rules are genuinely
unsure. The rules cover the phrasings a driver actually uses, they are cheap
enough to be free, and every classification records WHICH path produced it, so
a bad route can be read out of the log instead of being reproduced by hand.

PHASE
-----
Phase A implements scene_description, specific_object_question,
visual_follow_up and non_visual_question. The other four types in the spec are
still CLASSIFIED here -- so the log shows the real distribution of what drivers
ask before the behaviour exists -- and are marked `phase_b`, which the
orchestrator handles by falling back to the nearest Phase A behaviour rather
than pretending to a capability that is not built yet.
"""
import json
import re
import time

import config

# --- the type vocabulary ----------------------------------------------------
SCENE = "scene_description"
OBJECT = "specific_object_question"
FOLLOW_UP = "visual_follow_up"
COMPARISON = "object_comparison"
LANDMARK = "location_or_landmark_question"
READ_TEXT = "read_visible_text"
CLARIFY = "clarification_required"
NON_VISUAL = "non_visual_question"

PHASE_A_TYPES = {SCENE, OBJECT, FOLLOW_UP, NON_VISUAL}
PHASE_B_TYPES = {COMPARISON, LANDMARK, READ_TEXT, CLARIFY}

# Below this the rules do not trust themselves and the model is asked.
RULE_CONFIDENCE_MIN = 0.6
# Ask a model when the rules are unsure. Off makes the router purely local and
# deterministic, at the cost of missing phrasings nobody anticipated.
USE_MODEL_FALLBACK = True

# --- vocabulary -------------------------------------------------------------
_OBJECT_NOUNS = (
    r"car|cars|vehicle|truck|lorry|pickup|suv|van|bus|coupe|sedan|saloon|wagon|"
    r"hatchback|convertible|jeep|bike|motorbike|motorcycle|cyclist|bicycle|"
    r"rider|pedestrian|person|guy|woman|man|thing|one"
)
_SIDE_WORDS = (
    r"left|right|ahead|in front|front|behind|beside us|next to us|over there|"
    r"up there|back there|oncoming|lane"
)
# "the silver one" is a demonstrative as much as "that" is -- an adjective plus
# "one" is how drivers point at a car they can see and cannot name, and it is
# the spec's own example phrasing.
_DEMONSTRATIVE = r"\b(that|those|this|these|it|its|the ones?|the \w+ ones?)\b"

_SCENE_PATTERNS = [
    r"\bwhat (do|can) you see\b",
    r"\bwhat'?s (out there|around|going on|happening|it like)\b",
    r"\bwhat are we (looking at|driving (through|past|into))\b",
    r"\bdescribe (the|this|what)\b",
    r"\bwhat does it look like\b",
    r"\bhow'?s (the road|traffic|it look)",
    r"\bwhat'?s (the road|traffic) like\b",
    r"\bwhat'?s (up )?ahead\b",
    r"\btell me what you see\b",
    r"\bwhat'?s in front of us\b",
]

_OBJECT_PATTERNS = [
    r"\bwhat (kind|type|sort|make|model) of\b",
    r"\bwhat (car|truck|vehicle|bike|suv|van|bus)\b",
    r"\bwhat'?s that\b",
    r"\bwhat is that\b",
    r"\bwho'?s (that|the)\b",
    r"\bis that an?\b",
    r"\bwhat'?s the (\w+\s+)?(%s)\b" % _OBJECT_NOUNS,
    r"\b(?:the|that)\s+\w+\s+ones?\b",
]

_FOLLOW_UP_PATTERNS = [
    r"\bwhat year\b",
    r"\bwhat (model|trim|engine|generation)\b",
    r"\bhow (old|fast|much|many|rare)\b",
    r"\bis (it|that one|that thing) (a|an|the|any|fast|rare|old|new|worth)\b",
    r"\bare (they|those) (rare|fast|good|any good)\b",
    r"\bwhat'?s it worth\b",
    r"\bdoes it have\b",
    r"\btell me more\b",
    r"\bwhat about (it|that one)\b",
]

_COMPARISON_PATTERNS = [
    r"\bwhich (one|of (them|those))\b", r"\bcompare\b",
    r"\b(bigger|faster|newer|older|nicer|closer) than\b",
    r"\bor the\b.*\?", r"\bboth of (them|those)\b",
]

_READ_TEXT_PATTERNS = [
    r"\bwhat does (that|the|it) .{0,20}(say|read)\b",
    r"\bread (the|that|it)\b",
    r"\bwhat'?s (written|printed) on\b",
    r"\bwhat (sign|plate|number plate|licence plate|license plate)\b",
]

_LANDMARK_PATTERNS = [
    r"\bwhere are we\b",
    r"\bwhat (building|street|road|place|town|shop|store|restaurant)\b",
    r"\bwhat'?s that (building|place|shop|sign for)\b",
    r"\bwhat exit\b",
]

# Questions that mention nothing visual at all and must not touch the camera.
_NON_VISUAL_PATTERNS = [
    r"^\s*(hey|hi|hello|yo|morning|evening)\b",
    r"\bhow (are you|'?s it going)\b",
    r"\bplay (some |the )?(music|something)\b",
    r"\bwhat time\b", r"\bwhat'?s the weather\b",
    r"\bhow (long|far) (until|to|till)\b",
    r"\bnavigate to\b", r"\btake me to\b", r"\bcall\b", r"\btext\b",
    r"\bturn (up|down) the\b",
]


def _any(patterns, text):
    return any(re.search(p, text) for p in patterns)


def _count(patterns, text):
    return sum(1 for p in patterns if re.search(p, text))


def extract_reference(text: str):
    """The noun phrase the driver used for the object, verbatim where possible.

    Kept as the driver's own words rather than normalised: the resolver needs
    "the silver one on the left" intact, because every part of it -- colour,
    side, the fact that it is a single object -- is evidence.
    """
    # Most specific first. The point is to hand the resolver as much of the
    # driver's own wording as possible -- a bare "that" is a valid reference and
    # a poor one, so it is the last thing tried, not the first thing matched.
    patterns = [
        # "that car on the left", "the silver one in front"
        r"\b((?:that|this|the)\s+(?:\w+\s+){0,3}?(?:%s)\b(?:\s+(?:on|to|in|at|over)\s+(?:the\s+)?(?:%s))?)"
        % (_OBJECT_NOUNS, _SIDE_WORDS),
        # "the silver one on the left", "the one on the left", "the ones ahead"
        r"\b((?:the|that)\s+(?:\w+\s+)?ones?(?:\s+(?:on|in|to|at|over)\s+(?:the\s+)?(?:%s))?)\b"
        % _SIDE_WORDS,
        # "that on the left" — the noun sits earlier in the sentence
        # ("what kind of car is that on the left"), so it is not in the phrase.
        r"\b((?:that|those|this|these)\s+(?:on|to|in|at|over)\s+(?:the\s+)?(?:%s))\b"
        % _SIDE_WORDS,
        r"\b((?:that|those|the)\s+\w+\s+one)\b",
        r"\b(that\s+one)\b",
        r"\b(that|those|this|these)\b",
    ]
    for p in patterns:
        m = re.search(p, text)
        if m:
            return m.group(1).strip()
    return None


def classify(question: str, has_referent: bool = False,
             use_model: bool = None) -> dict:
    """Driver utterance -> a routing decision.

    `has_referent` is whether an active visual referent is alive; it is what
    separates "what year is it" (a follow-up about the car we were just
    discussing) from the same words with nothing to attach them to.
    """
    t0 = time.perf_counter()
    text = (question or "").strip().lower()
    if not text:
        return _result(NON_VISUAL, None, 1.0, "empty", t0)

    scene_hits = _count(_SCENE_PATTERNS, text)
    object_hits = _count(_OBJECT_PATTERNS, text)
    follow_hits = _count(_FOLLOW_UP_PATTERNS, text)
    has_noun = bool(re.search(r"\b(%s)\b" % _OBJECT_NOUNS, text))
    has_demo = bool(re.search(_DEMONSTRATIVE, text))
    has_side = bool(re.search(r"\b(%s)\b" % _SIDE_WORDS, text))

    # Phase B types first: they are more specific than the Phase A ones and
    # would otherwise be swallowed by them ("what does that sign say" contains
    # "what does that").
    if _any(_READ_TEXT_PATTERNS, text):
        return _result(READ_TEXT, extract_reference(text), 0.85, "rules", t0)
    if _any(_COMPARISON_PATTERNS, text) and (has_noun or has_demo):
        return _result(COMPARISON, extract_reference(text), 0.8, "rules", t0)
    if _any(_LANDMARK_PATTERNS, text):
        return _result(LANDMARK, extract_reference(text), 0.8, "rules", t0)

    # A follow-up is a question with a reference but no new object named, asked
    # while a referent is alive. Without the referent the same words are a
    # question about nothing, and are better served by the ordinary path.
    if follow_hits and has_referent and not has_side:
        return _result(FOLLOW_UP, None, 0.85, "rules", t0)

    if object_hits and (has_noun or has_demo):
        return _result(OBJECT, extract_reference(text), 0.85, "rules", t0)
    if scene_hits:
        return _result(SCENE, None, 0.85, "rules", t0)

    # Weaker evidence: naming a thing AND where it is, as a question, is a
    # question about something out of the window even in a phrasing not listed
    # above. Both halves are required -- "what's the plan?" names no object and
    # "is it far?" names no place, and neither should reach for the camera.
    if (has_demo or has_noun) and has_side and "?" in question:
        return _result(OBJECT, extract_reference(text), 0.65, "rules", t0)
    if _any(_NON_VISUAL_PATTERNS, text):
        return _result(NON_VISUAL, None, 0.8, "rules", t0)
    if follow_hits and has_referent:
        return _result(FOLLOW_UP, None, 0.65, "rules", t0)

    use_model = USE_MODEL_FALLBACK if use_model is None else use_model
    if use_model:
        got = _classify_with_model(question, has_referent)
        if got is not None:
            got["latency_ms"] = round((time.perf_counter() - t0) * 1000, 1)
            return got
    return _result(NON_VISUAL, None, 0.4, "rules_default", t0)


def _result(request_type, reference, confidence, method, t0) -> dict:
    return {
        "request_type": request_type,
        "object_reference": reference,
        "requires_full_frame": request_type in (SCENE, OBJECT, FOLLOW_UP,
                                                COMPARISON, LANDMARK, READ_TEXT),
        "requires_object_crop": request_type in (OBJECT, FOLLOW_UP, READ_TEXT,
                                                 COMPARISON),
        "confidence": confidence,
        "method": method,
        "phase_b": request_type in PHASE_B_TYPES,
        "latency_ms": round((time.perf_counter() - t0) * 1000, 1),
    }


_CLASSIFIER_PROMPT = """Classify what the driver just said to their in-car assistant.

Types:
- scene_description: asking about the surroundings in general
- specific_object_question: asking about one particular thing they can see
- visual_follow_up: a further question about the object already being discussed
- object_comparison: comparing two visible things
- location_or_landmark_question: asking where they are or about a place/building
- read_visible_text: asking what a sign, plate or label says
- non_visual_question: anything not about what is currently visible

An active visual referent {referent_state}.

Reply with ONLY this JSON:
{{"request_type": "<type>", "object_reference": "<the words naming the object, or null>"}}"""


def _classify_with_model(question: str, has_referent: bool):
    """One small call, only when the rules were not sure. None on any failure."""
    try:
        from openai import OpenAI

        client = OpenAI()
        prompt = _CLASSIFIER_PROMPT.format(
            referent_state=("exists (they may be continuing about it)"
                            if has_referent else "does not exist"))
        r = client.chat.completions.create(
            model=config.OPENAI_CHAT_MODEL,
            messages=[{"role": "system", "content": prompt},
                      {"role": "user", "content": question}],
            max_completion_tokens=80,
            reasoning_effort="none",
        )
        raw = (r.choices[0].message.content or "").strip()
        m = re.search(r"\{.*\}", raw, re.S)
        if not m:
            return None
        obj = json.loads(m.group(0))
        rt = str(obj.get("request_type") or "").strip()
        if rt not in PHASE_A_TYPES | PHASE_B_TYPES:
            return None
        ref = obj.get("object_reference")
        ref = str(ref).strip() if ref and str(ref).lower() != "null" else None
        return _result(rt, ref, 0.7, "model", time.perf_counter())
    except Exception as e:
        print(f"[router] model fallback failed: {type(e).__name__}: {e}", flush=True)
        return None


def is_visual(request_type: str) -> bool:
    return request_type != NON_VISUAL
