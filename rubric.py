"""rubric.py — grade a reading against the hazard engine's own rules.

WHY THE RUBRIC IS PARSED AND NOT TYPED OUT
------------------------------------------
The RIO Road Hazard Reasoning Engine prompt carries two lists that are worth
more as grading criteria than as instructions to a 2B model: a FALSE POSITIVE
CONTROL list of ten things that must not be elevated, and five PRIORITY
LEVELS. Both are read out of prompts/rio_hazard_engine.txt at import.

A hand-copied copy of either would be correct on the day it was written and
silently wrong the first time somebody edited the prompt -- and "silently
wrong" for a grading rubric means every arm scoring well against rules nobody
is applying any more. The same reasoning as eyeread.prompt_nouns, for the
same reason.

WHAT CAN BE GRADED MECHANICALLY, AND WHAT CANNOT
------------------------------------------------
Six of the ten false-positive rules are checkable against geometry, because
they name a measurable relationship: a vehicle in an ADJACENT lane, a
pedestrian NOT in the path, a DISTANT object, a parked vehicle showing NO
ACTIVITY. The detector gives lane membership, range and range rate, so
"elevated a car that is measured to be beside us and not closing" is a fact.

Four are not: normal curves, normal traffic braking, irrelevant signs and
harmless roadside objects. The detector holds six classes and none of them is
a sign, a curve or a traffic-light state, so there is no measurement to
contradict a reading that elevates one. Those are counted under one honest
heading -- `elevated_uncorroborated` -- which says what it is: the reading
made something important that nothing in this system can see. That is not the
same as proving it wrong, and this module does not pretend otherwise.

THE SCALE CONFLICT, STATED RATHER THAN QUIETLY RESOLVED
--------------------------------------------------------
Two scales are in play and they are not the same:

    the judgement schema    none | watch | advise | urgent
    the hazard engine       NORMAL | WATCH | CAUTION | HIGH | CRITICAL-CANDIDATE

The first was specified as matching the post-training target; the second is
the grading rubric. They differ in count and in the boundary that matters
most -- the engine splits "act now" into HIGH and CRITICAL-CANDIDATE, and the
four-level scale does not. `to_engine` and `to_target` below map between them
in one place so both can be reported, and nothing in this file picks a winner.
"""
import re
from pathlib import Path

PROMPT_PATH = Path(__file__).resolve().parent / "prompts" / "rio_hazard_engine.txt"

try:
    _PROMPT = PROMPT_PATH.read_text()
except OSError:
    _PROMPT = ""


def _section(name: str) -> str:
    """The body of one ==== delimited section of the engine prompt. -> str."""
    m = re.search(
        r"=+\s*\n" + re.escape(name) + r"\s*\n=+\s*\n(.*?)(?=\n=+\s*\n[A-Z])",
        _PROMPT, re.S)
    return m.group(1).strip() if m else ""


def _bullets(body: str):
    return [ln.strip("- ").strip() for ln in body.split("\n")
            if ln.strip().startswith("-")]


# --- the false-positive control list, read off the prompt -------------------
FALSE_POSITIVE_RULES = _bullets(_section("FALSE POSITIVE CONTROL"))

# --- the worked examples, read off the prompt -------------------------------
# PARSED, NOT TRANSCRIBED, for the reason everything else in this file is:
# a copy goes stale the day somebody edits the prompt, and a stale copy here
# means the echo detector is watching for examples that are no longer in the
# prompt while the ones that are go unwatched. 23c3dcd is what that costs.
WORKED_EXAMPLES = [t for t in (_section("EXAMPLE"), _section("SECOND EXAMPLE"))
                   if t]


# --- the priority levels, read off the prompt -------------------------------
def _levels():
    body = _section("PRIORITY LEVELS")
    out = []
    for ln in body.split("\n"):
        t = ln.strip()
        if t and t == t.upper() and re.fullmatch(r"[A-Z-]+", t):
            out.append(t)
    return out


PRIORITY_LEVELS = _levels() or ["NORMAL", "WATCH", "CAUTION", "HIGH",
                                "CRITICAL-CANDIDATE"]
# Where the line between silence and a driver alert sits, per the prompt's own
# wording: WATCH "should normally produce no driver alert", CAUTION "may
# require driver awareness". So CAUTION is the first level that is not silent.
ELEVATED_FROM = "CAUTION"


def is_elevated(priority: str) -> bool:
    try:
        return (PRIORITY_LEVELS.index((priority or "").upper())
                >= PRIORITY_LEVELS.index(ELEVATED_FROM))
    except ValueError:
        return False


# The two vocabularies, mapped in one place. See the header.
_TARGET_TO_ENGINE = {"none": "NORMAL", "watch": "WATCH",
                     "advise": "CAUTION", "urgent": "HIGH"}
_ENGINE_TO_TARGET = {"NORMAL": "none", "WATCH": "watch", "CAUTION": "advise",
                     "HIGH": "urgent", "CRITICAL-CANDIDATE": "urgent"}


def to_engine(risk: str) -> str:
    return _TARGET_TO_ENGINE.get((risk or "").lower(), "NORMAL")


def to_target(priority: str) -> str:
    return _ENGINE_TO_TARGET.get((priority or "").upper(), "none")


# --- scenery the detector cannot corroborate --------------------------------
# Deliberately the categories the false-positive list names and the detector
# does not track. A reading may mention any of these freely; what is counted
# is ELEVATING one, at CAUTION or above, with no track behind it.
_SCENERY = {
    "sign": r"\b(sign|signs|signage|signpost)\b",
    "curve": r"\b(curve|curves|bend|bends|curvature)\b",
    "marking": r"\b(lane marking\w*|road marking\w*|dashed line\w*|"
               r"solid line\w*|yellow line\w*|white line\w*)\b",
    "barrier": r"\b(barrier|barriers|guardrail|guard rail|median|divider)\b",
    "roadside object": r"\b(tree|trees|hedge|hedges|pole|poles|bush|bushes|"
                       r"fence|fencing)\b",
}

# How far away an object stops being a plausible interaction. The engine says
# "distant objects with no plausible interaction" and does not give a number;
# this one is ours and is stated rather than hidden. 60 m at 25 m/s is about
# 2.4 seconds of closing even head-on, and the headway loop's own bands are
# all inside it.
DISTANT_M = 60.0
# Below this the track is not moving relative to us in any way worth calling
# activity. Matches the floor eyeread uses for a direction claim.
STILL_MS = 0.5


def grade(prose: str, judgement: dict, state: dict) -> dict:
    """Score one reading against the engine's own false-positive rules.

    -> {"elevated", "violations": [...], "n_violations", "unchecked"}

    A violation is always ANCHORED TO A MEASUREMENT: it names the track, what
    the reading did with it, and what the geometry says instead. A count with
    no anchor would be an opinion with a number next to it.
    """
    out = {"elevated": False, "priority": None, "violations": [],
           "n_violations": 0, "unchecked": []}
    if not judgement:
        return out

    priority = (judgement.get("priority")
                or to_engine(judgement.get("risk") or ""))
    out["priority"] = priority
    out["elevated"] = is_elevated(priority) or bool(
        judgement.get("driver_alert_recommended"))
    if not out["elevated"]:
        # Nothing was elevated, so no false positive is possible. The engine's
        # default is silence and a silent reading cannot break the silence
        # rules -- it can only miss something, which the shadow counts cover.
        return out

    tracks = {t["id"]: t for t in (state or {}).get("tracks") or []}
    ids = [int(i) for i in (judgement.get("about") or [])]
    v = out["violations"]

    for tid in ids:
        t = tracks.get(tid)
        if t is None:
            continue                     # a bad cite; eyeread counts that
        lab = (t.get("label") or "object").lower()
        rr = t.get("range_rate_ms")
        rng = t.get("last_range_m")
        in_lane = bool(t.get("in_lane") or t.get("is_lead"))
        closing = rr is not None and rr < -STILL_MS

        if not in_lane and not closing and lab in ("car", "truck", "bus"):
            v.append({"rule": "cars safely traveling in adjacent lanes",
                      "track": tid,
                      "measured": f"{lab} {t.get('side') or 'off the ego lane'}"
                                  + (f", opening at {rr:.1f} m/s" if rr and rr > 0
                                     else ", not closing")})
        if not in_lane and lab == "motorcycle" and not closing:
            v.append({"rule": "motorcycles traveling normally", "track": tid,
                      "measured": f"motorcycle {t.get('side') or 'off the ego lane'}, not closing"})
        if not in_lane and lab in ("pedestrian",):
            v.append({"rule": "pedestrians safely on sidewalks", "track": tid,
                      "measured": f"pedestrian {t.get('side') or 'off the ego lane'}"})
        if not in_lane and lab in ("cyclist",):
            v.append({"rule": "cyclists safely separated from traffic",
                      "track": tid,
                      "measured": f"cyclist {t.get('side') or 'off the ego lane'}"})
        if rng is not None and rng > DISTANT_M and not closing:
            v.append({"rule": "distant objects with no plausible interaction",
                      "track": tid,
                      "measured": f"{lab} at {rng:.0f} m, not closing"})
        if rr is not None and abs(rr) < STILL_MS and not in_lane:
            v.append({"rule": "parked vehicles showing no activity",
                      "track": tid,
                      "measured": f"{lab} at {rng if rng is None else round(rng)} m, "
                                  f"range rate {rr:.1f} m/s"})

    # Elevated something the detector cannot see at all.
    if not ids:
        text = (prose or "").lower()
        for kind, pat in _SCENERY.items():
            if re.search(pat, text):
                v.append({"rule": f"elevated a {kind} the detector does not track",
                          "track": None,
                          "measured": "no track cited and nothing of this kind "
                                      "is measurable by this system"})
                break

    out["n_violations"] = len(v)
    # The four rules there is no measurement for. Named every time so the
    # score is never read as covering them.
    out["unchecked"] = ["normal curves", "normal traffic braking",
                        "signs that do not affect the current drive",
                        "harmless roadside objects"]
    return out


# ---------------------------------------------------------------------------
# The output schema, for guided decoding.
# ---------------------------------------------------------------------------
# A CUT-DOWN OF THE ENGINE'S OWN OUTPUT FORMAT, and the cut is the point.
#
# The engine prompt's schema has eighteen fields per hazard across four
# nesting levels -- scene, hazards[], occlusion{}, latent_risks[],
# important_visual_context[] -- every one of them an empty string in the
# prompt. Asked in prose that is the largest fill-in-the-blank this project
# has built, and the last two smaller ones came back populated with invented
# values (23c3dcd, and the "risk"/"should_speak" object that replaced the
# reading entirely). Under GUIDED DECODING the grammar guarantees the shape,
# so the failure mode is no longer malformed JSON -- it is a well-formed
# object full of confabulation, which is worse because it looks parsed.
#
# So the schema keeps the fields that are CHECKABLE against something:
#   priority        graded by rubric.grade against the false-positive list
#   about           checked against the detector's track ids
#   should_speak    scored against whether the headway loop actually fired
#   occluded        the one thing the engine adds that the geometry cannot,
#                   and the reason it is worth having a VLM at all
#   why             the one free-text field, one line, for the card
# and drops the fields that nothing can contradict. A confidence float, a
# time_horizon_seconds and a predicted_conflict string on a 2B model are
# three more numbers nobody can check.
JUDGEMENT_SCHEMA = {
    "type": "object",
    "properties": {
        "priority": {"type": "string", "enum": list(PRIORITY_LEVELS)},
        "should_speak": {"type": "boolean"},
        "why": {"type": "string", "maxLength": 200},
        "about": {"type": "array", "items": {"type": "integer"},
                  "maxItems": 8},
        "occluded": {"type": "boolean"},
    },
    "required": ["priority", "should_speak", "why", "about", "occluded"],
    "additionalProperties": False,
}
