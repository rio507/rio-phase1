"""What RIO says about a maneuver. Deterministic templates, no model (§23).

THE THREE OPPORTUNITIES
-----------------------
Not three mandatory calls — three chances to be useful, each of which RIO takes
only if it is still true and still needed:

  EARLY      "Right turn coming up."            optional, no camera, no distance
  PRIMARY    "Turn right by the Shell station." the differentiated one
             "Take the next right."             ...and its canonical fallback
  IMMINENT   "Right here."                      only when it is still needed

The primary call REPLACES distance narration. RIO does not say "Turn right in
200 feet. Turn right by the Shell." — it says "Turn right by the Shell." A
driver who can see the Shell does not need the number, and a driver who cannot
gets "Take the next right", which is the same sentence a passenger would use.

WHY THERE IS NO LLM HERE, AND WHY THERE NEVER WILL BE ON THIS PATH
------------------------------------------------------------------
Every sentence RIO can say about a maneuver is enumerable before the drive
starts: a direction, a road name, and at most one landmark from a list fetched
at route load. Enumerating them costs a dictionary lookup and removes latency,
hallucination, a validation layer, a test surface and an entire class of
failure state.

VARIED, WITHOUT A MODEL AND WITHOUT A DICE ROLL
-----------------------------------------------
A car that says "Left turn coming up." nine times in twenty minutes stops
sounding like someone in the passenger seat and starts sounding like a GPS,
which is the one thing RIO is defined against. So the EARLY line and the
CONTEXTUAL line — the two that are never time-critical — are drawn from a SET
of phrasings in her register rather than from a single template.

The set is fixed, and so is the choice. `_variant` indexes it by the
maneuver's position in the route plus a per-generation offset, which buys the
three things a random draw does not:

  a drive never repeats itself     consecutive maneuvers step through the set
                                   rather than landing on the same line twice
  a sentence is accountable        the offset is arithmetic on the journey id
                                   and the generation, both of which are in
                                   the drive log — so what she said on any
                                   turn of any drive can be recomputed, which
                                   a dice roll makes impossible. Two separate
                                   drives to the same place are two journeys
                                   and may phrase a turn differently, which is
                                   the point for anyone who drives it daily.
  /nav/voice stays a lookup        the text is chosen once, at route build,
                                   and stored — nothing decides anything about
                                   language while the car is moving

What varies is the WORDING. What never varies is the CONTENT: every phrasing
in every set carries the direction and, when the provider gave one, the road
name — checked in the selftest rather than promised here. The IMMINENT call is
excluded from all of it and stays one fixed template per maneuver type,
permanently: at two seconds' notice "Right here." is not a phrasing choice.

The text for every call of every maneuver is computed HERE, at route time, and
stored on the route. Nothing generates language while the car is moving; the
timing path only ever looks a string up, and /nav/voice re-reads the same
table, so what is spoken and what is logged cannot drift apart.
"""
from typing import Optional

import config

from . import model as M

# Call types. These are the arbiter's `call_type` and the /nav/voice address.
EARLY = "early"
PRIMARY = "primary"
IMMINENT = "imminent"
ARRIVAL = "arrival"

CALL_TYPES = (EARLY, PRIMARY, IMMINENT, ARRIVAL)

_DIR_WORD = {M.LEFT: "left", M.RIGHT: "right"}


def _road_phrase(maneuver: "M.CanonicalManeuver") -> str:
    """" onto Lincoln Boulevard", or "" when the provider gave no road name.

    Kept as a suffix rather than baked into each template so the no-road-name
    case — which is common, and is exactly §23's `LEFT + NO_ANCHOR -> "Take the
    next left."` — is the same sentence minus a phrase, not a separate table.
    """
    name = (maneuver.road_name or "").strip()
    return f" onto {name}" if name else ""


# THE PHRASINGS, and the rule they all obey: the direction is in every one of
# them, and `{road}` — which expands to " onto Lincoln Boulevard" or to nothing
# at all — is in every one of them too. A set member that could drop either is
# a set member that makes a call less useful than the template it replaced.
#
# Index 0 of each set is the line this file said before there were sets, so a
# route that draws 0 everywhere is the old behaviour exactly.
_EARLY_TURN = (
    "{Dir} turn coming up{road}.",
    "{Dir} coming up{road}.",
    "Next one's a {dir}{road}.",
    "Coming up on a {dir}{road}.",
)
_EARLY_UTURN = (
    "U-turn coming up.",
    "U-turn's next.",
    "Coming up on a U-turn.",
)
_EARLY_ROUNDABOUT = (
    "Roundabout coming up.",
    "Roundabout's next.",
    "Coming up on a roundabout.",
)
_EARLY_RAMP = (
    "Exit coming up.",
    "Exit's next.",
    "Coming up on an exit.",
)
# THE ONE PLACE A CALL WAS ADDED RATHER THAN REPHRASED. A long leg with no
# maneuver on it used to produce nothing at all, which is correct in the sense
# that there is no turn to call and wrong in the sense that a passenger says
# "stay on this" and a GPS says nothing. No distance, like every other early
# line: the planner decides WHEN this is worth saying, and a leg length said
# out loud here would be a second, worse answer to that question.
_EARLY_KEEP = (
    "Stay on {road_this}.",
    "Stay on {road_this} for now.",
    "We're on {road_this} for a while.",
)


def _variant(options, index: int) -> str:
    """One phrasing out of a set, chosen by position rather than by chance.

    `index` is the maneuver's sequence plus the route generation's offset, so
    consecutive maneuvers walk the set instead of repeating, and the same route
    built twice says the same words both times.
    """
    return options[index % len(options)]


def _fill(template: str, maneuver: "M.CanonicalManeuver") -> str:
    d = _DIR_WORD.get(maneuver.direction) or ""
    name = (maneuver.road_name or "").strip()
    return template.format(
        dir=d, Dir=d.capitalize(),
        road=_road_phrase(maneuver),
        # A leg is described by the road it is ON, a turn by the road it goes
        # ONTO — and a leg whose road the provider did not name is "this",
        # which is what a passenger says when they mean the one we are on.
        road_this=name or "this")


def early_text(maneuver: "M.CanonicalManeuver", variant: int = 0) -> Optional[str]:
    """The optional preparation line. No distance, no camera, no urgency."""
    d = _DIR_WORD.get(maneuver.direction)
    if maneuver.type == M.TURN and d:
        return _fill(_variant(_EARLY_TURN, variant), maneuver)
    if maneuver.type == M.UTURN:
        return _variant(_EARLY_UTURN, variant)
    if maneuver.type == M.ROUNDABOUT:
        return _variant(_EARLY_ROUNDABOUT, variant)
    if maneuver.type == M.RAMP:
        return _variant(_EARLY_RAMP, variant)
    if maneuver.type in (M.KEEP, M.STRAIGHT):
        return _fill(_variant(_EARLY_KEEP, variant), maneuver)
    return None


def primary_text(maneuver: "M.CanonicalManeuver") -> str:
    """The instruction, with no landmark. The canonical fallback (§27).

    This is what RIO says whenever visual context is missing, uncertain,
    duplicated, stale or simply switched off — which is most of the time, and
    is not a failure. It is a complete, correct navigation instruction on its
    own; the landmark version below is the same instruction, said better.
    """
    d = _DIR_WORD.get(maneuver.direction)
    if maneuver.type == M.TURN and d:
        return f"Take the next {d}{_road_phrase(maneuver)}."
    if maneuver.type == M.UTURN:
        return f"Make a U-turn{_road_phrase(maneuver)}."
    # Merges, ramps, forks, roundabouts and anything a provider hands back that
    # this vocabulary does not model: the provider's own instruction is the
    # deterministic, correct thing to say, and inventing a shorter phrasing for
    # a freeway interchange is how a driver ends up in the wrong lane.
    text = (maneuver.instruction or "").strip()
    if text:
        return text if text.endswith((".", "!", "?")) else text + "."
    return f"Continue{_road_phrase(maneuver)}."


# The contextual sets, one per relation. Same rule as the early sets — every
# phrasing carries the direction, the road name when there is one, and the
# landmark, because a contextual line that drops the road is a landmark
# description rather than an instruction.
_CONTEXTUAL = {
    "NEAR": (
        "Turn {dir} by {label}{road}.",
        "{Dir}{road}, by {label}.",
        "{Dir} at {label}{road}.",
    ),
    "JUST_AFTER": (
        "Turn {dir} just after {label}{road}.",
        "{Dir}{road}, just past {label}.",
        "Just past {label}, {dir}{road}.",
    ),
    "JUST_BEFORE": (
        "Turn {dir} just before {label}{road}.",
        "{Dir}{road}, just before {label}.",
        "Before {label}, {dir}{road}.",
    ),
}


def contextual_text(maneuver: "M.CanonicalManeuver", spoken_label: str,
                    relation: str, variant: int = 0) -> Optional[str]:
    """The differentiated line: the same turn, described by what is out there.

    Only for the maneuver families a landmark can actually describe. "Merge
    onto the 101 by the Shell station" is not how anyone speaks, and a merge is
    out of V1.1 scope anyway.
    """
    d = _DIR_WORD.get(maneuver.direction)
    if maneuver.type != M.TURN or not d or not spoken_label:
        return None
    options = _CONTEXTUAL.get(relation)
    if not options:
        return None
    return _variant(options, variant).format(
        dir=d, Dir=d.capitalize(), label=spoken_label,
        road=_road_phrase(maneuver))


def imminent_text(maneuver: "M.CanonicalManeuver") -> Optional[str]:
    """The backup at the junction. Two words, because there is no time for more.

    Stays armed even when a contextual call has already been spoken (§11C): the
    contextual line explains the turn, this one confirms it is *this* one. Its
    own timing and validity decide whether it is ever heard.
    """
    d = _DIR_WORD.get(maneuver.direction)
    if maneuver.type == M.TURN and d:
        return f"{d.capitalize()} here."
    if maneuver.type in (M.RAMP, M.FORK):
        return "Take this exit."
    if maneuver.type == M.UTURN:
        return "Turn around here."
    return None


# ---------------------------------------------------------------------------
# THE IMMINENT CALL IS A CLOSED SET, AND THAT IS WHY IT CAN BE A FILE
# ---------------------------------------------------------------------------
# Every other call names a road. "Take the next left onto Cloverfield
# Boulevard" cannot be pre-rendered, because there is no set of roads to render
# — which is the same constraint the tire clips run into, and the reason those
# say the thing that is true of all four corners.
#
# The imminent call names nothing. It is two words, deliberately (see
# imminent_text), and the whole set of sentences it can ever produce is four:
#
#     "Left here."  "Right here."  "Take this exit."  "Turn around here."
#
# A closed set of fixed sentences is a set that can be rendered once, offline,
# in her voice, and played from disk at the junction with no network, no queue
# and no deadline to miss. Which matters here more than anywhere else in
# navigation: this is the one line whose worst case IS the point. Dictating it
# meant a budget, a budget meant a timeout, and a timeout meant that the most
# time-critical sentence in the system was also the one most likely to come out
# in the fallback voice — measured at exactly that, one line in eleven, the
# only one that fell back on a clean drive.
#
# So it stops being spoken and starts being played. The dictation path is kept
# behind it, for a clip that is missing or will not decode.
#
# ENUMERATED FROM imminent_text RATHER THAN TYPED OUT. A second list of these
# sentences is a second list to forget: add a maneuver type tomorrow and a
# hand-written table silently stops covering it, which is a turn called in the
# wrong voice at the worst moment. This asks the function.
_CLIP_ID_CHARS = str.maketrans({" ": "_", ".": "", "'": ""})


def imminent_clip_id(text: str) -> str:
    """A stable file name for one imminent sentence. "Left here." -> left_here."""
    return (text or "").strip().lower().translate(_CLIP_ID_CHARS).strip("_")


def imminent_clips() -> dict:
    """{clip_id: sentence} for every imminent line that exists.

    Built by asking imminent_text for one of each shape it answers to, so the
    set cannot fall behind the function that produces it.
    """
    out = {}
    for kind in (M.TURN, M.RAMP, M.FORK, M.UTURN):
        for direction in (M.LEFT, M.RIGHT, M.UNKNOWN):
            text = imminent_text(M.CanonicalManeuver(
                id="_probe", sequence=0, type=kind, direction=direction,
                road_name="", latitude=0.0, longitude=0.0,
                route_distance_position=0.0, polyline_index=0))
            if text:
                out[imminent_clip_id(text)] = text
    return out


def arrival_text(destination_name: str, side: str) -> str:
    """"Your destination is on the right." — and only when the provider said so.

    UNKNOWN omits the side. There is no camera path to this sentence and no
    inference: a side is either provider data or it is not said (§28).
    """
    name = (destination_name or "").strip()
    if side == M.LEFT:
        return "Your destination is on the left."
    if side == M.RIGHT:
        return "Your destination is on the right."
    return f"You've arrived at {name}." if name else "You've arrived."


def route_offset(journey_id: str, generation_id: int) -> int:
    """Where in each phrasing set this route generation starts.

    Per GENERATION rather than per journey, so the same turn is not guaranteed
    the same words after a reroute — the drive has changed, and a line repeated
    verbatim from the plan that was just abandoned is the one place this would
    sound like a recording. Arithmetic on the id rather than a random seed:
    reproducible, and no state to carry.
    """
    return sum(ord(c) for c in (journey_id or "")) + int(generation_id or 0)


def variant_for(route: "M.CanonicalRoute",
                maneuver: "M.CanonicalManeuver") -> int:
    """The index this maneuver's phrasings are drawn at. One definition, two
    callers: the speech table here and the anchor lines in landmarks.py, which
    must agree or one turn's early call and contextual call come out of
    different halves of the register."""
    return route_offset(getattr(route, "journey_id", ""),
                        getattr(route, "generation_id", 0)) + maneuver.sequence


def build(maneuver: "M.CanonicalManeuver", destination_name: str = "",
          arrival_side: str = M.UNKNOWN, variant: int = 0) -> dict:
    """Every line this maneuver can produce, ahead of time.

    `anchors` is filled in separately by the landmark stage, which adds one
    prepared sentence per candidate — so even the contextual line is a lookup
    at drive time, never a formatting step.
    """
    if maneuver.type == M.ARRIVE:
        return {
            EARLY: "Almost there.",
            PRIMARY: arrival_text(destination_name, arrival_side),
            ARRIVAL: arrival_text(destination_name, arrival_side),
        }
    out = {PRIMARY: primary_text(maneuver)}
    e = early_text(maneuver, variant)
    if e:
        out[EARLY] = e
    # NOT varied, and this is the line that must not be. Two seconds from a
    # junction the driver is listening for a word, not a sentence, and every
    # millisecond of novelty is a millisecond of parsing.
    #
    # It is also the line that is not spoken at all any more: `clips` names the
    # pre-rendered file the browser plays instead, off disk, at the junction.
    # Named by the SERVER for the same reason the sentence is — the browser
    # holds neither, it is told both — and named next to the sentence so the
    # two cannot come to disagree about which line the file says.
    i = imminent_text(maneuver)
    if i:
        out[IMMINENT] = i
        out.setdefault("clips", {})[IMMINENT] = imminent_clip_id(i)
    return out


def text_for(route: "M.CanonicalRoute", maneuver_id: str, call_type: str,
             anchor_id: Optional[str] = None) -> Optional[str]:
    """The one text source: (route, maneuver, call type, anchor) -> sentence.

    Returns None for anything that does not resolve, which is what keeps
    /nav/voice a lookup rather than a text-to-speech endpoint. The browser
    sends coordinates into a table; it never sends RIO a sentence to say.
    """
    if not route:
        return None
    man = route.maneuver(maneuver_id)
    if not man:
        return None
    if anchor_id and call_type == PRIMARY:
        for a in man.anchors:
            if a.get("anchor_id") == anchor_id:
                return a.get("speech") or man.speech.get(PRIMARY)
        return None      # an anchor that is not on this route is not a sentence
    # `clips` lives in the same dict and is not a sentence. CALL_TYPES is
    # closed so no caller can ask for it, but a lookup that would return a
    # dict to a text endpoint is worth refusing by name rather than by luck.
    if call_type not in CALL_TYPES:
        return None
    return man.speech.get(call_type)


def destination_reply(status: str, name: str = "", candidates=None,
                      query: str = "") -> str:
    """What RIO says when the driver asks to be taken somewhere.

    Deterministic, like everything else on this path, and for a sharper reason
    than usual: a model composing "Routing to LAX" is a model that can compose
    "Routing to LAS". The destination in this sentence is the one the provider
    resolved, spelled the way the provider spelled it, or it is a question.
    """
    if status == "resolved":
        return f"Routing to {name}." if name else "Routing there now."
    if status == "ambiguous":
        names = [c.get("display_name") or c.get("formatted_address", "")
                 for c in (candidates or []) if c]
        names = [n for n in names if n][:3]
        if len(names) >= 3:
            return (f"I found a few — {names[0]}, {names[1]}, or {names[2]}. "
                    "Which one?")
        if len(names) == 2:
            return f"I found two — {names[0]} or {names[1]}. Which one?"
        return "I found more than one place by that name. Which one did you mean?"
    return (f"I couldn't find {query}." if query
            else "I couldn't find that place.")


def ttl_ms(call_type: str) -> int:
    """How long this call stays true, in milliseconds.

    Expire, never catch up: a queued instruction that outlived its window is
    dropped rather than played late. "Right here" three seconds late is a turn
    already missed being announced into a junction the car is leaving.
    """
    return int(float(config.NAV_SPEECH_TTL_S.get(call_type, 5.0)) * 1000)
