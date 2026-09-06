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
    """The backup at the junction. Short, because there is no time for more.

    Stays armed even when a contextual call has already been spoken (§11C): the
    contextual line explains the turn, this one confirms it is *this* one. Its
    own timing and validity decide whether it is ever heard.

    EVERY MANEUVER THE PROVIDER CAN EMIT HAS A LINE HERE, OR HAS ONE
    DELIBERATELY WITHHELD. It used to cover three shapes out of ten, and the
    silence was not a decision — it was the shapes nobody had written yet:

      A FORK WAS BEING CALLED AN EXIT. FORK_LEFT and FORK_RIGHT both came back
      as "Take this exit.", which is not what a fork is. A fork is the road
      splitting under you and the answer is which side to be on; "take this
      exit" at one is an instruction to leave a road the route stays on. That
      is the only line here that was WRONG rather than missing, and it is the
      reason this function was worth re-reading rather than extending.

      KEEP, MERGE AND ROUNDABOUT HAD NOTHING AT ALL. Six of Google's enum
      values (KEEP_LEFT/RIGHT, MERGE, MERGE_LEFT/RIGHT, ROUNDABOUT_LEFT/RIGHT)
      produced no junction call whatever, which on a freeway is precisely where
      one is wanted: the early call is a minute back and the instruction is the
      provider's own long sentence.

      AND SO DID THE CATCH-ALL. Anything this vocabulary does not recognise
      becomes TURN/UNKNOWN by design (see providers/google._MANEUVER_MAP), and
      TURN with no direction fell through to None. So the one maneuver shape
      guaranteed to exist the day a provider adds an enum was the one shape
      with no line. "This one." says the only thing that is still true when the
      direction is not known, which is also exactly what this call is for.

    KEEP AND FORK SAY THE SAME WORDS, and that is not an oversight. Bearing
    left at a fork and keeping left at a split are one action to a driver, and
    two near-identical sentences two seconds from a junction is the novelty the
    docstring above exists to refuse. One meaning, one line.

    WHAT IS STILL DELIBERATELY SILENT: STRAIGHT (there is no junction to
    confirm), DEPART (nothing has happened yet), and ARRIVE (build() gives
    arrival its own lines and a backup call at a destination is a confirmation
    of nothing).
    """
    d = _DIR_WORD.get(maneuver.direction)

    if maneuver.type == M.TURN:
        # No direction: the unrecognised-provider-enum case. It cannot say
        # which way, and it can still say WHICH JUNCTION, which is the half of
        # this call that matters when a contextual line has already explained
        # the turn.
        return f"{d.capitalize()} here." if d else "This one."
    if maneuver.type == M.UTURN:
        return "Turn around here."
    if maneuver.type == M.RAMP:
        # Left-hand exits exist and this does not name the side. Kept as it
        # was: the primary call carries the provider's own wording, and a
        # two-word line that says "exit" at the exit is not wrong, only terse.
        return "Take this exit."
    if maneuver.type in (M.FORK, M.KEEP):
        return f"Stay {d}." if d else None
    if maneuver.type == M.MERGE:
        return f"Merge {d}." if d else "Merge."
    if maneuver.type == M.ROUNDABOUT:
        # NOT "Second exit.", and the reason is that nothing here knows which
        # exit it is: CanonicalManeuver.exit_information exists on the model
        # and no provider populates it. A number said at a roundabout is a
        # number the driver will act on, so an invented one is the worst thing
        # this file could produce -- the same rule ArrivalInfo follows when it
        # refuses to guess a side (§28). What IS provider data is the
        # direction, so that is what gets said.
        return f"{d.capitalize()} at the roundabout." if d else None
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


# THE WHOLE CANONICAL VOCABULARY, not a list of the types that had lines when
# this was written. That distinction is the entire point of enumerating rather
# than listing: the first version of this walked four types, which was exactly
# the four that already had sentences -- so it would have rendered clips for
# the coverage that existed and stayed silent about the coverage that did not.
# A set built from the answers can only ever confirm what it already knew.
#
# Asked of model.py instead, which is the vocabulary a provider is mapped INTO
# (providers/google._MANEUVER_MAP) and therefore the real bound on what can
# arrive. A type added there gets probed here without anyone remembering to.
_ALL_TYPES = (M.TURN, M.MERGE, M.RAMP, M.FORK, M.ROUNDABOUT, M.KEEP,
              M.STRAIGHT, M.UTURN, M.DEPART, M.ARRIVE)
_ALL_DIRECTIONS = (M.LEFT, M.RIGHT, M.STRAIGHT_DIR, M.UNKNOWN)

# THE MANEUVERS THAT GET NO JUNCTION CALL, WRITTEN DOWN AS A DECISION.
#
# Silence and an unwritten sentence look identical from outside this file, and
# for six enum values they were the same thing until they were read. So the
# ones that are meant to be silent are named here, and the test asserts against
# THIS rather than against whatever imminent_text currently happens to return —
# which makes a new type with no line a failure instead of a fourth entry
# nobody notices.
#
#   STRAIGHT   there is no junction to confirm. The early call already says
#              "Stay on Lincoln", which is the whole of what can be said.
#   DEPART     nothing has happened yet.
#   ARRIVE     build() gives arrival its own lines, and a backup call at a
#              destination confirms nothing — the driver is looking at it.
IMMINENT_SILENT = frozenset({M.STRAIGHT, M.DEPART, M.ARRIVE})

# ...AND THE SECOND REASON A SHAPE IS SILENT, which is not about the type.
#
# "Stay left." without a side is not a shorter instruction, it is a different
# one, and there is no honest two-word version of a fork whose direction is
# unknown — unlike a turn, where "This one." still confirms the junction, or a
# merge, where "Merge." is complete on its own. These three are the shapes
# where the direction IS the instruction.
#
# No provider emits them: FORK, KEEP and ROUNDABOUT are directional in every
# entry of _MANEUVER_MAP. They are named anyway, because "unreachable today"
# and "would be handled correctly" are different claims and the second one is
# the one worth holding.
_NEEDS_DIRECTION = frozenset({M.FORK, M.KEEP, M.ROUNDABOUT})


def imminent_silent(kind: str, direction: str = M.UNKNOWN) -> bool:
    """Is this shape MEANT to have no junction call?

    The whole reason this is a function and not an absence: silence and an
    unwritten sentence are indistinguishable from outside speech.py, and for
    six of Google's enum values they were the same thing until somebody read
    the file. Now a shape that goes quiet without being named here is a test
    failure rather than a gap nobody notices.
    """
    return kind in IMMINENT_SILENT or (
        kind in _NEEDS_DIRECTION and direction not in (M.LEFT, M.RIGHT))


def imminent_shapes() -> dict:
    """{(type, direction): sentence or None} for every shape in the model.

    The None entries are as much of the answer as the sentences are: they are
    the maneuvers that deliberately have no junction call, and a test that
    could not see them could not tell "decided against" from "never written".
    """
    return {
        (kind, direction): imminent_text(M.CanonicalManeuver(
            id="_probe", sequence=0, type=kind, direction=direction,
            road_name="", latitude=0.0, longitude=0.0,
            route_distance_position=0.0, polyline_index=0))
        for kind in _ALL_TYPES for direction in _ALL_DIRECTIONS
    }


def imminent_clips() -> dict:
    """{clip_id: sentence} for every imminent line that exists.

    Built by asking imminent_text for one of each shape the model can hold, so
    the set cannot fall behind the function that produces it — which is what
    makes "render the junction calls" a command rather than a checklist.
    """
    return {imminent_clip_id(t): t
            for t in imminent_shapes().values() if t}


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
