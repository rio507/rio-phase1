"""What RIO says about a maneuver. Deterministic templates, no model (§23).

THE CADENCE IS GOOGLE MAPS', ON PURPOSE
---------------------------------------
This file used to speak its own dialect: "Right turn coming up." at 300 m,
"Take the next right onto 14th St." at 130 m, "Right here." at 35 m. Every one
of those sentences is defensible in isolation and the set of them was wrong,
which the drive of 2026-09-09 (session 738fbb82) shows in one line:

    t=76.6   NAV_CONTEXTUAL_CALL   "Take the next right onto 14th St."   31.1 m
    t=102.3  NAV_EARLY_GUIDANCE    "Coming up on a right onto 14th St."  28.2 m

The preparation line arrived AFTER the instruction and three metres closer to
the junction, because both tiers were timed in seconds-to-turn with a metre
floor underneath, and at 0.12 m/s in traffic both floors were already crossed
before the route had finished loading. A driver hearing that has no idea which
sentence is the one to act on.

Google's cadence is not better because it is Google's. It is better because
every driver already has it in their bones, and a car that says the words in a
different order at different distances is asking the driver to learn a second
system while driving. So:

    ROUTE START   immediately, the whole first move
                  "Head north on Lincoln Blvd, then turn right onto Ocean Ave."
    FAR           distance-phrased, on the ladder for the road class
                  "In half a mile, turn right onto Ocean Ave."
                  "In two miles, take exit 43 toward Sunset Blvd."
    NEAR          the instruction, with the road name and no hedging
                  "Turn right onto Ocean Ave."
    JUNCTION      two words, and only if NEAR was long enough ago to need them
                  "Turn right."
    ARRIVAL       "Your destination is on the right." then "You have arrived."

FIVE TIERS, NOT FIVE ANNOUNCEMENTS. Each is a chance the planner takes only if
it is still true and still adds something; the >10 s rule on JUNCTION is the
sharpest example and is enforced in rio_navplan.js, not here.

DISTANCE IS SAID, AND IT IS SAID ROUNDED
----------------------------------------
The previous design's stated reason for saying no distance at all was that
"the primary call REPLACES distance narration -- a driver who can see the Shell
does not need the number". That is true of the NEAR call, which still carries
no distance. It is not true of the FAR call, whose entire job is to say how
much road is left before the driver has to do anything. A far call with no
number is a car saying "something is coming" and leaving the driver to guess
whether to start moving over now.

The number is a rounded phrase from distance.py, computed at route load from
the TIER's nominal distance rather than from the live measurement, so two
drives down the same road say the same words at the same corner and nothing
formats language while the car is moving.

WHY THERE IS NO LLM HERE, AND WHY THERE NEVER WILL BE ON THIS PATH
------------------------------------------------------------------
Every sentence RIO can say about a maneuver is enumerable before the drive
starts: a direction, a road name, a rounded distance, and at most one landmark
from a list fetched at route load. Enumerating them costs a dictionary lookup
and removes latency, hallucination, a validation layer, a test surface and an
entire class of failure state.

VARIATION, AND WHERE IT IS NOW ALLOWED TO LIVE
----------------------------------------------
It used to live in the FAR and NEAR calls, drawn from a phrasing set indexed by
position. That was the right instinct aimed at the wrong lines. The whole value
of a conventional cadence is that the words do not move: "In half a mile, turn
right onto Ocean Ave." is the sentence a driver has heard ten thousand times
and can parse without listening to it, and a synonym for it is a sentence they
have to parse. So the navigation calls are now FIXED, one form each.

What varies instead is the ANCHOR line -- the one sentence in navigation that
Google cannot say, because it needs a camera. "Turn right by the Shell station"
is RIO's own, it is where her voice belongs, and it is still drawn from a set
indexed by `variant_for` so a drive does not repeat itself. See _CONTEXTUAL.

The text for every call of every maneuver is computed HERE, at route time, and
stored on the route. Nothing generates language while the car is moving; the
timing path only ever looks a string up, and /nav/voice re-reads the same
table, so what is spoken and what is logged cannot drift apart.
"""
import re
from typing import List, Optional

import config

from . import distance as D
from . import model as M

# Call types. These are the arbiter's `call_type` and the /nav/voice address.
DEPART = "depart"
FAR = "far"
FAR_MID = "far_mid"
NEAR = "near"
JUNCTION = "junction"
ARRIVAL = "arrival"
ARRIVED = "arrived"

CALL_TYPES = (DEPART, FAR, FAR_MID, NEAR, JUNCTION, ARRIVAL, ARRIVED)

# THE OLD NAMES, kept as an address and nothing else.
#
# `early/primary/imminent` were the wire format between the server's speech
# table, the browser's planner, /nav/voice and three test harnesses. Renaming
# them to what they now are is worth doing -- "primary" no longer describes the
# call that carries the road name -- but a phone running yesterday's cached
# page asking for "imminent" should get the junction line rather than silence
# at a junction. Lookup only: nothing WRITES these keys.
_LEGACY = {"early": FAR, "primary": NEAR, "imminent": JUNCTION}

_DIR_WORD = {M.LEFT: "left", M.RIGHT: "right"}


# ---------------------------------------------------------------------------
# THE TIER LADDER
# ---------------------------------------------------------------------------
# Where each call is made, in metres before the maneuver, by road class. These
# are Google's distances: half a mile and 150 m on surface streets, two miles
# and one mile and a quarter mile on a freeway. They travel to the browser in
# the route payload (`speech.tiers`), so the planner reads a number rather than
# holding a policy -- and so a drive log records the ladder the drive actually
# used.
#
# The JUNCTION row is a floor, not an announcement distance: it is where the
# two-word confirmation would go if the NEAR call is far enough behind to make
# it worth having. 150 m on a freeway rather than 35 is not a longer warning,
# it is the same ~5 seconds at 30 m/s that 35 m is at 7.
def _ladder(road_class: str) -> dict:
    table = getattr(config, "NAV_TIER_DISTANCES_M", None) or {}
    row = table.get(road_class) or table.get(M.SURFACE) or {}
    return dict(row)


def tiers_for(maneuver: "M.CanonicalManeuver",
              leg_m: Optional[float] = None) -> List[dict]:
    """[{call, at_m}] for this maneuver, outermost first.

    FAR_MID exists only on the highway ladder: a freeway exit gets a two-mile
    call and a one-mile call, because two miles out is where lane changes start
    and one mile out is where a driver who missed the first one still has room.

    A TIER FURTHER OUT THAN THE LEG IS LONG IS NOT A TIER, and dropping it here
    is the difference between a cadence and a set of thresholds. `leg_m` is the
    road between the previous maneuver (or the start of the route) and this
    one; a half-mile call on a 414 m leg cannot be made half a mile out, so it
    fires the moment the maneuver becomes active and says "in half a mile"
    while the car is 414 m away. That is not a rounding, it is wrong, and it
    is exactly what the replay of session a2da65cd produced before this
    existed: two far calls at 414 m and 485 m against an 805 m tier.

    The driver loses nothing. A 400 m leg gets the route-start line or the
    previous turn's instruction, and then the near call -- which is what Google
    does on a short block, and is why nobody hears "in half a mile" between two
    junctions four hundred metres apart.
    """
    row = _ladder(maneuver.road_class)
    out = []
    for call in (FAR, FAR_MID, NEAR, JUNCTION):
        at = row.get(call)
        if not at:
            continue
        # The near call and the junction call are the instruction and its
        # confirmation; they are made from wherever the car is when it becomes
        # due, however short the leg. Only the distance-PHRASED calls, whose
        # words contain a number, are dropped for want of room.
        if leg_m is not None and call in (FAR, FAR_MID) and float(at) > leg_m:
            continue
        out.append({"call": call, "at_m": round(float(at), 1)})
    return out


# ---------------------------------------------------------------------------
# THE ACTION PHRASE — the imperative every call is built out of
# ---------------------------------------------------------------------------
# Lower case, no full stop, no leading preposition: "turn right onto Ocean Ave".
# One function, because the FAR call, the NEAR call, the route-start line and
# the "then" continuation are the SAME instruction in four positions, and a
# system that formats them separately is a system where they can disagree about
# what the maneuver is.
def _road_suffix(maneuver: "M.CanonicalManeuver") -> str:
    name = (maneuver.road_name or "").strip()
    return f" onto {name}" if name else ""


def _exit_phrase(maneuver: "M.CanonicalManeuver") -> Optional[str]:
    """"take exit 43 toward Sunset Blvd" — provider data only, never invented."""
    info = maneuver.exit_information or {}
    number = str(info.get("number") or "").strip()
    toward = str(info.get("toward") or "").strip()
    if not number:
        return None
    out = f"take exit {number}"
    if toward:
        out += f" toward {toward}"
    return out


def action_phrase(maneuver: "M.CanonicalManeuver") -> str:
    """The instruction as a bare imperative clause."""
    d = _DIR_WORD.get(maneuver.direction)

    if maneuver.type == M.TURN:
        return f"turn {d}{_road_suffix(maneuver)}" if d else "take the next turn"
    if maneuver.type == M.UTURN:
        return "make a U-turn"
    if maneuver.type == M.RAMP:
        # The exit number is what is on the sign, so it leads when there is
        # one. Without it, the provider's own line is the honest fallback --
        # inventing a shorter phrasing for an interchange is how a driver ends
        # up in the wrong lane.
        ex = _exit_phrase(maneuver)
        if ex:
            return ex
        return f"take the exit{_road_suffix(maneuver)}"
    if maneuver.type == M.MERGE:
        return f"merge {d}{_road_suffix(maneuver)}" if d else f"merge{_road_suffix(maneuver)}"
    if maneuver.type in (M.FORK, M.KEEP):
        return f"keep {d}" if d else "keep going"
    if maneuver.type == M.ROUNDABOUT:
        # NOT "take the second exit": CanonicalManeuver.exit_information is
        # populated for ramps and no provider fills it for roundabouts, and a
        # number said at a roundabout is a number the driver will act on (§28).
        return f"go {d} at the roundabout" if d else "go around the roundabout"
    if maneuver.type == M.STRAIGHT:
        name = (maneuver.road_name or "").strip()
        return f"continue on {name}" if name else "continue straight"
    # Anything the vocabulary does not model: the provider's own sentence,
    # lowered into clause position.
    text = (maneuver.instruction or "").strip().rstrip(".")
    if text:
        return text[0].lower() + text[1:]
    return "continue"


def _sentence(clause: str) -> str:
    clause = (clause or "").strip()
    if not clause:
        return ""
    out = clause[0].upper() + clause[1:]
    return out if out.endswith((".", "!", "?")) else out + "."


# ---------------------------------------------------------------------------
# THE FIVE TIER SENTENCES
# ---------------------------------------------------------------------------
def far_text(maneuver: "M.CanonicalManeuver", at_m: float,
             units: str = D.IMPERIAL) -> str:
    """"In half a mile, turn right onto Ocean Ave."

    The distance comes from the TIER, not from where the car happens to be when
    the call fires. A call made at 790 m and one made at 815 m are both the
    half-mile call and both say "half a mile" — which is what makes the phrase
    a beat in the approach rather than a reading off an instrument.
    """
    return f"{D.in_phrase(at_m, units)}, {action_phrase(maneuver)}."


def near_text(maneuver: "M.CanonicalManeuver",
              chained: Optional["M.CanonicalManeuver"] = None) -> str:
    """"Turn right onto Ocean Ave." — or with the next move chained on.

    No "coming up", no "take the next": at 150 m the turn is the one in front
    of the driver and naming it is the whole instruction. `chained` is the
    following maneuver when it is close enough that the two are one move; see
    chain_window_m.
    """
    clause = action_phrase(maneuver)
    if chained is not None:
        clause += f", then {action_phrase(chained)}"
    return _sentence(clause)


# Google's DEPART step comes in two shapes, and only one of them composes.
#
#   "Head north on Lincoln Blvd"        the road you are ON. Complementary to
#                                       the move that follows it.
#   "Head northeast toward 16th St"     the road you are AIMED AT, which is
#                                       what it says when the road you are on
#                                       has no name — a car park, a driveway,
#                                       an unnamed segment.
#
# The second one names the road the first maneuver turns onto, so chaining it
# says that road twice. Measured on a live route to Griffith Observatory:
#
#     "Head northeast toward 16th St, then turn left onto 16th St."
#
# Not wrong, and not a sentence anyone says. The direction is the useful half
# of the head; the road name belongs on the instruction that acts on it.
_TOWARD = re.compile(r"\s*\b(?:toward|towards)\s+(.+)$", re.IGNORECASE)
_ON = re.compile(r"\bon\s+(.+)$", re.IGNORECASE)


def _on_road(head: str) -> str:
    """The road the depart step says we are ON, if it says one.

    Any `toward` clause comes off first: "Head north on Lincoln Blvd toward
    5th St" is a car on Lincoln, and a naive match to end-of-string reads the
    road as "Lincoln Blvd toward 5th St" and then matches nothing.
    """
    m = _ON.search(_TOWARD.sub("", head or ""))
    return m.group(1).strip() if m else ""


def _same_road(a: str, b: str) -> bool:
    """Are these the same road, allowing for punctuation and case?

    Both strings come out of the same route response, so the provider spells
    them the same way and normalising case and punctuation is enough. No
    abbreviation table: guessing that "St" and "Street" are the same road is a
    guess, and a wrong one drops a road name the driver needed.
    """
    def norm(t):
        return " ".join(re.sub(r"[^\w\s]", " ", (t or "").lower()).split())
    a, b = norm(a), norm(b)
    return bool(a) and a == b


def depart_text(route: "M.CanonicalRoute") -> Optional[str]:
    """"Head north on Lincoln Blvd, then turn right onto Ocean Ave."

    Said ONCE, immediately, the moment the route starts — which is the one
    thing the old cadence had no tier for at all. Session 738fbb82 started a
    route at t=76.1 and the first thing the driver heard was a turn call half a
    second later at 31 m; nothing ever told them what road they were on or
    which way they were pointing.

    Google's DEPART step is provider text and is used verbatim, with two
    subtractions, both of them a road said twice in one sentence:

      "...toward 16th St" + "turn left onto 16th St"   -> the toward goes
      "...on Lincoln Blvd" + "continue on Lincoln Blvd" -> the clause goes

    See the note above _TOWARD and the one in the body below.
    """
    head = (route.depart_instruction or "").strip().rstrip(".")
    first = None
    for m in route.maneuvers:
        if m.type not in (M.DEPART, M.ARRIVE):
            first = m
            break
    if not head and first is None:
        return None
    if first is None:
        return _sentence(head)
    if not head:
        return near_text(first)

    # "...toward 16th St" + "turn left onto 16th St" -> drop the toward.
    m_toward = _TOWARD.search(head)
    if m_toward and _same_road(m_toward.group(1), first.road_name):
        trimmed = head[:m_toward.start()].strip().rstrip(",")
        # Only if something is left worth saying. "Head northeast" is; a head
        # that was nothing BUT the toward phrase is not, and keeping it whole
        # is better than announcing a bare direction that came from nowhere.
        if trimmed:
            head = trimmed

    # "...on Lincoln Blvd" + "continue on Lincoln Blvd" -> drop the clause.
    #
    # The other half of the same redundancy, and it collapses the other way
    # round: there the road name belonged on the instruction, here the
    # instruction adds nothing the head has not said. "Head north on Lincoln
    # Blvd" already carries the road, the direction AND the fact that there is
    # nothing to do yet, which is the whole of what a continue-straight means.
    #
    # NARROW ON PURPOSE, on two counts. It is STRAIGHT only -- a KEEP's action
    # phrase is "keep left", which names no road, and collapsing on a road
    # comparison would delete a real instruction. And it is the SAME road only
    # -- Google maps NAME_CHANGE onto STRAIGHT, so "Head north on Lincoln Blvd,
    # then continue on Foo Ave" is a road that changes name under the car,
    # which is the opposite of redundant.
    if (first.type == M.STRAIGHT and first.road_name
            and _same_road(_on_road(head), first.road_name)):
        return _sentence(head)

    return _sentence(f"{head}, then {action_phrase(first)}")


# THE JUNCTION LINE IS A CLOSED SET, AND THAT IS WHY IT CAN BE A FILE.
#
# Every other call names a road or a distance. This one names neither: it is
# the two words a driver needs while their hands are already moving, and the
# whole set of sentences it can ever produce is small enough to render once,
# offline, in her voice, and play off disk at the junction with no network, no
# queue and no deadline to miss.
#
# That matters here more than anywhere else in navigation: this is the one line
# whose worst case IS the point. Dictating it meant a budget, a budget meant a
# timeout, and a timeout meant the most time-critical sentence in the system
# was also the one most likely to come out in the fallback voice — measured at
# exactly that, one line in eleven, the only one that fell back on a clean
# drive.
def junction_text(maneuver: "M.CanonicalManeuver") -> Optional[str]:
    """The two-word confirmation at the junction, or None where there is none.

    EVERY MANEUVER THE PROVIDER CAN EMIT HAS A LINE HERE, OR HAS ONE
    DELIBERATELY WITHHELD — see IMMINENT_SILENT and _NEEDS_DIRECTION below.
    Silence and an unwritten sentence look identical from outside this file,
    and for six of Google's enum values they were the same thing until somebody
    read it.
    """
    d = _DIR_WORD.get(maneuver.direction)

    if maneuver.type == M.TURN:
        # No direction: the unrecognised-provider-enum case. It cannot say
        # which way and it can still say WHICH JUNCTION, which is the half of
        # this call that matters when the near call already said the rest.
        return f"Turn {d}." if d else "This one."
    if maneuver.type == M.UTURN:
        return "Make a U-turn."
    if maneuver.type == M.RAMP:
        # Left-hand exits exist and this does not name the side; the near call
        # carried the exit number and the sign, which is what the driver is
        # looking for out of the windscreen.
        return "Take the exit."
    if maneuver.type in (M.FORK, M.KEEP):
        return f"Keep {d}." if d else None
    if maneuver.type == M.MERGE:
        return f"Merge {d}." if d else "Merge."
    if maneuver.type == M.ROUNDABOUT:
        return f"{d.capitalize()} at the roundabout." if d else None
    return None


# --- the contextual (landmark) line, which is the one that still varies ------
#
# Same rule as before: every phrasing carries the direction, the road name when
# there is one, and the landmark — a contextual line that drops the road is a
# landmark description rather than an instruction. This replaces the NEAR call
# when an anchor has been verified, so it is written in the NEAR call's
# register: an instruction, not a hedge.
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


def _variant(options, index: int) -> str:
    """One phrasing out of a set, chosen by position rather than by chance.

    `index` is the maneuver's sequence plus the route generation's offset, so
    consecutive maneuvers walk the set instead of repeating, and the same route
    built twice says the same words both times.
    """
    return options[index % len(options)]


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
        road=_road_suffix(maneuver))


# ---------------------------------------------------------------------------
# CLIPS
# ---------------------------------------------------------------------------
# ENUMERATED FROM junction_text RATHER THAN TYPED OUT. A second list of these
# sentences is a second list to forget: add a maneuver type tomorrow and a
# hand-written table silently stops covering it, which is a turn called in the
# wrong voice at the worst moment. This asks the function.
_CLIP_ID_CHARS = str.maketrans({" ": "_", ".": "", "'": "", "-": "_"})


def junction_clip_id(text: str) -> str:
    """A stable file name for one junction sentence. "Turn left." -> turn_left."""
    return (text or "").strip().lower().translate(_CLIP_ID_CHARS).strip("_")


# Kept under the old name too: tools/render_alerts.py and the audio manifest
# address clips by these functions, and one rename is not worth a silent miss.
imminent_clip_id = junction_clip_id

# THE WHOLE CANONICAL VOCABULARY, not a list of the types that had lines when
# this was written. That distinction is the entire point of enumerating rather
# than listing: a set built from the answers can only ever confirm what it
# already knew. Asked of model.py instead, which is the vocabulary a provider
# is mapped INTO (providers/google._MANEUVER_MAP) and therefore the real bound
# on what can arrive.
_ALL_TYPES = (M.TURN, M.MERGE, M.RAMP, M.FORK, M.ROUNDABOUT, M.KEEP,
              M.STRAIGHT, M.UTURN, M.DEPART, M.ARRIVE)
_ALL_DIRECTIONS = (M.LEFT, M.RIGHT, M.STRAIGHT_DIR, M.UNKNOWN)

# THE MANEUVERS THAT GET NO JUNCTION CALL, WRITTEN DOWN AS A DECISION.
#
#   STRAIGHT   there is no junction to confirm. The far call already said
#              "continue on Lincoln", which is the whole of what can be said.
#   DEPART     nothing has happened yet — and the route-start line is depart's
#              only sentence, said once, by depart_text.
#   ARRIVE     arrival has its own two lines and a backup call at a destination
#              confirms nothing — the driver is looking at it.
IMMINENT_SILENT = frozenset({M.STRAIGHT, M.DEPART, M.ARRIVE})

# ...AND THE SECOND REASON A SHAPE IS SILENT, which is not about the type.
#
# "Keep going." without a side is not a shorter instruction, it is a different
# one, and there is no honest two-word version of a fork whose direction is
# unknown — unlike a turn, where "This one." still confirms the junction, or a
# merge, where "Merge." is complete on its own. These three are the shapes
# where the direction IS the instruction.
_NEEDS_DIRECTION = frozenset({M.FORK, M.KEEP, M.ROUNDABOUT})


def imminent_silent(kind: str, direction: str = M.UNKNOWN) -> bool:
    """Is this shape MEANT to have no junction call?

    The whole reason this is a function and not an absence: silence and an
    unwritten sentence are indistinguishable from outside speech.py. Now a
    shape that goes quiet without being named here is a test failure rather
    than a gap nobody notices.
    """
    return kind in IMMINENT_SILENT or (
        kind in _NEEDS_DIRECTION and direction not in (M.LEFT, M.RIGHT))


def junction_shapes() -> dict:
    """{(type, direction): sentence or None} for every shape in the model.

    The None entries are as much of the answer as the sentences are.
    """
    return {
        (kind, direction): junction_text(M.CanonicalManeuver(
            id="_probe", sequence=0, type=kind, direction=direction,
            road_name="", latitude=0.0, longitude=0.0,
            route_distance_position=0.0, polyline_index=0))
        for kind in _ALL_TYPES for direction in _ALL_DIRECTIONS
    }


imminent_shapes = junction_shapes


def junction_clips() -> dict:
    """{clip_id: sentence} for every junction line that exists."""
    return {junction_clip_id(t): t
            for t in junction_shapes().values() if t}


imminent_clips = junction_clips


# ---------------------------------------------------------------------------
# ARRIVAL
# ---------------------------------------------------------------------------
def arrival_text(destination_name: str, side: str) -> str:
    """"Your destination is on the right." — the 150 m call.

    UNKNOWN omits the side and says "ahead", which is true of every
    destination and claims nothing. There is no camera path to this sentence
    and no inference: a side is either provider data or it is not said (§28).
    """
    if side == M.LEFT:
        return "Your destination is on the left."
    if side == M.RIGHT:
        return "Your destination is on the right."
    return "Your destination is ahead."


def arrived_text(destination_name: str = "") -> str:
    """"You have arrived." — the one at the kerb.

    Google's exact words, and deliberately not "You've arrived at 2411 Lincoln
    Blvd": by this point the driver is looking at the number on the building,
    and the address read back is the car narrating rather than navigating.
    """
    return "You have arrived."


# ---------------------------------------------------------------------------
# THE PHRASING INDEX
# ---------------------------------------------------------------------------
def route_offset(journey_id: str, generation_id: int) -> int:
    """Where in each phrasing set this route generation starts.

    Only the anchor lines vary now, but they vary for the same reasons they
    always did: reproducible from the drive log, no state to carry, and a turn
    phrased differently after a reroute because the drive has changed.
    """
    return sum(ord(c) for c in (journey_id or "")) + int(generation_id or 0)


def variant_for(route: "M.CanonicalRoute",
                maneuver: "M.CanonicalManeuver") -> int:
    """The index this maneuver's anchor phrasings are drawn at."""
    return route_offset(getattr(route, "journey_id", ""),
                        getattr(route, "generation_id", 0)) + maneuver.sequence


# ---------------------------------------------------------------------------
# BUILD
# ---------------------------------------------------------------------------
def chain_window_m() -> float:
    """How close the next maneuver has to be to ride on this one's near call."""
    return float(getattr(config, "NAV_CHAIN_WINDOW_M", 200.0))


def build(maneuver: "M.CanonicalManeuver", destination_name: str = "",
          arrival_side: str = M.UNKNOWN, variant: int = 0,
          chained: Optional["M.CanonicalManeuver"] = None,
          units: Optional[str] = None,
          leg_m: Optional[float] = None) -> dict:
    """Every line this maneuver can produce, ahead of time.

    `anchors` is filled in separately by the landmark stage, which adds one
    prepared sentence per candidate — so even the contextual line is a lookup
    at drive time, never a formatting step.
    """
    units = units or D.units_from_config()
    if maneuver.type == M.ARRIVE:
        # TWO LINES, and no far call. "Almost there." used to sit on this
        # maneuver's early tier and it is a sentence about a feeling rather
        # than about a distance; the two that are left are the two Google says
        # and the two a driver acts on -- which side to look at, and that this
        # is the place.
        at = float(getattr(config, "NAV_ARRIVAL_CALL_M", 150.0))
        if leg_m is not None:
            at = min(at, max(0.0, leg_m))
        return {
            ARRIVAL: arrival_text(destination_name, arrival_side),
            ARRIVED: arrived_text(destination_name),
            "tiers": [{"call": ARRIVAL, "at_m": round(at, 1)}],
        }

    tiers = tiers_for(maneuver, leg_m)
    out: dict = {"tiers": tiers}
    for tier in tiers:
        call, at_m = tier["call"], tier["at_m"]
        if call in (FAR, FAR_MID):
            out[call] = far_text(maneuver, at_m, units)
        elif call == NEAR:
            out[NEAR] = near_text(maneuver, chained)
        elif call == JUNCTION:
            j = junction_text(maneuver)
            if j:
                out[JUNCTION] = j
                out.setdefault("clips", {})[JUNCTION] = junction_clip_id(j)

    # A near call is the one line every maneuver must have, tier table or not:
    # it is the instruction. The ladder can drop a far call on a short leg and
    # a junction call on a shape that has none, never this.
    out.setdefault(NEAR, near_text(maneuver, chained))
    if chained is not None:
        out["chained_to"] = chained.id
    return out


def build_route(route: "M.CanonicalRoute", units: Optional[str] = None) -> str:
    """Fill in every maneuver's speech table and return the route-start line.

    ROUTE-LEVEL, because two of the five tiers are: the depart line needs the
    route's own heading sentence and the first maneuver, and the "then" chain
    needs the maneuver AFTER this one. `build()` alone could see neither, which
    is why a per-maneuver loop in service.py could not produce either sentence.
    """
    units = units or D.units_from_config()
    window = chain_window_m()
    mans = route.maneuvers or []
    prev_at = 0.0
    for i, man in enumerate(mans):
        # THE ROAD AVAILABLE FOR THIS MANEUVER'S APPROACH: from the maneuver
        # before it, or from the start of the route for the first one.
        leg_m = max(0.0, float(man.route_distance_position) - prev_at)
        prev_at = float(man.route_distance_position)
        chained = None
        nxt = mans[i + 1] if i + 1 < len(mans) else None
        # Chained only when the two are genuinely one move. An ARRIVE riding on
        # the last turn's near call would say "turn right onto Ocean, then your
        # destination is ahead", which is a different sentence and gets its own
        # tier.
        if (nxt is not None and nxt.type not in (M.ARRIVE, M.DEPART)
                and man.type not in (M.ARRIVE, M.DEPART)
                and 0 < (nxt.route_distance_position
                         - man.route_distance_position) <= window):
            chained = nxt
        man.speech = build(man, route.destination.display_name,
                           route.arrival.side, variant=variant_for(route, man),
                           chained=chained, units=units, leg_m=leg_m)
    return depart_text(route) or ""


def text_for(route: "M.CanonicalRoute", maneuver_id: str, call_type: str,
             anchor_id: Optional[str] = None) -> Optional[str]:
    """The one text source: (route, maneuver, call type, anchor) -> sentence.

    Returns None for anything that does not resolve, which is what keeps
    /nav/voice a lookup rather than a text-to-speech endpoint. The browser
    sends coordinates into a table; it never sends RIO a sentence to say.
    """
    if not route:
        return None
    call_type = _LEGACY.get(call_type, call_type)
    man = route.maneuver(maneuver_id)
    if not man:
        return None
    if anchor_id and call_type == NEAR:
        for a in man.anchors:
            if a.get("anchor_id") == anchor_id:
                return a.get("speech") or man.speech.get(NEAR)
        return None      # an anchor that is not on this route is not a sentence
    # `clips` and `tiers` live in the same dict and are not sentences.
    # CALL_TYPES is closed so no caller can ask for them, but a lookup that
    # would return a dict to a text endpoint is worth refusing by name.
    if call_type not in CALL_TYPES:
        return None
    return man.speech.get(call_type)


def destination_reply(status: str, name: str = "", candidates=None,
                      query: str = "") -> str:
    """What RIO says when the driver asks to be taken somewhere.

    FIRST PERSON, because the turn calls are hers. "Routing to Century City"
    is a status line from a machine; "I'll take you to Century City" is the
    person who is about to call every turn saying she has it. Nothing on this
    path may describe navigation in the third person — see the nav-voice lint
    in tools/nav_server_selftest.py.

    Deterministic, like everything else here, and for a sharper reason than
    usual: a model composing "I'll take you to LAX" is a model that can compose
    "I'll take you to LAS". The destination in this sentence is the one the
    provider resolved, spelled the way the provider spelled it, or it is a
    question.
    """
    if status == "resolved":
        return f"Got it — I'll take you to {name}." if name else "Got it, taking you there."
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
    dropped rather than played late. "Turn right" three seconds late is a turn
    already missed being announced into a junction the car is leaving.
    """
    call_type = _LEGACY.get(call_type, call_type)
    return int(float(config.NAV_SPEECH_TTL_S.get(call_type, 5.0)) * 1000)
