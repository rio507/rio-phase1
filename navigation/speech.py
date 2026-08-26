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
failure state. A model may later vary the phrasing of the EARLY line, which is
the one line that is never time-critical. The imminent call stays a template
permanently.

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


def early_text(maneuver: "M.CanonicalManeuver") -> Optional[str]:
    """The optional preparation line. No distance, no camera, no urgency."""
    d = _DIR_WORD.get(maneuver.direction)
    if maneuver.type == M.TURN and d:
        return f"{d.capitalize()} turn coming up."
    if maneuver.type == M.UTURN:
        return "U-turn coming up."
    if maneuver.type == M.ROUNDABOUT:
        return "Roundabout coming up."
    if maneuver.type in (M.RAMP, M.FORK, M.MERGE, M.KEEP):
        return "Exit coming up." if maneuver.type == M.RAMP else None
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


def contextual_text(maneuver: "M.CanonicalManeuver", spoken_label: str,
                    relation: str) -> Optional[str]:
    """The differentiated line: the same turn, described by what is out there.

    Only for the maneuver families a landmark can actually describe. "Merge
    onto the 101 by the Shell station" is not how anyone speaks, and a merge is
    out of V1.1 scope anyway.
    """
    d = _DIR_WORD.get(maneuver.direction)
    if maneuver.type != M.TURN or not d or not spoken_label:
        return None
    if relation == "NEAR":
        return f"Turn {d} by {spoken_label}."
    if relation == "JUST_AFTER":
        return f"Turn {d} just after {spoken_label}."
    if relation == "JUST_BEFORE":
        return f"Turn {d} just before {spoken_label}."
    return None


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


def build(maneuver: "M.CanonicalManeuver", destination_name: str = "",
          arrival_side: str = M.UNKNOWN) -> dict:
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
    e = early_text(maneuver)
    if e:
        out[EARLY] = e
    i = imminent_text(maneuver)
    if i:
        out[IMMINENT] = i
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
    return man.speech.get(call_type)


def ttl_ms(call_type: str) -> int:
    """How long this call stays true, in milliseconds.

    Expire, never catch up: a queued instruction that outlived its window is
    dropped rather than played late. "Right here" three seconds late is a turn
    already missed being announced into a junction the car is leaving.
    """
    return int(float(config.NAV_SPEECH_TTL_S.get(call_type, 5.0)) * 1000)
