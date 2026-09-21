"""The reading against the tracks, when they contradict.

THE RULE, AND IT IS ASYMMETRIC ON PURPOSE
-----------------------------------------
THE DETECTOR WINS ON EXISTENCE. If RF-DETR is holding a confirmed pedestrian
and a confirmed motorcycle, "TRAFFIC: none" is false, and it may not be handed
to RIO as what the camera sees. The detector is measured geometry with a size
floor, a plausibility gate and a tracker behind it; the reading is a caption
from a 2B model that was shown one frame.

THE DETECTOR DOES NOT WIN ON ABSENCE, and this is the half that would be wrong
to make symmetric. RF-DETR has seven classes. A tractor, a horse, a fallen
branch, a person behind a windscreen, anything at all past the size floor --
none of those are tracks, and all of them are things a reading may correctly
mention. "The detector sees nothing" is not evidence that the model invented
something, so a reading that REPORTS traffic is never contradicted here.

So exactly one contradiction is recognised: a reading that claims the road is
empty while the detector is holding road users on it.

WHAT IS NOT CONTESTED, AND WHY NOT
----------------------------------
RISK. "RISK: none seen" is not an existence claim -- it is a judgement about
what could bite in the next few seconds, and a tracked pedestrian on a footpath
is not a risk. Overruling that with a track count would be the detector winning
on something it has no opinion about, and would fire on nearly every urban
frame until nobody read the marker. Existence is the claim the detector can
settle; it settles that one and no others.

ROAD. Lanes and surface are not road users.

WHAT HAPPENS TO A CONTESTED READING
-----------------------------------
It is NOT deleted and NOT rewritten. The model said what it said and that stays
on the record -- rewriting a model's output and then showing it as the model's
output is how a card stops being evidence. The field is MARKED, the detector's
own account is attached beside it, and both go to the card and to RIO together
with a rule telling her which one is authoritative on existence. She is told,
in as many words, that the camera read the road as clear and the tracker
disagrees, and to go with the tracker.
"""
import re

# What a claim of "nothing there" looks like in this field. Matched as whole
# words against the field only, so "none" inside "no one else" is not a hit and
# "two cars, none braking" is not either -- that one is caught by the presence
# test below instead.
# Leading filler that carries no claim: "the road is clear" and "road clear"
# are the same sentence, and a list of every article would be the wrong shape.
_LEAD_RE = re.compile(r"^(?:the|a|an)\s+", re.IGNORECASE)

# ...and the trailing place-words that say WHERE the nothing is, which does not
# change that it is nothing: "none ahead", "no traffic in the lane ahead".
_TAIL_RE = re.compile(
    r"\s+(?:seen|visible|observed|detected|present|here|there|ahead|"
    r"in\s+view|in\s+sight|on\s+the\s+road|on\s+it|in\s+the\s+lane|"
    r"in\s+front|nearby|around|at\s+all|of\s+(?:traffic|note)|"
    r"is|are|in|the|this|frame|lane|road|street|view)\b",
    re.IGNORECASE)

_EMPTY_RE = re.compile(
    r"^(?:"
    r"none|"
    r"no\s+(?:traffic|vehicles?|cars?|road\s+users?|one|other\s+\w+)|"
    r"nothing|"
    r"clear|"
    r"empty|"
    r"(?:road|lane|street)\s+(?:is\s+)?(?:clear|empty)|"
    r"unremarkable|"
    r"n/?a|-{1,2}"
    r")$", re.IGNORECASE)


def claims_empty(text: str) -> bool:
    """Does this TRAFFIC field claim there is nothing on the road?

    Deliberately narrow: a whole-field match on a short set of ways of saying
    nothing. A field this cannot classify is left alone rather than guessed at,
    because a false contest is worse than a missed one -- it puts a warning on
    a card that is telling the truth, and a marker nobody believes is a marker
    that does nothing.
    """
    t = (text or "").strip().strip(".,;!").strip()
    if not t:
        return False
    t = _LEAD_RE.sub("", t)
    # ASKED AT EVERY LENGTH, LONGEST FIRST. Peeling all the place-words off
    # before asking gets "no road users" wrong -- "road" comes off, "no users"
    # is left, and that is not a phrase this knows. So the question is put to
    # the whole field first and to each shorter form after, and the first yes
    # wins. The loop rather than one pass because "none in view at all" is two
    # of them.
    seen = set()
    while t and t not in seen:
        if _EMPTY_RE.match(t):
            return True
        seen.add(t)
        t = _TAIL_RE.sub("", t).strip().strip(".,;").strip()
    return False


def _phrase(classes: dict) -> str:
    """{'pedestrian': 1, 'car': 2} -> 'two cars and one pedestrian'."""
    words = {1: "one", 2: "two", 3: "three", 4: "four", 5: "five",
             6: "six", 7: "seven", 8: "eight", 9: "nine"}
    parts = []
    for label, n in sorted(classes.items(), key=lambda kv: (-kv[1], kv[0])):
        name = label if n == 1 else (label + "s")
        parts.append(f"{words.get(n, n)} {name}")
    if not parts:
        return "nothing"
    if len(parts) == 1:
        return parts[0]
    return ", ".join(parts[:-1]) + " and " + parts[-1]


def check(fields, census) -> dict:
    """Compare a split reading against the detector's census.

    `fields` is rio_prompts.split_sensor_reading(...)["fields"] -- it is COPIED,
    never mutated, so a caller holding the parsed reading still has what the
    model said.

    -> {"fields": [...], "contested": [...], "detector": {...}|None}

    Each returned field keeps `name`, `text` and `present`, and a contested one
    gains `contested: True`, `detector` (the tracker's account in words) and
    `why`. `contested` is the list of field names, so a caller can ask the one
    question it cares about without walking the rows.
    """
    out = [dict(f) for f in (fields or [])]
    if not census or not census.get("n_tracked"):
        return {"fields": out, "contested": [], "detector": None}

    account = _phrase(census["classes"])
    contested = []
    for f in out:
        if f.get("name") != "TRAFFIC":
            continue
        # A field the model did not write at all is not a claim of absence --
        # it is a missing field, which the card already says plainly. Only a
        # written claim can be contradicted.
        if not f.get("present") or not claims_empty(f.get("text")):
            continue
        f["contested"] = True
        f["detector"] = account
        f["why"] = (f"the camera read the road as clear; the detector is "
                    f"tracking {account}")
        contested.append(f["name"])
    return {
        "fields": out,
        "contested": contested,
        "detector": {
            "account": account,
            "classes": dict(census["classes"]),
            "n_tracked": census["n_tracked"],
            "vulnerable": list(census.get("vulnerable") or []),
            "age_s": census.get("age_s"),
        } if contested else None,
    }


# WHAT RIO IS TOLD, when they disagree. One sentence, attached to the reading's
# own rules, and it says which instrument to believe rather than leaving her to
# weigh two sources she has no way of weighing.
def rule_for(result: dict) -> str:
    """-> the extra line for look()'s `rules`, or "" when nothing is contested."""
    if not result or not result.get("contested"):
        return ""
    det = result["detector"] or {}
    account = det.get("account") or "road users"
    return (
        " THE TWO INSTRUMENTS DISAGREE, AND THE TRACKER IS RIGHT ABOUT WHAT IS "
        f"THERE. The reading above says the road is clear. The detector is "
        f"holding {account} on it, measured and tracked across frames. Do not "
        f"repeat the reading's claim that there is no traffic: there is. Say "
        f"what is actually there if the driver asked about traffic, and "
        f"otherwise answer their question without mentioning the contradiction."
    )
