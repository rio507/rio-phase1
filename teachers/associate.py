"""The model said "the white pickup ahead". Which box is that?

WHY THIS IS A MODULE AND NOT A PROMPT
-------------------------------------
Both teachers are asked which single road user matters most right now, and both
answer in English. The deterministic pipeline, at the same instant, has a set
of tracked boxes with classes, ranges and lane membership. Those two things are
only comparable if the sentence can be pinned to a box -- and once it is, three
questions that were unanswerable become arithmetic:

  * do the two models agree about which road user matters? (same track id)
  * does either agree with RIO's lead lock? (track id == lead_id)
  * how often can a named actor be found in the detector's output at all?
    (the matched-track rate on the tally strip -- an unmatched rate that
    climbs is either a model describing things RF-DETR cannot see, or a
    detector missing what matters, and the corpus says which.)

WHY IT IS DETERMINISTIC, AND WHY THAT MATTERS MORE THAN BEING CLEVER
--------------------------------------------------------------------
The obvious implementation is to ask a language model which box it meant. That
would make the association as opaque as the thing it is trying to check, and it
would not be reproducible: re-running the corpus a year from now would produce
different associations from the same rows. So this is word lists and geometry.
It is dumber, it abstains more often, and every match it makes can be re-derived
from the row by anyone reading it.

ABSTAINING IS A RESULT. `matched: false` with a reason is the honest answer for
"a pedestrian on the far pavement" when the detector found no pedestrian, and
the tally strip counts it. A forced match to the nearest car would be a silent
lie in a training corpus, which is the worst place to put one.
"""
import re

# --- what the models call things, mapped to what the detector calls them ----
# Ordered longest-phrase-first within each class so "pickup truck" wins over
# "truck" and "fire truck" does not become a bus. The detector only has six
# classes (headway/detect.COCO_TO_LABEL), so the job is to funnel a large
# vocabulary into a small one, and to be honest when a word has no funnel --
# "traffic light", "cone" and "barrier" are real things the models name and
# RF-DETR does not detect, so they map to nothing and the match abstains.
CLASS_WORDS = {
    "truck": ("semi truck", "semi-truck", "pickup truck", "box truck",
              "dump truck", "delivery truck", "garbage truck", "tanker",
              "lorry", "pickup", "truck", "hgv", "tractor trailer",
              "articulated"),
    "bus": ("school bus", "coach", "bus"),
    "car": ("sedan", "hatchback", "estate", "station wagon", "suv", "crossover",
            "minivan", "van", "taxi", "cab", "police car", "car", "vehicle",
            "automobile"),
    "motorcycle": ("motorcycle", "motorbike", "motorcyclist", "scooter",
                   "moped", "rider"),
    "cyclist": ("cyclist", "bicyclist", "bicycle", "bike", "e-bike"),
    "pedestrian": ("pedestrian", "person", "people", "walker", "jogger",
                   "child", "man", "woman", "someone"),
}

# Words that name something the detector has no class for. Recognised on
# purpose: "no such class" is a better refusal than "no match found", because
# it tells a reader the association failed for a reason nothing about this
# drive can fix.
UNDETECTABLE_WORDS = (
    "traffic light", "traffic signal", "stop sign", "stop light", "sign",
    "cone", "barrier", "barricade", "pole", "kerb", "curb", "median",
    "crosswalk", "crossing", "junction", "intersection", "roundabout",
    "lane marking", "road marking", "pothole", "debris", "animal", "dog",
    "deer", "gate", "boom", "tram", "train", "traffic",
)

SIDE_WORDS = {
    "left": ("on the left", "to the left", "left side", "left-hand", "leftmost",
             "nearside" if False else "left lane", "left"),
    "right": ("on the right", "to the right", "right side", "right-hand",
              "rightmost", "right lane", "right"),
    "ahead": ("directly ahead", "straight ahead", "in front of the ego",
              "in front", "ahead", "leading", "lead vehicle", "in our lane",
              "same lane", "ego lane", "preceding"),
}

# Words that say the actor is not going the same way. Used only to explain a
# refusal: nothing in the deterministic pipeline knows an oncoming car from a
# leading one, so an actor described as oncoming cannot be matched with any
# confidence and says so rather than grabbing the nearest box.
OPPOSING_WORDS = ("oncoming", "opposing", "approaching from the opposite",
                  "coming towards", "coming toward", "head-on", "head on")


def _first_sentence(text: str) -> str:
    """The clause the actor is named in — the rest is the "why".

    Both prompts ask for a road user AND a reason, and the reason routinely
    names other road users ("...because the cyclist behind it may pull out").
    Matching on the whole answer would let the reason outvote the answer, which
    is exactly backwards.
    """
    t = (text or "").strip()
    if not t:
        return ""
    # Cut at the first reason marker or sentence end, whichever comes first.
    cut = len(t)
    for marker in (" because ", " since ", " as it ", " which ", ";", " — ",
                   " - ", ". ", "\n"):
        i = t.lower().find(marker)
        if i > 0:
            cut = min(cut, i)
    return t[:cut].strip()


def parse_phrase(text: str) -> dict:
    """An English answer -> what it is asking for. Never raises.

    -> {"phrase", "wanted_class", "wanted_side", "undetectable", "opposing"}
    """
    phrase = _first_sentence(text)
    low = " " + re.sub(r"[^a-z0-9\- ]+", " ", phrase.lower()) + " "
    low = re.sub(r"\s+", " ", low)

    wanted_class, best_at = None, None
    for label, words in CLASS_WORDS.items():
        for w in words:
            i = low.find(" " + w + " ")
            if i < 0:
                i = low.find(" " + w)
            if i >= 0 and (best_at is None or i < best_at):
                # A tie on position goes to the LONGER word, which is why the
                # lists are ordered: "pickup truck" and "truck" both start at
                # the same index and only one of them is the answer.
                wanted_class, best_at = label, i
                break
    undetectable = None
    for w in UNDETECTABLE_WORDS:
        if (" " + w) in low:
            undetectable = w
            break

    wanted_side, side_at = None, None
    for side, words in SIDE_WORDS.items():
        for w in words:
            i = low.find(" " + w)
            if i >= 0 and (side_at is None or i < side_at):
                wanted_side, side_at = side, i
                break

    opposing = any((" " + w) in low for w in OPPOSING_WORDS)
    return {"phrase": phrase, "wanted_class": wanted_class,
            "wanted_side": wanted_side, "undetectable": undetectable,
            "opposing": opposing}


def _side_of(box, width):
    """Which third of the picture the box's centre sits in."""
    if not box or not width:
        return None
    cx = (float(box[0]) + float(box[2])) / 2.0
    if cx < width / 3.0:
        return "left"
    if cx > 2.0 * width / 3.0:
        return "right"
    return "ahead"


def score_track(track, want, width):
    """How well one RF-DETR track fits the parsed phrase. -> (score, why).

    The weights are stated rather than tuned, because a tuned weight in a
    corpus tool is a weight nobody can reproduce. Class is worth most: a model
    saying "cyclist" and a box labelled `car` is not the same object however
    well positioned. Side is worth about half of that. Everything else is a
    tie-break, and the tie-breaks are ordered so the result cannot depend on
    dict iteration.
    """
    why = []
    score = 0.0
    label = track.get("label")
    if want["wanted_class"]:
        if label == want["wanted_class"]:
            score += 0.55
            why.append("class")
        elif {label, want["wanted_class"]} <= {"car", "truck", "bus"}:
            # Large-vehicle confusion is the detector's most common one and the
            # models' too. Worth a partial credit, never a full one.
            score += 0.22
            why.append("class~")
        else:
            return 0.0, ["class-mismatch"]
    side = _side_of(track.get("box"), width)
    if want["wanted_side"]:
        if side == want["wanted_side"]:
            score += 0.28
            why.append("side")
        elif want["wanted_side"] == "ahead" and track.get("member"):
            # "ahead" and "in our lane" are the same claim, and lane membership
            # is a better answer to it than which third of the frame the box
            # falls in.
            score += 0.28
            why.append("in-corridor")
        else:
            score -= 0.10
            why.append("side-mismatch")
    elif track.get("member"):
        score += 0.06
        why.append("in-corridor")

    # Tie-breaks, in a fixed order. Nearer is more likely to be what a model
    # calls "the most safety-relevant", and so is bigger in frame.
    rng = track.get("range_m")
    if isinstance(rng, (int, float)) and rng > 0:
        score += max(0.0, 0.12 * (1.0 - min(1.0, float(rng) / 80.0)))
        why.append("range")
    box = track.get("box") or []
    if len(box) == 4 and width:
        frac = (float(box[2]) - float(box[0])) / float(width)
        score += max(0.0, min(0.08, frac * 0.16))
        why.append("size")
    if track.get("is_lead"):
        score += 0.05
        why.append("lead")
    return score, why


# Below this the best fit is not a fit. Set where a class match alone (0.55)
# still passes and a bare tie-break pile does not: the association must rest on
# something the model actually said, not on one box being closer than another.
MATCH_FLOOR = 0.45


def associate(text: str, tracks: list, image_w: float, image_h: float = None) -> dict:
    """A model's named actor -> the RF-DETR track it means. -> ASSOCIATION_FIELDS.

    `tracks` is the `scene_objects` list from the headway result at t0 -- the
    same list the overlay draws -- so a match is a track id the dashboard can
    highlight and the corpus can be re-checked against.
    """
    want = parse_phrase(text)
    out = {
        "matched": False, "track_id": None, "label": None, "box": None,
        "range_m": None, "score": 0.0, "reason": "",
        "phrase": want["phrase"], "wanted_class": want["wanted_class"],
        "wanted_side": want["wanted_side"],
    }
    if not want["phrase"]:
        out["reason"] = "no_answer"
        return out
    if want["opposing"]:
        # Honest refusal: nothing here knows which way a box is facing, so a
        # match would be a guess wearing a track id.
        out["reason"] = "opposing_traffic_not_tracked"
        return out
    if not want["wanted_class"]:
        out["reason"] = ("names_undetectable_class:" + want["undetectable"]
                         if want["undetectable"] else "no_road_user_named")
        return out
    live = [t for t in (tracks or []) if t and t.get("box")]
    if not live:
        out["reason"] = "no_tracks_at_t0"
        return out

    # Deterministic ordering before scoring, so two equal scores always resolve
    # the same way: by track id, which is assigned in first-seen order.
    live = sorted(live, key=lambda t: (int(t.get("id") or 0)))
    best, best_score, best_why = None, -1.0, []
    for t in live:
        s, why = score_track(t, want, image_w)
        if s > best_score:
            best, best_score, best_why = t, s, why

    if best is None or best_score < MATCH_FLOOR:
        out["score"] = round(max(0.0, best_score), 3)
        out["reason"] = ("no_track_of_class:" + want["wanted_class"]
                         if best_score <= 0 else "below_match_floor")
        return out

    out.update({
        "matched": True,
        "track_id": int(best.get("id")) if best.get("id") is not None else None,
        "label": best.get("label"),
        "box": [round(float(v), 1) for v in best.get("box")],
        "range_m": best.get("range_m"),
        "score": round(best_score, 3),
        "reason": "+".join(best_why),
    })
    return out


def agree(a: dict, b: dict) -> bool:
    """Did the two models pick the same road user? Only ever true on evidence.

    Both unmatched is NOT agreement -- it is two abstentions, and counting it
    as agreement would make the tally strip read best when the association is
    working worst.
    """
    return bool(a and b and a.get("matched") and b.get("matched")
                and a.get("track_id") is not None
                and a.get("track_id") == b.get("track_id"))
