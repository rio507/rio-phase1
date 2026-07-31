"""Reference resolution — "the silver one on the left" -> a track id.

Design ref: docs/visual_qa.md §7.

This is the step that makes a conversation about the road possible at all. The
driver does not say "candidate 27"; they say "that car on the left", and every
part of that phrase is evidence about which of the tracked objects they mean:
a side, a class, sometimes a colour, sometimes only a demonstrative and the
assumption that there is one obvious answer.

THE ORDER MATTERS
-----------------
Geometry first, model second, and never the other way round.

Side and class come straight out of the scene graph, which got them from the
lane corridor and the detector -- measurements, not opinions. When those alone
leave one candidate clearly ahead, the answer is already correct and no model
is called: the fast path is also the trustworthy one.

Only when two candidates are both plausible does the question become "which of
these two does it LOOK like", and that is the one part a VLM is genuinely
better at than arithmetic. Qwen then gets the frame with the candidates marked
on it and picks a number (enrich.QwenAdapter.resolve).

AMBIGUITY IN PHASE A
--------------------
When nothing separates the candidates, this returns the best one WITH
`ambiguous` set and the alternatives listed. It does not invent certainty. In
Phase A the orchestrator passes that uncertainty down to the answer, so RIO
hedges rather than asserting; asking a clarifying question back is Phase B,
and the flag this produces is what that will hang off.
"""
import re
import time
from dataclasses import dataclass, field
from typing import Optional

import config
import enrich as enrich_mod
import scene as scene_mod

# How far ahead the top candidate must be before the answer is considered
# settled without asking a model. Expressed as a fraction of the top score, so
# it scales with how much evidence the phrase actually carried.
DECISIVE_MARGIN = 0.25

# ...and an absolute floor underneath it, which is what makes a cue-less
# question honest. "What is that?" carries no side, no class and no colour, so
# every candidate scores on the weak priors alone and the relative test above
# passes on a difference of 0.07 between two cars that are equally plausible to
# anyone looking out of the window. Measured on the test clip at t=42.5 s (a
# black saloon beside us, a white one a lane further over): margin 0.000.
#
# Below this floor the question genuinely has more than one answer, and the
# right move is to ask -- which is what Phase B added.
DECISIVE_MIN_MARGIN = 0.15
# ...and this is the margin below which even the model's pick is reported as
# ambiguous rather than certain.
AMBIGUOUS_MARGIN = 0.12

# How many candidates are worth putting in front of the model. More than this on
# screen at once and the numbers stop being legible on the annotated frame.
MAX_CANDIDATES = 5

# --- cue vocabularies -------------------------------------------------------
_SIDE_CUES = [
    (r"\b(on|to|from)?\s*(the\s+)?left\b|\bleft(-|\s)hand\b|\bnearside\b", "left"),
    (r"\b(on|to|from)?\s*(the\s+)?right\b|\bright(-|\s)hand\b|\boffside\b", "right"),
    (r"\bahead\b|\bin front\b|\bup front\b|\bfront of us\b|\bwe'?re following\b", "ahead"),
]

_CLASS_CUES = [
    (r"\bpickup|\bute\b|\bflatbed\b", {"labels": {"truck", "car"},
                                       "fine": {"pickup"}}),
    (r"\bsemi\b|\blorry\b|\barticulated\b|\bbig rig\b|\bbox truck\b",
     {"labels": {"truck"}, "fine": {"semi_truck", "box_truck"}}),
    (r"\btrucks?\b", {"labels": {"truck"}, "fine": {"pickup", "box_truck",
                                                    "semi_truck"}}),
    (r"\bsuvs?\b|\bjeep\b|\bcrossover\b|\b4x4\b", {"labels": {"car", "truck"},
                                                   "fine": {"suv", "crossover"}}),
    (r"\bminivans?\b|\bvans?\b", {"labels": {"car", "truck", "bus"},
                                  "fine": {"van", "minivan"}}),
    (r"\bbus(es)?\b|\bcoach\b", {"labels": {"bus"}, "fine": {"bus"}}),
    (r"\bmotorbike\b|\bmotorcycles?\b|\bbiker?\b", {"labels": {"motorcycle"},
                                                    "fine": {"motorcycle"}}),
    (r"\bcyclists?\b|\bbicycles?\b|\bpush ?bike\b|\brider\b",
     {"labels": {"cyclist"}, "fine": {"bicycle"}}),
    (r"\bpedestrians?\b|\bperson\b|\bguy\b|\bwoman\b|\bman\b|\bwalker\b",
     {"labels": {"pedestrian"}, "fine": set()}),
    (r"\bcoupe\b|\bsports ?car\b", {"labels": {"car"}, "fine": {"coupe"}}),
    (r"\bsedans?\b|\bsaloons?\b", {"labels": {"car"}, "fine": {"sedan"}}),
    (r"\bwagons?\b|\bestates?\b", {"labels": {"car"}, "fine": {"wagon"}}),
    (r"\bhatchbacks?\b", {"labels": {"car"}, "fine": {"hatchback"}}),
    # Generic, and last: "car" should not beat "pickup" when both appear.
    (r"\bcars?\b|\bvehicles?\b", {"labels": {"car", "truck", "bus", "van",
                                             "motorcycle"}, "fine": set()}),
]

_COLOUR_SYNONYMS = {
    "grey": "grey", "gray": "grey", "silver": "silver", "white": "white",
    "black": "black", "red": "red", "blue": "blue", "navy": "blue",
    "green": "green", "yellow": "yellow", "orange": "orange", "brown": "brown",
    "beige": "beige", "tan": "beige", "cream": "beige", "gold": "gold",
    "purple": "purple", "maroon": "maroon", "bronze": "bronze",
}
# Colours a human eye reads as each other. A "silver" car called "grey" or
# "white" is the single most common near-miss in this vocabulary, and treating
# them as a hard mismatch would reject the right vehicle.
_COLOUR_NEIGHBOURS = {
    "silver": {"grey", "white"}, "grey": {"silver", "white"},
    "white": {"silver", "grey"}, "beige": {"brown", "gold"},
    "gold": {"beige", "bronze", "yellow"}, "bronze": {"brown", "gold"},
    "maroon": {"red", "brown"}, "brown": {"beige", "bronze", "maroon"},
}

_NEAREST_CUES = r"\bclosest\b|\bnearest\b|\bright (in front|there)\b|\bjust ahead\b"
_FAR_CUES = r"\bway (up )?ahead\b|\bin the distance\b|\bfar (ahead|off)\b|\bfurther\b"


@dataclass
class Resolution:
    track_id: Optional[str] = None
    candidate_id: Optional[int] = None
    method: str = "none"
    confidence: float = 0.0
    ambiguous: bool = False
    alternatives: list = field(default_factory=list)
    cues: dict = field(default_factory=dict)
    scores: list = field(default_factory=list)
    info: dict = field(default_factory=dict)
    latency_ms: float = 0.0

    def to_log(self) -> dict:
        return {
            "track_id": self.track_id,
            "method": self.method,
            "confidence": round(self.confidence, 3),
            "ambiguous": self.ambiguous,
            "alternatives": self.alternatives,
            "cues": self.cues,
            "scores": self.scores[:6],
            "latency_ms": round(self.latency_ms, 1),
            "info": self.info,
        }


def parse_cues(phrase: str, question: str = "") -> dict:
    """Everything the driver's words say about which object they mean."""
    text = f"{phrase or ''} {question or ''}".lower()
    cues = {"side": None, "labels": None, "fine": None, "colour": None,
            "nearest": False, "far": False}

    for pattern, side in _SIDE_CUES:
        if re.search(pattern, text):
            cues["side"] = side
            break
    for pattern, spec in _CLASS_CUES:
        if re.search(pattern, text):
            cues["labels"] = set(spec["labels"])
            cues["fine"] = set(spec["fine"])
            break
    for word, canonical in _COLOUR_SYNONYMS.items():
        if re.search(r"\b%s\b" % word, text):
            cues["colour"] = canonical
            break
    cues["nearest"] = bool(re.search(_NEAREST_CUES, text))
    cues["far"] = bool(re.search(_FAR_CUES, text))
    return cues


def _colour_match(want: str, got: str) -> float:
    if not want or not got:
        return 0.0
    if want == got:
        return 1.0
    if got in _COLOUR_NEIGHBOURS.get(want, set()):
        return 0.6
    return -1.0        # a stated colour that clearly does not match is evidence AGAINST


def score_object(obj, cues: dict, n_objects: int) -> tuple:
    """-> (score, why). Higher is a better match for what the driver said."""
    score = 0.0
    why = []

    if cues.get("side"):
        side = scene_mod.side_of(obj.position)
        if side == cues["side"]:
            score += 1.0
            why.append("side")
        elif side == "unknown":
            score += 0.1
        else:
            # Wrong side is close to disqualifying: it is the cue a driver is
            # least likely to get wrong about their own field of view.
            score -= 1.2
            why.append("side_mismatch")

    if cues.get("labels"):
        if obj.label in cues["labels"]:
            score += 0.7
            why.append("class")
        else:
            score -= 0.9
            why.append("class_mismatch")
        if cues.get("fine") and obj.fine_label:
            if obj.fine_label in cues["fine"]:
                score += 0.6
                why.append("fine_class")
            else:
                score -= 0.3

    if cues.get("colour"):
        got = (obj.attributes or {}).get("color")
        m = _colour_match(cues["colour"], got)
        if m > 0:
            score += 1.0 * m
            why.append("colour")
        elif m < 0:
            score -= 0.8
            why.append("colour_mismatch")
        # got is None -> no evidence either way; enrichment has not run for it.

    if cues.get("nearest") and obj.depth_m is not None:
        score += 0.5
        why.append("nearest_cue")

    # Weak priors, deliberately small: they break ties between candidates the
    # phrase does not separate, and must never overturn a stated cue.
    if obj.depth_m is not None:
        score += 0.25 * max(0.0, 1.0 - obj.depth_m / 60.0)
    if obj.is_lead:
        score += 0.15
    score += 0.1 * min(1.0, obj.confidence)
    if n_objects == 1:
        # Only one thing out there; a bare "that" can only mean it.
        score += 0.5
        why.append("only_candidate")

    return score, why


def _describe(obj) -> str:
    """One line per candidate for the model's legend."""
    bits = [obj.label.replace("_", " ")]
    if obj.fine_label:
        bits.append(f"({obj.fine_label.replace('_', ' ')})")
    colour = (obj.attributes or {}).get("color")
    if colour:
        bits.insert(0, colour)
    where = obj.position.replace("_", " ")
    dist = "" if obj.depth_m is None else f", about {obj.depth_m:.0f} m away"
    return f"{' '.join(bits)} — {where}{dist}"


def resolve(question: str, phrase: Optional[str], graph, frame,
            cache: enrich_mod.EnrichmentCache, adapter=None,
            allow_model: bool = True) -> Resolution:
    """Which tracked object is the driver talking about?

    `graph` is the scene graph for `frame`. The graph may be rebuilt inside this
    function after enrichment, so the caller should use `res.track_id` rather
    than holding on to an object from the graph it passed in.
    """
    t0 = time.perf_counter()
    cues = parse_cues(phrase, question)
    objs = list(graph.objects)
    if not objs:
        return Resolution(method="no_objects", cues=_cues_log(cues),
                          latency_ms=(time.perf_counter() - t0) * 1000)

    scored = sorted(((score_object(o, cues, len(objs)), o) for o in objs),
                    key=lambda s: -s[0][0])

    # A colour was asked for and nothing has been enriched yet: pay for the
    # attribute on the few candidates that are otherwise plausible, then score
    # again. This is the one place enrichment is worth its latency, because
    # colour is often the ONLY thing separating two cars in the same lane.
    enriched = {}
    if cues.get("colour") and frame is not None:
        want = [o.track_id for (_, o) in scored[:config.ENRICH_MAX_OBJECTS]
                if not (o.attributes or {}).get("color")]
        if want:
            enriched = enrich_mod.enrich_objects(frame, want, cache, adapter)
            if enriched:
                for (_, o) in scored:
                    e = enriched.get(o.track_id)
                    if e:
                        o.attributes.update(
                            {k: v for k, v in (e.get("attributes") or {}).items() if v})
                        o.fine_label = o.fine_label or e.get("fine_label")
                scored = sorted(((score_object(o, cues, len(objs)), o) for o in objs),
                                key=lambda s: -s[0][0])

    (top_score, top_why), top = scored[0]
    second = scored[1][0][0] if len(scored) > 1 else None
    margin = None if second is None else (top_score - second)

    log_scores = [{"track_id": o.track_id, "score": round(s, 3), "why": w}
                  for (s, w), o in scored[:6]]

    decisive = (second is None
                or (top_score > 0
                    and margin >= DECISIVE_MARGIN * abs(top_score)
                    and margin >= DECISIVE_MIN_MARGIN))
    if decisive and top_score > 0:
        return Resolution(
            track_id=top.track_id, candidate_id=top.candidate_id,
            method="geometry", confidence=_confidence(top_score, margin),
            ambiguous=False,
            alternatives=[o.track_id for (_, o) in scored[1:4]],
            cues=_cues_log(cues), scores=log_scores,
            info={"margin": None if margin is None else round(margin, 3),
                  "enriched": list(enriched.keys())},
            latency_ms=(time.perf_counter() - t0) * 1000)

    # Two or more are plausible. A VLM can settle that — but ONLY when the
    # driver's words actually contain something to look for.
    #
    # This gate was added after watching it fail. Asked a bare "What is that?"
    # with a black saloon beside us and a white one a lane over, Qwen picked a
    # third vehicle entirely, on the far side of the road, and reported itself
    # sure. It was not being stupid: the question contains no side, no class and
    # no colour, so there is nothing in the image that could distinguish the
    # right answer from the wrong one. The ambiguity is in the LANGUAGE, and no
    # amount of looking at the road resolves it.
    #
    # So a cue-less reference goes straight to ambiguous, which is what makes
    # RIO ask instead of guess.
    discriminating = bool(cues.get("side") or cues.get("labels")
                          or cues.get("colour") or cues.get("nearest")
                          or cues.get("far"))
    plausible = [(s, o) for (s, _why), o in scored if s > -0.5][:MAX_CANDIDATES]
    if allow_model and discriminating and frame is not None and len(plausible) > 1:
        adapter = adapter or enrich_mod.get_adapter()
        cands = [(o.candidate_id, o.box, _describe(o)) for _, o in plausible]
        try:
            cid, info = adapter.resolve(frame.jpeg, question or phrase or "", cands)
        except Exception as e:
            cid, info = None, {"error": f"{type(e).__name__}: {e}"}
        if cid is not None:
            picked = next((o for _, o in plausible if o.candidate_id == cid), None)
            if picked is not None:
                sure = bool(info.get("sure"))
                return Resolution(
                    track_id=picked.track_id, candidate_id=picked.candidate_id,
                    method="vlm", confidence=0.75 if sure else 0.55,
                    ambiguous=not sure,
                    alternatives=[o.track_id for _, o in plausible
                                  if o.candidate_id != cid],
                    cues=_cues_log(cues), scores=log_scores,
                    info={"vlm": {k: v for k, v in info.items() if k != "legend"},
                          "enriched": list(enriched.keys())},
                    latency_ms=(time.perf_counter() - t0) * 1000)

    # Nothing separated them. Say so; do not manufacture certainty.
    return Resolution(
        track_id=top.track_id if top_score > 0 else None,
        candidate_id=top.candidate_id if top_score > 0 else None,
        method="ambiguous", confidence=_confidence(top_score, margin),
        ambiguous=True,
        alternatives=[o.track_id for (_, o) in scored[1:4]],
        cues=_cues_log(cues), scores=log_scores,
        info={"margin": None if margin is None else round(margin, 3),
              "enriched": list(enriched.keys()),
              # Why it stayed ambiguous: nothing in the phrase pointed at
              # anything, so the model was never consulted.
              "no_discriminating_cue": not discriminating},
        latency_ms=(time.perf_counter() - t0) * 1000)


def describe(obj) -> str:
    """A driver-readable description of one candidate. Public: the clarifying
    question is built from these, and the log records them so a bad question can
    be read back rather than re-enacted."""
    return _describe(obj)


def resolve_among(text: str, graph, allowed_track_ids, frame=None,
                  cache: enrich_mod.EnrichmentCache = None,
                  adapter=None) -> Resolution:
    """Which of the objects RIO just offered did the driver pick?

    A much easier problem than open resolution, and a different one: the set is
    known, small, and was described out loud a moment ago, so the driver's words
    are chosen to distinguish THESE candidates from each other. "The black one"
    is not a description of a car, it is a contrast with the white one.

    Scoring is therefore the same machinery restricted to the offered set, with
    one addition: ordinal answers ("the first one", "the second") are positional
    references to the order the question used, which no amount of looking at the
    road can resolve.
    """
    t0 = time.perf_counter()
    allowed = list(allowed_track_ids or [])
    objs = [o for o in graph.objects if o.track_id in allowed] if graph else []
    cues = parse_cues(text, "")

    # Ordinals refer to the ORDER RIO offered them in, not to anything visible.
    ordinal = _ordinal_index(text)
    if ordinal is not None and 0 <= ordinal < len(allowed):
        tid = allowed[ordinal]
        return Resolution(
            track_id=tid, candidate_id=scene_mod.candidate_id_of(tid),
            method="ordinal", confidence=0.8, ambiguous=False,
            alternatives=[t for t in allowed if t != tid],
            cues=_cues_log(cues),
            info={"ordinal": ordinal, "offered": allowed},
            latency_ms=(time.perf_counter() - t0) * 1000)

    if not objs:
        # Everything offered has since aged out of the graph. The driver's
        # answer is still meaningful -- they picked one of the things RIO
        # described -- but there is no live object to attach it to.
        return Resolution(method="offered_gone", cues=_cues_log(cues),
                          alternatives=allowed,
                          info={"offered": allowed},
                          latency_ms=(time.perf_counter() - t0) * 1000)

    if cues.get("colour"):
        want = [o.track_id for o in objs if not (o.attributes or {}).get("color")]
        if want and frame is not None and cache is not None:
            got = enrich_mod.enrich_objects(frame, want, cache, adapter,
                                            max_objects=len(want))
            for o in objs:
                e = got.get(o.track_id)
                if e:
                    o.attributes.update(
                        {k: v for k, v in (e.get("attributes") or {}).items() if v})
                    o.fine_label = o.fine_label or e.get("fine_label")

    scored = sorted(((score_object(o, cues, len(objs)), o) for o in objs),
                    key=lambda s: -s[0][0])
    (top_score, _why), top = scored[0]
    second = scored[1][0][0] if len(scored) > 1 else None
    margin = None if second is None else (top_score - second)
    log_scores = [{"track_id": o.track_id, "score": round(s, 3), "why": w}
                  for (s, w), o in scored]

    # A deliberately low bar. The driver has just been asked a direct question
    # and has answered it; the alternative to accepting a weak signal is asking
    # again, and asking twice is worse than being corrected once.
    resolved = top_score > 0 and (margin is None or margin > 0.01)
    return Resolution(
        track_id=top.track_id if resolved else None,
        candidate_id=top.candidate_id if resolved else None,
        method="clarified" if resolved else "clarify_unresolved",
        confidence=0.8 if resolved else 0.0,
        ambiguous=not resolved,
        alternatives=[o.track_id for (_s, _w), o in scored[1:]],
        cues=_cues_log(cues), scores=log_scores,
        info={"offered": allowed,
              "margin": None if margin is None else round(margin, 3)},
        latency_ms=(time.perf_counter() - t0) * 1000)


def _ordinal_index(text: str):
    """"the second one" -> 1. None for anything that is not an ordinal.

    Deliberately narrow: "the first one" and "the black one" both end in "one",
    and only the first of them is positional.
    """
    t = (text or "").lower()
    for word, idx in (("first", 0), ("second", 1), ("third", 2), ("last", -1)):
        if re.search(r"\b%s\b" % word, t):
            return idx
    return None


_NOUNS_RE = (r"car|cars|vehicle|truck|lorry|pickup|suv|van|bus|coupe|sedan|"
             r"saloon|wagon|hatchback|jeep|bike|motorbike|motorcycle|cyclist|"
             r"bicycle|rider|pedestrian|person")

# Connectors that separate the two halves of a comparison.
_COMPARE_SPLIT = r"\s+(?:or|versus|vs\.?|against|compared to|next to)\s+"


def split_comparison(question: str):
    """"the truck or the white car" -> ["the truck", "the white car"].

    Returns None when the utterance does not actually name two things, which is
    most of the time: "which one is faster" is a comparison in intent and names
    nothing, and guessing a second object out of it would be inventing half the
    question.
    """
    text = (question or "").strip()
    # "compare X to Y" gets its own pattern rather than putting bare " to " in
    # the connector list, where it would split "the car next to us" in half.
    m = re.match(r"\s*compare\s+(.+?)\s+(?:to|with|against)\s+(.+)$", text, re.I)
    if m:
        parts = [m.group(1).strip(" ?.,"), m.group(2).strip(" ?.,")]
        if all(re.search(r"\b(%s)\b|\bone\b" % _NOUNS_RE, p, re.I) for p in parts):
            return parts

    # Work from the part after a leading interrogative, so "which is bigger,
    # the truck or the car" splits on the right "or".
    m = re.search(r"[,:]\s*(.+)$", text)
    tail = m.group(1) if m else text
    parts = [p.strip(" ?.,") for p in re.split(_COMPARE_SPLIT, tail, flags=re.I)]
    parts = [p for p in parts if p and re.search(r"\b(%s)\b|\bone\b" % _NOUNS_RE, p, re.I)]
    return parts[:2] if len(parts) >= 2 else None


def _confidence(top_score: float, margin) -> float:
    if top_score <= 0:
        return 0.0
    base = min(1.0, top_score / 2.5)
    if margin is None:
        return min(1.0, base + 0.2)
    if margin < AMBIGUOUS_MARGIN:
        return min(base, 0.4)
    return min(1.0, base * (0.6 + 0.4 * min(1.0, margin)))


def _cues_log(cues: dict) -> dict:
    return {k: (sorted(v) if isinstance(v, set) else v) for k, v in cues.items()}
