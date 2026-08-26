"""Anchor validation and selection — the part that says no (§18-§20).

This is where most candidate landmarks die, and that is the design working.
Frequent rejection is correct behaviour: the cost of rejecting a good anchor is
that RIO says "Take the next right", which is a perfectly good instruction; the
cost of accepting a bad one is that RIO tells a driver to turn at a building
that is not there, or at the second of two identical signs. Those are not
comparable costs, and the thresholds are set accordingly.

HARD GATES FIRST, RANKING SECOND
--------------------------------
Every gate is a boolean and any failure is a rejection. There is deliberately
no multiplicative confidence score that lets a strong identity confidence
compensate for a stale observation, or a big bright sign compensate for there
being two of them — those are different kinds of wrong and they do not trade.

Only candidates that passed everything are ranked, and the ranking is a plain
ordering: salience, then spoken usefulness, then closeness of the relation, then
stability, then identity confidence. One anchor per maneuver, maximum.

WHAT PERCEPTION CONTRIBUTED, AND WHAT IT DID NOT
------------------------------------------------
Perception supplies: is it visible, how sure of the identity, how many
instances, how long it has been held, how fresh the last look was, and a depth
plausibility check. It does NOT supply the relation — that came from the map
(landmarks.py) — and it does not supply the maneuver, the direction, the
timing, or a vote on any of them.
"""
from dataclasses import dataclass
from typing import List, Optional, Tuple

import config

from .landmarks import JUST_AFTER, JUST_BEFORE, NEAR

ORDERED = (JUST_BEFORE, JUST_AFTER)


@dataclass
class VerifiedAnchor:
    """Everything navigation is allowed to know about a landmark (§17).

    Track ids, depth histories, bounding boxes, temporal windows, model
    confidences, image coordinates and scale rates stay inside the perception
    subsystem. What crosses the boundary is a label, a class, a relation, three
    confidences and an expiry — which is exactly what is needed to choose a
    sentence and to know when that sentence stops being true.
    """
    anchor_id: str
    label: str
    type: str
    turn_relation_to_anchor: str
    identity_confidence: float
    relation_confidence: float
    visibility_confidence: float
    valid_for_s: float
    valid_until: float

    def to_dict(self) -> dict:
        return {
            "anchor_id": self.anchor_id,
            "label": self.label,
            "type": self.type,
            "turn_relation_to_anchor": self.turn_relation_to_anchor,
            "identity_confidence": round(self.identity_confidence, 3),
            "relation_confidence": round(self.relation_confidence, 3),
            "visibility_confidence": round(self.visibility_confidence, 3),
            "valid_for_s": round(self.valid_for_s, 2),
            "valid_until": round(self.valid_until, 3),
        }


def resolve_relation(candidate: dict) -> Optional[Tuple[str, float, bool]]:
    """The relation this candidate may actually be spoken with.

    JUST_BEFORE and JUST_AFTER are claims about ORDER. Getting one backwards
    sends a driver through the junction, so they need substantially more
    confidence than NEAR — and when they do not have it they DEGRADE to NEAR
    rather than being dropped, because "by the Shell station" is still true and
    still useful. If NEAR cannot be supported either, there is no anchor and
    the canonical instruction is used (§16).

    Returns (relation, confidence, degraded) or None.
    """
    relation = candidate.get("relation")
    conf = float(candidate.get("relation_confidence") or 0.0)
    if relation in ORDERED:
        if conf >= config.NAV_ANCHOR_ORDERED_MIN_RELATION_CONFIDENCE:
            return relation, conf, False
        # Degrade. The landmark is still where it is; only the claim about
        # order is being withdrawn.
        if conf >= config.NAV_ANCHOR_MIN_RELATION_CONFIDENCE:
            return NEAR, conf, True
        return None
    if relation == NEAR and conf >= config.NAV_ANCHOR_MIN_RELATION_CONFIDENCE:
        return NEAR, conf, False
    return None


def validate(candidate: dict, obs: dict, now: float) -> Tuple[bool, List[str]]:
    """Every gate, in order, with every failure named.

    `obs` is the perception subsystem's report about this candidate:
        visible, identity_confidence, visibility_confidence, instances,
        observations, tracking_duration_s, last_seen_t, depth_m (optional),
        side (optional)

    Returns (ok, rejections). The rejection list is what goes in the log — a
    drive where anchors are rejected without a stated reason is a drive nobody
    can tune.
    """
    fails: List[str] = []

    if candidate.get("type") not in config.NAV_ANCHOR_TYPES:
        fails.append("allowed_anchor_type")

    if not obs.get("visible"):
        fails.append("not_visible")

    ident = float(obs.get("identity_confidence") or 0.0)
    if ident < config.NAV_ANCHOR_MIN_IDENTITY_CONFIDENCE:
        fails.append("identity_confidence")

    vis = float(obs.get("visibility_confidence") or 0.0)
    if vis < config.NAV_ANCHOR_MIN_VISIBILITY_CONFIDENCE:
        fails.append("visibility_confidence")

    # Persistence. A landmark seen in one frame is a detection; a landmark held
    # across several is an observation. Single-frame anchors are exactly the
    # ones that turn out to be a billboard, a reflection or a passing truck.
    if int(obs.get("observations") or 0) < config.NAV_ANCHOR_MIN_OBSERVATIONS:
        fails.append("tracking_observations")
    if float(obs.get("tracking_duration_s") or 0.0) < config.NAV_ANCHOR_MIN_TRACKING_DURATION_S:
        fails.append("tracking_duration")

    last_seen = obs.get("last_seen_t")
    if last_seen is None or (now - float(last_seen)) > config.NAV_ANCHOR_MAX_AGE_S:
        fails.append("observation_stale")

    # Scene uniqueness. Two Shell stations that could both plausibly be the one
    # meant is not a disambiguation problem in v1 — it is a rejection. "Take
    # the next left" is unambiguous and costs nothing (§20).
    if int(obs.get("instances") or 1) > 1:
        fails.append("scene_uniqueness")

    if not spatial_consistent(candidate, obs):
        fails.append("spatial_consistency")

    if resolve_relation(candidate) is None:
        fails.append("relation_confidence")

    return (not fails), fails


def spatial_consistent(candidate: dict, obs: dict) -> bool:
    """Does what the camera reports sit sensibly with what the map says (§15)?

    Depth is a CONSISTENCY signal and never a measurement RIO acts on. It is
    not asked where the intersection is — route geometry already knows that.
    It is asked whether a thing reported as the Shell 40 m from the maneuver is
    plausibly 40-ish metres away and in front of the car, rather than 200 m off
    or behind a wall. Missing depth is not a failure: most of the value is in
    catching the gross contradictions, and there are frames where depth is
    simply not available.
    """
    if config.NAV_VERIFY_DEPTH_ENABLED:
        depth = obs.get("depth_m")
        if depth is not None:
            d = float(depth)
            if d < config.NAV_VERIFY_DEPTH_MIN_M or d > config.NAV_VERIFY_DEPTH_MAX_M:
                return False
            # The map says how far the landmark is from the maneuver, and the
            # car is somewhere short of the maneuver; a depth wildly beyond the
            # distance to the maneuver plus that offset is a different object.
            expect_max = float(candidate.get("distance_to_maneuver_m") or 0.0) + \
                config.NAV_VERIFY_DEPTH_MAX_M
            if d > expect_max:
                return False

    side = obs.get("side")
    expected = candidate.get("side")
    if side and expected and side in ("LEFT", "RIGHT") and expected in ("LEFT", "RIGHT"):
        if side != expected:
            return False
    return True


def select(validated: List[dict]) -> Optional[dict]:
    """The one anchor, or none (§19).

    A plain lexicographic ordering rather than a weighted score, on purpose:
    with a score, a change to one term silently re-ranks everything, and there
    is no way to answer "why did it pick that one" in a log. This ordering can
    be read straight out of a rejection list.
    """
    if not validated:
        return None
    def key(entry):
        c, o = entry["candidate"], entry["observation"]
        return (
            -float(c.get("salience") or 0.0),
            -float(config.NAV_ANCHOR_TYPE_USEFULNESS.get(c.get("type"), 0.5)),
            float(c.get("distance_to_maneuver_m") or 1e9),
            -float(o.get("tracking_duration_s") or 0.0),
            -float(o.get("identity_confidence") or 0.0),
        )
    return sorted(validated, key=key)[0]


def build(candidate: dict, obs: dict, now: float) -> Optional[VerifiedAnchor]:
    """A validated candidate plus its observation -> the anchor navigation sees."""
    rel = resolve_relation(candidate)
    if rel is None:
        return None
    relation, relation_conf, _ = rel
    valid_for = float(config.NAV_ANCHOR_VALID_FOR_S)
    return VerifiedAnchor(
        anchor_id=candidate.get("anchor_id", ""),
        label=candidate.get("label", ""),
        type=candidate.get("type", ""),
        turn_relation_to_anchor=relation,
        identity_confidence=float(obs.get("identity_confidence") or 0.0),
        relation_confidence=relation_conf,
        visibility_confidence=float(obs.get("visibility_confidence") or 0.0),
        valid_for_s=valid_for,
        valid_until=now + valid_for,
    )
