"""Landmark candidates: what is near the turn RIO already knows about (§13).

THE INVERSION THAT MAKES THIS TRACTABLE
---------------------------------------
The naive version of contextual navigation points a camera at the world and
asks "what landmarks do you see?" That is open-world recognition, it is
unbounded, and every one of its failure modes ends with RIO confidently naming
something that is not there.

This asks the opposite question, and asks it of the map:

    The maneuver is at these coordinates. What useful, named, permanent things
    are within 90 m of it?

The answer is a short list — usually zero to three entries — fetched once,
before the drive. The camera's job then shrinks from recognition to
VERIFICATION: is *this specific expected thing* visible right now? That is a
question with a bounded answer, a defensible confidence, and a safe default.

RELATION IS MAP DATA, NOT PERCEPTION (§16 + addendum)
-----------------------------------------------------
`turn_relation_to_anchor` — whether the turn is NEAR the landmark, JUST_BEFORE
it or JUST_AFTER it — is computed here, from the landmark's coordinates against
the maneuver's coordinates and the route geometry. The camera never estimates
it. Vision answers "can the driver see the Shell", and the map answers "and the
turn is just past it"; neither is asked the other's question.

NEAR is the default because it needs the least spatial certainty. JUST_BEFORE
and JUST_AFTER are claims about ORDER, and a wrong one puts a driver through
the junction, so they need a wide, explicit margin before they are allowed —
and degrade to NEAR rather than being taken on trust.

LOOKUP BUDGET (addendum)
------------------------
One pass over the maneuvers at route load, capped, cached for the lifetime of
that route generation, refreshed only on reroute. Never per-frame, never on an
interval. A place lookup per frame is not a lookup, it is a subscription.
"""
import time
from typing import List, Optional

import config

from . import geo
from . import model as M
from . import speech as speech_mod

# Where the relation lands, said as the spec says it: relative to the LANDMARK.
NEAR = "NEAR"
JUST_BEFORE = "JUST_BEFORE"
JUST_AFTER = "JUST_AFTER"

# How precisely the provider's place coordinates can be trusted, as a plain
# multiplier on relation confidence. Places coordinates are a rooftop/centroid
# point for a business, not the position of its sign — good to a few metres and
# not better, and the relation must never claim more certainty than its inputs.
PROVIDER_COORD_PRECISION = 0.95


def brand_of(name: str):
    """Match a place name against the allowed brand table (§21).

    Substring on a normalised name, because providers return "Starbucks Coffee
    Company", "Shell Oil", "McDonald's #4412". Longest brand key first so
    "7-eleven" cannot be beaten by a shorter accidental match.

    Returns (brand_key, spoken_label, anchor_class, salience) or None. None is
    the common case and it means: not an allowed anchor, navigate normally.
    """
    n = " " + (name or "").lower().strip() + " "
    if len(n) < 3:
        return None
    for key in sorted(config.NAV_ANCHOR_BRANDS, key=len, reverse=True):
        spoken, cls, salience = config.NAV_ANCHOR_BRANDS[key]
        k = key.lower()
        # Word-boundary-ish: "76" must not match "Route 761 Deli", and "bp"
        # must not match "bpm Cafe".
        for sep_l in (" ", "-", "("):
            for sep_r in (" ", "-", ")", ",", "'", "."):
                if (sep_l + k + sep_r) in n:
                    return key, spoken, cls, salience
    return None


def _window(route: M.CanonicalRoute, maneuver: M.CanonicalManeuver, span_m: float = 200.0):
    """Vertex range around a maneuver, so a place cannot project onto a later leg.

    A route that doubles back on itself is common in a city, and an unbounded
    nearest-point search will happily decide the Shell 15 m before the turn is
    actually 900 m further along on the return leg — which flips the relation.
    """
    cum = geo.cumulative(route.geometry)
    lo = hi = maneuver.polyline_index
    while lo > 0 and cum[maneuver.polyline_index] - cum[lo] < span_m:
        lo -= 1
    while hi < len(route.geometry) - 2 and cum[hi] - cum[maneuver.polyline_index] < span_m:
        hi += 1
    return cum, lo, hi


def relation_for(route: M.CanonicalRoute, maneuver: M.CanonicalManeuver,
                 place: dict) -> Optional[dict]:
    """Where the TURN sits relative to this LANDMARK, from coordinates alone.

    Returns None when the landmark cannot describe this maneuver at all — too
    far from it, or too far off the road to be the thing the driver is looking
    at. Returning None is the expected outcome for most places near most turns.
    """
    cum, lo, hi = _window(route, maneuver)
    along, lateral, _ = geo.project(route.geometry, cum, place["lat"], place["lng"], lo, hi)
    straight = geo.haversine_m(maneuver.latitude, maneuver.longitude,
                               place["lat"], place["lng"])
    if straight > config.NAV_LANDMARK_MAX_DISTANCE_M:
        return None
    if lateral > config.NAV_RELATION_MAX_LATERAL_M:
        return None

    delta = along - maneuver.route_distance_position     # +ve: past the turn
    a = abs(delta)
    band = config.NAV_RELATION_NEAR_BAND_M
    ordered_min = config.NAV_RELATION_ORDERED_MIN_M
    ordered_max = config.NAV_RELATION_ORDERED_MAX_M

    if a <= band:
        relation = NEAR
        # Dead on the maneuver is the strongest NEAR there is; the edge of the
        # band is a weak one.
        conf = max(0.0, 1.0 - a / (band * 2.5))
    elif a <= ordered_max:
        relation = JUST_BEFORE if delta > 0 else JUST_AFTER
        # Ordered relations ramp from "barely outside NEAR" (untrustworthy) to
        # "unmistakably before/after" (trustworthy). The gate that consumes
        # this is deliberately high, so the ramp does the rejecting.
        t = min(1.0, max(0.0, (a - ordered_min) / 25.0))
        conf = 0.55 + 0.45 * t
        if a < ordered_min:
            # Inside the ambiguous margin: degrade to NEAR rather than assert
            # an order (§16). NEAR is still a true sentence here.
            relation = NEAR
            conf = max(0.0, 1.0 - a / (band * 2.5))
    else:
        return None

    # Off-road offset erodes confidence in the along-route position too: a
    # place 40 m off the line projects onto the route much less sharply than
    # one on the frontage.
    if lateral > 20.0:
        conf *= max(0.0, 1.0 - (lateral - 20.0) / config.NAV_RELATION_MAX_LATERAL_M)
    conf *= PROVIDER_COORD_PRECISION

    # Which side of travel the landmark is on, from the route's heading on the
    # APPROACH to the maneuver — the direction the driver is facing while the
    # landmark is in the windscreen, not the direction they leave in. Using the
    # departing leg puts every landmark at a left turn on the wrong side.
    # Used later as a spatial-consistency check on what the camera reports,
    # never as the thing that decides the relation.
    gj = max(1, maneuver.polyline_index)
    gi = gj - 1
    heading = geo.bearing_deg(route.geometry[gi][0], route.geometry[gi][1],
                              route.geometry[gj][0], route.geometry[gj][1])
    side = geo.side_of_line(maneuver.latitude, maneuver.longitude, heading,
                            place["lat"], place["lng"])

    return {
        "relation": relation,
        "relation_confidence": round(min(1.0, max(0.0, conf)), 3),
        "along_delta_m": round(delta, 1),
        "lateral_m": round(lateral, 1),
        "distance_to_maneuver_m": round(straight, 1),
        "side": side,
        "approach_heading_deg": round(heading, 1),
    }


def candidates_for(route: M.CanonicalRoute, maneuver: M.CanonicalManeuver,
                   places: List[dict]) -> List[dict]:
    """Allowed, resolvable, unambiguous candidates for one maneuver.

    Three filters, in this order and for these reasons:

      brand      only the classes §21 allows — big standardised signage a
                 driver reads without looking for it
      relation   it has to be describable relative to THIS turn
      uniqueness two of the same brand near one maneuver can never be spoken
                 safely ("turn by the Shell" — which Shell?), so BOTH go (§20).
                 No verbal disambiguation in v1; the fallback is a perfectly
                 good instruction.
    """
    scored = []
    for p in places:
        b = brand_of(p.get("name", ""))
        if not b:
            continue
        brand_key, spoken, cls, salience = b
        if cls not in config.NAV_ANCHOR_TYPES:
            continue
        rel = relation_for(route, maneuver, p)
        if not rel:
            continue
        text = speech_mod.contextual_text(maneuver, spoken, rel["relation"])
        if not text:
            continue
        scored.append(dict(rel, brand=brand_key, place_id=p.get("place_id", ""),
                           label=p.get("name", ""), spoken_label=spoken,
                           type=cls, salience=salience,
                           lat=p["lat"], lng=p["lng"], speech=text))

    # Map-level duplicate rejection. Doing it here as well as in the visual
    # verifier is not redundancy for its own sake: a duplicate the map already
    # knows about should never reach the camera, be verified, be ranked and
    # then be thrown away at the last moment.
    by_brand = {}
    for c in scored:
        by_brand.setdefault(c["brand"], []).append(c)
    kept = []
    for brand, group in by_brand.items():
        if len(group) > 1:
            for c in group:
                c["rejected"] = "duplicate_brand_near_maneuver"
            continue
        kept.append(group[0])

    kept.sort(key=lambda c: (-c["salience"], -c["relation_confidence"],
                             c["distance_to_maneuver_m"]))
    kept = kept[:config.NAV_LANDMARK_MAX_CANDIDATES_PER_MANEUVER]
    for i, c in enumerate(kept):
        c["anchor_id"] = f"{maneuver.id}a{i}"
    return kept


def generate(route: M.CanonicalRoute, provider, now: Optional[float] = None) -> dict:
    """One pass over the route's maneuvers, once per route generation.

    Returns a small stats dict for the log. The route is mutated in place: each
    anchorable maneuver gains `anchors`, which is a list of PREPARED SENTENCES
    with their relations — nothing at drive time formats a string, it selects
    one.

    Failure here is not failure: a provider with no place data, an exhausted
    budget, a maneuver with nothing branded near it and a disabled feature all
    end the same way — no anchors, canonical navigation, no complaint.
    """
    stats = {"lookups": 0, "maneuvers_considered": 0, "candidates": 0,
             "rejected_duplicates": 0, "state": "ready",
             "t_ms": 0.0}
    t0 = time.time()
    if not config.NAV_LANDMARKS_ENABLED:
        route.landmarks_state = "disabled"
        stats["state"] = "disabled"
        return stats
    budget = int(config.NAV_LANDMARK_MAX_LOOKUPS_PER_ROUTE)
    for man in route.maneuvers:
        if not man.anchorable:
            continue
        stats["maneuvers_considered"] += 1
        if stats["lookups"] >= budget:
            stats["state"] = "budget_exhausted"
            break
        try:
            places = provider.landmarks_near(
                man.latitude, man.longitude,
                config.NAV_LANDMARK_SEARCH_RADIUS_M,
                tuple(config.NAV_ANCHOR_TYPES))
        except Exception as e:
            print(f"[nav] landmark lookup failed: {type(e).__name__}: {e}", flush=True)
            places = []
            stats["state"] = "provider_error"
        stats["lookups"] += 1
        cands = candidates_for(route, man, places)
        man.anchors = cands
        stats["candidates"] += len(cands)
    route.landmarks_state = stats["state"]
    route.landmark_lookups = stats["lookups"]
    stats["t_ms"] = round((time.time() - t0) * 1000.0, 1)
    return stats
