"""The navigation service: journeys, route generations, and the speech table.

This is the only place that knows a drive is a sequence of route generations
rather than a route. Everything else either owns one route (the tracker) or one
maneuver (the speech planner).

WHAT A GENERATION IS FOR
------------------------
A reroute does not modify a route — it replaces it, atomically, and increments
`generation_id`. That integer is the thing every queued announcement is
validated against at the moment it would be spoken (§25). "Right here." queued
against generation 1 is not merely stale after a reroute; it belongs to a plan
that no longer exists, and the difference matters because the maneuver index it
names may well be valid on generation 2 and mean a completely different turn.

WHAT THIS FILE MAY NOT DO
-------------------------
Decide anything about navigation truth beyond what the provider said. It
assigns identity, precomputes sentences, and asks the landmark stage for
candidates. It does not adjust a route, re-order maneuvers, infer a direction
or invent a destination.
"""
import os
import re
import threading
import time
from collections import OrderedDict
from typing import List, Optional

import config

from . import landmarks as landmarks_mod
from . import model as M
from . import speech as speech_mod
from .provider import DestinationCandidate, NavError, NavigationProvider

_MAX_ROUTES = 8

_lock = threading.Lock()
_ROUTES: "OrderedDict[str, M.CanonicalRoute]" = OrderedDict()
_JOURNEYS: dict = {}          # journey_id -> {"generation": int, "reroutes": int}
_provider: Optional[NavigationProvider] = None


# ---------------------------------------------------------------------------
# Provider selection
# ---------------------------------------------------------------------------
def get_provider() -> NavigationProvider:
    """The active NavigationProvider.

    Google when a key is present, the fixture provider otherwise — so a machine
    with no key still runs the whole of navigation against fixtures instead of
    failing at the first route. Which one is in use is stated in every route
    payload (`provider`), never inferred.
    """
    global _provider
    if _provider is None:
        if os.getenv("GOOGLE_MAPS_API_KEY", "").strip():
            from .providers.google import GoogleProvider
            _provider = GoogleProvider()
        else:
            from .fixtures import FixtureProvider
            print("[nav] no GOOGLE_MAPS_API_KEY — using the fixture provider", flush=True)
            _provider = FixtureProvider()
    return _provider


def set_provider(provider: Optional[NavigationProvider]) -> None:
    """Swap the provider. The tests' entire mechanism, and §36's escape hatch."""
    global _provider
    _provider = provider


# ---------------------------------------------------------------------------
# Destination resolution (§4)
# ---------------------------------------------------------------------------
_STREET_NUMBER = re.compile(r"^\s*\d+\s+\S")
_STOPWORDS = {"the", "a", "an", "to", "take", "me", "navigate", "directions",
              "go", "let's", "lets", "drive", "at", "of", "please"}


def _normalise(text: str) -> str:
    return re.sub(r"[^a-z0-9 ]+", " ", (text or "").lower()).strip()


def _keywords(text: str) -> set:
    return {w for w in _normalise(text).split() if w and w not in _STOPWORDS}


def clean_destination_phrase(spoken: str) -> str:
    """"Take me to LAX" -> "LAX". "Let's go to the Getty" -> "the Getty".

    Deterministic prefix stripping, so a spoken request and a typed one reach
    the geocoder as the same string. No model: the failure mode of a model here
    is a silently different destination.
    """
    t = (spoken or "").strip()
    t = re.sub(r"^\s*(?:hey\s+rio[, ]+|rio[, ]+)", "", t, flags=re.I)
    t = re.sub(r"^\s*(?:please\s+)?(?:can you\s+|could you\s+)?"
               r"(?:take me to|navigate to|directions to|drive to|"
               r"let'?s go to|let'?s head to|head to|go to|route to|"
               r"take us to|navigate|set a route to)\s+", "", t, flags=re.I)
    # Trailing politeness. "the Ferry Building please" geocodes differently
    # from "the Ferry Building", and a driver who says please means the same
    # place.
    t = re.sub(r"[\s,]*(?:please|thanks|thank you)\s*$", "", t, flags=re.I)
    return t.strip().rstrip("?.!").strip()


def resolve_destination(query: str, lat: Optional[float] = None,
                        lng: Optional[float] = None) -> dict:
    """One destination, or a question. Never a silent guess (§4).

    The rule is narrow on purpose:

      * a street address (leading house number) resolves to the top result —
        addresses are not ambiguous in the way place names are;
      * an exact name match wins outright;
      * otherwise, if two or more candidates are plausible readings of what was
        asked for, that is ambiguity and RIO asks. "The Getty" is two museums
        eight miles apart and picking one silently is the failure this exists
        to prevent.

    Returns {"status": "resolved"|"ambiguous"|"not_found", ...}.
    """
    phrase = clean_destination_phrase(query)
    if not phrase:
        return {"status": "not_found", "query": query}
    provider = get_provider()
    if _STREET_NUMBER.match(phrase):
        dest = provider.destination(query=phrase)
        if dest:
            return {"status": "resolved", "destination": dest, "query": phrase,
                    "reason": "street_address"}
        return {"status": "not_found", "query": phrase}

    cands: List[DestinationCandidate] = provider.suggest(phrase, lat, lng, limit=5)
    if not cands:
        dest = provider.destination(query=phrase)
        if dest:
            return {"status": "resolved", "destination": dest, "query": phrase,
                    "reason": "geocoded"}
        return {"status": "not_found", "query": phrase}

    want = _keywords(phrase)
    exact = [c for c in cands if _normalise(c.display_name) == _normalise(phrase)]
    if len(exact) == 1:
        dest = provider.destination(place_id=exact[0].provider_place_id,
                                    label=exact[0].formatted_address)
        if dest:
            return {"status": "resolved", "destination": dest, "query": phrase,
                    "reason": "exact_name"}

    plausible = [c for c in cands
                 if want and want.issubset(_keywords(c.display_name + " " + c.formatted_address))]
    if len(plausible) > 1:
        return {"status": "ambiguous", "query": phrase,
                "candidates": [c.to_dict() for c in plausible[:3]]}
    pick = plausible[0] if plausible else cands[0]
    if not plausible and len(cands) > 1:
        # Nothing matched cleanly and there is more than one reading: ask.
        return {"status": "ambiguous", "query": phrase,
                "candidates": [c.to_dict() for c in cands[:3]]}
    dest = provider.destination(place_id=pick.provider_place_id,
                               label=pick.formatted_address)
    if not dest:
        return {"status": "not_found", "query": phrase}
    return {"status": "resolved", "destination": dest, "query": phrase,
            "reason": "single_candidate"}


# ---------------------------------------------------------------------------
# Routes
# ---------------------------------------------------------------------------
def _remember(route: M.CanonicalRoute) -> None:
    with _lock:
        _ROUTES[route.route_id] = route
        while len(_ROUTES) > _MAX_ROUTES:
            _ROUTES.popitem(last=False)


def get_route(route_id: str) -> Optional[M.CanonicalRoute]:
    return _ROUTES.get(route_id)


def latest_route() -> Optional[M.CanonicalRoute]:
    """The most recently computed route, or None.

    Read by the visual conversation path, which wants to know where the car is
    headed when the driver asks about something out of the window. It is NOT a
    claim about progress: the tracker is client-side and the server does not
    know which maneuver is current. Callers must present this as "where we're
    going", never as "what's coming up next".
    """
    if not _ROUTES:
        return None
    return next(reversed(_ROUTES.values()))


def build_route(origin_lat: float, origin_lng: float,
                destination: M.CanonicalDestination,
                previous: Optional[M.CanonicalRoute] = None,
                with_landmarks: bool = True) -> M.CanonicalRoute:
    """Provider route -> a complete, speakable, anchored route generation.

    Order matters. Speech is built before landmarks so that every maneuver has
    a canonical instruction even if the landmark stage fails outright, which is
    §27 expressed as control flow rather than as a promise.
    """
    provider = get_provider()
    route = provider.route(origin_lat, origin_lng, destination)

    if previous is not None:
        route.journey_id = previous.journey_id
        with _lock:
            j = _JOURNEYS.setdefault(route.journey_id, {"generation": previous.generation_id,
                                                        "reroutes": 0})
            j["generation"] += 1
            j["reroutes"] += 1
            route.generation_id = j["generation"]
    else:
        route.journey_id = M.new_journey_id()
        route.generation_id = 1
        with _lock:
            _JOURNEYS[route.journey_id] = {"generation": 1, "reroutes": 0}

    for man in route.maneuvers:
        man.speech = speech_mod.build(man, route.destination.display_name,
                                      route.arrival.side)

    stats = {"state": "not_requested", "lookups": 0, "candidates": 0}
    if with_landmarks:
        try:
            stats = landmarks_mod.generate(route, provider)
        except Exception as e:
            # The landmark stage may never take a route down with it.
            print(f"[nav] landmark generation failed: {type(e).__name__}: {e}", flush=True)
            route.landmarks_state = "error"
            stats = {"state": "error", "lookups": 0, "candidates": 0}
    route.landmarks_state = stats.get("state", route.landmarks_state)
    _remember(route)
    return route


def reroute_allowed(journey_id: str) -> bool:
    """Anti-loop. A journey that has rerouted this many times is not being
    rerouted, it is oscillating, and each attempt costs a routing call."""
    j = _JOURNEYS.get(journey_id) or {}
    return int(j.get("reroutes", 0)) < int(config.NAV_REROUTE_MAX_PER_JOURNEY)


def journey_state(journey_id: str) -> dict:
    return dict(_JOURNEYS.get(journey_id) or {"generation": 0, "reroutes": 0})


def reset() -> None:
    """Tests only. A process holds at most one driver's journeys."""
    with _lock:
        _ROUTES.clear()
        _JOURNEYS.clear()


# ---------------------------------------------------------------------------
# Wire shapes
# ---------------------------------------------------------------------------
def timing_config() -> dict:
    """The tracker's thresholds, shipped WITH the route.

    The browser holds no navigation policy of its own. Every number it times
    against arrives in the route payload, so tuning happens in config.py and
    reaches the car on the next route rather than on the next deploy.
    """
    return {
        "gps_stale_timeout_s": config.NAV_GPS_STALE_TIMEOUT_S,
        "gps_accuracy_limit_m": config.NAV_GPS_ACCURACY_LIMIT_M,
        "gps_degraded_bias_s": config.NAV_GPS_DEGRADED_BIAS_S,
        "off_route_distance_m": config.NAV_OFF_ROUTE_DISTANCE_M,
        "off_route_persistence": config.NAV_OFF_ROUTE_PERSISTENCE,
        "reroute_debounce_s": config.NAV_REROUTE_DEBOUNCE_S,
        "progress_rewind_tolerance_m": config.NAV_PROGRESS_REWIND_TOLERANCE_M,
        "maneuver_passed_eps_m": config.NAV_MANEUVER_PASSED_EPS_M,
        "arrive_radius_m": config.NAV_ARRIVE_RADIUS_M,
        "projection_back_m": config.NAV_PROJECTION_BACK_M,
        "projection_fwd_m": config.NAV_PROJECTION_FWD_M,
        "heading_min_displacement_m": config.NAV_HEADING_MIN_DISPLACEMENT_M,
        "heading_max_sample_age_s": config.NAV_HEADING_MAX_SAMPLE_AGE_S,
        "heading_min_speed_ms": config.NAV_HEADING_MIN_SPEED_MS,
        "stationary_speed_ms": config.NAV_STATIONARY_SPEED_MS,
        "early_guidance_s": config.NAV_EARLY_GUIDANCE_S,
        "anchor_acquisition_s": config.NAV_ANCHOR_ACQUISITION_S,
        "context_call_s": config.NAV_CONTEXT_CALL_S,
        "near_turn_s": config.NAV_NEAR_TURN_S,
        "min_call_distance_m": config.NAV_MIN_CALL_DISTANCE_M,
        "max_call_distance_m": config.NAV_MAX_CALL_DISTANCE_M,
        "early_max_distance_m": config.NAV_EARLY_MAX_DISTANCE_M,
        "speed_floor_ms": config.NAV_SPEED_FLOOR_MS,
        "speed_nominal_ms": config.NAV_SPEED_NOMINAL_MS,
        "duplicate_instruction_cooldown_s": config.NAV_DUPLICATE_INSTRUCTION_COOLDOWN_S,
        "anchor_valid_for_s": config.NAV_ANCHOR_VALID_FOR_S,
        "speech_ttl_s": dict(config.NAV_SPEECH_TTL_S),
        "vision_enabled": bool(config.NAV_VISION_ENABLED),
    }


def wire(route: M.CanonicalRoute) -> dict:
    """The route as the browser receives it."""
    out = route.to_dict(geometry=True)
    out["timing"] = timing_config()
    return out


def summary(route: M.CanonicalRoute) -> dict:
    """The shape worth putting in the session log at NAV_ROUTE_STARTED.

    The geometry is deliberately absent: a few hundred vertices per reroute
    would dominate a drive's JSONL, and the route_id ties an event back to the
    full geometry if a review ever needs it. Every sentence RIO *intends* to be
    able to say is here, which is what makes the log reviewable against what it
    actually said.
    """
    return {
        "route_id": route.route_id,
        "journey_id": route.journey_id,
        "generation_id": route.generation_id,
        "provider": route.provider,
        "destination": route.destination.to_dict(),
        "origin": {"lat": route.origin_lat, "lng": route.origin_lng},
        "total_distance_m": route.total_distance_m,
        "duration_s": route.duration_s,
        "eta_epoch": route.eta_epoch,
        "arrival": route.arrival.to_dict(),
        "n_maneuvers": len(route.maneuvers),
        "n_points": len(route.geometry),
        "landmarks_state": route.landmarks_state,
        "landmark_lookups": route.landmark_lookups,
        "maneuvers": [
            {"id": m.id, "sequence": m.sequence, "type": m.type,
             "direction": m.direction, "road_name": m.road_name,
             "route_distance_position": round(m.route_distance_position, 1),
             "speech": m.speech,
             "anchors": [{"anchor_id": a["anchor_id"], "label": a["label"],
                          "relation": a["relation"],
                          "relation_confidence": a["relation_confidence"],
                          "speech": a["speech"]} for a in m.anchors]}
            for m in route.maneuvers
        ],
    }
