"""Acceptance tests for the server half of contextual navigation.

    python -m tools.nav_server_selftest
    python -m tools.nav_server_selftest --live     # + one real provider route

Runs the real code in-process against the fixture provider: no HTTP, no
routing key, no camera, no model, and without --live no network at all. The
client half — the tracker, the speech planner, the arbiter, the context
lifecycle — is `node tools/nav_selftest.js`, and between them they cover the
whole of §32's required list.

Seven parts, separated by what each one can prove:

  A. PROVIDER BOUNDARY — the canonical model is complete enough to navigate
     from, and the fixture provider substitutes for the real one with nothing
     downstream noticing. That is §36 exercised rather than asserted.

  B. DESTINATION RESOLUTION — "Take me to LAX" resolves; something that reads
     as two places asks which one. RIO never silently picks (§4).

  C. RELATION FROM MAP DATA — where the turn sits relative to a landmark, from
     coordinates alone, including the deliberate refusals: too far, too far off
     the road, and the ambiguous margin that degrades to NEAR (§16, addendum).

  D. CANDIDATE GENERATION — allowed brands only, one lookup pass per route
     generation, a hard budget cap, duplicates rejected before they can ever be
     spoken (§13, §20, §21, addendum).

  E. ANCHOR GATES — every gate in §18, each one failed on its own, plus the
     JUST_AFTER -> NEAR degrade and the "if even NEAR is unsafe, reject" case.

  F. VERIFICATION — the whole pipeline on simulated landmark observations,
     including the two failures that must be invisible to navigation: the
     vision model unavailable, and the camera not there at all.

  G. SPEECH — the table is deterministic, complete before the drive starts, and
     addressable only by (route, maneuver, call, anchor). Nothing formats a
     sentence while the car is moving, and no model is reachable from the
     navigation speech path at all.
"""
import argparse
import inspect
import os
import re
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import config                                        # noqa: E402
from navigation import anchors as anchors_mod        # noqa: E402
from navigation import fixtures                      # noqa: E402
from navigation import landmarks as landmarks_mod    # noqa: E402
from navigation import model as M                    # noqa: E402
from navigation import service                       # noqa: E402
from navigation import speech as speech_mod          # noqa: E402
from navigation import verify as verify_mod          # noqa: E402
from navigation.provider import DestinationCandidate, NavigationProvider  # noqa: E402

PASS, FAIL = [], []


def ok(cond, what):
    (PASS if cond else FAIL).append(what)
    print(("  ok    " if cond else "  FAIL  ") + what)


def section(name):
    print(f"\n=== {name} ===")


# ---------------------------------------------------------------------------
# Shared fixture: an L-shaped city route with landmarks placed by route
# position, so what the relation SHOULD be is written into the test rather than
# read back out of the code that computes it.
# ---------------------------------------------------------------------------
def build(places=None, **kw):
    route = fixtures.city_route()
    service.set_provider(fixtures.FixtureProvider(route, places or [], **kw))
    service.reset()
    return service.build_route(route.geometry[0][0], route.geometry[0][1],
                               route.destination)


def shell_near_turn(route, along=1185.0, lateral=12.0, name="Shell"):
    return fixtures.place_at(route, along, lateral, name)


# ---------------------------------------------------------------------------
# A. Provider boundary
# ---------------------------------------------------------------------------
def run_provider():
    section("A. provider boundary — the canonical model, and substitution")
    route = build()

    ok(route.provider == "fixture" and route.generation_id == 1,
       "a route arrives as a CanonicalRoute with a generation, from any provider")
    ok(all(isinstance(m, M.CanonicalManeuver) for m in route.maneuvers),
       "every maneuver is canonical — no provider dict survives the boundary")
    ok([m.direction for m in route.maneuvers] == ["LEFT", "RIGHT", "RIGHT"],
       "direction is normalised out of the provider's own vocabulary")
    ok(route.arrival.side in (M.LEFT, M.RIGHT, M.UNKNOWN),
       "arrival side is provider-supplied, and UNKNOWN is a legal answer")
    ok(all(m.route_distance_position >= 0 and m.polyline_index >= 0
           for m in route.maneuvers),
       "each maneuver is pinned to an exact vertex, not a nearest-point guess")

    wire = service.wire(route)
    for key in ("route_id", "generation_id", "geometry", "maneuvers", "timing",
                "arrival", "destination"):
        ok(key in wire, f"the wire shape carries {key}")
    ok(all(k in wire["timing"] for k in
           ("early_guidance_s", "context_call_s", "near_turn_s",
            "off_route_distance_m", "gps_stale_timeout_s")),
       "every threshold the browser times against ships WITH the route")

    # Reroute: same journey, next generation, and the destination object is
    # reused rather than re-resolved.
    r2 = service.build_route(route.geometry[0][0], route.geometry[0][1],
                             route.destination, previous=route)
    ok(r2.journey_id == route.journey_id and r2.generation_id == 2,
       "a reroute is the next generation of the same journey")
    ok(r2.route_id != route.route_id,
       "and a different route entirely — nothing is patched in place")

    # Anti-flap.
    j = route.journey_id
    for _ in range(config.NAV_REROUTE_MAX_PER_JOURNEY + 2):
        if service.reroute_allowed(j):
            service._JOURNEYS[j]["reroutes"] += 1
    ok(not service.reroute_allowed(j),
       "a journey that keeps rerouting is stopped rather than left to oscillate")

    class Bare(NavigationProvider):
        """A provider with no place data at all — the minimum contract."""
        name = "bare"

        def suggest(self, query, lat=None, lng=None, limit=5, session=None):
            return [DestinationCandidate("Somewhere", "Somewhere, CA", "p1")]

        def destination(self, query="", place_id="", label="", session=None):
            return fixtures.city_route().destination

        def route(self, origin_lat, origin_lng, destination):
            r = fixtures.city_route()
            r.origin_lat, r.origin_lng = origin_lat, origin_lng
            return r

    service.set_provider(Bare())
    service.reset()
    bare = service.build_route(34.043, -118.267, Bare().destination())
    ok(bare.landmarks_state == "ready" and
       all(not m.anchors for m in bare.maneuvers),
       "a provider with no place data routes normally with no anchors at all")
    ok(all(m.speech.get("primary") for m in bare.maneuvers),
       "and every maneuver still has a complete spoken instruction")


# ---------------------------------------------------------------------------
# B. Destination resolution
# ---------------------------------------------------------------------------
class ScriptedProvider(fixtures.FixtureProvider):
    """Suggestions on demand, so ambiguity can be posed exactly."""

    def __init__(self, suggestions, fail_suggest=False):
        super().__init__()
        self._suggestions = suggestions
        self.fail_suggest = fail_suggest
        self.resolved = []

    def suggest(self, query, lat=None, lng=None, limit=5, session=None):
        if session:
            self.sessions_seen.append(("suggest", session))
        if self.fail_suggest:
            # What a provider outage looks like from here: an empty list, never
            # an exception. The driver types the address in full instead.
            return []
        return [DestinationCandidate(*s) for s in self._suggestions][:limit]

    def destination(self, query="", place_id="", label="", session=None):
        if session:
            self.sessions_seen.append(("destination", session))
        self.resolved.append({"query": query, "place_id": place_id})
        return M.CanonicalDestination(
            display_name=label or query or "Resolved",
            formatted_address=label or query, latitude=34.0, longitude=-118.0,
            provider_place_id=place_id or "p")


def run_destination():
    section("B. destination resolution — resolve, or ask; never guess")

    phrases = {
        "Take me to LAX": "LAX",
        "Navigate to Griffith Observatory": "Griffith Observatory",
        "Directions to 123 Main Street": "123 Main Street",
        "Let's go to the Getty": "the Getty",
        "Hey RIO, take us to the Ferry Building please": "the Ferry Building",
    }
    for spoken, want in phrases.items():
        got = service.clean_destination_phrase(spoken)
        ok(got == want, f'"{spoken}" -> "{got}"')

    service.set_provider(ScriptedProvider([
        ("Los Angeles International Airport", "1 World Way, Los Angeles, CA", "p_lax"),
    ]))
    res = service.resolve_destination("Take me to LAX")
    ok(res["status"] == "resolved", "one candidate resolves without a question")

    service.set_provider(ScriptedProvider([
        ("Getty Center", "1200 Getty Center Dr, Los Angeles, CA", "p_center"),
        ("Getty Villa", "17985 Pacific Coast Hwy, Pacific Palisades, CA", "p_villa"),
    ]))
    res = service.resolve_destination("Let's go to the Getty")
    ok(res["status"] == "ambiguous",
       "two museums called the Getty is a question, not a coin toss")
    ok(len(res["candidates"]) == 2 and
       {c["display_name"] for c in res["candidates"]} == {"Getty Center", "Getty Villa"},
       "and both readings come back so the driver can pick")

    service.set_provider(ScriptedProvider([
        ("Griffith Observatory", "2800 E Observatory Rd, Los Angeles, CA", "p_obs"),
        ("Griffith Park", "4730 Crystal Springs Dr, Los Angeles, CA", "p_park"),
    ]))
    res = service.resolve_destination("Navigate to Griffith Observatory")
    ok(res["status"] == "resolved" and res["reason"] == "exact_name",
       "an exact name match wins outright even with a similar neighbour")

    service.set_provider(ScriptedProvider([]))
    res = service.resolve_destination("Directions to 123 Main Street")
    ok(res["status"] == "resolved" and res["reason"] == "street_address",
       "a street address is not ambiguous the way a place name is")

    res = service.resolve_destination("   ")
    ok(res["status"] == "not_found", "an empty request resolves to nothing at all")


# ---------------------------------------------------------------------------
# C. Relation from map data
# ---------------------------------------------------------------------------
def run_relation():
    section("C. relation — computed from coordinates, never from the camera")
    route = fixtures.city_route()
    man = route.maneuvers[0]                      # left turn at 1200 m
    turn_at = man.route_distance_position

    cases = [
        ("a Shell 14 m before the turn", turn_at - 14, 12, landmarks_mod.NEAR),
        ("a Shell right at the turn", turn_at - 2, 10, landmarks_mod.NEAR),
        ("a Starbucks 50 m before the turn", turn_at - 50, 10, landmarks_mod.JUST_AFTER),
        ("a CVS 50 m past the turn", turn_at + 50, 12, landmarks_mod.JUST_BEFORE),
    ]
    for name, along, lateral, want in cases:
        p = fixtures.place_at(route, along, lateral, "Shell")
        rel = landmarks_mod.relation_for(route, man, p)
        got = rel["relation"] if rel else None
        ok(got == want, f"{name} -> the turn is {want} it (got {got})")

    # The ambiguous margin: outside NEAR, inside the ordered minimum. Claiming
    # an order here is the mistake that sends a driver through the junction.
    p = fixtures.place_at(route, turn_at - 26, 10, "Shell")
    rel = landmarks_mod.relation_for(route, man, p)
    ok(rel and rel["relation"] == landmarks_mod.NEAR,
       "a landmark in the ambiguous margin degrades to NEAR rather than claiming order")

    # Confidence is map-data quality, and only that.
    close = landmarks_mod.relation_for(route, man, fixtures.place_at(route, turn_at, 8, "Shell"))
    edge = landmarks_mod.relation_for(route, man, fixtures.place_at(route, turn_at - 20, 40, "Shell"))
    ok(close["relation_confidence"] > edge["relation_confidence"],
       "a landmark on the corner is a more confident NEAR than one 40 m off the road "
       f"({close['relation_confidence']} vs {edge['relation_confidence']})")

    far = landmarks_mod.relation_for(route, man, fixtures.place_at(route, turn_at - 200, 10, "Shell"))
    ok(far is None, "a landmark 200 m from the turn cannot describe it at all")
    wide = landmarks_mod.relation_for(route, man, fixtures.place_at(route, turn_at, 80, "Shell"))
    ok(wide is None, "nor can one 80 m off the road — that is not roadside")

    right = landmarks_mod.relation_for(route, man, fixtures.place_at(route, turn_at - 20, 12, "Shell"))
    left = landmarks_mod.relation_for(route, man, fixtures.place_at(route, turn_at - 20, -12, "Shell"))
    ok(right["side"] == "RIGHT" and left["side"] == "LEFT",
       "side comes from the APPROACH heading, so it is the side the driver sees")


# ---------------------------------------------------------------------------
# D. Candidate generation
# ---------------------------------------------------------------------------
def run_candidates():
    section("D. candidates — allowed brands, one pass, capped, deduplicated")
    route = fixtures.city_route()
    turn = route.maneuvers[0].route_distance_position

    places = [
        shell_near_turn(route, turn - 14, 12, "Shell"),
        fixtures.place_at(route, turn - 50, 10, "Starbucks Coffee Company", "coffee_shop"),
        fixtures.place_at(route, turn - 18, 14, "Bob's Auto Repair", "car_repair"),
        fixtures.place_at(route, turn - 10, 16, "Alameda Dry Cleaning", "laundry"),
    ]
    provider = fixtures.FixtureProvider(route, places)
    service.set_provider(provider)
    service.reset()
    r = service.build_route(route.geometry[0][0], route.geometry[0][1], route.destination)
    m0 = r.maneuvers[0]
    labels = [a["label"] for a in m0.anchors]
    ok("Shell" in labels and "Starbucks Coffee Company" in labels,
       "branded fuel and major chain signage are allowed anchors")
    ok(not any("Bob's" in l or "Dry Cleaning" in l for l in labels),
       "an unbranded local business is not — there is no reliable sign to see")
    ok(all(a["speech"] for a in m0.anchors),
       "every candidate arrives with its sentence already written")
    ok(m0.anchors[0]["speech"] == "Turn left by the Shell station.",
       'the best candidate\'s line is "Turn left by the Shell station." — got "'
       + m0.anchors[0]["speech"] + '"')
    ok(m0.anchors[0]["salience"] >= m0.anchors[-1]["salience"],
       "candidates are ordered with the most recognisable sign first")

    ok(provider.landmark_calls <= config.NAV_LANDMARK_MAX_LOOKUPS_PER_ROUTE,
       f"lookups stay inside the budget ({provider.landmark_calls} of "
       f"{config.NAV_LANDMARK_MAX_LOOKUPS_PER_ROUTE})")
    anchorable = [m for m in r.maneuvers if m.anchorable]
    ok(provider.landmark_calls == len(anchorable),
       "one lookup per anchorable maneuver, and none for arrival or a merge")

    before = provider.landmark_calls
    service.wire(r)
    for m in r.maneuvers:
        _ = m.anchors
    ok(provider.landmark_calls == before,
       "reading the anchors again costs nothing — they are cached for the generation")

    # Duplicates: two Shells near one turn can never be spoken safely.
    dupes = [shell_near_turn(route, turn - 14, 12, "Shell"),
             shell_near_turn(route, turn - 30, -14, "Shell")]
    service.set_provider(fixtures.FixtureProvider(route, dupes))
    service.reset()
    r2 = service.build_route(route.geometry[0][0], route.geometry[0][1], route.destination)
    ok(not r2.maneuvers[0].anchors,
       "two Shell stations near one turn rejects BOTH — no verbal disambiguation in v1")

    # The budget cap, forced.
    old = config.NAV_LANDMARK_MAX_LOOKUPS_PER_ROUTE
    try:
        config.NAV_LANDMARK_MAX_LOOKUPS_PER_ROUTE = 1
        provider = fixtures.FixtureProvider(route, places)
        service.set_provider(provider)
        service.reset()
        r3 = service.build_route(route.geometry[0][0], route.geometry[0][1],
                                 route.destination)
        ok(provider.landmark_calls == 1 and r3.landmarks_state == "budget_exhausted",
           "the budget cap stops the pass and says so, rather than quietly spending")
        ok(all(m.speech.get("primary") for m in r3.maneuvers),
           "and the maneuvers it did not reach navigate normally")
    finally:
        config.NAV_LANDMARK_MAX_LOOKUPS_PER_ROUTE = old

    # Feature off.
    old_enabled = config.NAV_LANDMARKS_ENABLED
    try:
        config.NAV_LANDMARKS_ENABLED = False
        provider = fixtures.FixtureProvider(route, places)
        service.set_provider(provider)
        service.reset()
        r4 = service.build_route(route.geometry[0][0], route.geometry[0][1],
                                 route.destination)
        ok(provider.landmark_calls == 0 and r4.landmarks_state == "disabled",
           "with landmarks switched off nothing is fetched at all")
    finally:
        config.NAV_LANDMARKS_ENABLED = old_enabled


# ---------------------------------------------------------------------------
# E. Anchor gates
# ---------------------------------------------------------------------------
def good_observation(now):
    return {"visible": True, "identity_confidence": 0.92,
            "visibility_confidence": 0.80, "instances": 1, "observations": 3,
            "tracking_duration_s": 2.1, "last_seen_t": now, "side": "RIGHT",
            "depth_m": 42.0}


def good_candidate():
    return {"anchor_id": "m0a0", "label": "Shell", "type": "gas_station",
            "relation": landmarks_mod.NEAR, "relation_confidence": 0.75,
            "distance_to_maneuver_m": 18.0, "side": "RIGHT", "salience": 1.0}


def run_gates():
    section("E. anchor gates — every one of them, failed on its own")
    now = time.time()
    ok(anchors_mod.validate(good_candidate(), good_observation(now), now)[0],
       "a clean observation of an allowed brand passes")

    breaks = [
        ("not visible", {"visible": False}, "not_visible"),
        ("uncertain identity", {"identity_confidence": 0.4}, "identity_confidence"),
        ("barely legible", {"visibility_confidence": 0.2}, "visibility_confidence"),
        ("seen in one frame only", {"observations": 1}, "tracking_observations"),
        ("held for a fraction of a second", {"tracking_duration_s": 0.2}, "tracking_duration"),
        ("last seen five seconds ago", {"last_seen_t": now - 5.0}, "observation_stale"),
        ("two of them in view", {"instances": 2}, "scene_uniqueness"),
        ("reported 300 m away", {"depth_m": 300.0}, "spatial_consistency"),
        ("reported on the wrong side", {"side": "LEFT"}, "spatial_consistency"),
    ]
    for name, override, expect in breaks:
        obs = dict(good_observation(now), **override)
        passed, fails = anchors_mod.validate(good_candidate(), obs, now)
        ok(not passed and expect in fails,
           f"{name} -> rejected ({expect})" if not passed else f"{name} -> WRONGLY ACCEPTED")

    bad_type = dict(good_candidate(), type="florist")
    ok(not anchors_mod.validate(bad_type, good_observation(now), now)[0],
       "a class outside the allowed anchor types never gets as far as being ranked")

    # Relation degrade, and the refusal beyond it.
    ordered = dict(good_candidate(), relation=landmarks_mod.JUST_AFTER,
                   relation_confidence=0.95)
    ok(anchors_mod.resolve_relation(ordered)[0] == landmarks_mod.JUST_AFTER,
       "a confident JUST_AFTER is spoken as JUST_AFTER")
    weak = dict(ordered, relation_confidence=0.7)
    got = anchors_mod.resolve_relation(weak)
    ok(got and got[0] == landmarks_mod.NEAR and got[2] is True,
       'an uncertain JUST_AFTER degrades to NEAR — "by the Shell", not "just after" it')
    hopeless = dict(ordered, relation_confidence=0.3)
    ok(anchors_mod.resolve_relation(hopeless) is None,
       "and when even NEAR cannot be supported, there is no anchor at all")

    # Selection: one anchor, chosen by a readable ordering.
    entries = [
        {"candidate": dict(good_candidate(), anchor_id="a_conv", type="convenience_store",
                           salience=0.8, distance_to_maneuver_m=10.0),
         "observation": good_observation(now)},
        {"candidate": dict(good_candidate(), anchor_id="a_shell", salience=1.0,
                           distance_to_maneuver_m=30.0),
         "observation": good_observation(now)},
    ]
    best = anchors_mod.select(entries)
    ok(best["candidate"]["anchor_id"] == "a_shell",
       "the bigger, more recognisable sign wins over the nearer weaker one")
    ok(anchors_mod.select([]) is None, "and nothing at all is a legal answer")


# ---------------------------------------------------------------------------
# F. Verification
# ---------------------------------------------------------------------------
class ScriptedObserver(verify_mod.LandmarkObserver):
    """Simulated landmark observations, keyed by label (§32)."""

    def __init__(self, table, reason=None):
        self.table = table
        self.reason = reason

    def observe(self, session_key, candidates, now):
        if self.reason:
            return {"_reason": self.reason}
        out = {}
        for c in candidates:
            entry = self.table.get(c["label"])
            if entry is None:
                out[c["anchor_id"]] = {"visible": False, "observations": 0,
                                       "tracking_duration_s": 0.0,
                                       "identity_confidence": 0.0,
                                       "visibility_confidence": 0.0,
                                       "instances": 0, "last_seen_t": None}
            else:
                out[c["anchor_id"]] = dict(entry, last_seen_t=now)
        return out


def run_verification():
    section("F. verification — simulated observations, and the failures that hide")
    route = fixtures.city_route()
    turn = route.maneuvers[0].route_distance_position
    places = [shell_near_turn(route, turn - 14, 12, "Shell"),
              fixtures.place_at(route, turn - 50, 10, "Starbucks", "coffee_shop")]
    service.set_provider(fixtures.FixtureProvider(route, places))
    service.reset()
    r = service.build_route(route.geometry[0][0], route.geometry[0][1], route.destination)
    cands = r.maneuvers[0].anchors
    ok(len(cands) >= 1, "the maneuver has candidates to verify")

    seen = {"visible": True, "identity_confidence": 0.92, "visibility_confidence": 0.8,
            "instances": 1, "observations": 3, "tracking_duration_s": 2.0,
            "side": "RIGHT", "depth_m": 42.0}

    verify_mod.set_observer(ScriptedObserver({"Shell": seen}))
    res = verify_mod.verify("session", cands)
    ok(res["anchor"] and res["anchor"]["label"] == "Shell",
       "the Shell the map expected, seen by the camera, becomes the anchor")
    ok(res["anchor"]["turn_relation_to_anchor"] == landmarks_mod.NEAR,
       "carrying the relation the MAP computed, not one the camera guessed")
    ok(set(res["anchor"]) == {"anchor_id", "label", "type", "turn_relation_to_anchor",
                              "identity_confidence", "relation_confidence",
                              "visibility_confidence", "valid_for_s", "valid_until"},
       "and nothing else crosses the boundary — no boxes, tracks or depth history")

    verify_mod.set_observer(ScriptedObserver({"Shell": dict(seen, instances=2)}))
    ok(verify_mod.verify("session", cands)["anchor"] is None,
       "two Shells in view -> no anchor, and the canonical instruction is used")

    verify_mod.set_observer(ScriptedObserver({"Shell": dict(seen, identity_confidence=0.5)}))
    ok(verify_mod.verify("session", cands)["anchor"] is None,
       "an uncertain identity -> no anchor")

    verify_mod.set_observer(ScriptedObserver({}))
    res = verify_mod.verify("session", cands)
    ok(res["anchor"] is None and res["reason"] in ("not_visible", "no_candidate_passed"),
       "nothing visible -> no anchor, with a reason worth logging")

    verify_mod.set_observer(ScriptedObserver({}, reason="camera_unavailable"))
    res = verify_mod.verify("session", cands)
    ok(res["anchor"] is None and res["reason"] == "camera_unavailable",
       "no camera at all -> no anchor, and no error")

    class Exploding(verify_mod.LandmarkObserver):
        def observe(self, *a, **k):
            raise RuntimeError("the vision model is not loaded")

    verify_mod.set_observer(Exploding())
    res = verify_mod.verify("session", cands)
    ok(res["anchor"] is None and res["reason"] == "observer_error",
       "the vision model falling over -> no anchor, and navigation never hears about it")

    old = config.NAV_VISION_ENABLED
    try:
        config.NAV_VISION_ENABLED = False
        verify_mod.set_observer(ScriptedObserver({"Shell": seen}))
        ok(verify_mod.verify("session", cands)["reason"] == "vision_disabled",
           "vision switched off is a first-class, silent outcome")
    finally:
        config.NAV_VISION_ENABLED = old
    ok(verify_mod.verify("session", [])["reason"] == "no_candidates",
       "no candidates is likewise not an error")
    verify_mod.set_observer(None)


# ---------------------------------------------------------------------------
# G. Speech
# ---------------------------------------------------------------------------
def run_speech():
    section("G. speech — deterministic, precomputed, and addressable only by id")
    route = fixtures.city_route()
    turn = route.maneuvers[0].route_distance_position
    service.set_provider(fixtures.FixtureProvider(
        route, [shell_near_turn(route, turn - 14, 12, "Shell")]))
    service.reset()
    r = service.build_route(route.geometry[0][0], route.geometry[0][1], route.destination)

    m0, m1, arrive = r.maneuvers
    ok(m0.speech["early"] == "Left turn coming up.", "the early line prepares, with no distance")
    ok(m0.speech["primary"] == "Take the next left onto Lincoln Boulevard.",
       "the canonical instruction is a complete sentence on its own")
    ok(m0.speech["imminent"] == "Left here.", "the imminent backup is two words")
    ok(arrive.speech["arrival"] == "Your destination is on the right.",
       "arrival says the side the provider gave")

    plain = M.CanonicalManeuver(id="x", sequence=0, type=M.TURN, direction=M.RIGHT,
                                road_name="", latitude=0, longitude=0,
                                route_distance_position=0, polyline_index=0,
                                instruction="Turn right")
    ok(speech_mod.primary_text(plain) == "Take the next right.",
       'with no road name it is exactly "Take the next right."')
    ok(speech_mod.arrival_text("The Getty", M.UNKNOWN) == "You've arrived at The Getty.",
       "an UNKNOWN arrival side is omitted, never guessed")

    ok(m0.anchors[0]["speech"] == "Turn left by the Shell station.",
       "the contextual line is prepared at route load, not composed while driving")

    # Every sentence is addressable, and only by id.
    ok(speech_mod.text_for(r, "m0", "primary") == m0.speech["primary"],
       "/nav/voice resolves (route, maneuver, call) to the stored line")
    ok(speech_mod.text_for(r, "m0", "primary", m0.anchors[0]["anchor_id"]) ==
       "Turn left by the Shell station.",
       "...and (route, maneuver, call, anchor) to the contextual one")
    ok(speech_mod.text_for(r, "m0", "primary", "not_a_real_anchor") is None,
       "an anchor that is not on this route is not a sentence RIO can say")
    ok(speech_mod.text_for(r, "m99", "primary") is None,
       "nor is a maneuver that is not on it")
    ok(speech_mod.text_for(r, "m0", "freestyle") is None,
       "nor is a call type that does not exist")

    ok(all(m.speech for m in r.maneuvers),
       "every maneuver on the route has its lines before the drive starts")

    # The firewall, read out of the source: nothing on the navigation speech
    # path may reach a model. Same check headway/live_selftest.py runs against
    # live_policy.py, and for the same reason.
    banned = ("openai", "llm_interface", "import enrich", "import vision",
              "get_adapter", "requests.", "httpx", "model.generate")
    for mod in (speech_mod, anchors_mod, landmarks_mod):
        src = inspect.getsource(mod)
        hits = [b for b in banned if b in src]
        ok(not hits, f"{mod.__name__} cannot reach a model or the network "
                     + (f"(found {hits})" if hits else ""))


# ---------------------------------------------------------------------------
# H. Spoken destinations
# ---------------------------------------------------------------------------
def run_spoken():
    section("H. spoken destinations — one classifier, one resolver, no model")
    import router as request_router

    for phrase in ("Take me to LAX", "Navigate to Griffith Observatory",
                   "Directions to 123 Main Street", "Let's go to the Getty",
                   "Set a route to the Ferry Building"):
        r = request_router.classify(phrase, use_model=False)
        ok(r["request_type"] == request_router.NAVIGATION,
           f'"{phrase}" is a navigation request')

    # ...and the near misses that must NOT be. A driver asking how far it is,
    # or what that building is, has not asked to be taken anywhere, and routing
    # them as a destination would restart the drive.
    for phrase, why in (("how far is it", "a question about the route we are on"),
                        ("where are we", "a landmark question"),
                        ("what's that building", "a question about the world"),
                        ("how are my tires", "a question about the car"),
                        ("take me back", "names no destination at all")):
        r = request_router.classify(phrase, use_model=False)
        ok(r["request_type"] != request_router.NAVIGATION, f'"{phrase}" is not — {why}')

    ok(request_router.classify("Take me to LAX", use_model=False)["object_reference"]
       == "LAX",
       "the destination phrase is extracted by navigation's own cleaner, not a second one")
    ok(not request_router.is_visual(request_router.NAVIGATION),
       "a destination request never reaches the camera")

    ok(speech_mod.destination_reply("resolved", name="Griffith Observatory")
       == "Routing to Griffith Observatory.",
       "a resolved destination is confirmed with the provider's own name")
    two = [{"display_name": "Getty Center"}, {"display_name": "Getty Villa"}]
    ok(speech_mod.destination_reply("ambiguous", candidates=two)
       == "I found two — Getty Center or Getty Villa. Which one?",
       "an ambiguous one is a question naming both readings")
    ok(speech_mod.destination_reply("not_found", query="Xyzzy")
       == "I couldn't find Xyzzy.",
       "and one that cannot be found is said plainly, not improvised around")


# ---------------------------------------------------------------------------
# I. The visual observer's own arithmetic
# ---------------------------------------------------------------------------
class ScriptedAdapter:
    """Stands in for the resident VLM: a scripted answer per frame."""

    def __init__(self, per_frame):
        self.per_frame = list(per_frame)
        self.calls = 0

    def landmark(self, frame_jpeg, labels):
        reports = self.per_frame[min(self.calls, len(self.per_frame) - 1)]
        self.calls += 1
        return [dict(reports.get(l, {"visible": False, "identity": 0.0,
                                     "clarity": 0.0, "count": 0, "side": None,
                                     "box": None})) for l in labels]


def _ring_with(n_frames, spacing_s=1.0):
    """A frame ring holding n frames, spaced in wall time.

    push() stamps wall_t with the clock, so the spacing is applied afterwards —
    the observer's frame thinning is a claim about elapsed time and cannot be
    tested with frames that all arrived in the same millisecond.
    """
    import io

    import framebuf
    from PIL import Image

    # Real JPEG bytes, because the depth path decodes them. A ring full of
    # b"jpeg" would silently skip every check below it.
    buf = io.BytesIO()
    Image.new("RGB", (640, 360), (40, 44, 52)).save(buf, format="JPEG")
    jpeg = buf.getvalue()

    ring = framebuf.FrameRing(seconds=30.0, max_frames=16)
    now = time.time()
    for i in range(n_frames):
        rf = ring.push(jpeg, {"ok": True, "t": float(i),
                              "image": {"w": 640, "h": 360},
                              "scene_objects": []})
        rf.wall_t = now - (n_frames - 1 - i) * spacing_s
    return ring


def run_observer():
    section("I. the visual observer — persistence, uniqueness, and abstention")
    import framebuf

    route = fixtures.city_route()
    turn = route.maneuvers[0].route_distance_position
    service.set_provider(fixtures.FixtureProvider(
        route, [shell_near_turn(route, turn - 14, 12, "Shell")]))
    service.reset()
    r = service.build_route(route.geometry[0][0], route.geometry[0][1], route.destination)
    cands = r.maneuvers[0].anchors

    seen = {"visible": True, "identity": 0.9, "clarity": 0.8, "count": 1, "side": "right"}
    unseen = {"visible": False, "identity": 0.0, "clarity": 0.0, "count": 0}

    original_peek = framebuf.peek_ring
    try:
        ring = _ring_with(3)
        framebuf.peek_ring = lambda key: ring
        verify_mod.set_observer(verify_mod.VisionObserver())

        verify_mod.set_adapter(ScriptedAdapter([{"Shell": seen}] * 3))
        res = verify_mod.verify("session", cands)
        ok(res["anchor"] is not None,
           "a landmark held across three frames verifies")
        ok(res["observation"]["observations"] == 3 and
           res["observation"]["frames_examined"] == 3,
           "and the count of observations is the count of frames it was in")
        ok(res["observation"]["tracking_duration_s"] >= 1.9,
           f"tracking duration is elapsed wall time, not a frame count "
           f"({res['observation']['tracking_duration_s']:.1f} s)")
        ok(res["observation"]["depth_m"] is None,
           "no box means no depth, and no depth is not a rejection")

        # Seen once out of three: a sign flickering in and out of view is not
        # one a driver can be told to turn at.
        verify_mod.set_adapter(ScriptedAdapter([{"Shell": seen},
                                                {"Shell": unseen},
                                                {"Shell": unseen}]))
        ok(verify_mod.verify("session", cands)["anchor"] is None,
           "a landmark seen in one frame of three does not")

        # Two of them, in only one of the frames, still rejects.
        verify_mod.set_adapter(ScriptedAdapter([
            dict(Shell=dict(seen, count=2)), {"Shell": seen}, {"Shell": seen}]))
        res = verify_mod.verify("session", cands)
        ok(res["anchor"] is None and
           "scene_uniqueness" in (res["rejections"].get("m0a0") or []),
           "two Shells in a single frame is enough to reject the whole thing")

        # A model that answers with nonsense answers "no".
        class Nonsense:
            def landmark(self, jpeg, labels):
                return [{"visible": True, "identity": "very sure"}]
        verify_mod.set_adapter(Nonsense())
        res = verify_mod.verify("session", cands)
        ok(res["anchor"] is None and res["reason"] != "observer_error",
           "a malformed model reply is read as 'not visible' — not as 'probably', "
           "and not as a crash")

        # One frame in the window is not enough to persist anything.
        ring2 = _ring_with(1)
        framebuf.peek_ring = lambda key: ring2
        verify_mod.set_adapter(ScriptedAdapter([{"Shell": seen}]))
        res = verify_mod.verify("session", cands)
        ok(res["anchor"] is None and res["reason"] == "not_enough_frames",
           "a single frame in the window is not an observation")

        framebuf.peek_ring = lambda key: None
        ok(verify_mod.verify("session", cands)["reason"] == "camera_unavailable",
           "and no ring at all is the camera being absent, which is fine")

        # -- depth, against a stand-in for the real module ------------------
        # roi_depth returns (metres, confidence, stats). Reading that tuple as
        # anything else is a bug that hides perfectly: the check would simply
        # abstain on every frame for the life of the product, and nothing would
        # ever look wrong.
        framebuf.peek_ring = lambda key: ring
        import numpy as np
        import types

        calls = {}

        def fake_depth(conf, metres):
            mod = types.ModuleType("headway.depth")
            mod.depth_map = lambda img: np.zeros(img.shape[:2], dtype="float32")

            def roi_depth(dmap, box, shrink=0.6):
                calls["box"] = box
                return metres, conf, {"valid_frac": 0.9}
            mod.roi_depth = roi_depth
            return mod

        seen_box = dict(seen, box=[600, 300, 700, 380])
        for conf, metres, want, why in (
                (0.9, 42.0, 42.0, "a confident reading is used"),
                (0.1, 42.0, None, "a reading the depth model does not trust is not"),
                (0.9, float("nan"), None, "and NaN — too few valid pixels — abstains")):
            sys.modules["headway.depth"] = fake_depth(conf, metres)
            verify_mod.set_adapter(ScriptedAdapter([{"Shell": seen_box}] * 3))
            res = verify_mod.verify("session", cands)
            got = (res.get("observation") or {}).get("depth_m")
            ok(got == want or (want is None and got is None), why)

        # A normalised box must not be read as pixels: that samples the top-left
        # corner of the frame, which is the sky, and reports it confidently.
        sys.modules["headway.depth"] = fake_depth(0.9, 42.0)
        verify_mod.set_adapter(ScriptedAdapter(
            [{"Shell": dict(seen, box=[0.47, 0.42, 0.55, 0.53])}] * 3))
        verify_mod.verify("session", cands)
        ok(calls.get("box") and calls["box"][0] > 100 and calls["box"][1] > 100,
           f"a 0-1 box is scaled to the frame before it is sampled ({calls.get('box')})")
        sys.modules.pop("headway.depth", None)
    finally:
        framebuf.peek_ring = original_peek
        verify_mod.set_adapter(None)
        verify_mod.set_observer(None)


# ---------------------------------------------------------------------------
# J. Destination autocomplete
# ---------------------------------------------------------------------------
def run_autocomplete():
    section("J. autocomplete — predictions while typing, and the fallback under them")
    import app as app_mod
    from navigation.providers import google as google_mod

    picks = [("Griffith Observatory", "2800 E Observatory Rd, Los Angeles, CA", "p_obs"),
             ("Griffith Park", "4730 Crystal Springs Dr, Los Angeles, CA", "p_park")]
    provider = ScriptedProvider(picks)
    service.set_provider(provider)

    # -- the endpoint the box actually calls --------------------------------
    res = app_mod.nav_suggest_endpoint(q="griff", lat=34.05, lng=-118.24, session="ac_1")
    got = res["suggestions"]
    ok(len(got) == 2, f"typing three characters returns predictions ({len(got)})")
    ok(got[0]["display_name"] == "Griffith Observatory" and got[0]["provider_place_id"],
       "each one carries a name to show and a place id to resolve")

    # -- THE regression guard ------------------------------------------------
    # The panel and this endpoint have to agree on field names, and when they
    # silently disagreed the dropdown filled with blank rows that routed to
    # `undefined`. So the fields the panel reads are read OUT OF THE PANEL and
    # checked against what the endpoint emits, rather than being restated here
    # where they could drift in step.
    nav_js = open(os.path.join(os.path.dirname(os.path.dirname(
        os.path.abspath(__file__))), "static", "rio_nav.js")).read()
    mapper = re.search(r"function toSuggestion\(c\) \{(.*?)\n    \}", nav_js, re.S)
    ok(bool(mapper), "the panel maps suggestions in one place, toSuggestion()")
    if mapper:
        read = set(re.findall(r"c\.(\w+)", mapper.group(1)))
        emitted = set(got[0].keys())
        missing = read - emitted
        ok(not missing,
           f"every field the panel reads is emitted by /nav/suggest "
           f"({sorted(read)})" if not missing else f"panel reads {sorted(missing)}, "
           f"endpoint emits {sorted(emitted)}")
    ok("if (!id || !(main || secondary)) return null;" in nav_js,
       "a suggestion with no id or nothing to read is dropped, not rendered blank")

    # -- selection resolves through place_id --------------------------------
    provider.resolved = []
    dest = service.get_provider().destination(place_id="p_obs",
                                              label="Griffith Observatory",
                                              session="ac_1")
    ok(isinstance(dest, M.CanonicalDestination) and dest.provider_place_id == "p_obs",
       "picking a prediction resolves by place id into a canonical Destination")
    ok(provider.resolved and provider.resolved[-1]["place_id"] == "p_obs"
       and not provider.resolved[-1]["query"],
       "by id, never by re-searching the text — the same name can be two places")

    # -- the session id reaches both halves ---------------------------------
    provider.sessions_seen = []
    service.resolve_destination("Griffith Observatory", 34.05, -118.24, session="ac_2")
    kinds = [k for k, sid in provider.sessions_seen if sid == "ac_2"]
    ok("suggest" in kinds and "destination" in kinds,
       f"one typing session id spans the predictions and the resolution ({kinds})")

    # -- and Google turns it into a token with a lifecycle -------------------
    before = google_mod.sessions_open()
    t1 = google_mod._session_token("typing_1")
    t2 = google_mod._session_token("typing_1")
    ok(t1 and t1 == t2, "every keystroke of one session shares a provider token")
    ok(google_mod.sessions_open() == before + 1, "and only one token is held for it")
    ok(google_mod._consume_session("typing_1") == t1,
       "the details lookup consumes it — that is the call the session is billed as")
    ok(google_mod.sessions_open() == before,
       "after which nothing is held: the session is over")
    ok(google_mod._session_token("typing_1") != t1,
       "and the next thing typed gets a new token, never the spent one")
    google_mod._consume_session("typing_1")
    ok(google_mod._session_token(None) is None,
       "a resolution with no typing behind it — a spoken destination, a reroute — "
       "opens no session at all")

    # -- autocomplete down: submitting what you typed still works -----------
    broken = ScriptedProvider(picks, fail_suggest=True)
    service.set_provider(broken)
    ok(app_mod.nav_suggest_endpoint(q="griff", session="ac_3")["suggestions"] == [],
       "a provider outage returns no predictions, and no error")
    res = service.resolve_destination("Griffith Observatory", session="ac_3")
    ok(res["status"] == "resolved" and res["reason"] == "geocoded",
       "and the typed destination still resolves, by the path that does not need them")
    ok(broken.resolved and broken.resolved[-1]["query"] == "Griffith Observatory",
       "through the provider's own lookup, on the text the driver actually typed")

    # -- the panel's own wiring, read out of the panel -----------------------
    ok("setTimeout(" in nav_js and "clearTimeout(suggestTimer)" in nav_js,
       "the panel debounces rather than asking on every keystroke")
    ok("'&session=' + encodeURIComponent(session)" in nav_js,
       "the session id rides on every suggest request")
    ok("endSuggestSession();" in nav_js and "session: session" in nav_js,
       "picking a suggestion sends the session id and then ends the session")
    ok("if (typedAt !== suggestSeq) return;" in nav_js,
       "a slow reply for an older prefix is discarded rather than repainting the list")
    ok("routeToQuery(elDest.value.trim())" in nav_js,
       "and Enter still submits whatever is in the box, suggestions or not")


# ---------------------------------------------------------------------------
# Optional: one real provider route
# ---------------------------------------------------------------------------
def run_live():
    section("LIVE — one real route through the production provider")
    service.set_provider(None)
    service.reset()
    provider = service.get_provider()
    print(f"  provider: {provider.name}")
    res = service.resolve_destination("Griffith Observatory", 34.0522, -118.2437)
    ok(res["status"] in ("resolved", "ambiguous"),
       f"a real destination resolves or asks ({res['status']})")
    if res["status"] != "resolved":
        return
    r = service.build_route(34.0522, -118.2437, res["destination"])
    ok(len(r.maneuvers) > 1, f"{len(r.maneuvers)} maneuvers came back")
    ok(len(r.geometry) > 50, f"{len(r.geometry)} geometry points — enough to track against")
    ok(all(m.speech.get("primary") for m in r.maneuvers),
       "every maneuver has a spoken instruction")
    print(f"  landmarks: {r.landmarks_state}, {r.landmark_lookups} lookups")
    for m in r.maneuvers[:6]:
        line = m.speech.get("primary")
        anchor = m.anchors[0]["speech"] if m.anchors else ""
        print(f"    {m.id} {m.type:<10} {m.direction:<8} \"{line}\""
              + (f"   [{anchor}]" if anchor else ""))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--live", action="store_true",
                    help="also compute one route through the production provider")
    args = ap.parse_args()

    run_provider()
    run_destination()
    run_relation()
    run_candidates()
    run_gates()
    run_verification()
    run_speech()
    run_spoken()
    run_observer()
    run_autocomplete()
    if args.live:
        run_live()

    print("\n" + "=" * 72)
    total = len(PASS) + len(FAIL)
    print(f"{len(PASS)}/{total} checks passed")
    if FAIL:
        print("\nFAILED:")
        for f in FAIL:
            print(f"  - {f}")
    print("=" * 72)
    return 1 if FAIL else 0


if __name__ == "__main__":
    sys.exit(main())
