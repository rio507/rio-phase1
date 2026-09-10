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
           ("tier_distances_m", "junction_min_gap_s", "chain_window_m",
            "arrival_call_m", "units",
            "off_route_distance_m", "gps_stale_timeout_s")),
       "every threshold the browser times against ships WITH the route")
    ok(all("tiers" in m["speech"] for m in wire["maneuvers"]),
       "...and each maneuver carries its OWN ladder, so the browser reads a "
       "distance rather than holding a policy")
    ok(wire.get("depart_speech"),
       "...and the route-start line rides on the route, not on a maneuver")

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

        # THE MINIMUM CONTRACT, and it now includes taking the two arguments
        # this provider cannot do anything with. `avoid` arrives already
        # filtered against AVOID_SUPPORTED — which is empty here — so it is
        # always the empty list; `heading` is a hint a provider is free to
        # ignore. Accepting and dropping them is what a map with no such
        # controls is supposed to do.
        def route(self, origin_lat, origin_lng, destination,
                  avoid=None, heading=None):
            r = fixtures.city_route()
            r.origin_lat, r.origin_lng = origin_lat, origin_lng
            return r

    service.set_provider(Bare())
    service.reset()
    bare = service.build_route(34.043, -118.267, Bare().destination())
    ok(bare.landmarks_state == "ready" and
       all(not m.anchors for m in bare.maneuvers),
       "a provider with no place data routes normally with no anchors at all")
    ok(all(m.speech.get("near") or m.speech.get("arrival")
           for m in bare.maneuvers),
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
    line = m0.anchors[0]["speech"]
    ok("left" in line.lower() and "the Shell station" in line
       and "Lincoln Boulevard" in line,
       "the best candidate's line names the turn, the landmark and the road, "
       "whichever phrasing it was drawn at — got " + repr(line))
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
        ok(all(m.speech.get("near") or m.speech.get("arrival")
               for m in r3.maneuvers),
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
                              "visibility_confidence", "valid_for_m"},
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
    ok(m0.speech["far"] == "In half a mile, turn left onto Lincoln Boulevard.",
       "the far call is distance-phrased and names the road — got "
       + repr(m0.speech["far"]))
    ok(m0.speech["near"] == "Turn left onto Lincoln Boulevard.",
       "the near call is the instruction, with no hedging — got "
       + repr(m0.speech["near"]))
    ok(m0.speech["junction"] == "Turn left.",
       "the junction call is two words — got " + repr(m0.speech["junction"]))
    ok("coming up" not in " ".join(str(v) for v in m0.speech.values()).lower(),
       'nothing anywhere still says "coming up"')
    ok("take the next" not in " ".join(str(v) for v in m0.speech.values()).lower(),
       '...nor "take the next", which is not what Google says at 150 m')
    ok(r.depart_speech ==
       "Head east on Venice Boulevard, then turn left onto Lincoln Boulevard.",
       "the route-start line is the whole first move — got "
       + repr(r.depart_speech))

    # --- THE DEPART STEP'S TWO SHAPES, AND THE ONE THAT SAID A ROAD TWICE ---
    #
    # Google's DEPART step names either the road you are ON ("Head north on
    # Lincoln Blvd") or the road you are AIMED AT, which is what it says when
    # the road you are on has no name ("Head northeast toward 16th St"). The
    # second one names the road the first maneuver turns onto, so chaining it
    # says that road twice. Measured on a live route to Griffith Observatory
    # (tools/live_tool_turns.py --script nav, 2026-09-10):
    #
    #     "Head northeast toward 16th St, then turn left onto 16th St."
    #
    # The rule is a comparison, not a rewrite: the "toward" clause is dropped
    # only when its road IS the road the next instruction turns onto.
    def depart_for(depart_instruction, road, kind=M.TURN, direction=M.LEFT):
        man = M.CanonicalManeuver(
            id="m0", sequence=0, type=kind, direction=direction, road_name=road,
            latitude=0.0, longitude=0.0, route_distance_position=100.0,
            polyline_index=0, instruction="")
        return speech_mod.depart_text(M.CanonicalRoute(
            route_id="r", journey_id="j", generation_id=1, provider="fixture",
            origin_lat=0.0, origin_lng=0.0,
            destination=M.CanonicalDestination("D", "D", 0.0, 0.0),
            total_distance_m=0.0, duration_s=0.0, geometry=[], maneuvers=[man],
            depart_instruction=depart_instruction))

    same = depart_for("Head northeast toward 16th St", "16th St")
    ok(same == "Head northeast, then turn left onto 16th St.",
       "same road: the toward clause is dropped and the name lands on the "
       "instruction that acts on it — got " + repr(same))
    ok(same.lower().count("16th st") == 1,
       "...so the road is said once, not twice")

    other = depart_for("Head northeast toward 16th St", "Ocean Ave",
                       direction=M.RIGHT)
    ok(other == "Head northeast toward 16th St, then turn right onto Ocean Ave.",
       "different roads: both are kept, because 'toward 16th St' is then real "
       "information about the way out — got " + repr(other))

    on_form = depart_for("Head north on Lincoln Blvd", "Ocean Ave",
                         direction=M.RIGHT)
    ok(on_form == "Head north on Lincoln Blvd, then turn right onto Ocean Ave.",
       "the 'on <road>' shape is never touched — it names the road under the "
       "car, which is different information — got " + repr(on_form))

    # Punctuation and case only. No abbreviation table: guessing that "St" and
    # "Street" are the same road is a guess, and a wrong one drops a road name
    # the driver needed.
    ok(depart_for("Head northeast toward 16TH ST.", "16th St")
       == "Head northeast, then turn left onto 16th St.",
       "the comparison survives case and punctuation")
    ok(depart_for("Head northeast toward 16th Street", "16th St")
       == "Head northeast toward 16th Street, then turn left onto 16th St.",
       "and stops there — a spelling this cannot prove is the same road keeps "
       "both halves rather than dropping one")

    # A head that is NOTHING but the toward clause has nothing left to say.
    ok(depart_for("Toward 16th St", "16th St")
       == "Toward 16th St, then turn left onto 16th St.",
       "a head with no direction in it is kept whole — a bare 'then turn left' "
       "came from nowhere")
    ok(arrive.speech["arrival"] == "Your destination is on the right.",
       "arrival says the side the provider gave")
    ok(arrive.speech["arrived"] == "You have arrived.",
       "and the line at the kerb is Google's own words")

    plain = M.CanonicalManeuver(id="x", sequence=0, type=M.TURN, direction=M.RIGHT,
                                road_name="", latitude=0, longitude=0,
                                route_distance_position=0, polyline_index=0,
                                instruction="Turn right")
    ok(speech_mod.near_text(plain) == "Turn right.",
       'with no road name the near call is exactly "Turn right."')
    ok(speech_mod.arrival_text("The Getty", M.UNKNOWN) ==
       "Your destination is ahead.",
       "an UNKNOWN arrival side says ahead, never a guessed side")

    ok(m0.anchors[0]["speech"] in [
        t.format(dir="left", Dir="Left", label="the Shell station",
                 road=" onto Lincoln Boulevard")
        for t in speech_mod._CONTEXTUAL["NEAR"]],
       "the contextual line is prepared at route load, not composed while "
       "driving — and is one of the NEAR phrasings")

    # Every sentence is addressable, and only by id.
    ok(speech_mod.text_for(r, "m0", "near") == m0.speech["near"],
       "/nav/voice resolves (route, maneuver, call) to the stored line")
    ok(speech_mod.text_for(r, "m0", "near", m0.anchors[0]["anchor_id"]) ==
       m0.anchors[0]["speech"],
       "...and (route, maneuver, call, anchor) to the contextual one, the "
       "same string that was stored — the choice was made once, at build")
    ok(speech_mod.text_for(r, "m0", "near", "not_a_real_anchor") is None,
       "an anchor that is not on this route is not a sentence RIO can say")
    ok(speech_mod.text_for(r, "m99", "near") is None,
       "nor is a maneuver that is not on it")
    ok(speech_mod.text_for(r, "m0", "freestyle") is None,
       "nor is a call type that does not exist")
    # A PHONE RUNNING YESTERDAY'S CACHED PAGE still asks for "imminent". It
    # gets the junction line rather than silence at a junction.
    ok(speech_mod.text_for(r, "m0", "imminent") == m0.speech["junction"],
       "the old call names still resolve, so a stale client is not mute")
    ok(speech_mod.text_for(r, "m0", "primary") == m0.speech["near"],
       "...for all three of them")

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
# G1b. THE LADDER — Google's distances, by road class
# ---------------------------------------------------------------------------
def run_cadence():
    section("G1b. cadence — the ladder a driver already knows")

    surface = fixtures.city_route()
    service.set_provider(fixtures.FixtureProvider(surface, []))
    service.reset()
    rs = service.build_route(surface.geometry[0][0], surface.geometry[0][1],
                             surface.destination)
    m0 = rs.maneuvers[0]
    tiers = {t["call"]: t["at_m"] for t in m0.speech["tiers"]}
    ok(m0.road_class == M.SURFACE, "a city street is on the surface ladder")
    ok(abs(tiers["far"] - 804.7) < 1,
       f"the far call is half a mile out ({tiers['far']} m)")
    ok(tiers.get("far_mid") is None,
       "and there is no one-mile call on a surface street")
    ok(tiers["near"] == 150.0, f"the near call is at 150 m ({tiers['near']} m)")
    ok(tiers["junction"] == 35.0, "and the junction floor is 35 m")

    hwy = fixtures.highway_route()
    service.set_provider(fixtures.FixtureProvider(hwy, []))
    service.reset()
    rh = service.build_route(hwy.geometry[0][0], hwy.geometry[0][1],
                             hwy.destination)
    h0 = rh.maneuvers[0]
    ht = {t["call"]: t["at_m"] for t in h0.speech["tiers"]}
    ok(h0.road_class == M.HIGHWAY, "a 29 m/s ramp approach is on the fast ladder")
    ok(abs(ht["far"] - 3218.7) < 1, f"the far call is two miles out ({ht['far']} m)")
    ok(abs(ht["far_mid"] - 1609.3) < 1,
       f"...with a one-mile call under it ({ht['far_mid']} m)")
    ok(abs(ht["near"] - 402.3) < 1, f"the near call is a quarter mile ({ht['near']} m)")
    ok(h0.speech["far"] == "In two miles, take exit 43 toward Sunset Blvd.",
       "and it names the exit and the sign — got " + repr(h0.speech["far"]))
    ok(h0.speech["far_mid"] == "In one mile, take exit 43 toward Sunset Blvd.",
       "the one-mile call says one mile — got " + repr(h0.speech["far_mid"]))
    ok(h0.speech["near"] == "Take exit 43 toward Sunset Blvd.",
       "and the near call drops the distance, not the exit number")
    ok(h0.speech["junction"] == "Take the exit.", "the junction call is two words")

    # THE EXIT NUMBER IS PROVIDER DATA OR IT IS NOT SAID.
    no_number = M.CanonicalManeuver(
        id="x", sequence=0, type=M.RAMP, direction=M.RIGHT, road_name="the 10",
        latitude=0, longitude=0, route_distance_position=0, polyline_index=0,
        instruction="Take the ramp onto the 10", road_class=M.HIGHWAY)
    ok("exit " not in speech_mod.near_text(no_number).lower()
       or "exit number" not in speech_mod.near_text(no_number).lower(),
       "a ramp with no exit number never invents one — "
       + repr(speech_mod.near_text(no_number)))

    # CHAINING: two junctions 120 m apart are one sentence.
    ch = fixtures.chained_route()
    service.set_provider(fixtures.FixtureProvider(ch, []))
    service.reset()
    rc = service.build_route(ch.geometry[0][0], ch.geometry[0][1], ch.destination)
    c0, c1 = rc.maneuvers[0], rc.maneuvers[1]
    ok(c0.speech["near"] == "Turn left onto Ocean Ave, then turn right onto 2nd St.",
       'the near call chains the second turn with "then" — got '
       + repr(c0.speech["near"]))
    ok(c0.speech.get("chained_to") == c1.id,
       "and the maneuver it swallowed is named, so the planner can stay quiet "
       "about it")
    ok("far" not in c0.speech,
       "a 600 m leg gets NO half-mile call — the distance in that sentence is "
       "written at route load and cannot be said 600 m out")
    # ...and where there IS a far call, it does not chain: half a mile out the
    # second turn is not yet a thing the driver can act on.
    far_man = rs.maneuvers[1]          # 900 m of Lincoln before the Fell turn
    ok("far" in far_man.speech and "then" not in far_man.speech["far"],
       "the far call does NOT chain — " + repr(far_man.speech.get("far")))
    # ...and a turn far enough away is not chained.
    ok(rs.maneuvers[0].speech.get("chained_to") is None,
       "two turns 900 m apart stay two announcements")
    # THE LADDER IS TRIMMED TO THE LEG, which is the rule that stops a
    # half-mile call on a 400 m block. Session a2da65cd's replay produced two
    # of exactly those, at 414 m and 485 m against an 805 m tier.
    short = [m for m in rc.maneuvers if m.type == M.TURN]
    ok(all("far" not in m.speech for m in short),
       "no maneuver on a route of short blocks carries a far call at all")


def run_distance_phrasing():
    section("G1c. distance — the closed set of phrases, and the roundings")
    from navigation import distance as dist_mod

    cases = [
        (30, "100 feet"), (46, "150 feet"), (76, "250 feet"),
        (152, "500 feet"), (160, "500 feet"),
        (300, "a quarter mile"), (402, "a quarter mile"),
        (700, "half a mile"), (805, "half a mile"),
        (1207, "three quarters of a mile"),
        (1609, "one mile"), (3218, "two miles"), (4828, "three miles"),
    ]
    for meters, want in cases:
        got = dist_mod.phrase(meters)
        ok(got == want, f"{meters} m -> {want!r}" + ("" if got == want else f" (got {got!r})"))

    ok(dist_mod.in_phrase(805) == "In half a mile",
       "and it composes into a sentence opener")

    # NOTHING BELOW 100 FEET, ever: at 60 ft the driver is in the junction and
    # a number is worse than the two words that belong there.
    ok(dist_mod.phrase(10) == "100 feet",
       "nothing is ever announced closer than a hundred feet")

    # SPOKEN, NOT PRINTED.
    ok("2 miles" not in dist_mod.phrase(3218),
       'the mile count is a word, not a digit — "two miles", never "2 miles"')

    # METRIC IS ITS OWN LADDER, not the imperial one converted.
    metric = [(300, "300 meters"), (805, "800 meters"), (1609, "two kilometers"),
              (1000, "one kilometer"), (60, "50 meters")]
    for meters, want in metric:
        got = dist_mod.phrase(meters, dist_mod.METRIC)
        ok(got == want, f"metric: {meters} m -> {want!r}"
           + ("" if got == want else f" (got {got!r})"))

    ok(len(dist_mod.announce_ladder()) <= 12,
       "the set of phrases is closed and small — "
       + str(len(dist_mod.announce_ladder())))

    # ...AND THE CONFIG SWITCH REACHES THE ROUTE.
    route = fixtures.city_route()
    service.set_provider(fixtures.FixtureProvider(route, []))
    service.reset()
    old = config.NAV_UNITS
    try:
        config.NAV_UNITS = "metric"
        rm = service.build_route(route.geometry[0][0], route.geometry[0][1],
                                 route.destination)
        ok("meters" in rm.maneuvers[0].speech["far"]
           or "kilometer" in rm.maneuvers[0].speech["far"],
           "NAV_UNITS = metric changes what the far call says — "
           + repr(rm.maneuvers[0].speech["far"]))
    finally:
        config.NAV_UNITS = old


# ---------------------------------------------------------------------------
# G1d. SHE IS THE NAVIGATION — no third person anywhere on the spoken path
# ---------------------------------------------------------------------------
# On the drive of 2026-09-09 a driver asked for directions and was told the
# navigation would handle it. There is no "the navigation": the turn calls are
# RIO, and the fact that a deterministic planner fires them is architecture,
# not something the driver is told about.
#
# The lint runs over everything that can reach the driver's ears or the model's
# context: every sentence the speech table can produce, and every string a nav
# tool result can carry. A model that is handed "the system will call it out"
# will say it back.
THIRD_PERSON = (
    "the car will", "the car's", "the vehicle will", "the car announces",
    "the car does", "the system", "the navigation system", "navigation will",
    "the nav system", "the gps will", "it will call", "you'll hear",
    "will call it out", "will let you know", "will tell you",
    "the turn-by-turn",
)

# A NEGATIVE EXAMPLE IS NOT A VIOLATION, and the instructions are full of them
# on purpose: never "the car will tell you" is the sentence that stops the
# model saying it. A substring lint that cannot tell the two apart forces the
# instructions to stop naming the failure they are preventing, which is the
# one thing they most need to do.
#
# TWO CONDITIONS, BOTH REQUIRED, and one of them alone is not enough:
#
#   the phrase is QUOTED     an instruction names a forbidden wording by
#                            quoting it; a sentence that USES the wording does
#                            not put it in quotes.
#   the sentence NEGATES     "never", "not", "n't" somewhere in it.
#
# Either on its own lets real failures through. "Don't worry about it, the
# navigation system will tell you." negates and is exactly the sentence this
# exists to catch, which is why the quote test is not optional -- it was the
# first thing this lint got wrong.
_NEGATORS = ("never", "not ", "n't", "must not", "instead of", "rather than",
             "stop saying")
# Straight and curly, and each delimiter matched against ITSELF -- a single
# character class spanning both kinds ends "you'll hear it" at the apostrophe
# and loses the quotation it was meant to find.
_QUOTED = (re.compile(r'"([^"]{3,160})"'),
           re.compile(r"'([^']{3,160})'"),
           re.compile("\u201c([^\u201d]{3,160})\u201d"))


def _norm(text: str) -> str:
    """Lower case, and every apostrophe the same apostrophe.

    The live run of 2026-09-10 answered "You\u2019ll hear the turns as they come
    up" and this lint missed it: a TTS transcript uses U+2019 and the phrase
    list uses U+0027. A lint that only fires on straight quotes never fires on
    anything a model actually said.
    """
    return (text or "").lower().replace("\u2019", "'").replace("\u02bc", "'")


def _sentences(text: str):
    """Split on sentence ends, EXCEPT inside quotation marks.

    A quoted sentence is one unit. The instructions quote the wording they are
    forbidding, verbatim and with its own full stop -- and a splitter that cuts
    inside the quotation hands the lint a fragment with the forbidden phrase in
    it and the "never" left behind in the sentence before. That is a lint that
    fails on the instruction telling the model not to do the thing, which is
    the one sentence that has to be allowed to exist.
    """
    # HARD WRAPPING IS NOT PUNCTUATION. The instructions are prose wrapped at
    # 78 columns, so a single newline falls in the middle of sentences -- and a
    # splitter that treats it as a sentence end cuts 'which is\nnot a sentence
    # you say' in half and leaves the forbidden quotation in the half without
    # the "not". Blank lines still separate; single ones are spaces.
    text = re.sub(r"(?<!\n)\n(?!\n)", " ", text or "")
    out, buf, quote = [], [], None
    for ch in text:
        buf.append(ch)
        if quote:
            if ch == quote:
                quote = None
            continue
        if ch in "\"'\u201c\u2018":
            # An apostrophe inside a word ("don't") is not an opening quote.
            prev = buf[-2] if len(buf) > 1 else " "
            if ch == "'" and prev.isalnum():
                continue
            quote = {"\u201c": "\u201d", "\u2018": "\u2019"}.get(ch, ch)
            continue
        if ch in ".!?\n":
            out.append("".join(buf))
            buf = []
    if buf:
        out.append("".join(buf))
    return out


def nav_voice_lint(text: str):
    """Which forbidden third-person phrases this text ASSERTS, if any."""
    hits = []
    for sentence in _sentences(text):
        low = _norm(sentence)
        negates = any(n in low for n in _NEGATORS)
        quoted = " ".join(_norm(m.group(1))
                          for rx in _QUOTED for m in rx.finditer(sentence))
        for p in THIRD_PERSON:
            if p not in low:
                continue
            if negates and p in quoted:
                continue          # named in order to be forbidden
            hits.append(p)
    return sorted(set(hits))


# EVERY STRING A NAV TOOL RESULT CAN CARRY, read out of the file that writes
# them rather than listed here.
#
# The results are built in static/rio_realtime.js, in JavaScript, so they
# cannot be called from this suite -- and a hand-kept list of their sentences
# is a list that stops covering the result someone adds tomorrow. What CAN be
# done exactly is to read the source and pull every `rules:` and `note:`
# string out of the navigation tools, which is what the model actually reads.
_RESULT_KEYS = ("rules", "note")


def _nav_tool_result_strings():
    """[(where, text)] for every rules/note string in the nav tool handlers."""
    src = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                       "static", "rio_realtime.js")
    text = open(src, encoding="utf-8").read()
    # The navigation tools are the block from navStatus() to the LOCAL_TOOLS
    # table; everything in it is a navigation answer.
    start = text.find("function navStatus(")
    end = text.find("var LOCAL_TOOLS")
    if start < 0 or end < 0:
        return []
    block = text[start:end]

    out = []
    for key in _RESULT_KEYS:
        for m in re.finditer(key + r"\s*:\s*", block):
            i = m.end()
            if i >= len(block) or block[i] not in "'\"":
                continue
            # A concatenated string literal: 'a ' + 'b ' + 'c'. Walk it.
            parts, j = [], i
            while j < len(block) and block[j] in "'\"":
                quote = block[j]
                j += 1
                buf = []
                while j < len(block) and block[j] != quote:
                    if block[j] == "\\":
                        j += 1
                        if j < len(block):
                            buf.append(block[j])
                    else:
                        buf.append(block[j])
                    j += 1
                j += 1
                parts.append("".join(buf))
                # Skip whitespace and a '+' to reach the next literal.
                k = j
                while k < len(block) and block[k] in " \t\r\n":
                    k += 1
                if k < len(block) and block[k] == "+":
                    k += 1
                    while k < len(block) and block[k] in " \t\r\n":
                        k += 1
                    j = k
                else:
                    break
            line = block[:m.start()].count("\n") + 1
            out.append((f"rio_realtime.js:{line}:{key}", "".join(parts)))
    return out


# THE SENTENCES THAT WERE ACTUALLY THERE, kept as the lint's own proof.
#
# A lint that has never failed is a lint nobody can trust, and these are not
# invented examples: the first two are verbatim from the session instructions
# and the nav tool results as they stood on the drive of 2026-09-09, and the
# last two are what a model handed them says to a driver.
_KNOWN_BAD = (
    "The car announces things itself — turn instructions, health warnings.",
    "the navigation system does that itself, at the moment it matters.",
    "The car will call it out when you get there.",
    "Don't worry about it, the navigation system will tell you.",
    # From the live run of 2026-09-10, verbatim, curly apostrophe and all --
    # the answer to "do I need to watch the screen for the directions?"
    "No. You\u2019ll hear the turns as they come up.",
)


def run_nav_voice():
    section("G1d. voice — RIO is the navigation, not a bystander to it")

    # --- THE LINT CATCHES WHAT WAS ACTUALLY SAID ----------------------------
    missed = [t for t in _KNOWN_BAD if not nav_voice_lint(t)]
    ok(not missed,
       "the lint fails on the sentences that were really there"
       + (f" — MISSED: {missed}" if missed else f" ({len(_KNOWN_BAD)} of them)"))
    # ...and does NOT fail on the instruction that forbids them, which is the
    # sentence the instructions most need to be allowed to contain.
    ok(not nav_voice_lint(
        'Never "the car will tell you", never "the navigation system will '
        'call it out", never "you\'ll hear it".'),
       "and not on the instruction that forbids them, quoted in full")

    route = fixtures.city_route()
    service.set_provider(fixtures.FixtureProvider(route, []))
    service.reset()
    r = service.build_route(route.geometry[0][0], route.geometry[0][1],
                            route.destination)

    # --- EVERY SPOKEN NAVIGATION LINE ---------------------------------------
    lines = [r.depart_speech]
    for m in r.maneuvers:
        for _k, v in m.speech.items():
            if isinstance(v, str):
                lines.append(v)
    bad = [(ln, nav_voice_lint(ln)) for ln in lines if nav_voice_lint(ln)]
    ok(not bad, "no spoken navigation line describes navigation in the third "
                "person" + (f" — {bad}" if bad else f" ({len(lines)} lines)"))

    # The destination reply, which is the exact sentence the drive of
    # 2026-09-09 got wrong.
    reply = speech_mod.destination_reply("resolved", "Century City")
    ok(not nav_voice_lint(reply),
       "the routing confirmation is in her own voice — " + repr(reply))
    ok("i'll take you" in reply.lower(),
       "...and it is first person — " + repr(reply))

    # --- EVERY NAV TOOL RESULT ----------------------------------------------
    results = _nav_tool_result_strings()
    ok(len(results) >= 8,
       f"the lint can see the navigation tool results at all ({len(results)} "
       f"strings)")
    offenders = [(where, hits) for where, txt in results
                 for hits in [nav_voice_lint(txt)] if hits]
    ok(not offenders,
       "no navigation tool result hands the model a third-person sentence to "
       "parrot" + (f" — {offenders}" if offenders
                   else f" ({len(results)} results checked)"))
    firsts = [w for w, txt in results
              if re.search(r"\b(i|i'?ve|i'?ll|you call|you are|yourself)\b",
                           txt.lower())]
    ok(len(firsts) >= max(1, len(results) // 3),
       f"and the ones that describe who is speaking put her in it "
       f"({len(firsts)}/{len(results)})")

    # --- THE SESSION INSTRUCTIONS THE MODEL IS ACTUALLY GIVEN ---------------
    #
    # The composed text, not the source file: comments in realtime.py QUOTE the
    # old sentences on purpose, as the record of what was wrong, and a lint
    # that reads the source cannot tell a quotation from an instruction.
    import realtime
    instr = realtime.instructions()
    hits = nav_voice_lint(instr)
    ok(not hits, "the session instructions never model that phrasing for her"
                 + (f" — {hits}" if hits else f" ({len(instr)} chars)"))
    low = instr.lower()
    ok("you call the turns" in low,
       "...and they say plainly that the turn calls are hers")
    ok("i'll call each turn as we get there" in low,
       "...with the sentence to use when a driver asks who is calling them")
    # The internal boundary is architecture and must not have been softened to
    # make room for the voice change.
    ok("you answer. you do not announce." in low,
       "the announce/answer boundary is still there, in those words")
    ok("never announce a turn" in low,
       "including the turn rule it has always had")

    # --- AND THE LIVE RIG LOOKS FOR THE SAME WORDS --------------------------
    #
    # tools/live_tool_turns.py asks "who's calling the turns?" out loud on a
    # real session, which is the only place the model's own answer can be
    # seen. It carries its own copy of this list because it runs against a
    # live server rather than importing this suite -- so the two are checked
    # against each other here, or a phrase added to one quietly stops being
    # looked for by the other.
    from tools import live_tool_turns as ltt
    ok(set(ltt._THIRD_PERSON) == set(THIRD_PERSON),
       "the live rig looks for exactly these phrases too"
       + ("" if set(ltt._THIRD_PERSON) == set(THIRD_PERSON)
          else f" — drift: {set(THIRD_PERSON) ^ set(ltt._THIRD_PERSON)}"))
    ok(any(t.get("lint") == "third_person" for t in ltt.NAV_SCRIPT),
       "and the live nav script actually asks the question")
    ok(any("calling the turns" in (t.get("say") or "").lower()
           for t in ltt.NAV_SCRIPT),
       '...in those words — "Who\'s calling the turns?"')


def run_variation():
    section("G2. phrasing — fixed where a driver relies on it, varied where she owns it")
    import persona
    route = fixtures.city_route()
    turn = route.maneuvers[0].route_distance_position
    service.set_provider(fixtures.FixtureProvider(
        route, [shell_near_turn(route, turn - 14, 12, "Shell"),
                shell_near_turn(route, route.maneuvers[1].route_distance_position - 14,
                                12, "Chevron")]))
    service.reset()
    r = service.build_route(route.geometry[0][0], route.geometry[0][1],
                            route.destination)
    turns = [m for m in r.maneuvers if m.type == M.TURN]

    # THE CONTENT, which is the half that may not vary. A call that drops the
    # direction leaves the driver guessing, and one that drops the road name is
    # worse than the template it replaced.
    for m in turns:
        word = "left" if m.direction == M.LEFT else "right"
        for call in ("far", "near"):
            line = m.speech.get(call, "")
            ok(word in line.lower(),
               f"{m.id}'s {call} call says which way to turn ({line!r})")
            ok(m.road_name in line,
               f"...and which road it goes onto ({m.road_name!r} in {line!r})")
            ok(not persona.lint(line),
               f"...and is in her register: {persona.lint(line) or 'clean'}")

    # THE NAVIGATION CALLS NO LONGER VARY, and that is the change. A driver
    # parses "In half a mile, turn left onto Lincoln Boulevard." without
    # listening to it; a synonym for it is a sentence they have to listen to.
    forms = set()
    for gen in range(1, 8):
        alt = speech_mod.build(turns[0], "", M.UNKNOWN, variant=gen)
        forms.add((alt["far"], alt["near"], alt["junction"]))
    ok(len(forms) == 1,
       "every tier is byte-identical across every variant index — "
       + str(len(forms)) + " form(s)")

    # ...AND THE ANCHOR LINE STILL DOES, because it is the one sentence in
    # navigation that is hers rather than the map's.
    anchored = [m for m in turns if m.anchors]
    ok(len(anchored) >= 2, "the fixture put a landmark at two turns")
    if len(anchored) >= 2:
        ok(anchored[0].anchors[0]["speech"] != anchored[1].anchors[0]["speech"]
           or anchored[0].anchors[0]["label"] != anchored[1].anchors[0]["label"],
           "consecutive anchored turns are not phrased identically ("
           + " / ".join(repr(m.anchors[0]["speech"]) for m in anchored) + ")")

    # REPRODUCIBLE FROM WHAT IS LOGGED, which is the property a bug report
    # needs: (journey, generation, sequence) is written to the drive log, and
    # those three redraw the sentence exactly.
    mans = r.maneuvers
    for i, m in enumerate(mans):
        nxt = mans[i + 1] if i + 1 < len(mans) else None
        chained = nxt if (m.speech.get("chained_to")
                          and nxt and nxt.id == m.speech["chained_to"]) else None
        ok(speech_mod.build(m, r.destination.display_name, r.arrival.side,
                            variant=speech_mod.variant_for(r, m),
                            chained=chained) == m.speech,
           f"{m.id}'s stored lines are what its journey, generation and "
           "position redraw — nothing decided at drive time")

    # ...and a REROUTE is a new generation, which the anchor lines are allowed
    # to sound different for: the plan changed.
    offsets = {speech_mod.route_offset("j-abc", g) for g in (1, 2, 3)}
    ok(len(offsets) == 3,
       "each generation of a journey starts at its own place in the set")


def run_preferences():
    section("G3. preferences — what the map can keep off a route, and what it cannot")
    route = fixtures.city_route()
    prov = fixtures.FixtureProvider(route, [])
    service.set_provider(prov)
    service.reset()

    wanted, unsupported = service.split_preferences(
        ["highways", "the scenic way", "TOLLS", ""])
    ok(wanted == ["highways", "tolls"],
       "what the provider advertises is honoured, case and blanks aside")
    ok(unsupported == ["the scenic way"],
       "and what it does not is carried OUT as data rather than dropped — "
       "the driver gets told, instead of assuming it was done")

    r = service.build_route(route.geometry[0][0], route.geometry[0][1],
                            route.destination, avoid=wanted, heading=91.5)
    ok(prov.last_avoid == ["highways", "tolls"],
       "the preferences reach the provider that has to honour them")
    ok(prov.last_heading == 91.5,
       "and so does the heading, so a reroute does not open with a U-turn")
    ok(r.generation_id == 1, "an ordinary route is still generation 1")

    # A provider with no such controls is not asked for them, and says so by
    # advertising nothing rather than by failing a call.
    class NoPrefs(fixtures.FixtureProvider):
        AVOID_SUPPORTED = ()
    service.set_provider(NoPrefs(route, []))
    wanted2, unsupported2 = service.split_preferences(["highways"])
    ok(wanted2 == [] and unsupported2 == ["highways"],
       "against a map with no preference controls, everything asked for is "
       "reported back as something it cannot do")

    # The real provider's own table, checked rather than assumed: these are
    # the three the Routes API has modifiers for.
    from navigation.providers.google import GoogleProvider
    ok(set(GoogleProvider.AVOID_SUPPORTED) == {"highways", "tolls", "ferries"},
       "and the production provider advertises exactly what it can modify")


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
       == "Got it — I'll take you to Griffith Observatory.",
       "a resolved destination is confirmed with the provider's own name, in "
       "her own voice")
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

    # -- and the browser cannot run a stale copy of it ----------------------
    # This is the other half of the same failure. A panel from before a change,
    # talking to endpoints from after it, is two different programs — and it
    # presents as the server being broken. The page stamps every local asset
    # with the file's own mtime, so a changed file is a changed URL.
    page = app_mod._stamp_assets(open(os.path.join(os.path.dirname(os.path.dirname(
        os.path.abspath(__file__))), "static", "index.html")).read())
    stamped = re.findall(r'src="(/static/[^"]+)"', page)
    ok(stamped and all("?v=" in u for u in stamped),
       f"every script the page loads is stamped with its version ({len(stamped)} of them)")
    ok(any("rio_nav.js?v=" in u for u in stamped),
       "the navigation panel included — the file this whole section is about")
    unchanged = app_mod._stamp_assets('<script src="/static/does_not_exist.js"></script>')
    ok("?v=" not in unchanged,
       "a missing asset is left alone, so it 404s visibly instead of being hidden")


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
    run_cadence()
    run_distance_phrasing()
    run_nav_voice()
    run_variation()
    run_preferences()
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
