"""Synthetic routes and landmarks — the whole test surface, offline.

Two jobs:

  1. The test harness. Everything about navigation is assertable from simulated
     GPS and simulated landmark observations (§32), and nothing in the required
     test list needs a network, a camera or a car.
  2. The answer to "the routing API is not enabled on this key yet". RIO is
     developed against fixtures and the provider is swapped in later without a
     line changing anywhere downstream — which is the provider boundary's whole
     claim, exercised rather than asserted.

The geometry is deliberately plain: axis-aligned legs, a vertex every 10 m, so
a projection failure shows up as an obviously wrong number rather than as a
plausible one.
"""
import math
from typing import List, Optional

from . import geo
from . import model as M
from .provider import DestinationCandidate, NavError, NavigationProvider

M_PER_DEG = 111320.0


def _grid(lat0: float, lng0: float):
    k_lng = M_PER_DEG * math.cos(math.radians(lat0))
    return (lambda x, y: [lat0 + y / M_PER_DEG, lng0 + x / k_lng],
            lambda p: ((p[1] - lng0) * k_lng, (p[0] - lat0) * M_PER_DEG))


def city_route(lat0: float = 34.0430, lng0: float = -118.2673,
               leg_m: tuple = (1200.0, 900.0, 500.0)) -> M.CanonicalRoute:
    """An L: east, left onto Lincoln, north, right onto Fell, east, arrive.

    Two anchorable turns and an arrival, which is the smallest route that can
    exercise maneuver ordering, per-maneuver anchoring and arrival together.
    """
    to_ll, _ = _grid(lat0, lng0)
    pts: List[List[float]] = [to_ll(0, 0)]
    x = y = 0.0
    for d in range(10, int(leg_m[0]) + 1, 10):
        pts.append(to_ll(d, 0))
    x = leg_m[0]
    for d in range(10, int(leg_m[1]) + 1, 10):
        pts.append(to_ll(x, d))
    y = leg_m[1]
    for d in range(10, int(leg_m[2]) + 1, 10):
        pts.append(to_ll(x + d, y))

    cum = geo.cumulative(pts)
    i_turn1 = int(leg_m[0] / 10)
    i_turn2 = i_turn1 + int(leg_m[1] / 10)
    i_end = len(pts) - 1

    dest = M.CanonicalDestination(
        display_name="Test Destination", formatted_address="Test Destination, Los Angeles, CA",
        latitude=pts[i_end][0], longitude=pts[i_end][1], provider_place_id="fixture_dest")

    maneuvers = [
        M.CanonicalManeuver(id="m0", sequence=0, type=M.TURN, direction=M.LEFT,
                            road_name="Lincoln Boulevard",
                            latitude=pts[i_turn1][0], longitude=pts[i_turn1][1],
                            route_distance_position=cum[i_turn1], polyline_index=i_turn1,
                            instruction="Turn left onto Lincoln Boulevard",
                            step_distance_m=leg_m[1], approach_speed_ms=11.0),
        M.CanonicalManeuver(id="m1", sequence=1, type=M.TURN, direction=M.RIGHT,
                            road_name="Fell Street",
                            latitude=pts[i_turn2][0], longitude=pts[i_turn2][1],
                            route_distance_position=cum[i_turn2], polyline_index=i_turn2,
                            instruction="Turn right onto Fell Street",
                            step_distance_m=leg_m[2], approach_speed_ms=11.0),
        M.CanonicalManeuver(id="m2", sequence=2, type=M.ARRIVE, direction=M.RIGHT,
                            road_name="", latitude=pts[i_end][0], longitude=pts[i_end][1],
                            route_distance_position=cum[i_end], polyline_index=i_end,
                            instruction="Arrive at Test Destination",
                            step_distance_m=0.0, approach_speed_ms=11.0),
    ]
    return M.CanonicalRoute(
        route_id=M.new_route_id(), journey_id="", generation_id=0, provider="fixture",
        origin_lat=pts[0][0], origin_lng=pts[0][1], destination=dest,
        total_distance_m=cum[-1], duration_s=cum[-1] / 11.0,
        geometry=pts, maneuvers=maneuvers,
        arrival=M.ArrivalInfo(side=M.RIGHT),
        depart_instruction="Head east on Venice Boulevard",
        route_length_m=cum[-1])


def highway_route(lat0: float = 34.0430, lng0: float = -118.2673) -> M.CanonicalRoute:
    """A freeway leg with an exit on it, for the fast announcement ladder.

    The surface fixture cannot exercise the two-mile and one-mile calls at all
    -- its longest leg is 1200 m -- so a route that runs 6 km at 29 m/s and
    leaves at a numbered exit is the smallest thing that can. `road_class` and
    `exit_information` are set the way providers/google._road_class and
    ._exit_information would set them for these steps.
    """
    to_ll, _ = _grid(lat0, lng0)
    pts: List[List[float]] = []
    for d in range(0, 6001, 20):
        pts.append(to_ll(d, 0))
    for d in range(20, 801, 20):
        pts.append(to_ll(6000 + d, d * 0.4))
    cum = geo.cumulative(pts)
    i_exit = 6000 // 20
    i_end = len(pts) - 1
    dest = M.CanonicalDestination(
        display_name="Sunset Plaza", formatted_address="Sunset Plaza, Los Angeles, CA",
        latitude=pts[i_end][0], longitude=pts[i_end][1], provider_place_id="fixture_hwy")
    maneuvers = [
        M.CanonicalManeuver(id="m0", sequence=0, type=M.RAMP, direction=M.RIGHT,
                            road_name="", latitude=pts[i_exit][0],
                            longitude=pts[i_exit][1],
                            route_distance_position=cum[i_exit], polyline_index=i_exit,
                            instruction="Take exit 43 toward Sunset Blvd",
                            road_class=M.HIGHWAY,
                            exit_information={"number": "43", "toward": "Sunset Blvd"},
                            step_distance_m=800.0, approach_speed_ms=29.0),
        M.CanonicalManeuver(id="m1", sequence=1, type=M.ARRIVE, direction=M.LEFT,
                            road_name="", latitude=pts[i_end][0], longitude=pts[i_end][1],
                            route_distance_position=cum[i_end], polyline_index=i_end,
                            instruction="Arrive at Sunset Plaza",
                            step_distance_m=0.0, approach_speed_ms=12.0),
    ]
    return M.CanonicalRoute(
        route_id=M.new_route_id(), journey_id="", generation_id=0, provider="fixture",
        origin_lat=pts[0][0], origin_lng=pts[0][1], destination=dest,
        total_distance_m=cum[-1], duration_s=cum[-1] / 25.0,
        geometry=pts, maneuvers=maneuvers,
        arrival=M.ArrivalInfo(side=M.LEFT),
        depart_instruction="Head north on the 405",
        route_length_m=cum[-1])


def chained_route(lat0: float = 34.0430, lng0: float = -118.2673) -> M.CanonicalRoute:
    """Two turns 120 m apart -- one move to a driver, and one "then" sentence."""
    to_ll, _ = _grid(lat0, lng0)
    pts: List[List[float]] = []
    for d in range(0, 601, 10):
        pts.append(to_ll(d, 0))
    for d in range(10, 121, 10):
        pts.append(to_ll(600, d))
    for d in range(10, 401, 10):
        pts.append(to_ll(600 + d, 120))
    cum = geo.cumulative(pts)
    i1, i2, i_end = 60, 72, len(pts) - 1
    dest = M.CanonicalDestination(
        display_name="Second Street", formatted_address="2nd St, Los Angeles, CA",
        latitude=pts[i_end][0], longitude=pts[i_end][1], provider_place_id="fixture_chain")
    maneuvers = [
        M.CanonicalManeuver(id="m0", sequence=0, type=M.TURN, direction=M.LEFT,
                            road_name="Ocean Ave", latitude=pts[i1][0], longitude=pts[i1][1],
                            route_distance_position=cum[i1], polyline_index=i1,
                            instruction="Turn left onto Ocean Ave",
                            step_distance_m=120.0, approach_speed_ms=11.0),
        M.CanonicalManeuver(id="m1", sequence=1, type=M.TURN, direction=M.RIGHT,
                            road_name="2nd St", latitude=pts[i2][0], longitude=pts[i2][1],
                            route_distance_position=cum[i2], polyline_index=i2,
                            instruction="Turn right onto 2nd St",
                            step_distance_m=400.0, approach_speed_ms=11.0),
        M.CanonicalManeuver(id="m2", sequence=2, type=M.ARRIVE, direction=M.UNKNOWN,
                            road_name="", latitude=pts[i_end][0], longitude=pts[i_end][1],
                            route_distance_position=cum[i_end], polyline_index=i_end,
                            instruction="Arrive at Second Street",
                            step_distance_m=0.0, approach_speed_ms=11.0),
    ]
    return M.CanonicalRoute(
        route_id=M.new_route_id(), journey_id="", generation_id=0, provider="fixture",
        origin_lat=pts[0][0], origin_lng=pts[0][1], destination=dest,
        total_distance_m=cum[-1], duration_s=cum[-1] / 11.0,
        geometry=pts, maneuvers=maneuvers, arrival=M.ArrivalInfo(side=M.UNKNOWN),
        depart_instruction="Head east on Main St", route_length_m=cum[-1])


def place_at(route: M.CanonicalRoute, along_m: float, lateral_m: float,
             name: str, primary_type: str = "gas_station",
             place_id: Optional[str] = None) -> dict:
    """A landmark placed by ROUTE POSITION, which is how a test wants to think.

    `along_m` is where it sits along the route, `lateral_m` how far off the
    line (positive = right of travel). That makes "a Shell 15 m before the
    turn, 12 m off the road" a one-liner, and it means the relation the
    generator computes from the coordinates is checkable against the intent
    they were written with.
    """
    cum = geo.cumulative(route.geometry)
    p = geo.point_at(route.geometry, cum, along_m)
    q = geo.point_at(route.geometry, cum, min(cum[-1], along_m + 5.0))
    brg = geo.bearing_deg(p[0], p[1], q[0], q[1])
    # Offset perpendicular to travel.
    off = math.radians(brg + 90.0)
    k_lng = M_PER_DEG * math.cos(math.radians(p[0]))
    lat = p[0] + (lateral_m * math.cos(off)) / M_PER_DEG
    lng = p[1] + (lateral_m * math.sin(off)) / k_lng
    return {"place_id": place_id or f"fixture_{name.lower().replace(' ', '_')}_{int(along_m)}",
            "name": name, "primary_type": primary_type,
            "types": [primary_type, "point_of_interest"], "lat": lat, "lng": lng}


class FixtureProvider(NavigationProvider):
    """A provider backed by a prepared route and a prepared place list.

    Used by the harness, and usable in production-shaped development when a
    routing key is unavailable — the point being that nothing downstream can
    tell which provider it is talking to.
    """
    name = "fixture"

    def __init__(self, route: Optional[M.CanonicalRoute] = None,
                 places: Optional[List[dict]] = None,
                 fail_with: Optional[str] = None):
        self._route = route or city_route()
        self._places = places or []
        self._fail_with = fail_with
        self.route_calls = 0
        self.landmark_calls = 0
        self.sessions_seen = []

    def suggest(self, query, lat=None, lng=None, limit=5, session=None):
        # Sessions are recorded rather than used: a provider with no session
        # concept still has to accept one, and a test wants to see that the
        # same id arrived on every keystroke and again on the selection.
        if session:
            self.sessions_seen.append(("suggest", session))
        d = self._route.destination
        return [DestinationCandidate(d.display_name, d.formatted_address,
                                     d.provider_place_id, d.latitude, d.longitude)]

    def destination(self, query="", place_id="", label="", session=None):
        if session:
            self.sessions_seen.append(("destination", session))
        return self._route.destination

    # The same three the real map honours, so the offline suites exercise the
    # honoured path and the refused one rather than only the refused one.
    AVOID_SUPPORTED = ("highways", "tolls", "ferries")

    def route(self, origin_lat, origin_lng, destination,
              avoid=None, heading=None):
        # Recorded so a selftest can assert what was ASKED for, which is the
        # only part of a preference this provider can honour.
        self.last_avoid = list(avoid or [])
        self.last_heading = heading
        self.route_calls += 1
        if self._fail_with:
            raise NavError(self._fail_with)
        import copy
        r = copy.deepcopy(self._route)
        r.route_id = M.new_route_id()
        r.origin_lat, r.origin_lng = origin_lat, origin_lng
        return r

    def landmarks_near(self, lat, lng, radius_m, types):
        self.landmark_calls += 1
        out = []
        for p in self._places:
            if geo.haversine_m(lat, lng, p["lat"], p["lng"]) <= radius_m:
                out.append(dict(p))
        return out
