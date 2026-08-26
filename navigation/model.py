"""The canonical navigation model — RIO's own vocabulary for a route.

WHY THIS FILE EXISTS
--------------------
Because "Google returns a route" and "RIO knows where the driver is going" are
two different facts, and only the second one is allowed to spread. Everything
downstream of a NavigationProvider — the route tracker, the landmark lookup,
the visual verifier, the speech planner, the panel — speaks the types in this
file and nothing else. A provider's response shape stops at the provider.

That boundary is what makes provider substitution (§36) an afternoon rather
than a rebuild: another provider means another `NavigationProvider` producing
these same objects. It is also what keeps the licensing question containable —
provider-derived data lives in a known, small set of fields.

WHAT IS AUTHORITATIVE HERE
--------------------------
All of it. Route, maneuver sequence, direction, coordinates, along-route
position, arrival side. Perception may never write to any of it (§2); the one
thing perception contributes anywhere in navigation is a VerifiedAnchor, which
is a *description* attached to a maneuver that already exists and cannot alter
a single field of it.
"""
import time
import uuid
from dataclasses import dataclass, field
from typing import List, Optional

# --- maneuver vocabulary ----------------------------------------------------
# Deliberately small. These are the distinctions that change what RIO SAYS or
# how it is timed; a provider taxonomy with forty entries collapses into this,
# and anything unrecognised becomes TURN/UNKNOWN rather than a new code path.
TURN = "TURN"
MERGE = "MERGE"
RAMP = "RAMP"
FORK = "FORK"
ROUNDABOUT = "ROUNDABOUT"
KEEP = "KEEP"
STRAIGHT = "STRAIGHT"
UTURN = "UTURN"
DEPART = "DEPART"
ARRIVE = "ARRIVE"

LEFT = "LEFT"
RIGHT = "RIGHT"
STRAIGHT_DIR = "STRAIGHT"
UNKNOWN = "UNKNOWN"

# Maneuver families that are worth a landmark. A landmark is a thing the driver
# picks out of a windscreen and turns at; "continue straight" has nothing to
# pick out, and a freeway interchange is out of scope for V1.1 (§21).
ANCHORABLE = {TURN, ROUNDABOUT, UTURN}


@dataclass
class CanonicalDestination:
    display_name: str
    formatted_address: str
    latitude: float
    longitude: float
    provider_place_id: Optional[str] = None

    def to_dict(self) -> dict:
        return {
            "display_name": self.display_name,
            "formatted_address": self.formatted_address,
            "lat": self.latitude,
            "lng": self.longitude,
            "provider_place_id": self.provider_place_id,
        }


@dataclass
class ArrivalInfo:
    """Which side of the road the destination is on — LEFT, RIGHT or UNKNOWN.

    Provider-supplied only. UNKNOWN is a normal, frequent answer and it means
    RIO omits the side rather than guessing one (§28). The camera is never
    asked; a destination side inferred from a dashcam is exactly the kind of
    false certainty this design refuses.
    """
    side: str = UNKNOWN

    def to_dict(self) -> dict:
        return {"side": self.side}


@dataclass
class CanonicalManeuver:
    id: str
    sequence: int
    type: str
    direction: str
    road_name: str
    latitude: float
    longitude: float
    route_distance_position: float      # metres along the route geometry
    polyline_index: int                 # the exact vertex, not a nearest-point guess
    instruction: str = ""               # the provider's own line, for the panel and the log
    step_distance_m: Optional[float] = None
    approach_speed_ms: Optional[float] = None
    lane_information: Optional[dict] = None
    exit_information: Optional[dict] = None
    # Filled in after the route is built, by the landmark candidate generator
    # (V1.1) and the speech planner. Both are OPTIONAL: a maneuver with neither
    # is a maneuver RIO navigates normally.
    anchors: List[dict] = field(default_factory=list)
    speech: dict = field(default_factory=dict)

    @property
    def anchorable(self) -> bool:
        return self.type in ANCHORABLE and self.direction in (LEFT, RIGHT)

    def to_dict(self) -> dict:
        return {
            "id": self.id,
            "sequence": self.sequence,
            "type": self.type,
            "direction": self.direction,
            "road_name": self.road_name,
            "lat": self.latitude,
            "lng": self.longitude,
            "route_distance_position": round(self.route_distance_position, 1),
            "polyline_index": self.polyline_index,
            "instruction": self.instruction,
            "step_distance_m": self.step_distance_m,
            "lane_information": self.lane_information,
            "exit_information": self.exit_information,
            "anchors": self.anchors,
            "speech": self.speech,
        }


@dataclass
class CanonicalRoute:
    """One route generation.

    `generation_id` is the reroute counter for a journey, and it is the thing
    speech validity is checked against (§25). A queued "Right here." from
    generation 1 must never play after a reroute has produced generation 2, and
    comparing route_ids would work only because they happen to differ —
    the generation says what is actually meant: this instruction belongs to a
    plan that no longer exists.
    """
    route_id: str
    journey_id: str
    generation_id: int
    provider: str
    origin_lat: float
    origin_lng: float
    destination: CanonicalDestination
    total_distance_m: float
    duration_s: float
    geometry: List[List[float]]
    maneuvers: List[CanonicalManeuver]
    arrival: ArrivalInfo = field(default_factory=ArrivalInfo)
    depart_instruction: str = ""
    created_at: float = field(default_factory=time.time)
    route_length_m: float = 0.0
    # Set by the landmark generator so the panel and the log can tell "no
    # landmarks near this route" from "we never looked".
    landmarks_state: str = "not_requested"
    landmark_lookups: int = 0

    @property
    def eta_epoch(self) -> float:
        return self.created_at + self.duration_s

    def maneuver(self, maneuver_id: str) -> Optional[CanonicalManeuver]:
        for m in self.maneuvers:
            if m.id == maneuver_id:
                return m
        return None

    def to_dict(self, geometry: bool = True) -> dict:
        out = {
            "route_id": self.route_id,
            "journey_id": self.journey_id,
            "generation_id": self.generation_id,
            "provider": self.provider,
            "created_at": self.created_at,
            "origin": {"lat": self.origin_lat, "lng": self.origin_lng},
            "destination": self.destination.to_dict(),
            "total_distance_m": self.total_distance_m,
            "duration_s": self.duration_s,
            "eta_epoch": self.eta_epoch,
            "route_length_m": round(self.route_length_m, 1),
            "arrival": self.arrival.to_dict(),
            "depart_instruction": self.depart_instruction,
            "landmarks_state": self.landmarks_state,
            "landmark_lookups": self.landmark_lookups,
            "maneuvers": [m.to_dict() for m in self.maneuvers],
        }
        if geometry:
            out["geometry"] = [[round(p[0], 6), round(p[1], 6)] for p in self.geometry]
        return out


def new_journey_id() -> str:
    return uuid.uuid4().hex[:12]


def new_route_id() -> str:
    return uuid.uuid4().hex[:12]
