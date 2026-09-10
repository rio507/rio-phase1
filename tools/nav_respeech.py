"""Rebuild a logged route's speech table with today's phrasing.

    cat route.json | python -m tools.nav_respeech

Reads the `maneuvers` block of a NAV_ROUTE_STARTED payload on stdin and writes
back the same maneuvers with a `speech` table and a `road_class` built by the
CURRENT navigation/speech.py, plus the route's own start line.

WHY THIS EXISTS AT ALL, rather than the replay harness holding its own copy of
the templates: tools/nav_drive_replay.js drives a REAL recorded drive back
through the tracker and the planner, and the drive it replays was recorded
against whatever phrasing was live that day. Replaying it against the sentences
in the log would test the cadence that has already been replaced.

So the geometry, the speeds and the fix timeline stay exactly as they were
recorded, and only the WORDS are rebuilt — by the one module that writes them,
asked over a pipe, so there is no second table to drift.

The maneuvers in a log carry type, direction, road name and position and
nothing else, which is precisely the input speech.build needs. What a log does
NOT carry is `approach_speed_ms`, so road class is recomputed from the drive's
own recorded speed where the caller supplies one and defaults to SURFACE
otherwise — stated here because a replay that silently called a city street a
freeway would be a two-mile announcement nobody could account for.
"""
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from navigation import model as M          # noqa: E402
from navigation import speech as speech_mod  # noqa: E402


def _maneuver(d: dict, road_class: str) -> M.CanonicalManeuver:
    return M.CanonicalManeuver(
        id=d.get("id", ""), sequence=int(d.get("sequence") or 0),
        type=d.get("type") or M.TURN, direction=d.get("direction") or M.UNKNOWN,
        road_name=d.get("road_name") or "",
        latitude=float(d.get("lat") or 0.0), longitude=float(d.get("lng") or 0.0),
        route_distance_position=float(d.get("route_distance_position") or 0.0),
        polyline_index=int(d.get("polyline_index") or 0),
        instruction=d.get("instruction") or "",
        road_class=road_class,
        exit_information=d.get("exit_information"),
    )


def respeech(payload: dict) -> dict:
    """{maneuvers, destination, arrival, depart_instruction, road_class} -> speech."""
    dest = payload.get("destination") or {}
    arrival = (payload.get("arrival") or {}).get("side") or M.UNKNOWN
    default_class = payload.get("road_class") or M.SURFACE
    per_man = payload.get("road_class_by_maneuver") or {}

    mans = [_maneuver(d, per_man.get(d.get("id"), default_class))
            for d in payload.get("maneuvers") or []]
    route = M.CanonicalRoute(
        route_id=payload.get("route_id") or "replay",
        journey_id=payload.get("journey_id") or "replay",
        generation_id=int(payload.get("generation_id") or 1),
        provider="replay", origin_lat=0.0, origin_lng=0.0,
        destination=M.CanonicalDestination(
            display_name=dest.get("display_name") or "",
            formatted_address=dest.get("formatted_address") or "",
            latitude=float(dest.get("lat") or 0.0),
            longitude=float(dest.get("lng") or 0.0)),
        total_distance_m=float(payload.get("total_distance_m") or 0.0),
        duration_s=0.0, geometry=[], maneuvers=mans,
        arrival=M.ArrivalInfo(side=arrival),
        depart_instruction=payload.get("depart_instruction") or "",
    )
    depart = speech_mod.build_route(route)
    return {
        "depart_speech": depart,
        "maneuvers": [{"id": m.id, "road_class": m.road_class, "speech": m.speech}
                      for m in mans],
    }


def main() -> int:
    payload = json.load(sys.stdin)
    json.dump(respeech(payload), sys.stdout)
    return 0


if __name__ == "__main__":
    sys.exit(main())
