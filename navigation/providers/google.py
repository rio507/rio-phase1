"""GoogleProvider — Routes v2, Geocoding, Places. The only Google-shaped file.

Everything about Google's request bodies, field masks, maneuver enum and
response shape is confined to this module. Downstream code — the tracker, the
landmark stage, the verifier, the speech planner, the panel — sees
CanonicalRoute and nothing else, which is what makes §36's "architecture must
allow provider substitution without rebuilding" true rather than aspirational.

LICENSING (see LICENSING.md): this file is the entire surface of the question.
Whether an API technically returns routing data and whether it may be used for
in-vehicle turn-by-turn with synthesized speech are different questions, and
the second one is answered before production, not by observing that the first
one returns HTTP 200.

Field masks are narrow on purpose. The Routes API bills partly on the mask, and
an unbounded one also drags a few hundred kB of alternate geometry through the
phone's connection on every reroute.
"""
import math
import os
import re
import time
from typing import List, Optional

import httpx

import config

from .. import geo
from .. import model as M
from ..provider import DestinationCandidate, NavError, NavigationProvider

ROUTES_URL = "https://routes.googleapis.com/directions/v2:computeRoutes"
GEOCODE_URL = "https://maps.googleapis.com/maps/api/geocode/json"
AUTOCOMPLETE_URL = "https://places.googleapis.com/v1/places:autocomplete"
PLACE_DETAILS_URL = "https://places.googleapis.com/v1/places/{place_id}"
NEARBY_URL = "https://places.googleapis.com/v1/places:searchNearby"

FIELD_MASK = ",".join([
    "routes.distanceMeters",
    "routes.duration",
    "routes.legs.endLocation",
    "routes.legs.steps.distanceMeters",
    "routes.legs.steps.staticDuration",
    "routes.legs.steps.startLocation",
    "routes.legs.steps.endLocation",
    "routes.legs.steps.navigationInstruction",
    "routes.legs.steps.polyline.encodedPolyline",
])

HTTP_TIMEOUT_S = 12.0

# Nominal approach speed, clamped before anything is derived from it. A step
# average of 2 m/s (gridlock) and one of 45 m/s (a mis-timed motorway step) are
# both real and both useless as an approach speed.
V_MIN_MS, V_MAX_MS = 4.5, 33.0

# --- Google's maneuver enum -> RIO's small vocabulary -----------------------
# Left as a table rather than a chain of substring tests because the mapping is
# the provider's contract with RIO, and a table can be read against the API
# documentation in one pass.
_MANEUVER_MAP = {
    "TURN_LEFT": (M.TURN, M.LEFT),
    "TURN_RIGHT": (M.TURN, M.RIGHT),
    "TURN_SLIGHT_LEFT": (M.TURN, M.LEFT),
    "TURN_SLIGHT_RIGHT": (M.TURN, M.RIGHT),
    "TURN_SHARP_LEFT": (M.TURN, M.LEFT),
    "TURN_SHARP_RIGHT": (M.TURN, M.RIGHT),
    "TURN_U_TURN_LEFT": (M.UTURN, M.LEFT),
    "TURN_U_TURN_RIGHT": (M.UTURN, M.RIGHT),
    "UTURN_LEFT": (M.UTURN, M.LEFT),
    "UTURN_RIGHT": (M.UTURN, M.RIGHT),
    "RAMP_LEFT": (M.RAMP, M.LEFT),
    "RAMP_RIGHT": (M.RAMP, M.RIGHT),
    "MERGE": (M.MERGE, M.UNKNOWN),
    "MERGE_LEFT": (M.MERGE, M.LEFT),
    "MERGE_RIGHT": (M.MERGE, M.RIGHT),
    "FORK_LEFT": (M.FORK, M.LEFT),
    "FORK_RIGHT": (M.FORK, M.RIGHT),
    "ROUNDABOUT_LEFT": (M.ROUNDABOUT, M.LEFT),
    "ROUNDABOUT_RIGHT": (M.ROUNDABOUT, M.RIGHT),
    "STRAIGHT": (M.STRAIGHT, M.STRAIGHT_DIR),
    "NAME_CHANGE": (M.STRAIGHT, M.STRAIGHT_DIR),
    "KEEP_LEFT": (M.KEEP, M.LEFT),
    "KEEP_RIGHT": (M.KEEP, M.RIGHT),
    "DEPART": (M.DEPART, M.UNKNOWN),
    "DESTINATION": (M.ARRIVE, M.UNKNOWN),
    "DESTINATION_LEFT": (M.ARRIVE, M.LEFT),
    "DESTINATION_RIGHT": (M.ARRIVE, M.RIGHT),
}

_ONTO = re.compile(r"\b(?:onto|on to)\s+(.+)$", re.IGNORECASE)


def _api_key() -> str:
    key = os.getenv("GOOGLE_MAPS_API_KEY", "").strip()
    if not key:
        raise NavError("GOOGLE_MAPS_API_KEY is not set")
    return key


def _clean_instruction(text: str) -> str:
    """Google's first line only.

    A navigationInstruction is often two lines — the maneuver, then a landmark
    ("Turn left onto Mission St\\nPass by Chase Bank (on the right in 2.1 km)").
    The second line is for a screen. Note that RIO does NOT take its contextual
    landmarks from this string: a landmark RIO speaks has to have been verified
    as visible, and this line is neither verified nor spatially resolved.
    """
    return (text or "").split("\n")[0].strip()


def _road_name(instruction: str) -> str:
    """"Turn left onto Lincoln Blvd" -> "Lincoln Blvd".

    The Routes API does not return the target road as its own field at this
    field-mask level, so it is parsed from the instruction — deterministically,
    and only the `onto` form. Anything else yields "", and every template
    handles a missing road name by simply not saying one.
    """
    m = _ONTO.search(instruction or "")
    if not m:
        return ""
    name = m.group(1).strip().rstrip(".")
    # "onto Main St toward the freeway" — the road is the part before `toward`.
    for cut in (" toward ", " towards ", " and ", " then "):
        i = name.lower().find(cut)
        if i > 0:
            name = name[:i]
    return name.strip()


def _short_label(label: str) -> str:
    """The spoken form of a destination.

    A geocoder returns "1 Ferry Building, San Francisco, CA 94105, USA", which
    is right on the panel and absurd out loud — by the time RIO finished saying
    the postcode you would have passed it.
    """
    parts = [p.strip() for p in (label or "").split(",") if p.strip()]
    if not parts:
        return "your destination"
    head = parts[0]
    if head.replace("-", "").isdigit() and len(parts) > 1:
        head = f"{head} {parts[1]}"
    return head


class GoogleProvider(NavigationProvider):
    name = "google"

    # -- destination resolution ---------------------------------------------
    def suggest(self, query: str, lat: Optional[float] = None,
                lng: Optional[float] = None, limit: int = 5) -> List[DestinationCandidate]:
        query = (query or "").strip()
        if len(query) < 3:
            return []
        body = {"input": query}
        if lat is not None and lng is not None:
            body["locationBias"] = {"circle": {
                "center": {"latitude": lat, "longitude": lng}, "radius": 50000.0}}
        try:
            r = httpx.post(AUTOCOMPLETE_URL, json=body, timeout=HTTP_TIMEOUT_S, headers={
                "X-Goog-Api-Key": _api_key(),
                "X-Goog-FieldMask": "suggestions.placePrediction.placeId,"
                                    "suggestions.placePrediction.text.text,"
                                    "suggestions.placePrediction.structuredFormat",
            })
            r.raise_for_status()
            data = r.json()
        except Exception as e:
            # Never fatal: the plain text the driver typed still geocodes.
            print(f"[nav] suggest failed: {type(e).__name__}: {e}", flush=True)
            return []
        out = []
        for s in (data.get("suggestions") or [])[:limit]:
            p = s.get("placePrediction") or {}
            fmt = p.get("structuredFormat") or {}
            if not p.get("placeId"):
                continue
            out.append(DestinationCandidate(
                display_name=(fmt.get("mainText") or {}).get("text", "")
                             or (p.get("text") or {}).get("text", ""),
                formatted_address=(p.get("text") or {}).get("text", ""),
                provider_place_id=p["placeId"],
            ))
        return out

    def destination(self, query: str = "", place_id: str = "",
                    label: str = "") -> Optional[M.CanonicalDestination]:
        if place_id:
            d = self._place_details(place_id)
            if d:
                return d
            # Details unavailable: the Routes API accepts the place id directly,
            # and the route's own end location pins the coordinates. Losing the
            # details call costs a nicer label, not the destination.
            return M.CanonicalDestination(
                display_name=_short_label(label) or "your destination",
                formatted_address=label or "", latitude=0.0, longitude=0.0,
                provider_place_id=place_id)
        q = (query or "").strip()
        if not q:
            return None
        g = self._geocode(q)
        if not g:
            return None
        return M.CanonicalDestination(
            display_name=_short_label(label or g["label"]),
            formatted_address=g["label"],
            latitude=g["lat"], longitude=g["lng"], provider_place_id=g.get("place_id"))

    def _place_details(self, place_id: str) -> Optional[M.CanonicalDestination]:
        try:
            r = httpx.get(PLACE_DETAILS_URL.format(place_id=place_id),
                          timeout=HTTP_TIMEOUT_S, headers={
                              "X-Goog-Api-Key": _api_key(),
                              "X-Goog-FieldMask": "id,displayName,formattedAddress,location",
                          })
            r.raise_for_status()
            d = r.json()
        except Exception as e:
            print(f"[nav] place details failed: {type(e).__name__}: {e}", flush=True)
            return None
        loc = d.get("location") or {}
        if "latitude" not in loc:
            return None
        return M.CanonicalDestination(
            display_name=(d.get("displayName") or {}).get("text", "") or _short_label(
                d.get("formattedAddress", "")),
            formatted_address=d.get("formattedAddress", ""),
            latitude=float(loc["latitude"]), longitude=float(loc["longitude"]),
            provider_place_id=d.get("id") or place_id)

    def _geocode(self, address: str) -> Optional[dict]:
        try:
            r = httpx.get(GEOCODE_URL, timeout=HTTP_TIMEOUT_S,
                          params={"address": address, "key": _api_key()})
            r.raise_for_status()
            data = r.json()
        except Exception as e:
            print(f"[nav] geocode failed: {type(e).__name__}: {e}", flush=True)
            return None
        if data.get("status") != "OK" or not data.get("results"):
            return None
        top = data["results"][0]
        loc = (top.get("geometry") or {}).get("location") or {}
        if "lat" not in loc:
            return None
        return {"label": top.get("formatted_address") or address,
                "lat": float(loc["lat"]), "lng": float(loc["lng"]),
                "place_id": top.get("place_id"),
                "n_results": len(data["results"])}

    def geocode_point(self, address: str) -> Optional[dict]:
        """Address -> {label, lat, lng}. Used for the desk start-point override."""
        return self._geocode(address)

    # -- routing -------------------------------------------------------------
    def route(self, origin_lat: float, origin_lng: float,
              destination: M.CanonicalDestination) -> M.CanonicalRoute:
        # A place id in preference to coordinates even when both are known: a
        # business's coordinates are a rooftop or a centroid, and the provider
        # routes a place id to the entrance it knows about. "Arrive at the
        # Getty Center" landing on the far side of the building is the
        # difference.
        if destination.provider_place_id:
            waypoint = {"placeId": destination.provider_place_id}
        elif destination.latitude or destination.longitude:
            waypoint = {"location": {"latLng": {"latitude": destination.latitude,
                                                "longitude": destination.longitude}}}
        elif destination.formatted_address:
            waypoint = {"address": destination.formatted_address}
        else:
            raise NavError("no destination given")

        body = {
            "origin": {"location": {"latLng": {"latitude": origin_lat,
                                               "longitude": origin_lng}}},
            "destination": waypoint,
            "travelMode": "DRIVE",
            "routingPreference": "TRAFFIC_AWARE",
            # HIGH_QUALITY is what makes route tracking possible at all: the
            # overview geometry is ~20 points for 3.5 km, which projects a
            # position onto a chord that can sit 60 m off the actual road and
            # reads as permanently off-route.
            "polylineQuality": "HIGH_QUALITY",
            "computeAlternativeRoutes": False,
            "languageCode": "en-US",
            "units": "METRIC",
        }
        try:
            r = httpx.post(ROUTES_URL, json=body, timeout=HTTP_TIMEOUT_S, headers={
                "Content-Type": "application/json",
                "X-Goog-Api-Key": _api_key(),
                "X-Goog-FieldMask": FIELD_MASK,
            })
        except Exception as e:
            raise NavError(f"routes request failed: {type(e).__name__}: {e}")
        if r.status_code != 200:
            try:
                detail = (r.json().get("error") or {}).get("message", "")
            except Exception:
                detail = r.text[:200]
            raise NavError(f"routes API {r.status_code}: {detail}")
        routes = (r.json().get("routes") or [])
        if not routes:
            raise NavError("no route found to that destination")
        return self._build(routes[0], destination, origin_lat, origin_lng)

    def _build(self, route: dict, destination: M.CanonicalDestination,
               origin_lat: float, origin_lng: float) -> M.CanonicalRoute:
        legs = route.get("legs") or []
        steps = []
        for leg in legs:
            steps.extend(leg.get("steps") or [])
        if not steps:
            raise NavError("route has no steps")

        # The geometry is stitched from the per-step polylines rather than taken
        # whole, so every maneuver has an EXACT vertex on it. Snapping a
        # maneuver's lat/lng onto an independently decoded overview line is the
        # classic source of "announced the wrong turn": two junctions 20 m apart
        # snap to the same vertex, and the tracker then cannot tell them apart.
        points: List[List[float]] = []
        step_start_index = []
        for st in steps:
            pts = geo.decode_polyline(
                ((st.get("polyline") or {}).get("encodedPolyline")) or "")
            if not pts:
                ll = ((st.get("startLocation") or {}).get("latLng")) or {}
                if "latitude" in ll:
                    pts = [[ll["latitude"], ll["longitude"]]]
            if not points:
                step_start_index.append(0)
                points.extend(pts)
                continue
            step_start_index.append(len(points) - 1)     # the joint is shared
            points.extend(pts[1:] if pts and pts[0] == points[-1] else pts)

        if len(points) < 2:
            raise NavError("route geometry too short to follow")
        cum = geo.cumulative(points)

        def _step_speed(idx: int) -> float:
            if idx < 0 or idx >= len(steps):
                return 11.0
            st = steps[idx]
            d = float(st.get("distanceMeters") or 0)
            secs = float(str(st.get("staticDuration") or "0s").rstrip("s") or 0)
            if d <= 0 or secs <= 0:
                return 11.0
            return min(V_MAX_MS, max(V_MIN_MS, d / secs))

        maneuvers: List[M.CanonicalManeuver] = []
        arrival_side = M.UNKNOWN
        # Step 0 is DEPART: its instruction describes where the driver already
        # is, so it is a route summary line and never an approach announcement.
        for i in range(1, len(steps)):
            nav_i = steps[i].get("navigationInstruction") or {}
            instruction = _clean_instruction(nav_i.get("instructions"))
            if not instruction:
                continue
            raw = (nav_i.get("maneuver") or "").upper()
            mtype, direction = _MANEUVER_MAP.get(raw, (M.TURN, M.UNKNOWN))
            if mtype == M.ARRIVE:
                # The final step is arrival; it is appended below as one
                # maneuver with the destination's own coordinates, and this is
                # where the provider tells us which side of the road it is on.
                arrival_side = direction if direction in (M.LEFT, M.RIGHT) else M.UNKNOWN
                continue
            pi = step_start_index[i]
            seq = len(maneuvers)
            maneuvers.append(M.CanonicalManeuver(
                id=f"m{seq}", sequence=seq, type=mtype, direction=direction,
                road_name=_road_name(instruction),
                latitude=points[pi][0], longitude=points[pi][1],
                route_distance_position=cum[pi], polyline_index=pi,
                instruction=instruction,
                step_distance_m=steps[i].get("distanceMeters"),
                approach_speed_ms=_step_speed(i - 1),
            ))

        # Arrival is a maneuver like any other: it has a place on the geometry,
        # it has approach timing, and the driver needs telling about it. Making
        # it a maneuver is what keeps the tracker free of an arrival special
        # case.
        end_ll = ((legs[-1].get("endLocation") or {}).get("latLng")) or {}
        if "latitude" in end_ll:
            dest_lat, dest_lng = float(end_ll["latitude"]), float(end_ll["longitude"])
        elif destination.latitude or destination.longitude:
            dest_lat, dest_lng = destination.latitude, destination.longitude
        else:
            dest_lat, dest_lng = points[-1]
        if not (destination.latitude or destination.longitude):
            destination.latitude, destination.longitude = dest_lat, dest_lng
        seq = len(maneuvers)
        maneuvers.append(M.CanonicalManeuver(
            id=f"m{seq}", sequence=seq, type=M.ARRIVE, direction=arrival_side,
            road_name="", latitude=dest_lat, longitude=dest_lng,
            route_distance_position=cum[-1], polyline_index=len(points) - 1,
            instruction=f"Arrive at {destination.display_name}",
            step_distance_m=steps[-1].get("distanceMeters"),
            approach_speed_ms=_step_speed(len(steps) - 1),
        ))

        duration_s = float(str(route.get("duration") or "0s").rstrip("s") or 0)
        return M.CanonicalRoute(
            route_id=M.new_route_id(), journey_id="", generation_id=0,
            provider=self.name,
            origin_lat=origin_lat, origin_lng=origin_lng,
            destination=destination,
            total_distance_m=float(route.get("distanceMeters") or cum[-1]),
            duration_s=duration_s,
            geometry=points, maneuvers=maneuvers,
            arrival=M.ArrivalInfo(side=arrival_side),
            depart_instruction=_clean_instruction(
                (steps[0].get("navigationInstruction") or {}).get("instructions")),
            route_length_m=cum[-1],
        )

    # -- landmark candidates -------------------------------------------------
    def landmarks_near(self, lat: float, lng: float, radius_m: float,
                       types: tuple) -> List[dict]:
        """Places Nearby Search around ONE maneuver.

        Called once per anchorable maneuver at route load and never again for
        that route generation (see landmarks.py for the budget and the cache).
        Ranked by distance because the question is "what is at this junction",
        not "what is popular near this junction".
        """
        try:
            r = httpx.post(NEARBY_URL, timeout=HTTP_TIMEOUT_S, json={
                "includedTypes": list(types),
                "maxResultCount": 10,
                "rankPreference": "DISTANCE",
                "locationRestriction": {"circle": {
                    "center": {"latitude": lat, "longitude": lng},
                    "radius": float(radius_m)}},
            }, headers={
                "X-Goog-Api-Key": _api_key(),
                "Content-Type": "application/json",
                "X-Goog-FieldMask": "places.id,places.displayName,places.location,"
                                    "places.primaryType,places.types,places.businessStatus",
            })
            r.raise_for_status()
            data = r.json()
        except Exception as e:
            # No landmarks is a normal outcome, not an error: RIO navigates
            # canonically and says nothing about the surroundings.
            print(f"[nav] nearby search failed: {type(e).__name__}: {e}", flush=True)
            return []
        out = []
        for p in (data.get("places") or []):
            loc = p.get("location") or {}
            if "latitude" not in loc:
                continue
            if (p.get("businessStatus") or "OPERATIONAL") != "OPERATIONAL":
                continue        # a closed business still has a sign, sometimes
            out.append({
                "place_id": p.get("id", ""),
                "name": (p.get("displayName") or {}).get("text", ""),
                "primary_type": p.get("primaryType", ""),
                "types": p.get("types") or [],
                "lat": float(loc["latitude"]), "lng": float(loc["longitude"]),
            })
        return out
