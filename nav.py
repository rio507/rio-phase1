"""nav.py — WebNavProvider: the server half of RIO's navigation foundation.

The NavigationProvider split
----------------------------
A NavigationProvider answers exactly one question: *what is the route?* It does
NOT drive the car through it. This module is the `web` provider:

    Google Routes API  ──►  route + maneuvers + precomputed speech
                            (this file, once per destination / reroute)

    GPS position       ──►  which maneuver, how far, which tier, speak now
                            (static/rio_navcore.js, ~1 Hz in the browser)

Google owns routing. We never invent, shortcut or "correct" a route: if the
driver leaves it, we ask Google for a new one (`reroute`) rather than patching
the old one. What we own is *progression* — watching the host position against
the returned polyline and deciding when a maneuver is close enough to talk
about. That is deliberate: progression must keep working at 1 Hz with no
network, and a round-trip per turn is not something to put between a driver and
a turn they are 4 seconds from missing.

Swapping in an embedded/offline provider later means replacing `compute_route`
with something that returns the same shape. Nothing downstream of this file
knows the word "Google".

The speech floor
----------------
Every maneuver gets its full announcement text computed HERE, at route time,
for all three approach tiers. Nothing generates language while the car is
moving: the timing path only ever looks a string up. `/nav/voice` re-formats
from the same stored templates, so the text that is spoken and the text that is
logged are produced by one function and cannot drift apart.

Units are metric ("In 300 meters, turn right onto Lincoln") to match the phrase
the design asked for; the imperial variant is a formatter change, nothing else.
"""

import math
import os
import time
import uuid
from collections import OrderedDict
from typing import Optional

import httpx

# ---------------------------------------------------------------------------
# Approach tiers. Seconds of TIME-to-maneuver at the current speed, not metres:
# 300 m of clear highway and 300 m of downtown are not the same warning, and a
# driver needs the same number of seconds to act in both.
#
# Provisional values, like everything else in this codebase's first cut of a
# policy: they are the design's starting points, tuned from real drives later.
# ---------------------------------------------------------------------------
TIER_SECONDS = OrderedDict([("far", 30.0), ("mid", 12.0), ("near", 4.0)])

ROUTES_URL = "https://routes.googleapis.com/directions/v2:computeRoutes"
GEOCODE_URL = "https://maps.googleapis.com/maps/api/geocode/json"
SUGGEST_URL = "https://places.googleapis.com/v1/places:autocomplete"

# Only the fields we actually consume. The Routes API bills partly on the field
# mask, and an unbounded mask would also drag a few hundred kB of alternate
# geometry through the phone's connection on every reroute.
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

# Nominal approach speed is clamped before it becomes a spoken distance. A step
# average of 2 m/s (gridlock) would otherwise announce "in 60 meters" half a
# block early, and 45 m/s (a mis-timed motorway step) would announce a turn
# from a kilometre and a half away.
V_MIN_MS, V_MAX_MS = 4.5, 33.0
NOMINAL_MIN_M, NOMINAL_MAX_M = 25.0, 2000.0

EARTH_R_M = 6371008.8


class NavError(RuntimeError):
    """A routing request failed in a way the driver needs told about."""


# ---------------------------------------------------------------------------
# Route registry
# ---------------------------------------------------------------------------
# Single-driver process, same assumption `_last_talk` in app.py already makes.
# `/nav/voice` needs the route to look a maneuver's text up, which is the whole
# reason the browser never sends speech text to the server: the announcement is
# addressed by (route_id, maneuver, tier) and anything else is refused, exactly
# as /headway_voice refuses a line that is not in its table.
_ROUTES: "OrderedDict[str, dict]" = OrderedDict()
_MAX_ROUTES = 8


def get_route(route_id: str) -> Optional[dict]:
    return _ROUTES.get(route_id)


def _remember(route: dict) -> None:
    _ROUTES[route["route_id"]] = route
    while len(_ROUTES) > _MAX_ROUTES:
        _ROUTES.popitem(last=False)


def _api_key() -> str:
    key = os.getenv("GOOGLE_MAPS_API_KEY", "").strip()
    if not key:
        raise NavError("GOOGLE_MAPS_API_KEY is not set")
    return key


# ---------------------------------------------------------------------------
# Geometry
# ---------------------------------------------------------------------------

def decode_polyline(encoded: str) -> list:
    """Google's encoded polyline -> [[lat, lng], ...].

    The standard algorithm. Kept here rather than pulled from a dependency
    because it is twenty lines and the browser needs the decoded points anyway
    — sending [lat,lng] pairs means the client needs no geometry library and
    the map, the projection and the log all read the same points.
    """
    points, index, lat, lng = [], 0, 0, 0
    n = len(encoded)
    while index < n:
        for is_lat in (True, False):
            result, shift = 0, 0
            while True:
                if index >= n:
                    return points
                b = ord(encoded[index]) - 63
                index += 1
                result |= (b & 0x1F) << shift
                shift += 5
                if b < 0x20:
                    break
            delta = ~(result >> 1) if (result & 1) else (result >> 1)
            if is_lat:
                lat += delta
            else:
                lng += delta
        points.append([lat * 1e-5, lng * 1e-5])
    return points


def haversine_m(a_lat: float, a_lng: float, b_lat: float, b_lng: float) -> float:
    p1, p2 = math.radians(a_lat), math.radians(b_lat)
    dp = p2 - p1
    dl = math.radians(b_lng - a_lng)
    h = math.sin(dp / 2) ** 2 + math.cos(p1) * math.cos(p2) * math.sin(dl / 2) ** 2
    return 2 * EARTH_R_M * math.asin(min(1.0, math.sqrt(h)))


def _cumulative(points: list) -> list:
    """Along-route distance at each vertex, in metres."""
    cum = [0.0] * len(points)
    for i in range(1, len(points)):
        cum[i] = cum[i - 1] + haversine_m(points[i - 1][0], points[i - 1][1],
                                          points[i][0], points[i][1])
    return cum


# ---------------------------------------------------------------------------
# Speech
# ---------------------------------------------------------------------------

def format_distance(m: float) -> str:
    """Spoken distance. Rounded the way a person says it, never to the metre.

    "In 287 meters" is a machine talking. The rounding is coarser the further
    out the maneuver is, because that is where the number matters least.
    """
    m = max(0.0, float(m))
    if m >= 1000:
        km = m / 1000.0
        if km >= 10:
            return f"{round(km)} kilometers"
        txt = f"{km:.1f}".rstrip("0").rstrip(".")
        return f"{txt} kilometer" + ("" if txt == "1" else "s")
    if m >= 100:
        n = int(round(m / 50.0) * 50)
    elif m >= 30:
        n = int(round(m / 10.0) * 10)
    else:
        n = max(10, int(round(m / 5.0) * 5))
    return f"{n} meters"


def _clean_instruction(text: str) -> str:
    """Google's first line only.

    A navigationInstruction is often two lines — the maneuver, then a landmark
    ("Turn left onto Mission St\\nPass by Chase Bank (on the right in 2.1 km)").
    The second line is useful on a screen and is noise in a car, so it is
    dropped from speech and kept nowhere.
    """
    return (text or "").split("\n")[0].strip()


def _lower_first(text: str) -> str:
    """"Turn right onto 10th St" -> "turn right onto 10th St", for mid-sentence.

    Left alone when the first word is an acronym or a route number (US-101,
    I-80), which must not be spoken lowercase.
    """
    if not text:
        return text
    head = text.split(" ", 1)[0]
    if head.isupper() or not head[0].isalpha():
        return text
    return text[0].lower() + text[1:]


def _short_label(label: str) -> str:
    """The spoken form of a destination.

    A geocoder returns "1 Ferry Building, San Francisco, CA 94105, USA", which
    is right on the panel and absurd out loud — by the time RIO finished saying
    the postcode you would have passed it. The first component is what a person
    would say, with the town kept when the first component is only a number.
    """
    parts = [p.strip() for p in (label or "").split(",") if p.strip()]
    if not parts:
        return "your destination"
    head = parts[0]
    if head.replace("-", "").isdigit() and len(parts) > 1:
        head = f"{head} {parts[1]}"
    return head


def _announcements(instruction: str, approach_v_ms: float) -> dict:
    """The three tier announcements for one maneuver, precomputed.

    Each tier carries a template and a fully-formed fallback string. The
    template exists because the honest distance is the one at the moment the
    tier actually fires, which depends on the speed then; the baked string
    exists because the speech floor's whole promise is that a usable sentence
    exists for every maneuver before the drive starts, with no formatting, no
    network and no model between the driver and it.
    """
    out = {}
    body = _lower_first(instruction)
    for tier, secs in TIER_SECONDS.items():
        nominal = min(NOMINAL_MAX_M, max(NOMINAL_MIN_M, secs * approach_v_ms))
        if tier == "near":
            # At four seconds the distance is not information, it is delay.
            template = instruction
        else:
            template = "In {d}, " + body
        out[tier] = {
            "template": template,
            "text": template.replace("{d}", format_distance(nominal)),
            "nominal_m": round(nominal, 1),
            "tier_s": secs,
        }
    return out


def announcement_text(route_id: str, maneuver_index: int, tier: str,
                      dist_m: Optional[float] = None) -> Optional[str]:
    """The exact sentence for (route, maneuver, tier) — the only text source.

    Returns None for anything not on a live route, which is what makes
    /nav/voice a lookup rather than a text-to-speech endpoint.
    """
    route = _ROUTES.get(route_id)
    if not route:
        return None
    if tier not in TIER_SECONDS:
        return None
    for man in route["maneuvers"]:
        if man["index"] == maneuver_index:
            slot = man["announce"][tier]
            if dist_m is None or "{d}" not in slot["template"]:
                return slot["text"]
            return slot["template"].replace("{d}", format_distance(dist_m))
    return None


# ---------------------------------------------------------------------------
# Destination resolution
# ---------------------------------------------------------------------------

def suggest(query: str, lat: Optional[float] = None, lng: Optional[float] = None,
            limit: int = 5) -> list:
    """Places autocomplete, server side.

    Server side rather than the JS widget for two reasons: the key stays on one
    surface, and a Google-rendered dropdown cannot be made to look like the rest
    of this dashboard. Failure here is not an error the driver should see — the
    plain text they typed still geocodes — so callers get an empty list.
    """
    query = (query or "").strip()
    if len(query) < 3:
        return []
    body = {"input": query}
    if lat is not None and lng is not None:
        body["locationBias"] = {"circle": {
            "center": {"latitude": lat, "longitude": lng}, "radius": 50000.0}}
    try:
        r = httpx.post(SUGGEST_URL, json=body, timeout=HTTP_TIMEOUT_S, headers={
            "X-Goog-Api-Key": _api_key(),
            "X-Goog-FieldMask": "suggestions.placePrediction.placeId,"
                                "suggestions.placePrediction.text.text,"
                                "suggestions.placePrediction.structuredFormat",
        })
        r.raise_for_status()
        data = r.json()
    except Exception as e:
        print(f"[nav] suggest failed: {type(e).__name__}: {e}", flush=True)
        return []
    out = []
    for s in (data.get("suggestions") or [])[:limit]:
        p = s.get("placePrediction") or {}
        fmt = p.get("structuredFormat") or {}
        if not p.get("placeId"):
            continue
        out.append({
            "place_id": p["placeId"],
            "text": (p.get("text") or {}).get("text", ""),
            "main": (fmt.get("mainText") or {}).get("text", ""),
            "secondary": (fmt.get("secondaryText") or {}).get("text", ""),
        })
    return out


def geocode(address: str) -> Optional[dict]:
    """Address -> {label, lat, lng}, or None if it cannot be resolved."""
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
    if "lat" not in loc or "lng" not in loc:
        return None
    return {"label": top.get("formatted_address") or address,
            "lat": float(loc["lat"]), "lng": float(loc["lng"])}


# ---------------------------------------------------------------------------
# Routing
# ---------------------------------------------------------------------------

def compute_route(origin_lat: float, origin_lng: float,
                  destination: str = "", place_id: str = "",
                  label: str = "", reroute_of: Optional[str] = None) -> dict:
    """Ask Google for a route and return it in RIO's shape.

    `reroute_of` carries the previous route_id through so a drive can be
    reconstructed from the log as one journey rather than a series of unrelated
    routes.
    """
    if place_id:
        waypoint = {"placeId": place_id}
        dest_label = label or destination or "destination"
        dest_ll = None
    else:
        query = (destination or "").strip()
        if not query:
            raise NavError("no destination given")
        geo = geocode(query)
        if geo:
            waypoint = {"location": {"latLng": {"latitude": geo["lat"],
                                                "longitude": geo["lng"]}}}
            dest_label = label or geo["label"]
            dest_ll = (geo["lat"], geo["lng"])
        else:
            # Geocoding is a convenience, not a dependency: the Routes API takes
            # a free-text address itself, so a geocode outage costs a prettier
            # label and nothing else.
            waypoint = {"address": query}
            dest_label = label or query
            dest_ll = None

    body = {
        "origin": {"location": {"latLng": {"latitude": origin_lat,
                                           "longitude": origin_lng}}},
        "destination": waypoint,
        "travelMode": "DRIVE",
        "routingPreference": "TRAFFIC_AWARE",
        # HIGH_QUALITY is what makes client-side progression possible at all:
        # the overview geometry is ~20 points for 3.5 km, which projects a
        # position onto a chord that can sit 60 m off the actual road and reads
        # as permanently off-route.
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
        detail = ""
        try:
            detail = (r.json().get("error") or {}).get("message", "")
        except Exception:
            detail = r.text[:200]
        raise NavError(f"routes API {r.status_code}: {detail}")

    data = r.json()
    routes = data.get("routes") or []
    if not routes:
        raise NavError("no route found to that destination")

    return _build(routes[0], dest_label, dest_ll,
                  (origin_lat, origin_lng), reroute_of)


def _build(route: dict, dest_label: str, dest_ll, origin_ll, reroute_of) -> dict:
    """Google's route -> RIO's route: one polyline, one maneuver list, one
    announcement per maneuver per tier."""
    legs = route.get("legs") or []
    steps = []
    for leg in legs:
        steps.extend(leg.get("steps") or [])
    if not steps:
        raise NavError("route has no steps")

    # The route polyline is stitched from the per-step polylines rather than
    # taken whole, so every maneuver has an exact vertex index on it. Snapping a
    # maneuver's lat/lng onto an independently-decoded overview line is the
    # classic source of "announced the wrong turn": two nearby junctions can
    # snap to the same vertex.
    points: list = []
    step_start_index = []
    for st in steps:
        pts = decode_polyline(((st.get("polyline") or {}).get("encodedPolyline")) or "")
        if not pts:
            ll = ((st.get("startLocation") or {}).get("latLng")) or {}
            if "latitude" in ll:
                pts = [[ll["latitude"], ll["longitude"]]]
        if not points:
            step_start_index.append(0)
            points.extend(pts)
            continue
        step_start_index.append(len(points) - 1)  # the joint vertex is shared
        points.extend(pts[1:] if pts and pts[0] == points[-1] else pts)

    if len(points) < 2:
        raise NavError("route geometry too short to follow")
    cum = _cumulative(points)

    def _step_speed(idx: int) -> float:
        """Average speed of the step that *leads into* a maneuver."""
        if idx < 0 or idx >= len(steps):
            return 11.0
        st = steps[idx]
        d = float(st.get("distanceMeters") or 0)
        secs = float(str(st.get("staticDuration") or "0s").rstrip("s") or 0)
        if d <= 0 or secs <= 0:
            return 11.0
        return min(V_MAX_MS, max(V_MIN_MS, d / secs))

    maneuvers = []
    # Step 0 is DEPART: its instruction is where the driver already is, so it is
    # a route summary line, never an approach announcement.
    for i in range(1, len(steps)):
        instruction = _clean_instruction(
            (steps[i].get("navigationInstruction") or {}).get("instructions"))
        if not instruction:
            continue
        pi = step_start_index[i]
        maneuvers.append({
            "index": len(maneuvers),
            "type": (steps[i].get("navigationInstruction") or {}).get("maneuver", ""),
            "instruction": instruction,
            "poly_index": pi,
            "along_m": round(cum[pi], 1),
            "lat": points[pi][0],
            "lng": points[pi][1],
            "step_distance_m": steps[i].get("distanceMeters"),
            "announce": _announcements(instruction, _step_speed(i - 1)),
        })

    # Arrival is a maneuver like any other: it has a location on the polyline,
    # it has approach tiers, and the driver needs to be told about it. Modelling
    # it as one keeps the progression engine free of an arrival special case.
    end_ll = ((legs[-1].get("endLocation") or {}).get("latLng")) or {}
    if "latitude" in end_ll:
        dest_lat, dest_lng = end_ll["latitude"], end_ll["longitude"]
    elif dest_ll:
        dest_lat, dest_lng = dest_ll
    else:
        dest_lat, dest_lng = points[-1]
    spoken_dest = _short_label(dest_label)
    arrive_instruction = f"Arrive at {spoken_dest}"
    ann = _announcements(arrive_instruction, _step_speed(len(steps) - 1))
    ann["near"] = dict(ann["near"],
                       template=f"Arriving at {spoken_dest}",
                       text=f"Arriving at {spoken_dest}")
    maneuvers.append({
        "index": len(maneuvers),
        "type": "ARRIVE",
        "instruction": arrive_instruction,
        "poly_index": len(points) - 1,
        "along_m": round(cum[-1], 1),
        "lat": dest_lat,
        "lng": dest_lng,
        "step_distance_m": steps[-1].get("distanceMeters"),
        "announce": ann,
    })

    duration_s = float(str(route.get("duration") or "0s").rstrip("s") or 0)
    out = {
        "route_id": uuid.uuid4().hex[:12],
        "provider": "web",
        "t": time.time(),
        "reroute_of": reroute_of,
        "destination": {"label": dest_label, "lat": dest_lat, "lng": dest_lng},
        "origin": {"lat": origin_ll[0], "lng": origin_ll[1]},
        "distance_m": route.get("distanceMeters"),
        "duration_s": duration_s,
        "eta_epoch": time.time() + duration_s,
        "depart_instruction": _clean_instruction(
            (steps[0].get("navigationInstruction") or {}).get("instructions")),
        "polyline": [[round(p[0], 6), round(p[1], 6)] for p in points],
        "route_length_m": round(cum[-1], 1),
        "maneuvers": maneuvers,
        "tiers_s": dict(TIER_SECONDS),
    }
    _remember(out)
    return out


def summary(route: dict) -> dict:
    """The small shape worth putting in the session log at route_set.

    The polyline is deliberately not in it: a few hundred vertices per reroute
    would dominate a drive's JSONL, and the route_id ties an event back to the
    full geometry if a review ever needs it.
    """
    return {
        "route_id": route["route_id"],
        "provider": route["provider"],
        "reroute_of": route.get("reroute_of"),
        "destination": route["destination"],
        "origin": route["origin"],
        "distance_m": route["distance_m"],
        "duration_s": route["duration_s"],
        "eta_epoch": route["eta_epoch"],
        "n_maneuvers": len(route["maneuvers"]),
        "n_points": len(route["polyline"]),
        "maneuvers": [
            {"index": m["index"], "type": m["type"], "instruction": m["instruction"],
             "along_m": m["along_m"],
             "announce": {t: m["announce"][t]["text"] for t in TIER_SECONDS}}
            for m in route["maneuvers"]
        ],
    }
