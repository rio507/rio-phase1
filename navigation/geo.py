"""Geometry every navigation stage shares — and only geometry.

Route tracking, landmark relation and the simulator all need the same three
things: a distance between two points, a decoded polyline, and the ability to
project a position onto that polyline. They must agree exactly. A landmark
whose along-route position was computed by one projection and compared against
a maneuver's along-route position computed by another is how "just after the
Shell" becomes "just before" — so there is one implementation, here, and every
caller uses it.

Nothing in this module knows what a route is, who provided it, or what RIO
intends to say. It is arithmetic.
"""
import math
from typing import List, Optional, Tuple

EARTH_R_M = 6371008.8
M_PER_DEG = 111320.0


def haversine_m(a_lat: float, a_lng: float, b_lat: float, b_lng: float) -> float:
    """Great-circle distance in metres."""
    p1, p2 = math.radians(a_lat), math.radians(b_lat)
    dp = p2 - p1
    dl = math.radians(b_lng - a_lng)
    h = math.sin(dp / 2) ** 2 + math.cos(p1) * math.cos(p2) * math.sin(dl / 2) ** 2
    return 2 * EARTH_R_M * math.asin(min(1.0, math.sqrt(h)))


def decode_polyline(encoded: str) -> List[List[float]]:
    """Google's encoded polyline -> [[lat, lng], ...].

    The standard algorithm, kept here rather than pulled from a dependency
    because it is twenty lines and the browser needs the decoded points anyway:
    sending [lat, lng] pairs means the client needs no geometry library, and the
    map, the projection and the log all read the same points.

    It lives in geo.py rather than in the Google provider because the encoding
    is not Google's — it is a common interchange format, and the next provider
    is as likely to use it as not.
    """
    points: List[List[float]] = []
    index, lat, lng = 0, 0, 0
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


def cumulative(points: List[List[float]]) -> List[float]:
    """Along-route distance at each vertex, in metres."""
    cum = [0.0] * len(points)
    for i in range(1, len(points)):
        cum[i] = cum[i - 1] + haversine_m(points[i - 1][0], points[i - 1][1],
                                          points[i][0], points[i][1])
    return cum


def project(points: List[List[float]], cum: List[float],
            lat: float, lng: float,
            lo: int = 0, hi: Optional[int] = None) -> Tuple[float, float, int]:
    """Nearest point on a polyline to (lat, lng).

    Returns (along_m, lateral_m, vertex_index). Local flat-earth metres: over
    the few hundred metres any single query spans, the error is centimetres and
    the inner loop stays arithmetic.

    `lo`/`hi` bound the search. An unbounded search on a route that crosses
    itself will happily snap a position onto a later leg, which reads as the
    driver having teleported two miles down the route — so callers that know
    roughly where they are say so.
    """
    if len(points) < 2:
        return 0.0, 0.0, 0
    hi = len(points) - 2 if hi is None else min(hi, len(points) - 2)
    lo = max(0, min(lo, hi))
    k_lng = M_PER_DEG * math.cos(math.radians(lat))
    best = None
    for i in range(lo, hi + 1):
        ax = (points[i][1] - lng) * k_lng
        ay = (points[i][0] - lat) * M_PER_DEG
        bx = (points[i + 1][1] - lng) * k_lng
        by = (points[i + 1][0] - lat) * M_PER_DEG
        dx, dy = bx - ax, by - ay
        len2 = dx * dx + dy * dy
        t = 0.0 if len2 <= 0 else -(ax * dx + ay * dy) / len2
        t = 0.0 if t < 0 else (1.0 if t > 1 else t)
        px, py = ax + t * dx, ay + t * dy
        d2 = px * px + py * py
        if best is None or d2 < best[0]:
            best = (d2, i, cum[i] + t * math.sqrt(len2))
    if best is None:
        return 0.0, 0.0, 0
    return best[2], math.sqrt(best[0]), best[1]


def point_at(points: List[List[float]], cum: List[float], m: float):
    """The position `m` metres along the polyline. The simulator's geometry."""
    if not points:
        return None
    if m <= 0:
        return points[0][0], points[0][1]
    total = cum[-1]
    if m >= total:
        return points[-1][0], points[-1][1]
    lo, hi = 0, len(cum) - 1
    while lo < hi - 1:
        mid = (lo + hi) // 2
        if cum[mid] <= m:
            lo = mid
        else:
            hi = mid
    seg = cum[hi] - cum[lo]
    f = 0.0 if seg <= 0 else (m - cum[lo]) / seg
    return (points[lo][0] + (points[hi][0] - points[lo][0]) * f,
            points[lo][1] + (points[hi][1] - points[lo][1]) * f)


def bearing_deg(a_lat: float, a_lng: float, b_lat: float, b_lng: float) -> float:
    """Initial bearing from A to B, degrees clockwise from true north.

    Used for the iOS heading fallback (a browser Geolocation fix often carries
    no heading at all) and for checking that a landmark sits on the side of the
    road the driver is about to be looking at.
    """
    p1, p2 = math.radians(a_lat), math.radians(b_lat)
    dl = math.radians(b_lng - a_lng)
    y = math.sin(dl) * math.cos(p2)
    x = math.cos(p1) * math.sin(p2) - math.sin(p1) * math.cos(p2) * math.cos(dl)
    return (math.degrees(math.atan2(y, x)) + 360.0) % 360.0


def side_of_line(a_lat: float, a_lng: float, heading_deg: float,
                 p_lat: float, p_lng: float) -> str:
    """Which side of a heading through A the point P falls on: LEFT/RIGHT.

    Flat-earth cross product. Only ever used as a plausibility hint — a
    landmark's side is map data, never something the camera is asked to decide.
    """
    k_lng = M_PER_DEG * math.cos(math.radians(a_lat))
    ex = math.sin(math.radians(heading_deg))
    ey = math.cos(math.radians(heading_deg))
    vx = (p_lng - a_lng) * k_lng
    vy = (p_lat - a_lat) * M_PER_DEG
    cross = ex * vy - ey * vx
    return "LEFT" if cross > 0 else "RIGHT"
