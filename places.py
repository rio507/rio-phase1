"""places.py — what is actually around the car, from Google Places (New).

RIO knows a great deal about restaurants in the abstract and nothing whatever
about the ones on this street. Asked "what's good round here" she used to
answer from the model's own memory, which is a description of the world as it
was when the weights were trained: places that have closed, prices that have
moved, opening hours that were never true on a Tuesday. A confident wrong
answer about a business is worse than no answer, because the driver acts on it
— they drive there.

So this is the same arrangement the rest of RIO's knowledge already has. The
camera answers what is out of the window, the tracker answers where we are, the
vehicle context answers how the car is, and none of them are the model. This
answers what is nearby, and the model's job is to say it well.

ONE CALL PER QUESTION
---------------------
Places (New) bills per request AND by field mask: the fields asked for decide
which SKU the request lands in, so a mask is a bill, not a preference. FIELD_MASK
below is exactly what RIO speaks — name, rating, review count, price, open-now,
address, id, coordinates — and nothing else. Closing times, photos, reviews and
editorial summaries are all deliberately absent: each would add cost to EVERY
question for one clause in one sentence.

The other half of "one call" is that nothing here fans out. Distance is computed
from coordinates the search already returned, and the drive-time figure is an
ESTIMATE derived from that distance — not a Routes call per result, which would
be five billed requests to decorate a sentence. It is labelled as an estimate
everywhere it appears, and the real ETA arrives the moment RIO actually routes
there.

LOCATION
--------
Two ways to know where to look, and no third:

  the car    a GPS fix the browser attaches to the tool call, used as a
             location bias for "near me" / "round here"
  the words  an area the driver named, which goes into the text query and lets
             Google resolve it ("good coffee in Santa Monica")

With neither, this returns `need_location` and RIO asks. It does not fall back
to the last route's origin, a city centroid or the office: a search silently
run 30 km from the driver returns real, correct, useless results, and they look
exactly like good ones.
"""
import math
import os
import threading
import time

import httpx

import config

SEARCH_TEXT_URL = "https://places.googleapis.com/v1/places:searchText"

# The bill. Every field here is spoken by RIO; nothing here is decoration.
# Adding one adds it to every place question this system ever answers.
FIELD_MASK = ",".join([
    "places.id",
    "places.displayName",
    "places.formattedAddress",
    "places.location",
    "places.rating",
    "places.userRatingCount",
    "places.priceLevel",
    "places.currentOpeningHours.openNow",
])

# Google's enum -> the number of currency symbols a person would say.
PRICE_LEVEL = {
    "PRICE_LEVEL_FREE": 0,
    "PRICE_LEVEL_INEXPENSIVE": 1,
    "PRICE_LEVEL_MODERATE": 2,
    "PRICE_LEVEL_EXPENSIVE": 3,
    "PRICE_LEVEL_VERY_EXPENSIVE": 4,
}

_last = {}                    # session_key -> {"t", "query", "results"}
_last_lock = threading.Lock()


def _api_key() -> str:
    """The same key navigation uses, read the same way.

    One key, one place it comes from (.env, gitignored), and it never leaves
    the server — which is the whole reason this tool is answered here rather
    than in the panel like nav_status is.
    """
    key = os.getenv("GOOGLE_MAPS_API_KEY", "").strip()
    if not key:
        raise RuntimeError("GOOGLE_MAPS_API_KEY is not set")
    return key


def haversine_m(lat1, lng1, lat2, lng2) -> float:
    r = 6371000.0
    p1, p2 = math.radians(lat1), math.radians(lat2)
    dp = math.radians(lat2 - lat1)
    dl = math.radians(lng2 - lng1)
    a = (math.sin(dp / 2) ** 2
         + math.cos(p1) * math.cos(p2) * math.sin(dl / 2) ** 2)
    return 2 * r * math.asin(math.sqrt(a))


def drive_minutes(distance_m: float) -> float:
    """Straight-line metres -> a spoken "about N minutes".

    An ESTIMATE, and named one everywhere it is passed on. A routed time would
    cost a Routes request per result — five billed calls to decorate one
    sentence — and the number that actually matters is the one the navigation
    system produces when RIO routes there, which is real and arrives seconds
    later.

    The detour factor is what makes it honest rather than optimistic: roads are
    not straight lines, and quoting the crow-flies time would understate every
    result in a grid city by a third.
    """
    road_m = float(distance_m) * config.PLACES_DETOUR_FACTOR
    return road_m / max(1.0, config.PLACES_DRIVE_SPEED_MS) / 60.0


def _fix_of(where) -> tuple:
    """The browser's GPS fix -> (lat, lng) or None, with a reason when None.

    A stale fix is refused rather than used. The car has been moving; a
    ten-minute-old position is a different neighbourhood, and "near me" answered
    from it is wrong in the one way the driver cannot detect.
    """
    if not isinstance(where, dict):
        return None, "no_fix"
    try:
        lat = float(where.get("lat"))
        lng = float(where.get("lng"))
    except (TypeError, ValueError):
        return None, "no_fix"
    if not (math.isfinite(lat) and math.isfinite(lng)):
        return None, "no_fix"
    age = where.get("age_s")
    if age is not None:
        try:
            if float(age) > config.PLACES_FIX_MAX_AGE_S:
                return None, "stale_fix"
        except (TypeError, ValueError):
            pass
    return (lat, lng), None


def _shape(place: dict, index: int, origin) -> dict:
    """One Places result as RIO says it.

    Every field is either from the response or derived from coordinates in it.
    Nothing is defaulted to a plausible value: a place with no rating comes back
    with `rating: None`, because "no rating" and "unrated" are different things
    to say and neither of them is 4.0.
    """
    loc = place.get("location") or {}
    lat = loc.get("latitude")
    lng = loc.get("longitude")
    out = {
        "index": index,
        "name": (place.get("displayName") or {}).get("text", ""),
        "place_id": place.get("id", ""),
        "address": place.get("formattedAddress", ""),
        "rating": place.get("rating"),
        "ratings_count": place.get("userRatingCount"),
        "price_level": PRICE_LEVEL.get(place.get("priceLevel")),
        "open_now": (place.get("currentOpeningHours") or {}).get("openNow"),
        "lat": lat, "lng": lng,
        "distance_m": None,
        "drive_minutes_est": None,
    }
    if origin and lat is not None and lng is not None:
        d = haversine_m(origin[0], origin[1], float(lat), float(lng))
        out["distance_m"] = round(d)
        out["distance_km"] = round(d / 100.0) / 10.0
        out["drive_minutes_est"] = max(1, round(drive_minutes(d)))
    return out


def remember(session_key: str, query: str, results: list) -> None:
    with _last_lock:
        _last[str(session_key or "default")] = {
            "t": time.time(), "query": query, "results": results}
        # An unbounded dict keyed by session is a leak with a long fuse.
        if len(_last) > 64:
            oldest = sorted(_last.items(), key=lambda kv: kv[1]["t"])[:32]
            for k, _v in oldest:
                _last.pop(k, None)


def last_results(session_key: str) -> dict:
    """What RIO last read out, if it is recent enough to still be what "the
    second one" means.

    This is what makes the follow-through work without a second billed call:
    the results are already in the conversation, each carries its place_id, and
    "take me to the second one" is start_navigation with that id rather than a
    fresh search for a phrase that names no place.
    """
    with _last_lock:
        got = _last.get(str(session_key or "default"))
        if not got:
            return {}
        if (time.time() - got["t"]) > config.PLACES_CACHE_TTL_S:
            return {}
        return dict(got)


def find_places(query: str, near: str = "", open_now: bool = False,
                count: int = None, where=None, session_key: str = "default") -> dict:
    """Text search, once, and shape the answer for speech."""
    t0 = time.time()
    query = (query or "").strip()
    near = (near or "").strip()
    if not query:
        return {"ok": False, "note": "no query",
                "rules": "Ask the driver what they are looking for."}
    if not config.PLACES_ENABLED:
        return {"ok": False, "note": "place search is switched off",
                "rules": "Say you cannot look that up right now. Do not answer "
                         "from memory."}

    origin, fix_note = _fix_of(where)
    # A model can send "3", or "all", or nothing, whatever the schema says the
    # type is. An unreadable count is the default rather than an exception: the
    # driver asked a question, and failing it over an argument they never saw
    # would be the wrong end to be strict at.
    try:
        want = int(count) if count not in (None, "") else config.PLACES_MAX_RESULTS
    except (TypeError, ValueError):
        want = config.PLACES_MAX_RESULTS
    body = {
        "textQuery": f"{query} in {near}" if near else query,
        "maxResultCount": max(1, min(want, config.PLACES_MAX_RESULTS)),
    }
    if open_now:
        body["openNow"] = True
    if near:
        # An area was named, so the words carry the location and the car's
        # position must NOT bias the search: "coffee in Santa Monica" asked
        # from downtown is a question about Santa Monica.
        area = near
    elif origin:
        area = "near the car"
        body["locationBias"] = {"circle": {
            "center": {"latitude": origin[0], "longitude": origin[1]},
            "radius": float(config.PLACES_BIAS_RADIUS_M)}}
    else:
        # Neither. Guessing a location here produces results that are real,
        # correct and useless, and they look exactly like good ones.
        return {
            "ok": False, "note": fix_note or "no_fix", "need_location": True,
            "rules": "You do not know where the car is, so you cannot answer "
                     "this yet. Ask the driver which area to search in — one "
                     "short question — and call this again with `near` set to "
                     "what they say. Do NOT search anyway and do NOT name a "
                     "place from memory.",
        }

    try:
        r = httpx.post(SEARCH_TEXT_URL, timeout=config.PLACES_TIMEOUT_S,
                       json=body, headers={
                           "X-Goog-Api-Key": _api_key(),
                           "Content-Type": "application/json",
                           "X-Goog-FieldMask": FIELD_MASK,
                       })
        r.raise_for_status()
        data = r.json()
    except Exception as e:
        # Every failure is the same answer to the driver: she could not look it
        # up. The alternative — falling back to what the model remembers — is
        # the exact behaviour this tool exists to remove, and it is worse for
        # being invisible.
        print(f"[places] search failed: {type(e).__name__}: {e}", flush=True)
        return {
            "ok": False, "note": f"{type(e).__name__}",
            "took_ms": round((time.time() - t0) * 1000, 1),
            "rules": "The search did not come back. Say plainly that you could "
                     "not pull that up right now, in your own words. Do NOT "
                     "name a business from memory, do NOT guess, and do not "
                     "offer one you 'think' is there.",
        }

    places = data.get("places") or []
    results = [_shape(p, i + 1, origin) for i, p in enumerate(places)]
    results = [r for r in results if r["name"]]
    remember(session_key, body["textQuery"], results)

    if not results:
        return {
            "ok": True, "n": 0, "results": [], "query": query, "area": area,
            "open_now_filter": bool(open_now),
            "took_ms": round((time.time() - t0) * 1000, 1),
            "rules": "Nothing came back for that. Say so plainly and offer to "
                     "try something else or somewhere else. Do NOT fill the "
                     "silence with a place you remember.",
        }

    return {
        "ok": True,
        "n": len(results),
        "query": query,
        "area": area,
        "open_now_filter": bool(open_now),
        "distances_from": "the car" if origin else None,
        "results": results,
        "took_ms": round((time.time() - t0) * 1000, 1),
        "attribution": "Powered by Google",
        "rules": (
            "Answer ONLY from this list. Every name, rating, price and opening "
            "state you say must be in it, and if it is not here you do not know "
            "it. Say the best two or three, not all of them: name, what makes "
            "it worth picking (the rating, how close it is, whether it is open "
            "now), and offer the rest if they want more. Ratings are out of "
            "five and spoken as such. `drive_minutes_est` is an ESTIMATE from "
            "distance, so say 'about four minutes', never 'four minutes'. "
            "If the driver picks one, call start_navigation with that result's "
            "place_id AND its name — the place is already resolved, so passing "
            "the id skips looking it up again and cannot land on a different "
            "branch of the same chain."
        ),
    }
