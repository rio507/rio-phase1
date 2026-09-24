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


# WHEN A CATEGORY WORD IS NOT ENOUGH, AND ONLY THEN.
#
# Places text search matches the WORDS. Asked for a "movie theater" from
# Pacific Palisades on 2026-09-24 it returned Theatre Palisades (a community
# playhouse, 4.4 km) and an "Outdoor Theater" -- theatres, correctly, and not
# cinemas. The API has an `includedType` that constrains by what a place IS
# rather than by what it is called, and nothing here was sending one.
#
# ADDED PER CATEGORY, ON A MEASUREMENT, because a type is not free: it is a
# filter, and a filter costs results. All six of these were run both ways on
# the same frame of the same drive:
#
#   movie theater  free text -> Theatre Palisades, Outdoor Theater  (wrong kind)
#                  typed     -> nothing within 5 km, which is TRUE: the nearest
#                               cinema is Laemmle NoHo at 19.9 km
#   cinema         free text -> "Uplifting Cinema Pvt. Ltd.", "Journey Cinema"
#                               -- production companies
#                  typed     -> nothing, again true
#   gas station    free text -> ChargePoint, Tesla Supercharger, Electric
#                               Circuit -- chargers, not fuel
#                  typed     -> Conserv Fuel, Village 76, 76, Chevron
#
#   coffee shops   free text -> 5 results including Cafe Mimosa at 3.5 km
#                  typed     -> 3, and CAFE MIMOSA IS GONE. The type is wrong
#                               here and it costs the nearest answer.
#   ev charging    identical both ways -- the words already say it
#   restaurants    identical both ways
#   hospital       free text -> Sunset Urgent Care; typed -> nothing. Arguable
#                               either way, so it is not in the table.
#
# So the table holds the two collisions where a category word means a DIFFERENT
# category in plain English, and nothing else. Anything added here should come
# with the same two lines of evidence, because "it seems more precise" is how
# coffee would have lost its nearest result.
_CATEGORY_TYPES = (
    (("movie theater", "movie theatre", "movie theaters", "movie theatres",
      "cinema", "cinemas", "the movies", "a movie", "picture house",
      "multiplex"), "movie_theater"),
    (("gas station", "gas stations", "petrol station", "petrol stations",
      "petrol", "fuel station", "filling station", "fill up"), "gas_station"),
)


def _included_type(query: str):
    """The Places type this query is unambiguously about. -> str|None

    Substring rather than token match, so "nearest movie theater" and "a movie
    theatre near me" both land. Longest phrases are listed first inside each
    group for the same reason.
    """
    q = " " + (query or "").lower().strip() + " "
    for phrases, kind in _CATEGORY_TYPES:
        for ph in phrases:
            if ph in q:
                return kind
    return None


# HOW WIDE, AND WHO IS ALLOWED TO DECIDE.
#
# `nearby` is the default and is a wall. The other two exist because a wall
# with no door made "what about further out" unanswerable -- see
# config.PLACES_WIDER_RADIUS_M. The scope comes from the MODEL, which is the
# only thing in the system that heard the driver say "further", and the tool
# description is explicit that it may not widen a question that did not ask.
# Whatever it chose is on the record (see _fix_meta) so a silent widening is a
# thing somebody can find rather than a thing somebody suspects.
_SCOPES = ("nearby", "wider", "anywhere")


def _scope_radius(scope: str) -> float:
    """-> the wall for this scope, or 0.0 for no wall."""
    if scope == "wider":
        return float(config.PLACES_WIDER_RADIUS_M)
    if scope == "anywhere":
        return 0.0
    return float(config.PLACES_NEARBY_RADIUS_M)


def _fix_meta(where, origin, radius_m: float, scope: str = "nearby",
              kind: str = None) -> dict:
    """Everything about the position this search used. -> dict

    Provenance rather than a coordinate: "it was 9 km out" and "it was 9 km out
    from a fix taken four minutes ago with 35 m of accuracy" are different
    findings with different fixes, and only the second one can be read off a
    log.
    """
    w = where if isinstance(where, dict) else {}
    return {
        "lat": origin[0] if origin else None,
        "lng": origin[1] if origin else None,
        # The browser's Geolocation watch is the only producer; named anyway,
        # because "which fix" stops being obvious the moment there are two.
        "source": "browser_geolocation" if origin else None,
        "age_s": (round(float(w["age_s"]), 1)
                  if isinstance(w.get("age_s"), (int, float)) else None),
        "accuracy_m": (round(float(w["accuracy_m"]), 1)
                       if isinstance(w.get("accuracy_m"), (int, float)) else None),
        "max_age_s": float(config.PLACES_FIX_MAX_AGE_S),
        # A WALL OR A HINT, said in the record rather than inferred from the
        # radius. They are different requests and they were the whole bug.
        # "wall" rather than "restriction": the limit is ours and is applied
        # after the call, because the API's own restriction does not rank by
        # distance and returned the wrong five. See the note in find_places.
        "constraint": "wall" if radius_m else ("none" if origin else None),
        "radius_m": radius_m or None,
        "rank": "DISTANCE" if origin else None,
        # HOW WIDE, AND WHETHER ANYBODY ASKED. `nearby` is the default and the
        # wall; anything else means the model decided the driver asked to look
        # further, and that decision belongs in the record -- a silent widening
        # of "what's nearby" is the one failure mode this parameter introduces
        # and the only way to catch it is to log what was chosen.
        "scope": scope,
        # ...and what kind of place the words were taken to mean. Null is the
        # ordinary case: most queries need no type. See _CATEGORY_TYPES.
        "included_type": kind,
    }


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
                count: int = None, where=None, session_key: str = "default",
                scope: str = "nearby") -> dict:
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
    # Set only on the restricted path. Zero means "no wall was applied", which
    # is the correct state for a named area, for `anywhere`, and for the filter
    # below.
    radius_m = 0.0
    scope = str(scope or "nearby").strip().lower()
    if scope not in _SCOPES:
        # An unreadable scope is the NARROW one, never the wide one. A model
        # that sends nonsense must not thereby widen a question the driver
        # asked about here.
        scope = "nearby"
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
    # WHAT KIND OF PLACE, where the words alone get it wrong. See
    # _CATEGORY_TYPES for the measurement behind each entry. Applied on the
    # named-area path too: "a cinema in Santa Monica" has the same collision
    # with playhouses that "a cinema near me" does.
    kind = _included_type(query)
    if kind:
        body["includedType"] = kind
    if near:
        # An area was named, so the words carry the location and the car's
        # position must NOT bias the search: "coffee in Santa Monica" asked
        # from downtown is a question about Santa Monica.
        area = near
    elif origin:
        # RESTRICTED TO THE CAR, NOT BIASED TOWARD IT. See
        # config.PLACES_NEARBY_RADIUS_M for the drive that changed this: with a
        # bias, all five results landed outside the radius and the nearest
        # coffee RIO offered was 9.5 km away while there was one at 3.5 km.
        #
        # A RECTANGLE BECAUSE THE API TAKES NOTHING ELSE. searchText's
        # `locationRestriction` accepts a rectangle only -- a circle is
        # rejected outright (HTTP 400, "Unknown name \"circle\" at
        # 'location_restriction'"), which is why this is a bounding box and why
        # the haversine filter below is not belt-and-braces but the other half
        # of the shape: the box's corners reach 1.41x the radius and those
        # corners are the ones that read as "not nearby".
        radius_m = _scope_radius(scope)
        area = {"nearby": "near the car",
                "wider": "further out from the car",
                "anywhere": "anywhere, nearest first"}[scope]
        # A BIAS TO CHOOSE THE CANDIDATES, AND OUR OWN WALL TO ENFORCE THE
        # LIMIT. This was a locationRestriction from 48c882d until 2026-09-24,
        # and the restriction was the wrong instrument -- measured on the drive
        # above, same frame, same query, five results each:
        #
        #   coffee, 5 km   restriction  Alfred 4759, Palisades Garden 4881,
        #                               + two outside. MISSES Cafe Mimosa at
        #                               3544 and Waterlily at 4050 -- the two
        #                               nearest coffees in the set.
        #                  bias         Cafe Mimosa 3544, Waterlily 4050,
        #                               Alfred 4759, Palisades Garden 4881
        #   cinema, 25 km  restriction  Burbank x3 at ~25.8 km, Universal
        #                               20441, NoHo 19889
        #                  bias         AMC Santa Monica 8973, Laemmle Monica
        #                               9058, Town Center 10505, Royal 10672,
        #                               Westwood 10720
        #
        # rankPreference=DISTANCE IS HONOURED WITH A BIAS AND NOT WITH A
        # RESTRICTION. Under a restriction Places returns an arbitrary subset
        # of what is inside the box -- at 25 km it handed back the three
        # Burbank cinemas and skipped the AMC eleven kilometres closer, which
        # is the AMC the driver was asking about. maxResultCount is 5, so WHICH
        # five come back is the whole answer, and only the bias picks the
        # nearest five.
        #
        # THE WALL DID NOT GO AWAY, IT MOVED HERE. The haversine filter below
        # is what enforces the radius, and it is a stronger guarantee than the
        # restriction ever was: a true circle rather than a box whose corners
        # reach 1.41x, applied to coordinates we measured rather than to a
        # shape we asked a vendor to honour. 48c882d's fault was a bias with
        # NO filter and NO sort; this is a bias with both.
        body["locationBias"] = {"circle": {
            "center": {"latitude": origin[0], "longitude": origin[1]},
            "radius": float(radius_m or config.PLACES_WIDER_RADIUS_M)}}
        # ...AND NEAREST FIRST. The old code took Places' relevance order
        # untouched and never sorted, so even within a good set the order was
        # not the order a driver means by "nearby".
        body["rankPreference"] = "DISTANCE"
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
    shaped = [_shape(p, i + 1, origin) for i, p in enumerate(places)]
    shaped = [r for r in shaped if r["name"]]

    # THE CORNERS OF THE BOX, CUT OFF. _box is the smallest rectangle around
    # the circle, so a result can satisfy the restriction and still be 1.41x
    # the radius away on a diagonal. Dropped rather than kept-and-labelled: the
    # radius is what "nearby" means and a result outside it is the thing this
    # whole path exists not to say.
    too_far = []
    if radius_m:
        inside = []
        for r in shaped:
            d = r.get("distance_m")
            if d is None or d <= radius_m:
                inside.append(r)
            else:
                too_far.append(r)
        shaped = inside

    # NEAREST FIRST, HERE AS WELL AS IN THE REQUEST. rankPreference is the
    # server's ordering and this is ours; they agree, and the one that has to
    # be true is this one, because it is the order RIO reads them out in. A
    # result with no coordinates sorts last rather than first -- unknown
    # distance is not zero distance.
    shaped.sort(key=lambda r: (r.get("distance_m") is None,
                               r.get("distance_m") or 0))
    for i, r in enumerate(shaped):
        r["index"] = i + 1
    results = shaped
    remember(session_key, body["textQuery"], results)

    if not results:
        # NOTHING CLOSE IS AN ANSWER, AND IT IS NOT "nothing came back".
        # Distinguished because the two need different sentences: one means the
        # search found no such thing anywhere, the other means there is such a
        # thing but not near the car, and only the second one invites "do you
        # want me to look further out". The search is NOT re-run wider here --
        # reaching further and still calling it nearby is the fault this path
        # was rewritten to remove.
        if radius_m:
            # EMPTY UNDER A WALL IS "NOTHING CLOSE", WHICHEVER WAY IT GOT THERE.
            # Two routes to no results and they must not be told apart in the
            # answer: the API returned nothing inside the box (the usual case,
            # because the restriction filters server-side), or it returned
            # corners that the haversine cut. Only the second leaves anything
            # behind to measure, so `nearest_outside_km` is a bonus and never a
            # condition -- keying the branch on it is how the common case fell
            # through to "nothing came back for that", which says the wrong
            # thing: there ARE coffee shops, just not near the car.
            #
            # AND NO SECOND SEARCH. Finding out how far the nearest one really
            # is would cost another billed call and would be the widening this
            # path exists to refuse. She says there is nothing close, which is
            # true and is the whole answer.
            near_km = (round(min(r["distance_m"] for r in too_far) / 100.0) / 10.0
                       if too_far else None)
            out = {
                "ok": True, "n": 0, "results": [], "query": query,
                "area": area, "note": "nothing_close",
                "searched_radius_m": radius_m,
                "nearest_outside_km": near_km,
                "open_now_filter": bool(open_now),
                "fix": _fix_meta(where, origin, radius_m, scope, kind),
                "took_ms": round((time.time() - t0) * 1000, 1),
            }
            out["rules"] = (
                "There is nothing of that kind close to the car. Say that "
                "plainly -- there is nothing nearby -- and do NOT read out "
                "anything further away as though it were near, and do NOT "
                "name a place from memory."
                + (f" If it helps, the nearest one found was roughly "
                   f"{near_km} km off." if near_km else "")
                # THE DOOR IN THE WALL, NAMED. Without this she had no way to
                # answer "what about further out" and simply could not: on
                # 2026-09-24 a driver was correctly told there was no cinema
                # close, asked about ones beyond, and got nothing back.
                + " Then OFFER to look further out. If they say yes, or if "
                  "they ask for somewhere further, call this again with "
                  "scope='wider'. Do not do that on your own -- widening a "
                  "question they asked about here is how a place half an hour "
                  "away gets called nearby. They can also name an area, which "
                  "goes in `near`."
                if scope == "nearby" else
                " They already asked you to look further out, so do not offer "
                "to widen again. Offer a named area instead, which goes in "
                "`near`."
            )
            return out
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
        # WHAT WAS ASKED, SO THE NEXT BAD ANSWER IS ARGUABLE FROM THE LOG.
        #
        # On 2026-09-24 the drive log recorded the query, the result count and
        # five names -- and not the coordinate, its age, the radius, whether it
        # was a bias or a wall, or how far any result was. So "why were those
        # far away" could not be answered from the record at all; it took
        # replaying the call against a coordinate recovered from a nav row
        # logged twenty-three seconds later. That is the difference between a
        # log and a receipt. app.py copies this block onto the tool_call row.
        "fix": _fix_meta(where, origin, radius_m, scope, kind),
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
