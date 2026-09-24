"""places_nearby_selftest.py — "nearby" means nearby, and it is a wall not a hint.

    python tools/places_nearby_selftest.py

THE DRIVE, 2026-09-24, session 4989d12e. "What coffee shops are nearby", asked
from Pacific Palisades with a live fix, and every one of the five answers was
outside the 8 km the search had been biased toward:

    Happy Days Cafe          13183 m   Sherman Oaks
    The Morning Mood         21197 m   Granada Hills
    LOKL HAUS                 9458 m   Santa Monica
    Urth Caffe Santa Monica  10725 m   Santa Monica
    Valley Grounds Coffee    13237 m   Sherman Oaks

while Cafe Mimosa was 3.5 km away and Alfred Coffee 4.8 km, in the village the
car was sitting in. Nothing failed. `locationBias` is a SUGGESTION, Places
decided relevance beat proximity, and the results came back in relevance order
because nothing here sorted them.

WHY THIS IS OFFLINE. The transport is stubbed and the assertions are about the
REQUEST WE SEND and the SHAPE WE RETURN -- neither of which should depend on
what Google has indexed this morning. A test that needs the network to say
whether we asked for a restriction is a test that goes red for the wrong
reason. The live evidence lives in the commit message; this is the guard.

WHAT IT ASSERTS
    1. A fix means a RESTRICTION, never a bias, and nearest-first ranking.
    2. The rectangle really contains the circle -- searchText refuses a circle
       (HTTP 400), so the box is the only shape available and its corners have
       to be cut off afterwards or "within 5 km" means 7.1 km diagonally.
    3. Results come back sorted by distance, renumbered, corners dropped.
    4. Nothing inside the radius is `nothing_close` -- BOTH ways it can happen
       -- and the search is never widened to fill the silence.
    5. A NAMED AREA is not restricted at all: "coffee in Santa Monica" asked
       from the Palisades is a question about Santa Monica.
    6. The result carries enough provenance to argue with it later.
"""
import json
import math
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO))

import config                                                    # noqa: E402
import places                                                    # noqa: E402

checks = 0
failures = 0


def ok(what, cond, extra=""):
    global checks, failures
    checks += 1
    if not cond:
        failures += 1
    print(("  ok   " if cond else "  FAIL ") + what
          + (f" — {extra}" if extra else ""), flush=True)


def section(n, title):
    print(f"\n=== {n}. {title} ===")


# The car, where it actually was on the drive.
ORIGIN = (34.0769965651506, -118.5623867652752)
WHERE = {"lat": ORIGIN[0], "lng": ORIGIN[1], "age_s": 2.4, "accuracy_m": 35.0}


def at(distance_m, bearing_deg=90.0, name=None):
    """A place `distance_m` from the car on `bearing_deg`. -> a Places result.

    Built from the origin rather than typed in, so the fixture cannot disagree
    with the haversine the code under test uses.
    """
    R = 6371000.0
    br = math.radians(bearing_deg)
    lat1, lng1 = math.radians(ORIGIN[0]), math.radians(ORIGIN[1])
    lat2 = math.asin(math.sin(lat1) * math.cos(distance_m / R)
                     + math.cos(lat1) * math.sin(distance_m / R) * math.cos(br))
    lng2 = lng1 + math.atan2(
        math.sin(br) * math.sin(distance_m / R) * math.cos(lat1),
        math.cos(distance_m / R) - math.sin(lat1) * math.sin(lat2))
    return {
        "id": f"p{int(distance_m)}",
        "displayName": {"text": name or f"Cafe {int(distance_m)}m"},
        "formattedAddress": "somewhere",
        "location": {"latitude": math.degrees(lat2),
                     "longitude": math.degrees(lng2)},
    }


class Sent:
    """The last request body, captured instead of sent."""
    body = None


def stub(results):
    """Replace the transport. Returns nothing; restores nothing -- each test
    installs its own, which is why every one of them sets `results`."""
    class R:
        status_code = 200

        def raise_for_status(self):
            pass

        def json(self):
            return {"places": results}

    def post(url, **kw):
        Sent.body = kw.get("json")
        return R()

    places.httpx.post = post


def main():
    radius = float(config.PLACES_NEARBY_RADIUS_M)

    section(1, "A FIX IS A WALL, NOT A HINT")
    stub([at(1000), at(2000)])
    r = places.find_places("coffee shops", where=WHERE)
    b = Sent.body
    ok("the request carries a locationRestriction", "locationRestriction" in b)
    ok("...and NOT a locationBias — the bias is what let every result out",
       "locationBias" not in b)
    ok("...as a rectangle, the only shape searchText accepts",
       "rectangle" in (b.get("locationRestriction") or {}))
    ok("...and asks for the nearest first",
       b.get("rankPreference") == "DISTANCE", str(b.get("rankPreference")))

    section(2, "THE RECTANGLE CONTAINS THE CIRCLE")
    box = places._box(ORIGIN, radius)
    lo, hi = box["low"], box["high"]
    n = places.haversine_m(hi["latitude"], ORIGIN[1], ORIGIN[0], ORIGIN[1])
    e = places.haversine_m(ORIGIN[0], hi["longitude"], ORIGIN[0], ORIGIN[1])
    ok("the box reaches the radius due north", abs(n - radius) < radius * 0.02,
       f"{n:.0f} m vs {radius:.0f} m")
    ok("...and due east", abs(e - radius) < radius * 0.02,
       f"{e:.0f} m vs {radius:.0f} m")
    corner = places.haversine_m(hi["latitude"], hi["longitude"],
                                ORIGIN[0], ORIGIN[1])
    ok("...and OVERSHOOTS at the corner, which is why the filter exists",
       corner > radius * 1.3, f"{corner:.0f} m")
    ok("the box is centred on the car",
       lo["latitude"] < ORIGIN[0] < hi["latitude"]
       and lo["longitude"] < ORIGIN[1] < hi["longitude"])

    section(3, "NEAREST FIRST, AND THE CORNERS CUT OFF")
    # Deliberately out of order, and one of them is a corner the box admits
    # and the radius does not.
    stub([at(4000, name="four"), at(radius * 1.2, 45.0, name="corner"),
          at(1000, name="one"), at(2500, name="two-five")])
    r = places.find_places("coffee shops", where=WHERE)
    names = [x["name"] for x in r["results"]]
    dists = [x["distance_m"] for x in r["results"]]
    ok("the far corner is dropped though the rectangle admitted it",
       "corner" not in names, str(names))
    ok("...and everything inside the radius is kept", len(r["results"]) == 3,
       str(len(r["results"])))
    ok("results are sorted nearest first", dists == sorted(dists), str(dists))
    ok("...and renumbered so index matches the order RIO reads them in",
       [x["index"] for x in r["results"]] == [1, 2, 3])
    ok("every result is inside the radius",
       all(d <= radius for d in dists), str(dists))

    section(4, "NOTHING CLOSE IS AN ANSWER, AND IT IS NOT 'NOTHING EXISTS'")
    # (a) the API returned only corners — the filter empties the list.
    stub([at(radius * 1.2, 45.0, name="corner")])
    r = places.find_places("coffee shops", where=WHERE)
    ok("filtered to empty -> nothing_close", r.get("note") == "nothing_close",
       str(r.get("note")))
    ok("...and it says how far the nearest actually was",
       r.get("nearest_outside_km") is not None,
       str(r.get("nearest_outside_km")))
    # (b) the API returned nothing at all — the usual case, because the
    #     restriction filters server-side. This is the one that used to fall
    #     through to "nothing came back for that".
    stub([])
    r = places.find_places("coffee shops", where=WHERE)
    ok("empty from the API -> nothing_close too",
       r.get("note") == "nothing_close", str(r.get("note")))
    ok("...with no nearest to report, and that is not an error",
       r.get("nearest_outside_km") is None)
    ok("...the radius it searched is on the record",
       r.get("searched_radius_m") == radius)
    ok("she is told to say there is nothing NEARBY",
       "nothing nearby" in r["rules"].lower())
    ok("...and forbidden from reading a far one out as near",
       "further away" in r["rules"].lower())
    ok("...and from answering out of memory",
       "memory" in r["rules"].lower())
    ok("THE SEARCH IS NOT WIDENED — one request, no retry at a bigger radius",
       (Sent.body.get("locationRestriction") or {}).get("rectangle") is not None
       and places.haversine_m(
           Sent.body["locationRestriction"]["rectangle"]["high"]["latitude"],
           ORIGIN[1], ORIGIN[0], ORIGIN[1]) <= radius * 1.02)

    section(5, "A NAMED AREA IS THE DRIVER'S, NOT THE CAR'S")
    stub([at(20000, name="far but in the named place")])
    r = places.find_places("coffee", near="Santa Monica", where=WHERE)
    b = Sent.body
    ok("no restriction when the driver named the area",
       "locationRestriction" not in b)
    ok("...and no bias either", "locationBias" not in b)
    ok("...the area is in the query text", "Santa Monica" in b["textQuery"])
    ok("...and a far result is kept, because they asked for over there",
       r["n"] == 1, str(r["n"]))
    ok("...the record says no wall was applied",
       r["fix"]["constraint"] == "none", str(r["fix"]["constraint"]))

    section(6, "THE RECORD CAN BE ARGUED WITH AFTERWARDS")
    stub([at(1000)])
    r = places.find_places("coffee shops", where=WHERE)
    f = r["fix"]
    ok("the coordinate that was sent", f["lat"] == ORIGIN[0]
       and f["lng"] == ORIGIN[1])
    ok("...its age", f["age_s"] == 2.4, str(f["age_s"]))
    ok("...its accuracy", f["accuracy_m"] == 35.0, str(f["accuracy_m"]))
    ok("...where it came from", f["source"] == "browser_geolocation")
    ok("...the staleness threshold it was judged against",
       f["max_age_s"] == float(config.PLACES_FIX_MAX_AGE_S))
    ok("...whether it was a wall or a hint",
       f["constraint"] == "restriction", str(f["constraint"]))
    ok("...the radius", f["radius_m"] == radius, str(f["radius_m"]))
    ok("...and the ranking asked for", f["rank"] == "DISTANCE")
    ok("every result carries its distance",
       all(x.get("distance_m") is not None for x in r["results"]))

    section(7, "A STALE FIX IS STILL REFUSED")
    stub([at(1000)])
    old = dict(WHERE, age_s=config.PLACES_FIX_MAX_AGE_S + 1)
    r = places.find_places("coffee shops", where=old)
    ok("a fix past its age is not searched from", r.get("ok") is False,
       str(r.get("note")))
    ok("...and she is told to ask which area instead",
       bool(r.get("need_location")))

    print("\n" + "-" * 62)
    print(f"  {'PASS' if not failures else 'FAIL'}: "
          f"{failures} failure(s) of {checks} checks")
    return 1 if failures else 0


if __name__ == "__main__":
    sys.exit(main())
