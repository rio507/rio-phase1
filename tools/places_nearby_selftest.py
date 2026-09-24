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

AND THE INSTRUMENT CHANGED AGAIN ON 2026-09-24. 48c882d fixed this with a
`locationRestriction`, which was the wrong tool for a second reason nobody had
measured: Places does not honour `rankPreference=DISTANCE` under a restriction.
Asked for a cinema within 25 km it returned three in Burbank at ~25.8 km and
skipped the AMC eleven kilometres nearer; asked for coffee within 5 km it
skipped the two nearest. `maxResultCount` is 5, so WHICH five come back is the
whole answer.

So the bias is back -- as the way to ask the API for the NEAREST candidates --
and the wall is the haversine filter here, which is a true circle rather than a
box whose corners reach 1.41x, and is ours rather than a shape a vendor is
asked to honour. The original fault was a bias with no filter and no sort.

WHAT IT ASSERTS
    1. The request BIASES to the car and asks for distance ranking; the limit
       is enforced here, not there.
    2. The wall is a true circle: just inside is kept, just outside is dropped.
    3. Results come back sorted by distance, renumbered, far ones dropped.
    4. Nothing inside the radius is `nothing_close` -- BOTH ways it can happen
       -- the search is never widened to fill the silence, and she is pointed
       at scope='wider' instead of left stuck.
    5. A NAMED AREA is not constrained at all: "coffee in Santa Monica" asked
       from the Palisades is a question about Santa Monica.
    6. The result carries enough provenance to argue with it later.
    7. A stale fix is refused.
    8. SCOPE: nearby is a wall, wider is a bigger wall, anywhere has none --
       and an unreadable scope is the NARROW one.
    9. A category word that means a different category in plain English is
       constrained by type.
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

    section(1, "THE API CHOOSES THE NEAREST; WE ENFORCE THE LIMIT")
    stub([at(1000), at(2000)])
    r = places.find_places("coffee shops", where=WHERE)
    b = Sent.body
    # A BIAS, AND THAT IS NOT A RETURN TO THE BUG. 48c882d replaced the bias
    # with a locationRestriction and 2026-09-24 measured what that costs:
    # Places does not honour rankPreference=DISTANCE under a restriction, so
    # at 25 km it returned three Burbank cinemas and skipped the AMC eleven
    # kilometres nearer, and at 5 km it skipped the two nearest coffees. The
    # bias is how the API is asked for the NEAREST candidates; the wall is the
    # haversine filter below, which is ours and is a true circle.
    ok("the request carries a locationBias centred on the car",
       "locationBias" in b
       and abs((b["locationBias"]["circle"]["center"]["latitude"])
               - ORIGIN[0]) < 1e-9)
    ok("...and NOT a locationRestriction, which does not rank by distance",
       "locationRestriction" not in b)
    ok("...and asks for the nearest first",
       b.get("rankPreference") == "DISTANCE", str(b.get("rankPreference")))
    ok("...sized to the scope's radius",
       b["locationBias"]["circle"]["radius"] == radius,
       str(b["locationBias"]["circle"]["radius"]))

    section(2, "THE WALL IS OURS, AND IT IS A CIRCLE")
    # A bias is a suggestion -- the whole finding of 48c882d -- so the only
    # thing that can make "within 5 km" true is measuring it here.
    stub([at(radius * 0.99, 45.0, name="just inside"),
          at(radius * 1.01, 45.0, name="just outside")])
    r = places.find_places("coffee shops", where=WHERE)
    names = [x["name"] for x in r["results"]]
    ok("a result inside the radius is kept", "just inside" in names, str(names))
    ok("...and one just outside is dropped, whatever the bias let through",
       "just outside" not in names, str(names))
    ok("the wall is on the record as ours", r["fix"]["constraint"] == "wall",
       str(r["fix"]["constraint"]))

    section(3, "NEAREST FIRST, AND THE FAR ONES CUT OFF")
    stub([at(4000, name="four"), at(radius * 1.2, 45.0, name="far"),
          at(1000, name="one"), at(2500, name="two-five")])
    r = places.find_places("coffee shops", where=WHERE)
    names = [x["name"] for x in r["results"]]
    dists = [x["distance_m"] for x in r["results"]]
    ok("the far one is dropped though the bias let it through",
       "far" not in names, str(names))
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
    ok("THE SEARCH IS NOT WIDENED — one request, still at the near radius",
       (Sent.body.get("locationBias") or {}).get("circle", {}).get("radius")
       == radius,
       str((Sent.body.get("locationBias") or {}).get("circle", {}).get("radius")))
    ok("...and she is pointed at scope='wider' rather than left stuck",
       "wider" in r["rules"], r["rules"][-90:])

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
    ok("...whether a wall was applied", f["constraint"] == "wall",
       str(f["constraint"]))
    ok("...how wide the driver asked to look", f["scope"] == "nearby",
       str(f["scope"]))
    ok("...and what kind of place the words were taken to mean",
       "included_type" in f, str(f.get("included_type")))
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

    section(8, "SCOPE: THE WALL MOVES WHEN ASKED AND NEVER ON ITS OWN")
    wider = float(config.PLACES_WIDER_RADIUS_M)
    stub([at(1000), at(radius * 1.5, 90.0, name="mid"),
          at(wider * 1.5, 90.0, name="miles off")])
    r = places.find_places("cinema", where=WHERE)
    ok("the default is nearby, and it is the narrow one",
       r["fix"]["scope"] == "nearby" and r["fix"]["radius_m"] == radius,
       f'{r["fix"]["scope"]} {r["fix"]["radius_m"]}')
    ok("...so only the near one survives", [x["name"] for x in r["results"]]
       == ["Cafe 1000m"], str([x["name"] for x in r["results"]]))

    r = places.find_places("cinema", where=WHERE, scope="wider")
    names = [x["name"] for x in r["results"]]
    ok("scope='wider' raises the wall", r["fix"]["radius_m"] == wider,
       str(r["fix"]["radius_m"]))
    ok("...and reaches the one further out", "mid" in names, str(names))
    ok("...but it is STILL A WALL — 'further out' is a bigger circle, not "
       "the absence of one", "miles off" not in names, str(names))
    ok("...and the bias sent is the wider one",
       Sent.body["locationBias"]["circle"]["radius"] == wider)

    r = places.find_places("cinema", where=WHERE, scope="anywhere")
    names = [x["name"] for x in r["results"]]
    ok("scope='anywhere' drops the wall", r["fix"]["radius_m"] is None
       and r["fix"]["constraint"] == "none", str(r["fix"]["constraint"]))
    ok("...and keeps everything", "miles off" in names, str(names))
    ok("...still nearest-first, because 'anywhere' is not 'forget where I am'",
       [x["distance_m"] for x in r["results"]]
       == sorted(x["distance_m"] for x in r["results"]))

    # THE ONE FAILURE MODE A SCOPE PARAMETER INTRODUCES.
    r = places.find_places("cinema", where=WHERE, scope="ANYWHERE-ish")
    ok("AN UNREADABLE SCOPE IS THE NARROW ONE — a model sending something "
       "unexpected must not thereby widen a question asked about here",
       r["fix"]["scope"] == "nearby" and r["fix"]["radius_m"] == radius,
       f'{r["fix"]["scope"]} {r["fix"]["radius_m"]}')
    r = places.find_places("cinema", where=WHERE, scope=None)
    ok("...and so is no scope at all", r["fix"]["scope"] == "nearby")

    section(9, "A CATEGORY WORD THAT MEANS ANOTHER CATEGORY")
    # Measured on the drive: free text for "movie theater" returned Theatre
    # Palisades, a playhouse. See places._CATEGORY_TYPES for the both-ways
    # numbers behind every entry.
    stub([at(1000)])
    places.find_places("movie theater", where=WHERE)
    ok("a movie theater is constrained to movie_theater",
       Sent.body.get("includedType") == "movie_theater",
       str(Sent.body.get("includedType")))
    places.find_places("nearest cinema", where=WHERE)
    ok("...and so is a cinema", Sent.body.get("includedType") == "movie_theater")
    places.find_places("gas station", where=WHERE)
    ok("a gas station is constrained to gas_station, which is what kept the "
       "EV chargers out", Sent.body.get("includedType") == "gas_station",
       str(Sent.body.get("includedType")))
    places.find_places("coffee shops", where=WHERE)
    ok("COFFEE IS DELIBERATELY NOT TYPED — measured, the type dropped the "
       "nearest two", Sent.body.get("includedType") is None,
       str(Sent.body.get("includedType")))
    places.find_places("tacos", where=WHERE)
    ok("...and neither is anything the words already say",
       Sent.body.get("includedType") is None)
    r = places.find_places("cinema", near="Santa Monica", where=WHERE)
    ok("the type applies to a named area too — a playhouse in Santa Monica is "
       "the same collision", Sent.body.get("includedType") == "movie_theater")

    print("\n" + "-" * 62)
    print(f"  {'PASS' if not failures else 'FAIL'}: "
          f"{failures} failure(s) of {checks} checks")
    return 1 if failures else 0


if __name__ == "__main__":
    sys.exit(main())
