"""Local news: the ways a claim from a stranger gets spoken when it shouldn't.

    python -m tools.news_selftest
    python -m tools.news_selftest --offline   # no API calls, no model calls

Weather can only mislead by being old. News can be wrong at the source, right
but stale, right but about a town with the same name, or two outlets flatly
contradicting each other — and all four arrive looking exactly like a fact. So
the sections here are the four amendments, in the order they bite:

  A. honesty    undated is refused; past the intent's window is refused; a
                conflicting pair produces a hedge and never a confident pick;
                a lone social post is not news.
  B. entity     the dangerous result is not the irrelevant one, it is the one
                with the RIGHT name in the WRONG state. And a place question
                with no confirmed entity is a question, not a search.
  F. routing    every scope and both modes classify from the wording; an
                ambiguous "any news?" defaults to local and offers wider; an
                ambiguous "story" gives background and offers news; follow-ups
                retain scope.
  C. budget     the per-drive cap refuses rather than overspending, and the
                caps exist at all.

Sections A, B, F and C are pure logic and run offline — which is the point:
the honesty rules must hold without a network, or they are not rules. The live
section spends real money and is skipped by --offline.

Exit code is the number of failures.
"""
import argparse
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
sys.path.insert(0, str(Path(__file__).resolve().parent))

import assert_guard as _guard                              # noqa: E402
from dotenv import load_dotenv                              # noqa: E402

load_dotenv(Path(__file__).resolve().parent.parent / ".env")

import config                                               # noqa: E402
import localnews as ln                                      # noqa: E402

OK, BAD = "ok  ", "FAIL"
_fails = []
# A VERDICT WITH NO TOTAL CANNOT TELL YOU IT RAN. This counted failures only, so
# "PASS: 0 failure(s)" was printed by a suite asserting dozens of things and
# would have read identically from one asserting none -- which is how 23f4185,
# source_selftest.js and https_selftest's own unreachable section E all stayed
# hidden. tools/suite_sweep.py flags it as NO COUNT. A list rather than an int so
# ok() needs no `global`.
_checks = []


def ok(name, cond, extra=""):
    _checks.append(1)
    if cond:
        print(f"  {OK} {name}")
    else:
        _fails.append(name)
        print(f"  {BAD} {name}{('  ' + str(extra)) if extra else ''}")


# The guarded assertions live in one place now: tools/assert_guard.py, shared
# with the other suites that had the same shape. This file is where the failure
# was first seen, which is why the account of it is in that module's docstring.
ok_all, ok_none, non_empty = _guard.bind(ok, order="name_first")


def section(t):
    print(f"\n== {t}")


NOW = time.time()
LOC = {"ok": True, "neighborhood": "Downtown", "city": "Santa Monica",
       "county": "Los Angeles County", "state": "California",
       "latitude": 34.0195, "longitude": -118.4912}


def iso(ago_h):
    return datetime.fromtimestamp(NOW - ago_h * 3600, timezone.utc).isoformat()


def item(**over):
    base = {"headline": "Something happened", "source": "Daily Press",
            "url": "https://example.test/x", "published": iso(1),
            "summary": "A thing occurred downtown.", "category": "general",
            "city": "Santa Monica", "source_type": "news_org",
            "relevance": 0.8}
    base.update(over)
    return base


def cls_of(**over):
    c = {"scope": "local", "mode": "news", "timeframe": "24h",
         "category": "general", "topic": None, "offer_other": None}
    c.update(over)
    return c


# ---------------------------------------------------------------------------
# A. Honesty
# ---------------------------------------------------------------------------

def t_honesty():
    section("A. nothing is spoken that the fields cannot support")

    a = ln.audit([item(published=None)], cls_of(), LOC, now=NOW)
    ok("an undated result is refused",
       not a["kept"] and a["dropped"][0]["why"] == "undated",
       a["dropped"])

    for bad_date in ("", "   ", "sometime last week", "yesterday", None):
        a = ln.audit([item(published=bad_date)], cls_of(), LOC, now=NOW)
        ok(f"an unparseable date ({bad_date!r}) is refused, not guessed",
           not a["kept"])

    a = ln.audit([item(published=iso(72))], cls_of(timeframe="24h"), LOC, now=NOW)
    ok("older than the intent's window is refused",
       not a["kept"] and a["dropped"][0]["why"] == "too_old")

    # ...and the window really does move with the intent, or the rule above is
    # just a constant pretending to be a policy.
    a = ln.audit([item(published=iso(10))], cls_of(timeframe="6h",
                                                   category="traffic"),
                 LOC, now=NOW)
    ok("a 10h-old incident is refused on the 6h traffic clock", not a["kept"])
    a = ln.audit([item(published=iso(10))], cls_of(timeframe="24h"), LOC, now=NOW)
    ok("...and kept on the 24h general clock", len(a["kept"]) == 1)

    a = ln.audit([item(published=iso(-48))], cls_of(), LOC, now=NOW)
    ok("a future publication date is refused",
       not a["kept"] and a["dropped"][0]["why"] == "future_dated")

    # The conflicting pair. Neither may win.
    pair = [
        item(headline="Pier concert proceeding tonight as planned",
             summary="Organisers say the pier concert is going ahead tonight.",
             published=iso(3), category="events"),
        item(headline="Pier concert cancelled tonight", source="KCRW",
             summary="The pier concert tonight has been cancelled.",
             published=iso(2), category="events"),
    ]
    a = ln.audit(pair, cls_of(), LOC, now=NOW)
    ok("a contradicting pair is detected", len(a["conflicts"]) == 1, a["conflicts"])
    ok("...and BOTH survive, so neither can be silently picked",
       len(a["kept"]) == 2)
    ranked = ln.rank(a["kept"], cls_of())
    ok("...ranking does not drop one of them", len(ranked) == 2)

    # Unrelated items that happen to contain opposed words are NOT a conflict.
    unrelated = [
        item(headline="Third Street shop reopens after refit",
             summary="The shop on Third Street has reopened.", published=iso(2)),
        item(headline="Beach lot closed for resurfacing",
             summary="The beach parking lot is closed this week.",
             published=iso(3)),
    ]
    a = ln.audit(unrelated, cls_of(), LOC, now=NOW)
    ok("two unrelated items are not called a conflict",
       not a["conflicts"], a["conflicts"])

    a = ln.audit([item(source_type="social", headline="Saw a big crash myself",
                       summary="huge crash on Lincoln right now")],
                 cls_of(), LOC, now=NOW)
    ok("a lone social post is never news on its own",
       not a["kept"] and a["dropped"][0]["why"] == "social_only")

    a = ln.audit([item(source_type="social", headline="Saw a crash",
                       summary="crash on Lincoln"),
                  item(source_type="official",
                       headline="Lincoln Blvd collision, lanes blocked",
                       summary="Police report a collision on Lincoln Blvd.")],
                 cls_of(), LOC, now=NOW)
    ok("...but social alongside an official source survives as supplement",
       len(a["kept"]) == 2)

    # Every kept result carries what the rules require RIO to be able to say.
    a = ln.audit([item()], cls_of(), LOC, now=NOW)
    k = a["kept"][0]
    for f in ("source", "published", "published_ts", "age_s", "age_h",
              "geo_score", "geo_level", "source_score", "url"):
        ok(f"a kept result carries {f}", f in k)

    ok("the spoken rules tell her to state the age",
       "age_h" in ln.RULES_NEWS and "old" in ln.RULES_NEWS.lower())
    ok("the spoken rules forbid resolving a conflict",
       "may not pick" in ln.RULES_NEWS)
    ok("the spoken rules forbid filling a gap from memory",
       "memory" in ln.RULES_NEWS)


# ---------------------------------------------------------------------------
# B. Entity confusion
# ---------------------------------------------------------------------------

def t_entity():
    section("B. the right name in the wrong place is the dangerous result")

    a = ln.audit([item(headline="Santa Monica bar brawl",
                       city="Santa Monica, New Mexico")],
                 cls_of(), LOC, now=NOW)
    ok("a same-named town in another state is discarded",
       not a["kept"] and a["dropped"][0]["why"] == "elsewhere", a["dropped"])

    ok("...and so is its abbreviation form",
       ln.geo_match({"city": "Santa Monica, NM"}, LOC)[1] == "elsewhere")
    ok("CA and California are the same place",
       ln.geo_match({"city": "Santa Monica, CA"}, LOC)[1] == "city"
       and ln.geo_match({"city": "Santa Monica, California"}, LOC)[1] == "city")
    ok("the neighbourhood outranks the city",
       ln.geo_match({"city": "Downtown"}, LOC)[0]
       > ln.geo_match({"city": "Santa Monica"}, LOC)[0])
    ok("a city we have nothing to do with is elsewhere",
       ln.geo_match({"city": "Portland, Oregon"}, LOC)[1] == "elsewhere")
    ok("an unstated city is 'unstated', not a match and not a discard",
       ln.geo_match({"city": None}, LOC)[1] == "unstated")

    # The geographic bar only applies where geography is the question.
    a = ln.audit([item(city="Portland, Oregon")], cls_of(scope="topic"),
                 LOC, now=NOW)
    ok("a topic question does NOT discard on geography", len(a["kept"]) == 1)
    a = ln.audit([item(city="Portland, Oregon")], cls_of(scope="local"),
                 LOC, now=NOW)
    ok("...but a local one does", not a["kept"])

    # A place question with no confirmed entity must ASK, not search.
    r = ln.search_local_news(34.0195, -118.4912,
                             question="Any news about this place?",
                             session_key="_t_entity")
    ok("an unconfirmed 'this place' does not search",
       r.get("ok") is False and r.get("note") == "unconfirmed_entity",
       r.get("note"))
    ok("...and it asks instead", r.get("need_entity") is True)
    ok("...with rules that forbid guessing between similar names",
       "do not guess" in (r.get("rules") or "").lower()
       or "do NOT search" in (r.get("rules") or ""))

    # The place query uses the confirmed name AND its city -- never bare OCR.
    qs = ln.build_queries(LOC, cls_of(scope="place"), place_name="The Viper Room")
    ok("a confirmed place query quotes the entity",
       any('"The Viper Room"' in q for q in qs), qs)
    ok("...and pins it to the city",
       any("Santa Monica" in q for q in qs), qs)


# ---------------------------------------------------------------------------
# F. Scope and mode
# ---------------------------------------------------------------------------

SHAPES = [
    ("Any news around here?", "local", "news"),
    ("What's going on where I am?", "local", "news"),
    ("Anything happening nearby?", "local", "news"),
    ("Any news about this place?", "place", "news"),
    ("Why is traffic so bad here?", "local", "news"),
    ("What happened up ahead?", "local", "news"),
    ("What's in the news today?", "world", "news"),
    ("Anything big happening?", "world", "news"),
    ("What's happening with the Lakers?", "topic", "news"),
    ("Any news on that wildfire?", "topic", "news"),
    ("What's going on with tariffs?", "topic", "news"),
    ("Anything happening with the fires near us?", "mixed", "news"),
    ("What's the story of this place?", "place", "background"),
    ("What's this neighborhood known for?", "local", "background"),
    ("Why is that place famous?", "place", "background"),
    ("Tell me about this area.", "local", "background"),
    ("Any stories about this place?", "place", "news"),
    ("What's the story with that building?", "place", "news"),
    ("What's the deal with this place?", "place", "background"),
    ("Any news?", "local", "news"),
]


def t_routing():
    section("F. the words decide the scope and the mode")
    for q, scope, mode in SHAPES:
        c = ln.classify(q)
        ok(f"{q!r} -> {scope}/{mode}",
           c["scope"] == scope and c["mode"] == mode,
           f'got {c["scope"]}/{c["mode"]}')

    # The distinction the spec cares most about, stated as its own check: the
    # preposition really is the whole difference.
    ok("'stories ABOUT' is news, 'story OF' is background",
       ln.classify("Any stories about this place?")["mode"] == "news"
       and ln.classify("What's the story of this place?")["mode"] == "background")

    ok("an ambiguous 'any news?' defaults to local",
       ln.classify("Any news?")["scope"] == "local")
    ok("...and offers the wider view in the same breath",
       ln.classify("Any news?")["offer_other"] == "world")
    ok("an ambiguous 'story' gives background and offers the news",
       ln.classify("What's the deal with this place?")["mode"] == "background"
       and ln.classify("What's the deal with this place?")["offer_other"] == "news")

    # Timeframes by intent.
    for q, tf in (("Why is traffic bad?", "6h"),
                  ("What happened here yesterday?", "yesterday"),
                  ("Any news about this place?", "30d"),
                  ("What's the story of this place?", "unrestricted"),
                  ("Any breaking news right now?", "2h")):
        ok(f"{q!r} -> timeframe {tf}", ln.classify(q)["timeframe"] == tf,
           ln.classify(q)["timeframe"])

    # Follow-ups retain scope: the caller passes the previous scope, and the
    # override must actually override the words.
    r = ln.classify("What else?")
    ok("a bare follow-up alone classifies as local", r["scope"] == "local")
    # The override path is what carries scope across a follow-up.
    import inspect
    src = inspect.getsource(ln.search_local_news)
    ok("search_local_news accepts a scope override for follow-ups",
       "scope in (SCOPE_LOCAL" in src)

    # Ranking weights change when geography stops being part of the question.
    ok("topic/world ranking drops proximity",
       config.NEWS_WEIGHTS_FLAT["geo"] == 0.0)
    ok("...and raises source reliability",
       config.NEWS_WEIGHTS_FLAT["source"] > config.NEWS_WEIGHTS_GEO["source"])
    ok("...and the geographic weights are the spec's 40/30/20/10",
       [config.NEWS_WEIGHTS_GEO[k] for k in ("geo", "recency", "relevance", "source")]
       == [0.40, 0.30, 0.20, 0.10])

    # The spec's worked example: a close, fresh closure outranks a distant,
    # older city-government story.
    near = item(headline="Lincoln Blvd closed for utility work",
                summary="Utility work closes lanes on Lincoln Blvd.",
                published=iso(0.75), city="Santa Monica",
                source_type="official", relevance=0.95, category="construction")
    far = item(headline="County budget vote scheduled",
               summary="The county board will vote on the budget.",
               published=iso(18), city="Los Angeles County",
               source_type="news_org", relevance=0.5, category="government")
    a = ln.audit([far, near], cls_of(), LOC, now=NOW)
    ranked = ln.rank(a["kept"], cls_of())
    ok("a near fresh closure outranks a distant older story",
       ranked and "Lincoln" in ranked[0]["headline"],
       [r["headline"] for r in ranked])


# ---------------------------------------------------------------------------
# C. Budget
# ---------------------------------------------------------------------------

def t_budget():
    section("C. the budget refuses rather than overspending")
    ok("there is a per-question query cap",
       int(config.NEWS_MAX_QUERIES_PER_QUESTION) > 0)
    ok("there is a per-drive search cap",
       int(config.NEWS_MAX_SEARCHES_PER_DRIVE) > 0)

    qs = ln.build_queries(LOC, cls_of(), "")
    ok("the query builder respects the per-question cap",
       len(qs) <= int(config.NEWS_MAX_QUERIES_PER_QUESTION), qs)
    ok("...and expands smallest-area-first",
       qs and "Downtown" in qs[0], qs)

    key = "_t_budget"
    ln.reset_spend(key)
    ln._charge(key, int(config.NEWS_MAX_SEARCHES_PER_DRIVE), 0.5)
    r = ln.search_local_news(34.0195, -118.4912, question="Any news round here?",
                             session_key=key)
    ok("past the per-drive cap the tool refuses",
       r.get("ok") is False and r.get("note") == "budget_exhausted", r.get("note"))
    ok("...and tells her not to answer from memory",
       "memory" in (r.get("rules") or "").lower())
    ln.reset_spend(key)
    ok("a new drive resets the budget",
       ln.spend_of(key)["searches"] == 0)

    # A cached answer must cost nothing, or the cache is decoration.
    ok("background is cached for far longer than traffic",
       config.NEWS_TTL_BACKGROUND_S > config.NEWS_TTL_GENERAL_S
       > config.NEWS_TTL_TRAFFIC_S)


# ---------------------------------------------------------------------------
# Live
# ---------------------------------------------------------------------------

def t_live():
    section("LIVE. it actually answers, and the gate holds on real results")
    key = "_t_live"
    ln.forget(key)
    loc = ln.location_context(34.0195, -118.4912, key)
    ok("reverse geocoding resolves the area", loc.get("ok") is True, loc.get("note"))
    ok("...into the four components the spec names",
       all(loc.get(k) for k in ("city", "county", "state")),
       {k: loc.get(k) for k in ("neighborhood", "city", "county", "state")})
    print(f"     -> {ln.area_words(loc)}")

    t0 = time.time()
    r = ln.search_local_news(34.0195, -118.4912,
                             question="Any news around here?", session_key=key)
    took = time.time() - t0
    ok("a local news question answers", r.get("ok") is True, r.get("note"))
    if r.get("ok"):
        print(f"     {took:.1f}s, {r.get('searches')} searches, "
              f"${r.get('est_cost_usd')}, {r['n']} kept, "
              f"{r['dropped_n']} dropped")
        for x in r["results"][:4]:
            print(f"       {x['age_h']:>5}h {x['geo_level']:<13} "
                  f"{x['source'][:24]:<24} {x['headline'][:54]}")
        ok_all("every spoken result is dated", r["results"],
               lambda x: x.get("published_ts"))
        ok_all("every spoken result carries a source", r["results"],
               lambda x: (x.get("source") or "").strip())
        ok_all("every spoken result corroborates geographically", r["results"],
               lambda x: x["geo_score"] >= config.NEWS_MIN_GEO_MATCH,
               [(x["city"], x["geo_level"]) for x in r["results"]])
        ok_all("nothing older than the window survived", r["results"],
               lambda x: x["age_s"] <= ln.TIMEFRAME_S["24h"])

    # The second identical question must be free.
    r2 = ln.search_local_news(34.0195, -118.4912,
                              question="Any news around here?", session_key=key)
    ok("the same question again is served from cache",
       r2.get("cached") is True, r2.get("cached"))
    spend = ln.spend_of(key)
    ok("...and cost nothing more", spend["searches"] == r.get("searches", 0),
       spend)
    print(f"     drive spend so far: {spend}")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--offline", action="store_true")
    a = ap.parse_args()
    t_honesty()
    t_entity()
    t_routing()
    t_budget()
    if not a.offline:
        t_live()
    print("\n" + "-" * 58)
    print(f'  {"PASS" if not _fails else "FAIL"}: {len(_fails)} failure(s)'
          f' of {len(_checks)} checks')
    for f in _fails:
        print(f"    - {f}")
    return len(_fails)


if __name__ == "__main__":
    raise SystemExit(main())
