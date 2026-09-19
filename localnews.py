"""localnews.py — what is happening around the car, and what this place is.

THE DIFFERENCE FROM WEATHER, WHICH DECIDES THE WHOLE DESIGN
-----------------------------------------------------------
Weather is numbers from one authority. Ask twice, get the same answer; the only
way it can mislead is by being old, so `weather.py` needs one clock and one
distance and it is honest.

News is claims from strangers. It can be wrong at the source, right but stale,
right but about a different town, or two outlets flatly contradicting each
other — and every one of those arrives looking exactly like a fact. RIO's
credibility does not survive her stating something she cannot support, so
nothing here is spoken unless the fields underneath it say it may be:

    source          who said it
    published       WHEN they said it. No date, no sentence -- see audit()
    geo_match       what makes it relevant to HERE rather than to somewhere
                    with the same name
    source_type     an official notice and an anonymous post are not the same
                    claim and must not sound the same

`audit()` is where that is enforced, and it runs before ranking rather than
after, because a result that may not be spoken should not be able to win.

WHY THIS IS NOT deep_dive, HAVING MEASURED deep_dive
-----------------------------------------------------
It reaches the same model, through the same API, with the same key. It is not a
second search stack and there is no new vendor here. But it is a different CALL,
for two reasons that were observed rather than assumed. (Both were recorded
against a probe that was never committed -- this file and config.py both cited
`tools/news_probe.py`, which does not exist in any commit. The 1520-token /
30.6-second failure below is specific enough to be a real run; it is simply not
one anybody can re-run from this repo. See config.py's money block, where the
same missing citation had carried a wrong number with it.)

  escalate() returns PROSE. Amendment A requires every result to carry a source
  and a timestamp and requires RIO's sentence to be supportable by them. You
  cannot enforce "nothing undated is ever spoken" against a paragraph. This
  asks for a strict JSON schema instead, so the date is a field that can be
  refused rather than a phrase that can be believed.

  escalate()'s budget provably fails this shape. Asked the local-news question
  with its 320 + 1200 token ceiling it spent all 1520 tokens reasoning across
  six searches and returned nothing at all, after 30.6 seconds. That budget is
  sized for a question with one or two searches behind it, which is what
  deep_dive is for.

BACKGROUND is the exception and it DOES go to deep_dive's shape, because it is
not news: no dates to audit, no geography to corroborate, unrestricted
timeframe, and prose is the right output. See `background()`.

WHAT IT COSTS, WHICH IS WHY THE CAPS ARE NOT DECORATION
--------------------------------------------------------
Observed: a local news question is 3-6 web searches and 22-53 seconds — at
$10.00 per 1000 search calls plus 8-10k input tokens stuffed into context per
search. The searches and the seconds are counted from real responses.

THE DOLLAR FIGURE THAT USED TO BE HERE ($0.06-$0.15) WAS ARITHMETIC AT THE
WRONG PRICE and is withdrawn rather than restated. `est_cost_usd` below priced
tokens at gpt-5's tier while NEWS_MODEL is gpt-5.6-sol -- input understated 4x,
output 3x. Re-pricing the old numbers puts the same shape nearer $0.24-$0.31,
which is an estimate and is not going in a header as a measurement. config.py's
money block has the whole account. Nothing overspent: the per-drive cap has its
teeth on the SEARCH COUNT, not on the money.

Either way it is the most expensive thing RIO does by a wide margin -- hundreds
of times a weather refresh, for one question. So: a hard per-question search cap
in the instruction, a hard per-drive cap enforced in Python that refuses rather
than overspends, and a cache that makes the second question about the same cell
free. The token half of the bill is the LARGER half, which is an argument for
the query cap and not only for the search cap.

Nothing here runs per frame, and nothing here runs unasked. V1 answers when
asked; the proactive path is deliberately absent (amendment E).
"""
import json
import math
import os
import re
import threading
import time
from datetime import datetime, timezone

import httpx

import config

GEOCODE_URL = "https://maps.googleapis.com/maps/api/geocode/json"

# ---------------------------------------------------------------------------
# Scope and mode, decided from the words rather than by a model
# ---------------------------------------------------------------------------
# A classifier call would cost a round trip and a fraction of a cent to decide
# how to spend eight cents, which is defensible -- but it would also make the
# routing non-deterministic, and "why did it search the whole world when I
# asked about this street" is a question that needs an answer you can read.
# So it is rules, and the selftest pins every shape in the spec.

SCOPE_LOCAL, SCOPE_PLACE = "local", "place"
SCOPE_TOPIC, SCOPE_WORLD, SCOPE_MIXED = "topic", "world", "mixed"
MODE_NEWS, MODE_BACKGROUND = "news", "background"

# "here" in all the ways a driver says it.
_HERE = re.compile(
    r"\b(around here|round here|near ?by|nearby|near (?:me|us|here)|where i am|"
    r"where we are|in this area|this area|this neighou?rhood|this town|"
    r"this city|locally|local|on this (?:street|road)|up ahead|ahead|here)\b")

# "this place" -- a specific entity rather than an area.
_THIS_PLACE = re.compile(
    r"\b(this place|that place|this building|that building|this restaurant|"
    r"that restaurant|this hotel|that hotel|this bar|that bar|this spot|"
    r"that spot|this shop|that shop|this store|that store|this venue|"
    r"that venue|this club|that club)\b")

# Definitional wording -> BACKGROUND. "The story OF" is history; "any stories
# ABOUT" is news, and the preposition really is the whole distinction.
_BACKGROUND = re.compile(
    r"\b(the (?:story|history) (?:of|behind)|what'?s the story of|"
    r"known for|famous for|why is (?:it|this|that)(?:\s+\w+)? famous|"
    r"why'?s (?:it|this|that)(?:\s+\w+)? famous|tell me about|what is this place|"
    r"what'?s this place|history of|used to be|origin of)\b")

# Present/recent wording -> NEWS, even when the noun is "story".
_NEWSY = re.compile(
    r"\b(any (?:news|stories)|anything happening|what'?s happening|"
    r"what'?s going on|going on|what happened|breaking|latest|today|tonight|"
    r"this week|right now|currently|recently|in the news)\b")

# Traffic/incident wording -> LOCAL, category traffic, short timeframe.
_TRAFFIC = re.compile(
    r"\b(traffic|accident|crash|collision|road closure|closed|closure|"
    r"backed up|back ?up|jam|congestion|why is it slow|police|fire|"
    r"detour|construction|roadwork|road work|blocked)\b")

# World wording -> no geography at all.
_WORLD = re.compile(
    r"\b(in the news today|the news today|world news|national news|"
    r"anything big|big news|headlines|top stories|what'?s in the news|"
    r"around the world|globally)\b")

# "what's the deal with" is genuinely ambiguous and the spec says so: it gets
# background first, with the news offered after.
_AMBIGUOUS_STORY = re.compile(
    r"\b(what'?s the deal with|what'?s up with (?:this|that) place|"
    r"what about (?:this|that) place)\b")

# Categories the spec names, matched on the words a driver would use.
_CATEGORY_WORDS = [
    ("traffic", r"\b(traffic|congestion|backed up|jam|slow)\b"),
    ("accident", r"\b(accident|crash|collision|pile ?up)\b"),
    ("construction", r"\b(construction|roadwork|road work|utility work|paving)\b"),
    ("crime", r"\b(crime|shooting|robbery|stabbing|assault|burglary)\b"),
    ("emergency", r"\b(emergency|evacuat|wildfire|fire|flood|earthquake|alert)\b"),
    ("weather", r"\b(storm|rain|flooding|snow|wind)\b"),
    ("events", r"\b(event|concert|game|festival|parade|march|protest)\b"),
    ("government", r"\b(council|mayor|city hall|ordinance|vote|election)\b"),
    ("business", r"\b(opened|opening|closed down|business|restaurant|store)\b"),
]


def classify(question: str, has_place_context: bool = False) -> dict:
    """The driver's words -> {scope, mode, timeframe, category, topic}.

    Order matters and is the spec's, not convenience:

      1. background wording wins over everything, because "the story OF this
         place" contains "this place" and would otherwise look like news.
      2. explicit world wording wins next -- it is unambiguous and it is the
         one scope where geography must NOT constrain.
      3. traffic wording implies local with a short clock, whatever else is in
         the sentence: "why is traffic bad" is never a world question.
      4. "this place" -> place scope.
      5. "here" words -> local.
      6. a topic with no geography -> topic; a topic WITH geography -> mixed.
      7. nothing matched -> LOCAL, which is the spec's default and the
         differentiated case, with the wider view offered in the same breath.
    """
    q = " " + (question or "").lower().strip() + " "
    q = q.replace("’", "'")

    ambiguous_story = bool(_AMBIGUOUS_STORY.search(q))
    backgroundish = bool(_BACKGROUND.search(q))
    newsy = bool(_NEWSY.search(q))

    # 1. Background. `_NEWSY` can co-occur ("tell me about this place today"),
    # and when it does the definitional phrasing still wins -- the driver asked
    # what the place IS.
    mode = MODE_NEWS
    offer_other = None
    if backgroundish and not (newsy and not _BACKGROUND.search(q)):
        mode = MODE_BACKGROUND
        offer_other = "news"
    elif ambiguous_story:
        # Genuinely ambiguous: background first, then offer the news.
        mode = MODE_BACKGROUND
        offer_other = "news"

    category = "general"
    for name, pat in _CATEGORY_WORDS:
        if re.search(pat, q):
            category = name
            break

    # 2-7. Scope.
    topic = None
    if mode == MODE_BACKGROUND:
        scope = SCOPE_PLACE if (_THIS_PLACE.search(q) or has_place_context) \
            else SCOPE_LOCAL
    elif _WORLD.search(q) and not _HERE.search(q):
        scope = SCOPE_WORLD
    elif _TRAFFIC.search(q):
        scope = SCOPE_LOCAL
    elif _THIS_PLACE.search(q):
        scope = SCOPE_PLACE
    elif _HERE.search(q):
        # "fires near us" -- a named subject AND a here-word is MIXED.
        topic = _topic_of(q)
        scope = SCOPE_MIXED if topic else SCOPE_LOCAL
    else:
        topic = _topic_of(q)
        if topic:
            scope = SCOPE_TOPIC
        else:
            scope = SCOPE_LOCAL
            offer_other = offer_other or "world"

    timeframe = _timeframe_for(q, scope, category, mode)
    return {"scope": scope, "mode": mode, "timeframe": timeframe,
            "category": category, "topic": topic,
            # What to offer in the same breath when the answer is thin. The
            # spec's "Nothing much nearby. Want the bigger headlines?"
            "offer_other": offer_other}


_TOPIC_STOP = re.compile(
    r"\b(what|whats|what's|is|are|any|news|about|happening|going|on|with|the|"
    r"a|an|of|to|in|at|for|tell|me|latest|there|anything|stories|story|"
    r"do|does|did|you|know|hey|rio|please|so|update|updates)\b")


def _topic_of(q: str) -> str:
    """The subject of a topic question, or "" when there is not one.

    Deliberately crude: it strips the question scaffolding and keeps what is
    left if it looks like a subject. A wrong topic here costs a worse search,
    not a wrong claim -- the honesty gates are downstream and do not depend on
    this being right.
    """
    # Every candidate preposition, RIGHTMOST valid one wins. `re.search` takes
    # the leftmost match, which is wrong here twice over: the subject sits at
    # the end of the sentence, and "what's going ON with tariffs" would match
    # the particle of `going on` and lose the real topic after `with`.
    cand = ""
    # Match the PREPOSITION only and look back a word by hand. Matching
    # `word + preposition` together consumes the word after it too, so in
    # "going on with tariffs" the `going on` match swallows the space before
    # `with` and the real preposition is never examined.
    for m in re.finditer(r"\b(with|about|on|regarding)\s+", q):
        prep = m.group(1)
        before = (q[:m.start()].strip().split() or [""])[-1]
        # "what's going ON where I am" is a phrasal verb followed by a place,
        # not "on <subject>".
        if prep == "on" and before in {"going", "happening", "coming",
                                       "moving", "later", "early", "based"}:
            continue
        tail = q[m.end():].strip()
        if len(tail) >= 3:
            cand = tail
    if not cand:
        return ""
    # "what's going ON where I am" is a phrasal verb followed by a place, not
    # "on <subject>". Without this the particle of `going on` reads as the
    # preposition of a topic question and "where i am" becomes the subject --
    # which then classifies a plain local question as MIXED and searches for a
    # topic that does not exist.
    if prep == "on" and before in {"going", "happening", "coming", "moving",
                                   "later", "early", "based"}:
        return ""
    # A candidate made only of here-words names the place the driver already
    # is, which is the location context's job and not a topic.
    if not _HERE.sub(" ", " " + cand + " ").strip(" ,."):
        return ""
    cand = _TOPIC_STOP.sub(" ", cand)
    cand = re.sub(r"[^\w\s'-]+", " ", cand)
    cand = " ".join(cand.split())
    return cand if len(cand) >= 3 else ""


def _timeframe_for(q: str, scope: str, category: str, mode: str) -> str:
    """The spec's recency defaults, with the wording able to override them."""
    if mode == MODE_BACKGROUND:
        return "unrestricted"
    if re.search(r"\byesterday\b", q):
        return "yesterday"
    if re.search(r"\b(breaking|right now|just now|just happened)\b", q):
        return "2h"
    if scope == SCOPE_WORLD:
        # "What's in the news TODAY" means today. A week of headlines is not
        # the question and would bury the answer.
        return "24h"
    if scope in (SCOPE_TOPIC, SCOPE_MIXED):
        # The spec's "ongoing topic: 7 days", and it needs to be the DEFAULT
        # rather than something unlocked by the word "ongoing". Measured: "what's
        # happening with the Lakers" on a 24 hour window returned nothing at all
        # across five searches -- not because the gates rejected anything (zero
        # dropped) but because a subject that is merely continuing does not
        # generate a story every day. A topic question asked on a quiet day
        # should get the thread of it, not silence.
        return "2h" if re.search(r"\b(breaking|right now)\b", q) else "7d"
    if category in ("traffic", "accident", "emergency", "construction"):
        return "6h"
    if scope == SCOPE_PLACE:
        return "30d"
    return "24h"


TIMEFRAME_S = {"2h": 2 * 3600.0, "6h": 6 * 3600.0, "24h": 24 * 3600.0,
               "7d": 7 * 86400.0, "30d": 30 * 86400.0,
               "yesterday": 48 * 3600.0, "unrestricted": None}


# ---------------------------------------------------------------------------
# Where the car is, in words a search engine understands
# ---------------------------------------------------------------------------

_geo_cache = {}
_geo_lock = threading.Lock()


def _api_key() -> str:
    key = os.getenv("GOOGLE_MAPS_API_KEY", "").strip()
    if not key:
        raise RuntimeError("GOOGLE_MAPS_API_KEY is not set")
    return key


def haversine_m(lat1, lng1, lat2, lng2) -> float:
    r = 6371000.0
    p1, p2 = math.radians(lat1), math.radians(lat2)
    dp, dl = math.radians(lat2 - lat1), math.radians(lng2 - lng1)
    a = (math.sin(dp / 2) ** 2
         + math.cos(p1) * math.cos(p2) * math.sin(dl / 2) ** 2)
    return 2 * r * math.asin(math.sqrt(a))


_WANT = {
    "neighborhood": "neighborhood",
    "sublocality": "neighborhood",
    "locality": "city",
    "administrative_area_level_2": "county",
    "administrative_area_level_1": "state",
    "postal_code": "postal_code",
}


def location_context(lat, lng, session_key: str = "default") -> dict:
    """GPS -> {neighborhood, city, county, state}. One Geocoding request.

    This is the object the whole layer is built on, and it is deliberately the
    same object whether the next step is news, events, traffic or history --
    the spec's "local intelligence, not just news". Nothing below cares which.

    Cached on a coarse cell, because a reverse geocode a hundred metres later
    returns the same four words and there is no reason to buy them twice.
    """
    try:
        lat, lng = float(lat), float(lng)
    except (TypeError, ValueError):
        return {"ok": False, "note": "no_fix"}
    if not (math.isfinite(lat) and math.isfinite(lng)):
        return {"ok": False, "note": "no_fix"}

    cell = _cell_of(lat, lng)
    with _geo_lock:
        got = _geo_cache.get(cell)
        if got and (time.time() - got["fetched_at"]) < config.NEWS_GEO_TTL_S:
            out = dict(got)
            out["cached"] = True
            return out

    try:
        r = httpx.get(GEOCODE_URL, timeout=config.NEWS_GEO_TIMEOUT_S,
                      params={"latlng": f"{lat},{lng}", "key": _api_key()})
        r.raise_for_status()
        data = r.json()
    except Exception as e:
        print(f"[news] reverse geocode failed: {type(e).__name__}: {e}",
              flush=True)
        return {"ok": False, "note": f"{type(e).__name__}"}

    if data.get("status") != "OK" or not data.get("results"):
        return {"ok": False, "note": f"geocode_{data.get('status', 'EMPTY')}"}

    # Scan EVERY result, not just the first. Google returns the same point at a
    # dozen granularities and the neighbourhood is frequently not on the
    # street-address one -- taking results[0] alone silently loses the smallest
    # area, which is the one the spec expands outward FROM.
    found = {}
    for res in data["results"]:
        for comp in res.get("address_components") or []:
            for t in comp.get("types") or []:
                key = _WANT.get(t)
                if key and key not in found:
                    found[key] = comp.get("long_name")

    ctx = {
        "ok": True,
        "latitude": round(lat, 5), "longitude": round(lng, 5),
        "neighborhood": found.get("neighborhood"),
        "city": found.get("city"),
        "county": found.get("county"),
        "state": found.get("state"),
        "postal_code": found.get("postal_code"),
        "label": (data["results"][0].get("formatted_address") or ""),
        "fetched_at": time.time(),
        "cached": False,
        "billed_requests": 1,
    }
    with _geo_lock:
        _geo_cache[cell] = dict(ctx)
        if len(_geo_cache) > 256:
            for k, _v in sorted(_geo_cache.items(),
                                key=lambda kv: kv[1]["fetched_at"])[:128]:
                _geo_cache.pop(k, None)
    return ctx


def _cell_of(lat, lng) -> str:
    """A coarse geographic cell, for the caches.

    ~0.01 degrees is roughly a kilometre of latitude: small enough that the
    neighbourhood name is still right, large enough that a car at traffic
    speed does not mint a new cell every few seconds.
    """
    return f"{round(float(lat), 2):.2f},{round(float(lng), 2):.2f}"


def area_words(loc: dict) -> str:
    """The location as a search engine wants it: smallest area, then outward."""
    bits = [loc.get("neighborhood"), loc.get("city"), loc.get("county"),
            loc.get("state")]
    return ", ".join([b for b in bits if b])


# ---------------------------------------------------------------------------
# The queries
# ---------------------------------------------------------------------------

def build_queries(loc: dict, cls: dict, place_name: str = "") -> list:
    """Structured context -> several searches, smallest area outward.

    The spec's geographic expansion is here rather than as a retry loop: the
    queries are ordered neighbourhood-first and the model runs them in order,
    so a city-wide result only ever arrives after the neighbourhood has been
    asked. Expansion is a property of the QUERY LIST, which means a log of the
    list is a log of how far RIO had to go to find anything.
    """
    scope, cat = cls["scope"], cls["category"]
    city = loc.get("city") or ""
    hood = loc.get("neighborhood") or ""
    county = loc.get("county") or ""
    state = loc.get("state") or ""
    topic = cls.get("topic") or ""
    qs = []

    if scope == SCOPE_WORLD:
        qs = ["top news headlines today", "major world news today"]
    elif scope == SCOPE_TOPIC:
        qs = [f"{topic} news latest", f"{topic} news today"]
    elif scope == SCOPE_MIXED:
        where = city or county or state
        qs = [f"{topic} {where} news today", f"{topic} near {where} latest",
              f"{topic} news today"]
    elif scope == SCOPE_PLACE and place_name:
        # The CONFIRMED entity plus its city -- amendment B. Never the raw
        # words off a sign, which is how you end up reading news about a
        # same-named restaurant in another state.
        qs = [f'"{place_name}" {city} news',
              f'"{place_name}" news latest']
    elif cat in ("traffic", "accident", "construction"):
        road = loc.get("road") or ""
        base = f"{road} {city}".strip() if road else city
        qs = [f"{base} traffic accident today",
              f"{base} road closure today",
              f"{city} traffic incidents today",
              f"{city} police activity today"]
    else:
        if hood and city:
            qs.append(f"{hood} {city} news")
        if city:
            qs += [f"{city} news today", f"{city} breaking news today"]
            if cat == "events":
                qs.append(f"{city} events today")
            else:
                qs.append(f"{city} road closures today")
        if county and not city:
            qs.append(f"{county} news today")
        if not qs and state:
            qs.append(f"{state} news today")

    # Dedupe, preserve order, and CAP. The cap is a cost control with teeth:
    # every query is a search and every search is a cent plus ~8k input tokens.
    seen, out = set(), []
    for q in qs:
        q = " ".join(q.split())
        if q and q.lower() not in seen:
            seen.add(q.lower())
            out.append(q)
    return out[:int(config.NEWS_MAX_QUERIES_PER_QUESTION)]


# ---------------------------------------------------------------------------
# Retrieval: the same vendor deep_dive uses, asked for fields instead of prose
# ---------------------------------------------------------------------------

RESULT_SCHEMA = {
    "type": "object", "additionalProperties": False,
    "properties": {"results": {"type": "array", "items": {
        "type": "object", "additionalProperties": False,
        "properties": {
            "headline": {"type": "string"},
            "source": {"type": "string"},
            "url": {"type": "string"},
            "published": {"type": ["string", "null"]},
            "summary": {"type": "string"},
            "category": {"type": "string"},
            "city": {"type": ["string", "null"]},
            "source_type": {"type": "string"},
            "relevance": {"type": "number"},
        },
        "required": ["headline", "source", "url", "published", "summary",
                     "category", "city", "source_type", "relevance"]}}},
    "required": ["results"]}

RETRIEVE_INSTRUCTIONS = (
    "You are the retrieval step behind a car assistant. You search and you "
    "report what you found as structured data. You never write prose and you "
    "never answer the driver.\n"
    "THE DATE RULE, which matters more than coverage: only include an item if "
    "the page carries a REAL publication date that you actually saw. Put it in "
    "`published` as ISO 8601 with a timezone offset where the page gives one. "
    "NEVER guess, infer or approximate a date. If you cannot date it, leave it "
    "out entirely — an undated item is not a result, it is a liability.\n"
    "Do NOT include: status pages, 'no incidents currently reported' pages, "
    "live dashboards, directory listings, event calendars with no publication "
    "date, or a homepage. Those are pages, not news.\n"
    "`source` is the publishing organisation as a person would name it. "
    "`source_type` is one of: official (government, police, fire, transit "
    "agency, the business itself), news_org (an established newsroom), "
    "aggregator, blog, social, unknown.\n"
    "`city` is the city the item is ABOUT, not the city the outlet is in. Null "
    "if it is not about a place.\n"
    "`relevance` is 0 to 1: how well this item answers the question asked.\n"
    "Return an empty list if nothing qualifies. An empty list is a correct and "
    "useful answer and is much better than a padded one."
)


def _client():
    from openai import OpenAI

    return OpenAI(timeout=float(config.NEWS_TIMEOUT_S))


def retrieve(queries: list, cls: dict, loc: dict, place_name: str = "") -> dict:
    """Run the searches, get structured results back. One model call.

    The search count is CAPPED in the instruction and COUNTED in the response,
    so what it actually cost is measured rather than assumed. The counted
    number is what the per-drive budget is debited by -- an instruction is a
    request, and a budget that trusted a request would not be a budget.
    """
    t0 = time.time()
    tf = cls["timeframe"]
    when = {"2h": "in the last 2 hours", "6h": "in the last 6 hours",
            "24h": "in the last 24 hours", "7d": "in the last 7 days",
            "30d": "in the last 30 days",
            "yesterday": "yesterday"}.get(tf, "recently")

    where = area_words(loc) if cls["scope"] in (
        SCOPE_LOCAL, SCOPE_PLACE, SCOPE_MIXED) else ""
    ask = [f"Run these searches (at most "
           f"{int(config.NEWS_MAX_QUERIES_PER_QUESTION)}), then report what "
           f"qualifies:"]
    ask += [f"  - {q}" for q in queries]
    ask.append(f"\nOnly items published {when}.")
    if where:
        ask.append(f"The car is at: {where}. Items must be about that area — "
                   f"a same-named place somewhere else is not a result.")
    if place_name:
        ask.append(f"The subject is specifically: {place_name}"
                   + (f" in {loc.get('city')}" if loc.get("city") else ""))

    try:
        r = _client().responses.create(
            model=config.NEWS_MODEL,
            instructions=RETRIEVE_INSTRUCTIONS,
            input="\n".join(ask),
            tools=[{"type": "web_search"}],
            text={"format": {"type": "json_schema", "name": "news_results",
                             "schema": RESULT_SCHEMA, "strict": True}},
            max_output_tokens=int(config.NEWS_MAX_TOKENS),
        )
    except Exception as e:
        print(f"[news] retrieve failed: {type(e).__name__}: {e}", flush=True)
        return {"ok": False, "note": f"{type(e).__name__}",
                "took_ms": round((time.time() - t0) * 1000, 1), "searches": 0}

    searches = sum(1 for i in (getattr(r, "output", None) or [])
                   if getattr(i, "type", "") == "web_search_call")
    usage = getattr(r, "usage", None)
    raw = (getattr(r, "output_text", "") or "").strip()
    try:
        results = (json.loads(raw) or {}).get("results") or []
    except json.JSONDecodeError:
        print(f"[news] unparseable retrieval payload ({len(raw)} chars)",
              flush=True)
        return {"ok": False, "note": "unreadable_results", "searches": searches,
                "took_ms": round((time.time() - t0) * 1000, 1)}

    return {
        "ok": True, "results": results, "searches": searches,
        "took_ms": round((time.time() - t0) * 1000, 1),
        "input_tokens": getattr(usage, "input_tokens", None),
        "output_tokens": getattr(usage, "output_tokens", None),
        "est_cost_usd": round(
            searches * config.NEWS_SEARCH_COST_USD
            + (getattr(usage, "input_tokens", 0) or 0) * config.NEWS_IN_COST_USD
            + (getattr(usage, "output_tokens", 0) or 0) * config.NEWS_OUT_COST_USD,
            4),
    }


# ---------------------------------------------------------------------------
# audit() — amendment A. Nothing reaches ranking that may not be spoken.
# ---------------------------------------------------------------------------

def _parse_ts(s):
    if not s or not isinstance(s, str):
        return None
    t = s.strip().replace("Z", "+00:00")
    for fmt in (None, "%Y-%m-%d", "%Y-%m-%dT%H:%M:%S", "%Y-%m-%d %H:%M:%S"):
        try:
            dt = datetime.fromisoformat(t) if fmt is None \
                else datetime.strptime(t, fmt)
            if dt.tzinfo is None:
                dt = dt.replace(tzinfo=timezone.utc)
            return dt.timestamp()
        except (ValueError, TypeError):
            continue
    return None


# Opposing status words. Narrow ON PURPOSE: a general contradiction detector is
# a research project, and a wide one that fires on unrelated pairs would make
# RIO hedge everything, which is its own kind of dishonesty. This catches the
# case that actually happens locally -- two outlets disagreeing about whether
# something is open, cancelled, or contained.
_OPPOSED = [
    ({"closed", "closure", "shut", "blocked"}, {"open", "reopened", "cleared", "reopens"}),
    ({"cancelled", "canceled", "called off", "postponed"}, {"proceeding", "going ahead", "on as planned", "confirmed"}),
    ({"contained", "under control"}, {"spreading", "out of control", "growing"}),
    ({"arrested", "charged", "in custody"}, {"released", "cleared", "no charges"}),
    ({"injured", "hurt", "casualties"}, {"no injuries", "unharmed", "nobody hurt"}),
    ({"confirmed", "verified"}, {"denied", "disputed", "unconfirmed"}),
]

_SUBJECT_STOP = frozenset(
    "the a an of in on at to for and or is are was were be been by with from "
    "this that these those it its as after before over under near about new "
    "today yesterday says said report reports reported".split())


def _subject_tokens(r: dict) -> set:
    text = f"{r.get('headline','')} {r.get('summary','')}".lower()
    words = re.findall(r"[a-z][a-z'-]{3,}", text)
    return {w for w in words if w not in _SUBJECT_STOP}


def find_conflicts(results: list) -> list:
    """Pairs of results that say opposite things about the same subject.

    Returns [(i, j, "closed vs open"), ...]. The caller's instruction when this
    is non-empty is to report the disagreement or say nothing -- never to pick
    the more convenient one, which is what a ranked list would otherwise do
    silently by putting one of them first.
    """
    out = []
    for i in range(len(results)):
        ti = f"{results[i].get('headline','')} {results[i].get('summary','')}".lower()
        si = _subject_tokens(results[i])
        for j in range(i + 1, len(results)):
            tj = f"{results[j].get('headline','')} {results[j].get('summary','')}".lower()
            sj = _subject_tokens(results[j])
            # Same story, roughly: they have to be ABOUT the same thing before
            # opposite words mean a contradiction rather than two topics.
            if len(si & sj) < 2:
                continue
            for left, right in _OPPOSED:
                lh = any(w in ti for w in left)
                rh = any(w in tj for w in right)
                lh2 = any(w in tj for w in left)
                rh2 = any(w in ti for w in right)
                if (lh and rh) or (lh2 and rh2):
                    a = next(w for w in left if w in (ti if lh else tj))
                    b = next(w for w in right if w in (tj if rh else ti))
                    out.append((i, j, f"{a} vs {b}"))
                    break
    return out


# US state abbreviations, so "Santa Monica, CA" and "Santa Monica, California"
# are the same claim. Bounded data rather than cleverness: the alternative is a
# substring test, and a substring test is what let "Santa Monica, New Mexico"
# through as a match for Santa Monica, California the first time this was run.
_US_ABBR = {
    "alabama": "al", "alaska": "ak", "arizona": "az", "arkansas": "ar",
    "california": "ca", "colorado": "co", "connecticut": "ct",
    "delaware": "de", "florida": "fl", "georgia": "ga", "hawaii": "hi",
    "idaho": "id", "illinois": "il", "indiana": "in", "iowa": "ia",
    "kansas": "ks", "kentucky": "ky", "louisiana": "la", "maine": "me",
    "maryland": "md", "massachusetts": "ma", "michigan": "mi",
    "minnesota": "mn", "mississippi": "ms", "missouri": "mo", "montana": "mt",
    "nebraska": "ne", "nevada": "nv", "new hampshire": "nh",
    "new jersey": "nj", "new mexico": "nm", "new york": "ny",
    "north carolina": "nc", "north dakota": "nd", "ohio": "oh",
    "oklahoma": "ok", "oregon": "or", "pennsylvania": "pa",
    "rhode island": "ri", "south carolina": "sc", "south dakota": "sd",
    "tennessee": "tn", "texas": "tx", "utah": "ut", "vermont": "vt",
    "virginia": "va", "washington": "wa", "west virginia": "wv",
    "wisconsin": "wi", "wyoming": "wy",
    "district of columbia": "dc",
}
_ABBR_TO_STATE = {v: k for k, v in _US_ABBR.items()}


def _canon_region(s: str) -> str:
    """"CA" and "California" -> the same string. Anything else, itself."""
    t = " ".join((s or "").lower().replace(".", "").split())
    return _ABBR_TO_STATE.get(t, t)


def geo_match(result: dict, loc: dict) -> tuple:
    """How this result connects to where the car is. -> (score, level)

    A LEVEL rather than a distance, and the honesty is in admitting that.
    Computing real distances would mean geocoding every result's city -- a
    billed request per item to decorate one sentence, which is the same trade
    places.py refuses for drive times. The level is what the ranking actually
    needs, and it is what amendment B's geographic corroboration test needs
    too, so one mechanism serves both.
    """
    raw = (result.get("city") or "").lower().strip()
    if not raw:
        return 0.15, "unstated"
    parts = [p.strip() for p in raw.split(",") if p.strip()]
    canon = {_canon_region(p) for p in parts}
    ours = {k: " ".join((loc.get(k) or "").lower().split())
            for k in ("neighborhood", "city", "county", "state")}

    # THE REGION TEST, FIRST. When a result names its region -- "Santa Monica,
    # New Mexico" -- and that region is not ours, it is elsewhere no matter how
    # exactly the town name matches. This is the whole of amendment B: the
    # dangerous result is not the irrelevant one, it is the one with the right
    # name in the wrong state, because it survives every other check and reads
    # as local.
    if len(parts) >= 2:
        tail = _canon_region(parts[-1])
        our_state = _canon_region(ours["state"])
        if our_state and tail and tail != our_state \
                and tail not in (ours["county"], ours["city"]):
            return 0.0, "elsewhere"

    # Exact component equality, never substring: "Santa Monica" must BE one of
    # the parts, not merely appear somewhere inside the string.
    for key, score, level in (("neighborhood", 1.0, "neighborhood"),
                              ("city", 0.85, "city"),
                              ("county", 0.5, "county"),
                              ("state", 0.2, "state")):
        v = ours[key]
        if v and (v in parts or _canon_region(v) in canon):
            return score, level
    return 0.0, "elsewhere"


def audit(results: list, cls: dict, loc: dict, now: float = None) -> dict:
    """The gate. -> {kept, dropped, conflicts}

    Runs BEFORE ranking, deliberately. A result that may not be spoken should
    never be in a position to be the top one -- ranking first and filtering
    after is how the best-scoring unusable item becomes the answer.

    Every drop is recorded with a reason. A question that found nothing and a
    question whose every result was undated are very different events and must
    not look the same in a log.
    """
    now = time.time() if now is None else now
    window = TIMEFRAME_S.get(cls.get("timeframe"))
    geo_scoped = cls.get("scope") in (SCOPE_LOCAL, SCOPE_PLACE, SCOPE_MIXED)
    kept, dropped = [], []

    for r in results:
        if not isinstance(r, dict) or not (r.get("headline") or "").strip():
            dropped.append({"why": "empty", "headline": ""})
            continue
        head = r.get("headline", "")

        # 1. UNDATED IS REFUSED. The single hardest rule in this file.
        ts = _parse_ts(r.get("published"))
        if ts is None:
            dropped.append({"why": "undated", "headline": head,
                            "source": r.get("source")})
            continue

        age = now - ts
        # A publication date in the future is a parse artefact or a bad page;
        # either way it is not something to reason about. A small tolerance
        # covers clock skew and timezone-less pages.
        if age < -config.NEWS_FUTURE_TOLERANCE_S:
            dropped.append({"why": "future_dated", "headline": head})
            continue
        age = max(0.0, age)

        # 2. OLDER THAN THE INTENT'S WINDOW IS REFUSED. "What's happening" and
        # a three-day-old story are not the same question answered well.
        if window is not None and age > window:
            dropped.append({"why": "too_old", "headline": head,
                            "age_h": round(age / 3600.0, 1),
                            "window": cls.get("timeframe")})
            continue

        # 3. GEOGRAPHIC CORROBORATION (amendment B). Only where geography is
        # what made the question local in the first place.
        gscore, glevel = geo_match(r, loc)
        if geo_scoped and gscore < config.NEWS_MIN_GEO_MATCH:
            dropped.append({"why": "elsewhere", "headline": head,
                            "city": r.get("city"), "level": glevel})
            continue

        out = dict(r)
        out["published_ts"] = ts
        out["age_s"] = round(age, 1)
        out["age_h"] = round(age / 3600.0, 1)
        out["geo_score"] = gscore
        out["geo_level"] = glevel
        out["source_score"] = config.NEWS_SOURCE_WEIGHTS.get(
            (r.get("source_type") or "unknown").lower(), 0.3)
        kept.append(out)

    conflicts = find_conflicts(kept)

    # 4. A LONE SOCIAL POST IS NOT NEWS. Supplemental evidence, the spec says,
    # and amendment A says never presented as fact on its own. So if everything
    # that survived is social, nothing survived.
    if kept and all((k.get("source_type") or "").lower() == "social"
                    for k in kept):
        dropped += [{"why": "social_only", "headline": k.get("headline")}
                    for k in kept]
        kept = []

    return {"kept": kept, "dropped": dropped, "conflicts": conflicts}


# ---------------------------------------------------------------------------
# Ranking
# ---------------------------------------------------------------------------

def rank(kept: list, cls: dict, now: float = None) -> list:
    """Score and sort. Weights are config, and they move with scope.

    Proximity's 40% only makes sense when geography is part of the question.
    For topic and world it is redistributed to recency and relevance, and
    source reliability rises -- because without geography there is nothing
    corroborating the claim except who made it, which is exactly the spec's
    reasoning and amendment F's warning.
    """
    now = time.time() if now is None else now
    w = (config.NEWS_WEIGHTS_GEO if cls.get("scope") in
         (SCOPE_LOCAL, SCOPE_PLACE, SCOPE_MIXED) else config.NEWS_WEIGHTS_FLAT)
    window = TIMEFRAME_S.get(cls.get("timeframe")) or (7 * 86400.0)

    for r in kept:
        recency = max(0.0, 1.0 - (r["age_s"] / window)) if window else 0.5
        rel = r.get("relevance")
        rel = float(rel) if isinstance(rel, (int, float)) else 0.5
        r["score"] = round(
            w["geo"] * r["geo_score"] + w["recency"] * recency
            + w["relevance"] * max(0.0, min(1.0, rel))
            + w["source"] * r["source_score"], 4)
    kept.sort(key=lambda r: -r["score"])
    return kept


# ---------------------------------------------------------------------------
# The cache, and the budget that can actually refuse
# ---------------------------------------------------------------------------
# Two different jobs that look similar. The cache stops RIO buying the same
# answer twice; the budget stops a drive buying too many different ones. Only
# the second can say no.

_news_cache = {}                 # (cell, scope, mode, category) -> payload
_news_lock = threading.Lock()
_spend = {}                      # session_key -> {"searches", "questions", "usd"}
_spend_lock = threading.Lock()


def _ttl_for(cls: dict) -> float:
    if cls.get("mode") == MODE_BACKGROUND:
        return float(config.NEWS_TTL_BACKGROUND_S)
    cat = cls.get("category")
    if cat in ("traffic", "accident", "construction"):
        return float(config.NEWS_TTL_TRAFFIC_S)
    if cat == "emergency":
        return float(config.NEWS_TTL_EMERGENCY_S)
    if cat == "events":
        return float(config.NEWS_TTL_EVENTS_S)
    return float(config.NEWS_TTL_GENERAL_S)


def _cache_key(loc: dict, cls: dict, place_name: str) -> tuple:
    geo = _cell_of(loc.get("latitude", 0), loc.get("longitude", 0)) \
        if cls["scope"] in (SCOPE_LOCAL, SCOPE_MIXED) else "-"
    return (geo, cls["scope"], cls["mode"], cls["category"],
            (cls.get("topic") or "").lower(), (place_name or "").lower())


def _cached(key, cls):
    with _news_lock:
        got = _news_cache.get(key)
        if not got:
            return None
        if (time.time() - got["fetched_at"]) > _ttl_for(cls):
            return None
        return dict(got)


def _remember(key, payload):
    with _news_lock:
        _news_cache[key] = payload
        if len(_news_cache) > 128:
            for k, _v in sorted(_news_cache.items(),
                                key=lambda kv: kv[1]["fetched_at"])[:64]:
                _news_cache.pop(k, None)


def spend_of(session_key: str) -> dict:
    with _spend_lock:
        return dict(_spend.get(str(session_key or "default"))
                    or {"searches": 0, "questions": 0, "usd": 0.0})


def _charge(session_key: str, searches: int, usd: float) -> None:
    with _spend_lock:
        s = _spend.setdefault(str(session_key or "default"),
                              {"searches": 0, "questions": 0, "usd": 0.0})
        s["searches"] += int(searches or 0)
        s["questions"] += 1
        s["usd"] = round(s["usd"] + float(usd or 0.0), 4)


def reset_spend(session_key: str = None) -> None:
    """A new drive starts with a full budget. Tests use it too."""
    with _spend_lock:
        if session_key is None:
            _spend.clear()
        else:
            _spend.pop(str(session_key), None)


def forget(session_key: str = None) -> None:
    with _news_lock:
        _news_cache.clear()
    with _geo_lock:
        _geo_cache.clear()
    reset_spend(session_key)


# ---------------------------------------------------------------------------
# What RIO is allowed to do with what comes back
# ---------------------------------------------------------------------------

RULES_NEWS = (
    "Answer from these results and nothing else. Every fact you say must be in "
    "one of them: you did not know any of this before you read it, and what "
    "you remember from training about this place is older than what is here.\n"
    "ONE TO THREE SENTENCES, the thing itself rather than a list of headlines. "
    "Not 'here are the top stories' — say what is actually going on, the way a "
    "passenger who just looked at their phone would. If there is more, offer "
    "it in a short clause rather than continuing into it.\n"
    "SAY HOW OLD IT IS WHEN IT MATTERS. `age_h` is hours since publication. "
    "Anything where the age changes what the driver should do — an incident, a "
    "closure, a fire, anything still unfolding — gets its age said out loud: "
    "'reported about an hour ago'. Do not make something sound like it is "
    "happening now when it was this morning.\n"
    "ATTRIBUTE WHEN IT IS CONTESTED OR CONSEQUENTIAL. Name the source for "
    "anything political, anything about blame or a person, anything about "
    "safety: 'the city says', 'the Daily Press reported'. For the ordinary "
    "stuff — a farmers market, a road being resurfaced — just say it.\n"
    "IF `conflicts` IS NOT EMPTY, the sources disagree. Say so, briefly, or "
    "say nothing about that item. You may not pick the tidier version, and you "
    "may not resolve it yourself.\n"
    "Never read a URL out loud. Never say 'according to several reports' as "
    "padding — only when the sourcing genuinely is the point.\n"
    "If the list is empty, say plainly that there is nothing much, and offer "
    "what `offer_other` suggests if it is set. Do NOT fill the gap from "
    "memory, and do not reach for something you half-remember about the area."
)

RULES_BACKGROUND = (
    "This is background, not news: what the place is and why it is interesting. "
    "Two or three sentences of what actually makes it worth knowing — the way "
    "somebody who lives in the city would tell you in the car, not an "
    "encyclopedia entry and not a founding date unless they asked.\n"
    "You may use what you know as well as what was retrieved. The honesty "
    "rules do not relax because there is no timestamp: say what is known and "
    "what is not, never invent a date, a number or an event, and if you are "
    "not certain of something say so in the sentence rather than stating it "
    "and hoping. 'Widely reported' and 'I'm not certain' are different claims "
    "and a driver can hear the difference.\n"
    "Anything contested — who something is named after, who did what first, "
    "anything political about the area — is attributed or left out.\n"
    "If it has also been in the news recently and that was retrieved, offer "
    "that at the end in one short clause rather than folding it in."
)

RULES_NO_ENTITY = (
    "You do not know which place the driver means, and Google could not "
    "confirm one. ASK — one short question, naming what you think you saw if "
    "you saw anything. Do NOT search, do not guess between two places with "
    "similar names, and do not say anything about a business you have not "
    "confirmed. A confident answer about the wrong branch of the wrong chain "
    "in the wrong state is the exact failure this is guarding."
)

RULES_BUDGET = (
    "You have looked things up enough times this drive that the budget for it "
    "is gone. Say plainly that you cannot look anything else up right now — in "
    "your own words, without explaining budgets or tools — and answer from "
    "what is already in the conversation if you can. Do NOT answer a news "
    "question from memory."
)


# ---------------------------------------------------------------------------
# The tool
# ---------------------------------------------------------------------------

def search_local_news(latitude=None, longitude=None, question: str = "",
                      place_name: str = "", scope: str = "", mode: str = "",
                      categories=None, timeframe: str = "",
                      session_key: str = "default",
                      route_ahead: list = None) -> dict:
    """The one entry point. Question + GPS -> what RIO may say, and the rules.

    `scope` and `mode` may be passed to override the classifier, but they are
    not required and usually should not be sent: the wording decides, and the
    classifier is deterministic so its decision is reproducible from the
    transcript. They exist for the follow-up case -- amendment F's "follow-ups
    keep scope until the driver changes it" -- where the CALLER knows what the
    last question was and the words alone no longer say ("what else?").
    """
    t0 = time.time()
    if not config.NEWS_ENABLED:
        return {"ok": False, "note": "news is switched off",
                "rules": RULES_BUDGET}

    cls = classify(question, has_place_context=bool(place_name))
    if scope in (SCOPE_LOCAL, SCOPE_PLACE, SCOPE_TOPIC, SCOPE_WORLD, SCOPE_MIXED):
        cls["scope"] = scope
    if mode in (MODE_NEWS, MODE_BACKGROUND):
        cls["mode"] = mode
        cls["timeframe"] = _timeframe_for(" " + (question or "").lower() + " ",
                                          cls["scope"], cls["category"], mode)
    if timeframe in TIMEFRAME_S:
        cls["timeframe"] = timeframe
    if categories:
        first = [c for c in (categories if isinstance(categories, list)
                             else [categories]) if c]
        if first:
            cls["category"] = str(first[0])

    # THE BUDGET, checked before anything is spent rather than after.
    spent = spend_of(session_key)
    if spent["searches"] >= int(config.NEWS_MAX_SEARCHES_PER_DRIVE):
        return {"ok": False, "note": "budget_exhausted", "classified": cls,
                "spent": spent, "rules": RULES_BUDGET}

    # Location, for every scope that is about somewhere.
    loc = {"ok": True}
    needs_geo = cls["scope"] in (SCOPE_LOCAL, SCOPE_PLACE, SCOPE_MIXED)
    if needs_geo:
        loc = location_context(latitude, longitude, session_key)
        if not loc.get("ok"):
            return {"ok": False, "note": loc.get("note") or "no_fix",
                    "need_location": True, "classified": cls,
                    "rules": ("You do not know where the car is, so you cannot "
                              "answer a question about round here. Ask which "
                              "area they mean — one short question — and do "
                              "NOT answer from memory.")}

    # AMENDMENT B. A place question with no confirmed entity is a question, not
    # a search. The confirmation itself is the caller's job (find_places /
    # the visual path) -- this refuses to proceed without it.
    if cls["scope"] == SCOPE_PLACE and not (place_name or "").strip():
        return {"ok": False, "note": "unconfirmed_entity", "classified": cls,
                "need_entity": True, "rules": RULES_NO_ENTITY}

    key = _cache_key(loc, cls, place_name)
    hit = _cached(key, cls)
    if hit:
        hit["cached"] = True
        hit["age_s"] = round(time.time() - hit["fetched_at"], 1)
        hit["classified"] = cls
        return hit

    if cls["mode"] == MODE_BACKGROUND:
        out = _background(loc, cls, place_name, session_key)
    else:
        out = _news(loc, cls, place_name, session_key, route_ahead)

    out["took_ms"] = round((time.time() - t0) * 1000, 1)
    if out.get("ok"):
        out["fetched_at"] = time.time()
        _remember(key, dict(out))
        out["cached"] = False
    return out


def _news(loc, cls, place_name, session_key, route_ahead=None) -> dict:
    queries = build_queries(loc, cls, place_name)
    if not queries:
        return {"ok": False, "note": "no_queries", "classified": cls,
                "rules": RULES_NEWS}

    got = retrieve(queries, cls, loc, place_name)
    _charge(session_key, got.get("searches", 0), got.get("est_cost_usd", 0.0))
    if not got.get("ok"):
        return {"ok": False, "note": got.get("note"), "classified": cls,
                "queries": queries, "searches": got.get("searches", 0),
                "rules": ("You could not look it up. Say so plainly, in your "
                          "own words, and do NOT answer from memory.")}

    checked = audit(got["results"], cls, loc)
    ranked = rank(checked["kept"], cls)[:int(config.NEWS_MAX_SPOKEN_RESULTS)]

    if route_ahead:
        for r in ranked:
            r["on_route"] = _on_route(r, route_ahead)

    return {
        "ok": True,
        "classified": cls,
        "location": {k: loc.get(k) for k in
                     ("neighborhood", "city", "county", "state")} if loc.get("city") else {},
        "queries": queries,
        "results": ranked,
        "n": len(ranked),
        "conflicts": [{"a": ranked[i]["headline"], "b": ranked[j]["headline"],
                       "disagreement": why}
                      for i, j, why in checked["conflicts"]
                      if i < len(ranked) and j < len(ranked)],
        "dropped": checked["dropped"],
        "dropped_n": len(checked["dropped"]),
        # What the dashboard must display. See LICENSING.md §4.
        "citations": citations_of_results(ranked),
        "offer_other": cls.get("offer_other"),
        "searches": got.get("searches"),
        "est_cost_usd": got.get("est_cost_usd"),
        "retrieve_ms": got.get("took_ms"),
        "rules": RULES_NEWS,
    }


def _background(loc, cls, place_name, session_key) -> dict:
    """Not news, so not the retrieval path. deep_dive's shape, measured at ~9 s.

    This is the one place the model's own knowledge is welcome, because the
    question is what a place IS rather than what just happened to it. The date
    gate does not apply -- there are no dates to gate -- so the honesty burden
    moves entirely into the rules and into not inventing, which is what the
    selftest checks instead.
    """
    subject = (place_name or "").strip()
    if not subject:
        subject = ", ".join([x for x in (loc.get("neighborhood"),
                                         loc.get("city"), loc.get("state")) if x])
    if not subject:
        return {"ok": False, "note": "no_subject", "classified": cls,
                "rules": RULES_NO_ENTITY}

    t0 = time.time()
    try:
        r = _client().responses.create(
            model=config.NEWS_MODEL,
            instructions=(
                "You are the background step behind a car assistant's spoken "
                "answer. Say what this place is and why it is interesting, in "
                "plain prose to be READ ALOUD: no markdown, no lists, no URLs, "
                "no citations.\n"
                "TWO OR THREE SENTENCES. The character of the place and what it "
                "is actually known for — not a founding date, not a population, "
                "not an encyclopedia opening line.\n"
                "Never invent a date, a number, a name or an event. If you are "
                "not certain of something, say so in the sentence or leave it "
                "out. Attribute anything contested rather than asserting it."),
            input=f"What is the story of {subject}? What is it known for?",
            tools=([{"type": "web_search"}]
                   if config.NEWS_BACKGROUND_SEARCH else []),
            max_output_tokens=int(config.NEWS_BACKGROUND_MAX_TOKENS),
        )
    except Exception as e:
        print(f"[news] background failed: {type(e).__name__}: {e}", flush=True)
        return {"ok": False, "note": f"{type(e).__name__}", "classified": cls,
                "rules": ("You could not look that up. Say so plainly and do "
                          "not invent a history for the place.")}

    searches = sum(1 for i in (getattr(r, "output", None) or [])
                   if getattr(i, "type", "") == "web_search_call")
    usage = getattr(r, "usage", None)
    cost = round(searches * config.NEWS_SEARCH_COST_USD
                 + (getattr(usage, "input_tokens", 0) or 0) * config.NEWS_IN_COST_USD
                 + (getattr(usage, "output_tokens", 0) or 0) * config.NEWS_OUT_COST_USD, 4)
    _charge(session_key, searches, cost)
    text = (getattr(r, "output_text", "") or "").strip()
    if not text:
        return {"ok": False, "note": "empty_background", "classified": cls,
                "rules": ("You could not look that up. Say so plainly and do "
                          "not invent a history for the place.")}
    return {
        "ok": True, "classified": cls, "mode": MODE_BACKGROUND,
        "subject": subject, "background": text,
        "citations": citations_of_response(r),
        "searches": searches, "est_cost_usd": cost,
        "offer_other": cls.get("offer_other"),
        "retrieve_ms": round((time.time() - t0) * 1000, 1),
        "rules": RULES_BACKGROUND,
    }


def citations_of_results(results: list) -> list:
    """The audited results, shaped for a citation UI and nothing else.

    OpenAI's web search terms require inline citations to be "clearly visible
    and clickable" wherever web results, or information drawn from them, are
    shown to a person. RIO's answer is spoken, so the dashboard is the only
    surface that can carry them — and this is the payload it renders.

    Deliberately a SEPARATE field from `results` rather than the panel reading
    the scoring internals. What a citation must contain is a licensing
    question, not a ranking one, and it should not silently change the next
    time a weight is renamed.
    """
    out = []
    for r in results:
        url = (r.get("url") or "").strip()
        if not url:
            # A citation with nothing to click is not a citation. It is left
            # out rather than rendered dead, and the result is still spoken --
            # the obligation attaches to what is DISPLAYED.
            continue
        out.append({
            "source": (r.get("source") or "").strip() or "unknown source",
            "headline": (r.get("headline") or "").strip(),
            "url": url,
            "published": r.get("published"),
            "age_h": r.get("age_h"),
            # What made it relevant to HERE. Null for topic and world, where
            # geography is not part of the question and a distance would be an
            # invented one.
            "where": r.get("city") or None,
            "geo_level": r.get("geo_level"),
            "source_type": r.get("source_type"),
        })
    return out


def citations_of_response(resp) -> list:
    """The url_citation annotations off a prose response.

    The background path answers in prose and has no result list, but when it
    searches, its answer IS "information contained in web results" and carries
    the same obligation. The annotations are where the API puts the sources, so
    this is the same citation payload arriving by a different door.
    """
    out, seen = [], set()
    for item in (getattr(resp, "output", None) or []):
        if getattr(item, "type", "") != "message":
            continue
        for c in (getattr(item, "content", None) or []):
            for a in (getattr(c, "annotations", None) or []):
                if getattr(a, "type", "") != "url_citation":
                    continue
                url = (getattr(a, "url", "") or "").strip()
                if not url or url in seen:
                    continue
                seen.add(url)
                title = (getattr(a, "title", "") or "").strip()
                out.append({
                    "source": _host_of(url), "headline": title, "url": url,
                    "published": None, "age_h": None, "where": None,
                    "geo_level": None, "source_type": "unknown",
                })
    return out


def _host_of(url: str) -> str:
    m = re.match(r"https?://(?:www\.)?([^/]+)", url or "")
    return m.group(1) if m else (url or "")[:40]


def _on_route(result: dict, route_ahead: list) -> bool:
    """Does this item sit on the road the car is about to drive?

    `route_ahead` is a list of {lat, lng} sampled from the live route by the
    caller — the panel, which is the only thing that knows where along the
    route the car is. Absent navigation it is None and every result is simply
    not marked, which is the honest default: "not on your route" and "there is
    no route" must not look the same.

    Matched on the road NAME appearing in the item, not on coordinates: a news
    item does not carry a position, and geocoding each one would be a billed
    request per result. Named roads are what closures are reported by.
    """
    names = {str(p.get("road") or "").lower().strip()
             for p in route_ahead if isinstance(p, dict)}
    names = {n for n in names if len(n) > 3}
    if not names:
        return False
    text = f"{result.get('headline','')} {result.get('summary','')}".lower()
    return any(n in text for n in names)


def status() -> dict:
    with _news_lock:
        cached = len(_news_cache)
    with _geo_lock:
        geo = len(_geo_cache)
    with _spend_lock:
        spend = {k: dict(v) for k, v in _spend.items()}
    return {
        "enabled": bool(config.NEWS_ENABLED),
        "model": config.NEWS_MODEL,
        "max_queries_per_question": config.NEWS_MAX_QUERIES_PER_QUESTION,
        "max_searches_per_drive": config.NEWS_MAX_SEARCHES_PER_DRIVE,
        "min_geo_match": config.NEWS_MIN_GEO_MATCH,
        "ttl_s": {"traffic": config.NEWS_TTL_TRAFFIC_S,
                  "emergency": config.NEWS_TTL_EMERGENCY_S,
                  "general": config.NEWS_TTL_GENERAL_S,
                  "events": config.NEWS_TTL_EVENTS_S,
                  "background": config.NEWS_TTL_BACKGROUND_S},
        "cached_answers": cached, "cached_areas": geo,
        "spend": spend,
    }
