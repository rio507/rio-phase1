"""Every shape of news question, answered out of live data, end to end.

    python -m tools.news_shapes
    python -m tools.news_shapes --only local,world
    python -m tools.news_shapes --lat 34.0195 --lng -118.4912

The selftest proves the gates HOLD. This shows what gets said when they do.
It runs the real classifier, the real retrieval, the real audit and ranking,
then hands the surviving results and the real rule text to the conversational
model and prints what RIO would actually say for each shape the spec names:

    local       "Any news around here?"
    traffic     "Why is traffic so bad?"
    place       "Any news about this place?"      (with a confirmed entity)
    unconfirmed "Any news about this place?"      (without one -- must ASK)
    topic       "What's happening with the Lakers?"
    world       "What's in the news today?"
    background  "What's the story of this place?"

IT SPENDS REAL MONEY: roughly 4 searches a shape at $0.010 each plus tokens,
so a full run is on the order of half a dollar. That is why `--only` exists and
why this is not wired into boot.sh. The per-drive budget in config applies here
exactly as it does in the car, which is itself worth watching: a full run is a
realistic "driver who will not stop asking" and it is the case the cap is for.

Why a tool rather than a paragraph in a commit: these seven sentences are the
feature. Everything else exists to make them supportable, and the only way to
see whether RIO sounds like a passenger rather than a news reader is to read
them.
"""
import argparse
import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from dotenv import load_dotenv                              # noqa: E402

load_dotenv(Path(__file__).resolve().parent.parent / ".env")

import config                                               # noqa: E402
import localnews as ln                                      # noqa: E402

VOICE = (
    "You are RIO, riding shotgun — she/her. Sharp, easygoing. Contractions "
    "always, fragments fine. Straight into it: no preamble, no 'great "
    "question'. Never call yourself an AI. Never address the driver by name."
)


def say(rules: str, payload: str) -> str:
    from openai import OpenAI

    r = OpenAI(timeout=60).responses.create(
        model=config.OPENAI_CHAT_MODEL,
        instructions=VOICE + "\n\n" + rules,
        input=payload, max_output_tokens=500,
        reasoning={"effort": "none"})
    return (getattr(r, "output_text", "") or "").strip()


def block(r: dict) -> str:
    """What the voice model is handed: the audited results and nothing else."""
    if r.get("background"):
        return (f"SUBJECT: {r.get('subject')}\n"
                f"BACKGROUND:\n{r['background']}\n"
                f"offer_other: {r.get('offer_other')}")
    slim = [{k: x.get(k) for k in
             ("headline", "source", "source_type", "published", "age_h",
              "summary", "category", "city", "geo_level", "score")}
            for x in (r.get("results") or [])]
    return (f"RESULTS ({len(slim)}):\n{json.dumps(slim, indent=1)}\n"
            f"conflicts: {json.dumps(r.get('conflicts') or [])}\n"
            f"offer_other: {r.get('offer_other')}")


SHAPES = {
    "local":       ("Any news around here?", ""),
    "traffic":     ("Why is traffic so bad here?", ""),
    "place":       ("Any news about this place?", "Santa Monica Pier"),
    "unconfirmed": ("Any news about this place?", None),   # None = pass nothing
    "topic":       ("What's happening with the Lakers?", ""),
    "world":       ("What's in the news today?", ""),
    "background":  ("What's the story of this place?", "Santa Monica Pier"),
}


def run_shape(name, lat, lng, key):
    question, place = SHAPES[name]
    cls = ln.classify(question, has_place_context=bool(place))
    print(f"\n{'=' * 72}\n{name.upper()}  —  {question!r}"
          + (f"   [entity: {place}]" if place else ""))
    print(f"  classified: scope={cls['scope']} mode={cls['mode']} "
          f"timeframe={cls['timeframe']} category={cls['category']} "
          f"topic={cls['topic']!r}")

    t0 = time.time()
    r = ln.search_local_news(lat, lng, question=question,
                             place_name=(place or ""), session_key=key)
    took = time.time() - t0

    if not r.get("ok"):
        print(f"  -> refused: {r.get('note')}  ({took:.1f}s)")
        # A refusal is a shape too: the unconfirmed-entity case must ASK.
        out = say(r.get("rules") or "", f'DRIVER: "{question}"\n'
                  f"(You could not proceed: {r.get('note')})")
        print(f"  RIO: {out}")
        return r

    if r.get("cached"):
        print(f"  -> cached, {took:.1f}s, $0.00")
    else:
        print(f"  -> {took:.1f}s, {r.get('searches')} searches, "
              f"${r.get('est_cost_usd')}")
    if r.get("queries"):
        for q in r["queries"]:
            print(f"     query: {q}")
    if r.get("results") is not None:
        print(f"     kept {r.get('n')}, dropped {r.get('dropped_n')}")
        for d in (r.get("dropped") or [])[:4]:
            print(f"       DROPPED [{d['why']}] {str(d.get('headline'))[:52]}")
        for x in r["results"][:4]:
            print(f"       {x['age_h']:>6}h {x['geo_level']:<13} "
                  f"{x['source'][:22]:<22} {x['headline'][:50]}")
    if r.get("conflicts"):
        print(f"     CONFLICTS: {r['conflicts']}")

    out = say(r["rules"], f'DRIVER: "{question}"\n\n' + block(r))
    print(f"  RIO: {out}")
    return r


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--lat", type=float, default=34.0195)
    ap.add_argument("--lng", type=float, default=-118.4912)
    ap.add_argument("--only", default="",
                    help="comma-separated shape names; default is all")
    a = ap.parse_args()

    key = "_shapes"
    ln.forget(key)
    loc = ln.location_context(a.lat, a.lng, key)
    print(f"Car at {a.lat}, {a.lng}")
    if loc.get("ok"):
        print(f"  -> {ln.area_words(loc)}")
    else:
        print(f"  -> reverse geocode failed: {loc.get('note')}")
        return 1

    want = [s.strip() for s in a.only.split(",") if s.strip()] or list(SHAPES)
    for name in want:
        if name not in SHAPES:
            print(f"\n(unknown shape {name!r}; known: {', '.join(SHAPES)})")
            continue
        try:
            run_shape(name, a.lat, a.lng, key)
        except Exception as e:
            print(f"  !! {type(e).__name__}: {e}")

    print(f"\n{'=' * 72}")
    print(f"  drive spend: {ln.spend_of(key)}")
    print(f"  cap is {config.NEWS_MAX_SEARCHES_PER_DRIVE} searches per drive.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
