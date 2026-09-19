# Local intelligence — news, incidents, and what a place is

RIO already knows where the car is, what is out of the window, and what the sky
is doing. This is the layer that knows what is *happening*.

It is built as local intelligence rather than as a news feature, which in
practice means one thing: the location context object (`location_context()`) is
produced once from GPS and knows nothing about news. News is its first
consumer. Events, alerts, road conditions and points of interest would be the
next ones and would not change it.

```
  GPS ─► reverse geocode ─► location context ─┬─► news / incidents  (here)
        (Geocoding API)     neighborhood      ├─► weather           (weather.py)
                            city              ├─► places            (places.py)
                            county            └─► events, alerts, … (later)
                            state
```

---

## 1. Why news is not weather, and what follows from it

Weather is numbers from one authority. Ask twice, get the same answer; the only
way it can mislead is by being old, so `weather.py` needs one clock and one
distance and it is honest.

News is claims from strangers. It can be wrong at the source, right but stale,
right but about a town with the same name, or two outlets flatly contradicting
each other — and **all four arrive looking exactly like a fact.** So nothing is
spoken unless the fields underneath it say it may be:

| field | what it defends against |
|---|---|
| `published` | a six-hour-old incident described as happening now |
| `source` + `source_type` | an anonymous post sounding like a police notice |
| `geo_level` | the right name in the wrong state |
| `conflicts` | two outlets disagreeing, silently resolved in RIO's favour |

`audit()` enforces these and runs **before** ranking, deliberately: a result
that may not be spoken should never be in a position to be the top one.
Filtering after ranking is how the best-scoring unusable item becomes the
answer.

---

## 2. The gates, in the order they bite

1. **Undated is refused.** Not hedged, not "reportedly" — dropped. The
   retrieval step is instructed to return `null` rather than guess a date, and
   it does: a first probe returned 7 of 9 results undated, which is exactly the
   behaviour wanted, and all 7 were dropped. Adding "only items you can date,
   and no status pages" to the instruction took the same query to 5 of 5 dated.
2. **Older than the intent's window is refused.** 6 h for an incident, 24 h for
   general local, 7 d for a topic, 30 d for a named place, unrestricted for
   background. "What happened here yesterday" uses yesterday, not 24 h.
3. **Geographic corroboration** (local/place/mixed only). Below county level
   the result is discarded. This is amendment B, and §4 is why it is not a
   substring test.
4. **A lone social post is not news.** If everything that survived is social,
   nothing survived. Alongside an official source it is kept as supplement.
5. **Conflicts are detected and both sides kept**, so RIO cannot silently
   prefer one. Her instruction is to report the disagreement or say nothing.

---

## 3. Scope and mode, decided from the words

Deterministic rules, not a model call — a classifier round trip to decide how
to spend eight cents is defensible, but "why did it search the whole world when
I asked about this street" needs an answer you can read.

| the driver says | scope | mode |
|---|---|---|
| "any news around here", "why is traffic bad", "what happened up ahead" | local | news |
| "any news about this place", "any stories about this place" | place | news |
| "what's happening with the Lakers", "any news on that wildfire" | topic | news |
| "what's in the news today", "anything big happening" | world | news |
| "anything happening with the fires near us" | mixed | news |
| "what's the story **of** this place", "what's this neighbourhood known for" | place/local | background |
| "what's the deal with this place" (ambiguous) | place | background, offers news |

**"Stories about" is news; "the story of" is history.** The preposition really
is the whole distinction, and it is pinned in the selftest.

An ambiguous "any news?" defaults to **local** — the differentiated case, and
what a passenger would assume — and sets `offer_other: world` so RIO offers the
wider view in the same breath.

### Two classifier bugs worth remembering

Both were caught by sweeping the spec's own question list, and both were silent:

- *"What's going on where I am?"* classified as **mixed** with topic
  `"where i am"`. The topic extractor matched the `on` of the phrasal verb
  "going on". Fixed by looking back a word — and then by matching the
  preposition *alone*, because matching `word + preposition` consumed the space
  before `with` and made *"going on **with** tariffs"* lose its real topic.
- *"Why is that place famous?"* missed the background pattern, which allowed
  "why is that famous" but not a noun in between.

---

## 4. Entity confusion: the right name in the wrong state

The dangerous result is never the irrelevant one. It is the one that matches
perfectly and is 800 miles away, because it survives every other check and
reads as local.

`geo_match()` compares **address components**, never substrings. The first
version used `v in city or city in v`, and a crafted test put
`"Santa Monica, New Mexico"` through it as a **city-level match** for Santa
Monica, California — because the string really does contain it. It now splits
on commas, canonicalises US state abbreviations (`CA` ≡ `California`), and
refuses outright when a result names a region that is not ours.

A **place** question with no confirmed entity does not search at all. It comes
back `unconfirmed_entity` and RIO asks — live, that is literally *"Which place
do you mean?"*. Confirmation is `find_places`' job; this refuses to proceed
without it.

---

## 5. What it costs — searches and seconds observed, dollars withdrawn

**The dollar column below was wrong and is struck rather than restated.** It was
`est_cost_usd` from `localnews.retrieve()`, which multiplies real token counts by
the rates in `config.py` — and those rates priced tokens at gpt-5's standard tier
($1.25 / $10.00) while `NEWS_MODEL` is `gpt-5.6-sol` ($5.00 / $30.00). Input was
understated **4×**, output **3×**. The table also said *measured*, citing
`tools/news_probe.py`, which **does not exist in any commit**: the searches and
the seconds are counted from real responses, the money never was.

| shape | searches | time | ~~cost~~ |
|---|---:|---:|---:|
| local news | 4–5 | 34–46 s | ~~$0.11–0.12~~ |
| place (confirmed entity) | 3 | 30 s | ~~$0.08~~ |
| traffic (narrow, nothing found) | 1 | 13 s | ~~$0.03~~ |
| topic (7 d) | 6 | 57 s | ~~$0.15~~ |
| world | 3 | 35 s | ~~$0.09~~ |
| **background** | **0–1** | **4–9 s** | ~~$0.008–0.014~~ |
| any of them, cached | 0 | ~0 s | $0.00 |

At **$10.00 per 1,000 `web_search` calls** plus ~8–10k input tokens per search.

### What one real question cost, at the corrected rates

`tools/news_selftest.py` run live (not `--offline`), Santa Monica, 2026-09-19:

| shape | searches | time | cost |
|---|---:|---:|---:|
| local news | 4 | 60.0 s | **$0.2326** |

Roughly **2×** the $0.11 that used to be quoted for this shape — which is what a
4× input and 3× output correction predicts. Read it as **one point, not a
range**: one question, one location, one run. The struck table above is left
struck rather than rescaled around this, because one measurement does not
reconstruct six.

It is still an *estimate* in one specific sense worth keeping straight: the
**token counts are real**, off the response's own `usage`, and the **prices are
list**. That is what `est_cost_usd` is for and it is the same method as before —
the only thing that changed is that the prices are now the ones `gpt-5.6-sol` is
billed at. It is not an invoice.

**The uncomfortable half of that run:** it took the full `NEWS_TIMEOUT_S`, and
after `audit()` it kept **nothing** — 4 searches, 1 result dropped, 0 spoken. So
23 cents bought a question RIO could not answer. The caps bound what a question
may *spend*; nothing bounds what a question may spend **for no answer**. That is
a real gap, not a rounding error, and it is worth a cap of its own.

**Nothing overspent because of this.** `NEWS_MAX_SEARCHES_PER_DRIVE` has its
teeth on the search count, not on the money, so the budget refused exactly when
it always did. What the wrong rates reached is every surface that *quotes* a
figure: `/news_spend`, `tools/news_selftest.py`, `tools/news_shapes.py`, and
`LICENSING.md` §4, which published the range to counsel.

One correction that survives the fix and matters for the caps: **the token half
is the larger half of the bill**, not the smaller one. That is an argument for
`NEWS_MAX_QUERIES_PER_QUESTION` and not only for the per-drive search cap.

That is why the caps are not decoration:

- `NEWS_MAX_QUERIES_PER_QUESTION = 4` — in the instruction.
- `NEWS_MAX_SEARCHES_PER_DRIVE = 24` — **enforced in Python** against the
  searches actually counted in each response, refusing rather than overspending.
  The instruction is a request; the budget is not. Observed: a 4-query question
  ran 5 searches, which is exactly why spend is debited from the count and not
  from the cap.
- Caching: 5 min traffic, 25 min general, 3 h events, **a week for background** —
  what a neighbourhood *is* does not change, and it is the most repeatable
  question there is.

**Latency is 13–57 s**, far past amendment C's 4 s line, so a holding line is
mandatory and is written into both the tool description and the session
instructions. Reasoning effort is *not* the lever — `low` and `medium` measured
24.8 s and 22.2 s. The time is the searches.

---

## 6. Does it route through deep_dive? Partly, and here is the measurement

**It is the same vendor, key and API that `deep_dive` already uses.** There is
no second search stack and no new licensing surface beyond §4 of LICENSING.md.

But local/place/topic/world news is a **different call**, for two measured
reasons:

1. **`escalate()` returns prose.** Amendment A requires every result to carry a
   source and a timestamp, and requires RIO's sentence to be supportable by
   them. You cannot enforce "nothing undated is ever spoken" against a
   paragraph. This asks for a strict JSON schema, so a date is a field that can
   be *refused* rather than a phrase that can be believed.
2. **`escalate()`'s budget provably fails this shape.** Given the local-news
   question with its 320 + 1200 token ceiling, it spent all 1520 tokens
   reasoning across six searches and returned **nothing at all** after 30.6 s.
   That budget is sized for a question with one or two searches behind it,
   which is what `deep_dive` is for.

**Background does route to `deep_dive`'s shape** — prose, the model's own
knowledge welcome, no date gate because there are no dates to gate. Measured at
9 s and $0.014 against 34 s and $0.12 for the retrieval path. It is a different
question and it gets the cheaper, better-suited call.

---

## 7. Files

| file | what |
|---|---|
| `localnews.py` | classifier, location context, queries, retrieval, audit, ranking, caches, budget |
| `config.py` | the `NEWS_*` block: caps, weights, TTLs, measured prices |
| `realtime.py` | `search_local_news` tool + schema + session instructions |
| `tools/news_selftest.py` | amendments A, B, F and C, offline; plus a live pass |
| `tools/news_shapes.py` | every question shape, live, end to end |
| `static/index.html` | the Sources card: `RIO.ui.sources()`, markup and CSS |
| `static/rio_realtime.js` | emits `LIVE_SOURCES` when a tool result carries citations |
| `tools/sources_card_selftest.py` | the card, in a real browser |
| `LICENSING.md` §4 | **what the card solves, and the half it does not** |

## 8. Citations: the Sources card

OpenAI's web search terms require inline citations be "clearly visible and
clickable" wherever web results, or information drawn from them, are shown to a
person. RIO's answer is spoken, so the dashboard is the only surface that can
carry them.

Every result that survives `audit()` keeps its `url`, `source`, `headline` and
`published`; `citations_of_results()` shapes those into a payload built for the
UI rather than for ranking, and the background path gets the same payload from
the response's `url_citation` annotations. `rio_realtime.js` emits
`LIVE_SOURCES` before asking for the spoken answer — the card has to be there
before she starts talking, not after she stops — and `RIO.ui.sources()` renders
a block per answer, headed by the question, kept in scrollback.

Two rules that are enforced rather than assumed, because the obligation
attaches to what is *displayed*:

- **A citation with nothing to click is not rendered.** Dropped server-side and
  again in the renderer. A browser test caught the renderer producing a dead
  row — headline and source, no link — when only the server filter existed,
  which is exactly the "not clickable" case the requirement names.
- **The card renders even for a superseded turn.** If a search ran and informed
  something shown to the driver, the obligation attached; it does not depend on
  which turn won.

**What it does not solve**, and this belongs here as much as in LICENSING: a
driver listening with the phone face-down, pocketed, or the screen off has no
visible UI at all. The citations exist, they are correct, and nobody can see
them. That is the open question for counsel in LICENSING.md §4, and it is the
same question Places (§2) and Weather (§3) ask about *their* display
obligations — worth answering once, as one question about spoken products.

---

## 9. Not built, on purpose

**No proactive news** (amendment E). V1 answers when asked. When proactive
does come it enters through the arbiter as an advisory finding under the
existing suppression rules — the same path `weather_policy.py` uses, at
`P.CONVO`, never a new speech path.

**Route relevance is present but thin.** `_on_route()` matches road names from
a caller-supplied route sample, because a news item carries no coordinates and
geocoding each one would be a billed request per result. Absent navigation
nothing is marked, which is the honest default: "not on your route" and "there
is no route" must not look the same.
