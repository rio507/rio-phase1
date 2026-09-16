# LICENSING — what RIO uses, and what has to be answered before it ships

Two separate questions, and only the second one is open:

1. **Can the code run?** Every model weight in this repository was chosen for a
   permissive licence and the choice is enforced in code, not in a comment.
2. **May the data be used this way, in a car, out loud, for money?** An API
   returning HTTP 200 is not an answer to that. This file is where the
   unanswered parts are written down so they are settled deliberately, before
   a production release, rather than discovered afterwards.

---

## 1. Model weights — settled, and guarded in code

| Component | Weights | Licence | Guard |
|---|---|---|---|
| Object detection | RF-DETR nano/small (Roboflow) | Apache-2.0 | `headway/detect.py::_assert_apache()` reads upstream's own `license` field at load time |
| Depth | Depth Anything V2 **Metric-Small** | Apache-2.0 | `headway/depth.py::_assert_apache_small()` refuses Base/Large, which are CC-BY-NC-4.0 |
| Lanes | UFLDv2 (CULane) | research-permissive; review before release | — |
| Local VLM | Qwen3-VL-8B | Tongyi Qianwen licence; review before release | — |

**YOLO is deliberately absent.** The Ultralytics line is AGPL-3.0, which is
commercially unusable for this product. That is why detection is RF-DETR.

Depth Base/Large would measurably improve range estimation and cannot be used.
The guard exists so that a future contributor who reaches for the better
checkpoint gets an exception rather than a licensing incident.

---

## 2. Navigation provider data — OPEN, and required before production

RIO's routing, geocoding, place autocomplete and landmark lookups currently
come from Google APIs, reached only through `navigation/providers/google.py`;
conversational place search is the same vendor through a second, deliberately
separate door (`places.py`), because it is a different use with different terms
— see the sub-section below.
Everything downstream of that file speaks RIO's canonical model and knows
nothing about Google (`navigation/model.py`, `navigation/provider.py`).

**The question is not "does the API return the data".** It does. The question
is whether the terms permit *this* use: turn-by-turn guidance, in a moving
vehicle, spoken by a synthesised voice, in a commercial product, with the
route and place names held in memory for the duration of a drive.

### Review scope — settle every line before a production release

- **Turn-by-turn usage rights.** Whether the Routes API's terms permit
  real-time navigation guidance, or whether that use requires the Navigation
  SDK instead.
- **Navigation SDK requirements.** If turn-by-turn requires it: platform,
  attribution, UI and telemetry obligations that come with it.
- **Synthesized speech restrictions.** Whether route instructions and
  provider-derived place names may be spoken by a TTS voice, and whether
  attribution must be audible or may be visual.
- **In-vehicle usage.** Terms specific to automotive/head-unit contexts, which
  are frequently distinct from mobile and web terms.
- **Attribution.** What must be displayed, where, and whether the dashboard's
  current map attribution is sufficient for a route that is *heard* rather
  than seen.
- **Caching and retention.** How long a route, its geometry and its maneuvers
  may be held. RIO currently holds up to 8 route generations in memory for the
  life of the process and writes a route summary to the session log.
- **Route content restrictions.** Whether route data may be combined with
  other data sources, re-derived, or logged for later review — which is
  exactly what `sessions.log_nav` does.
- **Places usage restrictions.** Nearby Search and Place Details results are
  used to pick contextual landmarks. Terms on pre-fetching, caching, and using
  place data outside a Google map need reading closely.
- **Places-derived business names in spoken guidance.** RIO says "Turn left by
  the Shell station", where "Shell" originated in a Places response. Whether a
  business name obtained from Places may be spoken as navigational context —
  and whether a brand's own trademark position matters here independently of
  the API terms — is the single most product-specific question in this file.
- **Autocomplete sessions.** Destination typing is grouped into billed
  autocomplete sessions with a session token, minted server-side and consumed
  by the Place Details lookup that resolves the selection. Confirm the token
  lifetime and the requirement that a session end in a details call are being
  honoured as the terms describe them, and that RIO's opaque session id (which
  is what the browser sees) raises nothing of its own.
- **Places data caching and retention.** Landmark candidates are fetched once
  per route generation and cached for that generation's lifetime (minutes),
  never written to disk. Confirm that is inside the permitted caching window,
  and confirm what may appear in the session log — currently the label, the
  relation and the confidences, not the place id.

#### Places SEARCH, spoken as an answer (`places.py`) — a distinct use

The landmark questions above are about place data used as *navigational
context*. `find_places` is a different use of the same API and needs its own
answers: the driver asks "what's good round here", a Text Search runs, and RIO
**reads business names, ratings, review counts, price levels and opening status
out loud**. That is Places content presented as the substance of an answer, not
as a landmark beside a turn.

- **Display requirements away from a map.** Google's terms attach display
  obligations to Places content — attribution ("Powered by Google"), and rules
  about showing ratings and reviews. RIO's answer is *audible* and the dashboard
  may not be showing a map at that moment. Settle what must be displayed, where,
  and whether an audible answer changes it. The tool result carries an
  `attribution` field so that whatever the answer is can be implemented in one
  place.
- **Ratings and review counts read aloud.** Whether a rating and its review
  count may be spoken, and whether they must be attributed to Google when they
  are. RIO currently says "four point four" without naming the source.
- **Third-party content in a synthesised voice.** Same question as the spoken
  route instructions above, for a different content class: business names and
  editorial-adjacent facts rather than road names.
- **Prohibition on re-use.** Confirm what may be retained from a search.
  `places.py` keeps the last result list in memory for
  `PLACES_CACHE_TTL_S` (3 minutes) so "take me to the second one" resolves, and
  `sessions.log_live` records the query and the returned NAMES for review.
  Confirm both, and in particular whether logging names is caching.
- **place_id retention and re-use.** A place_id from a search is passed into
  `start_navigation` to skip re-resolution. Google's terms treat place ids as
  cacheable indefinitely where other fields are not; confirm that, and confirm
  that using one as a routing destination is within scope.
- **Field mask as a licensing surface, not only a billing one.** `FIELD_MASK`
  in `places.py` is currently id, name, address, location, rating, review count,
  price level and open-now. Any field added there is a new content class with
  its own display terms — photos and reviews especially — so the mask should be
  read as part of this review and not treated as a performance setting.

### What is already true, and makes substitution cheap

Provider substitution does not require rebuilding any of:

- the route tracker (`static/rio_navcore.js`) — geometry and GPS only;
- contextual navigation logic (`navigation/landmarks.py`, `anchors.py`) —
  operates on canonical coordinates;
- visual verification (`navigation/verify.py`) — never sees provider data
  beyond a label;
- the speech system (`navigation/speech.py`) — templates over the canonical
  model;
- arbiter integration (`static/rio_speech.js`) — knows nothing of navigation.

A different provider means one more implementation of `NavigationProvider`.
`navigation/fixtures.py::FixtureProvider` is a working second implementation,
exercised by the test suite on every run, which is what keeps that claim
honest rather than aspirational.

If a provider's terms turn out to forbid this use, the options are: a
different routing provider, an embedded/offline engine, or shipping without
contextual landmarks (the architecture supports zero anchors as a first-class
mode — see §27 of the design).

---

## 3. Weather provider data — VERIFIED where it could be, OPEN where it could not

A third door to the same vendor, deliberately separate from routing
(`navigation/providers/google.py`) and from conversational place search
(`places.py`) for the same reason those two are separate from each other: it is
a different product, with different terms, doing a different job. All of it is
reached through `weather.py` and nothing downstream of that file knows Google
exists.

### What was actually checked, rather than assumed

Amendment A asked for verification before belief. This was run against the
live endpoint on the project's existing `GOOGLE_MAPS_API_KEY`
(`python -m tools.weather_probe`, kept in the tree so it can be re-run when
any of this is doubted):

| Question | Answer, measured |
|---|---|
| Enabled on the existing key? | **Yes** — `currentConditions:lookup` returns HTTP 200 with no key or project change |
| Endpoints used | `currentConditions:lookup`, `forecast/hours:lookup`, `forecast/days:lookup` |
| Forecast horizon | hourly **240 h max** (24 per page, `nextPageToken` beyond); daily **10 days max** |
| Units | a **request** parameter (`unitsSystem`), not a property of the response. METRIC is the default; RIO asks for IMPERIAL and carries the unit NAMES out with every number |
| Field mask | `fields` is supported and works, but unlike Places (New) it does **not** select a SKU — it is a payload reduction only |
| Latency | ~250–500 ms per endpoint; ~790 ms for a full three-call refresh |
| Cost | **one SKU, "Weather Usage" (9DB8-727A-ACFE)**, billed per request regardless of endpoint: first **10,000 requests/month free**, then **$0.15 / 1,000** (0.15¢ each) to 100k, tiering down to $0.038 / 1,000 above 5M |
| Rate limits | not published on the usage-and-billing page; **OPEN** (see below) |
| **Severe weather alerts** | **THERE IS NO ALERTS ENDPOINT.** Four candidate paths all 404 |
| **Geographic coverage** | **not global.** Japan, South Korea and mainland China return HTTP 404 *"Information is not supported for this location"* on every endpoint; Australia, Germany, Brazil, Nigeria, India and the US answer normally |

### The alerts finding, because it changes what the code may claim

The spec asks for "severe weather when available" and the context carries an
`alerts` field. Google's Weather API has no alerts endpoint, so **that list is
always empty**. It is kept only so the shape does not change if a source is
ever added, and `weather.py` documents — and `tools/weather_selftest.py`
asserts — that an empty `alerts` means **NOT KNOWN** and never "no severe
weather". Those are different claims and only one of them is true. If severe
weather warnings are ever a product requirement, they need a different
provider (NWS/CAP in the US, MeteoAlarm in the EU) and their own row in this
file, because government alert feeds carry redistribution terms of their own.

### The coverage finding, which is a product question before it is a legal one

Google's Weather API does not cover everywhere. Probing found Japan, South
Korea and mainland China refusing every endpoint with a 404 while comparable
markets answered normally. `weather.py` raises `NoCoverage` for that case and
returns `note: "no_coverage"` — the same honest refusal a network failure gets,
because the driver's experience should be identical, but named differently so
that a drive through an uncovered region does not fill the log with what look
like network faults.

RIO therefore degrades correctly rather than silently in those markets: she
says she cannot pull the forecast and forecasts nothing from the sky. **What
she does not do is work there.** If any of those markets is a launch market,
this is a second-provider decision and not a licensing one, and it should be
made before the feature is promised rather than after.

### Attribution — answered, and it lands in the same unsolved place as Places

Google's Weather API policies require, verbatim:

> "Source: Includes weather data from Google"

displayed **on or next to the data used**, "clearly visible", never removed,
hidden, obscured or modified. The string is carried out of `weather.py` on
every successful context as `attribution`, so whatever is decided can be
implemented in one place.

**The unsolved part is the same one `places.py` has and it is worse here.**
RIO's weather answer is *audible*, in a car, and the dashboard may not be in
front of the driver — or may not be showing anything at all. A requirement to
display something "clearly visible" has no obvious meaning for a sentence
spoken by a synthesised voice to someone watching the road. This must be
settled before a production release:

- Whether the dashboard's existing attribution satisfies it for data that is
  *heard*, or whether audible attribution is required.
- Whether a spoken weather figure ("about seventy percent") is "data used"
  for the purposes of that clause.
- Whether the answer differs when the panel is backgrounded, the phone is
  locked, or the car is being driven with the screen off.

Note what is NOT a problem here, having been checked: the policies place **no
restriction on displaying weather data away from a Google map**, which was the
open question that made the Places case hard. Weather data does not have to
sit on a map.

### Still open, and to settle before production

- **Caching and retention.** The Weather API is **not named** in the Maps
  Platform Service Specific Terms' caching clauses, which enumerate Geocoding,
  Directions, Solar and others at 30 consecutive calendar days, and exempt
  place IDs entirely. Weather's absence from that list is not permission.
  RIO holds one context per session in memory for minutes
  (`WEATHER_REFRESH_S` 600 s, hard limit `WEATHER_MAX_AGE_S` 900 s) and writes
  nothing to disk, which is inside any plausible reading — but "inside any
  plausible reading" is what this file exists to replace with an answer.
- **Synthesized speech.** Same question as the spoken route instructions and
  the spoken business names, for a third content class: may Weather API
  content be read aloud by a TTS voice in a commercial product. Unanswered for
  all three; answering it once probably answers it three times.
- **In-vehicle use.** Automotive and head-unit terms are frequently distinct
  from web and mobile terms. Confirm the Weather API's position specifically —
  do not infer it from the Routes review.
- **Rate limits / QPM.** Not documented on the usage-and-billing page. Get the
  real number before phase 2, whose route sampling is precisely the thing that
  would find it.
- **Derived claims.** `weather.py` computes `next_precipitation` — the first
  forecast hour crossing a probability threshold — from Google's hourly data,
  and RIO speaks it ("rain around three"). That is a *derivation* from provider
  content rather than a field of it. It is the same class of question as
  `places.py`'s drive-time estimate and should be read alongside it.
- **Logging.** Decide what may appear in a session log. Currently a weather
  context is not written to the session log at all; if that changes, the
  question is whether logging a temperature and a probability is caching.

### What is already true, and makes substitution cheap

The same property the navigation stack has. `weather.py` is the only file that
knows Google's response shape; everything else — the tool bridge in
`realtime.py`, the proactive gate in `weather_policy.py`, the honesty limits,
the selftests — operates on the normalized context. A different weather
provider is one more implementation of `get_weather_context`, and the
`alerts` question above is the most likely reason to need one.

---

## 4. Other services

| Service | Used for | To review |
|---|---|---|
| ElevenLabs | RIO's voice | commercial synthesis rights; whether provider-derived text may be sent to a third-party TTS |
| OpenAI | conversation, visual Q&A | data handling for camera-derived imagery; note that **no LLM is in the navigation path at all** — see `navigation/speech.py` |

---

## 5. Privacy posture, stated so it can be checked

Not a licence question, but adjacent and easy to lose track of:

- Camera frames are held in RAM for a few seconds (`framebuf.py`) and are
  written to disk only when `config.RING_PERSIST` is deliberately turned on.
- Navigation events log **route-relative** position — which maneuver, how far
  to it — not a trail of coordinates. The one exception is the off-route
  point, which cannot be understood without knowing where it happened.
- No API key is ever logged. The Maps JavaScript key is injected into the page
  at serve time because a browser map cannot work otherwise; routing,
  geocoding, autocomplete and place search all go through the server.
