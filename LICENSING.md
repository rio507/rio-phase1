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

## 3. Other services

| Service | Used for | To review |
|---|---|---|
| ElevenLabs | RIO's voice | commercial synthesis rights; whether provider-derived text may be sent to a third-party TTS |
| OpenAI | conversation, visual Q&A | data handling for camera-derived imagery; note that **no LLM is in the navigation path at all** — see `navigation/speech.py` |

---

## 4. Privacy posture, stated so it can be checked

Not a licence question, but adjacent and easy to lose track of:

- Camera frames are held in RAM for a few seconds (`framebuf.py`) and are
  written to disk only when `config.RING_PERSIST` is deliberately turned on.
- Navigation events log **route-relative** position — which maneuver, how far
  to it — not a trail of coordinates. The one exception is the off-route
  point, which cannot be understood without knowing where it happened.
- No API key is ever logged. The Maps JavaScript key is injected into the page
  at serve time because a browser map cannot work otherwise; routing,
  geocoding, autocomplete and place search all go through the server.
