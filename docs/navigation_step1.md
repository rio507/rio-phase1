# Navigation — step 1: WebNavProvider + speech arbiter + turn-by-turn

**Status:** shipped, desk-verified against real Google routes. Thresholds are
provisional prototype values, in the same sense as the headway ladder: starting
points for tuning against real drives, not validated numbers.

Plain turn-by-turn in the browser dashboard. No route reasoning, no LLM in the
timing path, no second voice.

---

## The split

```
  destination ──► /nav/route ──► Google Routes API v2        nav.py
                                 route + maneuvers + the
                                 announcement for every
                                 maneuver, precomputed
                                        │
                                        ▼
  GPS fix ──────► rio_navcore.js ──► which maneuver, how far, which tier
   (~1 Hz)        no DOM, no network      │
                                          ▼
                  rio_speech.js ──► one mouth, priority-arbitrated
                                          │
                                          ▼
                  /nav/voice ──► ElevenLabs, RIO's voice
```

**Google owns routing. RIO owns progression.** We never invent, shorten or
"correct" a route. Leaving it produces a `reroute` event, which asks Google for
a new one from where the car actually is — not a patch onto the old geometry.

Progression is client-side on purpose. A turn announcement is worth nothing
late, so nothing between "4 seconds from the turn" and RIO saying so may depend
on the network. It is also the one part of navigation that has to keep working
when signal drops: the route is already in hand.

Swapping in an embedded or offline provider later means another `compute_route`
returning the same shape. Nothing downstream knows the word "Google".

---

## Approach tiers are TIME, not distance

| tier | fires at | typical spoken line |
|------|----------|---------------------|
| far  | 30 s to the maneuver | "In 400 meters, turn left onto The Embarcadero" |
| mid  | 12 s | "In 150 meters, turn left onto The Embarcadero" |
| near | 4 s  | "Turn left onto The Embarcadero" |

300 m of open highway and 300 m of downtown are the same distance and completely
different warnings. Seconds are what a driver needs to act.

Rules the engine enforces:

- **Only the finest applicable tier fires.** A route set 60 m before a turn says
  "turn right onto Lincoln" — it does not start a countdown that is already false.
- **Tiers never run backwards** for a maneuver.
- **Speed floor of 3 m/s** when computing time-to-maneuver. Without it, creeping
  in traffic makes every maneuver hours away and nothing is ever announced —
  exactly where announcements matter.
- The **spoken distance is the live one** at the moment the tier fires (the
  server formats it from `dist_m`), so a slow approach says "in 90 meters" and a
  fast one says "in 500 meters" from the same tier.

Off-route needs `3` consecutive fixes beyond `45 m` before it rewrites anything;
one bad fix in a tunnel is not a wrong turn.

## The speech floor

Every maneuver's announcement text is computed **once, at route time**, for all
three tiers, in `nav.py`. Nothing generates language while the car is moving —
the timing path only ever looks a string up. `/nav/voice` re-formats from the
same stored templates, so what is spoken and what is logged come from one
function and cannot drift.

`/nav/voice` is addressed by `(route_id, maneuver, tier)` and refuses anything
that does not resolve to a maneuver on a live route. The browser never sends
RIO a sentence to say — the same discipline `/headway_voice` already had.

## The arbiter: one mouth

| priority | channel |
|---|---|
| P1 | safety — headway warnings. Pre-empts anything, mid-word. |
| P2 | the near-tier turn. The one nav line that is worthless late. |
| P3 | every other nav announcement. |

Two rules beyond the ladder:

- **Supersede by group.** A fresher item replaces an older one in the same group
  — playing or queued. Nav groups by maneuver, so `near` cuts off its own `far`.
  Headway groups as a channel, which preserves the behaviour the red tier had.
- **Expire, never catch up.** A queued announcement that outlived its window is
  dropped. Turn-by-turn played late is worse than silence.

Pre-empted lines are never resumed. The headway path was folded through this in
the same change: there is no longer any code that can play audio without asking.

## Logging

Every nav event lands in the session JSONL under kind `"nav"` — `route_set`
(server-side, with the full maneuver list and every precomputed line),
`maneuver_approach` (tier, remaining distance, time-to-maneuver, speed and its
source), `maneuver_complete`, `reroute`, `arrived`, plus `speech` recording what
the arbiter actually did with each announcement. The 1 Hz `progress` tick is
deliberately **not** logged; it would bury the events that matter.

One kind rather than one per event, unlike `lane_drift`: these events *are* the
stream, and their order is the thing under review.

## Desk testing

The nav panel has a **Simulate Drive** control: it walks the host position along
the route's own polyline at a set speed, through the same position handler a
real GPS fix goes through. Nothing about progression, tiers, the arbiter or the
logging can tell the difference. The start-point box next to it routes from
somewhere other than where the laptop is.

`node tools/nav_selftest.js [route.json]` runs the whole thing without a
browser: progression over a synthetic route, tier order, GPS jitter, off-route,
and the arbiter's queue. Pass a route saved from `/nav/route` to run the
progression checks against real Google geometry as well.

## API key

`GOOGLE_MAPS_API_KEY` lives in `.env` (gitignored) and is injected into the page
by `app.py`'s `index()` at serve time, replacing the `__GOOGLE_MAPS_API_KEY__`
placeholder that is what actually sits in `static/index.html`. The Maps
JavaScript API needs a key the browser can see, so "keep it out of the browser"
was never available — "keep it out of the repository" is, and that is the line
held. Routing, geocoding and autocomplete all go through the server, so only the
map render uses the browser-visible key.

The page must be served from `/`. `/static/index.html` is the raw file with the
placeholder still in it; the map is the only thing that breaks, and it breaks
visibly.

## Known edges, for step 2

- **Closely-spaced maneuvers chatter.** Two turns 190 m apart produce a full
  far/mid/near ladder for each, back to back. Google says "turn right onto 10th,
  then left onto Mission" as one line. Chained-maneuver phrasing is the obvious
  next refinement.
- **Lane guidance, exit numbers and roundabout counts** are not requested from
  the API and are not spoken.
- **Traffic is fetched once**, at route time. There is no periodic ETA refresh;
  the panel re-derives ETA from remaining distance and current speed instead.
- **Arrival is modelled as a maneuver**, which is what keeps the engine free of
  an arrival special case, but it means the arrival line uses the same tier
  ladder as a turn.
