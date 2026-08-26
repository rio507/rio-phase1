# Navigation v1 — contextual turn-by-turn

**Status:** V1.0 (deterministic navigation, no vision) and V1.1 (contextual
anchors) both implemented and desk-verified against real routes. Thresholds are
provisional prototype values in the same sense as the headway ladder: starting
points for tuning against real drives, not validated numbers.

---

## The one idea

> The navigation provider determines **where** the driver needs to go. RIO uses
> map context and computer vision to describe that **already-known** maneuver
> in a more natural way.

```
  instead of   "In 500 feet, turn left."
  RIO can say  "Turn left by the Shell station."
```

Vision is never asked where the turn is. RIO already knows. Vision is asked one
question and one only: *is there something the driver can clearly see that
makes this known instruction easier to understand?* If yes, RIO says it. If
not — which is most of the time — RIO says "Take the next left", and that is a
complete instruction, not a degraded one.

**False certainty is worse than conventional navigation.** Frequent anchor
rejection is correct behaviour.

---

## The pipeline

```
  NAVIGATION PROVIDER            navigation/providers/google.py   (the only
        │                                                          Google-shaped
        ▼                                                          file)
  CANONICAL ROUTE MODEL          navigation/model.py
        │
        ▼
  DETERMINISTIC ROUTE TRACKER    static/rio_navcore.js   ~1 Hz, no network
        │
        ▼
  LANDMARK CANDIDATE GENERATOR   navigation/landmarks.py  once per generation
        │
        ▼
  VISUAL VERIFIER                navigation/verify.py     reuses the existing
        │                                                 detector / VLM / depth
        ▼
  CONTEXT VALIDATOR + SELECTOR   navigation/anchors.py    hard gates, then rank
        │
        ▼
  DETERMINISTIC SPEECH PLANNER   navigation/speech.py + static/rio_navplan.js
        │
        ▼
  SPEECH ARBITER                 static/rio_speech.js     one mouth
        │
        ▼
  TTS                            /nav/voice -> ElevenLabs
```

---

## The authority firewall

Perception and language models may **never** determine or modify: the route,
the next road, the turn direction, the maneuver sequence, maneuver
coordinates, route progress, whether a maneuver has been passed, rerouting,
the arrival side, or when an instruction becomes time-critical.

Those come from the provider and the deterministic tracker. Vision's only
possible effect on navigation is to improve a sentence, and its only possible
failure is that the sentence is not improved.

The firewall is structural rather than promised:

- `navigation/speech.py`, `anchors.py` and `landmarks.py` cannot import a
  model or reach the network, and `tools/nav_server_selftest.py` asserts it by
  reading their source — the same check `headway/live_selftest.py` runs
  against `live_policy.py`.
- The verifier returns a `VerifiedAnchor` and nothing else. It cannot write to
  a route.
- `/nav/voice` is addressed by `(route_id, maneuver_id, call_type, anchor_id)`
  and refuses anything that does not resolve to a line already stored on a
  live route. The browser never sends RIO a sentence to say.

---

## The provider boundary

```
NavigationProvider ──► CanonicalDestination
                       CanonicalRoute { generation_id, geometry, maneuvers[] }
                       CanonicalManeuver { type, direction, road_name,
                                           coordinates, route_distance_position,
                                           lane_information?, exit_information? }
                       ArrivalInfo { side: LEFT | RIGHT | UNKNOWN }
```

Nothing downstream knows the word "Google". Two implementations exist today —
`GoogleProvider` and `FixtureProvider` — and the second is exercised on every
test run, which is what keeps "the provider is substitutable" from being a
claim nobody has tested. See `LICENSING.md` for what must be answered about
provider data before production.

A **journey** is a sequence of route generations. A reroute replaces a route
atomically and increments `generation_id`; that integer is what every queued
announcement is validated against before it is allowed to speak.

---

## Time, not distance

Every threshold is seconds-to-maneuver at the current speed, with distance
clamps at the extremes. 200 m of downtown and 200 m of arterial are the same
distance and completely different warnings.

| opportunity | fires at | says |
|---|---|---|
| EARLY (optional) | ~25 s | "Right turn coming up." |
| PRIMARY | ~6 s | "Turn right by the Shell station." / "Take the next right." |
| IMMINENT (only if it adds something) | ~2.5 s | "Right here." |

Anchor acquisition starts at ~11 s, so verification has time to look at
several frames before the primary call is due.

Rules the planner enforces:

- **The primary call REPLACES distance narration.** Never "Turn right in 200
  feet. Turn right by the Shell." Just "Turn right by the Shell."
- **The imminent call stays armed** after a contextual call. It is skipped only
  when the primary line was spoken moments earlier — at a crawl the distance
  clamp can bring both due within a second, and two instructions stacked back
  to back is RIO talking over itself.
- **Nothing is begun too late to finish.** Inside ~4 s the full instruction is
  skipped in favour of "Left here.", because a sentence still playing when the
  driver has to act is worse than a shorter one that finished.
- **A route set inside the window does not prepare.** No countdown that is
  already false.

---

## Two state machines, plus GPS health

```
maneuver:  UPCOMING -> APPROACHING -> IMMINENT -> EXECUTING -> PASSED
context:   INACTIVE -> ACQUIRING -> VERIFIED -> CALLED -> EXPIRED
GPS:       GPS_OK | GPS_DEGRADED | GPS_STALE
route:     ON_ROUTE | OFF_ROUTE_CANDIDATE | OFF_ROUTE_CONFIRMED
```

The maneuver machine has no VISUAL_CONTEXT state and never will.
`maneuver_state = APPROACHING` alongside `context_state = ACQUIRING` is the
normal case; collapsing them is how a camera failure starts being able to
stall a turn instruction.

**GPS health is not off-route.** A stale fix means we do not know where the car
is — it does not mean the car left the route, and it never causes a reroute.
When GPS is degraded near a maneuver, speech is biased **earlier**, never
later: a fix we do not trust is a reason to give the driver more room.

Off-route is distance-from-polyline plus persistence, and nothing else. No road
network matching, no lane inference, no probabilistic road inference — those
are what make an off-route detector confidently wrong. A deviation smaller than
the fix's own stated accuracy does not count towards the run at all.

On a confirmed off-route: tracking stops, a replacement route is requested,
the route is replaced atomically, `generation_id` increments, and every
announcement belonging to the old generation is invalidated. Reroutes are
debounced and capped per journey.

**iOS heading fallback.** Safari routinely reports `speed: null` and
`heading: null`. Both are derived from consecutive fixes — but only when the
derivation means something: fresh samples, real displacement, usable accuracy,
and actually moving. A heading from two fixes 3 m apart while parked is noise
with a compass rose on it, and the tracker says `heading_source` so a
surprising announcement can be explained afterwards.

---

## Landmark-first, and the relation comes from the map

The camera is never asked to discover landmarks in the open world. The map is
asked first:

> The maneuver is at these coordinates. What useful, named, permanent things
> are within 90 m of it?

That turns vision from open-world recognition into **verification of a short
closed list** — bounded, answerable, and safe to be wrong about.

`turn_relation_to_anchor` describes where the **turn** is relative to the
**landmark**, and is computed deterministically from coordinates:

| relation | means | line |
|---|---|---|
| `NEAR` | the turn is at the landmark | "Turn left by the Shell station." |
| `JUST_AFTER` | the landmark comes first | "Turn right just after the Starbucks." |
| `JUST_BEFORE` | the turn comes first | "Turn left just before the CVS." |

NEAR is the default because it needs the least spatial certainty. The other two
are claims about **order**, and a wrong one sends a driver through the
junction — so they demand a much wider margin, and **degrade to NEAR** rather
than being taken on trust. If even NEAR cannot be supported, there is no
anchor.

Relation confidence derives from map-data quality — coordinate precision,
distance from the maneuver, offset from the road — and never from perception.

**Lookup budget.** Candidates are fetched once per route generation, in one
pass over the maneuvers at route load, capped
(`NAV_LANDMARK_MAX_LOOKUPS_PER_ROUTE`), cached for that generation's lifetime,
and refreshed only on reroute. Never per-frame, never on an interval.

---

## What vision actually does

`navigation/verify.py` reuses what RIO already has — no second perception
stack:

- `framebuf.FrameRing`, the few seconds of road already retained for visual
  conversation;
- `enrich.get_adapter()`, the resident VLM, asked one small closed question
  (`landmark()`);
- `headway/depth.py`, Depth Anything V2 Metric-Small, as a plausibility check.

It answers: is the expected landmark visible, is the identity confident, has it
persisted across several observations, is it visually stable, is it spatially
consistent with the maneuver, is it unique enough to reference, and is the
observation fresh.

**Depth is a consistency signal only.** It is never asked how far the
intersection is — route geometry already knows. It is asked whether a thing
claimed to be the Shell 40 m from the maneuver is plausibly in front of the car
at a plausible range. Missing depth is not a rejection.

### The gates (all hard, no trading)

`allowed_anchor_type`, `identity_confidence`, `visibility_confidence`,
`tracking_duration` and observation count, `observation_is_fresh`,
`spatial_consistency`, `scene_uniqueness`, `relation_confidence`.

Any failure rejects. There is deliberately no multiplicative score that lets a
strong identity compensate for a stale observation, or a big bright sign
compensate for there being two of them.

**Two of the same brand in view rejects both.** No verbal disambiguation in
v1 — "Take the next left" is unambiguous and costs nothing. Duplicates are also
rejected at the map level, before the camera is ever asked.

Only candidates that passed everything are ranked, and the ranking is a plain
ordering — salience, spoken usefulness, closeness of the relation, stability,
identity confidence — so a log can answer "why that one". **One anchor per
maneuver, maximum.**

### The contract

```
VerifiedAnchor { label, type, turn_relation_to_anchor,
                 identity_confidence, relation_confidence,
                 visibility_confidence, valid_until }
```

Track ids, depth history, bounding boxes, temporal history, model confidences,
image coordinates and scale rates stay inside the perception subsystem.

### Scope, deliberately small

City streets, daytime, reasonable visibility, straightforward maneuvers, one
anchor maximum. Branded fuel (Shell, Chevron, Mobil, 76, …) and major chain
signage (Starbucks, McDonald's, CVS, Walgreens, …).

Not supported, on purpose: highway-speed contextual instructions, complex
interchanges, night, construction, moving vehicles, pedestrians, vegetation,
temporary signs, multi-anchor descriptions, building descriptions. Traffic
lights come after branded landmarks are reliable. RIO will never say "turn
after the white SUV."

---

## Speech, and why there is no LLM in it

Every sentence RIO can say about a maneuver is enumerable before the drive
starts: a direction, a road name, and at most one landmark from a list fetched
at route load. So they are enumerated, at route load, in `navigation/speech.py`:

```
LEFT  + NEAR       + Shell     -> "Turn left by the Shell station."
RIGHT + JUST_AFTER + Starbucks -> "Turn right just after the Starbucks."
LEFT  + no anchor              -> "Take the next left onto Lincoln Boulevard."
RIGHT + imminent               -> "Right here."
arrival + provider side        -> "Your destination is on the right."
arrival + UNKNOWN side         -> "You've arrived at the Getty Center."
```

That removes latency, hallucination, a validation layer, a test surface and a
whole class of failure state. A model may later vary the phrasing of the EARLY
line — the one line that is never time-critical. **The imminent call stays a
template permanently.**

Arrival side is provider-supplied or omitted. There is no camera path to it.

### Validity is checked at dequeue

```
NavigationSpeechCandidate { text, maneuver_id, route_generation, priority,
                            call_type, created_at, expires_at }
```

The arbiter asks `valid()` the moment before speaking, not when the line was
queued:

```
active_maneuver_id == candidate.maneuver_id
&& route_generation == candidate.route_generation
&& maneuver_not_passed
&& not expired
```

Invalid lines are dropped silently. "Right here." created two seconds before a
junction and dequeued three seconds later, after a gap warning finished, is a
lie about a turn already taken.

### The arbiter — unchanged, one mouth

| priority | channel |
|---|---|
| P1 | safety — headway warnings |
| P2 | vehicle health, critical only |
| P3 | the imminent turn |
| P4 | every other navigation line |
| P5 | conversation |

Navigation groups by maneuver (`nav:m3`), so the imminent call supersedes its
own primary line rather than queueing behind it. Expire, never catch up.
Pre-empted lines are never resumed.

---

## Fallback is normal behaviour

No candidate, landmark not visible, uncertain identity, unstable tracking,
duplicated landmark, uncertain relation, stale observation, inconsistent depth,
camera unavailable, VLM unavailable, vision switched off — all end the same
way: the canonical instruction, and a logged reason.

`tools/nav_selftest.js` asserts that each of those drives is **word for word**
the drive RIO would have made with no vision configured at all.

---

## Destination entry

Typing predicts. The box debounces (~250 ms), asks `/nav/suggest`, and renders
what comes back; picking a prediction resolves **by place id**, never by
re-searching the text, because the same name is frequently two places.

Autocomplete is proxied through the server for two reasons: the key stays on
one surface, and a provider-rendered dropdown cannot be made to look like the
rest of this dashboard.

**Typing sessions.** RIO mints an opaque id for one typing session — everything
between the first keystroke and the selection — and sends it with every suggest
request and again with the selection. Providers may do what they like with it
behind the boundary: Google turns it into an autocomplete session token, which
groups a dozen keystroke requests and the final lookup into one billed session.
The browser never holds a provider token, which is the API-key rule applied to
the thing the key is spent on. The session is consumed by the resolution and a
new one is minted the next time the driver types; a resolution with no typing
behind it (a spoken destination, a reroute) opens none at all.

**Submitting is the path underneath.** Suggestions are a convenience on top of
it, never a gate in front of it: Enter or **Route** resolves whatever is in the
box through the same provider, so an autocomplete outage costs prediction and
nothing else. An ambiguous submission still asks which one was meant, and the
choice renders through the same list.

One thing worth knowing about the seam: the panel maps a suggestion to a row in
exactly one function (`toSuggestion`), which drops any entry without an id or
without something to read. When the panel and the endpoint once disagreed about
field names, the dropdown filled with blank rows that routed to `undefined` —
visible only after a click. A future drift now shows as "no suggestions" and
the driver simply submits what they typed. `tools/nav_server_selftest.py` reads
the field names out of that function and checks them against what
`/nav/suggest` actually emits.

---

## Dashboard

Destination, ETA, next maneuver, distance to it, and four states shown
separately because they are genuinely independent: maneuver state, GPS state,
route state, context state. Plus the selected anchor with its three confidences
as a debug line. No map UI is required — a route you can hear is the product.

---

## Observability

Every event lands in the session JSONL under kind `"nav"`:
`NAV_ROUTE_STARTED`, `NAV_MANEUVER_SELECTED`, `NAV_EARLY_GUIDANCE`,
`NAV_CONTEXT_ACQUISITION_STARTED`, `NAV_ANCHOR_CANDIDATE`,
`NAV_ANCHOR_VERIFIED`, `NAV_ANCHOR_REJECTED`, `NAV_CONTEXTUAL_CALL`,
`NAV_NEAR_TURN`, `NAV_MANEUVER_PASSED`, `NAV_GPS_DEGRADED`, `NAV_GPS_STALE`,
`NAV_OFF_ROUTE_CANDIDATE`, `NAV_OFF_ROUTE_CONFIRMED`, `NAV_REROUTE_STARTED`,
`NAV_REROUTE_COMPLETE`, `NAV_SPEECH_EXPIRED`, `NAV_SPEECH_INVALIDATED`,
`NAV_ARRIVED`.

The 1 Hz `NAV_PROGRESS` tick is deliberately **not** logged; it would bury the
events that matter. Never an API key, in any field. No persistent precise
location history: events carry route-relative position, except the off-route
point, which cannot be understood without knowing where it happened.

---

## Configuration

Every threshold is in `config.py` under "Contextual navigation", and the ones
the browser times against are shipped **with the route** in `timing`, so tuning
reaches the car on the next route rather than on the next deploy:

`gps_stale_timeout`, `gps_accuracy_limit`, `off_route_distance`,
`off_route_persistence`, `reroute_debounce`, `early_guidance_seconds`,
`anchor_acquisition_seconds`, `context_call_seconds`, `near_turn_seconds`,
`min_call_distance`, `max_call_distance`, `anchor_min_tracking_duration`,
`anchor_min_identity_confidence`, `anchor_min_visibility_confidence`,
`anchor_min_relation_confidence`, `anchor_max_age`,
`duplicate_instruction_cooldown`.

---

## Testing

```
node tools/nav_selftest.js                     # tracker, planner, arbiter
node tools/nav_selftest.js route.json          # ...against real geometry
python -m tools.nav_server_selftest            # provider, relations, gates, speech
python -m tools.nav_server_selftest --live     # + one real provider route
```

Between them they cover the required list: normal navigation; the contextual
landmark; no landmark; a low-confidence landmark; a landmark that disappears; a
duplicated landmark; an uncertain relation degrading to NEAR; context plus
imminent; stale speech dropped at dequeue; reroute invalidation; GPS stale
without off-route; GPS degraded speaking earlier; missing iOS heading derived
from GPS deltas; the LLM unavailable; the camera unavailable.

The dashboard's **Simulate Drive** walks the host position along the route's
own geometry through the same position handler a real fix goes through —
nothing about tracking, planning, the arbiter or the logging can tell the
difference.

---

## API key

`GOOGLE_MAPS_API_KEY` lives in `.env` (gitignored) and is injected into the
page at serve time, replacing the `__GOOGLE_MAPS_API_KEY__` placeholder in
`static/index.html`. The Maps JavaScript API needs a key the browser can see,
so "keep it out of the browser" was never available — "keep it out of the
repository" is, and that is the line held. Routing, geocoding, autocomplete and
place search all go through the server.

Required APIs: **Routes API**, **Geocoding API**, **Places API (New)** —
including Autocomplete, Place Details and Nearby Search — and **Maps
JavaScript API** for the optional map. If any is not enabled, the console
reports it in the error body and `navigation/fixtures.py::FixtureProvider`
keeps the whole system developable in the meantime.

---

## Known edges, for next

- **Closely-spaced maneuvers.** Two turns 190 m apart still produce two full
  sets of calls. Chained phrasing ("turn right, then left onto Mission") is the
  obvious next refinement.
- **Lane guidance and exit numbers** are modelled in `CanonicalManeuver` and
  are neither requested from the provider nor spoken.
- **Traffic is fetched once,** at route time; the panel re-derives ETA from
  remaining distance and current speed.
- **Anchors are city-street only.** Highway interchanges are excluded by
  maneuver type, not by a speed test — that is the conservative direction, and
  a speed gate would be the natural addition.
